"""Source-anchored *partial* tariff-to-issuer cash-flow transmission.

The original US-import-tariff-on-Mexico target remains unpriced unless an
equivalent contract exists. This component asks a different, explicitly
conditional question: what is the direct contribution-cash-flow effect IF that
tariff causes some loss of Mexico-linked freight business at Union Pacific?

The 2026-Q3 SEC statement covers shipments *to and from* Mexico. The share
affected by US import tariffs and the causal freight-volume response are both
unknown; their separate default domains are [0, 1]. A 2026 quarter is only a
flat daily run-rate anchor for a specified future horizon, not a revenue
forecast. Bounds apply solely to this fixed cash-flow model, not total stock
return, total portfolio loss or any market probability.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import re

from .business_interruption import BusinessInterruptionError, evaluate_business_interruption
from .business_interruption_assessment import _issuer_exposure, _positions, _time
from .business_interruption_sources import (
    BusinessSourceError, SOURCE_RECEIVED, SOURCE_SHA256, SOURCE_URL,
    checked_unp_capitalization, load_unp_mexico_exposure_anchor,
)


SCHEMA = 'castle.mexico_tariff_cashflow_component.v1'
STATUS = 'DECLARED_DIRECT_MEXICO_FREIGHT_CHANNEL_NOT_EVENT_PNL'
TARGET_ID = 'us_mexico_tariff_increase_by_2027_01_26'
TARGET_TITLE = 'US increases tariff rate on imports from Mexico by 2027-01-26'
CONDITION = 'If a US import tariff on Mexico causes Union Pacific to lose Mexico-linked freight volume'
DEFAULT_DOMAINS = {
    'us_import_affected_share_of_mexico_freight': {'lower': 0., 'upper': 1.},
    'tariff_caused_freight_volume_loss_fraction': {'lower': 0., 'upper': 1.},
    'incremental_contribution_margin': {'lower': 0., 'upper': 1.},
    'freight_catchup_fraction': {'lower': 0., 'upper': 1.},
    'immediate_tax_shield_fraction': {'lower': 0., 'upper': 1.},
}
DURATIONS = [7, 30, 90]


def _domain(value, label):
    if not isinstance(value, dict) or set(value) != {'lower', 'upper'}:
        raise BusinessInterruptionError(label + ' requires exactly lower and upper')
    lo, hi = value['lower'], value['upper']
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (lo, hi)) or not 0 <= lo <= hi <= 1):
        raise BusinessInterruptionError(label + ' is not a valid fraction interval')
    return {'lower': float(lo), 'upper': float(hi)}


def _assumptions(domains):
    if not isinstance(domains, dict) or set(domains) != set(DEFAULT_DOMAINS):
        raise BusinessInterruptionError('All five explicit causal domains are required')
    return {key: _domain(domains[key], key) for key in DEFAULT_DOMAINS}


def _business_assumptions(domains):
    imports = domains['us_import_affected_share_of_mexico_freight']
    response = domains['tariff_caused_freight_volume_loss_fraction']
    return {
        'affected_fraction': {'lower': imports['lower'] * response['lower'],
                              'upper': imports['upper'] * response['upper']},
        'contribution_margin': deepcopy(domains['incremental_contribution_margin']),
        'catchup_fraction': deepcopy(domains['freight_catchup_fraction']),
        'tax_shield_fraction': deepcopy(domains['immediate_tax_shield_fraction']),
        'annual_discount_rate': 0., 'start_delay_days': 0,
        'catchup_delay_days': 120,
        'additional_cost_mode': 'EXCLUDED_FROM_COMPONENT',
    }


def evaluate_direct_mexico_channel(anchor, *, domains, exposure, capitalization):
    """Return sharp rectangular-domain cash-flow bounds under fixed timing.

    This does not estimate causal elasticities. ``affected_fraction`` is the
    product of the two distinct tariff-related domains, whose interval endpoints
    are attainable together; no independent draws or hidden point values.
    """
    domains = _assumptions(domains)
    if (not isinstance(anchor, dict) or anchor.get('issuer') != 'UNP'
            or anchor.get('revenue_metric') != 'Reported freight revenues from shipments to AND from Mexico'
            or anchor.get('mexico_linked_freight_revenue_usd') != 820_000_000.
            or anchor.get('period_start') != '2026-07-01'
            or anchor.get('period_end') != '2026-09-30'
            or anchor.get('calendar_days') != 92
            or anchor.get('mexico_import_only_fraction_identified') is not False
            or anchor.get('tariff_effect_identified') is not False):
        raise BusinessSourceError('A reviewed two-direction Mexico freight baseline is required')
    capital = capitalization if isinstance(capitalization, dict) else {}
    model = evaluate_business_interruption(
        {'issuer': 'UNP', 'revenue_usd': 820_000_000.,
         'period_start': anchor['period_start'], 'period_end': anchor['period_end'],
         'revenue_metric': anchor['revenue_metric'], 'source_record_id': anchor['source_record_id']},
        duration_days=list(DURATIONS), assumptions=_business_assumptions(domains),
        exposure=exposure,
        equity_value={'issuer': 'UNP', 'market_cap_usd': capital.get('market_capitalization_usd'),
                      'as_of': capital.get('received_at_utc'),
                      'source_record_id': capital.get('source_record_id')},
        equity_pricing_weights=None)
    rows = []
    for result in model['duration_scenarios']:
        normalized = result['equity_normalization']
        if normalized['status'] != 'FIXED_DENOMINATOR_CASHFLOW_COMPONENT_ONLY':
            raise BusinessSourceError('This model cannot project the cash-flow channel within the equity denominator')
        rows.append({'horizon_days': result['duration_calendar_days'],
                     'company_direct_cashflow_loss_pv_usd': result['company_after_tax_cashflow_component_loss_pv_usd'],
                     'known_unp_position_component_change_usd': normalized['known_issuer_portfolio_component_contrast_usd']})
    return model, rows, domains


def relevant_original_mexico_tariff(event):
    """An exact recorded original-target binding, not an LLM text similarity."""
    return (isinstance(event, dict) and event.get('event_id', event.get('id')) == TARGET_ID
            and event.get('title') == TARGET_TITLE
            and event.get('family') == 'TRADE_TARIFFS'
            and event.get('event_date') == '2027-01-26'
            and event.get('source_defined_scenario') is not True)


def build_mexico_tariff_component(event, *, portfolio, inputs, evaluated_at,
                                  original_assessment_at=None, domains=None):
    """Enrich one exact original event without altering Q, rank or P&L gates."""
    if not relevant_original_mexico_tariff(event):
        return None
    try:
        now = _time(evaluated_at)
        original = _time(original_assessment_at or evaluated_at)
        if now < original:
            raise BusinessSourceError('Financial enrichment precedes the original event evaluation')
        anchor, source = load_unp_mexico_exposure_anchor(evaluated_at=now)
        cap = checked_unp_capitalization((inputs.get('instrument_metadata') or {}).get('UNP'), evaluated_at=now)
        exposure = _issuer_exposure(portfolio, inputs, now)
        model, rows, assumptions = evaluate_direct_mexico_channel(
            anchor, domains=deepcopy(domains or DEFAULT_DOMAINS), exposure=exposure, capitalization=cap)
        original_exposure = model['original_portfolio_exposure']
        if original_exposure['known_issuer_portfolio_weight'] <= 0:
            raise BusinessSourceError('No reviewed UNP common-equity issuer exposure')
        component = {
            'schema': SCHEMA, 'status': STATUS, 'event_id': TARGET_ID,
            'condition': CONDITION, 'evaluated_at': now.isoformat(),
            'original_assessment_at': original.isoformat(),
            'separate_later_enrichment': now > original,
            'filing_source': {'url': SOURCE_URL, 'sha256': SOURCE_SHA256,
                             'received_at': SOURCE_RECEIVED,
                             'period_start': '2026-07-01', 'period_end': '2026-09-30',
                             'mexico_linked_to_and_from_freight_revenue_usd': 820_000_000.,
                             'calendar_days': 92,
                             'source_record_id': anchor['source_record_id']},
            'equity_source': {key: cap[key] for key in
                              ('source_url', 'response_sha256', 'received_at_utc', 'source_record_id', 'basis')},
            'exposure': {'portfolio_notional_usd': portfolio.notional,
                         'original_positions': _positions(exposure['positions']),
                         'known_issuer_weight': original_exposure['known_issuer_portfolio_weight'],
                         'unresolved_fund_weight': original_exposure['unresolved_fund_portfolio_weight'],
                         'complete_issuer_exposure': original_exposure['complete_issuer_exposure']},
            'declared_domains': assumptions,
            'scenarios': rows,
            'mexico_import_share_identified': False,
            'tariff_caused_volume_response_identified': False,
            'future_quarter_revenue_identified': False,
            'physical_probability_P': None, 'prediction_market_Q': None,
            'total_stock_return_identified': False,
            'full_portfolio_conditional_pnl_identified': False,
            'primary_impact_allowed': False, 'economic_ranking_allowed': False,
            'already_priced_in_identified': False,
            'source_or_model_calls': 0,
            'interpretation': 'Direct lost freight-contribution component IF a tariff causes volume loss on the relevant US-import subset. Each horizon begins at hypothetical onset of lost traffic, not at the assessment or event-announcement date. A 2026-Q3 both-direction revenue proxy is spread uniformly over that hypothetical interval.',
            'excluded_channels': ['export-related traffic not affected by the specified US import tariff',
                                  'substitution, diversion and possible added shipments',
                                  'extra operating costs, price changes, permanent demand and network spillovers',
                                  'financing, risk premia, terminal value and future stock-return repricing'],
            'bounds_method': model['bounds_method'] + '_PRODUCT_OF_TWO_DISTINCT_TARIFF_FRACTIONS',
            'source_scope_note': 'Original filing bytes and captured capitalization are checked locally. Their source statements are not an estimate of a tariff-induced import-volume fall.',
        }
        return {'public_component': component, 'private_mexico_tariff_inputs': {
            'model': model, 'filing': source, 'capitalization': cap}}
    except (BusinessSourceError, BusinessInterruptionError, ValueError, TypeError, KeyError, OSError, OverflowError):
        return {'status': 'MEXICO_TARIFF_COMPONENT_UNAVAILABLE',
                'reason': 'Exact target, original economic source, capitalization or exposure could not be verified; no estimate substituted.'}


_TOP_LEVEL = {
    'schema', 'status', 'event_id', 'condition', 'evaluated_at', 'original_assessment_at',
    'separate_later_enrichment', 'filing_source', 'equity_source', 'exposure',
    'declared_domains', 'scenarios', 'mexico_import_share_identified',
    'tariff_caused_volume_response_identified', 'future_quarter_revenue_identified',
    'physical_probability_P', 'prediction_market_Q', 'total_stock_return_identified',
    'full_portfolio_conditional_pnl_identified', 'primary_impact_allowed',
    'economic_ranking_allowed', 'already_priced_in_identified', 'source_or_model_calls',
    'interpretation', 'excluded_channels', 'bounds_method', 'source_scope_note',
}


def mexico_component_view(value, *, event_id=None, portfolio_notional=None,
                          portfolio_positions=None):
    """Strictly display-check a saved component. NOT source authentication.

    An edited JSON may invent source bytes: only the original build rechecked
    the retained SEC and provider records. This shape/arithmetic check cannot
    promote unrelated market probabilities or full portfolio P&L.
    """
    try:
        if not isinstance(value, dict) or set(value) != _TOP_LEVEL:
            return None
        v = deepcopy(value)
        if (v['schema'] != SCHEMA or v['status'] != STATUS or v['event_id'] != TARGET_ID
                or (event_id is not None and event_id != TARGET_ID) or v['condition'] != CONDITION):
            return None
        for key in ('mexico_import_share_identified', 'tariff_caused_volume_response_identified',
                    'future_quarter_revenue_identified', 'total_stock_return_identified',
                    'full_portfolio_conditional_pnl_identified', 'primary_impact_allowed',
                    'economic_ranking_allowed', 'already_priced_in_identified'):
            if v[key] is not False:
                return None
        if (v['physical_probability_P'] is not None or v['prediction_market_Q'] is not None
                or type(v['source_or_model_calls']) is not int or v['source_or_model_calls'] != 0):
            return None
        now, original = _time(v['evaluated_at']), _time(v['original_assessment_at'])
        if now < original or v['separate_later_enrichment'] is not (now > original):
            return None
        filing, cap = v['filing_source'], v['equity_source']
        if (filing != {'url': SOURCE_URL, 'sha256': SOURCE_SHA256,
                       'received_at': SOURCE_RECEIVED,
                       'period_start': '2026-07-01', 'period_end': '2026-09-30',
                       'mexico_linked_to_and_from_freight_revenue_usd': 820_000_000.,
                       'calendar_days': 92,
                       'source_record_id': 'SEC_0000100885_26_000321_MEXICO_BUSINESS_PARAGRAPH'}
                or now < _time(SOURCE_RECEIVED)
                or (now.date() - datetime(2026, 9, 30).date()).days > 120):
            return None
        received = _time(cap['received_at_utc'])
        if (not isinstance(cap, dict)
                or set(cap) != {'source_url', 'response_sha256', 'received_at_utc',
                                'source_record_id', 'basis'}
                or not 0 <= (now - received).total_seconds() <= 3 * 86400
                or cap['source_url'] != 'https://api.massive.com/v3/reference/tickers/UNP?date=' + received.date().isoformat()
                or not isinstance(cap['response_sha256'], str)
                or re.fullmatch(r'[a-f0-9]{64}', cap['response_sha256']) is None
                or any(not isinstance(cap[k], str) or not cap[k] or len(cap[k]) > 1000
                       for k in ('source_record_id', 'basis'))):
            return None
        exposure = v['exposure']
        if (not isinstance(exposure, dict) or
                set(exposure) != {'portfolio_notional_usd', 'original_positions',
                                 'known_issuer_weight', 'unresolved_fund_weight',
                                 'complete_issuer_exposure'}):
            return None
        n, w, u = (exposure[key] for key in
                   ('portfolio_notional_usd', 'known_issuer_weight', 'unresolved_fund_weight'))
        def finite(x):
            return type(x) in (int, float) and math.isfinite(x)
        if (not all(finite(x) for x in (n, w, u)) or not n > 0 or not 0 < w <= 1
                or not 0 <= u <= 1 - w + 1e-12
                or exposure['complete_issuer_exposure'] is not (u == 0)
                or portfolio_notional is not None and portfolio_notional != n):
            return None
        originals = _positions(exposure['original_positions'])
        if portfolio_positions is not None and originals != _positions(portfolio_positions):
            return None
        direct = math.fsum(row['weight'] for row in originals
                           if row['symbol'] == 'UNP' and row['type'] == 'stock')
        fund = math.fsum(row['weight'] for row in originals if row['type'] == 'etf')
        if w < direct - 1e-12 or w + u > direct + fund + 1e-12:
            return None
        domains = _assumptions(v['declared_domains'])
        if (not isinstance(v['scenarios'], list)
                or len(v['scenarios']) != len(DURATIONS)
                or not isinstance(v['interpretation'], str) or len(v['interpretation']) > 1500
                or not isinstance(v['source_scope_note'], str) or len(v['source_scope_note']) > 1000
                or not isinstance(v['bounds_method'], str) or len(v['bounds_method']) > 500
                or not isinstance(v['excluded_channels'], list)
                or any(not isinstance(s, str) or len(s) > 400 for s in v['excluded_channels'])):
            return None
        f, response = domains['us_import_affected_share_of_mexico_freight'], domains['tariff_caused_freight_volume_loss_fraction']
        margin, catchup, shield = (domains['incremental_contribution_margin'],
                                   domains['freight_catchup_fraction'],
                                   domains['immediate_tax_shield_fraction'])
        ownership_ratio = None
        for duration, row in zip(DURATIONS, v['scenarios']):
            if (not isinstance(row, dict)
                    or set(row) != {'horizon_days', 'company_direct_cashflow_loss_pv_usd',
                                    'known_unp_position_component_change_usd'}
                    or type(row['horizon_days']) is not int or row['horizon_days'] != duration):
                return None
            cash, change = (row[k] for k in
                            ('company_direct_cashflow_loss_pv_usd',
                             'known_unp_position_component_change_usd'))
            if (any(not isinstance(pair, list) or len(pair) != 2 or
                    any(not finite(x) for x in pair) or pair[0] > pair[1]
                    for pair in (cash, change))):
                return None
            base = 820e6 * duration / 92
            lo = base * f['lower'] * response['lower'] * margin['lower'] * (1 - catchup['upper']) * (1 - shield['upper'])
            hi = base * f['upper'] * response['upper'] * margin['upper'] * (1 - catchup['lower']) * (1 - shield['lower'])
            if (not math.isclose(cash[0], lo, rel_tol=1e-11, abs_tol=1e-5)
                    or not math.isclose(cash[1], hi, rel_tol=1e-11, abs_tol=1e-5)):
                return None
            # A saved document may validate arithmetic and common ownership
            # consistency; it cannot reauthenticate the private market cap.
            if hi > 0:
                scale = -change[0] / hi
                if (not math.isfinite(scale) or not 0 < scale < 1
                        or ownership_ratio is not None and not math.isclose(
                            scale, ownership_ratio, rel_tol=1e-11, abs_tol=1e-14)):
                    return None
                ownership_ratio = scale
            elif not all(math.isclose(item, 0., abs_tol=1e-10, rel_tol=0)
                         for item in change):
                return None
            if ownership_ratio is not None and (
                    not math.isclose(change[0], -ownership_ratio * hi, rel_tol=1e-11, abs_tol=1e-7)
                    or not math.isclose(change[1], -ownership_ratio * lo, rel_tol=1e-11, abs_tol=1e-7)):
                return None
        return v
    except (BusinessInterruptionError, BusinessSourceError, KeyError, TypeError, ValueError,
            OverflowError, AttributeError):
        return None

