"""Join a reviewed issuer cash-flow mechanism without promoting event P&L.

The broad railroad-strike question remains unanswered. A different, explicitly
conditional question is evaluated: what is the finite freight contribution
channel if UNP's own operations stop? No return fit, midpoint probability, or
new provider request is performed. Original ranking/admission are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re

from .business_interruption import BusinessInterruptionError, evaluate_business_interruption
from .business_interruption_sources import (
    BusinessSourceError, SOURCE_URL, SOURCE_SHA256, SOURCE_RECEIVED,
    checked_unp_capitalization, load_unp_source_anchor,
)
from .portfolio import recover_exposures


SCHEMA = 'castle.business_interruption.presentation.v1'
STATUS = 'DECLARED_CASHFLOW_COMPONENT_NOT_TOTAL_EVENT_PNL'
CONDITION = "If Union Pacific's own freight operations stop"
DEFAULT_ASSUMPTIONS = {
    'affected_fraction': {'lower': 1., 'upper': 1.},
    'contribution_margin': {'lower': 0., 'upper': 1.},
    'catchup_fraction': {'lower': 0., 'upper': 1.},
    'tax_shield_fraction': {'lower': 0., 'upper': 1.},
    'annual_discount_rate': 0., 'start_delay_days': 0,
    'catchup_delay_days': 30, 'additional_cost_mode': 'EXCLUDED_FROM_COMPONENT',
}
_ISO = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)')


def _time(value):
    if isinstance(value, datetime):
        value = value.isoformat()
    if not isinstance(value, str) or not _ISO.fullmatch(value):
        raise ValueError('An aware ISO timestamp is required.')
    moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return moment.astimezone(timezone.utc)


def relevant_rail_interruption(event):
    """Routing only, never proof that any market event implies UNP shutdown."""
    if not isinstance(event, dict):
        return False
    title = str(event.get('title') or '').lower()
    if re.search(r'\b(passenger|commuter|nj transit|amtrak|subway|metro|option|strike price)\b', title):
        return False
    return bool(re.search(r'\b(strike|lockout|work stoppage|shutdown|interruption)\b', title)
                and re.search(r'\b(railroad|railway|freight rail|class i rail|union pacific|unp)\b', title))


def _issuer_exposure(portfolio, inputs, evaluated_at):
    ledger = recover_exposures(portfolio, inputs.get('holdings') or {}, now=evaluated_at)
    rows = []
    for position in portfolio.positions:
        row = {'symbol': position.symbol, 'asset_type': position.asset_type, 'weight': position.weight,
               'issuer_fraction': 1. if position.asset_type == 'stock' and position.symbol == 'UNP' else 0.}
        if position.asset_type == 'etf':
            entries = [e for e in ledger.entries if e.original_symbol == position.symbol]
            known = math.fsum(e.portfolio_weight for e in entries
                             if e.status == 'look_through' and e.asset_type == 'stock' and e.symbol == 'UNP')
            unknown = math.fsum(e.portfolio_weight for e in entries if e.status == 'unresolved_etf')
            if position.weight and unknown >= position.weight - 1e-12 and not known:
                row['issuer_fraction'] = None
            elif position.weight:
                row['issuer_fraction'] = known / position.weight
                row['unresolved_issuer_fraction'] = unknown / position.weight
                refs = sorted((e.symbol, e.asset_type, e.status, e.portfolio_weight,
                               e.source_url or '', e.as_of or '', e.retrieved_at or '') for e in entries)
                source_rows_present = any(e.status == 'look_through' and e.source_url for e in entries)
                if not source_rows_present:
                    row['issuer_fraction'] = None
                    row.pop('unresolved_issuer_fraction', None)
                else:
                    row['holdings_source_record_id'] = 'NORMALIZED_HOLDINGS_LEDGER_ROWS_NOT_SOURCE_AUTHENTICATION_' + sha256(
                        json.dumps(refs).encode()).hexdigest()
        rows.append(row)
    return {'notional_usd': portfolio.notional, 'positions': rows}


def _positions(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError('Original positions are required.')
    result = []
    for row in rows:
        symbol, kind, weight = row['symbol'], row.get('asset_type', row.get('type')), row['weight']
        if (not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.^/-]{0,31}', symbol)
                or kind not in ('stock', 'etf', 'cash') or isinstance(weight, bool)
                or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1):
            raise ValueError('Malformed original position.')
        result.append({'symbol': symbol, 'type': kind, 'weight': weight})
    if len({r['symbol'] for r in result}) != len(result) or not math.isclose(math.fsum(r['weight'] for r in result), 1, abs_tol=1e-10, rel_tol=0):
        raise ValueError('Original allocation is duplicated or incomplete.')
    return sorted(result, key=lambda row: row['symbol'])


def build_business_interruption_component(event, *, portfolio, inputs, evaluated_at,
                                          original_assessment_at=None):
    if not relevant_rail_interruption(event):
        return None
    try:
        now = _time(evaluated_at)
        original_time = _time(original_assessment_at or evaluated_at)
        if now < original_time:
            raise BusinessSourceError('A later component cannot precede its original assessment.')
        anchor, filing = load_unp_source_anchor(evaluated_at=now)
        cap = checked_unp_capitalization((inputs.get('instrument_metadata') or {}).get('UNP'), evaluated_at=now)
        exposure = _issuer_exposure(portfolio, inputs, now)
        model = evaluate_business_interruption(anchor, duration_days=[1, 7, 14],
            assumptions=deepcopy(DEFAULT_ASSUMPTIONS), exposure=exposure,
            equity_value={'issuer': 'UNP', 'market_cap_usd': cap['market_capitalization_usd'],
                          'as_of': cap['received_at_utc'], 'source_record_id': cap['source_record_id']},
            equity_pricing_weights=[0., .5, 1.])
        observed = model['original_portfolio_exposure']
        if observed['known_issuer_portfolio_weight'] <= 0:
            raise BusinessSourceError('No verified common-equity UNP sleeve is present in the captured portfolio.')
        scenarios = []
        for row in model['duration_scenarios']:
            values = row['equity_normalization']['known_issuer_portfolio_component_contrast_usd']
            if values is None:
                raise BusinessSourceError('The cash-flow component cannot be normalized within the reviewed equity domain.')
            scenarios.append({'duration_days': row['duration_calendar_days'],
                'company_component_loss_range_usd': row['company_after_tax_cashflow_component_loss_pv_usd'],
                'known_position_component_change_range_usd': values,
                'pricing_sensitivity': [{
                    'assumed_equity_embedded_weight': item['assumed_equity_embedded_weight'],
                    'yes_component_range_usd': item['known_issuer_portfolio_yes_component_usd'],
                    'no_component_range_usd': item['known_issuer_portfolio_no_component_usd'],
                } for item in row['equity_pricing_sensitivity']]})
        view = {'schema': SCHEMA, 'status': STATUS, 'issuer': 'UNP',
            'event_id': str(event.get('event_id') or event.get('id') or ''),
            'condition': CONDITION, 'evaluated_at': now.isoformat(),
            'original_assessment_at': original_time.isoformat(),
            'separate_later_enrichment': now > original_time,
            'source_scope': 'ORIGINAL_FILING_AND_CAPITALIZATION_BYTES_RECHECKED_AT_COMPONENT_BUILD',
            'filing_source': {'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'received_at': SOURCE_RECEIVED,
                'period_start': anchor['period_start'], 'period_end': anchor['period_end'],
                'freight_revenue_usd': anchor['revenue_usd'], 'calendar_days': 92},
            'equity_source': {key: cap[key] for key in
                ('source_url', 'response_sha256', 'received_at_utc', 'source_record_id', 'basis')},
            'exposure': {'portfolio_notional_usd': portfolio.notional,
                'original_positions': _positions(exposure['positions']),
                'known_issuer_weight': observed['known_issuer_portfolio_weight'],
                'unresolved_fund_weight': observed['unresolved_fund_portfolio_weight'],
                'complete_issuer_exposure': observed['complete_issuer_exposure']},
            'scenarios': scenarios, 'declared_assumptions': deepcopy(DEFAULT_ASSUMPTIONS),
            'pricing_weights_are_assumptions_not_probabilities': True,
            'physical_probability_P': None, 'prediction_market_Q': None,
            'event_to_issuer_interruption_verified': False,
            'total_stock_return_identified': False, 'full_portfolio_conditional_pnl_identified': False,
            'primary_impact_allowed': False, 'economic_ranking_allowed': False,
            'interpretation': model['interpretation'], 'unmodeled_channels': model['unmodeled_channels'],
            'bounds_method': model['bounds_method'],
            'mechanism_fingerprint_sha256': model['mechanism_fingerprint_sha256'],
            'source_or_model_calls': 0,
            'source_authenticity_note': 'Financial anchors use local original-byte consistency checks, not a provider signature or historical publication-time proof. Fund exposure relies on upstream capture checks; normalized ledger-row fingerprints are not independent source authentication.',
        }
        # Private reproduction retains the denominator and every extremizing
        # witness. Browser/shareable exports omit this original-provider input.
        return {'public_component': view, 'private_business_interruption_inputs': {
            'model': model, 'filing': filing, 'equity_source': cap}}
    except (BusinessSourceError, BusinessInterruptionError, ValueError, TypeError, KeyError, OSError, OverflowError):
        return {'status': 'BUSINESS_COMPONENT_UNAVAILABLE',
                'reason': 'The reviewed issuer, source dates, original capture or common-equity exposure could not be reconstructed; no component substituted.'}


def component_view(value, *, event_id=None, portfolio_notional=None, portfolio_positions=None):
    """Validate a recorded component for display, never promote a saved flag.

    Does not reread original sources or authenticate edited JSON. It enforces
    the narrow saved schema, assumptions, shared-state arithmetic and scope.
    """
    try:
        if not isinstance(value, dict) or value.get('schema') != SCHEMA or value.get('status') != STATUS:
            return None
        v = deepcopy(value)
        if (v.get('issuer') != 'UNP' or v.get('condition') != CONDITION
                or not isinstance(v.get('event_id'), str) or not v['event_id']
                or event_id is not None and v['event_id'] != event_id):
            return None
        for key in ('event_to_issuer_interruption_verified', 'total_stock_return_identified',
                    'full_portfolio_conditional_pnl_identified', 'primary_impact_allowed', 'economic_ranking_allowed'):
            if v.get(key) is not False:
                return None
        if (v.get('physical_probability_P') is not None or v.get('prediction_market_Q') is not None
                or v.get('pricing_weights_are_assumptions_not_probabilities') is not True
                or v.get('declared_assumptions') != DEFAULT_ASSUMPTIONS):
            return None
        for name in ('affected_fraction', 'contribution_margin', 'catchup_fraction', 'tax_shield_fraction'):
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   for x in v['declared_assumptions'][name].values()):
                return None
        if any(isinstance(v['declared_assumptions'][name], bool) for name in
               ('annual_discount_rate', 'start_delay_days', 'catchup_delay_days')):
            return None
        now, original = _time(v.get('evaluated_at')), _time(v.get('original_assessment_at'))
        if now < original or v.get('separate_later_enrichment') is not (now > original):
            return None
        filing, cap, exposure = v['filing_source'], v['equity_source'], v['exposure']
        if (filing != {'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'received_at': SOURCE_RECEIVED,
                       'period_start': '2026-07-01', 'period_end': '2026-09-30',
                       'freight_revenue_usd': 6813000000., 'calendar_days': 92}
                or now < _time(filing['received_at']) or (now.date() - datetime(2026, 9, 30).date()).days > 120
                or not 0 <= (now - _time(cap['received_at_utc'])).total_seconds() <= 259200
                or cap['source_url'] != 'https://api.massive.com/v3/reference/tickers/UNP?date=' + _time(cap['received_at_utc']).date().isoformat()
                or not re.fullmatch(r'[a-f0-9]{64}', cap['response_sha256'])):
            return None
        def finite(x):
            return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
        n, w, u = (exposure[key] for key in ('portfolio_notional_usd', 'known_issuer_weight', 'unresolved_fund_weight'))
        if (not all(finite(x) for x in (n, w, u)) or n <= 0 or not 0 < w <= 1 or not 0 <= u <= 1 - w + 1e-12
                or portfolio_notional is not None and n != portfolio_notional
                or exposure.get('complete_issuer_exposure') is not (u == 0)):
            return None
        originals = _positions(exposure['original_positions'])
        if portfolio_positions is not None and originals != _positions(portfolio_positions):
            return None
        direct = math.fsum(p['weight'] for p in originals if p['symbol'] == 'UNP' and p['type'] == 'stock')
        funds = math.fsum(p['weight'] for p in originals if p['type'] == 'etf')
        if w < direct - 1e-12 or w + u > direct + funds + 1e-12:
            return None
        rows = v['scenarios']
        if (not isinstance(rows, list) or [r.get('duration_days') for r in rows] != [1, 7, 14]
                or any(type(r.get('duration_days')) is not int for r in rows)):
            return None
        def pair(x):
            return isinstance(x, list) and len(x) == 2 and all(finite(a) for a in x) and x[0] <= x[1]
        base = None
        for row in rows:
            loss, change = row['company_component_loss_range_usd'], row['known_position_component_change_range_usd']
            days = row['duration_days']
            if (not pair(loss) or not pair(change) or loss[0] != 0 or change[1] != 0
                    or not math.isclose(loss[1], 6813000000. / 92 * days, rel_tol=1e-12)
                    or not -n*w <= change[0] < 0):
                return None
            base = change[0] / days if base is None else base
            if not math.isclose(change[0], base * days, rel_tol=1e-12):
                return None
            pricing = row['pricing_sensitivity']
            if not isinstance(pricing, list) or [p['assumed_equity_embedded_weight'] for p in pricing] != [0., .5, 1.]:
                return None
            for p in pricing:
                q = p['assumed_equity_embedded_weight']
                if not finite(q):
                    return None
                yes, no = p['yes_component_range_usd'], p['no_component_range_usd']
                if (not pair(yes) or not pair(no) or any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12)
                        for a, b in zip(yes + no, [change[0]*(1-q), 0., 0., -change[0]*q]))):
                    return None
        if not re.fullmatch(r'[a-f0-9]{64}', v.get('mechanism_fingerprint_sha256', '')):
            return None
        if (not isinstance(v.get('unmodeled_channels'), list) or not 1 <= len(v['unmodeled_channels']) <= 10
                or any(not isinstance(s, str) or len(s) > 1000 for s in v['unmodeled_channels'])
                or not isinstance(v.get('interpretation'), str) or len(v['interpretation']) > 1500):
            return None
        # Saved exports cannot use extra fields to republish original vendor
        # datasets or smuggle unreviewed model fields into the browser.
        top_keys = {'schema', 'status', 'issuer', 'event_id', 'condition', 'evaluated_at',
            'original_assessment_at', 'separate_later_enrichment', 'source_scope', 'filing_source',
            'equity_source', 'exposure', 'scenarios', 'declared_assumptions',
            'pricing_weights_are_assumptions_not_probabilities', 'physical_probability_P', 'prediction_market_Q',
            'event_to_issuer_interruption_verified', 'total_stock_return_identified',
            'full_portfolio_conditional_pnl_identified', 'primary_impact_allowed', 'economic_ranking_allowed',
            'interpretation', 'unmodeled_channels', 'bounds_method', 'mechanism_fingerprint_sha256',
            'source_or_model_calls', 'source_authenticity_note'}
        if (set(v) != top_keys
                or set(exposure) != {'portfolio_notional_usd', 'original_positions', 'known_issuer_weight', 'unresolved_fund_weight', 'complete_issuer_exposure'}
                or set(cap) != {'source_url', 'response_sha256', 'received_at_utc', 'source_record_id', 'basis'}
                or any(set(p) != {'symbol', 'type', 'weight'} for p in exposure['original_positions'])
                or any(set(r) != {'duration_days', 'company_component_loss_range_usd',
                    'known_position_component_change_range_usd', 'pricing_sensitivity'} for r in rows)
                or any(set(p) != {'assumed_equity_embedded_weight', 'yes_component_range_usd', 'no_component_range_usd'}
                       for r in rows for p in r['pricing_sensitivity'])):
            return None
        return v
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None
