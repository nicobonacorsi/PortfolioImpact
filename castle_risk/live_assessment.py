"""One portfolio -> acquired exposures -> proposals -> exact matches -> scenarios.

All network activity is initiated by the caller. Missing credentials or sources
produce explicit partial results. Public contract fallback never impersonates a
Claude run; historical archives never masquerade as current market receipts.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .adapters import DataAccessError, normalize_quote, parse_time
from .provider_access import ValidatedKalshiClient as KalshiClient
from .acquisition_status import massive_acquisition_stage
from .agent import AgentError, SemanticAgent
from .agent_budgeted_discovery import (
    limited_client_budget, expanded_client_budget, source_first_discovery,
    R59_SEARCH_QUERIES, R59_MARKETS_PER_QUERY, InterruptedDiscovery,
)
from .agentic_bridge import catalog_record
from .event_release_dates import fetch_release_dates
from .economic_binding import review_economic_binding
from .default_event_economics import _betas as select_announcement_day_betas
from .instrument_exposures import build_instrument_exposures
from .instrument_identity import resolve_portfolio_types
from .live_catalog import BoundedKalshiCatalog, CatalogSearchError
from .live_data import MassiveAssessmentClient, acquire_portfolio_inputs, verify_saved_responses
from .live_event_brief import prepare_live_event_brief, rank_events_by_modeled_contrast
from .option1_brief import portfolio_input_fingerprint
from .portfolio import parse_portfolio, recover_exposures
from .semantics import make_contract_target
from .source_question_refinement import derive_source_defined_questions
from .empirical_event_impact import (
    estimate_instrument_betas, fit_event_day_association, fit_event_study,
    beta_scaled_contrasts, event_study_contrasts,
    build_binary_event_impact, build_taiwan_channel_scenario,
)

FAMILY_SERIES = {'FED_POLICY': 'KXFEDDECISION', 'US_INFLATION': 'KXCPI'}
FAMILY_STUDY = {'FED_POLICY': 'FOMC', 'US_INFLATION': 'CPI'}
# Pause before the one extra attempt at a rate-limited catalog search.
RATE_LIMIT_PAUSE_SECONDS = 10
MACRO_QUOTE_MAX_AGE_SECONDS = 300


def _market_series(market):
    return market.series_ticker or market.ticker.split('-')[0]


def refresh_macro_quotes(client, market_objects, payroll_markets, *, requested_series, evaluated_at,
                         expanded=False):
    """Refetch macro receipts after discovery, without changing their clocks.

    A successful GET supplies new Market objects and their original receipt/raw
    response lineage. A failed or cached read never acquires a new timestamp.
    Each retained series is replaced with its final bounded snapshot. Omission
    from that page does not prove a contract has closed or that no market exists.
    Existing strict quote/source checks decide whether the returned data are usable.
    """
    originals = dict(market_objects)
    payrolls = None if payroll_markets is None else dict(payroll_markets)
    ledger = {"status": "NO_REFRESH_NEEDED", "checked_at_utc": evaluated_at.isoformat(),
              "max_receipt_age_seconds": MACRO_QUOTE_MAX_AGE_SECONDS, "series": [],
              "timestamp_scope": "HTTP_RECEIPT_TIME_NOT_EXCHANGE_QUOTE_UPDATE_TIME",
              "timestamps_rewritten": False}
    for series in dict.fromkeys(requested_series):
        source = (payrolls or {}) if series == 'KXPAYROLLS' else originals
        previous = {key: market for key, market in source.items() if _market_series(market) == series}
        if not previous:
            ledger['series'].append({'series': series, 'status': 'NO_PREVIOUS_MACRO_QUOTES_TO_REFRESH',
                                     'contracts_before': 0})
            continue
        ages = []
        for market in previous.values():
            try:
                ages.append((evaluated_at - parse_time(market.quote_timestamp)).total_seconds())
            except (ValueError, TypeError):
                ages.append(None)
        # Refresh every retained series, even if presently inside the ceiling:
        # a previous series' retry may otherwise push a skipped receipt past 300s.
        row = {'series': series, 'contracts_before': len(previous),
               'previous_receipts': sorted({m.retrieved_at for m in previous.values()}),
               'oldest_previous_receipt_age_seconds': max((age for age in ages if age is not None), default=None),
               'request_scope': ('EXPANDED_UP_TO_FOUR_OPEN_MARKET_PAGES_LIMIT_400' if expanded
                                 else 'ONE_OPEN_MARKET_PAGE_LIMIT_100')}
        try:
            fetched = client.list_markets(series_ticker=series,
                                          limit=400 if expanded else 100,
                                          max_pages=4 if expanded else 1)
        except DataAccessError as exc:
            row.update(status='REFRESH_FAILED_ORIGINAL_RECEIPTS_RETAINED', reason=str(exc))
        else:
            replacement = {market.ticker: market for market in fetched}
            if any(_market_series(market) != series for market in fetched):
                row.update(status='REFRESH_REJECTED_DIFFERENT_SERIES',
                           reason='The refresh response contains a different series; original receipts remain unchanged.')
            else:
                if series == 'KXPAYROLLS':
                    payrolls = replacement
                else:
                    originals = {key: market for key, market in originals.items() if _market_series(market) != series}
                    originals.update(replacement)
                row.update(status='REFETCHED_SOURCE_RECEIPTS', contracts_after=len(replacement),
                           refreshed_receipts=sorted({m.retrieved_at for m in fetched}),
                           previously_seen_not_in_bounded_refresh=sorted(set(previous) - set(replacement)),
                           absence_from_refresh_proves_contract_closed=False)
        ledger['series'].append(row)
    attempted = [row for row in ledger['series'] if row['status'].startswith(('REFRESH_', 'REFETCHED_'))]
    if attempted:
        ledger['status'] = ('REFRESH_COMPLETED_SOURCE_GATES_STILL_APPLY'
                            if all(row['status'] == 'REFETCHED_SOURCE_RECEIPTS' for row in attempted)
                            else 'PARTIAL_REFRESH_ORIGINAL_RECEIPTS_NOT_REDATED')
    return originals, payrolls, ledger


def public_contract_fallback(exposures, market_objects, *, evaluated_at, horizon_days=90):
    """Inspect actual, dated contract questions when no Claude proposal is available.

One spread-closest-to-0.5 contract from the earliest expiry per relevant family
is a transparent inspection selection, not an expected-loss ranking. Economic
ranking happens after the independently declared scenarios are calculated.
    """
    events, groups, matches = [], {}, {}
    cutoff = evaluated_at + timedelta(days=horizon_days)
    for family, series in FAMILY_SERIES.items():
        if not exposures['channels_by_family'].get(family):
            continue
        eligible = []
        for market in market_objects.values():
            if market.series_ticker != series and not market.ticker.startswith(series + '-'):
                continue
            try:
                close = datetime.fromisoformat(market.close_time.replace('Z', '+00:00'))
            except (TypeError, ValueError):
                continue
            if not evaluated_at < close <= cutoff:
                continue
            q = normalize_quote(market, now=evaluated_at)
            if q.usable:
                eligible.append((close, abs((q.lower + q.upper) / 2 - .5), market.ticker, market))
        if not eligible:
            continue
        market = min(eligible, key=lambda row: row[:3])[3]
        record = catalog_record(market)
        target = make_contract_target(record)
        target.update(family=family, event_date=market.close_time[:10],
                      candidate_origin='SOURCE_DEFINED_CONTRACT_FALLBACK_NOT_CLAUDE',
                      economic_channel_components=exposures['channels_by_family'][family],
                      search_queries=[])
        events.append(target)
        groups[target['event_id']] = [record]
        matches[target['event_id']] = []
    return {'status': 'PUBLIC_EXACT_CONTRACT_FALLBACK_NOT_CLAUDE', 'events': events,
            'markets_by_event': groups, 'matches_by_event': matches,
            'searches': [], 'quarantined_provider_contracts': {},
            'coverage_status': 'TWO_PUBLIC_SERIES_ONLY_NOT_COMPLETE_MARKET_UNIVERSE'}


def _empty_inputs(now):
    return {'evaluated_at_utc': now.isoformat(), 'instrument_metadata': {}, 'holdings': {},
            'prices_by_symbol': {}, 'price_sources': {}, 'attempts': [],
            'provider_request_count': 0, 'benchmark_symbol': 'SPY',
            'limitations': ['Massive acquisition was not run; missing data are not imputed.']}


def _original_rows_for_live_forecast(report):
    """Bridge the internal exposure tuple to strict JSON-array forecast math.

    `build_instrument_exposures` deliberately keeps the original holdings as
    a tuple in its in-memory report. Both joint-impact engines require a list.
    JSON serialization silently converted that tuple to a list, so archived
    replay passed while the *actual live pipeline* failed before model calls.
    Convert the container only: never change original weights or NAV.
    """
    value = (report.get('portfolio') or {}).get('positions')
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 24:
        raise ValueError('Expected 1–24 funded original holdings in the live report.')
    if not all(isinstance(position, dict) for position in value):
        raise ValueError('Each funded original holding must be an object.')
    return list(value)


def build_assessment_from_evidence(*, portfolio_rows, notional, inputs, discovery,
                                    market_objects, evaluated_at, scenario_settings=None,
                                    release_dates=None, event_observations=None,
                                    refine_source_questions=False, default_scenarios=False,
                                    geopolitical_search=None, payroll_markets=None):
    """Deterministic assessment; injectable source records, no network or clock reads."""
    scenarios = scenario_settings or {}
    identity = resolve_portfolio_types(portfolio_rows,
        instrument_metadata=inputs.get('instrument_metadata'), evaluated_at=evaluated_at,
        require_auto_resolution=False)
    portfolio_rows = identity['portfolio_rows']
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    exposures = build_instrument_exposures(portfolio_rows, notional=notional,
        holdings=inputs.get('holdings'), instrument_metadata=inputs.get('instrument_metadata'),
        evaluated_at=evaluated_at)
    if refine_source_questions:
        refinement = derive_source_defined_questions(discovery, market_objects,
            portfolio_rows=portfolio_rows, notional=notional, evaluated_at=evaluated_at)
        report = refinement['original_brief']
        report['events'].extend(refinement['refinement_brief']['events'])
        report['source_question_refinement'] = {key: deepcopy(refinement[key]) for key in (
            'status', 'refinements', 'exclusions', 'selection_basis', 'max_refinements',
            'original_questions_preserved', 'no_provider_or_model_calls',
            'no_economic_scenarios_transferred', 'limitations')}
    else:
        report = prepare_live_event_brief(discovery, market_objects, portfolio_rows=portfolio_rows,
            notional=notional, evaluated_at=evaluated_at, execution_mode='LIVE')
    prices = inputs.get('prices_by_symbol', {})
    benchmark = inputs.get('benchmark_symbol', 'SPY')
    train_end = (evaluated_at.date() - timedelta(days=365)).isoformat()
    daily = estimate_instrument_betas(prices, benchmark, analysis_as_of=evaluated_at, train_end=train_end)
    studies, surprise_studies = {}, {}
    for family in ('FOMC', 'CPI'):
        dates = (release_dates or {}).get(family, {}).get('dates', [])
        studies[family] = fit_event_day_association(prices, dates, family, benchmark,
            analysis_as_of=evaluated_at, train_end=train_end)
        surprise_studies[family] = fit_event_study(prices, event_observations or [], family,
            analysis_as_of=evaluated_at, train_end=train_end)
    from .two_factor_response import two_factor_fits
    announcement_days = sorted({d for family in ('FOMC', 'CPI')
                                for d in (release_dates or {}).get(family, {}).get('dates', [])})
    two_factor = two_factor_fits([p.symbol for p in portfolio.positions if p.asset_type != 'cash'], prices,
        inputs.get('treasury_10y') or [], benchmark=benchmark, train_end=train_end,
        release_dates=announcement_days)
    from .issuer_factor_evidence import review_issuer_factor
    for symbol, fit in two_factor.items():
        # Independent check of the measured rate loading against the issuer's own
        # dated duration, when one is reviewed for this exact product. A check only:
        # the fit, not the issuer figure, prices the fund.
        issuer = review_issuer_factor(symbol, (inputs.get('instrument_metadata') or {}).get(symbol), evaluated_at)
        if issuer.get('status') == 'ISSUER_FACT_AND_PROVIDER_IDENTITY_REVIEWED' and issuer.get('duration_years') \
                and fit.get('rate_loading_per_bp') is not None:
            fit['issuer_duration_check'] = {
                'issuer_duration_years': issuer['duration_years'], 'fact_as_of': issuer['fact_as_of'],
                'official_source_url': issuer['official_source_url'], 'kind': issuer['kind'],
                'measured_years_per_10y_yield': -fit['rate_loading_per_bp'] * 1e4,
                'measured_se_years': fit['rate_loading_se_per_bp'] * 1e4,
                'note': ('Measured loading is on the 10-year yield alone, so it can differ from effective duration '
                         'when the curve does not move in parallel or spreads move with rates.')}
    ledger = recover_exposures(portfolio, inputs.get('holdings') or {}, now=evaluated_at)
    for event in report.get('events', []):
        family = event.get('family')
        if family == 'OTHER' and re.search(r'\btaiwan\b',
                event.get('title', '') + ' ' + event.get('description', ''), re.I):
            # Qualitative scenario routing supplies no semantic relationship or
            # quote admission. The independent matching decision stays intact.
            family = event['family'] = 'TAIWAN_SUPPLY'
            event['economic_family_basis'] = 'TAIWAN_TEXT_FOR_DECLARED_CHANNEL_SCENARIO_ONLY_NOT_EVENT_CERTIFICATION'
        binding = review_economic_binding(event, scenarios)
        settings = binding.pop('settings')
        event['economic_binding'] = binding
        q_interval = event.get('probability', {}).get('pricing_measure_Q_interval')
        scenario_q = q_interval if (binding['explicit_binding_valid'] or
                                    binding['legacy_numeric_payload_retained']) else None
        if family in FAMILY_STUDY:
            study_family = FAMILY_STUDY[family]
            contrasts = {}
            estimation_basis = 'NO_DECLARED_ECONOMIC_CONTRAST'
            if settings.get('surprise_contrast') is not None:
                contrasts = event_study_contrasts(surprise_studies[study_family],
                    settings['surprise_contrast'], settings.get('surprise_contrast_interval'))
                estimation_basis = 'DOCUMENTED_SURPRISE_STUDY_WITH_DECLARED_BRANCH_OUTCOMES'
            elif settings.get('benchmark_return_contrast') is not None:
                # Use the same chronological selection as the default source
                # model. A failed release-day check keeps the daily point and
                # the hull of both intervals; availability alone is not a pass.
                calibration = dict(studies[study_family])
                selected_rows, _, _, _, _ = select_announcement_day_betas(
                    portfolio, daily, studies[study_family])
                calibration['instruments'] = {
                    row['symbol']: row for row in selected_rows}
                contrasts = beta_scaled_contrasts(calibration, settings['benchmark_return_contrast'],
                    settings.get('benchmark_contrast_interval'), event_family=study_family)
                for row in selected_rows:
                    contrast = contrasts[row['symbol']]
                    if row['basis'] == 'DAILY_BETA_NO_ANNOUNCEMENT_DAY_ESTIMATE':
                        contrast['basis'] = 'DECLARED_BENCHMARK_SCENARIO_TIMES_DAILY_BETA_EVENT_SAMPLE_INSUFFICIENT'
                    elif row['basis'] == 'SPECIFICATION_UNRESOLVED_HULL_OF_DAILY_AND_ANNOUNCEMENT_DAY_INTERVALS':
                        contrast['basis'] = 'DECLARED_BENCHMARK_SCENARIO_TIMES_DAILY_BETA_WITH_ANNOUNCEMENT_DAY_INTERVAL_HULL'
                    elif row['basis'] == 'UNCHANGED_USD_CASH_OR_ZERO_WEIGHT':
                        contrast['basis'] = row['basis']
                estimation_basis = 'CHRONOLOGICALLY_CHECKED_RELEASE_DAY_ASSOCIATION_WITH_DAILY_POINT_AND_INTERVAL_HULL_FALLBACK'
            event['impact'] = build_binary_event_impact(portfolio, contrasts, scenario_q)
            event['impact']['estimation_basis'] = estimation_basis
            event['impact']['declared_settings'] = settings
            event['impact']['binary_indicator_surprise'] = {
                'unit': 'exact_event_indicator_not_policy_bp_or_cpi_pp',
                'expected_indicator': event['impact']['priced_in'].get('q_reference'),
                'YES': None if scenario_q is None else 1 - sum(scenario_q) / 2,
                'NO': None if scenario_q is None else -sum(scenario_q) / 2,
                'definition': 'Outcome indicator minus the declared quote-implied expectation; economic contrast is a separate assumption.'}
        elif family == 'TAIWAN_SUPPLY':
            event['impact'] = build_taiwan_channel_scenario(portfolio, ledger,
                nvda_return_contrast=settings.get('nvda_return_contrast'), q_interval=scenario_q,
                nvda_contrast_interval=settings.get('nvda_contrast_interval'),
                remaining_return_contrast=settings.get('remaining_return_contrast'),
                remaining_contrast_interval=settings.get('remaining_contrast_interval'))
        else:
            event['impact'] = build_binary_event_impact(portfolio,
                settings.get('instrument_contrasts', {}), scenario_q)
        event['impact']['economic_binding_status'] = binding['status']
        if not binding['explicit_binding_valid']:
            event['family_stress'] = deepcopy(event['impact'])
            event['family_stress']['interpretation'] = (
                'Declared family/channel sensitivity only. It is not an identified conditional effect of this event; '
                'identical settings across dates do not imply identical event returns. Do not add parent and child cases.')
            if family != 'TAIWAN_SUPPLY' and not binding['legacy_numeric_payload_retained']:
                event['impact'] = build_binary_event_impact(portfolio, {}, None)
                event['impact'].update(
                    status='NOT_ESTIMATED_EVENT_SPECIFIC_BRANCH_ASSUMPTIONS_REQUIRED',
                    scope='NO_EVENT_CONDITIONAL_PORTFOLIO_RETURN_IDENTIFIED',
                    economic_binding_status=binding['status'])
            elif family == 'TAIWAN_SUPPLY':
                # Preserve the auditable component sensitivity and break-even
                # arithmetic without promoting it to a binary event total.
                event['impact']['event_conditional_total_identified'] = False
                event['impact']['interpretation'] = event['family_stress']['interpretation']
                if event['impact'].get('delta_pnl_usd') is not None:
                    unknown = build_binary_event_impact(portfolio, {}, None)
                    for key in ('delta_pnl_usd', 'delta_pnl_interval_usd', 'priced_in'):
                        event['impact'][key] = unknown[key]
                    event['impact']['scope'] = 'IDENTIFIED_NVDA_CHANNEL_ONLY'
        event['impact']['event_definition'] = event.get('title')
        event['impact']['prior_A6_forward_transfer_status'] = 'FAIL_RETAINED_NOT_OVERRIDDEN'
        event['priced_in'] = deepcopy(event['impact']['priced_in'])
    strict_ranking_inputs = deepcopy(report.get('events', []))
    for event in strict_ranking_inputs:
        if not event['economic_binding']['explicit_binding_valid']:
            event['impact'] = {'delta_pnl_usd': None, 'delta_pnl_interval_usd': None}
    strict_rankings = {event['event_id']: event['ranking']
                      for event in rank_events_by_modeled_contrast(strict_ranking_inputs)}
    report['events'] = rank_events_by_modeled_contrast(report.get('events', []))
    for event in report['events']:
        event['event_ranking'] = strict_rankings[event['event_id']]
        event['ranking']['valid_for_event_ranking'] = event['economic_binding']['explicit_binding_valid']
        if event['economic_binding']['legacy_numeric_payload_retained']:
            event['ranking']['basis'] = 'LEGACY_FAMILY_SENSITIVITY_COMPATIBILITY_ONLY_USE_EVENT_RANKING_FOR_V2'
    fingerprint = portfolio_input_fingerprint(portfolio_rows, notional, inputs.get('holdings'))
    report.update(schema='castle.option1.live_assessment.v2',
        economic_binding_policy='EXACT_EVENT_SPECIFIC_ASSUMPTIONS_REQUIRED_FOR_COMPLETE_CASE',
        evaluated_at_utc=evaluated_at.isoformat(), portfolio=portfolio.to_dict(),
        instrument_identity=identity['audit'],
        input_fingerprint=fingerprint, exposures=exposures,
        empirical={'daily_betas': daily, 'release_day_studies': studies,
                   'documentary_surprise_studies': surprise_studies,
                   'two_factor_responses': two_factor,
                   'train_end': train_end, 'holdout_policy': 'Last 365 calendar days; never refit on holdout',
                   'prior_A6_forward_transfer_status': 'FAIL_RETAINED_NOT_OVERRIDDEN'},
        scenario_settings=scenarios, acquisition_attempts=inputs.get('attempts', []),
        holdings_quality_review=[{
            'symbol': position.symbol,
            'snapshot_present': position.symbol in inputs.get('holdings', {}),
            'reliable': inputs.get('holdings', {}).get(position.symbol, {}).get('reliable', False),
            'complete': inputs.get('holdings', {}).get(position.symbol, {}).get('complete', False),
            'quality_issues': inputs.get('holdings', {}).get(position.symbol, {}).get('metadata', {}).get('quality_issues', []),
            # Aggregate only: the excluded constituent rows themselves stay in the private capture.
            'excluded_negligible_offset_rows': len(inputs.get('holdings', {}).get(position.symbol, {}).get('metadata', {})
                                                   .get('excluded_negligible_offset_rows') or []),
            'excluded_offset_nav': inputs.get('holdings', {}).get(position.symbol, {}).get('metadata', {}).get('excluded_offset_nav', 0.0),
            'scope': 'Saved snapshot quality, separate from the original HTTP attempt status and source-weighted composition review.',
        } for position in portfolio.positions if position.asset_type == 'etf'],
        source_response_references=inputs.get('price_sources', {}),
        release_date_sources=release_dates or {},
        historical_archive_automatically_substituted=False,
        physical_probability_P=None,
        limitations=list(report.get('limitations', [])) + list(inputs.get('limitations', [])) + [
            'A quote-implied pricing weight is not physical P or an observed asset-specific pricing weight.',
            'Scenario outcomes and benchmark branch contrasts are declared assumptions, not inferred from event probability.',
            'Daily and event-day associations do not isolate causal effects; an earlier failed forward test of coefficient transfer stays on record.',
            'Missing impact or matching probability remains unknown; no zero or fabricated beta is substituted.',
            'Current coverage supports long-only USD stock/ETF/cash portfolios, not every financial instrument.',
            'Family stress figures are retained separately; a source-defined child never inherits its parent economic assumptions.',
            'Legacy exact-target numeric payloads are declared family sensitivities and do not pass the v2 complete-case gate without an explicit economic binding.',
        ])
    if default_scenarios:
        from .default_assessment import build_default_assessment
        report['default_analysis'] = build_default_assessment(report,
            market_objects=market_objects, inputs=inputs, evaluated_at=evaluated_at,
            geopolitical_search=geopolitical_search, payroll_markets=payroll_markets)
        report['primary_presentation'] = 'DEFAULT_SOURCE_OUTCOMES_AND_DECLARED_ANNOUNCEMENT_COMPONENTS'
    # This separate, source-hashed ledger does NOT upgrade plain equity holdings,
    # change position accounting or turn bond class into a security-master identity.
    from .source_class_quality import review_portfolio_source_classes
    report['class_aware_fund_reviews'] = review_portfolio_source_classes(inputs.get('holdings') or {},
                                                                         inputs.get('instrument_metadata') or {})
    # A missing event quote does not prevent an independently labelled factor
    # sensitivity. This layer never changes the event's admitted outcome table.
    from .scenario_response_surface import build_response_model
    report['scenario_response_model'] = build_response_model(
        portfolio, prices_by_symbol=prices,
        treasury_yields=inputs.get('treasury_10y') or [],
        two_factor_responses=two_factor, evaluated_at=evaluated_at,
        train_end=train_end, price_sources=inputs.get('price_sources') or {},
        benchmark=benchmark, proxy_prices=inputs.get('scenario_proxy_prices') or {},
        proxy_sources=inputs.get('scenario_proxy_sources') or {},
        release_dates=announcement_days, treasury_source=inputs.get('treasury_source') or {})
    return report


def public_assessment_view(report):
    """Export derived results without vendor price rows or a constituent dataset.

The private in-app report may inspect every holding. Shareable outputs retain
original-position classification, aggregate unresolved/NVDA channel weights,
model coefficients, current public quotes and source hashes only.
    """
    result = deepcopy(report)
    exposures = result.get('exposures', {})
    exposures.pop('agent_exposure_input', None)
    ledger = exposures.get('exposure_ledger', {})
    entries = ledger.pop('entries', [])
    ledger.pop('identified_by_symbol', None)
    aggregate = {}
    for row in entries:
        original = row.get('original_symbol')
        item = aggregate.setdefault(original, {'original_symbol': original, 'total_weight': 0.,
                                               'unresolved_weight': 0., 'identified_nvda_weight': 0.})
        weight = row.get('portfolio_weight', 0.)
        item['total_weight'] += weight
        if row.get('status') == 'unresolved_etf':
            item['unresolved_weight'] += weight
        if row.get('symbol') == 'NVDA' and row.get('status') in {'direct', 'look_through'}:
            item['identified_nvda_weight'] += weight
    ledger['aggregated_original_positions'] = list(aggregate.values())
    ledger['constituent_rows_omitted'] = True

    def aggregate_channels(rows):
        grouped = {}
        for row in rows:
            key = (row.get('original_symbol', row.get('original_position')),
                   row.get('instrument_class'), row.get('channel_status'), row.get('reason'))
            item = grouped.setdefault(key, {'original_symbol': key[0], 'instrument_class': key[1],
                'channel_status': key[2], 'reason': key[3], 'portfolio_weight': 0.})
            item['portfolio_weight'] += row.get('portfolio_weight', 0.)
        return list(grouped.values())
    for family, rows in exposures.get('channels_by_family', {}).items():
        exposures['channels_by_family'][family] = aggregate_channels(rows)
    for event in result.get('events', []):
        for key in ('exposure_channels', 'economic_channel_components'):
            if key in event:
                event[key] = aggregate_channels(event[key])
    omitted = {'raw_path', 'raw_paths', 'normalized_path', 'run_directory',
               'prices_by_symbol', 'scenario_proxy_prices', 'private_replay_inputs', 'holdings', 'request_body',
               'private_business_interruption_inputs', 'private_mexico_tariff_inputs'}
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in omitted}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    result = clean(result)
    result['export_scope'] = 'DERIVED_COEFFICIENTS_AGGREGATE_EXPOSURES_PUBLIC_QUOTES_AND_RESPONSE_HASHES_NO_MASSIVE_RAW_DATA'
    return result


def _research_profile(portfolio, inputs):
    from .discovery_context import exposure_research_profile
    from .fund_composition import FundCompositionError, analyze_holdings_snapshot
    compositions = {}
    for symbol, holdings in (inputs.get('holdings') or {}).items():
        try:
            compositions[symbol] = analyze_holdings_snapshot(holdings)
        except (FundCompositionError, OSError, ValueError, KeyError, TypeError):
            continue
    return exposure_research_profile(portfolio, inputs, compositions)


def _retain_partial_discovery_evidence(fallback: dict, partial: dict | None) -> dict:
    """Keep completed web/scout stages without admitting failed proposals.

    The original public-contract fallback still owns event definitions,
    matches and settlement probabilities. An interrupted LLM stage cannot
    replace those independent objects or their scope diagnostics.
    """
    if not partial:
        return fallback
    for key in ('web_research', 'risk_hypotheses', 'risk_scout_status'):
        if key in partial:
            fallback[key] = deepcopy(partial[key])
    fallback['partial_model_evidence_retained'] = True
    return fallback


def _geopolitical_capture_clients(data_dir, *, expanded_live, gamma_client=None):
    """Build bounded public-only search transports without changing injected budgets.

    Legacy/replay runs intentionally use the adapter's original 30-request cap.
    No network request is performed by client construction.
    """
    if not expanded_live:
        return None, gamma_client
    from .geopolitical_market_search import AuditedKalshiClient, GammaSearchClient
    root = Path(data_dir)
    return (
        AuditedKalshiClient(root / 'kalshi', max_requests=60, max_retries=2, cache_ttl=0),
        gamma_client if gamma_client is not None else GammaSearchClient(
            root / 'polymarket', max_requests=150, max_retries=2, cache_ttl=0),
    )


def _analogue_stage(client, portfolio, notional, events, as_of):
    """Long adjusted daily history for the past-episode replay; failures stay visible."""
    from .analogue_stress import analogue_stress_for_events, episode_set_for
    wanted = [e for e in events if episode_set_for(e.get('family'), ' '.join(
        str(e.get(k) or '') for k in ('title', 'original_question')))]
    if not wanted:
        return {}, {}, {'status': 'NOT_APPLICABLE_NO_EPISODE_FAMILY', 'request_count': 0}
    prices, failures = {}, []
    before = client.request_count
    for position in portfolio.positions:
        if position.asset_type == 'cash' or position.symbol in prices:
            continue
        try:
            prices[position.symbol] = client.daily_prices(
                position.symbol, start=date(2018, 1, 1), end=as_of.date(), max_pages=8)
        except (DataAccessError, ValueError) as exc:
            failures.append({'symbol': position.symbol, 'reason': str(exc)})
    rows = [{'symbol': p.symbol, 'weight': p.weight, 'asset_type': p.asset_type} for p in portfolio.positions]
    result = {} if failures else analogue_stress_for_events(wanted, rows, prices, notional=notional)
    stage = {'status': 'FAILED' if failures else 'MEASURED', 'request_count': client.request_count - before,
             'failures': failures, 'events': len(result)}
    return result, prices, stage


def run_live_assessment(*, portfolio_rows, notional=100000., root=None,
                        massive_key=None, anthropic_key=None, model=None,
                        scenario_settings=None, horizon_days=90, progress=None,
                        massive_client=None, kalshi_client=None, agent=None,
                        release_dates=None, event_observations=None,
                        include_geopolitical_search=False, gamma_client=None, include_payrolls=False,
                        run_directory=None, resume_archives=False):
    """Perform authorized reads and write one private replay bundle + derived report."""
    portfolio_rows = deepcopy(list(portfolio_rows))
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    started = datetime.now(timezone.utc)
    root = Path(root or Path.cwd())
    run_root = (root / 'runtime' / 'option1_live').resolve()
    if run_directory is None:
        run_dir = run_root / started.strftime('%Y%m%dT%H%M%S%fZ')
    else:
        import re
        supplied = Path(run_directory)
        if (not re.fullmatch(r'\d{8}T\d{12}Z', supplied.name) or
                supplied.is_symlink() or supplied.parent.resolve() != run_root):
            raise ValueError('Untrusted analysis directory')
        run_dir = supplied
        if (run_dir / 'derived_assessment.json').exists():
            raise ValueError('Refusing to overwrite a completed analysis')
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_archives and run_directory is None:
        raise ValueError('Cannot resume without a validated saved run directory')
    from .restart_recovery import prime_verified_http_cache
    archive_reuse = {'enabled': bool(resume_archives), 'verified_gets': {},
                     'fresh_final_macro_quotes_required': True,
                     'provenance': 'ORIGINAL_SAVED_RECEIPT_TIMESTAMPS_RETAINED'}
    notify = progress or (lambda message: None)
    massive_key = massive_key or os.environ.get('MASSIVE_API_KEY') or os.environ.get('POLYGON_API_KEY')
    anthropic_key = anthropic_key or os.environ.get('ANTHROPIC_API_KEY')
    # The R59 search breadth is selected only for the real credential-backed
    # application path; injected replay/test transports retain frozen R58 caps.
    expanded_live = bool(anthropic_key) and agent is None
    stages = {}
    notify('Retrieving instrument metadata, every ETF, and completed price sessions.')
    inputs = _empty_inputs(started)
    if massive_client is not None or massive_key:
        client = massive_client or MassiveAssessmentClient(run_dir / 'massive_private', api_key=massive_key,
            max_requests=max(200 if expanded_live else 40,
                             (40 if expanded_live else 12) * len(portfolio.positions)),
            max_retries=1, cache_ttl=0)
        if resume_archives and massive_client is None:
            archive_reuse['verified_gets']['Massive'] = prime_verified_http_cache(
                client, run_dir / 'massive_private')
        inputs = acquire_portfolio_inputs(portfolio_rows, notional=notional, client=client,
                                          evaluated_at=started,
                                          holdings_max_pages=12 if expanded_live and massive_client is None else 5,
                                          include_scenario_proxies=expanded_live and massive_client is None)
        stages['Massive'] = massive_acquisition_stage(inputs, request_count=client.request_count)
    else:
        stages['Massive'] = {'status': 'BLOCKED_MISSING_CREDENTIAL', 'request_count': 0}
    as_of = datetime.now(timezone.utc)
    identity = resolve_portfolio_types(inputs.get('resolved_portfolio_rows', portfolio_rows),
        instrument_metadata=inputs.get('instrument_metadata'), evaluated_at=as_of)
    portfolio_rows = identity['portfolio_rows']
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    inputs['instrument_identity'] = identity['audit']
    exposures = build_instrument_exposures(portfolio_rows, notional=notional,
        holdings=inputs['holdings'], instrument_metadata=inputs['instrument_metadata'], evaluated_at=as_of,
        issuer_siblings=inputs.get('issuer_siblings'))
    notify('Retrieving public current Kalshi contracts and complete settlement rules.')
    # Three retries with backoff (1, 2, 4 s) for 429, 5xx and network failures.
    # A single transient failure otherwise turns a discovery search into ADAPTER_ERROR.
    kalshi = kalshi_client or KalshiClient(run_dir / 'kalshi_public', max_requests=500 if expanded_live else 40,
                                         max_retries=3, cache_ttl=0)
    if resume_archives and kalshi_client is None:
        archive_reuse['verified_gets']['KalshiDiscovery'] = prime_verified_http_cache(
            kalshi, run_dir / 'kalshi_public')
    originals, provider_errors, requested_macro_series = {}, [], []
    for family, series in FAMILY_SERIES.items():
        if not exposures['channels_by_family'].get(family):
            continue
        requested_macro_series.append(series)
        try:
            originals.update({m.ticker: m for m in kalshi.list_markets(
                series_ticker=series, limit=400 if expanded_live else 100,
                max_pages=4 if expanded_live else 1)})
        except DataAccessError as exc:
            provider_errors.append({'series': series, 'reason': str(exc)})
    # Every jobs report in the horizon is a scheduled macro event for any invested
    # portfolio; its Kalshi ladder is read outside the discovery catalog and
    # refreshed alongside the other macro quotes before the final analysis.
    # The app and the CLI enable it, as they enable the geopolitical search.
    payroll_markets = None
    if include_payrolls and any(p.asset_type != 'cash' and p.weight for p in portfolio.positions):
        requested_macro_series.append('KXPAYROLLS')
        payroll_markets = {}
        try:
            payroll_markets = {m.ticker: m for m in kalshi.list_markets(
                series_ticker='KXPAYROLLS', limit=400 if expanded_live else 100,
                max_pages=4 if expanded_live else 1)}
        except DataAccessError as exc:
            provider_errors.append({'series': 'KXPAYROLLS', 'reason': str(exc)})
    discovery = None
    partial_discovery_evidence = None
    catalog = None
    notify('Running exposure-dependent Claude proposals when credentials are available.')
    if agent is not None or anthropic_key:
        try:
            agent = agent or SemanticAgent(api_key=anthropic_key,
                **({'model': model} if model else {}), archive_dir=run_dir / 'claude',
                reuse_verified_archives=bool(resume_archives),
                **(expanded_client_budget() if expanded_live else limited_client_budget()))
            catalog = BoundedKalshiCatalog(kalshi, as_of=as_of.date(), expanded=expanded_live)
            def search(query, limit):
                try:
                    markets = catalog.search(query, limit=limit)
                except CatalogSearchError as exc:
                    # A rate limit that outlasts the client's own retries gets one
                    # more attempt after a pause; both attempts stay in the ledger.
                    if 'HTTP 429' not in str(exc):
                        raise
                    time.sleep(RATE_LIMIT_PAUSE_SECONDS)
                    markets = catalog.search(query, limit=limit)
                originals.update({m.ticker: m for m in markets})
                return [catalog_record(m) for m in markets]
            discovery = source_first_discovery(agent, exposures['agent_exposure_input'], search,
                                              as_of.date().isoformat(), horizon_days=horizon_days,
                                              research_profile=_research_profile(portfolio, inputs),
                                              enable_web_research=(getattr(agent.client, '_is_real_transport', False)
                                                                   is True),
                                              max_searches=R59_SEARCH_QUERIES if expanded_live else 6,
                                              max_markets_per_match=R59_MARKETS_PER_QUERY if expanded_live else 3,
                                              stop_after_first_family_match=not expanded_live)
            stages['Claude'] = {'status': discovery['status'], 'request_count': agent.client.request_count,
                                'model': agent.client.model,
                                'discovery_execution_profile': 'R59_EXPANDED' if expanded_live else 'LEGACY_BOUNDED'}
            stages['WebResearch'] = {
                'status': discovery.get('web_research', {}).get('status', 'NOT_RUN'),
                'source_count': len(discovery.get('web_research', {}).get('sources') or []),
                'search_count': len(discovery.get('web_research', {}).get('searches') or []),
                'evidence_scope': 'DISCOVERY_LEADS_NOT_VERIFIED_PNL_OR_EVENT_PROBABILITY',
            }
            # Legacy status codes distinguish retained review candidates, not
            # raw retrieval counts. Preserve the specific search diagnosis so
            # excluded records are not presented as global market absence.
            for key in ('reason', 'excluded_candidate_count', 'excluded_candidate_count_basis',
                        'source_search_coverage'):
                if key in discovery:
                    stages['Claude'][key] = deepcopy(discovery[key])
        except AgentError as exc:
            stages['Claude'] = {'status': 'FAILED', 'reason': str(exc),
                                'request_count': getattr(getattr(agent, 'client', None), 'request_count', 0)}
            if isinstance(exc, InterruptedDiscovery):
                partial_discovery_evidence = exc.partial_evidence
                web = partial_discovery_evidence['web_research']
                stages['WebResearch'] = {
                    'status': web.get('status', 'NOT_RUN'),
                    'source_count': len(web.get('sources') or []),
                    'search_count': len(web.get('searches') or []),
                    'evidence_scope': 'DISCOVERY_LEADS_NOT_VERIFIED_PNL_OR_EVENT_PROBABILITY',
                    'retained_after_event_proposal_failure': True,
                }
        if catalog is not None:
            # Secret-free per-query strategy, GET count and sanitized failure reason.
            stages['Claude']['catalog_search_ledger'] = list(catalog.search_ledger)
    else:
        stages['Claude'] = {'status': 'BLOCKED_MISSING_CREDENTIAL', 'request_count': 0}
    notify('Checking official release dates and chronological model diagnostics.')
    if release_dates is None:
        release_dates = fetch_release_dates(run_dir / 'official_dates', through=started.date())
    evaluated = datetime.now(timezone.utc)
    used_public_fallback = discovery is None
    if used_public_fallback:
        discovery = public_contract_fallback(exposures, originals, evaluated_at=evaluated, horizon_days=horizon_days)
        discovery = _retain_partial_discovery_evidence(discovery, partial_discovery_evidence)
    geopolitical = None
    if include_geopolitical_search:
        from .geopolitical_market_search import (REVIEWED_SEARCH_FAMILIES, acquire_geopolitical_search,
                                                 documented_issuer_links, public_geopolitical_search_view,
                                                 reviewed_family_events)
        notify('Searching Kalshi and Polymarket for the proposed geopolitical risks.')
        # A held issuer's documented Taiwan, China-export or China-tariff link always
        # puts the reviewed risk on the list, whatever the model proposed; the model can
        # only add to them. The rate channel lists every non-cash ledger row with its class.
        reviewed = reviewed_family_events(discovery.get('events', []),
            documented_issuer_links(exposures['channels_by_family'].get('FED_POLICY', [])),
            start=as_of.date().isoformat(), horizon_days=horizon_days, families=REVIEWED_SEARCH_FAMILIES)
        # The expanded R59/R62 profile documents 60 reads per venue. Previously,
        # the default constructor silently imposed 30 and exhausted the local
        # Gamma budget in the frozen third portfolio. Explicitly budget *both*
        # venues, without mutating injected clients or retroactively upgrading
        # the frozen benchmark result. Replay/legacy runs retain the 30 cap.
        geo_kalshi_client, geo_gamma_client = _geopolitical_capture_clients(
            run_dir / 'geopolitical_public', expanded_live=expanded_live, gamma_client=gamma_client)
        if resume_archives:
            if geo_kalshi_client is not None:
                archive_reuse['verified_gets']['KalshiGeopolitical'] = prime_verified_http_cache(
                    geo_kalshi_client, run_dir / 'geopolitical_public' / 'kalshi')
            if geo_gamma_client is not None and gamma_client is None:
                archive_reuse['verified_gets']['PolymarketGeopolitical'] = prime_verified_http_cache(
                    geo_gamma_client, run_dir / 'geopolitical_public' / 'polymarket')
        # The model's own proposed queries are searched and every Gamma query is
        # read to its last page (an independent R85 audit found a Hormuz query left
        # with pages after the third). The ten-page reviewed bound still applies.
        geopolitical = acquire_geopolitical_search(discovery.get('events', []) + reviewed,
            data_dir=run_dir / 'geopolitical_public', kalshi_client=geo_kalshi_client,
            gamma_client=geo_gamma_client,
            include_target_queries=True, max_gamma_pages=10)
        geo_venues = [venue for target in geopolitical.get('targets', [])
                      for venue in target.get('closest_by_venue', {}).values()]
        geo_failed = any(venue.get('failed_requests', 0) or venue.get('queries_with_unfinished_pages')
                         or venue.get('status') != 'SEARCH_COMPLETED_WITHIN_BOUNDS' for venue in geo_venues)
        geo_count = geopolitical.get('verified_response_records', 0)
        stages['GeopoliticalSearch'] = {
            'status': ('NOT_APPLICABLE_NO_GEOPOLITICAL_TARGETS' if not geopolitical.get('targets') else
                       'FAILED' if not geo_count else 'PARTIAL' if geo_failed else 'RECORDED_BOUNDED_TWO_VENUE_SEARCH'),
            'verified_response_records': geo_count,
            'search_ledger': geopolitical.get('search_ledger', []),
        }
        evaluated = datetime.now(timezone.utc)
    # Model scouting, source lookup and two-venue research may take several
    # minutes. Refresh all retained macro receipts immediately before report assembly.
    # A separate bounded client reserves up to three series x four attempts;
    # discovery cannot consume its request budget or return its cached payload.
    notify('Refreshing macro quotes for the final analysis.')
    quote_client = kalshi if kalshi_client is not None else KalshiClient(
        run_dir / 'kalshi_final_public', max_requests=40 if expanded_live else 12,
        max_retries=3, cache_ttl=0)
    originals, payroll_markets, quote_refresh = refresh_macro_quotes(
        quote_client, originals, payroll_markets, requested_series=requested_macro_series,
        evaluated_at=datetime.now(timezone.utc), expanded=expanded_live)
    evaluated = datetime.now(timezone.utc)
    quote_refresh['evaluation_at_utc'] = evaluated.isoformat()
    if used_public_fallback:
        discovery = public_contract_fallback(exposures, originals, evaluated_at=evaluated, horizon_days=horizon_days)
        # Rebuilding the fallback after refreshing quotes must not erase real
        # web leads and the exposure-specific scout obtained before a timeout.
        discovery = _retain_partial_discovery_evidence(discovery, partial_discovery_evidence)
    for row in quote_refresh['series']:
        if row['status'] in {'REFRESH_FAILED_ORIGINAL_RECEIPTS_RETAINED', 'REFRESH_REJECTED_DIFFERENT_SERIES'}:
            provider_errors.append({'series': row['series'], 'phase': 'FINAL_QUOTE_REFRESH', 'reason': row['reason']})
    stages['Kalshi'] = {'status': 'PARTIAL' if provider_errors else 'READS_COMPLETED',
                        'request_count': kalshi.request_count + (quote_client.request_count if quote_client is not kalshi else 0),
                        'contracts_retrieved': len(originals),
                        'quote_refresh': quote_refresh,
                        'failures': provider_errors}
    build_kwargs = {
        'portfolio_rows': portfolio_rows, 'notional': notional,
        'inputs': inputs, 'discovery': discovery, 'market_objects': originals,
        'evaluated_at': evaluated, 'scenario_settings': scenario_settings,
        'release_dates': release_dates, 'event_observations': event_observations,
        'refine_source_questions': True, 'default_scenarios': True,
        'geopolitical_search': geopolitical, 'payroll_markets': payroll_markets}
    report = build_assessment_from_evidence(**build_kwargs)
    # Price-in evidence is a descriptive check of already completed marks, not
    # an attributed geopolitical repricing fraction, P(E), or a future forecast.
    # Reuse the synchronised original-holding Massive sessions acquired above;
    # never fetch a second proxy or add ETF constituent NAV.
    from .observed_portfolio_mark import observed_portfolio_mark
    report["observed_mark_context"] = observed_portfolio_mark(
        _original_rows_for_live_forecast(report), notional,
        inputs.get("prices_by_symbol"), as_of_utc=report["evaluated_at_utc"])
    observed_status = report["observed_mark_context"]["status"]
    stages["PriceInObservation"] = {
        "status": observed_status,
        "sessions": report["observed_mark_context"].get("synchronized_sessions", 0),
        "event_attribution_identified": False,
        "equity_price_in_fraction_identified": False,
    }
    # Keep the issuer-risk scouting layer independent of the four binary-event
    # proposal slots. No related Kalshi quote, family coefficient, or synthetic
    # probability is transferred to a broad source-unverified hazard.
    report['risk_watchlist'] = list(discovery.get('risk_hypotheses') or [])
    report['risk_watchlist_status'] = discovery.get('risk_scout_status', 'NOT_EXECUTED')
    report['web_research'] = deepcopy(discovery.get('web_research') or
                                      {'status': 'NOT_RUN', 'sources': [], 'searches': []})
    # Keep broad model proposals and exact source settlement questions separate.
    # The bounded supplement uses only already-acquired coherent source models.
    from .source_question_completion import supplement_source_questions
    report, build_kwargs = supplement_source_questions(
        report, build_kwargs=build_kwargs, horizon_days=horizon_days)
    # Promote only exact source outcome contrasts supported by the assembled
    # joint source probability model. No new network or inference calls.
    from .exact_event_binding import close_exact_source_scenarios
    supplementation = report.get('source_question_completion')
    report = close_exact_source_scenarios(report, build_kwargs=build_kwargs,
                                        scenario_settings=scenario_settings)
    if supplementation is not None:
        report['source_question_completion'] = supplementation
    if geopolitical is not None:
        report['geopolitical_search'] = public_geopolitical_search_view(geopolitical)
    report['acquisition_quality'] = inputs.get('acquisition_quality', {})
    if massive_client is not None or massive_key:
        notify('Replaying your holdings through comparable past episodes.')
        report['analogue_stress'], inputs['analogue_prices'], stages['Analogues'] = _analogue_stage(
            client, portfolio, notional, (report.get('default_analysis') or {}).get('events') or [], as_of)
    # R94: release-day band of this exact portfolio on past Kalshi-listed releases (derived; no requests).
    from .announcement_band import compute_report_bands
    try:
        report['announcement_bands'] = compute_report_bands(portfolio, inputs.get('prices_by_symbol') or {},
                                                            as_of=as_of.date().isoformat(), notional=notional)
    except Exception as exc:  # a failed band never blocks the assessment
        report['announcement_bands'] = {'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
    # R115 autonomous original-event forecasting, independent of contract Q and
    # the portfolio's already computed conditional P&L. Only the real,
    # credential-authorized interactive analysis may incur this extra model call.
    # Recorded examples and injected transports remain deterministic replays.
    report['autonomous_event_forecasts'] = {}
    point2_ledger = []
    if (expanded_live and agent is not None and
            getattr(getattr(agent, 'client', None), '_is_real_transport', False) is True):
        from .probability_assumptions import probability_question_context
        from .structural_event_forecast import forecast_events, MAX_EVENTS
        candidates = []
        for event in (report.get('default_analysis') or {}).get('events') or []:
            if not isinstance(event, dict):
                continue
            candidate = {'id': event.get('event_id'), 'title': event.get('title'),
                         'date': event.get('event_date'),
                         'source_defined_scenario': event.get('source_defined_scenario') is True}
            context = probability_question_context(
                {'capture': report.get('evaluated_at_utc')}, candidate)
            if context and context['status'] == 'OPEN_FOR_DECLARED_EVENT_FORECAST':
                if not any(existing['event_id'] == context['event_id'] for existing in candidates):
                    candidates.append(context)
            if len(candidates) >= MAX_EVENTS:
                break
        if candidates:
            # R125: event-targeted evidence and a generic two-venue contract
            # review precede the forecaster, with their own bounded budgets.
            # Their failures are recorded; the forecaster sees only what was
            # actually retrieved and never a substituted quote.
            notify('Searching event-specific evidence and prediction-market contracts for each original event.')
            from .point2_evidence_stage import run_point2_evidence_stage
            point2 = run_point2_evidence_stage(
                candidates, run_dir=run_dir, anthropic_key=anthropic_key, model=model, as_of=as_of,
                discovery_events=list(discovery.get('events') or []),
                default_events=list((report.get('default_analysis') or {}).get('events') or []),
                geopolitical_search=geopolitical, progress=notify,
                portfolio_search=report.get('web_research'),
                resume_archives=bool(resume_archives), archive_reuse=archive_reuse['verified_gets'])
            stages['EventEvidence'] = point2['stage']
            point2_ledger = point2['ledger']
            report['event_evidence_research'] = point2['public_evidence']
            report['event_market_reviews'] = point2['market_reviews']
            report['point2_catalog_search_ledger'] = point2.get('catalog_search_ledger', [])
            notify('Evaluating original-event probabilities and structural uncertainty.')
            try:
                from functools import partial
                from .official_event_evidence import retrieve_source_excerpts
                from .official_macro_evidence import retrieve_cpi_snapshot
                page_fetch_attempts = []
                report['point2_page_fetch_attempts'] = page_fetch_attempts
                forecasts = forecast_events(
                    candidates, report.get('web_research'), agent.client,
                    official_indicator_fetcher=retrieve_cpi_snapshot,
                    source_excerpt_fetcher=partial(retrieve_source_excerpts, attempt_ledger=page_fetch_attempts),
                    event_evidence=point2['evidence'],
                    market_reviews=point2['market_reviews'],
                    definition_details=point2['definition_details'],
                    replicates=2)
                report['autonomous_event_forecasts'] = forecasts
                accepted = sum(v.get('status') == 'MODEL_FORECAST_UNCALIBRATED'
                               for v in forecasts.values())
                stages['AutonomousForecast'] = {
                    'status': 'UNCALIBRATED_MODEL_RESULTS' if accepted else 'ABSTAINED',
                    'eligible_event_count': len(candidates),
                    'model_estimate_count': accepted,
                    'abstention_count': len(forecasts) - accepted,
                    'source_assurance': 'PER_EVENT_TARGETED_LEADS_DIGESTS_TIERED_PAGE_TEXT_AND_REVIEWED_CONTRACTS',
                    'direct_primary_page_observations': sum(
                        v.get('primary_page_observation_count', 0)
                        for v in forecasts.values()),
                    'direct_specialized_page_observations': sum(
                        v.get('specialized_page_observation_count', 0)
                        for v in forecasts.values()),
                    'abstention_classes': sorted(
                        str(v.get('abstention_class')) for v in forecasts.values()
                        if v.get('status') == 'ABSTAIN'),
                    'events_with_exact_market_quote': sum(
                        1 for v in forecasts.values() if v.get('exact_market_quotes')),
                    'replicates_per_event': 2,
                    'stability_statuses': sorted(
                        str((v.get('stability') or {}).get('status')) for v in forecasts.values()
                        if isinstance(v.get('stability'), dict)),
                    'evidence_sufficiency_statuses': sorted(
                        str(v.get('evidence_sufficiency')) for v in forecasts.values()
                        if v.get('status') == 'MODEL_FORECAST_UNCALIBRATED'),
                    'independent_primary_fact_verification': False,
                    'official_bls_indicator_observations': sum(
                        v.get('official_indicator_observation_count', 0)
                        for v in forecasts.values()),
                    'empirical_backtest_completed': False,
                }
            except (ValueError, TypeError, AgentError) as exc:
                stages['AutonomousForecast'] = {
                    'status': 'FAILED_CLOSED', 'eligible_event_count': len(candidates),
                    'model_estimate_count': 0, 'reason': type(exc).__name__,
                }
        else:
            stages['AutonomousForecast'] = {
                'status': 'NO_ELIGIBLE_ORIGINAL_EVENT', 'model_estimate_count': 0,
            }
    else:
        stages['AutonomousForecast'] = {
            'status': 'NOT_RUN_RECORDED_OR_NONINTERACTIVE',
            'model_estimate_count': 0,
        }
    # R116 point 3: one independent, explicitly uncalibrated scenario-stress
    # component for original event targets. It never overwrites the source A
    # response or the static analyst B hypothesis, and never reads event P/Q.
    report['autonomous_portfolio_impacts'] = {}
    if (expanded_live and agent is not None and
            getattr(getattr(agent, 'client', None), '_is_real_transport', False) is True):
        from .probability_assumptions import probability_question_context
        from .autonomous_event_impact import forecast_portfolio_impacts, MAX_EVENTS as IMPACT_MAX_EVENTS
        impact_candidates = []
        for event in (report.get('default_analysis') or {}).get('events') or []:
            if not isinstance(event, dict):
                continue
            candidate = {'id': event.get('event_id'), 'title': event.get('title'),
                         'date': event.get('event_date'),
                         'source_defined_scenario': event.get('source_defined_scenario') is True}
            context = probability_question_context(
                {'capture': report.get('evaluated_at_utc')}, candidate)
            if context and context.get('status') == 'OPEN_FOR_DECLARED_EVENT_FORECAST':
                if not any(c['event_id'] == context['event_id'] for c in impact_candidates):
                    impact_candidates.append(context)
            if len(impact_candidates) >= IMPACT_MAX_EVENTS:
                break
        if impact_candidates:
            notify('Checking 20-day event-conditional portfolio stress assumptions.')
            try:
                impacts = forecast_portfolio_impacts(
                    impact_candidates, _original_rows_for_live_forecast(report), notional,
                    report.get('web_research'), agent.client,
                    event_evidence=report.get('event_evidence_research'),
                    exposure_context=exposures.get('agent_exposure_input'),
                    observed_mark_context=report.get('observed_mark_context'))
                report['autonomous_portfolio_impacts'] = impacts
                admitted = sum(row.get('status') == 'MODEL_STRESS_UNCALIBRATED'
                               for row in impacts.values())
                paired_admitted = sum(
                    (row.get('paired_today_mark') or {}).get('status') ==
                    'MODEL_TODAY_MARK_STRESS_UNCALIBRATED' for row in impacts.values())
                stages['AutonomousImpact'] = {
                    'status': 'UNCALIBRATED_JOINT_STRESS' if admitted else 'ABSTAINED',
                    'event_count': len(impact_candidates),
                    'model_scenario_count': admitted,
                    'modeled_yes_from_today_scenario_count': paired_admitted,
                    'abstention_count': len(impacts) - admitted,
                    'return_horizon_trading_sessions': 20,
                    'provenance': 'MODEL_PROPOSED_JOINT_VECTORS_PUBLIC_SEARCH_LEADS',
                    'empirical_event_impact_validation': 'NOT_ESTABLISHED',
                    'portfolio_pnl_from_today_identified': False,
                }
            except (ValueError, TypeError, AgentError) as exc:
                stages['AutonomousImpact'] = {
                    'status': 'FAILED_CLOSED', 'event_count': len(impact_candidates),
                    'model_scenario_count': 0, 'reason': type(exc).__name__,
                    'input_validation_detail': (
                        str(exc)[:160] if type(exc).__name__ == 'JointImpactInputError' else None),
                }
        else:
            stages['AutonomousImpact'] = {
                'status': 'NO_PRECISE_ORIGINAL_EVENT', 'model_scenario_count': 0,
            }
    else:
        stages['AutonomousImpact'] = {
            'status': 'NOT_RUN_RECORDED_OR_NONINTERACTIVE', 'model_scenario_count': 0,
        }
    # R122 point 3: paired YES/NO scenarios marked at ONE calendar-fixed common
    # endpoint per event (versioned successor to the R118 per-branch onset
    # horizon). Separate output; R116/R118 results and A/B evidence untouched.
    report['common_endpoint_impacts'] = {}
    if (expanded_live and agent is not None and
            getattr(getattr(agent, 'client', None), '_is_real_transport', False) is True):
        from .probability_assumptions import probability_question_context
        from .autonomous_common_endpoint_impact import (
            forecast_common_endpoint_impacts, MAX_EVENTS as ENDPOINT_MAX_EVENTS)
        endpoint_candidates = []
        for event in (report.get('default_analysis') or {}).get('events') or []:
            if not isinstance(event, dict):
                continue
            candidate = {'id': event.get('event_id'), 'title': event.get('title'),
                         'date': event.get('event_date'),
                         'source_defined_scenario': event.get('source_defined_scenario') is True}
            context = probability_question_context(
                {'capture': report.get('evaluated_at_utc')}, candidate)
            if context and context.get('status') == 'OPEN_FOR_DECLARED_EVENT_FORECAST':
                if not any(c['event_id'] == context['event_id'] for c in endpoint_candidates):
                    endpoint_candidates.append(context)
            if len(endpoint_candidates) >= ENDPOINT_MAX_EVENTS:
                break
        if endpoint_candidates:
            notify('Checking common-endpoint event YES/NO scenario assumptions.')
            try:
                endpoint_impacts = forecast_common_endpoint_impacts(
                    endpoint_candidates, _original_rows_for_live_forecast(report), notional,
                    report.get('web_research'), agent.client,
                    event_evidence=report.get('event_evidence_research'),
                    observed_mark_context=report.get('observed_mark_context'),
                    exposure_context=exposures.get('agent_exposure_input'),
                    prices_by_symbol=inputs.get('prices_by_symbol'))
                report['common_endpoint_impacts'] = endpoint_impacts
                admitted_endpoint = sum(
                    row.get('status') == 'MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED'
                    for row in endpoint_impacts.values())
                stages['CommonEndpointImpact'] = {
                    'status': 'UNCALIBRATED_COMMON_ENDPOINT_STRESS' if admitted_endpoint else 'ABSTAINED',
                    'event_count': len(endpoint_candidates),
                    'model_scenario_count': admitted_endpoint,
                    'abstention_count': len(endpoint_impacts) - admitted_endpoint,
                    'endpoint_convention': 'ONE_CALENDAR_FIXED_SESSION_PER_EVENT_FOR_YES_AND_NO',
                    'provenance': 'MODEL_PROPOSED_JOINT_VECTORS_PUBLIC_SEARCH_LEADS',
                    'empirical_event_impact_validation': 'NOT_ESTABLISHED',
                    'portfolio_pnl_from_today_identified': False,
                }
            except (ValueError, TypeError, AgentError) as exc:
                stages['CommonEndpointImpact'] = {
                    'status': 'FAILED_CLOSED', 'event_count': len(endpoint_candidates),
                    'model_scenario_count': 0, 'reason': type(exc).__name__,
                    'input_validation_detail': (
                        str(exc)[:160] if type(exc).__name__ == 'JointImpactInputError' else None),
                }
        else:
            stages['CommonEndpointImpact'] = {
                'status': 'NO_PRECISE_ORIGINAL_EVENT', 'model_scenario_count': 0,
            }
    else:
        stages['CommonEndpointImpact'] = {
            'status': 'NOT_RUN_RECORDED_OR_NONINTERACTIVE', 'model_scenario_count': 0,
        }
    report['stages'] = stages
    archive_reuse['verified_http_archives_primed'] = sum(archive_reuse['verified_gets'].values())
    archive_reuse['model_archive_replays'] = sum(
        entry.get('status') == 'REPLAYED_VALIDATED_ARCHIVE'
        for entry in list(getattr(getattr(agent, 'client', None), 'request_ledger', [])) + list(point2_ledger))
    report['archive_reuse'] = archive_reuse
    report['started_at_utc'] = started.isoformat()
    missing_credential = any(s.get('status') == 'BLOCKED_MISSING_CREDENTIAL' for s in stages.values())
    stage_gaps = (stages['Massive']['status'] != 'ACQUIRED' or
                  stages['Claude']['status'] != 'COMPLETE_SOURCE_BACKED_REVIEW' or
                  stages['Kalshi']['status'] != 'READS_COMPLETED' or
                  stages.get('GeopoliticalSearch', {}).get('status') in {'FAILED', 'PARTIAL'})
    report['verification_status'] = ('PARTIAL_PUBLIC_RUN_AUTHENTICATED_ACCEPTANCE_NOT_COMPLETED'
        if missing_credential else 'PARTIAL_LIVE_RUN_COVERAGE_OR_REVIEW_GAPS' if stage_gaps
        else 'LIVE_READS_RECORDED_REVIEW_EACH_COVERAGE_AND_MODEL_GATE')
    report['provider_receipts'] = verify_saved_responses(run_dir)
    report['model_request_ledger'] = (list(getattr(getattr(agent, 'client', None), 'request_ledger', []))
                                      + list(point2_ledger))
    # This private bundle contains provider price/holdings inputs. Never include
    # it in the shareable ZIP or a UI export without a redistribution review.
    private_bundle = {'inputs': inputs, 'discovery': build_kwargs['discovery'],
                      'original_discovery': discovery,
                      'market_objects': {k: v.to_dict() for k, v in originals.items()},
                      'payroll_markets': {k: v.to_dict() for k, v in (payroll_markets or {}).items()},
                      'event_observations': event_observations or []}
    (run_dir / 'private_replay_inputs.json').write_text(json.dumps(private_bundle, indent=2, allow_nan=False), encoding='utf-8')
    report['run_directory'] = str(run_dir)
    report_path = run_dir / 'derived_assessment.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (run_dir / 'derived_assessment.sha256').write_text(digest + '  derived_assessment.json\n', encoding='utf-8')
    return report
