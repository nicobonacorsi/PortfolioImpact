"""Portfolio-dependent selection from the existing bounded offline source catalog.

The ranking measures linked portfolio NAV, not expected loss, event probability,
return sensitivity, or the likelihood of a price move. No source is acquired.
Original Kalshi objects and strict source engines are reused without mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from .a8_offline_demo import archived_a8_demo
from .nonmacro_primary_desk_case import taiwan_primary_page_desk_case
from .portfolio import Exposure, parse_portfolio, recover_exposures

# Instrument identity and type must both match. This is deliberately a bounded
# registry, not an inference that any supplied ticker is a verified security.
_EQUITY_FUNDS = frozenset({'QQQ', 'SPY', 'XLE'})
_TREASURY_FUNDS = frozenset({'TLT', 'IEF'})
_KNOWN_STOCKS = frozenset({'NVDA'})
_MACRO_FAMILIES = {'archive-fomc-oct26': 'FED_POLICY',
                   'archive-cpi-oct26': 'US_INFLATION'}
_TAIWAN_EVENT_ID = 'documentary-taiwan-commercial-blockade-2026'


def _instant(value: datetime | str | None) -> datetime:
    value = datetime.now(timezone.utc) if value is None else value
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Evaluation time must be a timezone-aware datetime or ISO timestamp.')
    return value.astimezone(timezone.utc)


def _component_channel(entry: Exposure, parents: dict[str, Any], family: str) -> dict | None:
    """Return one channel per existing accounting slice, never parent plus child."""
    parent = parents[entry.original_symbol]
    if entry.asset_type == 'cash' and entry.status != 'unresolved_etf':
        return None
    known_stock = (entry.status != 'unresolved_etf' and entry.asset_type == 'stock'
                   and entry.symbol in _KNOWN_STOCKS)
    if family == 'TAIWAN_SUPPLY':
        if not known_stock or entry.symbol != 'NVDA':
            return None
        reason = ('NVIDIA identifies TSMC among its semiconductor foundries. A Taiwan '
                  'commercial-access disruption is a plausible supply-chain channel; '
                  'neither the price effect nor contract equivalence follows from this link.')
        source = ('https://www.sec.gov/Archives/edgar/data/1045810/'
                  '000104581026000021/nvda-20260125.htm')
        kind = 'IDENTIFIED_ISSUER_SUPPLY_CHAIN'
        specific = True
    elif known_stock:
        reason = ('A policy-rate or inflation surprise can change discount rates and '
                  'earnings expectations for the identified equity. This is a broad '
                  'economic pathway, not a measured NVIDIA event sensitivity.')
        source = 'docs/PORTFOLIO_DEPENDENT_DISCOVERY.md'
        kind = 'IDENTIFIED_EQUITY_BROAD_MACRO_CHANNEL'
        specific = False
    elif parent.asset_type == 'etf' and parent.symbol in _TREASURY_FUNDS:
        reason = ('The original Treasury ETF has a nominal-yield/duration channel. '
                  'The amount linked is its existing NAV slice; holdings identity '
                  'and the event-induced yield move remain separate questions.')
        source = ('data/evidence/impact/tlt_duration.normalized.json' if parent.symbol == 'TLT'
                  else 'docs/PORTFOLIO_DEPENDENT_DISCOVERY.md')
        kind = 'TREASURY_FUND_PARENT_CHANNEL'
        specific = True
    elif parent.asset_type == 'etf' and parent.symbol in _EQUITY_FUNDS:
        reason = ('The original equity ETF has broad discount-rate and earnings channels. '
                  'This parent-instrument classification does not identify any constituent '
                  'or estimate an event beta. Each accounting slice is counted once.')
        source = 'docs/PORTFOLIO_DEPENDENT_DISCOVERY.md'
        kind = 'EQUITY_FUND_PARENT_CHANNEL'
        specific = False
    else:
        return None
    return {
        'instrument': entry.symbol,
        'original_position': entry.original_symbol,
        'asset_type': entry.asset_type,
        'identity_status': entry.status,
        'channel_kind': kind,
        'portfolio_weight': entry.portfolio_weight,
        'specific_channel': specific,
        'reason': reason,
        'evidence_reference': source,
        'holdings_as_of': entry.as_of,
        'holdings_retrieved_at': entry.retrieved_at,
        'holdings_source_url': entry.source_url,
        'estimated_return_sensitivity': None,
    }


def discover_archived_portfolio_events(
    *, portfolio_rows: list[dict[str, Any]], notional: float = 100_000.0,
    holdings: Mapping[str, Any] | None = None,
    evaluated_at: datetime | str | None = None, horizon_days: int = 90,
) -> dict[str, Any]:
    """Select and order existing sourced event candidates for the supplied portfolio.

    ``events`` is the JSON-compatible headline shortlist; ``macro_discovery``
    and ``originals`` can be passed directly to the existing A8 investigation.
    ``originals`` contains normalized Market objects and should not be JSON
    serialized. Exclude that key when exporting this report.

    The current portfolio is assessed at evaluated_at, while source receipt
    clocks remain historical. No historical ownership or point-in-time backtest
    is claimed. No provider/model call is made, including for unknown holdings.
    """
    if not isinstance(portfolio_rows, (list, tuple)) or not 1 <= len(portfolio_rows) <= 128:
        raise ValueError('One to 128 explicit portfolio rows are required.')
    if any(not isinstance(row, Mapping) for row in portfolio_rows):
        raise ValueError('Each portfolio row must be a mapping.')
    if holdings is not None and not isinstance(holdings, Mapping):
        raise ValueError('Holdings must be a keyed snapshot mapping or None.')
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or not 1 <= horizon_days <= 366:
        raise ValueError('horizon_days must be an integer from 1 to 366.')
    now = _instant(evaluated_at)
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    ledger = recover_exposures(portfolio, holdings or {}, now=now)
    parents = {p.symbol: p for p in portfolio.positions}
    raw_discovery, originals = archived_a8_demo()
    end_date = now.date() + timedelta(days=horizon_days)

    catalog = []
    for event in raw_discovery['events']:
        records = raw_discovery['markets_by_event'][event['event_id']]
        # The inherited loader re-reads source bytes. Do not let an unverified
        # record or a receipt after evaluation acquire a historical source claim.
        if any(r.get('source_receipt_status') != 'LOCAL_BYTES_VERIFIED' for r in records):
            raise ValueError('Archived source receipt failed independent byte verification.')
        receipts = sorted({r['retrieved_at'] for r in records})
        catalog.append({
            **deepcopy(event), 'family': _MACRO_FAMILIES[event['event_id']],
            'source_class': 'HASH_VERIFIED_ARCHIVED_KALSHI_CONTRACT_FAMILY',
            'source_available_at_utc': max(receipts, key=_instant),
            'source_receipts_utc': receipts,
            'source_urls': sorted({r['source_request_url'] for r in records}),
            'source_sha256': sorted({r['source_response_sha256'] for r in records}),
            'matched_contract_ids': [r['ticker'] for r in records],
            'matching_scope': 'SOURCE_FAMILY_RELEVANCE_ONLY_EXACT_YES_RULES_REQUIRE_INSPECTION',
            'probability_scope': 'HISTORICAL_EXACT_CONTRACT_Q_REFERENCES_NOT_BROAD_EVENT_PHYSICAL_P',
            'probability_next_action': 'Inspect a precise source-defined YES question and its dated bid/ask bounds.',
        })
    desk = taiwan_primary_page_desk_case(portfolio_rows=portfolio_rows, notional=notional)
    catalog.append({
        'event_id': _TAIWAN_EVENT_ID,
        'title': 'Taiwan main-island commercial-access blockade during 2026',
        'description': 'A supply-chain candidate from the existing manually reviewed primary-page desk case.',
        'event_date': '2026-12-31', 'family': 'TAIWAN_SUPPLY',
        'definition_origin': 'EXISTING_DOCUMENTARY_PRIMARY_PAGE_REVIEW_NOT_AGENT_CAPTURE',
        'source_class': 'DOCUMENTARY_PAGE_REVIEW_WITHOUT_ORIGINAL_PROVIDER_RECEIPT',
        # Only the review date is known. Do not invent a time within that day.
        'source_review_date': desk['research_review_as_of'],
        'source_available_at_utc': None, 'source_receipts_utc': [],
        'source_urls': [x['url'] for x in desk['sources']], 'source_sha256': [],
        'matched_contract_ids': [],
        'matching_scope': 'NOTHING_CERTIFIED_FROM_LOCAL_PROVIDER_BYTES',
        'probability_scope': 'UNAVAILABLE_NO_ADMITTED_CONTRACT_PRICE',
        'probability_next_action': 'Obtain original complete rules and a dated quote before assigning a probability.',
        'non_equivalence': desk['pairwise_settlement_review'],
    })

    candidates, excluded = [], []
    for candidate in catalog:
        event_date = datetime.fromisoformat(candidate['event_date']).date()
        reason = None
        if event_date < now.date():
            reason = 'EVENT_DEADLINE_PRECEDES_EVALUATION_DATE'
        elif event_date > end_date:
            reason = 'EVENT_DEADLINE_OUTSIDE_REQUESTED_HORIZON'
        elif candidate['source_available_at_utc'] is not None and _instant(candidate['source_available_at_utc']) > now:
            reason = 'SOURCE_RECEIPT_AFTER_EVALUATION_TIME'
        elif candidate.get('source_review_date') and now.date().isoformat() <= candidate['source_review_date']:
            reason = 'DOCUMENTARY_REVIEW_NOT_KNOWN_BEFORE_END_OF_REVIEW_DATE'
        if reason:
            excluded.append({'event_id': candidate['event_id'], 'title': candidate['title'], 'reason': reason})
            continue
        components = [c for entry in ledger.entries if entry.portfolio_weight > 0
                      if (c := _component_channel(entry, parents, candidate['family'])) is not None]
        linked = math.fsum(c['portfolio_weight'] for c in components)
        if linked <= 0:
            excluded.append({'event_id': candidate['event_id'], 'title': candidate['title'],
                             'reason': 'NO_DOCUMENTED_SUPPORTED_EXPOSURE_CHANNEL_IN_THIS_CATALOG'})
            continue
        if linked > 1 + 1e-8:
            raise ValueError('Discovery relevance would count portfolio NAV more than once.')
        for c in components:
            c['notional_usd'] = c['portfolio_weight'] * portfolio.notional
        specific = math.fsum(c['portfolio_weight'] for c in components if c['specific_channel'])
        candidates.append({
            **candidate,
            'relevance_portfolio_weight': linked,
            'relevance_notional_usd': linked * portfolio.notional,
            'specific_channel_weight_for_tie_break': specific,
            'relevance_components': components,
            'relevance_measure': 'UNWEIGHTED_LINKED_NAV_NOT_EXPECTED_LOSS_OR_MOVE_PROBABILITY',
            'physical_probability_P': None, 'current_quote_Q': None,
            'estimated_portfolio_move': None,
        })
    candidates.sort(key=lambda e: (-e['relevance_portfolio_weight'],
                                    -e['specific_channel_weight_for_tie_break'],
                                    e['event_date'], e['event_id']))
    for rank, candidate in enumerate(candidates, 1):
        candidate['inspection_priority'] = rank

    # Mapping coverage is separate from whether this finite catalog has an event
    # inside today's requested horizon. ETF identity uncertainty remains intact.
    supported, unsupported, cash = [], [], []
    for entry in ledger.entries:
        if entry.asset_type == 'cash' and entry.status != 'unresolved_etf':
            cash.append(entry.portfolio_weight)
        elif _component_channel(entry, parents, 'FED_POLICY') is not None:
            supported.append(entry.portfolio_weight)
        else:
            unsupported.append({'instrument': entry.symbol, 'original_position': entry.original_symbol,
                                'identity_status': entry.status, 'weight': entry.portfolio_weight})
    supported_weight = math.fsum(supported)
    unsupported_weight = math.fsum(c['weight'] for c in unsupported)
    cash_weight = math.fsum(cash)
    if not math.isclose(supported_weight + unsupported_weight + cash_weight, ledger.total_weight, abs_tol=1e-8):
        raise ValueError('Supported, unsupported and cash mass do not reconcile.')

    selected_macro_ids = [e['event_id'] for e in candidates if e['event_id'] in _MACRO_FAMILIES]
    selected_ids = set(selected_macro_ids)
    macro = deepcopy(raw_discovery)
    by_id = {e['event_id']: e for e in macro['events']}
    macro['events'] = [by_id[eid] for eid in selected_macro_ids]
    for key in ('markets_by_event', 'quarantined_provider_contracts', 'calendar_reviews_by_event',
                'matches_by_event', 'match_statuses'):
        macro[key] = {k: v for k, v in macro.get(key, {}).items() if k in selected_ids}
    macro['source_qualified_relevance_count'] = sum(len(v) for v in macro['markets_by_event'].values())
    macro['quarantined_count'] = sum(len(v) for v in macro['quarantined_provider_contracts'].values())
    macro['coverage_status'] = 'PORTFOLIO_FILTERED_BOUNDED_HISTORICAL_CATALOG_NOT_CURRENT_VENUE_COVERAGE'
    macro['portfolio_selection_method'] = 'EXPLICIT_SUPPORTED_EXPOSURE_CHANNELS_NO_FABRICATED_BETAS'
    selected_tickers = {m['ticker'] for records in macro['markets_by_event'].values() for m in records}
    return {
        'schema': 'castle.option1.portfolio_event_discovery.v1',
        'status': ('PORTFOLIO_DEPENDENT_ARCHIVED_CANDIDATES' if candidates
                   else 'NOTHING_SUPPORTED_IN_BOUNDED_CATALOG'),
        'evaluated_at_utc': now.isoformat(), 'horizon_days': horizon_days,
        'portfolio': portfolio.to_dict(), 'exposure_ledger': ledger.to_dict(),
        'events': candidates, 'excluded_events': excluded,
        'coverage': {
            'supported_instrument_weight': supported_weight,
            'unsupported_instrument_weight': unsupported_weight,
            'explicit_cash_weight': cash_weight,
            'unsupported_components': unsupported,
            'unresolved_etf_identity_weight': ledger.unresolved_weight,
            'known_nvda_direct_and_lookthrough_weight': math.fsum(
                entry.portfolio_weight for entry in ledger.entries
                if entry.symbol == 'NVDA' and entry.asset_type == 'stock'
                and entry.status != 'unresolved_etf'),
            'catalog_events_considered': len(catalog), 'events_selected': len(candidates),
            'all_tickers_supported': False, 'venue_coverage_exhaustive': False,
        },
        'ranking_rule': ('Descending linked NAV, then linked NAV with a specific issuer/supply or '
                         'Treasury-duration channel, then earlier event deadline and stable event ID. '
                         'The tie break is an explicit inspection policy, not an estimated risk score.'),
        'macro_discovery': macro,
        'originals': {k: v for k, v in originals.items() if k in selected_tickers},
        'new_provider_requests': 0, 'new_model_calls': 0,
        'limitations': [
            'This is portfolio-dependent selection from three existing archived/documentary candidates, not autonomous live news discovery.',
            'Linked NAV is an exposure-relevance proxy; event probabilities and conditional price changes are not inferred from it.',
            'The current input portfolio is analyzed using current admissible holdings; historical ownership at source receipt time is not asserted.',
            'Macro contract-family relevance does not prove equivalence to an economic event; exact source-defined YES rules control.',
            'Taiwan documentary excerpts have no admitted original provider quote or full-rule receipt in this public route.',
            'Parent ETF macro channels do not establish constituent identity; unresolved holdings remain explicit and weights are never renormalized.',
            'Unsupported instruments and absent catalog events are unmodeled, not safe or zero-risk.',
        ],
    }
