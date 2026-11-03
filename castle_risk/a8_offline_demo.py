"""Genuine archived Kalshi responses; scripted event candidates, NEVER live Claude.

Intended for a reliable walkthrough without incurring model/API charges. This
snapshot contains two independent real historical provider event families and
a rejection in each direction. No 'current' quote status is claimed.
"""
from __future__ import annotations

from .agentic_bridge import catalog_record
from .source_grounding import calendar_context, evaluate_market
from .workflow import load_real_snapshot


def archived_a8_demo():
    cpi = load_real_snapshot('cpi_oct26')['markets']
    fomc = load_real_snapshot('oct26')['markets']
    events = [
        {'event_id':'archive-cpi-oct26',
         'title':'October 2026 CPI release (BLS November 10)',
         'description':'Historical demonstration of the October CPI settlement-value market family',
         'event_date':'2026-11-10',
         'definition_origin':'SCRIPTED_OFFLINE_ARCHIVE'},
        {'event_id':'archive-fomc-oct26',
         'title':'Federal Reserve October 2026 policy rate decision',
         'description':'Historical demonstration of the October 27–28 FOMC meeting',
         'event_date':'2026-10-28',
         'definition_origin':'SCRIPTED_OFFLINE_ARCHIVE'},
    ]
    objects = {m.ticker: m for m in cpi + fomc}
    if len(objects) != len(cpi) + len(fomc):
        raise ValueError('Archived provider market identity duplicate')
    by_event = {events[0]['event_id']: [catalog_record(m) for m in cpi],
                events[1]['event_id']: [catalog_record(m) for m in fomc]}
    # A negative uses real archived Kalshi source bytes, not a fabricated Chile
    # or bank sanctions object. Independent source-family verifier rejects it.
    negatives = {}
    for event, other in zip(events, (fomc[0], cpi[0])):
        record = catalog_record(other)
        verdict = evaluate_market(event, record, calendar_context(event))
        if verdict.get('eligible'):
            raise ValueError('An archived wrong-family contract bypassed the negative gate')
        negatives[event['event_id']] = [{'ticker': other.ticker, 'reason': verdict['reason'],
                                         'source_url': record['source_url'],
                                         'source_receipt_status': record['source_receipt_status']}]
    return {
        'events': events,
        'markets_by_event': by_event,
        'quarantined_provider_contracts': negatives,
        'calendar_reviews_by_event': {e['event_id']: calendar_context(e) for e in events},
        'matches_by_event': {e['event_id']: [] for e in events},
        'match_statuses': {e['event_id']: 'NO_CLAUDE_MODEL_RUN_SCRIPTED_OFFLINE' for e in events},
        'status': 'SCRIPTED_OFFLINE_REAL_ARCHIVED_KALSHI_REPLAY_NOT_LIVE_CLAUDE',
        'execution_mode': 'DETERMINISTIC_SCENARIOS_REAL_SOURCE_ARCHIVE_NO_MODEL_OR_NETWORK',
        'request_count': 0, 'searches': [],
        'coverage_status': 'TWO_HISTORICAL_ARCHIVED_KALSHI_EVENTS_ONLY_NO_VENUE_COVERAGE',
        'all_market_availability_verified': False,
        'source_relevance_gate_version': 'A8_INDEPENDENT_ARCHIVED_DEMO',
        'source_qualified_relevance_count': len(cpi)+len(fomc),
        'quarantined_count': 2,
    }, objects
