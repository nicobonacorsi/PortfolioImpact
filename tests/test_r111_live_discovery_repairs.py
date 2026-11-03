"""R111 regression checks derived from the 2026-11-01 authenticated run.

No test contacts an API or elevates a model hypothesis to a priced event.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import io
import json

import pytest

from castle_risk.agent import AnthropicClient, AgentError, EXPANDED_EVENTS_SCHEMA, RISK_SLATE_SCHEMA
from castle_risk.agent_budgeted_discovery import InterruptedDiscovery, source_first_discovery
from castle_risk.live_assessment import _retain_partial_discovery_evidence
from castle_risk.live_catalog import _family, _query_topic_compatible
from castle_risk.live_event_brief import rank_events_by_modeled_contrast
from castle_risk.portfolio_impact_adapter import _risk_watchlist
from castle_risk.live_web_research import (
    WEB_RESEARCH_STATUS, balanced_source_leads, public_web_research, research_web_leads,
)


def _source(query, index):
    return {'url': f'https://www.sec.gov/{query}/page-{index}',
            'title': f'Sourced result {query} {index}',
            'search_query': query, 'evidence_scope': WEB_RESEARCH_STATUS}


def test_every_web_query_receives_source_slots_even_if_first_four_have_37_leads():
    leads = [_source(f'q{i}', j) for i in range(8) for j in range(9 + (i == 1))]
    original = deepcopy(leads)
    # Preserve the archived, explicitly requested 32-slot fairness check.
    # Current production uses a larger adaptive budget by default.
    selected = balanced_source_leads(leads, limit=32)
    assert len(selected) == 32
    assert Counter(row['search_query'] for row in selected) == {f'q{i}': 4 for i in range(8)}
    assert len({row['url'] for row in selected}) == 32
    assert leads == original  # selection does not mutate the original buffer


def test_web_budget_retains_all_75_returned_links_and_balances_larger_batches():
    from castle_risk.live_web_research import MAX_MODEL_CONTEXT_LEADS, MAX_PUBLIC_LEADS
    leads = [_source(f'q{i}', j) for i in range(8) for j in range(9 + (i < 3))]
    selected = balanced_source_leads(leads)
    assert len(leads) == len(selected) == 75
    assert MAX_PUBLIC_LEADS == MAX_MODEL_CONTEXT_LEADS == 96
    assert {row['url'] for row in selected} == {row['url'] for row in leads}
    two_query = [_source(f'q{i}', j) for i in range(2) for j in range(80)]
    assert len(balanced_source_leads(two_query)) == 96
    assert len(balanced_source_leads(two_query, limit=96)) == 96
    assert Counter(row['search_query'] for row in balanced_source_leads(two_query)) == {'q0': 48, 'q1': 48}
    assert leads == [_source(f'q{i}', j) for i in range(8) for j in range(9 + (i < 3))]


def test_real_web_search_tool_blocks_do_not_drop_latter_queries():
    blocks = []
    for i in range(8):
        tool_id = f'srvtoolu_{i}'
        blocks.extend([
            {'type': 'server_tool_use', 'id': tool_id, 'name': 'web_search',
             'input': {'query': f'issuer query {i}'}},
            {'type': 'web_search_tool_result', 'tool_use_id': tool_id,
             'content': [{'type': 'web_search_result', 'url': f'https://www.sec.gov/{i}/{j}',
                          'title': f'result {j}'} for j in range(10)]},
        ])
    response = {'content': blocks, 'stop_reason': 'end_turn',
                'usage': {'input_tokens': 70, 'output_tokens': 90,
                          'server_tool_use': {'web_search_requests': 8}}}
    client = AnthropicClient(api_key='unit-key', max_requests=2,
        min_interval_seconds=0, opener=lambda *_a, **_k: io.BytesIO(json.dumps(response).encode()))
    report = research_web_leads(client, {'portfolio': {'positions': [
        {'symbol': 'MSFT', 'weight': 1., 'asset_type': 'stock'}]}}, '2026-11-01', 90)
    assert report['status'] == 'SEARCHED_SOURCE_LEADS_NOT_VERIFIED'
    assert report['source_count'] == 80
    assert report['returned_valid_source_count'] == 80
    assert report['omitted_valid_source_count'] == 0
    assert report['distinct_queries_searched'] == 8
    assert report['retained_distinct_queries'] == 8
    assert Counter(row['search_query'] for row in report['sources']) == {
        f'issuer query {i}': 10 for i in range(8)}


def test_fable_structured_discovery_has_room_for_tool_output_without_changing_open_research():
    recorded = []

    def opener(request, timeout):
        recorded.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'content': [], 'stop_reason': 'end_turn',
                                     'usage': {'input_tokens': 12, 'output_tokens': 1}}).encode())

    client = AnthropicClient(api_key='unit-key', opener=opener, min_interval_seconds=0,
                             max_requests=5, max_request_bytes=100000,
                             max_output_tokens=65536, max_total_tokens=500000)
    client.request([], [])
    assert recorded[-1]['output_config'] == {'effort': 'max'}
    assert recorded[-1]['max_tokens'] == 65536
    for name, schema in [('submit_risk_slate', RISK_SLATE_SCHEMA),
                         ('submit_events', EXPANDED_EVENTS_SCHEMA)]:
        client.request([], [{'name': name, 'description': 'one structured answer',
                             'input_schema': schema, 'strict': True}])
        assert recorded[-1]['output_config'] == {'effort': 'high'}
        assert recorded[-1]['max_tokens'] == 16384
        assert client.ledger[-1]['output_token_ceiling_requested'] == 16384
        assert client.ledger[-1]['effort_requested'] == 'high'


def test_event_proposal_failure_retains_web_and_scout_without_promoting_events(monkeypatch):
    from castle_risk import live_web_research
    captured = public_web_research({'status': 'SEARCHED_SOURCE_LEADS_NOT_VERIFIED',
        'sources': [_source('q0', 0)], 'searches': ['q0']})
    monkeypatch.setattr(live_web_research, 'research_web_leads', lambda *args: captured)
    model_client = AnthropicClient(api_key='unit-key', opener=lambda *_a, **_k: None,
                                   min_interval_seconds=0)

    class Agent:
        client = model_client

        def scout_risks(self, *_args):
            return [{'risk_id': 'msft-cyber', 'risk_title': 'MSFT cyber incident',
                     'exposure_rationale': 'MSFT direct stock cybersecurity channel'}]

        def propose_events(self, *_args):
            raise AgentError('Simulated provider read timeout')

    portfolio = {'portfolio': {'positions': [{'symbol': 'MSFT', 'asset_type': 'stock', 'weight': 1}]}}
    with pytest.raises(InterruptedDiscovery) as error:
        source_first_discovery(Agent(), portfolio, lambda *_: [], '2026-11-01',
                               enable_web_research=True)
    partial = error.value.partial_evidence
    assert partial['web_research']['source_count'] == 1
    assert len(partial['risk_hypotheses']) == 1
    fallback = {'status': 'SOURCE_DEFINED_FALLBACK', 'events': [{'event_id': 'independent-cpi'}]}
    merged = _retain_partial_discovery_evidence(fallback, {**partial, 'events': [{'event_id': 'fake'}]})
    assert merged['events'] == [{'event_id': 'independent-cpi'}]
    assert merged['web_research']['sources'][0]['url'] == 'https://www.sec.gov/q0/page-0'
    assert merged['partial_model_evidence_retained'] is True


@pytest.mark.parametrize('query,expected', [
    ('Federal Reserve Basel III endgame final rule', None),
    ('Federal Reserve new bank capital requirements', None),
    ('Fed December 2026 rate decision', 'KXFEDDECISION'),
    ('FOMC December 2026 meeting', 'KXFEDDECISION'),
    ('Will CPI rise more than 0.2%?', 'KXCPI'),
])
def test_rate_search_does_not_misroute_bank_regulation(query, expected):
    assert _family(query) == expected


@pytest.mark.parametrize('query,title', [
    ('Microsoft earnings Azure growth', 'Will US average hourly earnings grow?'),
    ('Microsoft cybersecurity incident 8-K', 'Will AI incident reporting become federal law?'),
    ('JPMorgan Q4 2026 earnings', 'Will wage earnings rise in November?'),
    ('Federal Reserve Basel III endgame final rule', 'Will the Fed cut rates by 25bps?'),
    ('Treasury quarterly refunding November 2026', 'Will Treasury trade Bitcoin?'),
    ('gold ETF outflows', 'Will Trump Gold Card receipts top $50 million?'),
    ('US credit rating downgrade', 'Will France sovereign credit rating be downgraded?'),
])
def test_unrelated_realistic_catalog_titles_cannot_pass_topic_filter(query, title):
    assert not _query_topic_compatible(query, title, metadata=False)


@pytest.mark.parametrize('query,title', [
    ('Microsoft cybersecurity incident 8-K',
     'Will Microsoft disclose a material cybersecurity incident by January 2027?'),
    ('Microsoft earnings Azure growth',
     'Will MSFT report Azure growth below 30% in Q4 2026?'),
    ('Federal Reserve Basel III endgame final rule',
     'Will the Federal Reserve finalize Basel III bank capital requirements?'),
    ('gold ETF outflows', 'Will SPDR Gold Trust holdings fall in 2027?'),
    ('US credit rating downgrade', 'Will the U.S. credit rating be downgraded by December 2026?'),
])
def test_true_topic_candidates_survive_for_independent_rule_review(query, title):
    assert _query_topic_compatible(query, title, metadata=False)


def _event(event_id, point, lo, hi):
    return {'event_id': event_id, 'impact': {'delta_pnl_usd': point,
        'delta_pnl_interval_usd': [lo, hi], 'missing_symbols': []}}


def test_ranking_proves_only_strict_interval_dominance_and_preserves_unknowns():
    rows = rank_events_by_modeled_contrast([
        _event('A', 100., 90., 110.),
        _event('B', 20., 10., 30.),
        _event('C', 85., 50., 120.),
        {'event_id': 'D', 'impact': {'delta_pnl_usd': None, 'delta_pnl_interval_usd': None}},
    ])
    r = {e['event_id']: e['ranking'] for e in rows}
    assert r['A']['strictly_larger_movement_than_event_ids'] == ['B']
    assert r['B']['strictly_smaller_movement_than_event_ids'] == ['A', 'C']
    assert 'C' in r['A']['overlapping_interval_event_ids']
    assert 'A' in r['C']['overlapping_interval_event_ids']
    assert r['D']['abs_delta_pnl_usd'] is None
    assert not r['D']['strictly_larger_movement_than_event_ids']


def test_magnitude_interval_crossing_zero_never_certifies_strict_priority():
    rows = rank_events_by_modeled_contrast([
        _event('zero-crossing', 5., -90., 100.),
        _event('other', 40., 35., 45.),
    ])
    rankings = {event['event_id']: event['ranking'] for event in rows}
    assert rankings['zero-crossing']['abs_delta_pnl_interval_usd'] == [0, 100.]
    assert rankings['zero-crossing']['strictly_larger_movement_than_event_ids'] == []
    assert rankings['other']['strictly_larger_movement_than_event_ids'] == []


def test_unpriced_risk_watchlist_preserves_candidates_without_a_nav_proxy_rank():
    portfolio = [{'symbol': 'MSFT', 'weight': .3}, {'symbol': 'JPM', 'weight': .25},
                 {'symbol': 'GLD', 'weight': .45}]
    report = {'risk_watchlist': [
        {'risk_id': 'bank', 'risk_title': 'JPM credit',
         'exposure_rationale': 'JPM credit losses'},
        {'risk_id': 'cloud', 'risk_title': 'MSFT cloud',
         'exposure_rationale': 'MSFT capital expenditure'},
        {'risk_id': 'gold', 'risk_title': 'GLD flows',
         'exposure_rationale': 'GLD ETF flows'},
    ]}
    rows = _risk_watchlist(report, portfolio)
    assert [row['risk_id'] for row in rows] == ['bank', 'cloud', 'gold']
    assert [row['inspection_priority'] for row in rows] == [None, None, None]
    assert [row['linked_direct_portfolio_weight'] for row in rows] == [.25, .3, .45]
    assert all(row['probability'] is row['pnl'] is row['portfolio_impact_usd'] is None for row in rows)
    assert all(row['inspection_priority_basis'] == 'UNRANKED_UNTIL_CONDITIONAL_IMPACT_IS_IDENTIFIED' for row in rows)
