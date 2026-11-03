"""R113: scenario movement orders the display; named NAV does not.

These are offline mathematical and regression checks. Conditional stresses
are not empirical event odds, expected losses or blind forecast validation.
"""
from __future__ import annotations

from copy import deepcopy
import math

import pytest

from castle_risk.analyst_scenarios import propose_analyst_scenario
from castle_risk.impact_priority import (
    conditional_movement_priority, sort_risks_by_conditional_movement,
)
from tests.test_r111_portfolio_invariants import FROZEN_PORTFOLIOS
from tests.test_r112_economic_assumption_coverage import CASE, PROPOSALS, target


def _a(name, values, allowed=True, *, weight=1.0):
    return {'id': name, 'title': name, 'exposure': [{'symbol': 'MSFT', 'weight': weight}],
            'outcomes': [{'pnl': value} for value in values],
            'quality': {'model_admission': {'primary_impact_allowed': allowed}}}


def _b(case, key=1):
    event = target(PROPOSALS[key][1]); event['id'] = PROPOSALS[key][0]
    event['analyst_assumption'] = propose_analyst_scenario(case, event)
    assert event['analyst_assumption'] is not None
    return event


def test_small_linked_nav_can_be_first_when_conditional_dollars_are_larger():
    positions = [{'symbol': 'MSFT', 'weight': .8}, {'symbol': 'JPM', 'weight': .2}]
    large_holding_small_move = _a('MSFT-low-shock', [-100, 100], weight=.8)
    small_holding_large_move = _a('JPM-large-shock', [-900, 600], weight=.2)
    large_holding_small_move['exposure'][0]['symbol'] = 'MSFT'
    small_holding_large_move['exposure'][0]['symbol'] = 'JPM'
    rows = sort_risks_by_conditional_movement([
        large_holding_small_move, small_holding_large_move], positions, 100000)
    assert [row['id'] for row in rows] == ['JPM-large-shock', 'MSFT-low-shock']
    assert [row['impact_priority']['max_abs_conditional_pnl_usd'] for row in rows] == [900, 100]
    assert [row['impact_priority']['modeled_display_order'] for row in rows] == [1, 2]


def test_rejected_or_partial_outcomes_are_unranked_even_if_they_contain_big_numbers():
    positions = [{'symbol': 'MSFT', 'weight': 1}]
    unknown = _a('unadmitted', [-1000000], False)
    partial = _a('partial', [-200, None])
    valid = _a('admitted', [-70, 50])
    rows = sort_risks_by_conditional_movement([unknown, partial, valid], positions, 100000)
    assert [row['id'] for row in rows] == ['admitted', 'unadmitted', 'partial']
    assert rows[1]['impact_priority']['max_abs_conditional_pnl_usd'] is None
    assert rows[2]['impact_priority']['modeled_display_order'] is None
    assert all(row['impact_priority']['not_expected_loss'] for row in rows)
    assert all(row['impact_priority']['not_event_probability_weighted'] for row in rows)


def test_source_defined_market_and_what_if_are_not_portfolio_target_events():
    positions = [{'symbol': 'MSFT', 'weight': 1}]
    market = {**_a('related-contract', [-10000]), 'source_defined_scenario': True}
    what_if = {**_a('user-what-if', [-10000]), 'what_if_explorer': True}
    rows = sort_risks_by_conditional_movement([market, _a('target', [150]), what_if], positions, 100000)
    assert [row['id'] for row in rows] == ['target', 'related-contract', 'user-what-if']
    assert 'impact_priority' not in market and 'impact_priority' not in what_if
    assert rows[1].get('impact_priority') is None


def test_tied_magnitudes_keep_source_first_then_stable_original_order():
    positions = [{'symbol': 'MSFT', 'weight': 1}]
    rows = sort_risks_by_conditional_movement([
        _a('first-A', [-100]), _a('first-B', [-100]), _a('second-A', [100])], positions, 100000)
    assert [row['id'] for row in rows] == ['first-A', 'first-B', 'second-A']


def test_realistic_taiwan_declared_stress_is_separately_labelled_and_recomputed():
    taiwan = _b(CASE)
    priority = conditional_movement_priority(taiwan, CASE['positions'], CASE['notional'])
    assert priority['evidence_tier'] == 'B_DECLARED_ANALYST_SCENARIO'
    assert priority['max_abs_conditional_pnl_usd'] == pytest.approx(1790)
    assert 'ILLUSTRATIVE' in priority['basis']
    assert taiwan['analyst_assumption']['event_probability']['value'] is None
    assert taiwan['analyst_assumption']['result']['pricing']['q_empirically_identified'] is False


@pytest.mark.parametrize('change', [
    lambda p: p.update(status='PARTIAL'),
    lambda p: p['result'].update(complete=False),
    lambda p: p['result'].update(covered_weight=.75),
    lambda p: p['result'].update(known_event_gap_usd=999999),
    lambda p: p['result'].update(yes_pnl_usd=999999),
    lambda p: p['result']['pricing'].update(equity_weight_q=1.1),
    lambda p: p['position_return_gap_pct'].update(TSM=float('nan')),
    lambda p: p['position_return_gap_pct'].pop('IEF'),
    lambda p: p.update(unmodeled_symbols=['IEF']),
])
def test_tampered_or_incomplete_analyst_math_never_receives_numeric_priority(change):
    event = _b(CASE)
    change(event['analyst_assumption'])
    priority = conditional_movement_priority(event, CASE['positions'], CASE['notional'])
    assert priority['evidence_tier'] == 'C_UNQUANTIFIED'
    assert priority['max_abs_conditional_pnl_usd'] is None


def test_portfolio_change_invalidates_old_analyst_dollar_priority():
    taiwan = _b(CASE)
    new_case = deepcopy(CASE)
    new_case['positions'][0]['weight'] = .4
    new_case['positions'][1]['weight'] = .15
    assert conditional_movement_priority(taiwan, new_case['positions'], new_case['notional'])['evidence_tier'] == 'C_UNQUANTIFIED'
    assert conditional_movement_priority(taiwan, CASE['positions'], CASE['notional'] * 2)['evidence_tier'] == 'C_UNQUANTIFIED'


def test_all_8_independent_live_event_profiles_get_source_separated_navigation_dollars():
    rows = [_b(CASE, key) for key in range(len(PROPOSALS))]
    sorted_rows = sort_risks_by_conditional_movement(rows, CASE['positions'], CASE['notional'])
    assert len(sorted_rows) == 8
    magnitudes = [row['impact_priority']['max_abs_conditional_pnl_usd'] for row in sorted_rows]
    assert all(math.isfinite(value) for value in magnitudes)
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert sorted_rows[0]['id'] == 'taiwan_shipping_blockade'
    assert all(row['impact_priority']['evidence_tier'] == 'B_DECLARED_ANALYST_SCENARIO'
               for row in sorted_rows)


@pytest.mark.parametrize('symbols', FROZEN_PORTFOLIOS, ids=lambda s: '-'.join(s))
def test_twenty_portfolios_never_rank_unpriced_hypotheses_by_named_weight(symbols):
    weights = [.1, .5, .2, .2]
    positions = [{'symbol': symbol, 'weight': weight} for symbol, weight in zip(symbols, weights)]
    rows = [{**_a('unknown-' + symbol, [None]), 'exposure': [{'symbol': symbol, 'weight': weight}]}
            for symbol, weight in zip(symbols, weights)]
    sorted_rows = sort_risks_by_conditional_movement(rows, positions, 100000)
    assert [row['id'] for row in sorted_rows] == [row['id'] for row in rows]
    assert all(row['impact_priority']['max_abs_conditional_pnl_usd'] is None for row in sorted_rows)
