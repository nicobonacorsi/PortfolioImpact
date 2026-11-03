"""R112: source-supported mechanisms with clearly analyst-set magnitude gaps.

The tests never treat an event title as a real outcome, quote, or empirical
return estimate. No provider is contacted. Every dollar uses original NAV.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random

import pytest

from castle_risk.analyst_scenarios import propose_analyst_scenario
from castle_risk.scenario_lab import evaluate_scenario


CASE = {
    'id': 'independent-live-case', 'notional': 100000.,
    'positions': [
        {'symbol': 'XOM', 'weight': .30, 'type': 'stock'},
        {'symbol': 'TSM', 'weight': .25, 'type': 'stock'},
        {'symbol': 'LMT', 'weight': .25, 'type': 'stock'},
        {'symbol': 'IEF', 'weight': .20, 'type': 'etf'},
    ], 'lookthrough': [],
}


PROPOSALS = [
    ('us_sovereign_downgrade', "Moody's, S&P or Fitch downgrades the United States long-term sovereign credit rating by 2027-01-30"),
    ('taiwan_shipping_blockade', 'China declares or implements a blockade, maritime quarantine or customs inspection regime on Taiwan-bound shipping by 2027-01-30'),
    ('hormuz_disruption', 'Commercial tanker transit through the Strait of Hormuz suspended or materially interrupted by 2027-01-30'),
    ('shutdown', 'US federal government shutdown (lapse in appropriations covering the Department of Defense) begins by 2027-01-30'),
    ('opec_output_increase', 'OPEC+ formally announces an increase to aggregate crude production targets by 2026-12-31'),
    ('defense_funding_approval', 'Full-year FY2027 Department of Defense appropriations act signed into law by 2027-01-30'),
    ('tariff', 'US presidential proclamation imposing Section 232 tariffs on imported semiconductors published by 2027-01-30'),
    ('chip_export_controls', 'US BIS publishes a new rule in the Federal Register tightening export controls on advanced AI chips or advanced-node foundry services to China by 2027-01-30'),
]


def target(title, **additional):
    return {'id': 'model-researched-event', 'title': title,
            'quality': {'definition_review': {'status': 'DEFINED_FOR_REVIEW'},
                        'model_admission': {'primary_impact_allowed': False}},
            **additional}


@pytest.mark.parametrize(('key', 'title'), PROPOSALS)
def test_all_eight_independent_portfolio_events_have_separate_complete_analyst_scenarios(key, title):
    event = target(title, original_question='Could a shutdown, rating review or tariff affect the portfolio?')
    original = deepcopy(event)
    proposal = propose_analyst_scenario(CASE, event)
    assert proposal is not None
    assert proposal['key'] == key
    assert proposal['status'] == 'COMPLETE'
    assert proposal['result']['complete']
    assert proposal['unmodeled_symbols'] == []
    assert proposal['result']['covered_weight'] == pytest.approx(1)
    assert proposal['source']['supports'] == 'EVENT_DEFINITION_OR_MECHANISM_NOT_SHOCK_MAGNITUDE'
    assert proposal['event_probability']['value'] is None
    assert not proposal['prediction_market_probability_used']
    assert not proposal['changes_existing_primary_outcomes']
    assert not proposal['result']['event_effect_estimated']
    assert not proposal['result']['pricing']['q_empirically_identified']
    assert set(proposal['position_return_gap_pct']) == {'XOM', 'TSM', 'LMT', 'IEF'}
    gap = math.fsum(100000 * position['weight'] *
                    proposal['position_return_gap_pct'][position['symbol']] / 100
                    for position in CASE['positions'])
    assert proposal['result']['known_event_gap_usd'] == pytest.approx(gap)
    assert proposal['result']['yes_pnl_usd'] == pytest.approx(.5 * gap)
    assert proposal['result']['no_pnl_usd'] == pytest.approx(-.5 * gap)
    replay = evaluate_scenario(CASE, {
        'hypothesis': proposal['scenario_condition'], 'mode': 'binary_gap',
        'position_shocks': proposal['position_return_gap_pct'],
        'pricing_weight': .5, 'pricing_weight_range': [.25, .75]})
    assert replay['known_event_gap_usd'] == pytest.approx(gap)
    assert event == original


def test_funding_approval_is_never_an_inferred_shutdown_loss():
    title = PROPOSALS[5][1]
    proposal = propose_analyst_scenario(CASE, target(
        title, original_question='Could a shutdown or funding approval move Lockheed?'))
    assert proposal['key'] == 'defense_funding_approval'
    assert proposal['position_return_gap_pct'] == {'XOM': 0, 'TSM': 0, 'LMT': 1.5, 'IEF': 0}
    assert proposal['result']['known_event_gap_usd'] == pytest.approx(375)
    assert proposal['result']['yes_pnl_usd'] == pytest.approx(187.5)


def test_taiwan_direct_tsm_and_treasury_hedge_are_auditable_premises():
    proposal = propose_analyst_scenario(CASE, target(PROPOSALS[1][1]))
    assert proposal['position_return_gap_pct']['TSM'] == -12.
    assert proposal['position_return_gap_pct']['IEF'] == .6
    assert proposal['channels']['TSM'] == 'DECLARED_NAMED_STOCK_STRESS'
    assert proposal['position_parameter_basis']['TSM']['kind'] == 'DECLARED_ANALYST_RETURN_GAP_NOT_CALIBRATED'
    assert 'bond yields could instead rise' in proposal['alternative_economic_regime']


@pytest.mark.parametrize('title', [
    'OPEC+ formally announces a decrease in crude production targets',
    'Taiwan manufacturing outlook review without a transport blockade',
    'Pentagon funding bill is proposed but not enacted',
    'US BIS exports advanced chips to Canada without a China restriction',
])
def test_no_premature_named_analyst_scenario_for_different_conditions(title):
    assert propose_analyst_scenario(CASE, target(title)) is None


def test_unverified_new_fund_remains_partial_not_100_percent_modeled():
    case = deepcopy(CASE)
    case['positions'][-1] = {'symbol': 'MYSTERY', 'weight': .2, 'type': 'etf'}
    proposal = propose_analyst_scenario(case, target(PROPOSALS[2][1]))
    assert proposal['key'] == 'hormuz_disruption'
    assert proposal['status'] == 'PARTIAL'
    assert proposal['unmodeled_symbols'] == ['MYSTERY']
    assert proposal['result']['yes_pnl_usd'] is None
    assert proposal['result']['no_pnl_usd'] is None


def test_unknown_causal_relation_is_not_given_generic_global_negative_shock():
    assert propose_analyst_scenario(CASE, target('Unclassified trade delegation visits Canada')) is None


def test_twenty_new_weight_vectors_preserve_all_event_dollar_identities():
    rng = random.Random(112)
    symbols = ('XOM', 'TSM', 'LMT', 'MSFT', 'JPM', 'CAT', 'UNH',
               'GLD', 'TLT', 'IEF', 'BND', 'QQQ', 'VTI')
    funds = {'GLD', 'TLT', 'IEF', 'BND', 'QQQ', 'VTI'}
    for _ in range(20):
        selected = ['LMT', *rng.sample([s for s in symbols if s != 'LMT'], rng.randint(1, 6))]
        raw = [rng.uniform(.05, 2.0) for _ in selected]
        total = math.fsum(raw)
        nav = rng.uniform(1000, 2_000_000)
        case = {'notional': nav, 'positions': [
            {'symbol': symbol, 'weight': value / total,
             'type': 'etf' if symbol in funds else 'stock'}
            for symbol, value in zip(selected, raw)]}
        for expected, title in PROPOSALS:
            proposal = propose_analyst_scenario(case, target(title))
            assert proposal is not None and proposal['key'] == expected
            assert proposal['status'] == 'COMPLETE'
            gap = math.fsum(nav * position['weight'] *
                            proposal['position_return_gap_pct'][position['symbol']] / 100
                            for position in case['positions'])
            assert proposal['result']['known_event_gap_usd'] == pytest.approx(gap, abs=1e-7)
            assert proposal['result']['yes_pnl_usd'] == pytest.approx(.5 * gap, abs=1e-7)
            assert proposal['result']['no_pnl_usd'] == pytest.approx(-.5 * gap, abs=1e-7)
            assert proposal['event_probability']['value'] is None
