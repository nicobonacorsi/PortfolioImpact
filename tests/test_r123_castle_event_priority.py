"""Castle point 1: rank on verified point-3 paired event gaps, not stale fields.

All fixture scenarios are synthetic, not claims about actual price effects.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.autonomous_common_endpoint_impact import (
    attach_common_endpoint_impacts, forecast_common_endpoint_impacts,
)
from castle_risk.impact_priority import (
    conditional_movement_priority, sort_risks_by_conditional_movement,
)
from test_r122_autonomous_common_endpoint import (
    EVENT, MARK, POSITIONS, SOURCE, FakeClient, _case, _context,
)


def _verified_case():
    response = forecast_common_endpoint_impacts(
        [_context()], POSITIONS, 100_000., SOURCE, FakeClient(),
        observed_mark_context=MARK)
    case, risk = _case(response[EVENT['id']])
    attach_common_endpoint_impacts(case, response)
    assert risk['common_endpoint_impact']['status'] == 'MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED'
    return case, risk


def _priority(risk, case):
    return conditional_movement_priority(
        risk, case['positions'], case['notional'],
        observed_mark_context=case.get('observed_mark_context'))


def test_autonomous_paired_event_gap_is_a_distinct_reverified_navigation_tier():
    case, risk = _verified_case()
    priority = _priority(risk, case)
    assert priority['evidence_tier'] == 'B_AUTONOMOUS_COMMON_ENDPOINT_SCENARIO'
    assert priority['max_abs_conditional_pnl_usd'] == pytest.approx(
        abs(risk['common_endpoint_impact']['impact']['paired_event_gap_center_usd']))
    assert priority['max_abs_conditional_pnl_usd'] != pytest.approx(
        abs(risk['common_endpoint_impact']['impact']['event_yes_from_today_center_usd']))
    assert priority['not_event_probability_weighted'] is True
    assert priority['not_expected_loss'] is True
    assert priority['not_empirically_causal_or_calibrated'] is True


def test_verified_autonomous_impact_can_move_a_portfolio_event_ahead_of_unquantified():
    case, risk = _verified_case()
    unpriced = {'id': 'larger-named-NAV-but-no-admitted-model',
                'title': 'Hypothetical unknown event', 'outcomes': [],
                'quality': {'model_admission': {'primary_impact_allowed': False}}}
    sorted_rows = sort_risks_by_conditional_movement(
        [unpriced, risk], case['positions'], case['notional'],
        observed_mark_context=case['observed_mark_context'])
    assert [r['id'] for r in sorted_rows] == [EVENT['id'], unpriced['id']]
    assert sorted_rows[0]['impact_priority']['modeled_display_order'] == 1
    assert sorted_rows[1]['impact_priority']['max_abs_conditional_pnl_usd'] is None


@pytest.mark.parametrize('tamper', [
    lambda c, r: r['common_endpoint_impact']['impact'].update(paired_event_gap_center_usd=999999),
    lambda c, r: r['common_endpoint_impact'].update(paired_event_gap_center_usd=999999),
    lambda c, r: r['common_endpoint_impact']['scenario_branches'][0]['joint_paths']['central'][0].update(gap_pct=90),
    lambda c, r: r['common_endpoint_impact'].update(event_yes_from_today_center_usd=999999),
    lambda c, r: r.update(title='Different event predicate with same ID'),
    lambda c, r: r.update(date='2027-02-28'),
    lambda c, r: c.update(notional=200_000.),
    lambda c, r: c['positions'][0].update(weight=.29),
    lambda c, r: c['observed_mark_context'].update(last_completed_session='2026-10-29'),
    lambda c, r: c.pop('observed_mark_context'),
    lambda c, r: r['common_endpoint_impact']['impact'].update(model_class='EMPIRICALLY_IDENTIFIED_CAUSAL'),
])
def test_tampered_or_stale_autonomous_paired_scenarios_cannot_drive_event_order(tamper):
    case, risk = _verified_case()
    tamper(case, risk)
    priority = _priority(risk, case)
    assert priority['evidence_tier'] == 'C_UNQUANTIFIED'
    assert priority['max_abs_conditional_pnl_usd'] is None


def test_saved_v2_scenario_is_never_trusted_from_a_priority_field_alone():
    case, risk = _verified_case()
    sorted_once = sort_risks_by_conditional_movement(
        [risk], case['positions'], case['notional'],
        observed_mark_context=case['observed_mark_context'])
    stale = deepcopy(sorted_once[0]); stale.pop('common_endpoint_impact')
    sorted_again = sort_risks_by_conditional_movement(
        [stale], case['positions'], case['notional'],
        observed_mark_context=case['observed_mark_context'])
    assert sorted_again[0]['impact_priority']['evidence_tier'] == 'C_UNQUANTIFIED'

