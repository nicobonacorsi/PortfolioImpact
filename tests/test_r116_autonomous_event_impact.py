"""R116 injected autonomous impact and public-export tamper checks.

Synthetic model judgments here are not evidence of real financial prediction.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.agent import AgentError
from castle_risk.autonomous_event_impact import (
    BATCH_SCHEMA, forecast_portfolio_impacts, attach_autonomous_impacts,
    recheck_saved_autonomous_impacts,
)
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.strict_schema import transport_schema
from castle_risk.joint_credal_impact import portfolio_signature
from test_r116_joint_credal_impact import POSITIONS, _branches


EVENT = {"id": "HORMUZ_EVENT", "date": "2027-01-30",
         "title": "Commercial tanker transit through Hormuz interrupted by 2027-01-30"}
CAPTURE = "2026-11-01T18:10:22+00:00"
SOURCE = {"status": "SEARCHED_SOURCE_LEADS_NOT_VERIFIED",
          "sources": [{"url": "https://www.gao.gov/reports/example.pdf",
                       "title": "Documented contextual source lead",
                       "search_query": "Hormuz maritime traffic",
                       "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED"}]}


def _context(event=EVENT):
    result = probability_question_context({"capture": CAPTURE}, event)
    assert result is not None
    return result


def _proposal():
    branches = _branches()
    for branch in branches:
        branch["rationale"] = "A meaningful hypothetical price transmission with distinct risk channels."
        branch["counterevidence"] = "A diplomatic reversal or a different market regime could overturn it."
        branch["source_ids"] = ["W01"]
    return {"event_id": EVENT["id"], "status": "PROPOSED",
            "abstain_reason": "Not applicable", "branches": branches}


class FakeClient:
    model = "injected-test-model"

    def __init__(self, rows=None, *, fail=False):
        self.rows = rows if rows is not None else [_proposal()]
        self.fail = fail
        self.calls = []

    def structured(self, name, schema, payload, system=""):
        self.calls.append((name, deepcopy(payload)))
        assert name == "submit_portfolio_impacts"
        assert "P(branch | E)" in system
        assert payload["do_not_use_probability_P_or_quote_Q_or_pricing_q"] is True
        assert payload["horizon_trading_sessions"] == 20
        assert sum(row["weight"] for row in payload["original_portfolio_positions"]) == pytest.approx(1.)
        assert "secret" not in repr(payload).lower()
        if self.fail:
            raise AgentError("injected request failure")
        return {"assessments": deepcopy(self.rows)}


def _case():
    return {"user_supplied": True, "capture": CAPTURE, "notional": 100000.,
            "positions": deepcopy(POSITIONS),
            "instrument_identity": {"blocked_symbols": []},
            "risks": [{"id": EVENT["id"], "title": EVENT["title"], "date": EVENT["date"],
                       "event_probability_question": _context(),
                       "outcomes": [{"pnl": -1234., "q_reference": .4}],
                       "quality": {"definition_review": {"status": "DEFINED_FOR_REVIEW"}}}]}


def test_schema_is_valid_for_strict_anthropic_transport():
    transport_schema(BATCH_SCHEMA)


def test_proposed_impact_accepts_an_empty_abstain_reason_without_weakening_math():
    from castle_risk.agent import validate_schema
    proposal = _proposal()
    proposal['abstain_reason'] = ''
    for branch in proposal['branches']:
        # Legacy injected regressions omit the later R118 transport fields;
        # populate them here so the COMPLETE current schema is checked.
        branch['assumed_onset_date'] = '2026-11-02'
        branch['no_event_baseline_rationale'] = 'Hypothetical zero-drift baseline for this synthetic schema check.'
        branch['no_event_current_mark_paths'] = {
            label: [{'symbol': p['symbol'], 'no_event_return_pct': 0.} for p in POSITIONS]
            for label in ('adverse', 'central', 'favorable')}
    validate_schema({'assessments': [proposal]}, BATCH_SCHEMA)
    row = forecast_portfolio_impacts([_context()], POSITIONS, 100000, SOURCE, FakeClient([proposal]))[EVENT['id']]
    assert row['status'] == 'MODEL_STRESS_UNCALIBRATED'
    assert row['gap_center_usd'] == pytest.approx(-1700.)


def test_autonomous_joint_forecast_uses_original_nav_and_is_never_causal_or_probabilistic():
    client = FakeClient()
    result = forecast_portfolio_impacts([_context()], POSITIONS, 100000, SOURCE, client)
    assert len(client.calls) == 1
    row = result[EVENT["id"]]
    assert row["status"] == "MODEL_STRESS_UNCALIBRATED"
    assert row["gap_center_usd"] == pytest.approx(-1700.)
    assert row["gap_model_set_usd"] == pytest.approx([-4645, 990])
    assert row["portfolio_signature"] == portfolio_signature(POSITIONS, 100000)
    assert row["model_calibrated"] is False
    assert row["empirical_causal_estimate"] is False
    assert row["event_probability_used"] is False
    assert row["asset_pricing_q_used"] is False
    assert row["absolute_future_pnl_identified"] is False
    assert row["etf_lookthrough_added_to_nav"] is False
    assert len(row["scenario_branches"]) == 3
    assert row["today_mark_sensitivity"]["status"] == "EVENT_YES_PNL_FROM_TODAY_NOT_IDENTIFIED"
    assert row["today_mark_sensitivity"]["event_minus_no_event_gap_model_set_usd"] == pytest.approx([-4645, 990])
    assert row["today_mark_sensitivity"]["equity_priced_in_fraction_identified"] is False
    assert row["today_mark_sensitivity"]["event_yes_pnl_from_today_usd"] is None


def test_live_style_mislabelled_complete_vectors_are_sorted_without_mixing_holdings():
    # The real Claude probe on 2026-11-01 returned a whole-portfolio path
    # order mismatch. The canonicalizer relabels COMPLETE vectors, not assets.
    proposal = _proposal()
    branch = proposal["branches"][0]
    branch["joint_paths"]["adverse"], branch["joint_paths"]["central"] = (
        branch["joint_paths"]["central"], branch["joint_paths"]["adverse"])
    response = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                           SOURCE, FakeClient([proposal]))[EVENT["id"]]
    assert response["status"] == "MODEL_STRESS_UNCALIBRATED"
    assert response["gap_center_usd"] == pytest.approx(-1700.)
    assert response["gap_model_set_usd"] == pytest.approx([-4645, 990])
    assert response["joint_path_relabelings"] == [{
        "branch_id": "S1",
        "original_label_for_canonical": {
            "adverse": "central", "central": "adverse", "favorable": "favorable"},
        "rule": "SORT_WHOLE_COUPLED_VECTORS_BY_PORTFOLIO_GAP",
    }]
    original_joint_sets = sorted(
        sorted((v["symbol"], v["gap_pct"]) for v in vec)
        for vec in branch["joint_paths"].values()
    )
    admitted_joint_sets = sorted(
        sorted((v["symbol"], v["gap_pct"]) for v in vec)
        for vec in response["scenario_branches"][0]["joint_paths"].values()
    )
    assert admitted_joint_sets == original_joint_sets
    case = _case()
    attach_autonomous_impacts(case, {EVENT["id"]: response})
    assert "autonomous_event_impact" in case["risks"][0]
    recheck_saved_autonomous_impacts(case)
    assert "autonomous_event_impact" in case["risks"][0]


def test_public_adapter_preserves_A_and_only_attaches_snapshot_matched_model_C():
    forecast = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                          SOURCE, FakeClient())[EVENT["id"]]
    case = _case()
    original = deepcopy(case["risks"][0]["outcomes"])
    attach_autonomous_impacts(case, {EVENT["id"]: forecast})
    assert case["risks"][0]["autonomous_event_impact"]["gap_center_usd"] == pytest.approx(-1700.)
    assert case["risks"][0]["outcomes"] == original
    recheck_saved_autonomous_impacts(case)
    assert "autonomous_event_impact" in case["risks"][0]


@pytest.mark.parametrize("corrupt", [
    lambda r: r.update(gap_center_usd=-2000),
    lambda r: r.update(gap_model_set_usd=[-4645, float("nan")]),
    lambda r: r.update(gap_model_set_pct=[-4.645, 99.]),
    lambda r: r.update(portfolio_signature="0"*64),
    lambda r: r["today_mark_sensitivity"].update(event_yes_pnl_from_today_usd=999.),
    lambda r: r["today_mark_sensitivity"].update(physical_event_probability_used=True),
    lambda r: r.update(as_of_utc="2026-11-02T18:10:22+00:00"),
    lambda r: r.update(deadline="2027-01-29"),
    lambda r: r.update(event_probability_used=True),
    lambda r: r.update(model_calibrated=True),
    lambda r: r.update(etf_lookthrough_added_to_nav=True),
    lambda r: r["scenario_branches"][1]["weight_given_event"].update(center=.8),
    lambda r: r["scenario_branches"][0]["joint_paths"]["central"][0].update(gap_pct=9),
    lambda r: r["scenario_branches"][0]["joint_paths"]["central"].pop(),
    lambda r: r["scenario_branches"][0].update(residual=True),
    lambda r: r["source_leads"][0].update(url="https://127.0.0.1/private"),
    lambda r: r["source_leads"][0].update(url="https://www.gao.gov/?token=secret"),
    lambda r: r["source_leads"][0].update(id="W17"),
])
def test_tampered_model_c_cannot_be_replayed_or_promoted(corrupt):
    forecast = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                          SOURCE, FakeClient())[EVENT["id"]]
    altered = deepcopy(forecast)
    corrupt(altered)
    case = _case()
    attach_autonomous_impacts(case, {EVENT["id"]: altered})
    assert "autonomous_event_impact" not in case["risks"][0]
    assert case["risks"][0]["outcomes"][0]["pnl"] == -1234.


def test_saved_case_reinspection_drops_corrupted_model_C():
    forecast = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                          SOURCE, FakeClient())[EVENT["id"]]
    case = _case()
    attach_autonomous_impacts(case, {EVENT["id"]: forecast})
    case["risks"][0]["autonomous_event_impact"]["gap_center_usd"] = 50_000
    recheck_saved_autonomous_impacts(case)
    assert "autonomous_event_impact" not in case["risks"][0]


@pytest.mark.parametrize("modify_case", [
    lambda c: c.update(capture="2026-11-02T18:10:22+00:00"),
    lambda c: c.update(notional=200000),
    lambda c: c["positions"][0].update(weight=.31),
    lambda c: c["risks"][0].update(date="2027-01-29"),
    lambda c: c["risks"][0].update(title="Unrelated target renamed"),
    lambda c: c["instrument_identity"].update(blocked_symbols=["IEF"]),
])
def test_portfolio_or_target_drift_is_not_admitted(modify_case):
    forecast = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                          SOURCE, FakeClient())[EVENT["id"]]
    case = _case()
    modify_case(case)
    attach_autonomous_impacts(case, {EVENT["id"]: forecast})
    assert "autonomous_event_impact" not in case["risks"][0]


def test_no_relevant_source_or_provider_request_failure_abstains_without_zero_fill():
    client = FakeClient()
    result = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                        {"sources": []}, client)[EVENT["id"]]
    assert client.calls == []
    assert result["status"] == "ABSTAIN"
    assert result["gap_center_usd"] is None
    failed = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                        SOURCE, FakeClient(fail=True))[EVENT["id"]]
    assert failed["status"] == "ABSTAIN"


@pytest.mark.parametrize("mutate", [
    lambda p: p["branches"][0].update(source_ids=["W99"]),
    lambda p: p["branches"][0]["joint_paths"]["adverse"][0].update(gap_pct=101),
    lambda p: p["branches"][1]["weight_given_event"].update(center=.9),
    lambda p: p["branches"][0].update(trigger="no"),
])
def test_admission_rejects_unjustified_or_incoherent_generated_scenario(mutate):
    proposal = _proposal()
    mutate(proposal)
    result = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                        SOURCE, FakeClient([proposal]))[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["gap_center_usd"] is None


def test_scheduled_date_is_not_mislabeled_as_directional_event_effect():
    scheduled = {"id": "fomc-2027-01-27-scheduled",
                 "title": "FOMC policy decision on 2027-01-27 (scheduled)",
                 "date": "2027-01-27"}
    context = _context(scheduled)
    client = FakeClient([])
    result = forecast_portfolio_impacts([context], POSITIONS, 100000,
                                        SOURCE, client)[scheduled["id"]]
    assert result["status"] == "ABSTAIN"
    assert client.calls == []


def test_model_missing_and_duplicate_events_do_not_turn_into_estimates():
    missing = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                         SOURCE, FakeClient([]))[EVENT["id"]]
    duplicated = forecast_portfolio_impacts([_context()], POSITIONS, 100000,
                                            SOURCE, FakeClient([_proposal(), _proposal()]))[EVENT["id"]]
    assert missing["gap_center_usd"] is None
    assert duplicated["status"] == "ABSTAIN"


def test_one_event_per_r118_model_call_and_unknown_ids_cannot_modify_target():
    other = {"id": "HORMUZ_SECOND", "title": EVENT["title"], "date": EVENT["date"]}
    client = FakeClient([_proposal()])
    result = forecast_portfolio_impacts([_context(), _context(other)], POSITIONS,
                                        100000, SOURCE, client)
    assert len(client.calls) == 2
    assert [call[1]["original_event_targets"][0]["event_id"] for call in client.calls] == [
        EVENT["id"], other["id"]]
    assert all(len(call[1]["original_event_targets"]) == 1 for call in client.calls)
    assert result[EVENT["id"]]["status"] == "MODEL_STRESS_UNCALIBRATED"
    assert result[other["id"]]["status"] == "ABSTAIN"
