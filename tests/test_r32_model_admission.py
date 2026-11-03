"""R32 admission tests use synthetic models; they do not validate forecast skill."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from castle_risk.model_admission import (
    event_model_admission, apply_model_admission_to_event, eligible_contract_scenarios,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("R32 admission tests cannot contact providers")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def model_event(status="COMPLETE_DECLARED_MODEL", *, q_spread=100., check="NOT_TESTED"):
    return {
        "event_id": "synthetic-event", "family": "US_INFLATION", "event_date": "2026-11-10",
        "outcome_table": {"status": status, "missing_symbols": [], "missing_weight": 0.,
            "model_reliability": {"status": check, "portfolio_transfer_check": {"status": check}},
            "scenario_transfer_status": check,
            "outcomes": [{"outcome_id": "lower", "q_reference": .5, "pnl_if_outcome_usd": -100.,
                          "pnl_model_range_5_95_usd": [-200., 0.]},
                         {"outcome_id": "upper", "q_reference": .5, "pnl_if_outcome_usd": 100.,
                          "pnl_model_range_5_95_usd": [0., 200.]}],
            "summary": {"q_standard_deviation_usd": q_spread}},
        "distribution": {"status": "CONDITIONAL_SOURCE_DISTRIBUTION", "source_contract_ids": ["SYNTH"],
                         "atoms": [{"q_reference": .5}, {"q_reference": .5}]},
        "ranking": {"abs_delta_pnl_usd": 200., "selected_outcome_id": "lower"},
        "risk_ranking": {"q_standard_deviation_usd": q_spread},
    }


@pytest.mark.parametrize("check", ["PASS", "NOT_TESTED", "NOT_REPORTED"])
def test_complete_macro_is_conditional_never_forecast_certified(check):
    event = model_event(check=check)
    result = event_model_admission(event)
    assert result["state"] == "CONDITIONAL_SCENARIO"
    assert result["primary_impact_allowed"] and result["economic_ranking_allowed"]
    assert result["forward_prediction_certified"] is False
    assert result["market_probability_independent"] is True
    assert result["evidence_scope"] == "MODEL_CONDITIONAL_RESPONSE_NOT_FUTURE_TOTAL_RETURN"


def test_failed_payroll_is_diagnostic_without_mutating_values_or_quotes():
    event = model_event("COMPLETE_MEASURED_SURPRISE_RESPONSE", check="FAIL")
    original = deepcopy(event)
    result = event_model_admission(event)
    assert event == original
    assert result["state"] == "DIAGNOSTIC_ONLY"
    assert not result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    assert result["diagnostic_impact_available"]
    apply_model_admission_to_event(event)
    assert event["outcome_table"] == original["outcome_table"]
    assert event["distribution"] == original["distribution"]
    assert event["risk_ranking"]["q_standard_deviation_usd"] is None
    assert event["ranking"]["abs_delta_pnl_usd"] is None
    saved = event["economic_ranking_diagnostics"]
    assert saved["risk_ranking"]["q_standard_deviation_usd"] == 100.
    assert saved["component_ranking"]["abs_delta_pnl_usd"] == 200.
    apply_model_admission_to_event(event)
    assert event["economic_ranking_diagnostics"] == saved


def test_portfolio_fail_wins_over_inconsistent_overall_pass_label():
    event = model_event("COMPLETE_MEASURED_SURPRISE_RESPONSE", check="PASS")
    event["outcome_table"]["model_reliability"]["portfolio_transfer_check"]["status"] = "FAIL"
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


def test_overall_measured_response_fail_is_not_hidden_by_missing_portfolio_check():
    event = model_event("COMPLETE_MEASURED_SURPRISE_RESPONSE", check="FAIL")
    event["outcome_table"]["model_reliability"].pop("portfolio_transfer_check")
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


@pytest.mark.parametrize("where", ["raw_reliability", "display_reliability", "display_transfer", "display_portfolio"])
def test_conflicting_pass_does_not_erase_explicit_measured_fail(where):
    event = model_event("COMPLETE_MEASURED_SURPRISE_RESPONSE", check="PASS")
    if where == "raw_reliability":
        event["outcome_table"]["model_reliability"]["status"] = "FAIL"
    elif where == "display_reliability":
        event["quality"] = {"model_reliability": {"status": "FAIL"}}
    elif where == "display_transfer":
        event["quality"] = {"transfer_status": "FAIL"}
    else:
        event["quality"] = {"portfolio_transfer_status": "FAIL"}
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


def test_macro_component_fail_is_not_relabelled_portfolio_fail():
    event = model_event(check="NOT_TESTED")
    event["outcome_table"]["model_reliability"]["positions"] = [
        {"symbol": "SYNTH", "daily_association_check": {"status": "FAIL"}}]
    assert event_model_admission(event)["state"] == "CONDITIONAL_SCENARIO"


def test_unselected_legacy_comparison_fail_is_not_the_primary_models_test():
    event = model_event(check="NOT_TESTED")
    event["economics"] = {"status": "DECLARED_DEFAULT_COMPONENT_MODEL",
        "portfolio_transfer_status": "FAIL", "scenario_transfer_status": "FAIL",
        "model_reliability": {"status": "FAIL", "portfolio_transfer_check": {"status": "FAIL"}},
        "outcome_cases": [{"component_pnl_usd": 999.}]}
    before = deepcopy(event)
    assert event_model_admission(event)["state"] == "CONDITIONAL_SCENARIO"
    assert event == before


@pytest.mark.parametrize("where", ["event", "table", "quality"])
def test_identity_block_wins_over_complete_and_passing_model(where):
    event = model_event(check="PASS")
    target = event if where == "event" else event["outcome_table"] if where == "table" else event.setdefault("quality", {})
    target["instrument_identity_gate"] = {"blocked_symbols": ["SYNTHETF"]}
    result = event_model_admission(event)
    assert result["state"] == "UNAVAILABLE"
    assert result["evidence_scope"] == "INSTRUMENT_SCOPE_GATE"
    assert not result["economic_ranking_allowed"]


@pytest.mark.parametrize("patch", [
    {"legacy_fallback_allowed": False},
    {"status": "BLOCKED_INSTRUMENT_IDENTITY"},
    {"instrument_identity": {"blocked_symbols": ["SYNTHETF"]}},
])
def test_inherited_identity_blocks_do_not_escape_through_legacy_models(patch):
    event = model_event(check="PASS")
    event.update(patch)
    event["economics"] = {"outcome_cases": [{"component_pnl_usd": 99999.}], "scenario_transfer_status": "PASS"}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("status", ["PARTIAL_KNOWN_POSITIONS_ONLY", "UNSUPPORTED_ASSET_CLASS", "BLOCKED_MODEL"])
def test_partial_or_unsupported_table_cannot_fallback_to_complete_legacy(status):
    event = model_event(status)
    event["economics"] = {"status": "DECLARED_DEFAULT_COMPONENT_MODEL",
                          "outcome_cases": [{"component_pnl_usd": 123.}]}
    result = event_model_admission(event)
    assert result["state"] == "UNAVAILABLE" and result["diagnostic_impact_available"]


@pytest.mark.parametrize("patch", [{"missing_symbols": ["SYNTH"]}, {"missing_weight": .2}])
def test_complete_label_cannot_override_missing_portfolio_data(patch):
    event = model_event()
    event["outcome_table"].update(patch)
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_cash_identity_is_not_called_empirical_pass():
    event = model_event("IDENTIFIED_NOMINAL_USD_CASH_ZERO")
    for row in event["outcome_table"]["outcomes"]:
        row["pnl_if_outcome_usd"], row["pnl_model_range_5_95_usd"] = 0., [0., 0.]
    result = event_model_admission(event)
    assert result["state"] == "EXACT_IDENTITY"
    assert result["primary_impact_allowed"] and result["economic_ranking_allowed"]
    assert not result["forward_prediction_certified"]


@pytest.mark.parametrize("value", [1., None, True, float("nan")])
def test_cash_identity_rejects_nonzero_or_missing_numeric_values(value):
    event = model_event("IDENTIFIED_NOMINAL_USD_CASH_ZERO")
    event["outcome_table"]["outcomes"][0]["pnl_if_outcome_usd"] = value
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("bounds", [[-100., 100.], [0., 1.], [0., float("nan")], [True, 0.], [0.]])
def test_cash_zero_point_cannot_hide_contradictory_range(bounds):
    event = model_event("IDENTIFIED_NOMINAL_USD_CASH_ZERO")
    for row in event["outcome_table"]["outcomes"]:
        row["pnl_if_outcome_usd"], row["pnl_model_range_5_95_usd"] = 0., [0., 0.]
    event["outcome_table"]["outcomes"][0]["pnl_model_range_5_95_usd"] = bounds
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_cash_zero_in_one_field_does_not_override_nonzero_in_another():
    event = {"quality": {"model_kind": "NOMINAL_CASH_IDENTITY"},
             "outcomes": [{"pnl_if_outcome_usd": 0., "pnl": 10., "pnl_interval": [0., 0.]}]}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("where", ["event", "table", "target", "review"])
def test_related_tariff_scenario_never_ranks_original_target(where):
    event = model_event("COMPLETE_MEASURED_ANCHOR_MODEL", q_spread=1000000.)
    event["family"] = "TRADE_TARIFFS"
    if where == "event":
        target = event
    elif where == "table":
        target = event["outcome_table"]
    elif where == "target":
        target = event.setdefault("target_scope_review", {})
    else:
        target = event.setdefault("tariff_review", {}).setdefault("target_scope_review", {})
    target["source_scenario_is_exact_target_match"] = False
    result = apply_model_admission_to_event(event)
    assert result["state"] == "RELATED_SOURCE_ONLY"
    assert not result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    assert event["risk_ranking"]["q_standard_deviation_usd"] is None
    assert event["economic_ranking_diagnostics"]["risk_ranking"]["q_standard_deviation_usd"] == 1000000.


def test_separately_defined_tariff_child_is_usable_not_original_target_ranking():
    event = model_event("COMPLETE_MEASURED_ANCHOR_MODEL")
    event.update(source_defined_scenario=True, related_to_event_id="original-tariff-increase")
    event["tariff_review"] = {"target_scope_review": {"source_scenario_is_exact_target_match": False}}
    result = event_model_admission(event)
    assert result["state"] == "CONDITIONAL_SCENARIO"
    assert result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    event["instrument_identity_gate"] = {"blocked_symbols": ["SYNTHETF"]}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("kind,expected", [("MEASURED_ANCHOR", "RELATED_SOURCE_ONLY"),
                                          ("CHANNEL_AWARE", "CONDITIONAL_SCENARIO"),
                                          ("UNKNOWN", "UNAVAILABLE")])
def test_saved_display_case_admission(kind, expected):
    event = {"outcomes": [{"pnl": -12., "q_reference": .5}],
             "quality": {"model_kind": kind, "transfer_status": "NOT_TESTED"}}
    assert event_model_admission(event)["state"] == expected


@pytest.mark.parametrize("index", range(2))
def test_actual_archived_tariff_anchor_cards_cannot_price_the_action_window(index):
    path = Path(__file__).resolve().parents[1] / "web" / "portfolio_impact" / "data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    events = [event for case in data["cases"] for event in case["risks"]
              if event.get("quality", {}).get("model_kind") == "MEASURED_ANCHOR"]
    assert len(events) == 2
    event = events[index]
    before = deepcopy(event)
    assert event["date"] == "2027-01-23"
    assert all("31 Dec 2026" in row["label"] for row in event["outcomes"])
    assert all(row["question"] == "Recorded settlement distribution" for row in event["admitted"])
    result = event_model_admission(event)
    assert result["state"] == "RELATED_SOURCE_ONLY"
    assert result["evidence_scope"] == "UNATTESTED_SEPARATE_TARIFF_SOURCE_SCOPE"
    assert not result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    assert result["diagnostic_impact_available"]
    assert event == before


@pytest.mark.parametrize("kind", ["MEASURED_ANCHOR", "TARIFF_ANCHOR"])
@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_anchor_source_scope_requires_explicit_separate_scenario_marker(kind, flag):
    event = {"source_defined_scenario": flag, "outcomes": [{"pnl": -12., "q_reference": .5}],
        "quality": {"model_kind": kind, "model_status": "COMPLETE_MEASURED_ANCHOR_MODEL",
                    "portfolio_transfer_status": "FAIL", "model_admission": {
                        "state": "CONDITIONAL_SCENARIO", "primary_impact_allowed": True}}}
    result = event_model_admission(event)
    assert result["state"] == "RELATED_SOURCE_ONLY"
    assert not result["primary_impact_allowed"] and not result["economic_ranking_allowed"]


def test_explicit_separate_saved_anchor_still_admits_only_its_source_scenario():
    event = {"source_defined_scenario": True, "outcomes": [{"pnl": -12., "q_reference": .5}],
        "quality": {"model_kind": "MEASURED_ANCHOR", "model_status": "COMPLETE_MEASURED_ANCHOR_MODEL"}}
    result = event_model_admission(event)
    assert result["state"] == "CONDITIONAL_SCENARIO" and result["primary_impact_allowed"]
    assert not result["economic_ranking_allowed"]


def test_raw_anchor_model_without_separate_scope_cannot_rank_a_generic_target():
    event = model_event("COMPLETE_MEASURED_ANCHOR_MODEL")
    result = apply_model_admission_to_event(event)
    assert result["state"] == "RELATED_SOURCE_ONLY"
    assert event["risk_ranking"]["q_standard_deviation_usd"] is None
    assert event["outcome_table"]["summary"]["q_standard_deviation_usd"] == 100.


@pytest.mark.parametrize("status", ["NO_REVIEWED_DEFAULT_ECONOMIC_CHANNEL", "SOURCE_OUTCOME_DISTRIBUTION_UNAVAILABLE",
                                    "COMPLETE_MEASURED_ANCHOR_MODEL", "UNRECOGNIZED_STATUS"])
def test_saved_kind_cannot_override_contradictory_model_status(status):
    event = {"outcomes": [{"pnl": 100.}],
             "quality": {"model_kind": "CHANNEL_AWARE", "model_status": status}}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_saved_complete_label_with_unknown_kind_is_not_enough_for_admission():
    event = {"outcomes": [{"pnl": 100.}],
             "quality": {"model_kind": "UNAVAILABLE", "model_status": "COMPLETE_DECLARED_MODEL"}}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("status", ["UNKNOWN", "NO_REVIEWED_DEFAULT_ECONOMIC_CHANNEL", "COMPLETE_DECLARED_MODEL"])
@pytest.mark.parametrize("fail_field", ["transfer_status", "portfolio_transfer_status"])
def test_incompatible_public_model_scope_wins_over_reported_fail(status, fail_field):
    event = {"outcomes": [{"pnl": 100.}], "quality": {
        "model_kind": "MEASURED_PAYROLL", "model_status": status, fail_field: "FAIL"}}
    before = deepcopy(event)
    result = event_model_admission(event)
    assert result["state"] == "UNAVAILABLE"
    assert result["evidence_scope"] == "INCOMPATIBLE_PUBLIC_MODEL_METADATA"
    assert result["diagnostic_impact_available"] and not result["primary_impact_allowed"]
    assert event == before


@pytest.mark.parametrize("status", ["", "COMPLETE_MEASURED_SURPRISE_RESPONSE"])
def test_compatible_public_payroll_fail_still_has_diagnostic_state(status):
    event = {"outcomes": [{"pnl": 100.}], "quality": {
        "model_kind": "MEASURED_PAYROLL", "model_status": status, "transfer_status": "FAIL"}}
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


def test_saved_payroll_quality_portfolio_fail_is_diagnostic():
    event = {"outcomes": [{"pnl": -12., "q_reference": .5}], "quality": {
        "model_kind": "MEASURED_PAYROLL", "transfer_status": "PASS", "portfolio_transfer_status": "FAIL"}}
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


def archived_public_contract_events():
    path = Path(__file__).resolve().parents[1] / "web" / "portfolio_impact" / "data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [event for case in data["cases"] for event in case["risks"]
            if event.get("quality", {}).get("model_kind") == "DECLARED_CONTRACT_SCENARIO"]


@pytest.mark.parametrize("index", range(3))
def test_actual_archived_contract_events_retain_narrow_primary_source_scope(index):
    events = archived_public_contract_events()
    assert len(events) == 3
    event = events[index]
    before = deepcopy(event)
    result = event_model_admission(event)
    assert result["state"] == "CONDITIONAL_SCENARIO"
    assert result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    assert result["evidence_scope"] == "SOURCE_IDENTIFIED_CONTRACT_STRESS_SCENARIO"
    assert result["forward_prediction_certified"] is False
    assert "broader" in result["reason"]
    assert event == before
    assert all(row["question"] != event["original_question"] for row in event["outcomes"])


@pytest.mark.parametrize("patch", [
    {"contract_id": ""}, {"contract_id": "  "}, {"contract_id": "WRONG-CONTRACT"},
    {"contract_id": 123}, {"source_contract_ids": []}, {"source_contract_ids": [""]},
    {"source_contract_ids": [None]}, {"source_contract_ids": "2382819"},
    {"source_contract_ids": ["WRONG-CONTRACT"]}, {"question": ""}, {"question": "   "},
    {"question": None}, {"partial": True}, {"partial": None}, {"no_channel": True},
    {"no_channel": None}, {"pnl": None}, {"pnl": True}, {"pnl": float("inf")},
    {"q_reference": float("nan")}, {"q_reference": True}, {"q_reference": -0.1}, {"q_reference": 1.1},
])
def test_malformed_public_contract_row_fails_before_stale_admission_or_fail(patch):
    event = deepcopy(archived_public_contract_events()[0])
    event["outcomes"][0].update(patch)
    event["quality"].update(transfer_status="FAIL", portfolio_transfer_status="FAIL",
        model_admission={"state": "CONDITIONAL_SCENARIO", "primary_impact_allowed": True})
    before = json.dumps(event, sort_keys=True)
    result = event_model_admission(event)
    assert result["state"] == "UNAVAILABLE"
    assert result["evidence_scope"] == "INCOMPLETE_PUBLIC_SOURCE_CONTRACT_SCOPE"
    assert not result["primary_impact_allowed"] and not result["economic_ranking_allowed"]
    assert json.dumps(event, sort_keys=True) == before


@pytest.mark.parametrize("status", ["", "COMPLETE_DECLARED_MODEL", "UNKNOWN"])
def test_public_contract_kind_alone_does_not_admit_the_source_scope(status):
    event = deepcopy(archived_public_contract_events()[0])
    event["quality"]["model_status"] = status
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_public_contract_with_no_outcomes_is_unavailable():
    event = deepcopy(archived_public_contract_events()[0])
    event["outcomes"] = []
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("rows", [1, True, "not-an-array", {"pnl": 1.}, [None]])
def test_malformed_public_contract_outcomes_fail_closed_without_crash(rows):
    event = {"outcomes": rows, "quality": {"model_kind": "DECLARED_CONTRACT_SCENARIO",
                                           "model_status": "DECLARED_CONTRACT_SCENARIO"}}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_malformed_cash_outcomes_fail_closed_without_crash():
    event = {"outcomes": 1, "quality": {"model_kind": "NOMINAL_CASH_IDENTITY",
                                       "model_status": "IDENTIFIED_NOMINAL_USD_CASH_ZERO"}}
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_malformed_source_contract_collection_is_not_rankable():
    event = {"contract_first": {"contract_scenarios": 1}}
    assert eligible_contract_scenarios(event) == []
    assert event_model_admission(event)["state"] == "UNAVAILABLE"


def test_complete_public_source_scope_does_not_override_portfolio_failure():
    event = deepcopy(archived_public_contract_events()[0])
    event["quality"]["portfolio_transfer_status"] = "FAIL"
    assert event_model_admission(event)["state"] == "DIAGNOSTIC_ONLY"


def test_no_model_does_not_destroy_a_matching_market_distribution():
    event = {"distribution": {"source_contract_ids": ["SYNTH"], "atoms": [{"q_reference": .5}]}}
    before = deepcopy(event)
    result = event_model_admission(event)
    assert event == before
    assert result["state"] == "UNAVAILABLE"
    assert result["market_probability_independent"]


def contract_scenario(**changes):
    return {"status": "DECLARED_CONTRACT_SCENARIO", "contract": {"contract_id": "SYNTH-VALID"},
            "missing_symbols": [], "summary": {"q_standard_deviation_usd": 10.}, **changes}


@pytest.mark.parametrize("changes", [
    {"contract": {}}, {"contract": {"contract_id": "  "}}, {"contract": {"contract_id": None}},
    {"contract": {"contract_id": 123}}, {"missing_symbols": ["SYNTHETF"]}, {"missing_weight": .1},
    {"summary": {"q_standard_deviation_usd": float("nan")}},
    {"summary": {"q_standard_deviation_usd": float("inf")}},
    {"summary": {"q_standard_deviation_usd": -10.}},
    {"summary": {"q_standard_deviation_usd": True}},
    {"summary": {"q_standard_deviation_usd": None}},
    {"status": "PARTIAL_KNOWN_POSITIONS_ONLY"},
    {"instrument_identity_gate": {"blocked_symbols": ["SYNTHETF"]}},
    {"legacy_fallback_allowed": False},
])
def test_one_valid_contract_cannot_admit_a_different_ineligible_contract(changes):
    bad, good = contract_scenario(**changes), contract_scenario()
    event = {"contract_first": {"contract_scenarios": [bad, good]}}
    before = deepcopy(event)
    eligible = eligible_contract_scenarios(event)
    assert len(eligible) == 1 and eligible[0] is good
    # Canonical text also permits a stable immutability comparison for the
    # deliberately malformed NaN fixture (NaN is not equal to itself).
    assert json.dumps(event, sort_keys=True) == json.dumps(before, sort_keys=True)
    admission = event_model_admission(event)
    assert admission["economic_ranking_allowed"] and not admission["primary_impact_allowed"]
    assert admission["evidence_scope"] == "SOURCE_IDENTIFIED_CONTRACT_STRESS_SCENARIO"
    event["contract_first"]["contract_scenarios"] = [bad]
    assert not eligible_contract_scenarios(event)
    assert not event_model_admission(event)["economic_ranking_allowed"]


def test_valid_zero_contract_spread_is_not_rejected_as_missing():
    event = {"contract_first": {"contract_scenarios": [contract_scenario(
        summary={"q_standard_deviation_usd": 0.})]}}
    assert len(eligible_contract_scenarios(event)) == 1


def test_engine_uses_same_contract_eligibility_set_as_admission(monkeypatch):
    import castle_risk.default_assessment as module
    bad = contract_scenario(contract={"contract_id": "SYNTH-PARTIAL"}, missing_symbols=["SYNTHETF"],
                            summary={"q_standard_deviation_usd": 1e9})
    good = contract_scenario()
    question = "Taiwan supply disruption?"
    report = {"portfolio": {"notional": 100000., "positions": [
        {"symbol": "SYNTH", "asset_type": "stock", "weight": 1.}]},
        "empirical": {"daily_betas": {}},
        "events": [{"event_id": "taiwan", "title": "Taiwan supply disruption", "family": "TAIWAN_SUPPLY",
                    "event_date": "2026-12-31", "question_we_care_about": question}]}
    search = {"evaluated_at": "2026-10-27T00:00:00+00:00", "targets": [{"event_id": "taiwan",
        "family": "TAIWAN_SUPPLY", "event_date": "2026-12-31", "question_we_care_about": question,
        "candidates": [{"contract_id": "SYNTH"}]}]}
    monkeypatch.setattr(module, "build_distributions_from_markets", lambda *a, **kw: {})
    monkeypatch.setattr(module, "scheduled_macro_events", lambda *a, **kw: [])
    monkeypatch.setattr(module, "build_default_channel_sensitivity", lambda *a, **kw: {})
    monkeypatch.setattr(module, "contract_first_review", lambda *a, **kw: {
        "contract_scenarios": deepcopy([bad, good])})
    result = module.build_default_assessment(report, market_objects=[],
        inputs={"holdings": {}, "instrument_metadata": {}, "prices_by_symbol": {}},
        evaluated_at=datetime(2026, 10, 27, tzinfo=timezone.utc), geopolitical_search=search)
    event = next(row for row in result["events"] if row["event_id"] == "taiwan")
    assert event["risk_ranking"]["contract_id"] == "SYNTH-VALID"
    assert event["risk_ranking"]["q_standard_deviation_usd"] == 10.
    assert event["contract_first"]["contract_scenarios"][0] == bad
    assert event["model_admission"]["primary_impact_allowed"] is False


def test_engine_orders_admitted_conditional_models_before_failed_large_spread(monkeypatch):
    import castle_risk.default_assessment as module
    report = {"portfolio": {"notional": 100000., "positions": [
        {"symbol": "SYNTH", "asset_type": "stock", "weight": 1.}]},
        "empirical": {"daily_betas": {}},
        "events": [{"event_id": name, "title": name, "family": "US_PAYROLLS", "event_date": day}
                   for name, day in [("failed", "2026-11-06"), ("conditional", "2026-12-04")]]}
    monkeypatch.setattr(module, "build_distributions_from_markets", lambda *a, **kw: {})
    monkeypatch.setattr(module, "scheduled_macro_events", lambda *a, **kw: [])
    def synthetic_card(portfolio, original, *args, **kwargs):
        failed = original["event_id"] == "failed"
        event = model_event("COMPLETE_MEASURED_SURPRISE_RESPONSE", q_spread=1e8 if failed else 10.,
                            check="FAIL" if failed else "PASS")
        return {key: deepcopy(event[key]) for key in ("outcome_table", "distribution")}
    monkeypatch.setattr(module, "payroll_card", synthetic_card)
    result = module.build_default_assessment(report, market_objects=[],
        inputs={"holdings": {}, "instrument_metadata": {}, "prices_by_symbol": {}},
        evaluated_at=datetime(2026, 10, 27, tzinfo=timezone.utc))
    assert [e["event_id"] for e in result["events"]] == ["conditional", "failed"]
    conditional, failed = result["events"]
    assert conditional["model_admission"]["state"] == "CONDITIONAL_SCENARIO"
    assert conditional["risk_ranking"]["q_standard_deviation_usd"] == 10.
    assert conditional["risk_ranking"]["forward_prediction_certified"] is False
    assert failed["model_admission"]["state"] == "DIAGNOSTIC_ONLY"
    assert failed["risk_ranking"]["q_standard_deviation_usd"] is None
    assert failed["outcome_table"]["summary"]["q_standard_deviation_usd"] == 1e8
    assert failed["economic_ranking_diagnostics"]["risk_ranking"]["q_standard_deviation_usd"] == 1e8
