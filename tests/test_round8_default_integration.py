"""Scope and presentation joins using explicitly synthetic integration evidence.

These fixtures certify code behavior only. They are not provider captures,
authenticated acceptance, calibrated probabilities, or empirical event effects.
The macro ranking tests exercise the real economic component builder with a
small declared synthetic probability polytope; no fitted outputs are mocked.
"""
from copy import deepcopy
from datetime import datetime
import socket

import pytest

from castle_risk import adapters
from castle_risk.default_assessment import build_default_assessment


NOW = datetime.fromisoformat("2026-10-21T12:00:00+00:00")
LATER = "2026-10-21T14:55:00+00:00"
FIXTURE_SCOPE = "SYNTHETIC_INTEGRATION_FIXTURE_NOT_MARKET_OR_ACCEPTANCE_EVIDENCE"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("The default projection must not acquire data.")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(adapters, "_transport", deny)
    monkeypatch.setattr(adapters._ReadOnlyClient, "_get", deny)


def synthetic_report():
    return {
        "fixture_scope": FIXTURE_SCOPE,
        "portfolio": {"positions": [
            {"symbol": "NVDA", "weight": 1.0, "asset_type": "stock"}],
            "notional": 100000.0},
        "verification_status": "SYNTHETIC_AUTHENTICATED_ACCEPTANCE_NOT_COMPLETED",
        "physical_probability_P": None,
        "holdings_quality_review": [{"status": "UNKNOWN", "reason": FIXTURE_SCOPE}],
        "empirical": {
            "daily_betas": {"kind": "INSTRUMENT_DAILY_BETAS",
                "benchmark_symbol": "SPY", "analysis_as_of": NOW.isoformat(),
                "train_end": "2025-10-21", "fixture_scope": FIXTURE_SCOPE,
                "instruments": {"NVDA": {"beta": 1.5, "beta_interval": [1.4, 1.6],
                    "n_train": 500, "n_holdout": 200,
                    "transfer_check": {"status": "FAIL", "reason": FIXTURE_SCOPE}}}},
            "release_day_studies": {},
            "prior_A6_forward_transfer_status": "FAIL_RETAINED_NOT_OVERRIDDEN"},
        "events": [
            {"event_id": "synthetic-taiwan", "title": "Synthetic Taiwan supply risk",
             "family": "TAIWAN_SUPPLY", "event_date": "2027-01-19",
             "question_we_care_about": "Synthetic question: blockade, quarantine or armed action disrupts Taiwan supply by 2027-01-19.",
             "matching_comparisons": [{"decision": "UNKNOWN", "nested": {"fixture_scope": FIXTURE_SCOPE}}]},
            {"event_id": "synthetic-export", "title": "Synthetic export-control risk",
             "family": "EXPORT_CONTROLS", "event_date": "2027-01-19",
             "question_we_care_about": "Synthetic question: the US imposes new advanced-chip export restrictions by 2027-01-19.",
             "matching_comparisons": [{"decision": "UNKNOWN", "nested": {"fixture_scope": FIXTURE_SCOPE}}]},
        ],
        "limitations": [FIXTURE_SCOPE, "Windows cold-start remains unverified in this fixture."],
    }


def synthetic_search(report, *, stamp=NOW.isoformat()):
    targets = []
    for event in report["events"]:
        if event["family"] not in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}:
            continue
        candidate = {
            "contract_id": "SYNTHETIC-RELATED-" + event["event_id"],
            "market_question": "Synthetic narrower contract, not equivalent to the broad event",
            "decision": "RELATED_ONLY", "fixture": True,
            "source": {"fixture": True, "retrieved_at": stamp},
            "contract_quote_reference": {"interval": [.02, .03], "evaluated_at": stamp,
                "scope": FIXTURE_SCOPE, "transferred_to_parent_event": False},
        }
        targets.append({
            "event_id": event["event_id"], "event_date": event["event_date"],
            "question_we_care_about": event["question_we_care_about"],
            "decision": "NO_EXACT_FIT_WITHIN_SEARCHED_SCOPE", "fixture_scope": FIXTURE_SCOPE,
            "candidates": [candidate],
            "closest_by_venue": {"Polymarket": {"status": "SEARCH_COMPLETED_WITHIN_BOUNDS",
                "closest_candidate": deepcopy(candidate)}},
            "event_pricing_measure_Q_interval": None,
        })
    return {"fixture_scope": FIXTURE_SCOPE, "evaluated_at": stamp,
            "capture_evaluated_at": stamp, "targets": targets}


def project(report, **kwargs):
    return build_default_assessment(
        report, market_objects={}, inputs={}, evaluated_at=NOW, **kwargs)


def geo_cards(analysis):
    """Broad geopolitical cards; scheduled FOMC/CPI cards are added by the calendar rule."""
    return [event for event in analysis["events"] if event["family"] in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}]


def by_id(analysis):
    return {event["event_id"]: event for event in analysis["events"]}


def synthetic_distribution(*, q=(.35, .65, 0.), bounds=None,
                           third_source_bounds=(100., 100.), third_direction="hike"):
    """A declared test polytope; the source identity is explicitly fictional."""
    atoms = [
        {"outcome_id": "cut", "label": "Synthetic cut", "direction": "cut",
         "value": -25., "value_interval": [-25., -25.], "q_reference": q[0]},
        {"outcome_id": "hold", "label": "Synthetic hold", "direction": "hold",
         "value": 0., "value_interval": [0., 0.], "q_reference": q[1]},
        {"outcome_id": "third", "label": "Synthetic other outcome", "direction": third_direction,
         "value": None, "value_interval": list(third_source_bounds), "q_reference": q[2]},
    ]
    return {
        "event_ticker": "SYNTHETIC-FED", "family": "FED_POLICY", "event_date": "2026-10-28",
        "evaluated_at": NOW.isoformat(), "status": "SYNTHETIC_TEST_DISTRIBUTION",
        "metric_identity": "fed_target_rate_change", "probability_question": FIXTURE_SCOPE,
        "source": {"fixture": True, "identity": "SYNTHETIC-FED", "scope": FIXTURE_SCOPE},
        "atoms": atoms, "constraints": {"atom_ids": [atom["outcome_id"] for atom in atoms],
            "A_eq": [[1., 1., 1.]], "b_eq": [1.],
            "bounds": bounds if bounds is not None else [[mass, mass] for mass in q]},
    }


def add_macro(report):
    report["events"].append({
        "event_id": "synthetic-fed", "title": "Synthetic scheduled FOMC decision",
        "family": "FED_POLICY", "event_date": "2026-10-28",
        "question_we_care_about": "Synthetic broad policy question with no reassigned physical probability.",
        "matching_comparisons": [{"decision": "UNKNOWN", "fixture_scope": FIXTURE_SCOPE}],
    })


@pytest.mark.parametrize("event_id", ["synthetic-taiwan", "synthetic-export"])
def test_exact_question_and_horizon_join_preserves_related_contract_scope(event_id):
    report = synthetic_report()
    search = synthetic_search(report)
    card = by_id(project(report, geopolitical_search=search))[event_id]
    expected = next(target for target in search["targets"] if target["event_id"] == event_id)
    assert card["market_search"] == expected
    assert card["market_search"]["event_pricing_measure_Q_interval"] is None
    assert card["physical_probability_P"] is None
    assert not card["original_question_probability_reassigned"]
    assert card["economics"]["event_definition"] == card["original_question"]
    assert card["economics"]["matched_probability_required_for_priced_in"]
    assert card["ranking"]["abs_delta_pnl_usd"] is None


@pytest.mark.parametrize("event_id", ["synthetic-taiwan", "synthetic-export"])
@pytest.mark.parametrize("field,replacement", [
    ("question_we_care_about", "A different synthetic question sharing only the event ID"),
    ("event_date", "2027-02-19"),
])
def test_question_or_horizon_mismatch_cannot_be_bypassed_by_separate_clock(event_id, field, replacement):
    report = synthetic_report()
    search = synthetic_search(report, stamp=LATER)
    target = next(target for target in search["targets"] if target["event_id"] == event_id)
    target[field] = replacement
    card = by_id(project(report, geopolitical_search=search,
                         allow_separate_search_clock=True))[event_id]
    assert card["market_search"]["decision"] == "SEARCH_SCOPE_MISMATCH"
    assert card["market_search"]["candidates"] == []
    assert card["market_search"]["closest_by_venue"] == {}
    assert card["market_search"]["event_pricing_measure_Q_interval"] is None


def test_future_search_is_refused_by_default_for_both_broad_questions():
    report = synthetic_report()
    result = project(report, geopolitical_search=synthetic_search(report, stamp=LATER))
    for card in geo_cards(result):
        assert card["market_search"]["decision"] == "SEARCH_AFTER_MACRO_EVALUATION_NOT_JOINED"
        assert card["market_search"]["candidates"] == []
        assert card["market_search_context"]["search_evaluated_at"] == LATER
        assert not card["market_search_context"]["separate_recorded_clock_explicitly_requested"]
        assert card["market_search"]["event_pricing_measure_Q_interval"] is None


def test_explicit_separate_search_retains_both_clocks_without_reassigning_q():
    report = synthetic_report()
    search = synthetic_search(report, stamp=LATER)
    result = project(report, geopolitical_search=search, allow_separate_search_clock=True)
    for card in geo_cards(result):
        context = card["market_search_context"]
        assert context["macro_evaluated_at"] == NOW.isoformat()
        assert context["search_evaluated_at"] == context["capture_evaluated_at"] == LATER
        assert context["separate_recorded_clock_explicitly_requested"]
        assert context["scope"] == "SEPARATE_DATED_MARKET_SEARCH_NOT_A_QUOTE_AT_MACRO_CAPTURE_TIME"
        assert card["market_search"]["candidates"]
        assert card["market_search"]["event_pricing_measure_Q_interval"] is None
        assert card["physical_probability_P"] is None


def test_equivalent_timezone_representation_is_not_a_future_search():
    report = synthetic_report()
    result = project(report, geopolitical_search=synthetic_search(report, stamp="2026-10-21T08:00:00-04:00"))
    assert all(card["market_search"]["candidates"] for card in geo_cards(result))


def test_failed_search_remains_unavailability_not_evidence_of_no_market():
    report = synthetic_report()
    search = synthetic_search(report)
    for target in search["targets"]:
        target.update(decision="SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE", candidates=[],
                      closest_by_venue={"Kalshi": {"status": "SEARCH_FAILED", "closest_candidate": None},
                                        "Polymarket": {"status": "SEARCH_FAILED", "closest_candidate": None}})
    result = project(report, geopolitical_search=search)
    assert all(card["market_search"]["decision"] == "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE"
               for card in geo_cards(result))


def test_zero_feasible_mass_state_stays_visible_but_cannot_set_primary_rank():
    report = synthetic_report()
    add_macro(report)
    result = project(report, distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = by_id(result)["synthetic-fed"]
    cases = {case["outcome_id"]: case for case in card["economics"]["outcome_cases"]}
    assert cases["third"]["q_interval"] == pytest.approx([0., 0.])
    assert abs(cases["third"]["delta_pnl_usd"]) > card["ranking"]["abs_delta_pnl_usd"]
    assert card["ranking"]["selected_outcome_id"] in {"cut", "hold"}
    assert result["events"][0]["event_id"] == "synthetic-fed"
    assert result["events"][0]["ranking"]["display_order"] == 1


@pytest.mark.parametrize("source_bounds", [(0., None), (None, None)])
def test_open_or_opaque_source_magnitude_stress_cannot_set_primary_rank(source_bounds):
    report = synthetic_report()
    add_macro(report)
    source = synthetic_distribution(q=(.35, .45, .20), third_source_bounds=source_bounds)
    card = by_id(project(report, distribution_overrides={"SYNTHETIC-FED": source}))["synthetic-fed"]
    cases = {case["outcome_id"]: case for case in card["economics"]["outcome_cases"]}
    assert cases["third"]["q_interval"][0] > 0
    assert abs(cases["third"]["delta_pnl_usd"]) > card["ranking"]["abs_delta_pnl_usd"]
    assert cases["third"]["state_model"]["source_value_interval"] == list(source_bounds)
    assert card["ranking"]["selected_outcome_id"] == "cut"
    assert card["ranking"]["unbounded_source_tail_stresses_excluded_from_primary_order"]
    assert card["ranking"]["complement_expectation_may_still_require_declared_tail_assumptions"]
    assert card["economics"]["unrestricted_support_review"]["finite_economic_domain_is_declared_assumption"]


def test_zero_reference_weight_with_positive_possible_mass_is_not_excluded():
    report = synthetic_report()
    add_macro(report)
    source = synthetic_distribution(q=(.1, .9, 0.), bounds=[[0., .2], [.6, 1.], [0., .2]])
    card = by_id(project(report, distribution_overrides={"SYNTHETIC-FED": source}))["synthetic-fed"]
    third = next(case for case in card["economics"]["outcome_cases"] if case["outcome_id"] == "third")
    assert third["q_reference"] == 0
    assert third["q_interval"][1] > 0
    assert card["ranking"]["selected_outcome_id"] == "third"


def test_entirely_unbounded_source_states_leave_rank_unknown_without_hiding_event():
    report = synthetic_report()
    add_macro(report)
    source = synthetic_distribution(q=(.5, .5, 0.))
    for atom in source["atoms"]:
        atom.update(value=None, value_interval=[None, None])
    card = by_id(project(report, distribution_overrides={"SYNTHETIC-FED": source}))["synthetic-fed"]
    assert len(card["economics"]["outcome_cases"]) == 3
    assert card["ranking"]["selected_outcome_id"] is None
    assert card["ranking"]["abs_delta_pnl_usd"] is None


def test_projection_and_nested_edits_do_not_mutate_retained_report_or_evidence():
    report = synthetic_report()
    add_macro(report)
    report["events"].append({"event_id": "synthetic-exact-child", "parent_event_id": "synthetic-fed",
                             "fixture_scope": FIXTURE_SCOPE})
    search = synthetic_search({**report, "events": report["events"][:-1]})
    source = synthetic_distribution()
    inputs = {"holdings": {}, "fixture_scope": FIXTURE_SCOPE}
    markets = {}
    before = deepcopy((report, search, source, inputs, markets))
    result = build_default_assessment(report, market_objects=markets, inputs=inputs,
        evaluated_at=NOW, geopolitical_search=search,
        distribution_overrides={"SYNTHETIC-FED": source})
    assert (report, search, source, inputs, markets) == before
    assert "synthetic-exact-child" not in by_id(result)
    assert not result["original_authenticated_acceptance_upgraded"]
    assert result["new_model_or_provider_calls_by_this_function"] == 0
    assert report["empirical"]["prior_A6_forward_transfer_status"] == "FAIL_RETAINED_NOT_OVERRIDDEN"
    assert by_id(result)["synthetic-fed"]["economics"]["daily_transfer_failed_symbols"] == ["NVDA"]

    # Mutating derived nested structures must not alias retained evidence.
    by_id(result)["synthetic-fed"]["distribution"]["atoms"][0]["label"] = "Changed derived display"
    taiwan = by_id(result)["synthetic-taiwan"]
    taiwan["market_search"]["candidates"][0]["market_question"] = "Changed derived candidate"
    taiwan["original_matching_comparisons"][0]["nested"]["fixture_scope"] = "Changed derived comparison"
    assert (report, search, source, inputs, markets) == before


def test_scheduled_macro_events_are_added_from_the_official_calendar_without_duplicates():
    """Redesign (2026-10-23): every FOMC and CPI inside the 90-day horizon is listed, whatever the model proposed.

    r8 widened this from the next ones inside 45 days: the 15:21 live runs listed the
    December FOMC and CPI for one portfolio but not another, depending on proposals.
    """
    report = synthetic_report()
    result = project(report)
    scheduled = {(e["family"], e["event_date"]) for e in result["events"]}
    assert ("FED_POLICY", "2026-10-28") in scheduled
    assert ("US_INFLATION", "2026-11-10") in scheduled
    assert ("FED_POLICY", "2026-12-09") in scheduled  # 49 days after NOW: inside the 90-day horizon
    assert ("US_INFLATION", "2026-12-10") in scheduled
    origins = {e["event_date"]: e["definition_origin"] for e in result["events"] if e["family"] in {"FED_POLICY", "US_INFLATION"}}
    assert origins["2026-10-28"] == origins["2026-11-10"] == origins["2026-12-09"] == origins["2026-12-10"] == \
        "OFFICIAL_CALENDAR_SCHEDULED_MACRO"
    add_macro(report)
    result = project(report)
    fomc = [e for e in result["events"] if e["family"] == "FED_POLICY" and e["event_date"] == "2026-10-28"]
    assert [e["event_id"] for e in fomc] == ["synthetic-fed"]


def test_contract_first_scenario_joins_the_broad_card_without_replacing_it():
    """I3 (2026-10-23): an admitted blockade contract is priced as its own scenario; the broad card stays."""
    report = synthetic_report()
    search = synthetic_search(report)
    taiwan = next(t for t in search["targets"] if t["event_id"] == "synthetic-taiwan")
    taiwan["candidates"][0].update(
        venue="Polymarket", contract_end_time="2026-12-31T23:59:00Z", market_open_at_evaluation=True,
        rules_text=("China (People's Republic of China) establishes a naval blockade of Taiwan that prevents the "
                    "normal ingress or egress of foreign commercial traffic to or from Taiwan's main ports by 2026-12-31."))
    # An event description alone says nothing about whether YES or NO means
    # the action happens. The same description needs an explicit YES resolver.
    unbound = by_id(project(report, geopolitical_search=search))["synthetic-taiwan"]["contract_first"]
    assert unbound["candidate_reviews"][0]["decision"] == "REJECTED_DIFFERENT_EVENT"
    assert unbound["contract_scenarios"] == []
    taiwan["candidates"][0]["rules_text"] = ('This market will resolve to "Yes" if '
                                               + taiwan["candidates"][0]["rules_text"])
    card = by_id(project(report, geopolitical_search=search))["synthetic-taiwan"]
    assert card["status"] == "IDENTIFIED_CHANNEL_SENSITIVITY_WITH_UNMATCHED_EVENT_PROBABILITY"
    review = card["contract_first"]
    assert [r["decision"] for r in review["candidate_reviews"]] == ["ADMITTED_CONTRACT_SCENARIO"]
    scenario = review["contract_scenarios"][0]
    assert scenario["outcomes"][0]["pnl_if_outcome_usd"] < 0 < scenario["outcomes"][1]["pnl_if_outcome_usd"]
    assert card["risk_ranking"]["contract_id"] == taiwan["candidates"][0]["contract_id"]
    # The bound is stated for the reviewed Taiwan definition, whatever the fixture question's wording.
    from castle_risk.contract_scenarios import REVIEWED_RISKS
    probability = review["broad_risk"]["probability"]
    assert probability["status"] == "CONDITIONAL_LOWER_BOUND_ONLY"
    assert probability["lower_bounds_by_venue"] == []
    bound = probability["conditional_lower_bounds_by_venue"][0]
    assert bound["lower_bound"] is None and bound["conditional_lower_bound"] == .02
    assert not bound["unconditional_bound_available"]
    assert "No qualifying contract event occurred before evaluation" in bound["condition"]
    assert review["broad_risk"]["probability_definition"] == REVIEWED_RISKS["TAIWAN"]
    assert review["broad_risk"]["definition"] == card["original_question"]
    export = by_id(project(report, geopolitical_search=search))["synthetic-export"]
    assert export["contract_first"]["candidate_reviews"][0]["decision"] == "REJECTED_DIFFERENT_EVENT"
