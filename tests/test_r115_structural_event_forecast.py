"""R115 autonomous forecasting admission; all proposed values are synthetic.

This is an injected model-interface test, not measured live forecasting skill.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.agent import validate_schema
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.structural_event_forecast import BATCH_SCHEMA, forecast_events, attach_autonomous_forecasts
from castle_risk.strict_schema import transport_schema


EVENT = {
    "id": "HORMUZ_EVENT", "date": "2027-01-30",
    "title": "Commercial tanker transit through Hormuz interrupted by 2027-01-30",
}
CASE = {"capture": "2026-11-01T18:10:22+00:00"}
SOURCE = {"status": "SEARCHED_SOURCE_LEADS_NOT_VERIFIED",
          "sources": [{"url": "https://www.gao.gov/reports/example.pdf",
                       "title": "Documented contextual source lead",
                       "search_query": "Hormuz maritime traffic",
                       "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED"}]}


def _branch(i, mass, mass_low, mass_high, conditional, cond_low, cond_high):
    return {
        "id": f"scenario{i}",
        "trigger": f"Operational escalation indicator {i}",
        "residual": i == 2,
        "scenario_probability": {"low": mass_low, "center": mass, "high": mass_high},
        "event_conditional_probability": {"low": cond_low, "center": conditional, "high": cond_high},
        "rationale": "A specified modeled transmission mechanism with stated uncertainty.",
        "counterevidence": "A different diplomatic or operational trajectory may intervene.",
        "evidence_basis": "PARTLY_DOCUMENTED",
        "source_ids": ["W01"],
    }


def _proposal():
    return {
        "event_id": EVENT["id"], "status": "PROPOSED",
        "abstain_reason": "Not applicable",
        "scenarios": [
            _branch(0, .25, .1, .4, .2, .1, .3),
            _branch(1, .45, .2, .6, .8, .7, .9),
            _branch(2, .30, 0, .5, .3, .2, .4),
        ],
    }


class FakeClient:
    model = "injected-test-model"

    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def structured(self, name, schema, payload, system=""):
        self.calls.append((name, deepcopy(payload)))
        assert name == "submit_structural_forecasts"
        assert "not a market-data provider" in system.lower()
        return {"forecasts": deepcopy(self.proposals)}


def _context():
    result = probability_question_context(CASE, EVENT)
    assert result is not None
    return result


def test_strict_tool_schema_is_transport_compatible():
    transport_schema(BATCH_SCHEMA)


def test_proposed_forecast_may_correctly_have_empty_abstain_reason():
    # Actual Anthropic transport returned precisely this zero-length field in
    # a PROPOSED event, and the old minLength=1 schema threw away its P(E).
    proposed = _proposal()
    proposed['abstain_reason'] = ''
    validate_schema({'forecasts': [proposed]}, BATCH_SCHEMA)
    estimate = forecast_events([_context()], SOURCE, FakeClient([proposed]))[EVENT['id']]
    assert estimate['status'] == 'MODEL_FORECAST_UNCALIBRATED'
    assert estimate['probability_center'] == pytest.approx(.5)


def test_abstention_without_explanation_remains_an_explicit_abstention():
    abstention = {**_proposal(), 'status': 'ABSTAIN', 'abstain_reason': '', 'scenarios': []}
    validate_schema({'forecasts': [abstention]}, BATCH_SCHEMA)
    result = forecast_events([_context()], SOURCE, FakeClient([abstention]))[EVENT['id']]
    assert result['status'] == 'ABSTAIN'
    assert result['reason'] == 'MODEL_ABSTAINED_WITHOUT_EXPLANATION'
    assert result['probability_center'] is None


def test_verbose_genuine_abstention_reason_is_not_a_transport_failure():
    # A real Anthropic ABSTAIN on 2026-11-02 used 566 characters; the old
    # shared 550-character field silently became MODEL_REQUEST_FAILED.
    reason = 'Insufficient verified source evidence for an event estimate. ' * 10
    assert 550 < len(reason) < 1200
    decision = {**_proposal(), 'status': 'ABSTAIN', 'abstain_reason': reason, 'scenarios': []}
    validate_schema({'forecasts': [decision]}, BATCH_SCHEMA)
    result = forecast_events([_context()], SOURCE, FakeClient([decision]))[EVENT['id']]
    assert result['status'] == 'ABSTAIN'
    assert result['reason'].startswith('MODEL_ABSTAINED: ')
    assert result['probability_center'] is None


def test_autonomous_forecast_matches_original_predicate_not_market_or_portfolio_q():
    client = FakeClient([_proposal()])
    result = forecast_events([_context()], SOURCE, client)[EVENT["id"]]
    assert len(client.calls) == 1
    assert result["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert result["target_condition"] == EVENT["title"]
    assert result["probability_center"] == pytest.approx(.5)
    assert result["credal_interval"] == pytest.approx([.26, .69])
    assert result["physical_probability_empirically_identified"] is False
    assert result["forecast_calibrated"] is False
    assert result["equity_pricing_q_changed"] is False
    assert result["portfolio_pnl_changed"] is False
    assert result["source_leads"][0]["url"] == SOURCE["sources"][0]["url"]


def test_missing_retrieved_source_abstains_without_paid_model_call():
    client = FakeClient([_proposal()])
    result = forecast_events([_context()], {"sources": []}, client)[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["probability_center"] is None
    assert client.calls == []


@pytest.mark.parametrize("change", [
    lambda p: p["scenarios"][0].update(source_ids=["W99"]),
    lambda p: p["scenarios"][0].update(source_ids=[]),
    lambda p: p["scenarios"][1]["scenario_probability"].update(center=.3),
    lambda p: p["scenarios"][1]["event_conditional_probability"].update(low=.95),
    lambda p: p["scenarios"][2].update(residual=False),
])
def test_model_output_cannot_bypass_source_or_probabilistic_checks(change):
    proposal = _proposal()
    change(proposal)
    result = forecast_events([_context()], SOURCE, FakeClient([proposal]))[EVENT["id"]]
    # Missing one branch-specific source is allowed when others are linked.
    if proposal["scenarios"][0].get("source_ids") == []:
        assert result["status"] == "MODEL_FORECAST_UNCALIBRATED"
    else:
        assert result["status"] == "ABSTAIN"


def test_model_omission_or_duplicate_abstains_without_filling_missing_p():
    omitted = forecast_events([_context()], SOURCE, FakeClient([]))[EVENT["id"]]
    duplicated = forecast_events([_context()], SOURCE, FakeClient([_proposal(), _proposal()]))[EVENT["id"]]
    assert omitted["probability_center"] is None
    assert duplicated["status"] == "ABSTAIN"


def test_vacuous_structural_credal_interval_cannot_appear_as_precision():
    proposal = _proposal()
    for branch in proposal["scenarios"]:
        branch["event_conditional_probability"].update(low=0., high=1.)
    result = forecast_events([_context()], SOURCE, FakeClient([proposal]))[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "VACUOUS_CREDAL_INTERVAL_NO_INFORMATIVE_BOUND"


def test_scheduled_release_without_target_outcome_is_not_forecasted():
    scheduled = {**EVENT, "id": "fomc-2027-01-27-scheduled",
                 "title": "FOMC policy decision on 2027-01-27 (scheduled)",
                 "date": "2027-01-27"}
    context = probability_question_context(CASE, scheduled)
    client = FakeClient([])
    result = forecast_events([context], SOURCE, client)[scheduled["id"]]
    assert result["status"] == "ABSTAIN"
    assert client.calls == []


def test_public_serialization_rechecks_math_identity_and_preserves_pnl():
    forecast = forecast_events([_context()], SOURCE, FakeClient([_proposal()]))[EVENT["id"]]
    base = {"risks": [{"id": EVENT["id"], "title": EVENT["title"],
                       "date": EVENT["date"], "event_probability_question": _context(),
                       "outcomes": [{"pnl": -1200, "q_reference": .4}]}]}
    case = deepcopy(base)
    attach_autonomous_forecasts(case, {EVENT["id"]: forecast})
    assert case["risks"][0]["autonomous_event_forecast"]["probability_center"] == pytest.approx(.5)
    assert case["risks"][0]["outcomes"] == base["risks"][0]["outcomes"]
    for corruption in (
        lambda r: r.update(credal_interval=[.2, .69]),
        lambda r: r.update(probability_center=float("nan")),
        lambda r: r.update(credal_interval=[float("nan"), .69]),
        lambda r: r.update(deadline="2027-01-29"),
        lambda r: r.update(forecast_calibrated=True),
        lambda r: r["scenario_branches"][1]["scenario_probability"].update(center=.5),
        lambda r: r["source_leads"][0].update(id="W99"),
        lambda r: r["source_leads"][0].update(url="https://127.0.0.1/private"),
        lambda r: r["source_leads"][0].update(url="https://www.gao.gov/?token=secret"),
        lambda r: r.update(scenario_branches=7),
        lambda r: r["scenario_branches"][0].update(source_ids=9),
    ):
        changed = deepcopy(forecast)
        corruption(changed)
        case = deepcopy(base)
        attach_autonomous_forecasts(case, {EVENT["id"]: changed})
        assert "autonomous_event_forecast" not in case["risks"][0]
        assert case["risks"][0]["outcomes"] == base["risks"][0]["outcomes"]


def test_public_serialization_exposes_explicit_abstention_not_fake_zero():
    result = forecast_events([_context()], {"sources": []}, FakeClient([]))[EVENT["id"]]
    case = {"risks": [{"id": EVENT["id"], "event_probability_question": _context()}]}
    attach_autonomous_forecasts(case, {EVENT["id"]: result})
    published = case["risks"][0]["autonomous_event_forecast"]
    assert published["status"] == "ABSTAIN"
    assert published["probability_center"] is None
    assert published["credal_interval"] is None


def test_unrelated_portfolio_research_leads_are_not_treated_as_event_evidence():
    other = {"status": SOURCE["status"], "sources": [
        {"url": "https://www.gao.gov/reports/treasury-yields.pdf",
         "title": "US Treasury duration and yields",
         "search_query": "Federal Reserve Treasury interest rate outlook",
         "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED"}]}
    client = FakeClient([_proposal()])
    result = forecast_events([_context()], other, client)[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "NO_EVENT_RELEVANT_RETRIEVED_SOURCE_LEAD"
    assert client.calls == []


def test_source_citations_must_be_relevant_to_the_specific_event():
    other = {"url": "https://www.gao.gov/reports/treasury-yields.pdf",
             "title": "Treasury yields", "search_query": "Federal Reserve rates",
             "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED"}
    proposal = _proposal()
    proposal["scenarios"][0]["source_ids"] = ["W02"]
    client = FakeClient([proposal])
    result = forecast_events([_context()],
                             {**SOURCE, "sources": [*SOURCE["sources"], other]},
                             client)[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "UNVERIFIED_SOURCE_ID_REFERENCE"
    assert client.calls[0][1]["event_targets"][0]["event_relevant_source_ids"] == ["W01"]
