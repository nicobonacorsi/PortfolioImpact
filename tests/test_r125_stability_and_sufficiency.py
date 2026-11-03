"""R125 rigor gates: identical-evidence replicate stability and documented-mass sufficiency.

Synthetic model outputs only; the rule thresholds are declared conventions.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.structural_event_forecast import (
    attach_autonomous_forecasts, forecast_events, replicate_stability,
)
from test_r115_structural_event_forecast import EVENT, SOURCE, _context, _proposal


class SequenceClient:
    model = "injected-replicate-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def structured(self, name, schema, payload, system=""):
        self.payloads.append(deepcopy(payload))
        reply = self.replies.pop(0)
        return {"forecasts": [reply]}


def _shift(proposal, delta):
    """Move the first branch's conditional probability to change the centre."""
    shifted = deepcopy(proposal)
    r = shifted["scenarios"][1]["event_conditional_probability"]
    r["center"] = min(1.0, max(0.0, r["center"] + delta))
    r["low"] = min(r["low"], r["center"])
    r["high"] = max(r["high"], r["center"])
    return shifted


def test_two_identical_payload_replicates_are_requested_and_published_as_first_replicate():
    client = SequenceClient([_proposal(), _shift(_proposal(), -.05)])
    result = forecast_events([_context()], SOURCE, client, replicates=2)[EVENT["id"]]
    assert len(client.payloads) == 2 and client.payloads[0] == client.payloads[1]
    assert result["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert result["probability_center"] == pytest.approx(.5), "first replicate is published, not an average"
    assert result["stability"]["status"] == "STABLE"
    assert result["stability"]["numeric_replicates"] == 2
    assert len(result["replicate_forecasts"]) == 1
    assert result["replicate_forecasts"][0]["probability_center"] == pytest.approx(.5 - .45 * .05)


def test_materially_different_replicates_withhold_the_number_but_keep_both_proposals():
    far = _shift(_proposal(), -.6)  # centre drops by 0.27
    client = SequenceClient([_proposal(), far])
    result = forecast_events([_context()], SOURCE, client, replicates=2)[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "UNSTABLE_MODEL_JUDGMENT_ACROSS_REPLICATES"
    assert result["abstention_class"] == "MODEL_INSTABILITY_ACROSS_REPLICATES"
    assert result["probability_center"] is None and result["credal_interval"] is None
    centres = [r["probability_center"] for r in result["withheld_model_proposal"]["replicates"]]
    assert centres[0] == pytest.approx(.5) and centres[1] < .25
    case = {"risks": [{"id": EVENT["id"], "event_probability_question": _context()}]}
    attach_autonomous_forecasts(case, {EVENT["id"]: result})
    assert case["risks"][0]["autonomous_event_forecast"]["status"] == "ABSTAIN"


def test_replicate_that_abstains_on_the_same_evidence_counts_as_instability():
    abstain = {"event_id": EVENT["id"], "status": "ABSTAIN", "abstain_reason": "Evidence too thin on reflection.",
               "scenarios": []}
    result = forecast_events([_context()], SOURCE, SequenceClient([_proposal(), abstain]), replicates=2)[EVENT["id"]]
    assert result["status"] == "ABSTAIN" and result["abstention_class"] == "MODEL_INSTABILITY_ACROSS_REPLICATES"
    assert result["withheld_model_proposal"]["stability"]["evidence_abstentions"] == 1


def test_transport_failure_in_a_replicate_is_not_instability():
    class Flaky(SequenceClient):
        def structured(self, name, schema, payload, system=""):
            if len(self.payloads) == 1:
                self.payloads.append(deepcopy(payload))
                from castle_risk.agent import AgentError
                raise AgentError("injected timeout")
            return super().structured(name, schema, payload, system)
    result = forecast_events([_context()], SOURCE, Flaky([_proposal()]), replicates=2)[EVENT["id"]]
    assert result["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert result["stability"]["status"] == "INSUFFICIENT_REPLICATES"
    assert result["stability"]["transport_failures"] == 1


def test_stability_rule_thresholds_are_the_declared_conventions():
    def row(center, low, high, status="MODEL_FORECAST_UNCALIBRATED"):
        return {"status": status, "probability_center": center, "credal_interval": [low, high]}
    # gap 0.05 with mean width 0.275: negligible gap -> stable
    assert replicate_stability([row(.30, .20, .45), row(.35, .25, .50)])["status"] == "STABLE"
    # gap 0.15 against a mean width 0.275 (ratio 0.55) -> withheld
    assert replicate_stability([row(.30, .20, .45), row(.45, .30, .60)])["status"] == "UNSTABLE"
    # the live Hormuz probe A: gap 0.24, mean width 0.47 (ratio 0.51), centres mutually covered -> withheld
    probe_a = replicate_stability([row(.52, .2675, .7545), row(.3625, .17, .6075), row(.6025, .347, .8115)])
    assert probe_a["status"] == "UNSTABLE" and probe_a["mutual_center_coverage"] is True
    # the live tariff probe C: gap 0.1125, mean width about 0.41 (ratio 0.27) -> stable
    assert replicate_stability([row(.454, .2466, .656), row(.5325, .30, .72), row(.42, .22, .62)])["status"] == "STABLE"
    # gap 0.16 against mean width 0.40 (ratio 0.4) with mutual coverage -> high dispersion, published with warning
    assert replicate_stability([row(.30, .10, .50), row(.46, .25, .65)])["status"] == "HIGH_DISPERSION"
    # a centre outside the other replicate's interval is unstable even with overlapping intervals
    assert replicate_stability([row(.10, .05, .15), row(.30, .12, .40)])["status"] == "UNSTABLE"
    # disjoint intervals are unstable
    assert replicate_stability([row(.10, .05, .15), row(.30, .20, .40)])["status"] == "UNSTABLE"


def test_assumption_driven_probability_mass_is_withheld_with_the_proposal_visible():
    proposal = _proposal()
    # Branch 1 carries 0.36 of the 0.50 centre and is declared an undocumented assumption.
    proposal["scenarios"][1]["evidence_basis"] = "ASSUMPTION_NOT_DOCUMENTED"
    result = forecast_events([_context()], SOURCE, SequenceClient([proposal]))[EVENT["id"]]
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "EVIDENCE_INSUFFICIENT_ASSUMPTION_DRIVEN_PROBABILITY_MASS"
    assert result["abstention_class"] == "EVIDENCE_INSUFFICIENT_ASSUMPTION_DRIVEN_MASS"
    withheld = result["withheld_model_proposal"]
    assert withheld["probability_center"] == pytest.approx(.5)
    assert withheld["documented_probability_mass_share"] == pytest.approx(1 - .36 / .5)
    case = {"risks": [{"id": EVENT["id"], "event_probability_question": _context()}]}
    attach_autonomous_forecasts(case, {EVENT["id"]: result})
    assert case["risks"][0]["autonomous_event_forecast"]["withheld_model_proposal"]["probability_center"] == pytest.approx(.5)


def test_documented_and_partly_documented_shares_are_reported_and_uncited_branches_are_assumptions():
    proposal = _proposal()
    proposal["scenarios"][0]["evidence_basis"] = "DOCUMENTED_IN_SUPPLIED_SOURCES"
    proposal["scenarios"][1]["evidence_basis"] = "DOCUMENTED_IN_SUPPLIED_SOURCES"
    proposal["scenarios"][2]["source_ids"] = []
    proposal["scenarios"][2]["evidence_basis"] = "DOCUMENTED_IN_SUPPLIED_SOURCES"  # overridden by code
    result = forecast_events([_context()], SOURCE, SequenceClient([proposal]))[EVENT["id"]]
    assert result["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert result["branch_evidence_bases"] == ["DOCUMENTED_IN_SUPPLIED_SOURCES", "DOCUMENTED_IN_SUPPLIED_SOURCES",
                                               "ASSUMPTION_NOT_DOCUMENTED"]
    assert result["documented_probability_mass_share"] == pytest.approx(1 - .09 / .5)
    assert result["evidence_sufficiency"] == "PARTLY_DOCUMENTED", "an uncited branch keeps the forecast partly documented"
    fully = _proposal()
    for branch in fully["scenarios"]:
        branch["evidence_basis"] = "DOCUMENTED_IN_SUPPLIED_SOURCES"
    assert forecast_events([_context()], SOURCE, SequenceClient([fully]))[EVENT["id"]]["evidence_sufficiency"] == "DOCUMENTED"
    legacy = _proposal()
    for branch in legacy["scenarios"]:
        branch.pop("evidence_basis")
    result = forecast_events([_context()], SOURCE, SequenceClient([legacy]))[EVENT["id"]]
    assert result["evidence_sufficiency"] == "PARTLY_DOCUMENTED"
    assert result["scenario_branches"][0]["basis_adjusted_by_code"].startswith("LEGACY_RESPONSE")
