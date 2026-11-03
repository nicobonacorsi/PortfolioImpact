"""Offline regressions for event evidence integration.

Synthetic source leads and controlled model fixtures verify that event-specific
evidence reaches conditional impact calculations without network calls.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect

import pytest

from castle_risk.autonomous_common_endpoint_impact import forecast_common_endpoint_impacts
from castle_risk.autonomous_event_impact import forecast_portfolio_impacts
from castle_risk.live_assessment import run_live_assessment
from castle_risk.structural_event_forecast import impact_event_source_bundle
from test_r116_autonomous_event_impact import _context, _proposal, POSITIONS

URL = "https://www.gao.gov/products/event-contingency-plan"


def targeted_evidence(context, *, title="Agency contingency guidance", verified=True):
    """The digest URL was returned by a bounded per-event search."""
    return {context["event_id"]: {
        "schema": "castle.event_targeted_evidence_research.v1",
        "event_id": context["event_id"],
        "target_condition": context["target_condition"],
        "deadline": context["deadline"],
        "as_of_utc": context["as_of_utc"],
        "status": "TARGETED_LEADS_WITH_URL_VERIFIED_DIGEST",
        "leads": [{
            "url": URL, "title": title, "search_query": "agency continuity plans",
            "lead_origin": "EVENT_TARGETED_SEARCH",
        }],
        "findings": [{
            "source_url": URL, "finding": "A synthetic summary of an agency contingency notice.",
            "dated": "2026-11-01", "evidence_type": "OFFICIAL_STATEMENT",
            "direction": "CONTEXT", "url_verified_in_search_results": verified,
        }],
    }}


class CaptureAbstain:
    model = "injected-point3-model"

    def __init__(self):
        self.calls = []

    def structured(self, name, schema, payload, system=""):
        self.calls.append((name, deepcopy(payload), system))
        assert "unverified search-content notes" in system
        return {"assessments": [{
            "event_id": payload["original_event_targets"][0]["event_id"],
            "status": "ABSTAIN",
            "abstain_reason": "Synthetic choice to withhold a numerical shock.",
            "no_event_paths": {}, "no_event_baseline_rationale": "",
            "branches": [],
        }]}


def mark(context):
    return {
        "status": "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION",
        "as_of_utc": context["as_of_utc"],
        "last_completed_session": "2026-10-30",
    }


def test_targeted_digest_unblocks_legacy_and_common_endpoint_impact():
    context = _context()
    evidence = targeted_evidence(context)
    for name, forecaster, kwargs in [
        ("submit_portfolio_impacts", forecast_portfolio_impacts, {}),
        ("submit_common_endpoint_impacts", forecast_common_endpoint_impacts,
         {"observed_mark_context": mark(context)}),
    ]:
        client = CaptureAbstain()
        result = forecaster([context], POSITIONS, 100000., {"sources": []},
                            client, event_evidence=evidence, **kwargs)[context["event_id"]]
        assert len(client.calls) == 1, name
        tool, payload, _ = client.calls[0]
        assert tool == name
        assert payload["original_event_targets"][0]["event_relevant_source_ids"] == ["T01"]
        assert payload["source_leads_unverified"][0]["lead_origin"] == "EVENT_TARGETED_SEARCH"
        assert payload["targeted_evidence_digest_url_verified_untrusted"][0]["source_id"] == "T01"
        assert payload["targeted_evidence_digest_url_verified_untrusted"][0]["content_independently_verified"] is False
        assert result["status"] == "ABSTAIN"
        assert result["reason"].startswith("MODEL_ABSTAINED:")

    # Without the point-2 bundle, the old fail-closed behavior is preserved.
    client = CaptureAbstain()
    row = forecast_common_endpoint_impacts(
        [context], POSITIONS, 100000., {"sources": []}, client,
        observed_mark_context=mark(context))[context["event_id"]]
    assert row["reason"] == "NO_RELEVANT_PUBLIC_RESEARCH_LEAD"
    assert not client.calls


def test_legacy_impact_accepts_citations_to_targeted_sources_with_valid_arithmetic():
    context = _context()
    proposed = _proposal()
    for branch in proposed["branches"]:
        branch["source_ids"] = ["T01"]

    class Propose(CaptureAbstain):
        def structured(self, name, schema, payload, system=""):
            self.calls.append((name, deepcopy(payload), system))
            return {"assessments": [deepcopy(proposed)]}

    client = Propose()
    row = forecast_portfolio_impacts(
        [context], POSITIONS, 100000., {"sources": []}, client,
        event_evidence=targeted_evidence(context))[context["event_id"]]
    assert row["status"] == "MODEL_STRESS_UNCALIBRATED"
    assert row["source_leads"][0]["id"] == "T01"
    assert len(client.calls) == 1


@pytest.mark.parametrize("drift", ["event_id", "target_condition", "deadline", "as_of_utc", "schema"])
def test_cross_event_or_stale_evidence_never_unlocks_an_impact_request(drift):
    context = _context()
    evidence = targeted_evidence(context)
    evidence[context["event_id"]][drift] = "WRONG_OR_STALE"
    sources, digest = impact_event_source_bundle(context, [], evidence)
    assert sources == {}
    assert digest == []
    client = CaptureAbstain()
    row = forecast_common_endpoint_impacts(
        [context], POSITIONS, 100000., {"sources": []}, client,
        observed_mark_context=mark(context), event_evidence=evidence)[context["event_id"]]
    assert row["reason"] == "NO_RELEVANT_PUBLIC_RESEARCH_LEAD"
    assert not client.calls


def test_only_verified_digest_or_predicate_relevant_targeted_leads_are_reused():
    context = _context()
    evidence = targeted_evidence(context, verified=False)
    sources, digest = impact_event_source_bundle(context, [], evidence)
    assert sources == {} and digest == []

    evidence = targeted_evidence(context)
    evidence[context["event_id"]]["leads"][0]["url"] = "https://localhost/private"
    sources, digest = impact_event_source_bundle(context, [], evidence)
    assert sources == {} and digest == []


def test_duplicate_source_url_is_cited_once_and_digest_keeps_provenance():
    context = _context()
    generic = [{
        "id": "W01", "url": URL,
        "title": "Commercial tanker transit through Hormuz",
        "search_query": "Hormuz tanker flows",
        "lead_origin": "PORTFOLIO_LEVEL_SEARCH",
    }]
    sources, digest = impact_event_source_bundle(context, generic, targeted_evidence(context))
    assert list(sources) == ["W01"]
    assert len(digest) == 1 and digest[0]["source_id"] == "W01"


def test_live_runner_passes_identical_event_evidence_to_both_impact_engines():
    tree = ast.parse(inspect.getsource(run_live_assessment))
    expected = {"forecast_portfolio_impacts", "forecast_common_endpoint_impacts"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in expected:
            value = next((arg.value for arg in node.keywords if arg.arg == "event_evidence"), None)
            assert value is not None
            assert ast.unparse(value) == "report.get('event_evidence_research')"
            seen.add(node.func.id)
    assert seen == expected
