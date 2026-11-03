"""R125 point-2 stage orchestration with injected clients: explicit failures, tagged ledger."""
from __future__ import annotations

from datetime import datetime, timezone

from castle_risk.agent import AgentError
from castle_risk.point2_evidence_stage import (
    _definition_detail, _proposed_queries, _seed_candidates, run_point2_evidence_stage,
)
from castle_risk.probability_assumptions import probability_question_context
from test_r125_event_market_search import FakeCatalog, FakeGamma, _kalshi_market


CASE = {"capture": "2026-11-02T10:00:00+00:00"}
EVENT = {"id": "US_SOVEREIGN_DOWNGRADE_2026_12_31", "date": "2026-12-31",
         "title": "Moody's, S&P or Fitch lowers the long-term US sovereign credit rating by 2026-12-31"}
URL = "https://www.fitchratings.com/research/us-affirmed"


class StageClient:
    """Serves one targeted-search turn and one relation review per event."""
    model = "injected-stage-model"

    def __init__(self, *, fail_search=False):
        self.request_ledger = []
        self.fail_search = fail_search

    def request(self, messages, tools, system, server_tools=None, output_profile=None):
        if self.fail_search:
            raise AgentError("injected search failure")
        self.request_ledger.append({"web_search_requests_reported": 1})
        return {"stop_reason": "tool_use", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "US rating"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1",
             "content": [{"type": "web_search_result", "url": URL, "title": "Fitch affirms US"}]},
            {"type": "tool_use", "id": "t1", "name": "submit_event_evidence_digest", "input": {
                "event_id": EVENT["id"], "findings": [{
                    "source_url": URL, "finding": "Fitch affirmed the US rating with a negative outlook.",
                    "dated": "2026-10-29", "evidence_type": "OFFICIAL_STATEMENT", "direction": "TOWARD_NO"}],
                "current_state_summary": "Affirmed.", "evidence_gap": ""}}]}

    def structured(self, name, schema, payload, system=""):
        self.request_ledger.append({"tool": name})
        return {"event_id": EVENT["id"], "reviews": [{
            "contract_id": c["contract_id"], "venue": c["venue"], "relationship": "EQUIVALENT",
            "rule_evidence": ["resolves to Yes"], "difference": "Same agencies, action and deadline."}
            for c in payload["contracts"]]}


def test_stage_collects_evidence_reviews_details_and_tagged_ledger(tmp_path):
    context = probability_question_context(CASE, EVENT)
    discovery = [{"event_id": EVENT["id"], "description": "Any of the three agencies lowers the US long-term rating.",
                  "search_queries": ["US credit rating downgrade"]}]
    result = run_point2_evidence_stage(
        [context], run_dir=tmp_path, anthropic_key="unused", model=None,
        as_of=datetime(2026, 11, 2, tzinfo=timezone.utc), discovery_events=discovery, default_events=[],
        client=StageClient(), kalshi_client=object(), gamma_client=FakeGamma([]))
    # An injected plain object is not a usable Kalshi client: the venue read fails
    # explicitly inside the search ledger and the stage continues without raising.
    assert result["stage"]["status"] == "COMPLETED"
    kalshi_rows = [r for r in result["market_reviews"][EVENT["id"]]["venue_ledger"] if r["venue"] == "Kalshi"]
    assert kalshi_rows and all(r["status"] == "FAILED" for r in kalshi_rows)
    assert result["definition_details"][EVENT["id"]].startswith("Any of the three agencies")
    assert result["evidence"][EVENT["id"]]["findings"][0]["source_url"] == URL
    assert result["public_evidence"][EVENT["id"]]["findings"][0]["url_verified_in_search_results"] is True
    assert result["market_reviews"][EVENT["id"]]["queries"][0] == "US credit rating downgrade"
    assert all(entry["stage"] == "POINT2_EVIDENCE" for entry in result["ledger"])
    assert result["stage"]["targeted_research_events"] == 1
    assert result["stage"]["probability_produced_here"] is False


def test_stage_search_failure_is_explicit_and_market_review_still_runs(tmp_path, monkeypatch):
    context = probability_question_context(CASE, EVENT)
    catalog = FakeCatalog([_kalshi_market()])
    monkeypatch.setattr("castle_risk.point2_evidence_stage.BoundedKalshiCatalog", lambda *a, **k: catalog)
    monkeypatch.setattr("castle_risk.point2_evidence_stage.ValidatedKalshiClient", lambda *a, **k: object())
    result = run_point2_evidence_stage(
        [context], run_dir=tmp_path, anthropic_key="unused", model=None,
        as_of=datetime(2026, 11, 2, tzinfo=timezone.utc), discovery_events=[], default_events=[],
        client=StageClient(fail_search=True), gamma_client=FakeGamma([]))
    assert result["evidence"][EVENT["id"]]["status"] == "TARGETED_SEARCH_FAILED_NO_SOURCES"
    assert result["evidence"][EVENT["id"]]["leads"] == []
    assert result["market_reviews"][EVENT["id"]]["counts"]["exact"] == 1
    assert result["stage"]["status"] == "COMPLETED_WITH_FAILURES"
    assert result["stage"]["failures"][0] == {"scope": "TARGETED_RESEARCH", "event_id": EVENT["id"],
                                              "reason": "TARGETED_SEARCH_FAILED_NO_SOURCES"}


def test_helpers_read_details_queries_and_family_search_seeds():
    discovery = [{"event_id": "E", "question_we_care_about": "  Precise   question. ", "search_queries": ["a", 3, "b"]}]
    assert _definition_detail("E", discovery, []) == "Precise question."
    assert _definition_detail("E", [], [{"event_id": "E", "original_question": "from default"}]) == "from default"
    assert _definition_detail("X", discovery, []) is None
    assert _proposed_queries("E", discovery) == ["a", "b"]
    geo = {"targets": [{"event_id": "E", "candidates": [{
        "venue": "Polymarket", "contract_id": "55", "market_question": "Will China blockade Taiwan in 2026?",
        "rules_text": "Resolves YES if ...", "rules_sha256": "x", "contract_end_time": "2027-01-01T00:00:00Z",
        "market_open_at_evaluation": True,
        "contract_quote_reference": {"interval": [.02, .03], "scope": "GAMMA", "reasons": []},
        "source_url": "https://polymarket.com/event/china-blockade",
        "source": {"retrieved_at": "2026-11-02T10:00:00Z", "source_response_sha256": "y"}}]}]}
    seeds = _seed_candidates("E", geo)
    assert seeds[0]["contract_id"] == "55" and seeds[0]["quote_interval"] == [.02, .03]
    assert _seed_candidates("OTHER", geo) == []
