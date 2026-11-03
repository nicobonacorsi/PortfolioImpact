"""R125 event-targeted evidence research: URL-verified digests, no fabricated sources.

All transports are injected; no network, no credentials, no probabilities.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.agent import AgentError
from castle_risk.event_evidence_research import (
    DIGEST_SCHEMA, DIGEST_TOOL, LEAD_ORIGIN, attach_event_evidence, public_event_evidence,
    research_event_evidence,
)
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.strict_schema import transport_schema


EVENT = {"id": "US_SOVEREIGN_DOWNGRADE_2027_01_31", "date": "2027-01-31",
         "title": "Moody's, S&P or Fitch lowers the long-term US sovereign credit rating by 2027-01-31"}
CASE = {"capture": "2026-11-02T10:00:00+00:00"}
URL_A = "https://www.fitchratings.com/research/sovereigns/us-rating-affirmed"
URL_B = "https://www.reuters.com/markets/us/treasury-deficit-update"


def _context():
    context = probability_question_context(CASE, EVENT)
    assert context is not None
    return context


def _search_blocks(urls, query="US sovereign rating outlook"):
    return [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": query}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
         "content": [{"type": "web_search_result", "url": u, "title": "Result " + str(i),
                      "page_age": "November 1, 2026"} for i, u in enumerate(urls)]},
    ]


def _digest(findings, event_id=EVENT["id"]):
    return {"type": "tool_use", "id": "toolu_1", "name": DIGEST_TOOL,
            "input": {"event_id": event_id, "findings": findings,
                      "current_state_summary": "Ratings affirmed with negative outlook.",
                      "evidence_gap": "No agency has announced a review."}}


class FakeClient:
    model = "injected-test-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.request_ledger = []

    def request(self, messages, tools, system, server_tools=None, output_profile=None):
        self.calls.append({"messages": deepcopy(messages), "tools": tools, "server_tools": server_tools,
                           "output_profile": output_profile})
        assert tools and tools[0]["name"] == DIGEST_TOOL and tools[0]["strict"] is True
        assert server_tools and server_tools[0]["name"] == "web_search"
        assert output_profile == {"effort": "high", "max_tokens": 12_288}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        self.request_ledger.append({"web_search_requests_reported": reply.get("_web", 0)})
        return reply


def test_digest_schema_is_transport_compatible():
    transport_schema(DIGEST_SCHEMA)


def test_findings_must_cite_urls_actually_returned_by_the_search_tool():
    finding_ok = {"source_url": URL_A, "finding": "Fitch affirmed the US at AA+ with a negative outlook on 2026-10-29.",
                  "dated": "2026-10-29", "evidence_type": "OFFICIAL_STATEMENT", "direction": "TOWARD_NO"}
    fabricated = {**finding_ok, "source_url": "https://www.moodys.com/not-returned-by-search"}
    reply = {"stop_reason": "tool_use", "_web": 2,
             "content": _search_blocks([URL_A, URL_B]) + [_digest([finding_ok, fabricated])]}
    client = FakeClient([reply])
    row = research_event_evidence(client, _context())
    assert row["status"] == "TARGETED_LEADS_WITH_URL_VERIFIED_DIGEST"
    assert [lead["url"] for lead in row["leads"]] == [URL_A, URL_B]
    assert all(lead["lead_origin"] == LEAD_ORIGIN and lead["event_id"] == EVENT["id"] for lead in row["leads"])
    assert len(row["findings"]) == 1
    assert row["findings"][0]["source_url"] == URL_A
    assert row["findings"][0]["url_verified_in_search_results"] is True
    assert row["findings"][0]["content_independently_verified"] is False
    assert row["web_search_requests_reported"] == 2
    assert row["probability_produced"] is False
    payload = client.calls[0]["messages"][0]["content"]
    assert EVENT["title"] in payload and "no probability" in payload


def test_missing_digest_gets_one_nudge_without_new_search_budget_and_keeps_leads():
    first = {"stop_reason": "end_turn", "_web": 1, "content": _search_blocks([URL_A])}
    second = {"stop_reason": "tool_use", "_web": 0, "content": [_digest([
        {"source_url": URL_A, "finding": "The agency affirmed the rating and kept the outlook negative.",
         "dated": "", "evidence_type": "OFFICIAL_STATEMENT", "direction": "CONTEXT"}])]}
    client = FakeClient([first, second])
    row = research_event_evidence(client, _context())
    assert len(client.calls) == 2
    assert client.calls[1]["server_tools"][0]["max_uses"] == 1
    assert row["leads"][0]["url"] == URL_A
    assert len(row["findings"]) == 1
    assert row["model_requests"] == 2


def test_transport_failure_after_leads_is_partial_not_fabricated():
    first = {"stop_reason": "pause_turn", "_web": 1, "content": _search_blocks([URL_A])}
    client = FakeClient([first, AgentError("injected timeout")])
    row = research_event_evidence(client, _context())
    assert row["status"] == "PARTIAL_SEARCH_ERROR"
    assert row["leads"] and row["findings"] == []
    failed = research_event_evidence(FakeClient([AgentError("down")]), _context())
    assert failed["status"] == "TARGETED_SEARCH_FAILED_NO_SOURCES"
    assert failed["leads"] == [] and failed["findings"] == []


def test_digest_for_a_different_event_or_bad_schema_is_discarded():
    wrong_event = {"stop_reason": "tool_use", "_web": 1, "content": _search_blocks([URL_A]) + [_digest([
        {"source_url": URL_A, "finding": "Affirmed rating with negative outlook, per agency release.",
         "dated": "", "evidence_type": "OFFICIAL_STATEMENT", "direction": "CONTEXT"}], event_id="OTHER")]}
    row = research_event_evidence(FakeClient([wrong_event]), _context())
    assert row["digest_status"] == "DIGEST_FOR_DIFFERENT_EVENT"
    assert row["findings"] == [] and row["leads"]
    bad = {"stop_reason": "tool_use", "_web": 1, "content": _search_blocks([URL_A]) + [_digest([
        {"source_url": URL_A, "finding": "short", "dated": "", "evidence_type": "OFFICIAL_STATEMENT",
         "direction": "CONTEXT"}])]}
    row = research_event_evidence(FakeClient([bad]), _context())
    assert row["digest_status"] == "DIGEST_SCHEMA_REJECTED"


def test_private_or_tokenized_result_urls_are_never_retained():
    reply = {"stop_reason": "tool_use", "_web": 1, "content": _search_blocks([
        "https://127.0.0.1/internal", "http://www.reuters.com/plain", URL_B + "?token=abc"]) + [_digest([])]}
    row = research_event_evidence(FakeClient([reply]), _context())
    assert [lead["url"] for lead in row["leads"]] == [URL_B]


def test_public_view_and_attachment_require_identity_and_safe_fields():
    reply = {"stop_reason": "tool_use", "_web": 1, "content": _search_blocks([URL_A]) + [_digest([
        {"source_url": URL_A, "finding": "Rating affirmed; outlook negative, according to the agency page.",
         "dated": "2026-10-29", "evidence_type": "OFFICIAL_STATEMENT", "direction": "TOWARD_NO"}])]}
    row = research_event_evidence(FakeClient([reply]), _context())
    public = public_event_evidence(row)
    assert public["findings"][0]["url_verified_in_search_results"] is True
    assert "leads" in public and public["probability_produced"] is False
    case = {"risks": [{"id": EVENT["id"], "title": EVENT["title"], "date": EVENT["date"],
                       "event_probability_question": _context()}]}
    attach_event_evidence(case, {EVENT["id"]: row})
    assert case["risks"][0]["event_evidence_research"]["event_id"] == EVENT["id"]
    stale = deepcopy(row)
    stale["deadline"] = "2027-02-28"
    case = {"risks": [{"id": EVENT["id"], "event_probability_question": _context()}]}
    attach_event_evidence(case, {EVENT["id"]: stale})
    assert "event_evidence_research" not in case["risks"][0]
    with pytest.raises(ValueError):
        research_event_evidence(FakeClient([]), {"event_id": None})
