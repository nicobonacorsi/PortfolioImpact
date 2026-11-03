"""Event-identified semantic source links across probability and impact stages.

All evidence and model responses are injected. These are source-selection
regressions, not empirical forecasts or independent page fact checks.
"""
from __future__ import annotations

from copy import deepcopy

from castle_risk.autonomous_common_endpoint_impact import forecast_common_endpoint_impacts
from castle_risk.autonomous_event_impact import forecast_portfolio_impacts
from castle_risk.event_evidence_research import (
    DIGEST_SCHEMA, public_event_evidence, research_event_evidence,
)
from castle_risk.live_web_research import WEB_RESEARCH_STATUS
from castle_risk.strict_schema import transport_schema
from castle_risk.structural_event_forecast import (
    _event_source_ids, _identity_matched_event_evidence, _leads,
    impact_event_source_bundle, forecast_events,
)
from test_r116_autonomous_event_impact import POSITIONS, _context
from test_event_evidence_integration import CaptureAbstain, mark

TARGETED_URL = "https://www.imo.org/en/MediaCentre/PressBriefings/Pages/security-routing.aspx"
BROAD_URL = "https://www.lloydslist.com/safety/war-risk-coverage-guidance"
DISTRACTOR_URL = "https://www.reuters.com/world/middle-east/hormuz-tanker-outlook"
UNRETURNED_URL = "https://www.example.org/unreturned-source"
REASON_DIRECT = "Security routing changes determine whether commercial ships can transit the regional passage."
REASON_CHANNEL = "Changes to maritime war-risk cover affect whether shipowners can continue the planned transit."
REASON_REJECT = "The result covers a previous period and provides no evidence about the event's specified deadline."


def _broad_row(url, title, query):
    return {"url": url, "title": title, "search_query": query,
            "evidence_scope": WEB_RESEARCH_STATUS}


def _broad():
    return {"status": "SEARCHED_SOURCE_LEADS_NOT_VERIFIED", "sources": [
        _broad_row(BROAD_URL, "Cargo war-risk underwriting notices", "insurance cover for ship operators"),
        _broad_row(DISTRACTOR_URL, "Hormuz tanker traffic projections", "Hormuz tanker transit"),
    ]}


def _link(context, url, relation, reason, origin="EVENT_TARGETED_SEARCH"):
    return {
        "event_id": context["event_id"], "source_url": url,
        "relation_to_event": relation, "relevance_reason_untrusted": reason,
        "source_origin": origin,
        "url_verified_in_retrieved_search_records": True,
        "content_independently_verified": False,
        "relevance_assessment_scope": "MODEL_ASSESSED_SEMANTIC_LINK_NOT_FACT_VERIFICATION",
    }


def _reviewed(context):
    return {
        "schema": "castle.event_targeted_evidence_research.v1",
        "event_id": context["event_id"], "target_condition": context["target_condition"],
        "deadline": context["deadline"], "as_of_utc": context["as_of_utc"],
        "status": "TARGETED_LEADS_WITH_URL_VERIFIED_DIGEST",
        "source_relevance_status": "REVIEWED_IN_EVENT_DIGEST",
        "leads": [{"url": TARGETED_URL, "title": "Regional shipping security routing update",
                   "search_query": "freighter security routing", "lead_origin": "EVENT_TARGETED_SEARCH"}],
        "portfolio_context_leads": [
            {**_broad()["sources"][0], "lead_origin": "PORTFOLIO_LEVEL_SEARCH"},
            {**_broad()["sources"][1], "lead_origin": "PORTFOLIO_LEVEL_SEARCH"},
        ],
        "source_relevance_links": [
            _link(context, TARGETED_URL, "DIRECT_EVENT_EVIDENCE", REASON_DIRECT),
            _link(context, BROAD_URL, "ECONOMIC_OR_INSTITUTIONAL_CHANNEL", REASON_CHANNEL,
                  origin="PORTFOLIO_LEVEL_SEARCH"),
            _link(context, DISTRACTOR_URL, "NOT_RELEVANT", REASON_REJECT,
                  origin="PORTFOLIO_LEVEL_SEARCH"),
        ],
        "findings": [{
            "source_url": TARGETED_URL, "finding": "A cited model summary of a maritime routing notice.",
            "dated": "2026-11-02", "evidence_type": "OFFICIAL_STATEMENT",
            "direction": "CONTEXT", "url_verified_in_search_results": True,
        }],
    }


def test_transport_accepts_strict_semantic_review_schema():
    transport_schema(DIGEST_SCHEMA)
    assert "source_relevance_links" in DIGEST_SCHEMA["required"]


def test_unrelated_lexical_match_is_rejected_and_semantic_nonmatching_sources_admitted():
    context = _context()
    broad = _leads(_broad())
    row = _reviewed(context)
    evidence = {context["event_id"]: row}
    assert not _event_source_ids(context, broad, evidence_row=row) == {
        lead["id"] for lead in broad
    }
    sources, findings = impact_event_source_bundle(context, broad, evidence)
    assert {lead["url"] for lead in sources.values()} == {BROAD_URL, TARGETED_URL}
    assert DISTRACTOR_URL not in {lead["url"] for lead in sources.values()}
    assert len(findings) == 1 and findings[0]["source_url"] == TARGETED_URL
    assert all(lead["event_relevance_review"]["event_id"] == context["event_id"]
               for lead in sources.values())
    assert any(lead["event_relevance_review"]["relation_to_event"] ==
               "ECONOMIC_OR_INSTITUTIONAL_CHANNEL" for lead in sources.values())


def test_both_impact_engines_use_the_same_semantic_selection_without_extra_research():
    context = _context()
    evidence = {context["event_id"]: _reviewed(context)}
    for forecast, kwargs in (
        (forecast_portfolio_impacts, {}),
        (forecast_common_endpoint_impacts, {"observed_mark_context": mark(context)}),
    ):
        client = CaptureAbstain()
        result = forecast([context], POSITIONS, 100000, _broad(), client,
                          event_evidence=evidence, **kwargs)[context["event_id"]]
        assert result["status"] == "ABSTAIN"
        assert len(client.calls) == 1
        _, payload, _ = client.calls[0]
        assert {lead["url"] for lead in payload["source_leads_unverified"]} == {
            TARGETED_URL, BROAD_URL,
        }
        assert all(lead["event_relevance_review"]["source_url"] == lead["url"]
                   for lead in payload["source_leads_unverified"])


def test_new_semantic_review_blocks_keyword_only_sources_and_citation_conflicts():
    context = _context()
    row = _reviewed(context)
    row["source_relevance_links"] = [
        _link(context, DISTRACTOR_URL, "NOT_RELEVANT", REASON_REJECT,
              origin="PORTFOLIO_LEVEL_SEARCH"),
    ]
    row["findings"] = []
    sources, digest = impact_event_source_bundle(context, _leads(_broad()), {context["event_id"]: row})
    assert sources == {} and digest == []  # "Hormuz tanker" in title is not sufficient.

    # A negative reviewed relation vetoes a contradictory digest citation.
    row["leads"].append({"url": DISTRACTOR_URL, "title": "Hormuz tanker traffic",
                         "search_query": "Hormuz", "lead_origin": "EVENT_TARGETED_SEARCH"})
    row["source_relevance_links"][0]["source_origin"] = "EVENT_TARGETED_SEARCH"
    row["findings"].append({"source_url": DISTRACTOR_URL, "finding": "Conflicting synthetic finding.",
                            "dated": "", "evidence_type": "OTHER", "direction": "CONTEXT",
                            "url_verified_in_search_results": True})
    sources, digest = impact_event_source_bundle(context, [], {context["event_id"]: row})
    assert sources == {} and digest == []


def test_exact_identity_and_returned_url_provenance_prevent_cross_event_and_fabrication():
    context = _context()
    row = _reviewed(context)
    evidence = {context["event_id"]: row}
    assert _identity_matched_event_evidence(context, evidence) is row
    for key in ("target_condition", "deadline", "as_of_utc", "event_id"):
        altered = deepcopy(row)
        altered[key] = "different"
        sources, digest = impact_event_source_bundle(context, [], {context["event_id"]: altered})
        assert sources == {} and digest == [], key

    row = _reviewed(context)
    row["source_relevance_links"][0]["event_id"] = "OTHER_EVENT"
    row["source_relevance_links"].append(
        _link(context, UNRETURNED_URL, "DIRECT_EVENT_EVIDENCE", REASON_DIRECT))
    row["findings"] = []
    sources, digest = impact_event_source_bundle(context, [], {context["event_id"]: row})
    assert sources == {} and digest == []


def test_duplicate_url_keeps_stable_broad_id_and_the_digest_citation():
    context = _context()
    row = _reviewed(context)
    row["portfolio_context_leads"].append(
        {**_broad_row(TARGETED_URL, "Commercial ship security", "routing update"),
         "lead_origin": "PORTFOLIO_LEVEL_SEARCH"})
    broad = [{"id": "W01", "url": TARGETED_URL,
              "title": "Commercial ship security", "search_query": "routing update",
              "lead_origin": "PORTFOLIO_LEVEL_SEARCH"}]
    sources, digest = impact_event_source_bundle(context, broad, {context["event_id"]: row})
    assert list(sources) == ["W01"]
    assert digest[0]["source_id"] == "W01"
    assert sources["W01"]["event_relevance_review"]["source_url"] == TARGETED_URL


class OneSearchClient:
    model = "injected-relevance-review-model"

    def __init__(self, context):
        self.context, self.calls, self.request_ledger = context, [], []

    def request(self, messages, tools, system, server_tools=None, output_profile=None):
        self.calls.append((deepcopy(messages), deepcopy(tools), system))
        self.request_ledger.append({"web_search_requests_reported": 1})
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search",
                 "input": {"query": "Regional freighter security routing"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{
                    "type": "web_search_result", "url": TARGETED_URL,
                    "title": "Regional shipping security routing update",
                }]},
                {"type": "tool_use", "id": "d1", "name": "submit_event_evidence_digest",
                 "input": {
                     "event_id": self.context["event_id"],
                     "findings": [{
                         "source_url": TARGETED_URL,
                         "finding": "Security routing bulletin discusses operating restrictions for cargo vessels.",
                         "dated": "2026-11-02", "evidence_type": "OFFICIAL_STATEMENT",
                         "direction": "CONTEXT",
                     }],
                     "current_state_summary": "Operational routing bulletin found.",
                     "evidence_gap": "",
                     "source_relevance_links": [
                         {"event_id": self.context["event_id"], "source_url": TARGETED_URL,
                          "relation_to_event": "DIRECT_EVENT_EVIDENCE",
                          "relevance_reason": REASON_DIRECT},
                         {"event_id": self.context["event_id"], "source_url": BROAD_URL,
                          "relation_to_event": "ECONOMIC_OR_INSTITUTIONAL_CHANNEL",
                          "relevance_reason": REASON_CHANNEL},
                         {"event_id": self.context["event_id"], "source_url": UNRETURNED_URL,
                          "relation_to_event": "DIRECT_EVENT_EVIDENCE",
                          "relevance_reason": REASON_DIRECT},
                     ],
                 }},
            ],
        }


def test_existing_research_turn_emits_url_verified_semantic_links_without_extra_request():
    context = _context()
    client = OneSearchClient(context)
    raw = research_event_evidence(client, context, portfolio_search=_broad())
    assert len(client.calls) == 1
    assert raw["source_relevance_status"] == "REVIEWED_IN_EVENT_DIGEST"
    assert {link["source_url"] for link in raw["source_relevance_links"]} == {
        TARGETED_URL, BROAD_URL,
    }
    public = public_event_evidence(raw)
    assert public is not None
    assert {link["source_url"] for link in public["source_relevance_links"]} == {
        TARGETED_URL, BROAD_URL,
    }
    assert all(link["url_verified_in_retrieved_search_records"] is True
               and link["content_independently_verified"] is False
               for link in public["source_relevance_links"])
    assert all(link["event_id"] == context["event_id"] for link in public["source_relevance_links"])
    sources, digest = impact_event_source_bundle(
        context, _leads(_broad()), {context["event_id"]: public})
    assert {lead["url"] for lead in sources.values()} == {TARGETED_URL, BROAD_URL}
    assert digest and digest[0]["source_id"] in sources


def test_probability_forecaster_uses_same_reviewed_source_ids():
    context = _context()
    row = _reviewed(context)

    class ModelAbstain:
        model = "injected-forecast"

        def __init__(self):
            self.payloads = []

        def structured(self, name, schema, payload, system=""):
            self.payloads.append(deepcopy(payload))
            return {"forecasts": [{"event_id": context["event_id"], "status": "ABSTAIN",
                                   "abstain_reason": "Synthetic deliberate abstention.",
                                   "scenarios": []}]}

    model = ModelAbstain()
    forecast_events([context], _broad(), model,
                    event_evidence={context["event_id"]: row})
    assert len(model.payloads) == 1
    payload = model.payloads[0]
    assert {lead["url"] for lead in payload["unverified_public_source_leads"]} == {
        TARGETED_URL, BROAD_URL,
    }
    assert DISTRACTOR_URL not in {lead["url"] for lead in payload["unverified_public_source_leads"]}


def test_archived_v1_rows_remain_readable_without_fake_semantic_verification():
    context = _context()
    row = _reviewed(context)
    row.pop("source_relevance_status")
    row.pop("source_relevance_links")
    broad = _leads(_broad())
    assert _event_source_ids(context, broad, evidence_row=row)
    sources, _ = impact_event_source_bundle(context, broad, {context["event_id"]: row})
    assert "event_relevance_review" not in next(iter(sources.values()))
