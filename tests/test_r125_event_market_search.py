"""R125 generic two-venue contract search and relation review; offline fixtures only.

Nothing here certifies that any real contract is equivalent to any event.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from castle_risk.adapters import Market
from castle_risk.agent import AgentError
from castle_risk.event_market_search import (
    REVIEW_SCHEMA, apply_deadline_check, attach_event_market_reviews, content_tokens,
    event_market_queries, gamma_candidate, kalshi_candidate, market_implied_bounds,
    rank_candidates, review_contract_relations, review_event_markets,
)
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.strict_schema import transport_schema


CASE = {"capture": "2026-11-02T10:00:00+00:00"}
EVENT = {"id": "US_SOVEREIGN_DOWNGRADE_2026_12_31", "date": "2026-12-31",
         "title": "Moody's, S&P or Fitch lowers the long-term US sovereign credit rating by 2026-12-31"}
NOW = "2026-11-02T10:05:00+00:00"
RULES = ("If U.S. credit is downgraded by any of the three major credit ratings agencies by "
         "December 31, 2026, then the market resolves to Yes.")


def _context(event=EVENT):
    context = probability_question_context(CASE, event)
    assert context is not None
    return context


def _kalshi_market(ticker="KXCREDITRATING-26DEC31", close="2027-01-01T15:00:00Z", bid=.012, ask=.063):
    return Market(ticker=ticker, title="Will the U.S. credit rating be downgraded by December 31, 2026?",
                  rules_primary=RULES, event_ticker="KXCREDITRATING-26DEC31", series_ticker="KXCREDITRATING",
                  close_time=close, source_url="https://external-api.kalshi.com/trade-api/v2/markets/" + ticker,
                  retrieved_at=NOW, yes_bid=bid, yes_ask=ask, quote_timestamp=NOW, status="active",
                  metadata={"raw": {"ticker": ticker}})


class FakeCatalog:
    def __init__(self, markets, fail=False):
        self.markets, self.fail = markets, fail
        self.client = type("C", (), {"request_count": 0})()
        self.queries = []

    def search(self, query, *, limit=8):
        self.queries.append(query)
        self.client.request_count += 1
        if self.fail:
            from castle_risk.live_catalog import CatalogSearchError
            raise CatalogSearchError("injected")
        return list(self.markets)[:limit]


class FakeGamma:
    request_count = 0

    def __init__(self, events):
        self.events = events

    def _get(self, url):
        self.request_count += 1
        return ({"events": self.events, "pagination": {"hasMore": False}},
                {"retrieved_at": NOW, "sha256": "ab" * 32})


class FakeReviewer:
    model = "injected-review-model"

    def __init__(self, reviews, *, event_id=EVENT["id"], fail=False):
        self.reviews, self.event_id, self.fail = reviews, event_id, fail
        self.payloads = []

    def structured(self, name, schema, payload, system=""):
        assert name == "submit_contract_relations"
        assert "exact substrings" in system.lower() or "EXACT substrings" in system
        self.payloads.append(deepcopy(payload))
        if self.fail:
            raise AgentError("injected")
        return {"event_id": self.event_id, "reviews": deepcopy(self.reviews)}


def test_review_schema_is_transport_compatible_and_queries_are_bounded():
    transport_schema(REVIEW_SCHEMA)
    queries = event_market_queries(_context(), ["US credit rating downgrade", "x" * 200, 7],
                                   "Any of Moody's, S&P Global or Fitch Ratings lowers the US rating.")
    assert queries[0] == "US credit rating downgrade"
    assert all(2 <= len(q) <= 120 for q in queries) and len(queries) <= 4
    assert len({q.lower() for q in queries}) == len(queries)


def test_candidates_normalize_quotes_as_market_q_and_prefilter_by_subject():
    context = _context()
    kalshi = kalshi_candidate(_kalshi_market(), evaluated_at=NOW, query="credit rating")
    assert kalshi["quote_interval"] == pytest.approx([.012, .063])
    assert kalshi["market_open_at_evaluation"] is True
    closed = kalshi_candidate(_kalshi_market(close="2026-10-01T15:00:00Z"), evaluated_at=NOW, query="q")
    assert closed["quote_interval"] is None and closed["market_open_at_evaluation"] is False
    gamma = gamma_candidate({"id": "77", "question": "Will the US be downgraded in 2026?",
                             "description": "Resolves YES if any agency downgrades the US sovereign rating in 2026.",
                             "endDate": "2027-01-01T12:00:00Z", "active": True, "closed": False, "archived": False,
                             "acceptingOrders": True, "outcomes": '["Yes", "No"]', "bestBid": .02, "bestAsk": .05},
                            {"slug": "us-downgrade-2026"}, retrieved_at=NOW, response_sha="cd" * 32,
                            evaluated_at=NOW, query="downgrade")
    assert gamma["quote_interval"] == [.02, .05]
    assert gamma["source_url"] == "https://polymarket.com/event/us-downgrade-2026"
    unrelated = gamma_candidate({"id": "78", "question": "Will the Lakers win the title?",
                                 "description": "Resolves YES if the Los Angeles Lakers win the 2027 NBA championship.",
                                 "endDate": "2027-06-30T00:00:00Z", "active": True, "closed": False,
                                 "acceptingOrders": True, "outcomes": '["Yes", "No"]', "bestBid": .1, "bestAsk": .2},
                                None, retrieved_at=NOW, response_sha=None, evaluated_at=NOW, query="downgrade")
    review, excluded = rank_candidates(context, [kalshi, gamma, unrelated])
    assert {row["contract_id"] for row in review} == {"KXCREDITRATING-26DEC31", "77"}
    assert excluded[0]["contract_id"] == "78"
    assert excluded[0]["decision"] == "REJECTED_INCOMPATIBLE"
    assert excluded[0]["rejection_basis"] == "NO_SHARED_SUBJECT_TERM_PREFILTER"
    assert content_tokens("downgraded ratings agencies") == {"downgrad", "rating", "agenci"}


@pytest.mark.parametrize("relationship,close,expected,note_part", [
    ("EQUIVALENT", "2027-01-01T15:00:00Z", "EQUIVALENT", "WITHIN_ONE_DAY"),
    ("EQUIVALENT", "2026-12-01T15:00:00Z", "CONTRACT_YES_IMPLIES_EVENT", "EARLIER"),
    ("EQUIVALENT", "2027-03-01T15:00:00Z", "EVENT_IMPLIES_CONTRACT_YES", "LATER"),
    ("CONTRACT_YES_IMPLIES_EVENT", "2027-03-01T15:00:00Z", "RELATED_NO_ENTAILMENT", "EXPIRES_AFTER"),
    ("EVENT_IMPLIES_CONTRACT_YES", "2026-12-01T15:00:00Z", "RELATED_NO_ENTAILMENT", "EXPIRES_BEFORE"),
    ("RELATED_NO_ENTAILMENT", "not-a-date", "RELATED_NO_ENTAILMENT", "DATES_NOT_COMPARED"),
    ("EQUIVALENT", "not-a-date", "RELATED_NO_ENTAILMENT", "UNPARSEABLE"),
])
def test_code_enforces_deadline_arithmetic_on_model_relations(relationship, close, expected, note_part):
    final, note = apply_deadline_check(relationship, close, "2026-12-31")
    assert final == expected
    assert note_part in note


def test_review_verifies_rule_substrings_and_downgrades_unsupported_entailments():
    context = _context()
    candidates = [kalshi_candidate(_kalshi_market(), evaluated_at=NOW, query="credit rating")]
    good = {"contract_id": "KXCREDITRATING-26DEC31", "venue": "Kalshi", "relationship": "EQUIVALENT",
            "rule_evidence": ["downgraded by any of the three major credit ratings agencies by December 31, 2026"],
            "difference": "Same actors (three major agencies), same action, same deadline."}
    cards, status = review_contract_relations(FakeReviewer([good]), context, candidates)
    assert status == "REVIEWED"
    assert cards[0]["decision"] == "EXACT_MATCH_MODEL_REVIEWED"
    assert cards[0]["relationship"] == "EQUIVALENT"
    assert cards[0]["quote_is_market_Q_not_event_P"] is True
    assert "rules_text" not in cards[0]
    fabricated = {**good, "rule_evidence": ["the Treasury Secretary announces a default"]}
    cards, _ = review_contract_relations(FakeReviewer([fabricated]), context, candidates)
    assert cards[0]["relationship"] == "RELATED_NO_ENTAILMENT"
    assert cards[0]["decision"] == "RELATED_ONLY"
    assert "EVIDENCE_NOT_EXACT_SUBSTRING_OF_RULES" in cards[0]["review_flags"]
    assert "ENTAILMENT_WITHOUT_VERIFIED_RULE_EVIDENCE_DOWNGRADED" in cards[0]["review_flags"]
    partly_fabricated = {**good, "rule_evidence": [good["rule_evidence"][0],
                                                   "the Treasury Secretary announces a default"]}
    cards, _ = review_contract_relations(FakeReviewer([partly_fabricated]), context, candidates)
    assert cards[0]["relationship"] == "RELATED_NO_ENTAILMENT"
    assert cards[0]["decision"] == "RELATED_ONLY"
    assert "EVIDENCE_NOT_EXACT_SUBSTRING_OF_RULES" in cards[0]["review_flags"]
    assert "ENTAILMENT_WITH_INVALID_RULE_EVIDENCE_DOWNGRADED" in cards[0]["review_flags"]
    unrelated = {**good, "relationship": "UNRELATED", "rule_evidence": []}
    cards, _ = review_contract_relations(FakeReviewer([unrelated]), context, candidates)
    assert cards[0]["decision"] == "REJECTED_INCOMPATIBLE"


def test_review_batch_with_missing_duplicate_or_unknown_contracts_fails_closed():
    context = _context()
    candidates = [kalshi_candidate(_kalshi_market(), evaluated_at=NOW, query="q"),
                  kalshi_candidate(_kalshi_market(ticker="KXOTHER-26DEC31"), evaluated_at=NOW, query="q")]
    one = {"contract_id": "KXCREDITRATING-26DEC31", "venue": "Kalshi", "relationship": "EQUIVALENT",
           "rule_evidence": ["resolves to Yes"], "difference": "Identical settlement predicate."}
    cards, status = review_contract_relations(FakeReviewer([one]), context, candidates)
    assert status == "REVIEW_BATCH_INVALID" and all(c["decision"] == "NOT_REVIEWED" for c in cards)
    cards, status = review_contract_relations(FakeReviewer([one, one, {**one, "contract_id": "KXOTHER-26DEC31"}]),
                                              context, candidates)
    assert status == "REVIEW_BATCH_INVALID"
    cards, status = review_contract_relations(FakeReviewer([one], event_id="WRONG"), context, candidates)
    assert status == "REVIEW_BATCH_INVALID"
    cards, status = review_contract_relations(FakeReviewer([one], fail=True), context, candidates)
    assert status == "REVIEW_REQUEST_FAILED" and cards[0]["rejection_basis"] == "REVIEW_REQUEST_FAILED"


def test_market_implied_bounds_follow_entailment_direction_and_never_set_p():
    cards = [
        {"venue": "Kalshi", "contract_id": "A", "relationship": "CONTRACT_YES_IMPLIES_EVENT", "quote_interval": [.02, .03]},
        {"venue": "Polymarket", "contract_id": "B", "relationship": "EVENT_IMPLIES_CONTRACT_YES", "quote_interval": [.40, .45]},
        {"venue": "Kalshi", "contract_id": "C", "relationship": "RELATED_NO_ENTAILMENT", "quote_interval": [.9, .95]},
        {"venue": "Kalshi", "contract_id": "D", "relationship": "EQUIVALENT", "quote_interval": None},
    ]
    bounds = market_implied_bounds(cards)
    assert bounds["lower"] == .02 and bounds["upper"] == .45
    assert bounds["status"] == "QUOTE_IMPLIED_RANGE_UNDER_STATED_ASSUMPTIONS"
    assert bounds["assumptions"] and "not physical probabilities" in bounds["assumptions"][1]
    assert bounds["used_to_set_probability"] is False
    inconsistent = market_implied_bounds([
        {"venue": "Kalshi", "contract_id": "A", "relationship": "CONTRACT_YES_IMPLIES_EVENT", "quote_interval": [.5, .6]},
        {"venue": "Kalshi", "contract_id": "B", "relationship": "EVENT_IMPLIES_CONTRACT_YES", "quote_interval": [.1, .2]}])
    assert inconsistent["status"] == "INCONSISTENT_QUOTES_UNDER_ASSERTED_ENTAILMENTS"
    assert market_implied_bounds([])["status"] == "NO_QUOTED_ENTAILMENT"


def test_end_to_end_review_classifies_and_attaches_with_identity_gate():
    context = _context()
    catalog = FakeCatalog([_kalshi_market()])
    gamma = FakeGamma([{"slug": "us-downgrade-2026", "markets": [{
        "id": "77", "question": "Will the US be downgraded in 2026?",
        "description": "Resolves YES if any agency downgrades the US sovereign rating during 2026.",
        "endDate": "2027-01-01T12:00:00Z", "active": True, "closed": False, "archived": False,
        "acceptingOrders": True, "outcomes": '["Yes", "No"]', "bestBid": .02, "bestAsk": .05}]}])
    reviews = [
        {"contract_id": "KXCREDITRATING-26DEC31", "venue": "Kalshi", "relationship": "EQUIVALENT",
         "rule_evidence": ["downgraded by any of the three major credit ratings agencies by December 31, 2026"],
         "difference": "Same agencies, action and deadline."},
        {"contract_id": "77", "venue": "Polymarket", "relationship": "RELATED_NO_ENTAILMENT",
         "rule_evidence": [], "difference": "Any agency, whole-year window; not the same three-agency predicate."},
    ]
    reviewer = FakeReviewer(reviews)
    result = review_event_markets(reviewer, context, kalshi_catalog=catalog, gamma_client=gamma,
                                  proposed_queries=["US credit rating downgrade"], evaluated_at=NOW)
    assert result["status"] == "EXACT_MODEL_REVIEWED_CONTRACT_FOUND"
    assert result["counts"] == {"exact": 1, "entailment": 0, "related": 1, "rejected": 0, "not_reviewed": 0}
    assert result["nothing_fits_exactly"] is False
    assert result["market_implied_bounds"]["lower"] == pytest.approx(.012)
    assert result["market_implied_bounds"]["upper"] == pytest.approx(.063)
    assert result["quotes_are_market_Q_not_event_P"] is True
    assert reviewer.payloads[0]["event"]["target_condition"] == EVENT["title"]
    assert all("rules_text" in c for c in reviewer.payloads[0]["contracts"])
    case = {"risks": [{"id": EVENT["id"], "title": EVENT["title"], "date": EVENT["date"],
                       "event_probability_question": context}]}
    attach_event_market_reviews(case, {EVENT["id"]: result})
    attached = case["risks"][0]["prediction_market_review"]
    assert attached["decisions"]["exact"][0]["contract_id"] == "KXCREDITRATING-26DEC31"
    assert "rules_text" not in attached["decisions"]["exact"][0]
    changed = deepcopy(result)
    changed["target_condition"] = EVENT["title"] + " (edited)"
    case = {"risks": [{"id": EVENT["id"], "event_probability_question": context}]}
    attach_event_market_reviews(case, {EVENT["id"]: changed})
    assert "prediction_market_review" not in case["risks"][0]


def test_search_failures_are_explicit_and_not_evidence_of_absence():
    context = _context()
    result = review_event_markets(FakeReviewer([]), context, kalshi_catalog=FakeCatalog([], fail=True),
                                  gamma_client=None, evaluated_at=NOW)
    assert result["status"] == "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE"
    assert result["venue_ledger"] and all(row["status"] == "FAILED" for row in result["venue_ledger"])
    assert result["candidates_total"] == 0 and result["nothing_fits_exactly"] is True
