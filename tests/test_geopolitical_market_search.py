"""Source/semantic adversarial checks for bounded two-venue geopolitical search."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from castle_risk.adapters import KalshiClient
from castle_risk.geopolitical_market_search import (
    GammaSearchClient, GeopoliticalSearchError, _relevance,
    acquire_geopolitical_search, public_geopolitical_search_view,
    replay_geopolitical_search,
)


TARGETS = [
    {"event_id": "taiwan-supply-2027-01-19", "title": "Taiwan supply disruption",
     "description": "Chinese blockade, quarantine or armed action involving Taiwan by 2027-01-19",
     "event_date": "2027-01-19"},
    {"event_id": "chip-export-2027-01-19", "title": "US chip export restrictions",
     "description": "New or tightened US chip export restrictions affecting China by 2027-01-19",
     "event_date": "2027-01-19"},
]


@pytest.fixture(autouse=True)
def explicit_capture_clock(monkeypatch):
    # These are injected fixture responses, not live evidence; use an explicit
    # receipt clock so date-sensitive assertions never depend on the test day.
    from castle_risk import adapters
    monkeypatch.setattr(adapters, "utc_now", lambda: "2026-10-21T14:59:00+00:00")


def gamma_market(identifier="1", question="Will China blockade Taiwan in 2026?", end="2027-01-01T04:59:00Z"):
    return {"id": identifier, "question": question, "description": question + " A qualifying blockade prevents foreign commercial ingress or egress. Otherwise No.",
            "active": True, "closed": False, "archived": False,
            "acceptingOrders": True, "enableOrderBook": True,
            "endDate": end, "outcomes": '["Yes", "No"]', "bestBid": .02, "bestAsk": .04}


def kalshi_market():
    return {"ticker": "KXTAIWANLVL4-27JAN01", "event_ticker": "KXTAIWANLVL4",
            "title": "Will the US issue a Level 4 travel advisory for Taiwan before 2027?",
            "rules_primary": "If the US issues a Level 4 travel advisory for Taiwan before January 1, 2027, the market resolves Yes.",
            "rules_secondary": "Otherwise No.", "status": "active", "market_type": "binary",
            "close_time": "2027-01-01T05:00:00Z", "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.03",
            "notional_value_dollars": "1.0000"}


def transport(url, headers, timeout):
    assert timeout == 20
    assert "Authorization" not in headers
    path, params = urlsplit(url).path, parse_qs(urlsplit(url).query)
    if path.endswith("/series"):
        body = {"series": [{"ticker": "KXTAIWANLVL4", "title": "Taiwan Level 4 travel advisory"}]}
    elif path.endswith("/markets"):
        body = {"markets": [kalshi_market()], "cursor": ""}
    else:
        first = gamma_market()
        second = gamma_market("2", "Will China invade Taiwan by June 30, 2027?", "2027-07-01T03:59:00Z")
        third = gamma_market("3", "Will AI-chip export licensing become law this year?")
        body = {"events": [{"id": "10", "slug": "taiwan-events", "markets": [first, second, third]}],
                "pagination": {"hasMore": False, "totalResults": 1}}
    return 200, {"Date": "Wed, 21 Oct 2026 15:00:00 GMT"}, json.dumps(body).encode()


def collect(tmp_path, *, chosen_transport=transport, max_requests=30, targets=None):
    k = KalshiClient(tmp_path / "kalshi", transport=chosen_transport, min_interval=0, max_requests=max_requests)
    g = GammaSearchClient(tmp_path / "polymarket", transport=chosen_transport, min_interval=0, max_requests=max_requests)
    report = acquire_geopolitical_search(TARGETS if targets is None else targets,
        data_dir=tmp_path, kalshi_client=k, gamma_client=g, evaluated_at="2026-10-21T15:00:00+00:00")
    bundle = json.loads((tmp_path / "search_capture.json").read_text())
    return report, bundle, k, g


def alter_record(tmp_path, record, change):
    raw = tmp_path / record["raw_file"]
    receipt = tmp_path / record["receipt_file"]
    document = json.loads(raw.read_text())
    change(document)
    data = json.dumps(document).encode()
    raw.write_bytes(data)
    digest = sha256(data).hexdigest()
    meta = json.loads(receipt.read_text())
    meta["sha256"] = digest
    receipt.write_text(json.dumps(meta))
    record["source_response_sha256"] = digest


def test_both_venues_real_query_routes_and_distinct_questions(tmp_path):
    report, bundle, k, g = collect(tmp_path)
    assert k.request_count > 0 and g.request_count > 0
    assert all(record["fixture"] is True for record in bundle["source_records"])
    taiwan = report["targets"][0]
    assert taiwan["closest_by_venue"]["Kalshi"]["closest_candidate"]["contract_id"] == "KXTAIWANLVL4-27JAN01"
    candidate = taiwan["closest_by_venue"]["Polymarket"]["closest_candidate"]
    assert candidate["contract_id"] == "1"
    assert candidate["decision"] == "RELATED_ONLY"
    assert "quarantine" in candidate["difference"]
    assert candidate["event_pricing_measure_Q_interval"] is None
    assert candidate["exact_source_alternative"]["same_as_parent"] is False
    assert candidate["contract_quote_reference"]["interval"] is None
    assert "FIXTURE_NOT_LIVE_CAPTURE" in candidate["contract_quote_reference"]["reasons"]


def test_never_promotes_chip_legislation_to_all_export_controls(tmp_path):
    report, _, _, _ = collect(tmp_path)
    target = report["targets"][1]
    candidate = target["closest_by_venue"]["Polymarket"]["closest_candidate"]
    assert candidate["contract_id"] == "3"
    assert candidate["decision"] == "RELATED_ONLY"
    assert "narrower" in candidate["difference"]
    assert target["event_pricing_measure_Q_interval"] is None


@pytest.mark.parametrize("text", ["Chipotle headcount", "Which chipmunk wins?", "Boeing revenue"])
def test_chip_substrings_do_not_retrieve_unrelated_company(text):
    assert _relevance(text, "US_CHIP_EXPORT_POLICY") == 0


def test_nearest_contract_does_not_depend_on_price(tmp_path):
    first, bundle, _, _ = collect(tmp_path)
    for record in bundle["source_records"]:
        if record["venue"] == "Polymarket":
            alter_record(tmp_path, record, lambda doc: [row.update(bestBid=.9, bestAsk=.95)
                for event in doc["events"] for row in event["markets"]])
    second = replay_geopolitical_search(bundle)
    for one, two in zip(first["targets"], second["targets"]):
        for venue in ("Kalshi", "Polymarket"):
            a, b = one["closest_by_venue"][venue]["closest_candidate"], two["closest_by_venue"][venue]["closest_candidate"]
            assert (None if a is None else a["contract_id"]) == (None if b is None else b["contract_id"])


def test_zero_results_are_bounded_absence(tmp_path):
    def empty(url, headers, timeout):
        body = {"series": []} if urlsplit(url).path.endswith("/series") else {"pagination": {"hasMore": False, "totalResults": 0}}
        return 200, {}, json.dumps(body).encode()
    report, _, _, _ = collect(tmp_path, chosen_transport=empty)
    for target in report["targets"]:
        for venue in target["closest_by_venue"].values():
            assert venue["decision"] == "NONE_FOUND_WITHIN_SCOPE"
        assert target["no_match_is_not_global_absence"]


def test_failed_search_is_not_no_market_found(tmp_path):
    report, _, k, g = collect(tmp_path, chosen_transport=lambda *_: (403, {}, b"do not log this body"))
    assert k.max_requests == g.max_requests == 30
    for target in report["targets"]:
        assert target["decision"] == "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE"
        for venue in target["closest_by_venue"].values():
            assert venue["status"] == "SEARCH_FAILED"
            assert venue["decision"] == "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE"
    assert "do not log" not in json.dumps(report)


def test_original_client_request_cap_is_preserved(tmp_path):
    report, _, k, g = collect(tmp_path, max_requests=2)
    assert k.request_count <= 2 and g.request_count <= 2
    assert report["existing_request_caps"] == {"Kalshi": 2, "Polymarket": 2}
    assert any(row["status"] == "FAILED" for row in report["search_ledger"])


def test_replay_does_not_touch_network_and_preserves_input(tmp_path, monkeypatch):
    before = deepcopy(TARGETS)
    report, bundle, _, _ = collect(tmp_path)
    from castle_risk import adapters
    monkeypatch.setattr(adapters, "_transport", lambda *_: pytest.fail("Replay touched network"))
    replay = replay_geopolitical_search(bundle)
    assert replay["targets"] == report["targets"]
    assert TARGETS == before
    assert replay["new_http_calls_in_replay"] == 0


def test_body_mutation_rejected(tmp_path):
    _, bundle, _, _ = collect(tmp_path)
    path = tmp_path / bundle["source_records"][0]["raw_file"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(GeopoliticalSearchError, match="hash mismatch"):
        replay_geopolitical_search(bundle)


def test_receipt_mutation_rejected(tmp_path):
    _, bundle, _, _ = collect(tmp_path)
    path = tmp_path / bundle["source_records"][0]["receipt_file"]
    meta = json.loads(path.read_text())
    meta["retrieved_at"] = "2026-10-20T00:00:00Z"
    path.write_text(json.dumps(meta))
    with pytest.raises(GeopoliticalSearchError, match="hash mismatch"):
        replay_geopolitical_search(bundle)


@pytest.mark.parametrize("path", ["../secret.json", "/tmp/secret.json", "..\\secret.json"])
def test_path_traversal_rejected(tmp_path, path):
    _, bundle, _, _ = collect(tmp_path)
    bundle["source_records"][0]["raw_file"] = path
    with pytest.raises(GeopoliticalSearchError, match="path"):
        replay_geopolitical_search(bundle)


def test_stale_or_future_quote_is_not_admitted(tmp_path):
    _, bundle, _, _ = collect(tmp_path)
    report = replay_geopolitical_search(bundle, evaluated_at="2026-10-22T00:00:00Z")
    for target in report["targets"]:
        for card in target["candidates"]:
            assert card["contract_quote_reference"]["interval"] is None
            assert "FUTURE_OR_STALE_RECEIPT" in card["contract_quote_reference"]["reasons"]


def test_conflicting_source_rules_quarantined(tmp_path):
    _, bundle, _, _ = collect(tmp_path)
    record = next(row for row in bundle["source_records"] if row["venue"] == "Polymarket")
    alter_record(tmp_path, record, lambda doc: doc["events"][0]["markets"][0].update(description="Mutated rules"))
    report = replay_geopolitical_search(bundle)
    assert ["Polymarket", "1"] in report["conflicting_rules_contracts"]
    assert all(card["contract_id"] != "1" for target in report["targets"] for card in target["candidates"])


def test_closed_market_stays_rejected(tmp_path):
    def closed(url, headers, timeout):
        status, hdrs, raw = transport(url, headers, timeout)
        doc = json.loads(raw)
        for event in doc.get("events", []):
            for row in event["markets"]:
                row["closed"] = True
        return status, hdrs, json.dumps(doc).encode()
    report, _, _, _ = collect(tmp_path, chosen_transport=closed)
    for target in report["targets"]:
        for card in target["candidates"]:
            if card["venue"] == "Polymarket":
                assert card["decision"] == "REJECT"
                assert card["contract_quote_reference"]["interval"] is None


def test_public_view_does_not_export_local_paths(tmp_path):
    report, _, _, _ = collect(tmp_path)
    public = public_geopolitical_search_view(report)
    assert str(tmp_path) not in json.dumps(public)
    assert "raw_file" not in json.dumps(public)
    assert "receipt_file" not in json.dumps(public)
    assert report["capture_bundle_path"]


def test_unfinished_pages_explicit_and_bounded(tmp_path):
    def paged(url, headers, timeout):
        status, hdrs, raw = transport(url, headers, timeout)
        doc = json.loads(raw)
        if "events" in doc:
            doc["pagination"]["hasMore"] = True
        return status, hdrs, json.dumps(doc).encode()
    report, _, _, g = collect(tmp_path, chosen_transport=paged)
    assert g.request_count == 14
    assert report["targets"][0]["closest_by_venue"]["Polymarket"]["queries_with_unfinished_pages"]
    assert all(row["page"] <= 2 for row in report["search_ledger"])


def test_no_geopolitical_targets_makes_no_calls(tmp_path):
    report, _, k, g = collect(tmp_path, targets=[{"event_id": "fed", "title": "FOMC rate decision"}])
    assert report["targets"] == []
    assert k.request_count == g.request_count == 0


def test_malformed_market_payload_is_failure_not_empty_match(tmp_path):
    def malformed(url, headers, timeout):
        if urlsplit(url).path.endswith("/public-search"):
            return 200, {}, b'{"events": "invalid"}'
        return transport(url, headers, timeout)
    report, _, _, _ = collect(tmp_path, chosen_transport=malformed)
    assert all(t["closest_by_venue"]["Polymarket"]["status"] == "SEARCH_FAILED" for t in report["targets"])


def test_broad_category_tags_do_not_hide_chip_delivery_series(tmp_path):
    def tagged(url, headers, timeout):
        if urlsplit(url).path.endswith("/series"):
            rows = [{"ticker": "KXH200CHINA", "title": "Nvidia H200 chips delivered to China before 2027?", "tags": ["Jobs & Economy"]}]
            rows += [{"ticker": "KXSEMI" + str(i), "title": "Semiconductor production " + str(i)} for i in range(4)]
            return 200, {}, json.dumps({"series": rows}).encode()
        return transport(url, headers, timeout)
    report, _, _, _ = collect(tmp_path, chosen_transport=tagged)
    assert any(row["kind"] == "markets" and row["query"] == "KXH200CHINA" for row in report["search_ledger"])


def test_source_query_identity_cannot_be_relabelled(tmp_path):
    _, bundle, _, _ = collect(tmp_path)
    bundle["source_records"][0]["query"] = "Different category"
    with pytest.raises(GeopoliticalSearchError, match="query"):
        replay_geopolitical_search(bundle)


def test_fixed_taiwan_plan_searches_the_contract_wording_without_a_year(tmp_path):
    """Live r3/r4 (2026-10-23): the 2026 invasion contract appeared only under a year-free "China invade Taiwan" phrasing."""
    report, bundle, _, _ = collect(tmp_path)
    assert "China invade Taiwan" in bundle["query_plan"]["TAIWAN"]
    searched = {row["query"] for row in bundle["search_ledger"] if row["venue"] == "Polymarket"}
    assert "China invade Taiwan" in searched


def test_fuzzy_search_stops_at_relevance_tail_and_says_why(tmp_path):
    # Gamma matched 8,126 events for "new chip export restrictions" on 2026-10-30; page 2
    # onwards here holds only unrelated events, so the query stops there, not at the bound.
    def fuzzy(url, headers, timeout):
        path, params = urlsplit(url).path, parse_qs(urlsplit(url).query)
        if path.endswith("/series") or path.endswith("/markets"):
            return transport(url, headers, timeout)
        if params["page"] == ["1"]:
            status, hdrs, raw = transport(url, headers, timeout)
            doc = json.loads(raw)
        else:
            doc = {"events": [{"id": "99", "title": "New York mayor election",
                               "markets": [gamma_market("9", "Will the new mayor take office?")]}]}
            hdrs = {"Date": "Wed, 21 Oct 2026 15:00:00 GMT"}
        doc["pagination"] = {"hasMore": True, "totalResults": 8126}
        return 200, hdrs, json.dumps(doc).encode()
    k = KalshiClient(tmp_path / "kalshi", transport=fuzzy, min_interval=0, max_requests=60)
    g = GammaSearchClient(tmp_path / "polymarket", transport=fuzzy, min_interval=0, max_requests=150)
    report = acquire_geopolitical_search(TARGETS, data_dir=tmp_path, kalshi_client=k, gamma_client=g,
                                         evaluated_at="2026-10-21T15:00:00+00:00", max_gamma_pages=10)
    gamma_rows = [row for row in report["search_ledger"] if row["venue"] == "Polymarket"]
    assert max(row["page"] for row in gamma_rows) == 2
    assert all(row.get("stop_reason") == "RELEVANCE_TAIL_NO_FAMILY_MATCH_ON_PAGE" for row in gamma_rows if row["page"] == 2)
    for target in report["targets"]:
        venue = target["closest_by_venue"]["Polymarket"]
        assert venue["queries_with_unfinished_pages"] == []
        assert venue["queries_stopped_at_relevance_tail"]
        assert venue["status"] == "SEARCH_COMPLETED_WITHIN_BOUNDS"
