"""Deterministic adapter tests: synthetic transport responses are fixtures only."""
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from castle_risk.adapters import (
    DataAccessError, KalshiClient, MassiveClient, load_snapshot,
    normalize_market, normalize_quote, sanitize_url,
)

STAMP = "2026-10-16T15:30:00+00:00"


def market(**kwargs):
    return normalize_market({"ticker": "FIXTURE", "market_type": "binary", "status": "active",
                             "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.60", **kwargs}, STAMP)


class FixtureTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, headers.copy()))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return 200, {"Date": "Fri, 16 Oct 2026 15:30:00 GMT"}, json.dumps(value).encode()


def test_dollars_preferred_zero_not_missing_and_zero_size_excluded():
    m = market(yes_bid=80, yes_bid_dollars="0.0000", yes_bid_size_fp="1.00")
    assert m.yes_bid == 0
    assert normalize_quote(m, now=STAMP).lower == 0
    m = market(yes_bid_dollars="0.0000", yes_bid_size_fp="0.00")
    q = normalize_quote(m, now=STAMP)
    assert q.usable and q.lower == 0 and q.observed_yes_bid is None


def test_no_bid_complement_and_positive_size():
    book = {"orderbook_fp": {"yes_dollars": [["0.3", "10"], ["0.9", "0"], ["0.4", "2"]],
                             "no_dollars": [["0.55", "2"], ["0.2", "4"]]},
            "_provenance": {"retrieved_at": STAMP}}
    q = normalize_quote(market(), book, now=STAMP)
    assert q.usable and q.lower == .4 and q.upper == pytest.approx(.45)


def test_one_sided_empty_crossed_and_stale():
    q = normalize_quote(market(), {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.99", "50"]]}}, now=STAMP)
    assert q.usable and q.lower == 0 and q.upper == pytest.approx(.01)
    assert q.observed_yes_bid is None
    empty = normalize_quote(market(), {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}, now=STAMP)
    assert not empty.usable and "Insufficient" in empty.exclusion_reason
    crossed = normalize_quote(market(yes_bid_dollars=".8", yes_ask_dollars=".2"), now=STAMP)
    assert not crossed.usable and "Crossed" in crossed.exclusion_reason
    stale = normalize_quote(market(), now="2026-10-16T15:36:00Z")
    assert not stale.usable and "Stale" in stale.exclusion_reason


def test_metadata_updated_time_does_not_set_quote_timestamp():
    m = market(updated_time="2025-01-01T00:00:00Z")
    assert m.quote_timestamp == STAMP
    assert normalize_quote(m, now=STAMP).usable
    assert m.metadata["quote_exchange_timestamp"] is None


def test_invalid_quote_and_future_time():
    assert not normalize_quote(market(yes_bid_dollars="1.2"), now=STAMP).usable
    assert not normalize_quote(market(status="settled"), now=STAMP).usable
    assert not normalize_quote(market(market_type="scalar"), now=STAMP).usable
    assert not normalize_quote(market(), now="2026-10-16T15:20:00Z").usable
    assert not normalize_quote(market(), now=STAMP, discount_factor=0).usable


def test_market_cursor_pagination_cache_and_archive(tmp_path):
    transport = FixtureTransport({"markets": [{"ticker": "A"}], "cursor": "next"},
                                 {"markets": [{"ticker": "B"}], "cursor": ""})
    client = KalshiClient(tmp_path, transport=transport, min_interval=0)
    result = client.list_markets(limit=3)
    assert [m.ticker for m in result] == ["A", "B"]
    assert "cursor=next" in transport.calls[1][0]
    assert len(client.list_markets(limit=3)) == 2
    assert len(transport.calls) == 2
    metas = list(tmp_path.glob("*.meta.json"))
    assert len(metas) == 2 and all(json.loads(p.read_text())["fixture"] for p in metas)


def test_massive_header_only_and_fraction_weights(tmp_path):
    rows = [{"composite_ticker": "SPY", "effective_date": "2026-10-15", "processed_date": "2026-10-16", "constituent_ticker": "AAPL", "weight": .07, "asset_class": "Equity", "security_type": "Common Stock"}]
    transport = FixtureTransport({"results": rows}, {"results": rows})
    client = MassiveClient(tmp_path, api_key="TEST_SECRET_NOT_REAL", transport=transport, min_interval=0)
    snapshot = client.holdings("SPY", as_of_date="2026-10-16")
    assert snapshot["holdings"][0]["weight"] == .07
    assert snapshot["reliable"] and snapshot["complete"]
    assert snapshot["weight_basis"] == "fraction_of_nav"
    assert all("TEST_SECRET_NOT_REAL" not in url for url, _ in transport.calls)
    assert all(headers["Authorization"] == "Bearer TEST_SECRET_NOT_REAL" for _, headers in transport.calls)
    assert all("TEST_SECRET_NOT_REAL" not in p.read_text() for p in tmp_path.iterdir())


def test_massive_refuses_next_url_host_and_no_credentials(tmp_path):
    transport = FixtureTransport({"results": [], "next_url": "https://attacker.invalid/steal"})
    client = MassiveClient(tmp_path, api_key="TEST_SECRET_NOT_REAL", transport=transport, min_interval=0)
    with pytest.raises(DataAccessError, match="untrusted"):
        client.holdings("SPY", effective_date="2026-10-15")
    assert len(transport.calls) == 1
    with pytest.raises(DataAccessError, match="not configured"):
        MassiveClient(tmp_path, api_key="").holdings("SPY")


def test_pagination_secret_redacted_and_same_date_preserved(tmp_path):
    row = {"composite_ticker": "SPY", "effective_date": "2026-10-15", "processed_date": "2026-10-16", "constituent_ticker": "AAPL", "weight": .07, "asset_class": "Equity"}
    transport = FixtureTransport({"results": [row], "next_url": "https://api.massive.com/etf-global/v1/constituents?cursor=a&apiKey=TEST_SECRET_NOT_REAL"},
                                 {"results": [{**row, "constituent_ticker": "MSFT", "weight": .06}]})
    snapshot = MassiveClient(tmp_path, api_key="TEST_SECRET_NOT_REAL", transport=transport, min_interval=0).holdings("SPY", effective_date="2026-10-15")
    assert snapshot["reliable"] and len(snapshot["holdings"]) == 2
    assert all("apiKey" not in url for url, _ in transport.calls)
    assert all("TEST_SECRET_NOT_REAL" not in p.read_text() for p in tmp_path.iterdir())
    assert sanitize_url("https://api.massive.com/x?apiKey=SECRET&z=1").endswith("?z=1")


def test_errors_do_not_echo_credentials(tmp_path):
    transport = FixtureTransport(HTTPError("https://evil?apiKey=SECRET", 403, "SECRET", {}, None))
    client = MassiveClient(tmp_path, api_key="SECRET", transport=transport, min_interval=0)
    with pytest.raises(DataAccessError) as exc:
        client.holdings("SPY")
    assert "SECRET" not in str(exc.value)
    assert not list(tmp_path.iterdir())


def test_request_budget(tmp_path):
    transport = FixtureTransport({"market": {"ticker": "A"}})
    client = KalshiClient(tmp_path, max_requests=1, transport=transport, min_interval=0)
    client.get_market("A")
    with pytest.raises(DataAccessError, match="budget"):
        client.get_market("B")


def test_snapshot_checksum_and_receipt(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"markets": [{"ticker": "A"}]}))
    meta = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "retrieved_at": STAMP, "fixture": True}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    result = load_snapshot(path)
    assert result[0].quote_timestamp == STAMP and result[0].metadata["fixture"]
    path.write_text("{}")
    with pytest.raises(DataAccessError, match="SHA256"):
        load_snapshot(path)


def test_derivatives_are_not_promoted_to_stock(tmp_path):
    row = {"composite_ticker": "SPY", "effective_date": "2026-10-15", "processed_date": "2026-10-16", "constituent_ticker": "EQUITYSWAP", "weight": .1, "asset_class": "Equity", "security_type": "Equity swap"}
    transport = FixtureTransport({"results": [row]})
    snapshot = MassiveClient(tmp_path, api_key="TEST", transport=transport, min_interval=0).holdings("SPY", effective_date="2026-10-15")
    assert snapshot["holdings"][0]["asset_type"] == "other"
    assert "?" not in snapshot["source_url"]


def test_payout_and_discount_cannot_silently_change_orderbook_semantics():
    assert not normalize_quote(market(notional_value_dollars="2.00"), now=STAMP).usable
    book = {"orderbook_fp": {"yes_dollars": [[".4", "2"]], "no_dollars": [[".5", "2"]]}}
    assert not normalize_quote(market(), book, now=STAMP, discount_factor=.95).usable


def test_zero_quotes_without_positive_size_are_not_probability_evidence():
    m = market(yes_bid_dollars="0.0000", yes_ask_dollars="0.0000")
    q = normalize_quote(m, now=STAMP)
    assert not q.usable and "Insufficient" in q.exclusion_reason
    assert m.yes_bid is None and m.yes_ask is None
    explicit = market(yes_bid_dollars="0.0000", yes_bid_size_fp="2.00", yes_ask_dollars="0.0000", yes_ask_size_fp="3.00")
    q = normalize_quote(explicit, now=STAMP)
    assert q.usable and q.lower == 0 and q.upper == 0
    one_side = market(yes_bid_dollars="0.25", yes_ask_dollars="0.0000")
    q = normalize_quote(one_side, now=STAMP)
    assert q.usable and q.lower == .25 and q.upper == 1 and q.observed_yes_ask is None


def test_nonunit_payout_not_supported_even_if_reported():
    book = {"orderbook_fp": {"yes_dollars": [[".4", "2"]], "no_dollars": [[".5", "2"]]}}
    q = normalize_quote(market(notional_value_dollars="2.00"), book, now=STAMP, payout=2.0)
    assert not q.usable and "unit-dollar" in q.exclusion_reason


# Documented Massive example shape; these are synthesized fixtures, never live
# observations. Fields follow the official ETF constituents response schema.
def constituent(**changes):
    return {"asset_class": "Equity", "composite_ticker": "SPY",
            "constituent_name": "CAESARS ENTERTAINMENT INC COMMON STOCK",
            "constituent_rank": 1, "constituent_ticker": "CZR", "country_of_exchange": "US",
            "currency_traded": "USD", "effective_date": "2026-10-15",
            "figi": "BBG0074Q3NK6", "isin": "US12769G1004", "market_value": 63308625.6,
            "processed_date": "2026-10-16", "security_type": "Common Stock",
            "sedol": "BMWWGB0", "shares_held": 2398054, "us_code": "12769G100",
            "weight": 0.0000958005, **changes}


def response(*rows, next_url=None):
    value = {"status": "OK", "request_id": "synthetic-fixture", "results": list(rows)}
    if next_url:
        value["next_url"] = next_url
    return value


def snapshot_fixture(tmp_path, *responses, **kwargs):
    transport = FixtureTransport(*responses)
    client = MassiveClient(tmp_path, api_key="FIXTURE_NOT_A_CREDENTIAL", transport=transport,
                           min_interval=0, max_retries=0, **kwargs.pop("client_args", {}))
    result = client.holdings("SPY", as_of_date="2026-10-16", **kwargs)
    return result, transport


def test_massive_no_date_path_is_bounded_without_sort_or_latest_probe(tmp_path):
    from urllib.parse import parse_qs, urlsplit
    result, transport = snapshot_fixture(tmp_path, response(constituent()))
    query = parse_qs(urlsplit(transport.calls[0][0]).query)
    assert query == {"composite_ticker": ["SPY"], "effective_date.gte": ["2026-09-17"],
                     "effective_date.lte": ["2026-10-16"], "limit": ["5000"]}
    assert len(transport.calls) == 1 and result["reliable"]
    assert result["processed_date"] == "2026-10-16"
    assert result["metadata"]["point_in_time_verified"] is False
    assert result["metadata"]["selected_batch"] == ["2026-10-15", "2026-10-16"]
    assert Path(result["metadata"]["normalized_path"]).exists()
    assert result["metadata"]["pages"][0]["fixture"] is True


def test_massive_latest_batch_is_client_selected_after_all_pages_without_revision_mixing(tmp_path):
    old = constituent(effective_date="2026-10-14", processed_date="2026-10-15", weight=.10)
    earlier_revision = constituent(processed_date="2026-10-15", weight=.11)
    latest = constituent(weight=.12)
    result, transport = snapshot_fixture(
        tmp_path, response(old, latest, next_url="https://api.massive.com/etf-global/v1/constituents?cursor=page2"),
        response(earlier_revision))
    assert len(transport.calls) == 2 and result["complete"] and result["reliable"]
    assert len(result["holdings"]) == 1 and result["holdings"][0]["weight"] == .12
    assert len(result["metadata"]["candidate_batches"]) == 3
    assert result["metadata"]["weight_sum"] == .12
    assert result["metadata"]["unreported_nav_weight"] == .88


def test_massive_later_page_can_change_latest_effective_selection(tmp_path):
    result, _ = snapshot_fixture(
        tmp_path, response(constituent(effective_date="2026-10-14"), next_url="https://api.massive.com/etf-global/v1/constituents?cursor=page2"),
        response(constituent(weight=.25)))
    assert result["as_of"] == "2026-10-15" and result["holdings"][0]["weight"] == .25


@pytest.mark.parametrize("max_pages,max_requests", [(1, 3), (3, 1)])
def test_massive_incomplete_page_or_request_budget_never_trusts_observed_latest(tmp_path, max_pages, max_requests):
    result, transport = snapshot_fixture(
        tmp_path, response(constituent(), next_url="https://api.massive.com/etf-global/v1/constituents?cursor=page2"),
        max_pages=max_pages, client_args={"max_requests": max_requests})
    assert len(transport.calls) == 1
    assert not result["reliable"] and not result["complete"]
    assert result["metadata"]["selection_provisional"]
    assert "INCOMPLETE_QUERY" in result["metadata"]["quality_issues"]
    assert result["holdings"] and Path(result["metadata"]["normalized_path"]).exists()


def test_massive_cycle_and_mid_pagination_http_failure_preserve_raw_evidence(tmp_path):
    link = "https://api.massive.com/etf-global/v1/constituents?cursor=cycle"
    result, transport = snapshot_fixture(tmp_path / "cycle", response(constituent(), next_url=link),
                                         response(constituent(), next_url=link))
    assert len(transport.calls) == 2 and "PAGINATION_CYCLE" in result["metadata"]["quality_issues"]
    result, _ = snapshot_fixture(tmp_path / "http", response(constituent(), next_url=link),
                                 HTTPError(link, 400, "unsafe server details", {}, None))
    assert not result["reliable"]
    assert "PAGINATION_FETCH_FAILED" in result["metadata"]["quality_issues"]
    assert len(result["metadata"]["raw_paths"]) == 1
    assert "unsafe server details" not in json.dumps(result)


@pytest.mark.parametrize("changes,issue", [
    ({"processed_date": None}, "MISSING_OR_INVALID_BATCH_DATES"),
    ({"processed_date": "not-a-date"}, "MISSING_OR_INVALID_BATCH_DATES"),
    ({"processed_date": "2026-10-14"}, "PROCESSED_DATE_PRECEDES_EFFECTIVE_DATE"),
    ({"composite_ticker": "QQQ"}, "ROW_OUTSIDE_REQUESTED_FUND_OR_DATE_RANGE"),
    ({"effective_date": "2026-09-01"}, "ROW_OUTSIDE_REQUESTED_FUND_OR_DATE_RANGE"),
    ({"weight": -.01}, "NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT"),
    ({"weight": 6.0}, "NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT"),
    ({"weight": "NaN"}, "INVALID_WEIGHT"),
    ({"constituent_rank": {"bad": "rank"}}, "INVALID_CONSTITUENT_RANK"),
])
def test_massive_quality_gates_fail_closed(tmp_path, changes, issue):
    result, _ = snapshot_fixture(tmp_path, response(constituent(**changes)))
    assert not result["reliable"] and issue in result["metadata"]["quality_issues"]


def test_massive_processed_after_cutoff_is_not_historical_information(tmp_path):
    row = constituent(processed_date="2026-10-17")
    result, _ = snapshot_fixture(tmp_path, response(row))
    assert result["complete"] and not result["reliable"]
    assert result["metadata"]["excluded_processed_after_cutoff"] == 1
    assert result["metadata"]["point_in_time_verified"] is False


def test_massive_exact_duplicates_removed_but_conflicting_revisions_rejected(tmp_path):
    row = constituent(weight=.2)
    result, _ = snapshot_fixture(tmp_path / "exact", response(row, row))
    assert result["reliable"] and len(result["holdings"]) == 1
    assert result["metadata"]["exact_duplicate_rows_removed"] == 1
    assert result["metadata"]["weight_sum"] == .2
    changed = constituent(weight=.3)
    result, _ = snapshot_fixture(tmp_path / "conflict", response(row, changed))
    assert not result["reliable"]
    assert "CONFLICTING_DUPLICATE_SECURITY" in result["metadata"]["quality_issues"]


def test_massive_ticker_identifier_conflicts_not_silently_aggregated(tmp_path):
    # Two issuer-identical ticker strings may refer to different instruments.
    one = constituent(weight=.1)
    two = constituent(figi="BBG000B9XRY4", isin="US0378331005", constituent_rank=2, weight=.2)
    result, _ = snapshot_fixture(tmp_path, response(one, two))
    assert not result["reliable"]
    assert "CONFLICTING_DUPLICATE_SECURITY" in result["metadata"]["quality_issues"]


def test_massive_mixed_instruments_missing_weights_and_identifiers_are_preserved(tmp_path):
    rows = [constituent(weight=.4),
            constituent(constituent_ticker="SWAP", figi="", isin="", constituent_rank=2,
                        security_type="Equity Swap", weight=.1),
            constituent(constituent_ticker=None, figi="", isin="US0378331005", constituent_rank=3, weight=.2),
            constituent(constituent_ticker="UNKNOWN_WEIGHT", figi="", isin="", constituent_rank=4, weight=None)]
    result, _ = snapshot_fixture(tmp_path, response(*rows))
    assert result["reliable"] and len(result["holdings"]) == 4
    assert [h["asset_type"] for h in result["holdings"]] == ["stock", "other", "other", "stock"]
    assert result["holdings"][2]["symbol"] == "ISIN_US0378331005"
    assert result["metadata"]["identified_weight"] == .4
    assert result["metadata"]["unclassified_weight"] == pytest.approx(.3)
    assert result["metadata"]["unreported_nav_weight"] == pytest.approx(.3)
    assert result["metadata"]["missing_weight_row_count"] == 1
    assert [h["weight"] for h in result["holdings"]] == [.4, .1, .2, None]


def test_massive_invalid_identifiers_are_not_verified_identifiers(tmp_path):
    result, _ = snapshot_fixture(tmp_path, response(constituent(isin="AAPL", figi="unknown")))
    row = result["holdings"][0]
    assert row["identifier"] == "" and row["identifier_assurance"] == "syntax_only_not_security_master_verified"
    assert len(row["notes"]) == 3


def test_provider_constituent_name_survives_normalization_verbatim(tmp_path):
    from castle_risk.fund_composition import analyze_holdings_snapshot
    name = "BLACKROCK CASH FUNDS TREASURY SL AGENCY SHARES"
    source = constituent(constituent_ticker="XTSLA", constituent_name=name,
                         security_type="Open-End Fund", asset_class="Equity",
                         isin="US0669224778", figi="BBG00LFWD7W1", weight=.0002217)
    before = dict(source)
    result, _ = snapshot_fixture(tmp_path, response(source))
    normalized = result["holdings"][0]
    assert normalized["constituent_name"] == name
    assert normalized["name"] == name
    assert normalized["source_ticker"] == "XTSLA"
    assert normalized["identifiers"]["isin"] == "US0669224778"
    assert source == before
    # The signed composition rechecks original page hashes and original names,
    # including old normalized records whose adapter omitted the name entirely.
    normalized.pop("name")
    normalized.pop("constituent_name")
    composition = analyze_holdings_snapshot(result)
    assert composition["segments"]["MONEY_MARKET_FUND"]["net_weight"] == pytest.approx(.0002217)
    assert "name" not in normalized and "constituent_name" not in normalized


@pytest.mark.parametrize("name", [None, 42, {"text": "MONEY MARKET FUND"}])
def test_missing_or_malformed_provider_name_is_not_invented(name):
    from castle_risk.adapters import _normalize_constituent
    source = constituent(constituent_ticker="XTSLA", constituent_name=name,
                         security_type="Open-End Fund", asset_class="Equity")
    row = _normalize_constituent(source, "SPY")
    assert row["name"] is None and row["constituent_name"] is None


def test_open_end_equity_mutual_fund_name_does_not_become_cash(tmp_path):
    from castle_risk.fund_composition import analyze_holdings_snapshot
    name = "SYNTHETIC S&P 500 EQUITY MUTUAL FUND"
    result, _ = snapshot_fixture(tmp_path, response(constituent(
        constituent_ticker="OTHER_FUND", constituent_name=name,
        security_type="Open-End Fund", asset_class="Equity", weight=.6)))
    assert result["holdings"][0]["name"] == name
    composition = analyze_holdings_snapshot(result)
    assert composition["segments"]["EQUITY"]["net_weight"] == pytest.approx(.6)
    assert "MONEY_MARKET_FUND" not in composition["segments"]


def test_massive_empty_bounded_query_does_not_claim_new_entitlement_failure(tmp_path):
    result, transport = snapshot_fixture(tmp_path, response())
    assert len(transport.calls) == 1 and result["complete"] and not result["reliable"]
    assert result["metadata"]["quality_issues"] == ["NO_ELIGIBLE_DATED_BATCH_IN_BOUNDED_WINDOW"]
    assert "entitlement" not in json.dumps(result).lower()


def test_massive_nav_overallocation_is_not_renormalized(tmp_path):
    result, _ = snapshot_fixture(tmp_path, response(
        constituent(weight=.7), constituent(constituent_ticker="AAPL", figi="BBG000B9XRY4",
                                            isin="US0378331005", constituent_rank=2, weight=.4)))
    assert not result["reliable"] and result["metadata"]["weight_sum"] == pytest.approx(1.1)
    assert [h["weight"] for h in result["holdings"]] == [.7, .4]
