"""Acquisition guards use synthetic transports only; no provider credential is read."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
from urllib.parse import parse_qs, urlsplit

import pytest

from castle_risk import adapters
from castle_risk.live_data import (
    MassiveAssessmentClient, acquire_portfolio_inputs, verify_saved_responses,
)
from castle_risk.adapters import DataAccessError

AS_OF = date(2026, 10, 21)
RECEIVED = "2026-10-21T12:00:00+00:00"
FIXTURE_KEY = "FIXTURE_NOT_A_CREDENTIAL"


@pytest.fixture(autouse=True)
def isolated_clock_and_network(monkeypatch):
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    def denied(*args, **kwargs):
        raise AssertionError("Acquisition tests must use injected transport")

    monkeypatch.setattr(adapters, "datetime", FixtureClock)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


class Transport:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        assert self.payloads, "Unexpected request beyond fixture sequence"
        result = self.payloads.pop(0)
        status, payload = result if isinstance(result, tuple) else (200, result)
        return status, {}, json.dumps(payload).encode()


def client(tmp_path, *payloads, transport=None):
    transport = transport or Transport(*payloads)
    return MassiveAssessmentClient(tmp_path, api_key=FIXTURE_KEY, transport=transport,
                                    min_interval=0, max_retries=0, cache_ttl=0,
                                    max_requests=100), transport


def taxonomy_row(ticker="GLD", **changes):
    return {"composite_ticker": ticker, "asset_class": "Commodities",
            "effective_date": "2026-10-19", "processed_date": "2026-10-20",
            "leverage_style": "Long", "levered_amount": 1, "focus": "Gold", **changes}


def bar(day="2026-10-20", close=100.):
    timestamp = datetime.fromisoformat(day).replace(hour=4, tzinfo=timezone.utc).timestamp()
    return {"t": int(timestamp * 1000), "c": close}


def prices_payload(*rows, ticker="AAPL", **extra):
    return {"status": "OK", "ticker": ticker, "adjusted": True,
            "resultsCount": len(rows), "results": list(rows), **extra}


def test_metadata_identity_date_query_and_independent_receipt_hash(tmp_path):
    payload = {"status": "OK", "results": {"ticker": "AAPL", "name": "Apple",
               "type": "CS", "currency_name": "usd", "market": "stocks", "locale": "us",
               "sic_description": "ELECTRONIC COMPUTERS", "description": "Provider text"}}
    service, transport = client(tmp_path, payload)
    result = service.ticker_details("aapl", as_of=AS_OF)
    assert result["ticker"] == "AAPL" and result["type"] == "CS"
    assert result["sic_description"] == "ELECTRONIC COMPUTERS"
    assert result["metadata_as_of"] == AS_OF.isoformat()
    assert result["fixture"] is True and result["retrieved_at"] == RECEIVED
    assert hashlib.sha256(Path(result["raw_path"]).read_bytes()).hexdigest() == result["source_response_sha256"]
    assert urlsplit(transport.calls[0][0]).path == "/v3/reference/tickers/AAPL"
    assert parse_qs(urlsplit(transport.calls[0][0]).query) == {"date": ["2026-10-21"]}
    assert transport.calls[0][1]["Authorization"] == "Bearer " + FIXTURE_KEY
    assert all(FIXTURE_KEY not in path.read_text() for path in tmp_path.iterdir() if path.is_file())
    receipts = verify_saved_responses(tmp_path)
    assert len(receipts) == 1 and receipts[0]["matches_receipt"] and receipts[0]["fixture"]
    Path(result["raw_path"]).write_text('{"changed": true}')
    assert verify_saved_responses(tmp_path)[0]["matches_receipt"] is False


@pytest.mark.parametrize("result", [{"ticker": "JPM"}, [], None])
def test_metadata_rejects_wrong_identity_or_shape(tmp_path, result):
    service, _ = client(tmp_path, {"status": "OK", "results": result})
    with pytest.raises(DataAccessError):
        service.ticker_details("AAPL", as_of=AS_OF)


def test_metadata_rejects_provider_error_even_with_result_body(tmp_path):
    service, _ = client(tmp_path, {"status": "ERROR", "results": {"ticker": "AAPL", "type": "CS"}})
    with pytest.raises(DataAccessError):
        service.ticker_details("AAPL", as_of=AS_OF)


def test_taxonomy_latest_dated_revision_and_all_page_provenance(tmp_path):
    endpoint = "https://api.massive.com/etf-global/v1/taxonomies"
    first = {"status": "OK", "results": [taxonomy_row(effective_date="2026-10-16", processed_date="2026-10-17")],
             "next_url": endpoint + "?cursor=next&apiKey=" + FIXTURE_KEY}
    second = {"status": "OK", "results": [taxonomy_row(), taxonomy_row(),
              taxonomy_row(effective_date="2026-10-21", processed_date="2026-10-22")]}
    service, transport = client(tmp_path, first, second)
    result = service.taxonomy("GLD", as_of=AS_OF)
    assert result["effective_date"] == "2026-10-19" and result["processed_date"] == "2026-10-20"
    assert result["asset_class"] == "Commodities" and result["levered_amount"] == 1
    assert len(result["pages"]) == len(transport.calls) == 2
    assert result["source_response_sha256"] == result["pages"][1]["source_response_sha256"]
    assert all(page["fixture"] for page in result["pages"])
    assert "apiKey" not in transport.calls[1][0] and FIXTURE_KEY not in transport.calls[1][0]
    assert parse_qs(urlsplit(transport.calls[0][0]).query)["effective_date.gte"] == [(AS_OF - timedelta(days=89)).isoformat()]
    assert all(row["matches_receipt"] for row in verify_saved_responses(tmp_path))


@pytest.mark.parametrize("rows", [
    [taxonomy_row(), taxonomy_row(asset_class="Fixed Income")],
    [taxonomy_row(ticker="BND")],
    [taxonomy_row(processed_date=None)],
    [taxonomy_row(effective_date="2026-10-22", processed_date="2026-10-23")],
    [taxonomy_row(effective_date="2025-01-01", processed_date="2025-01-02")],
])
def test_taxonomy_rejects_conflicts_invalid_identity_or_out_of_window_dates(tmp_path, rows):
    service, _ = client(tmp_path, {"status": "OK", "results": rows})
    with pytest.raises(DataAccessError):
        service.taxonomy("GLD", as_of=AS_OF)


@pytest.mark.parametrize("endpoint", ["taxonomy", "prices"])
@pytest.mark.parametrize("bad_link", [
    "https://untrusted.invalid/anything", "http://api.massive.com/anything",
    "https://api.massive.com:8443/anything", "https://user@api.massive.com/anything",
    "https://api.massive.com/wrong-path", 123,
])
def test_pagination_never_sends_second_authenticated_request_to_untrusted_link(tmp_path, endpoint, bad_link):
    payload = ({"status": "OK", "results": [taxonomy_row()], "next_url": bad_link} if endpoint == "taxonomy"
               else prices_payload(bar(), next_url=bad_link))
    service, transport = client(tmp_path, payload)
    with pytest.raises(DataAccessError):
        if endpoint == "taxonomy":
            service.taxonomy("GLD", as_of=AS_OF)
        else:
            service.daily_prices("AAPL", start=date(2026, 10, 19), end=date(2026, 10, 20))
    assert len(transport.calls) == 1


@pytest.mark.parametrize("endpoint", ["taxonomy", "prices"])
def test_pagination_cycle_and_page_cap_do_not_claim_complete_data(tmp_path, endpoint):
    path = ("/etf-global/v1/taxonomies" if endpoint == "taxonomy"
            else "/v2/aggs/ticker/AAPL/range/1/day/2026-10-19/2026-10-20")
    link = "https://api.massive.com" + path + "?cursor=repeat"
    payload = ({"status": "OK", "results": [taxonomy_row()], "next_url": link} if endpoint == "taxonomy"
               else prices_payload(bar(), next_url=link))
    for cap, count in ((1, 1), (3, 2)):
        service, transport = client(tmp_path / str(cap), payload, payload)
        with pytest.raises(DataAccessError, match="incomplete|cycle"):
            if endpoint == "taxonomy":
                service.taxonomy("GLD", as_of=AS_OF, max_pages=cap)
            else:
                service.daily_prices("AAPL", start=date(2026, 10, 19), end=date(2026, 10, 20), max_pages=cap)
        assert len(transport.calls) == count


def test_daily_prices_preserve_complete_pagination_hashes_dates_and_adjustment_scope(tmp_path):
    endpoint = "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2026-10-19/2026-10-20"
    service, _ = client(tmp_path,
        prices_payload(bar("2026-10-20", 101.), next_url=endpoint + "?cursor=next"),
        prices_payload(bar("2026-10-19", 100.), bar("2026-10-20", 101.)))
    result = service.daily_prices("AAPL", start=date(2026, 10, 19), end=date(2026, 10, 20))
    assert result["prices"] == [{"date": "2026-10-19", "close": 100.}, {"date": "2026-10-20", "close": 101.}]
    assert len(result["receipts"]) == 2 and all(r["fixture"] for r in result["receipts"])
    assert result["adjustment"] == "split_adjusted_close_not_dividend_total_return"
    assert all(row["matches_receipt"] for row in verify_saved_responses(tmp_path))


@pytest.mark.parametrize("payload", [
    prices_payload(bar(), status="ERROR"),
    prices_payload(bar(), ticker="JPM"),
    prices_payload(bar(), adjusted=False),
    prices_payload(bar(), resultsCount=2),
    prices_payload(bar("2026-10-18")),
    prices_payload(bar("2026-10-21")),
    prices_payload(bar(close=0.)),
    prices_payload(bar(close=True)),
    prices_payload(bar(close="NaN")),
    prices_payload(bar(), bar(close=101.)),
    prices_payload(),
])
def test_daily_prices_reject_error_incomplete_identity_invalid_bars_and_conflicts(tmp_path, payload):
    service, _ = client(tmp_path, payload)
    with pytest.raises(DataAccessError):
        service.daily_prices("AAPL", start=date(2026, 10, 19), end=date(2026, 10, 20))


def test_each_owned_etf_gets_holdings_taxonomy_and_prices_with_no_stale_fallback(tmp_path):
    calls = []

    def route(url, headers, timeout):
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        calls.append((parsed.path, query))
        if parsed.path.startswith("/v3/reference/tickers/"):
            symbol = parsed.path.rsplit("/", 1)[-1]
            payload = {"status": "OK", "results": {"ticker": symbol, "type": "ETF" if symbol in {"GLD", "BND"} else "CS",
                       "currency_name": "usd", "market": "stocks", "locale": "us"}}
        elif parsed.path == "/etf-global/v1/constituents":
            symbol = query["composite_ticker"][0]
            if symbol == "GLD":
                return 403, {}, b'{}'
            payload = {"status": "OK", "results": [{"composite_ticker": symbol, "effective_date": "2026-10-19",
                       "processed_date": "2026-10-20", "constituent_ticker": "CASH", "weight": .01,
                       "asset_class": "Cash", "currency_traded": "USD"}]}
        elif parsed.path == "/etf-global/v1/taxonomies":
            symbol = query["composite_ticker"][0]
            payload = {"status": "OK", "results": [taxonomy_row(symbol, asset_class="Commodities" if symbol == "GLD" else "Fixed Income")]}
        elif parsed.path == "/fed/v1/treasury-yields":
            payload = {"status": "OK", "results": [{"date": "2026-10-19", "yield_10_year": 4.1},
                                                   {"date": "2026-10-20", "yield_10_year": 4.12}]}
        else:
            assert parsed.path.startswith("/v2/aggs/ticker/")
            symbol = parsed.path.split("/")[4]
            assert parsed.path.endswith("/2026-10-10/2026-10-20")
            payload = prices_payload(bar(), ticker=symbol)
        return 200, {}, json.dumps(payload).encode()

    service, _ = client(tmp_path, transport=route)
    portfolio = [{"symbol": "AAPL", "weight": .3, "asset_type": "stock"},
                 {"symbol": "JPM", "weight": .2, "asset_type": "stock"},
                 {"symbol": "GLD", "weight": .2, "asset_type": "etf"},
                 {"symbol": "BND", "weight": .2, "asset_type": "etf"},
                 {"symbol": "CASH", "weight": .1, "asset_type": "cash"}]
    result = acquire_portfolio_inputs(portfolio, notional=100_000, client=service,
        evaluated_at=datetime(2026, 10, 21, 14, tzinfo=timezone.utc), history_days=10,
        prior_holdings={"GLD": {"retained_prior": True}})
    for operation in ("/etf-global/v1/constituents", "/etf-global/v1/taxonomies"):
        assert sorted(query["composite_ticker"][0] for path, query in calls if path == operation) == ["BND", "GLD"]
    assert set(result["instrument_metadata"]) == {"AAPL", "JPM", "GLD", "BND"}
    assert result["instrument_metadata"]["GLD"]["type"] == "ETF"
    assert result["instrument_metadata"]["GLD"]["taxonomy"]["composite_ticker"] == "GLD"
    assert result["instrument_metadata"]["GLD"]["taxonomy"]["source_response_sha256"]
    assert set(result["holdings"]) == {"BND"}
    assert result["prior_holdings_not_substituted_for_live"] == ["GLD"]
    assert set(result["prices_by_symbol"]) == {"AAPL", "JPM", "GLD", "BND", "SPY"}
    assert [row["yield_pct"] for row in result["treasury_10y"]] == [4.1, 4.12]
    assert all(source["end"] == "2026-10-20" for source in result["price_sources"].values())
    assert any(row["symbol"] == "GLD" and row["operation"] == "holdings" and row["status"] == "UNAVAILABLE"
               for row in result["attempts"])
    assert result["provider_request_count"] == len(calls) == 14  # includes the 10Y yield series
