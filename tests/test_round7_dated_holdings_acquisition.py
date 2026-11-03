"""Realistic-size synthetic pagination; no live call or credential is used."""
from datetime import datetime, timezone
import json
import socket
from urllib.parse import parse_qs, urlsplit

import pytest

from castle_risk import adapters
from castle_risk.adapters import DataAccessError
from castle_risk.live_data import MassiveAssessmentClient, acquire_portfolio_inputs


AT = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
BASE = "https://api.massive.com/etf-global/v1/constituents"


@pytest.fixture(autouse=True)
def no_network_and_fixed_clock(monkeypatch):
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 21, 12, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    def denied(*args, **kwargs):
        raise AssertionError("An injected synthetic transport is required")
    monkeypatch.setattr(adapters, "datetime", FixtureClock)
    monkeypatch.setattr(adapters, "utc_now", lambda: AT.isoformat())
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def row(index, *, processed="2026-10-19", effective="2026-09-30", weight=1 / 6000):
    return {"composite_ticker": "WIDEFUND", "constituent_ticker": f"SEC{index}",
        "weight": weight, "asset_class": "Equity", "security_type": "Common Stock",
        "effective_date": effective, "processed_date": processed, "constituent_rank": index + 1}


class WideFixture:
    """A broad range spans revisions; an exact pair contains 6,000 securities."""
    def __init__(self, *, ignored_processed_filter=False, complete_range=False):
        self.calls = []
        self.ignored_processed_filter = ignored_processed_filter
        self.complete_range = complete_range

    def __call__(self, url, headers, timeout):
        query = parse_qs(urlsplit(url).query)
        self.calls.append(query)
        assert "sort" not in query
        if "cursor" in query:
            assert query["cursor"] == ["exact-page-two"]
            rows = [row(index) for index in range(5000, 6000)]
            if self.ignored_processed_filter:
                rows[0]["processed_date"] = "2026-10-16"
            payload = {"status": "OK", "results": rows}
        elif query["effective_date.gte"] != query["effective_date.lte"]:
            if self.complete_range:
                payload = {"status": "OK", "results": [row(0, weight=1)]}
            else:
                rows = [row(index, processed="2026-10-16" if index % 2 else "2026-10-19") for index in range(5000)]
                payload = {"status": "OK", "results": rows, "next_url": BASE + "?cursor=range-unfinished"}
        elif query["effective_date.gte"] == ["2026-09-30"] and query.get("processed_date") == ["2026-10-19"]:
            payload = {"status": "OK", "results": [row(index) for index in range(5000)],
                       "next_url": BASE + "?cursor=exact-page-two"}
        else:
            payload = {"status": "OK", "results": []}
        return 200, {}, json.dumps(payload).encode()


def client(tmp_path, transport, **settings):
    return MassiveAssessmentClient(tmp_path, api_key="SYNTHETIC_TEST_VALUE", transport=transport,
        min_interval=0, cache_ttl=0, max_retries=0, max_requests=40, **settings)


def test_more_than_5000_exact_pair_rows_are_retrieved_without_pooling_revisions(tmp_path):
    transport = WideFixture()
    service = client(tmp_path, transport)
    limits = (service.max_requests, service.max_retries, service.timeout, service.min_interval)
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21")
    assert result["complete"] is True and result["reliable"] is True
    assert len(result["holdings"]) == 6000
    assert result["metadata"]["selected_batch"] == ["2026-09-30", "2026-10-19"]
    assert result["metadata"]["weight_sum"] == pytest.approx(1)
    assert result["metadata"]["query"]["processed_date"] == "2026-10-19"
    assert result["metadata"]["raw_row_count"] == 6000
    assert len(result["metadata"]["pages"]) == 2
    search = result["metadata"]["acquisition_search"]
    assert search["latest_global_verified"] is False
    assert search["latest_within_complete_range_verified"] is False
    assert search["exact_pair_page_cap"] == 5
    assert search["termination"] == "EXACT_OBSERVED_PAIR_COMPLETED"
    assert service.request_count == len(transport.calls) == 8
    assert limits == (service.max_requests, service.max_retries, service.timeout, service.min_interval)
    assert transport.calls[-2]["processed_date"] == ["2026-10-19"]
    assert "range-unfinished" not in str(transport.calls[1:])


def test_exact_pair_still_capped_is_explicitly_unreliable(tmp_path):
    transport = WideFixture()
    service = client(tmp_path, transport)
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21", max_pages=1)
    assert result["complete"] is False and result["reliable"] is False
    assert "INCOMPLETE_QUERY" in result["metadata"]["quality_issues"]
    assert result["metadata"]["acquisition_search"]["termination"] == "EXACT_PAIR_REMAINS_INCOMPLETE"
    assert len(result["metadata"]["pages"]) == 1
    assert service.request_count <= 40


def test_ignored_processed_filter_cannot_become_a_reliable_snapshot(tmp_path):
    service = client(tmp_path, WideFixture(ignored_processed_filter=True))
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21")
    assert result["complete"] is False and result["reliable"] is False
    assert "PAGINATION_FETCH_FAILED" in result["metadata"]["quality_issues"]


def test_range_that_finishes_keeps_one_request_and_original_quality(tmp_path):
    transport = WideFixture(complete_range=True)
    service = client(tmp_path, transport)
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21")
    assert result["reliable"] is True
    assert service.request_count == 1
    assert result["metadata"]["acquisition_search"]["latest_within_complete_range_verified"] is True
    assert result["metadata"]["acquisition_search"]["latest_global_verified"] is False


def test_completed_invalid_snapshot_is_not_hidden_by_older_date_fallback(tmp_path):
    calls = []
    def transport(url, headers, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"results": [row(0, weight=1.01), row(1, weight=-.01)]}).encode()
    service = client(tmp_path, transport)
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21")
    assert result["complete"] is True and result["reliable"] is False
    assert len(calls) == 1
    assert "NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT" in result["metadata"]["quality_issues"]


def test_remaining_operations_keep_their_budget_and_limits_are_never_raised(tmp_path):
    service = client(tmp_path, WideFixture())
    service.max_requests = 4
    service.request_count = 1
    result = service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21", reserve_requests=2)
    assert service.request_count == 2
    assert service.max_requests == 4
    assert result["complete"] is False and result["reliable"] is False
    assert result["metadata"]["acquisition_search"]["termination"] == "EXACT_SEARCH_STOPPED_BY_BUDGET_OR_PROVIDER_FAILURE"


@pytest.mark.parametrize("settings", [{"max_pages": 6}, {"max_pages": True}, {"lookback_days": 31},
    {"max_date_probes": 31}, {"max_date_probes": 0}, {"reserve_requests": -1}])
def test_limits_reject_expansion_before_any_request(tmp_path, settings):
    transport = WideFixture()
    service = client(tmp_path, transport)
    with pytest.raises(ValueError):
        service.holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21", **settings)
    assert not transport.calls


def test_legacy_client_fallback_keeps_taxonomy_error_and_separates_holdings_quality():
    class LegacyClient:
        request_count = 0
        def ticker_details(self, symbol, **kwargs):
            self.request_count += 1
            return {"ticker": symbol, "type": "ETF"}
        def holdings(self, symbol, **kwargs):
            self.request_count += 1
            assert kwargs["max_pages"] == 5
            return {"complete": False, "reliable": False,
                "metadata": {"quality_issues": ["INCOMPLETE_QUERY"], "selected_batch": ["2026-09-30", "2026-10-19"]}}
        def taxonomy(self, symbol, **kwargs):
            self.request_count += 1
            raise DataAccessError("Data service returned HTTP 403; credentials and server body omitted")
        def daily_prices(self, symbol, **kwargs):
            self.request_count += 1
            return {"symbol": symbol, "prices": [{"date": "2026-10-20", "close": 100.}]}
    inputs = acquire_portfolio_inputs([{"symbol": "WIDEFUND", "weight": 1, "asset_type": "etf"}],
        notional=100_000, client=LegacyClient(), evaluated_at=AT)
    holding = next(a for a in inputs["attempts"] if a["operation"] == "holdings")
    taxonomy = next(a for a in inputs["attempts"] if a["operation"] == "taxonomy")
    assert holding["status"] == "OK" and holding["transport_status"] == "OK"
    assert holding["data_quality_status"] == "UNRELIABLE_OR_INCOMPLETE_SNAPSHOT"
    assert taxonomy["status"] == "UNAVAILABLE" and "403" in taxonomy["reason"]
    quality = inputs["acquisition_quality"]
    assert quality["all_required_transport_operations_succeeded"] is True
    assert quality["all_required_holdings_quality_passed"] is False
    assert quality["holdings_quality_gap_symbols"] == ["WIDEFUND"]
    assert quality["optional_taxonomy_unavailable_symbols"] == ["WIDEFUND"]
    assert quality["full_acceptance_redefined"] is False


def test_ticker_active_field_is_preserved_only_when_provider_returns_it(tmp_path):
    def transport(url, headers, timeout):
        return 200, {}, json.dumps({"status": "OK", "results": {
            "ticker": "INACTIVE", "type": "CS", "active": False, "currency_name": "usd"}}).encode()
    service = client(tmp_path, transport)
    result = service.ticker_details("INACTIVE", as_of=AT.date())
    assert result["active"] is False
    assert "asset_class" not in result


def test_r85_batch_still_being_written_falls_back_to_previous_effective_date(tmp_path):
    # The range probe sees today's batch, but an exact read returns no rows
    # (the provider is still writing it); the previous date is complete.
    calls = []
    def transport(url, headers, timeout):
        query = parse_qs(urlsplit(url).query)
        calls.append(query)
        if query["effective_date.gte"] != query["effective_date.lte"]:
            rows = [row(i, effective="2026-10-20", processed="2026-10-20") for i in range(10)]
            payload = {"status": "OK", "results": rows, "next_url": BASE + "?cursor=range-unfinished"}
        elif query["effective_date.gte"] == ["2026-10-19"] and "processed_date" not in query:
            payload = {"status": "OK", "results": [row(i, effective="2026-10-19", processed="2026-10-19", weight=.1)
                                                   for i in range(10)]}
        else:
            payload = {"status": "OK", "results": []}
        return 200, {}, json.dumps(payload).encode()
    result = client(tmp_path, transport).holdings_for_assessment("WIDEFUND", as_of_date="2026-10-21")
    assert result["complete"] is True and result["reliable"] is True
    assert result["metadata"]["selected_batch"] == ["2026-10-19", "2026-10-19"]
    assert result["metadata"]["acquisition_search"]["termination"] == "OBSERVED_EFFECTIVE_DATE_COMPLETED"
