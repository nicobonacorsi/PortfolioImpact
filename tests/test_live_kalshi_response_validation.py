"""Malformed-provider regressions; all HTTP bodies below are synthetic fixtures."""
import json
import socket

import pytest

from castle_risk.adapters import DataAccessError
from castle_risk.provider_access import AuditedKalshiClient, ValidatedKalshiClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Provider-shape tests must not make network requests")
    monkeypatch.setattr(socket.socket, "connect", blocked)


def client_for(tmp_path, payload, cls=ValidatedKalshiClient):
    def transport(url, headers, timeout):
        return 200, {}, json.dumps(payload).encode()
    return cls(tmp_path, transport=transport, min_interval=0,
               max_retries=0, cache_ttl=0)


@pytest.mark.parametrize("payload", [
    {}, {"markets": None}, {"markets": {}}, {"markets": "malformed"},
    {"markets": [None]}, {"markets": ["not a record"]},
    {"markets": [{"ticker": None}]}, {"markets": [{"ticker": True}]},
    {"markets": [{"ticker": " "}]},
    {"markets": [{"ticker": "FIXTURE", "rules_primary": None}]},
    {"markets": [{"ticker": "FIXTURE", "title": {"unexpected": "SECRET_SENTINEL"}}]},
    {"markets": [], "cursor": []}, {"markets": [], "cursor": {}},
    {"markets": [], "cursor": False},
])
def test_malformed_market_page_is_safe_schema_failure(tmp_path, payload):
    client = client_for(tmp_path, payload)
    with pytest.raises(DataAccessError, match="Kalshi response schema rejected") as error:
        client.list_markets()
    assert "SECRET_SENTINEL" not in str(error.value)
    assert client.request_count == 1
    # The original receipt is retained; a schema failure is not an absent source.
    assert len(list(tmp_path.glob("*.meta.json"))) == 1


@pytest.mark.parametrize("payload", [
    {}, {"market": None}, {"market": []}, {"market": {"ticker": []}},
    {"market": {"ticker": "FIXTURE", "close_time": 42}},
])
def test_malformed_single_market_is_safe_schema_failure(tmp_path, payload):
    with pytest.raises(DataAccessError, match="Kalshi response schema rejected"):
        client_for(tmp_path, payload).get_market("FIXTURE")


def test_missing_optional_rules_remain_unknown_not_rejected(tmp_path):
    result = client_for(tmp_path, {"markets": [{"ticker": "FIXTURE"}], "cursor": None}).list_markets()
    assert len(result) == 1 and result[0].ticker == "FIXTURE"
    assert result[0].rules_primary == result[0].rules_secondary == ""
    assert result[0].yes_bid is None and result[0].yes_ask is None


def test_empty_well_formed_page_and_unrelated_endpoint_are_preserved(tmp_path):
    assert client_for(tmp_path / "empty", {"markets": [], "cursor": ""}).list_markets() == []
    event = client_for(tmp_path / "event", {"event": {"ticker": "FIXTURE"}}).get_event("FIXTURE")
    assert event["event"]["ticker"] == "FIXTURE"


def test_audited_boundary_classifies_schema_failure(tmp_path):
    client = client_for(tmp_path, {"markets": [None]}, AuditedKalshiClient)
    with pytest.raises(DataAccessError) as error:
        client.list_markets()
    assert error.value.failure_class == "SCHEMA_REJECTED"
    assert client.failure_ledger[-1]["failure_class"] == "SCHEMA_REJECTED"


def test_live_workflow_retains_partial_report_after_malformed_macro_response(tmp_path, monkeypatch):
    import castle_risk.live_assessment as assessment

    for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def transport(url, headers, timeout):
        calls.append(url)
        payload = {"markets": None} if len(calls) == 1 else {"markets": [], "cursor": ""}
        return 200, {}, json.dumps(payload).encode()

    original_client = assessment.KalshiClient

    def fixture_client(data_dir, **kwargs):
        return original_client(data_dir, **{**kwargs, "transport": transport, "min_interval": 0})

    monkeypatch.setattr(assessment, "KalshiClient", fixture_client)
    report = assessment.run_live_assessment(
        portfolio_rows=[{"symbol": "NVDA", "weight": 1., "asset_type": "stock"}],
        root=tmp_path, release_dates={},
    )
    assert len(calls) == 2  # The second macro family is still inspected.
    assert report["stages"]["Kalshi"]["status"] == "PARTIAL"
    assert len(report["stages"]["Kalshi"]["failures"]) == 1
    assert report["verification_status"].startswith("PARTIAL")
    assert report["events"] == []
    assert all(row["matches_receipt"] for row in report["provider_receipts"])
    assert json.dumps(assessment.public_assessment_view(report), allow_nan=False)


class _ScriptedDiscoveryClaude:
    model = 'SCRIPTED_FAKE_NO_REAL_CLAUDE'
    request_count = max_requests = max_total_tokens = total_tokens = reserved_tokens = 0

    def structured(self, tool_name, schema, payload, system=''):
        self.request_count += 1
        if tool_name == 'submit_events':
            return {'events': [{'event_id': 'cpi', 'title': 'CPI October release', 'description': 'Macro hypothesis',
                                'event_date': '2026-11-10', 'exposure_rationale': 'Equity duration',
                                'search_queries': ['October 2026 CPI']}]}
        return {'matches': []}


def test_live_discovery_retries_a_transient_kalshi_failure_and_keeps_its_search_ledger(tmp_path, monkeypatch):
    """2026-10-23 live r3: one failed GET per lookup (no retry) became ADAPTER_ERROR with no recorded cause."""
    import castle_risk.adapters as adapters
    import castle_risk.live_assessment as assessment
    from castle_risk.agent import SemanticAgent

    for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: None)
    attempts = {}

    def transport(url, headers, timeout):
        attempts[url] = attempts.get(url, 0) + 1
        if attempts[url] == 1:
            return 429, {}, b'{"error": "SECRET_SENTINEL"}'
        return 200, {}, json.dumps({"markets": [], "cursor": ""}).encode()

    original_client = assessment.KalshiClient

    def fixture_client(data_dir, **kwargs):
        return original_client(data_dir, **{**kwargs, "transport": transport, "min_interval": 0})

    monkeypatch.setattr(assessment, "KalshiClient", fixture_client)
    report = assessment.run_live_assessment(
        portfolio_rows=[{"symbol": "NVDA", "weight": 1., "asset_type": "stock"}],
        root=tmp_path, release_dates={}, agent=SemanticAgent(client=_ScriptedDiscoveryClaude()),
    )
    assert attempts and all(count == 2 for count in attempts.values())
    assert report["stages"]["Kalshi"]["status"] == "READS_COMPLETED"
    ledger = report["stages"]["Claude"]["catalog_search_ledger"]
    assert [(row["query"], row["status"], row["source_http_gets"]) for row in ledger] == [("October 2026 CPI", "OK", 2)]
    assert "SECRET_SENTINEL" not in json.dumps(assessment.public_assessment_view(report), allow_nan=False)
