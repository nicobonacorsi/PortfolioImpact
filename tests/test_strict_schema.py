"""Offline raw-HTTP regression coverage: no test contacts Anthropic."""
from copy import deepcopy
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from castle_risk.agent import (AgentError, AnthropicClient, DEFAULT_MODEL, EVENT_SCHEMA,
    EVENTS_SCHEMA, MATCH_SCHEMA, MATCHES_SCHEMA, SEARCH_SCHEMA, FINISH_SCHEMA, validate_schema)
from castle_risk.evaluation import ORDINARY_SCHEMA
from castle_risk.strict_schema import SchemaError, transport_schema, transport_tools


SCHEMAS = [EVENT_SCHEMA, EVENTS_SCHEMA, SEARCH_SCHEMA, FINISH_SCHEMA,
           MATCH_SCHEMA, MATCHES_SCHEMA, ORDINARY_SCHEMA]
REMOVED = {"minLength", "maxLength", "maxItems", "minimum", "maximum",
           "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern"}


def scan(schema):
    assert not REMOVED & schema.keys()
    kind = schema["type"]
    if "object" in (kind if isinstance(kind, list) else [kind]):
        assert schema["additionalProperties"] is False
        for child in schema["properties"].values():
            scan(child)
    if "items" in schema:
        scan(schema["items"])
    if "minItems" in schema:
        assert schema["minItems"] in (0, 1)


def response(value=None, name="submit", **updates):
    payload = {"content": [] if value is None else [{"type": "tool_use", "id": "tool_1", "name": name, "input": value}],
               "usage": {"input_tokens": 100, "output_tokens": 20}, "stop_reason": "tool_use"}
    payload.update(updates)
    return io.BytesIO(json.dumps(payload).encode())


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_actual_schema_is_transformed_at_http_boundary(schema):
    before, captured = deepcopy(schema), []

    def opener(request, timeout):
        captured.append(json.loads(request.data))
        return response()

    c = AnthropicClient(api_key="mock-key", opener=opener, min_interval_seconds=0)
    c.request([], [{"name": "submit", "description": "Submit.", "input_schema": schema, "strict": True}])
    scan(captured[0]["tools"][0]["input_schema"])
    assert schema == before  # Local schema is never weakened in place.
    assert c.request_count == 1


def test_nullable_enums_and_closed_nested_objects():
    original = {"type": "object", "properties": {"x": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "relation": {"type": "string", "enum": ["IMPLIES", "NONE"]}}, "required": ["x", "relation"]}
    out = transport_schema(original)
    assert out["additionalProperties"] is False
    assert out["properties"]["x"]["type"] == ["number", "null"]
    assert out["properties"]["relation"]["enum"] == ["IMPLIES", "NONE"]
    validate_schema({"x": None, "relation": "NONE"}, original)
    with pytest.raises(AgentError, match="Unexpected"):
        validate_schema({"x": None, "relation": "NONE", "unapproved": 1}, original)


@pytest.mark.parametrize("schema,value", [
    ({"type": "string", "minLength": 2}, "x"),
    ({"type": "string", "maxLength": 2}, "xxx"),
    ({"type": "string", "pattern": "^[A-Z]+$"}, "lowercase"),
    ({"type": "array", "items": {"type": "integer"}, "maxItems": 1}, [1, 2]),
    ({"type": "array", "items": {"type": "integer"}, "minItems": 3}, [1, 2]),
    ({"type": "number", "minimum": 0}, -1),
    ({"type": "number", "maximum": 1}, 2),
    ({"type": "number", "exclusiveMinimum": 0}, 0),
    ({"type": "number", "exclusiveMaximum": 1}, 1),
    ({"type": "number", "multipleOf": 0.1}, 0.25),
    ({"type": ["number", "null"], "minimum": 0}, -1),
    ({"type": "integer", "enum": [1]}, True),
    ({"type": "string", "const": "NONE"}, "none"),
])
def test_transport_relaxation_does_not_weaken_local_acceptance(schema, value):
    transport_schema(schema)
    with pytest.raises(AgentError):
        validate_schema(value, schema)


@pytest.mark.parametrize("value,schema", [
    ({"events": [dict(event_id="a", title="x", description="x", event_date="2026-10-16",
                    exposure_rationale="x", search_queries=["x"])]}, EVENTS_SCHEMA),
    ({"query": "x", "event_id": "a"}, SEARCH_SCHEMA),
    ({"summary": ""}, FINISH_SCHEMA),
    ({"matches": [{"market_id": "x", "relationship": "NONE", "evidence": [], "explanation": ""}]}, MATCHES_SCHEMA),
    ({"relationships": [], "target_probability": 1.01}, ORDINARY_SCHEMA),
])
def test_invalid_tool_input_rejected_by_common_request_gate(value, schema):
    c = AnthropicClient(api_key="mock-key", opener=lambda *a, **kw: response(value), min_interval_seconds=0)
    with pytest.raises(AgentError):
        c.request([], [{"name": "submit", "description": "Submit.", "input_schema": schema, "strict": True}])
    assert c.request_ledger[-1]["status"] == "ERROR"
    assert c.request_ledger[-1]["input_tokens"] == 100
    assert c.request_ledger[-1]["cost_estimate_usd"] is not None


@pytest.mark.parametrize("update", [
    {"$ref": "https://example.invalid/schema"}, {"$ref": "#/definitions/x"},
    {"oneOf": [{"type": "string"}]}, {"anyOf": [{"type": "string"}]},
    {"allOf": [{"type": "string"}]}, {"unevaluatedProperties": False},
    {"additionalProperties": True}, {"additionalProperties": {"type": "string"}},
    {"required": ["absent"]}, {"format": "email"}, {"enum": [{}]},
])
def test_unsupported_structures_fail_before_network(update):
    schema = {"type": "object", "properties": {}, **update}
    c = AnthropicClient(api_key="mock-key", opener=lambda *a, **k: pytest.fail("No HTTP allowed"))
    with pytest.raises(AgentError):
        c.request([], [{"name": "submit", "description": "Submit.", "input_schema": schema, "strict": True}])
    assert c.request_count == 0 and not c.request_ledger


def test_recursive_python_schema_fails_closed():
    schema = {"type": "array"}
    schema["items"] = schema
    with pytest.raises(SchemaError):
        transport_schema(schema)


def test_success_attempt_ledger_is_sanitized_and_includes_cost(tmp_path):
    secret = "SECRET_TOKEN_MUST_NOT_APPEAR"
    c = AnthropicClient(api_key=secret, archive_dir=tmp_path, opener=lambda *a, **k:
                        response({"summary": secret}, name="finish_search"), min_interval_seconds=0)
    result = c.structured("finish_search", FINISH_SCHEMA, {})
    assert result["summary"] == "[REDACTED]"
    row = c.request_ledger[0]
    assert row["model"] == DEFAULT_MODEL == "claude-fable-5-1"
    assert row["cost_estimate_usd"] == pytest.approx(0.002)
    assert row["latency_seconds"] >= 0 and row["status"] == "HTTP_RESPONSE_VALIDATED"
    assert secret not in json.dumps(row)
    assert secret not in next(tmp_path.glob("*.json")).read_text()


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_max_one_attempt_including_retryable_errors(status, tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, status, "secret-body", {}, io.BytesIO(b"secret-body"))

    c = AnthropicClient(api_key="mock-key", archive_dir=tmp_path, opener=opener,
                        max_requests=1, min_interval_seconds=0)
    with pytest.raises(AgentError, match=f"HTTP {status}"):
        c.request([], [])
    assert len(calls) == c.request_count == len(c.request_ledger) == 1
    assert c.request_ledger[0]["cost_estimate_usd"] is None
    assert "secret-body" not in next(tmp_path.glob("*.json")).read_text()


def test_per_request_attempt_cap_prevents_one_method_consuming_comparison(monkeypatch):
    monkeypatch.setattr("castle_risk.agent.time.sleep", lambda x: None)
    c = AnthropicClient(api_key="mock-key", max_requests=2, max_attempts_per_request=1,
                       opener=lambda *a, **k: (_ for _ in ()).throw(HTTPError("fixed", 500, "hidden", {}, None)))
    for _ in range(2):
        with pytest.raises(AgentError, match="HTTP 500"):
            c.request([], [])
    assert c.request_count == 2


def test_retry_attempts_each_get_ledger_entry(monkeypatch):
    monkeypatch.setattr("castle_risk.agent.time.sleep", lambda x: None)
    answers = iter([500, 200])

    def opener(request, timeout):
        if next(answers) == 500:
            raise HTTPError(request.full_url, 500, "hidden", {}, None)
        return response()

    c = AnthropicClient(api_key="mock-key", max_requests=2, opener=opener)
    c.request([], [])
    assert [r["http_status"] for r in c.request_ledger] == [500, 200]


def test_network_failure_never_retried_and_unknown_cost_recorded():
    c = AnthropicClient(api_key="mock-key", max_requests=3,
                       opener=lambda *a, **k: (_ for _ in ()).throw(URLError("secret")))
    with pytest.raises(AgentError, match="not retried"):
        c.request([], [])
    assert c.request_count == 1
    assert c.request_ledger[0]["cost_estimate_usd"] is None
    assert c.request_ledger[0]["error"] == "NETWORK_FAILURE_BILLING_UNKNOWN"


@pytest.mark.parametrize("options", [{"max_request_bytes": 256}, {"max_total_tokens": 1}])
def test_input_and_token_reservation_budgets_fail_before_network(options):
    c = AnthropicClient(api_key="mock-key", opener=lambda *a, **k: pytest.fail("No HTTP allowed"), **options)
    with pytest.raises(AgentError, match="budget"):
        c.request([], [])
    assert c.request_count == 0


def test_refusal_is_billed_but_rejected():
    c = AnthropicClient(api_key="mock-key", opener=lambda *a, **k: response(stop_reason="refusal"))
    with pytest.raises(AgentError, match="refused"):
        c.request([], [])
    assert c.request_ledger[0]["input_tokens"] == 100
    assert c.request_ledger[0]["cost_estimate_usd"] is not None


def test_unknown_usage_never_fabricates_zero_cost():
    c = AnthropicClient(api_key="mock-key", opener=lambda *a, **k: response(usage={}))
    c.request([], [])
    assert c.request_ledger[0]["input_tokens"] is None
    assert c.request_ledger[0]["cost_estimate_usd"] is None


def test_unknown_model_does_not_get_default_pricing():
    c = AnthropicClient(api_key="mock-key", model="other-model", opener=lambda *a, **k: response())
    c.request([], [])
    assert c.request_ledger[0]["cost_estimate_usd"] is None


def test_opus_uses_high_effort_and_auto_tool_choice():
    captured = []
    def opener(request, timeout):
        captured.append(json.loads(request.data))
        return response()

    c = AnthropicClient(api_key="mock-key", model="claude-opus-5-5",
                        opener=opener, min_interval_seconds=0)
    c.request([], [])
    assert captured[0]["model"] == "claude-opus-5-5"
    assert captured[0]["output_config"] == {"effort": "high"}
    assert captured[0]["tool_choice"] == {"type": "auto"}


def test_fable_default_uses_max_effort_and_auto_tool_choice():
    captured = []
    def opener(request, timeout):
        captured.append(json.loads(request.data))
        return response()

    c = AnthropicClient(api_key="mock-key", opener=opener, min_interval_seconds=0)
    c.request([], [])
    assert captured[0]["model"] == "claude-fable-5-1"
    assert captured[0]["output_config"] == {"effort": "max"}
    assert captured[0]["tool_choice"] == {"type": "auto"}


def test_non_opus_model_keeps_its_own_effort_defaults():
    captured = []
    def opener(request, timeout):
        captured.append(json.loads(request.data))
        return response()

    c = AnthropicClient(api_key="mock-key", model="claude-sonnet-5-5",
                        opener=opener, min_interval_seconds=0)
    c.request([], [])
    assert captured[0]["model"] == "claude-sonnet-5-5"
    assert "output_config" not in captured[0]


def test_shared_evidence_keeps_full_rules_quotes_and_terms():
    from castle_risk.agent import SemanticAgent
    from castle_risk.evaluation import shared_market_evidence
    market = {"ticker": "PUBLIC-ID", "rules_primary": "Primary. " * 1000,
              "rules_secondary": "Secondary.", "yes_bid_dollars": "0.20", "yes_ask_dollars": "0.30",
              "contract_terms_text": "Complete terms", "retrieved_at": "2026-10-16T00:00:00Z",
              "notional_value_dollars": "1", "api_key": "MUST_NOT_LEAVE"}
    evidence = SemanticAgent._safe_market(market)
    assert evidence == shared_market_evidence(market)
    assert evidence["rules_primary"] == market["rules_primary"]
    assert evidence["yes_ask_dollars"] == "0.30" and evidence["contract_terms_text"] == "Complete terms"
    assert "MUST_NOT_LEAVE" not in json.dumps(evidence)
