"""R119 original-event isolation regressions; synthetic model outputs only.

These tests do not claim empirically verified probabilities.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.agent import AgentError
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.structural_event_forecast import forecast_events
from test_r115_structural_event_forecast import (
    EVENT, CASE, SOURCE, _proposal, _context,
)


SECOND = {
    "id": "HORMUZ_72_HOUR_STOP",
    "date": "2027-01-30",
    "title": "Commercial oil tanker transit through Hormuz ceases for 72 consecutive hours by 2027-01-30",
}


def _both():
    second = probability_question_context(CASE, SECOND)
    assert second is not None
    return [_context(), second]


def test_two_distinct_events_require_two_bounded_calls_and_independent_payloads():
    class Reply:
        model = "injected-isolation-test"

        def __init__(self):
            self.calls = []

        def structured(self, name, schema, payload, system=""):
            assert name == "submit_structural_forecasts"
            assert len(payload["event_targets"]) == 1
            event_id = payload["event_targets"][0]["event_id"]
            assert payload["unverified_public_source_leads"]
            assert {s["id"] for s in payload["unverified_public_source_leads"]} == set(
                payload["event_targets"][0]["event_relevant_source_ids"])
            self.calls.append(event_id)
            if event_id == EVENT["id"]:
                return {"forecasts": [_proposal()]}
            return {"forecasts": [{
                "event_id": event_id, "status": "ABSTAIN",
                "abstain_reason": "A defensible physical probability cannot be supplied.",
                "scenarios": [],
            }]}

    unrelated = {
        "url": "https://www.gao.gov/treasury/example.pdf",
        "title": "Treasury duration and monetary policy",
        "search_query": "U.S. central bank yields and Treasury securities",
        "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED",
    }
    sources = {**SOURCE, "sources": [*SOURCE["sources"], unrelated]}
    client = Reply()
    results = forecast_events(_both(), sources, client)
    assert client.calls == [EVENT["id"], SECOND["id"]]
    assert results[EVENT["id"]]["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert results[SECOND["id"]]["status"] == "ABSTAIN"
    assert results[SECOND["id"]]["probability_center"] is None


def test_failed_second_transport_cannot_erase_first_valid_forecast():
    class Reply:
        model = "injected-isolation-test"

        def structured(self, name, schema, payload, system=""):
            if payload["event_targets"][0]["event_id"] == EVENT["id"]:
                return {"forecasts": [_proposal()]}
            raise AgentError("injected second-event timeout")

    rows = forecast_events(_both(), SOURCE, Reply())
    assert rows[EVENT["id"]]["status"] == "MODEL_FORECAST_UNCALIBRATED"
    assert rows[SECOND["id"]]["status"] == "ABSTAIN"
    assert rows[SECOND["id"]]["reason"] == "FORECAST_MODEL_REQUEST_FAILED"


def test_unexpected_event_id_and_extra_forecasts_fail_closed():
    class Reply:
        model = "injected-isolation-test"

        def __init__(self, with_extra):
            self.with_extra = with_extra

        def structured(self, name, schema, payload, system=""):
            main = _proposal()
            main["event_id"] = payload["event_targets"][0]["event_id"]
            return {"forecasts": [main, {**main, "event_id": "UNREQUESTED_EVENT"}] if
                    self.with_extra else [{**main, "event_id": "UNREQUESTED_EVENT"}]}

    for extra in (True, False):
        row = forecast_events([_context()], SOURCE, Reply(extra))[EVENT["id"]]
        assert row["status"] == "ABSTAIN"
        assert row["probability_center"] is None
        assert row["reason"] in {
            "UNEXPECTED_EVENT_IN_STRUCTURED_RESPONSE", "MODEL_OMITTED_EVENT_FORECAST"
        }


def test_model_c_does_not_consume_market_quote_q_or_change_portfolio_pnl():
    class Reply:
        model = "injected-isolation-test"

        def structured(self, name, schema, payload, system=""):
            assert "market_quote" not in repr(payload)
            assert "portfolio_pnl" not in repr(payload)
            return {"forecasts": [_proposal()]}

    row = forecast_events([_context()], SOURCE, Reply())[EVENT["id"]]
    assert row["forecast_calibrated"] is False
    assert row["physical_probability_empirically_identified"] is False
    assert row["prediction_market_Q_substituted"] is False
    assert row["portfolio_pnl_changed"] is False
