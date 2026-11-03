"""R122 injected common-endpoint model proposals, attach/recheck boundary and tamper checks.

Synthetic model judgments here are not evidence of real financial prediction.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.autonomous_common_endpoint_impact import (
    BATCH_SCHEMA, attach_common_endpoint_impacts, endpoint_rule_for,
    forecast_common_endpoint_impacts, recheck_saved_common_endpoint_impacts,
)
from castle_risk.probability_assumptions import probability_question_context
from castle_risk.strict_schema import transport_schema
from test_r122_common_endpoint_impact import POSITIONS, fixture

EVENT = {"id": "HORMUZ_EVENT", "date": "2027-01-30",
         "title": "Commercial tanker transit through Hormuz interrupted by 2027-01-30"}
CAPTURE = "2026-11-02T03:30:00+00:00"
MARK = {"status": "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION", "as_of_utc": CAPTURE,
        "last_completed_session": "2026-10-30",
        "recent_20_session_mark_change_pct_current_nav": -1.2,
        "historical_portfolio_20_session_mark_change_range_pct_current_nav": [-6.5, 8.6],
        "source_adjustment": "SPLIT_ADJUSTED_CLOSE_EXCLUDES_CASH_DIVIDENDS",
        "equity_priced_in_fraction_identified": False}
SOURCE = {"status": "SEARCHED_SOURCE_LEADS_NOT_VERIFIED",
          "sources": [{"url": "https://www.gao.gov/reports/example.pdf",
                       "title": "Hormuz tanker transit interruption context lead",
                       "search_query": "Hormuz maritime traffic",
                       "evidence_scope": "WEB_SEARCH_RESULT_METADATA_NOT_INDEPENDENTLY_VERIFIED"}]}
PUBLIC_POSITIONS = [{**p, "asset_type": p["type"]} for p in POSITIONS]


def _context(event=EVENT):
    result = probability_question_context({"capture": CAPTURE}, event)
    assert result is not None
    return result


def _proposal():
    nop, branches = fixture()
    for branch in branches:
        branch["rationale"] = "A meaningful hypothetical price transmission with distinct risk channels."
        branch["counterevidence"] = "A diplomatic reversal or a different market regime could overturn it."
        branch["source_ids"] = ["W01"]
    return {"event_id": EVENT["id"], "status": "PROPOSED", "abstain_reason": "Not applicable",
            "no_event_paths": nop,
            "no_event_baseline_rationale": "Ordinary drift with moderate dispersion across the three regimes.",
            "branches": branches}


class FakeClient:
    model = "injected-test-model"

    def __init__(self, rows=None, *, fail=False):
        self.rows = rows if rows is not None else [_proposal()]
        self.fail = fail
        self.calls = []

    def structured(self, name, schema, payload, system=""):
        self.calls.append((name, deepcopy(payload)))
        assert name == "submit_common_endpoint_impacts"
        assert "YES = NO + GAP" in system
        assert payload["do_not_use_probability_P_or_quote_Q_or_pricing_q"] is True
        assert payload["common_endpoint_session_date"] == "2027-03-01"
        assert payload["regular_us_sessions_from_mark_to_endpoint"] == 81
        if self.fail:
            from castle_risk.agent import AgentError
            raise AgentError("transport failure")
        return {"assessments": deepcopy(self.rows)}


def test_schema_is_strict_transportable_and_endpoint_rule_is_deterministic():
    transport_schema(BATCH_SCHEMA)
    assert endpoint_rule_for({"event_id": "KXCPI-26OCT-T0.3"}) == "SCHEDULED_RELEASE_PLUS_2_SESSIONS"
    assert endpoint_rule_for({"event_id": "KXFEDDECISION-26DEC-H25"}) == "SCHEDULED_RELEASE_PLUS_2_SESSIONS"
    assert endpoint_rule_for({"event_id": "HORMUZ_EVENT"}) == "WINDOW_DEADLINE_PLUS_20_SESSIONS"


def test_proposed_common_endpoint_impact_accepts_empty_abstain_reason():
    from castle_risk.agent import validate_schema
    decision = _proposal()
    decision['abstain_reason'] = ''
    validate_schema({'assessments': [decision]}, BATCH_SCHEMA)
    results = forecast_common_endpoint_impacts(
        [_context()], POSITIONS, 100_000., SOURCE, FakeClient([decision]),
        observed_mark_context=MARK)
    assert results[EVENT['id']]['status'] == 'MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED'


def test_injected_proposal_is_verified_and_dollars_match_the_math_module():
    client = FakeClient()
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, client,
                                           observed_mark_context=MARK)
    row = out[EVENT["id"]]
    assert row["status"] == "MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED"
    assert row["common_endpoint_session_date"] == "2027-03-01"
    assert row["impact"]["sessions_from_last_completed_mark_to_endpoint"] == 81
    assert row["event_yes_from_today_center_usd"] == row["impact"]["event_yes_from_today_center_usd"]
    assert row["impact"]["horizon_calendar_end_is_common_across_branches"] is True
    assert row["model_calibrated"] is False and row["empirical_causal_estimate"] is False
    assert len(client.calls) == 1
    assert client.calls[0][1]["historical_horizon_reference_not_a_forecast"] == {"status": "NOT_AVAILABLE"}


def test_missing_mark_transport_failure_and_missing_leads_abstain_without_numbers():
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient(),
                                           observed_mark_context=None)
    assert out[EVENT["id"]]["status"] == "ABSTAIN"
    assert out[EVENT["id"]]["reason"].startswith("NO_MATCHING_COMPLETED_PRICE_MARK")
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient(fail=True),
                                           observed_mark_context=MARK)
    assert out[EVENT["id"]]["status"] == "ABSTAIN"
    assert out[EVENT["id"]]["event_yes_from_today_center_usd"] is None
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., {"status": "NOT_RUN", "sources": []},
                                           FakeClient(), observed_mark_context=MARK)
    assert out[EVENT["id"]]["reason"] == "NO_RELEVANT_PUBLIC_RESEARCH_LEAD"


def test_defective_model_vectors_abstain_rather_than_publish():
    bad = _proposal()
    bad["no_event_paths"]["central"][0]["no_event_return_pct"] = 500.  # schema bound violated at verifier level
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient([bad]),
                                           observed_mark_context=MARK)
    assert out[EVENT["id"]]["status"] == "ABSTAIN"
    assert out[EVENT["id"]]["reason"].startswith("INVALID_COMMON_ENDPOINT_IMPACT")
    unknown_source = _proposal()
    unknown_source["branches"][0]["source_ids"] = ["W09"]
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient([unknown_source]),
                                           observed_mark_context=MARK)
    assert out[EVENT["id"]]["reason"] == "NONMATCHING_SOURCE_LEAD_ID"


def _case(row):
    risk = {"id": EVENT["id"], "title": EVENT["title"], "date": EVENT["date"],
            "event_probability_question": _context()}
    return {"user_supplied": True, "capture": CAPTURE, "notional": 100_000., "positions": deepcopy(PUBLIC_POSITIONS),
            "observed_mark_context": deepcopy(MARK), "risks": [risk], "instrument_identity": {"blocked_symbols": []}}, risk


def test_attach_and_recheck_accept_consistent_rows_and_drop_tampered_ones():
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient(),
                                           observed_mark_context=MARK)
    case, risk = _case(out[EVENT["id"]])
    attach_common_endpoint_impacts(case, out)
    assert risk["common_endpoint_impact"]["status"] == "MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED"
    rechecked = recheck_saved_common_endpoint_impacts(deepcopy(case))
    assert "common_endpoint_impact" in rechecked["risks"][0]
    tampered = deepcopy(case)
    tampered["risks"][0]["common_endpoint_impact"]["event_yes_from_today_center_usd"] += 100.
    assert "common_endpoint_impact" not in recheck_saved_common_endpoint_impacts(tampered)["risks"][0]
    moved = deepcopy(case)
    moved["risks"][0]["common_endpoint_impact"]["impact"]["common_endpoint_session_date"] = "2027-02-26"
    assert "common_endpoint_impact" not in recheck_saved_common_endpoint_impacts(moved)["risks"][0]
    no_mark = deepcopy(case)
    del no_mark["observed_mark_context"]
    assert "common_endpoint_impact" not in recheck_saved_common_endpoint_impacts(no_mark)["risks"][0]
    other_deadline = deepcopy(case)
    other_deadline["risks"][0]["date"] = "2027-02-27"
    assert "common_endpoint_impact" not in recheck_saved_common_endpoint_impacts(other_deadline)["risks"][0]


def test_abstention_rows_are_attached_only_when_they_carry_no_numbers():
    out = forecast_common_endpoint_impacts([_context()], POSITIONS, 100_000., SOURCE, FakeClient(fail=True),
                                           observed_mark_context=MARK)
    case, risk = _case(out[EVENT["id"]])
    attach_common_endpoint_impacts(case, out)
    assert risk["common_endpoint_impact"]["status"] == "ABSTAIN"
    forged = deepcopy(out)
    forged[EVENT["id"]]["event_yes_from_today_center_usd"] = -5000.
    case2, risk2 = _case(forged[EVENT["id"]])
    attach_common_endpoint_impacts(case2, forged)
    assert "common_endpoint_impact" not in risk2
