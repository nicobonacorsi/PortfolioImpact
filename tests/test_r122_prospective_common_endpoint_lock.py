"""Audit the real 2 Nov 2026 common-endpoint scenario lock: provenance and arithmetic, not skill.

The cases are authentic Massive-marked, Anthropic-proposed scenario assumptions
saved before any outcome. These tests recompute every admitted case with the
deterministic verifier and check that nothing claims empirical validation.
"""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path

import pytest

from castle_risk.common_endpoint_impact import verified_saved_common_endpoint
from castle_risk.us_trading_calendar import common_endpoint_for_event, sessions_strictly_between

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "r122_prospective_common_endpoint_lock_20261102.json"

pytestmark = pytest.mark.skipif(not LOCK.exists(), reason="R122 common-endpoint lock not present")


def _lock():
    return json.loads(LOCK.read_text(encoding="utf-8"))


def test_lock_chronology_provenance_and_no_outcome_claims():
    lock = _lock()
    assert lock["schema"] == "castle.prospective_common_endpoint_lock.v1"
    assert lock["resolved_outcomes_at_lock"] == 0
    assert lock["empirical_skill_claimed"] is False
    assert lock["false_causal_identification_claimed"] is False
    assert lock["raw_prices_redistributed"] is False
    locked = datetime.fromisoformat(lock["locked_at_utc"])
    as_of = datetime.fromisoformat(lock["research_as_of_utc"])
    assert as_of <= locked
    assert lock["price_receipts"] and all(r["request_id"] and r["response_sha256"] for r in lock["price_receipts"].values())
    assert len(lock["cases"]) == 2
    assert lock["admitted_cases"] + lock["abstained_cases"] == sum(len(c["results"]) for c in lock["cases"])
    for entry in lock["anthropic_request_ledger"]:
        # A rejected structured answer is retained as a failed attempt; it never becomes a scenario.
        assert entry["http_status"] == 200
        assert entry["status"] in {"HTTP_RESPONSE_VALIDATED", "ERROR"}
        if entry["status"] == "ERROR":
            assert entry["error"] == "RESPONSE_REJECTED_BY_LOCAL_VALIDATION"
        assert entry["model"] == "claude-fable-5-1"
        assert datetime.fromisoformat(entry["retrieved_at"]) <= locked


def test_every_admitted_case_recomputes_exactly_and_shares_one_endpoint_per_event():
    lock = _lock()
    admitted = 0
    for case in lock["cases"]:
        positions = case["positions"]
        assert abs(sum(p["weight"] for p in positions) - 1) < 1e-12
        assert case["observed_mark_status"] == "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
        mark = case["last_completed_session"]
        assert date.fromisoformat(mark) < datetime.fromisoformat(lock["research_as_of_utc"]).date()
        for symbol, close in case["opening_split_adjusted_closes"].items():
            assert close is not None and close > 0, symbol
        for event_id, row in case["results"].items():
            assert row["event_id"] == event_id
            deadline = date.fromisoformat(row["deadline"])
            expected_endpoint = common_endpoint_for_event(deadline, row["endpoint_rule"]).isoformat()
            assert row["common_endpoint_session_date"] == expected_endpoint
            if row["status"] == "ABSTAIN":
                assert row["event_yes_from_today_center_usd"] is None
                continue
            admitted += 1
            assert row["status"] == "MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED"
            impact = row["impact"]
            assert impact["sessions_from_last_completed_mark_to_endpoint"] == sessions_strictly_between(
                date.fromisoformat(mark), date.fromisoformat(expected_endpoint))
            assert impact["horizon_calendar_end_is_common_across_branches"] is True
            assert verified_saved_common_endpoint(
                impact, positions, case["original_notional_usd"], row["no_event_paths"], row["scenario_branches"],
                as_of_utc=row["as_of_utc"], deadline=row["deadline"], endpoint_rule=row["endpoint_rule"],
                last_completed_session=mark)
            assert row["model_calibrated"] is False and row["empirical_causal_estimate"] is False
            assert impact["forward_returns_calibrated"] is False
            assert impact["future_pnl_observed_or_identified"] is False
            lo, hi = impact["event_yes_from_today_model_set_usd"]
            assert lo <= row["event_yes_from_today_center_usd"] <= hi
            assert abs((row["event_yes_from_today_center_usd"] - row["event_no_from_today_center_usd"])
                       - row["paired_event_gap_center_usd"]) < 1e-6
            reference = impact["historical_no_event_horizon_reference"]
            if reference is not None:
                assert reference["is_forward_forecast"] is False
                assert reference["horizon_sessions"] == impact["sessions_from_last_completed_mark_to_endpoint"]
    assert admitted == lock["admitted_cases"]
