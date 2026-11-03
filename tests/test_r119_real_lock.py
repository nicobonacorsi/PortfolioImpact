"""Verify frozen provider-backed R119 pilot is still unresolved, not a backtest.

The archive is an actual historical GitHub commit with four issued provider
message IDs. This local test verifies its math/chronology; it CANNOT validate
the external provider receipt, source contents, or future outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from castle_risk.credal_scenario import evaluate_structural_partition
from castle_risk.prospective_oos_score import evaluate_prospective_lock

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "r119_prospective_event_lock_20261101.json"


def test_live_prospective_event_lock_math_reproducible_without_outcomes():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["schema"] == "castle.prospective_event_lock.v1"
    assert lock["no_completed_outcomes_as_of_lock"] is True
    assert len(lock["events"]) == 4
    for row in lock["events"]:
        assert row["outcome_status"] == "UNRESOLVED_AT_LOCK"
        forecast = row["model_forecast"]
        assert forecast["provider_message_id"].startswith("msg_")
        assert forecast["forecast_is_uncalibrated"] is True
        audit = evaluate_structural_partition(forecast["scenario_branches"])
        assert forecast["probability_center"] == pytest.approx(
            audit["probability_center"], abs=1e-9)
        assert forecast["credal_interval"] == pytest.approx(
            audit["credal_interval"], abs=1e-9)
    result = evaluate_prospective_lock(
        lock, [], audit_time_utc="2026-11-01T23:55:00Z")
    assert result["status"] == "NO_RESOLVED_HOLDOUT_EVENTS_YET"
    assert result["resolved_event_count"] == 0
    assert result["model_brier"] is None
    assert result["model_log_loss"] is None
    assert result["forecast_empirically_calibrated"] is False


def test_live_market_quotes_do_not_get_labeled_as_physical_event_p():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    quotes = {r["event_id"]: r["market_quote"] for r in lock["events"]
              if r["market_quote"] is not None}
    assert len(quotes) == 2
    assert quotes["KXFEDDECISION-26DEC-H25"]["match_status"] == "EXACT_CONTRACT_RULES_AUDITED"
    assert quotes["KXCPI-26OCT-T0.3"]["match_status"] == "PROVISIONAL_MATCH_NOT_SCORED"
    for row in lock["events"]:
        assert row["model_forecast"]["probability_center"] is not None
        q = row["market_quote"]
        if q:
            assert q["yes_bid"] <= q["yes_midpoint_benchmark"] <= q["yes_ask"]
            assert q["captured_at_utc"] <= row["model_forecast"]["provider_response_received_at_utc"]
