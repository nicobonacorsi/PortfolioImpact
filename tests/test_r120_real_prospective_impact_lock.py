"""Audit actual 2 Nov 2026 provider-backed impact lock: provenance, not skill.

The twelve cases are real Massive/Anthropic source-derived *prospective*
scenario estimates. No future price marks or adjudicated outcomes exist in the
lock. These tests do not turn conditional guesses into an accurate forecast.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from json import loads
from math import fsum, sqrt
from pathlib import Path

import pytest

from castle_risk.impact_oos_score import (
    ImpactOosInputError, score_forward_portfolio_impacts,
)
from castle_risk.prospective_oos_score import evaluate_prospective_lock

ROOT = Path(__file__).resolve().parents[1]
IMPACT = ROOT / "data" / "r120_prospective_impact_lock_20261102.json"
PROBABILITY = ROOT / "data" / "r119_prospective_event_lock_20261101.json"
HISTORICAL = ROOT / "docs" / "r103" / "benchmark" / "economic_oos.json"

FUNDED_PORTFOLIOS = {
    "tech_duration": [("NVDA", .35), ("QQQ", .30), ("TLT", .20), ("USD", .15)],
    "energy_defense": [("XLE", .30), ("RTX", .25), ("IEF", .25), ("GLD", .20)],
    "global_trade": [("XOM", .30), ("TSM", .25), ("LMT", .25), ("IEF", .20)],
}
ENDPOINTS = {
    "KXCPI-26OCT-T0.3": "2026-11-12",
    "KXFEDDECISION-26DEC-H25": "2026-12-11",
    "TAIWAN_FUTURE_CARGO_QUARANTINE_2027_01_30": "2027-02-26",
    "HORMUZ_INTERRUPT_2027_01_30": "2027-02-26",
}


def _load(path):
    return loads(path.read_text(encoding="utf-8"))


def test_actual_lock_contains_all_twelve_precisely_original_events_and_positions():
    lock = _load(IMPACT)
    probability = _load(PROBABILITY)
    source = {e["event_id"]: e for e in probability["events"]}
    assert set(source) == set(ENDPOINTS)
    assert lock["schema"] == "castle.prospective_impact_lock.v1"
    assert lock["locked_cases"] == 12
    assert lock["cohort_count"] == 3 and lock["family_count"] == 4
    assert lock["resolved_outcomes_at_lock"] == 0
    assert lock["empirical_skill_claimed"] is False
    assert lock["false_causal_identification_claimed"] is False
    seen = set()
    tally = Counter()
    prices = {r["symbol"]: r for r in lock["opening_price_receipts"]}
    assert len(prices) == 10
    assert set(prices) == {"NVDA", "QQQ", "TLT", "XLE", "RTX", "IEF", "GLD", "XOM", "TSM", "LMT"}
    for ticker, receipt in prices.items():
        assert receipt["api_request_id"]
        assert receipt["price_session_date"] == "2026-10-30"
        assert receipt["split_adjusted_close"] > 0
        assert receipt["bar_time_ms"] == 1793332800000
    locked = datetime.fromisoformat(lock["locked_at_utc"].replace("Z", "+00:00"))
    for case in lock["events"]:
        key = case["case_id"]
        assert key not in seen
        seen.add(key)
        portfolio, event = key.split("__", 1)
        assert case["event_id"] == event
        assert case["event_condition"] == source[event]["target_condition"]
        assert case["event_family_id"] == source[event]["event_family_id"]
        assert case["fixed_price_return_end_session_date"] == ENDPOINTS[event]
        assert case["opening_close_session_date"] == "2026-10-30"
        assert case["status"] == "MODEL_ASSUMPTION_UNCALIBRATED"
        assert case["original_notional_usd"] == 100000
        assert case["provider"] == "Anthropic"
        assert case["provider_message_id"].startswith("msg_")
        assert case["exact_model_input_prompt"]
        assert case["prompt_variant"] in (
            "initial_verified_all_original_holdings", "strict_no_extra_cash_retry")
        assert datetime.fromisoformat(case["opening_mark_recorded_at_utc"].replace("Z", "+00:00")) <= datetime.fromisoformat(
            case["model_response_received_at_utc"].replace("Z", "+00:00")) <= locked
        holdings = case["positions"]
        assert [(p["symbol"], p["weight"]) for p in holdings] == FUNDED_PORTFOLIOS[portfolio]
        assert fsum(p["weight"] for p in holdings) == pytest.approx(1.0, abs=1e-12)
        for row in holdings:
            ticker = row["symbol"]
            assert row["opening_split_adjusted_close"] == (
                1 if ticker == "USD" else prices[ticker]["split_adjusted_close"])
            for branch in ("yes", "no"):
                lo, hi = row["uncalibrated_assumption_ranges_pct"][branch]
                center = row["predicted_" + branch + "_return_pct"]
                assert -100 <= lo <= center <= hi <= 200
                if ticker == "USD":
                    assert (lo, center, hi) == (0, 0, 0)
            assert "USD_CASH" not in [p["symbol"] for p in holdings]
        for branch in ("yes", "no"):
            recalculated = 1000 * fsum(
                row["weight"] * row["predicted_" + branch + "_return_pct"]
                for row in holdings)
            assert case["conditional_center_portfolio_pnl_usd"][branch] == pytest.approx(
                recalculated, abs=1e-8)
        assert case["future_event_independently_adjudicated_at_lock"] is False
        assert case["actual_future_price_return_at_lock"] is None
        assert case["causal_yes_minus_no_validated"] is False
        tally[event] += 1
    assert tally == Counter({id: 3 for id in ENDPOINTS})


def test_two_rejected_nonexistent_cash_positions_are_preserved_not_hidden():
    lock = _load(IMPACT)
    rejected = lock["discarded_nonconforming_attempts"]
    assert len(rejected) == lock["model_calls_initially_rejected_before_freeze"] == 2
    ids = {r["case_id"] for r in rejected}
    assert ids == {
        "energy_defense__KXCPI-26OCT-T0.3",
        "global_trade__KXCPI-26OCT-T0.3",
    }
    accepted = {c["case_id"]: c for c in lock["events"]}
    for row in rejected:
        assert "USD_CASH" in row["failure"]
        assert row["provider_message_id"] != accepted[row["case_id"]]["provider_message_id"]
        assert row["outcome_used_to_select_replacement"] is False
        assert accepted[row["case_id"]]["prompt_variant"] == "strict_no_extra_cash_retry"


def test_live_forecast_lock_is_pending_not_a_finished_skill_test():
    lock = _load(IMPACT)
    score = score_forward_portfolio_impacts(
        lock, [], audit_time_utc="2026-11-02T23:30:00Z",
        minimum_independent_event_families=2)
    assert score["status"] == "PENDING_NO_RESOLVED_FIXED_HORIZONS"
    assert score["locked_cases"] == 12
    assert score["scored_cases"] == 0
    assert score["model_rmse_pct_nav"] is None
    assert score["model_mae_pct_nav"] is None
    assert score["event_yes_minus_no_causal_effect_validated"] is False
    assert score["external_adjudication_and_provider_price_authenticity_certified"] is False
    ps = evaluate_prospective_lock(
        _load(PROBABILITY), [], audit_time_utc="2026-11-02T23:30:00Z")
    assert ps["status"] == "NO_RESOLVED_HOLDOUT_EVENTS_YET"
    assert ps["model_brier"] is None
    assert ps["locked_event_count"] == 4


def test_real_historical_heldout_price_transmission_still_fails_original_gates():
    """Recompute published independent 720-row aggregates, never re-fit on test."""
    report = _load(HISTORICAL)
    rows = report["portfolio_metrics"]
    assert len(rows) == 20
    assert sum(row["n_dates"] for row in rows) == 720
    pool = {name: sqrt(
        fsum(row["n_dates"] * row["rmse_fraction"][name] ** 2 for row in rows)
        / 720)
        for name in ("candidate", "zero", "spy")}
    for name, value in pool.items():
        assert value == pytest.approx(report["pooled_rmse_fraction"][name], abs=1e-12)
    wins = sum(row["rmse_fraction"]["candidate"] < min(
        row["rmse_fraction"]["spy"], row["rmse_fraction"]["zero"]) for row in rows)
    assert wins == report["portfolios_beating_both"] == 10
    assert 1 - pool["candidate"] / pool["zero"] == pytest.approx(.136003678, abs=1e-8)
    assert 1 - pool["candidate"] / pool["spy"] == pytest.approx(.003530813, abs=1e-8)
    assert report["event_block_bootstrap_95pct"]["versus_spy"][0] < 0
    assert report["gate_passed"] is False and report["status"] == "FAIL"


def test_exact_endpoint_and_input_identity_reject_fabricated_later_outcome():
    from copy import deepcopy
    lock = _load(IMPACT)
    changed = deepcopy(lock)
    changed["events"][0]["fixed_price_return_end_session_date"] = "2026-10-31"
    with pytest.raises(ImpactOosInputError):
        score_forward_portfolio_impacts(
            changed, [], audit_time_utc="2026-11-02T23:30:00Z")
