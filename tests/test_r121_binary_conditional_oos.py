"""R121 independent numerical, chronology and no-overclaim regression tests."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
import math
from pathlib import Path

import pytest

from castle_risk.binary_conditional_oos import (
    BinaryConditionalEvidenceError, evaluate_binary_conditional_screen,
    screen_archived_r107_payroll,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/r107_payroll_context.json").read_text(encoding="utf-8"))
ROWS = DATA["event_study"]["observations"]
ASOF = "2026-11-02T00:00:00Z"


def specifications(symbol="SPY", weight=1.):
    return [{"id": "core", "notional_usd": 100_000.,
             "positions": [
                 {"symbol": symbol, "weight": weight, "asset_type": "etf"},
                 {"symbol": "USD", "weight": 1. - weight, "asset_type": "cash"},
             ]}]


def call(rows=ROWS, *, threshold=0., op="lt", specs=None, train_end="2025-12-31", asof=ASOF):
    return evaluate_binary_conditional_screen(
        rows, specs if specs is not None else specifications(),
        threshold_jobs=threshold, operator=op,
        train_end=train_end, analysis_as_of_utc=asof,
        evidence_class="R107_KALSHI_DERIVED_EXPECTATION_NOT_SURVEY_CONSENSUS",
    )


def synthetic_clear_signal():
    start = date(2023, 1, 1)
    data = []
    for i in range(30):
        day = (start + timedelta(days=30 * i)).isoformat()
        yes = i % 2 == 0
        data.append({"date": day, "surprise_jobs": -10_000. if yes else 10_000.,
                     "return_fraction_by_symbol": {"SPY": .02 if yes else -.02}})
    train_end = data[19]["date"]
    return data, train_end


def test_real_original_100k_job_event_abstains_without_fake_zero_or_pnl():
    r = screen_archived_r107_payroll()
    assert r["status"] == "ABSTAIN_INSUFFICIENT_EVENT_BRANCH_SUPPORT"
    assert r["counts"] == {"training_events": 22, "training_yes": 1, "training_no": 21,
                           "heldout_events": 9, "heldout_yes": 1, "heldout_no": 8}
    assert r["conditional_portfolio_centers_usd"] is None
    assert r["heldout_metrics"] is None
    assert r["promoted_to_live_conditional_impact"] is False
    assert r["ex_ante_live_forecast_verified"] is False
    assert r["empirically_identified_causal_effect"] is False


def test_real_31_date_binary_payroll_heldout_both_baselines_and_date_blocks():
    # Exploratory AFTER-the-fact diagnostic on the already disclosed R107
    # dates; never advertise as a newly preregistered blind test.
    r = screen_archived_r107_payroll(threshold_jobs=0, operator="lt")
    assert r["counts"] == {"training_events": 22, "training_yes": 9, "training_no": 13,
                           "heldout_events": 9, "heldout_yes": 5, "heldout_no": 4}
    assert r["status"] == "RESEARCH_ONLY_RETRO_HOLDOUT_NO_INCREMENTAL_EVIDENCE"
    m = r["heldout_metrics"]
    assert m["distinct_later_event_dates"] == 9
    assert m["portfolio_event_rows"] == 63  # NOT 63 independent event dates
    assert m["total_portfolios"] == 7
    assert m["portfolios_beating_both"] == 3
    assert math.isclose(m["pooled_rmse_fraction"]["candidate"], .021679774492464183, abs_tol=1e-12)
    assert math.isclose(m["pooled_rmse_fraction"]["zero"], .020327890375585134, abs_tol=1e-12)
    assert math.isclose(m["pooled_rmse_fraction"]["unconditional_mean"], .02130265671044675, abs_tol=1e-12)
    assert m["relative_improvement"]["zero"] < 0
    assert m["relative_improvement"]["unconditional_mean"] < 0
    assert all(len(bounds) == 2 and bounds[0] <= bounds[1]
               for bounds in m["event_date_bootstrap_95pct"].values())
    assert r["conditional_portfolio_centers_usd"] is None
    assert r["promoted_to_live_conditional_impact"] is False


def test_real_yes_no_fits_are_training_only_and_cash_is_not_refunded():
    r = call(specs=specifications("SPY", .5))
    tr = [row for row in ROWS if row["date"] <= "2025-12-31"]
    yes = [row["return_fraction_by_symbol"]["SPY"] for row in tr if row["surprise_jobs"] < 0]
    pooled = math.fsum(row["return_fraction_by_symbol"]["SPY"] for row in tr) / len(tr)
    expected_yes = pooled + (len(yes) / (len(yes) + 10)) * (math.fsum(yes) / len(yes) - pooled)
    center = r["research_only_training_branch_centers"][0]
    assert math.isclose(center["conditional_train_only_yes_price_return_pct"], 50 * expected_yes, abs_tol=1e-12)
    assert math.isclose(center["conditional_train_only_yes_pnl_usd"], 100_000 * .5 * expected_yes,
                        abs_tol=1e-9)
    assert r["heldout_metrics"]["portfolio_event_rows"] == 9
    assert r["promoted_to_live_conditional_impact"] is False


def test_strong_synthetic_separation_still_cannot_claim_causality_or_live_skill():
    data, train_end = synthetic_clear_signal()
    r = call(data, train_end=train_end, asof="2027-01-01T00:00:00Z")
    assert r["counts"] == {"training_events": 20, "training_yes": 10, "training_no": 10,
                           "heldout_events": 10, "heldout_yes": 5, "heldout_no": 5}
    assert r["research_gate_passed"] is True
    assert r["status"] == "RESEARCH_ONLY_RETRO_HOLDOUT_SCREEN_PASS_NOT_LIVE_VALIDATED"
    assert r["heldout_metrics"]["relative_improvement"]["zero"] > .05
    assert r["heldout_metrics"]["event_date_bootstrap_95pct"]["zero"][0] > 0
    assert r["ex_ante_live_forecast_verified"] is False
    assert r["promoted_to_live_conditional_impact"] is False


def test_future_and_same_day_data_or_duplicate_dates_fail_closed():
    for rows, asof in (
        (ROWS, "2026-10-02T00:00:00Z"),
        (ROWS + [ROWS[0]], ASOF),
    ):
        with pytest.raises(BinaryConditionalEvidenceError):
            call(rows, asof=asof)
    with pytest.raises(BinaryConditionalEvidenceError):
        call(asof="2026-11-02")  # timezone-free timestamp


def test_forged_or_missing_return_and_unsupported_source_fails_closed():
    missing = deepcopy(ROWS)
    del missing[0]["return_fraction_by_symbol"]["SPY"]
    with pytest.raises(BinaryConditionalEvidenceError):
        call(missing)
    invalid = deepcopy(ROWS)
    invalid[0]["return_fraction_by_symbol"]["SPY"] = float("nan")
    with pytest.raises(BinaryConditionalEvidenceError):
        call(invalid)
    too_negative = deepcopy(ROWS)
    too_negative[0]["return_fraction_by_symbol"]["SPY"] = -1.3
    with pytest.raises(BinaryConditionalEvidenceError):
        call(too_negative)


def test_holding_weight_exact_nav_no_renormalization_and_operator_validation():
    p = specifications("SPY", .5)
    p[0]["positions"][1]["weight"] = .4
    with pytest.raises(BinaryConditionalEvidenceError):
        call(specs=p)
    p = specifications("SPY", .5)
    p[0]["positions"][1]["symbol"] = "SPY"
    with pytest.raises(BinaryConditionalEvidenceError):
        call(specs=p)
    with pytest.raises(BinaryConditionalEvidenceError):
        call(op="nonsense")
    with pytest.raises(BinaryConditionalEvidenceError):
        call(threshold=float("inf"))


def test_sparse_event_cannot_be_promoted_by_only_one_training_yes():
    sparse = call(threshold=-100_000, op="le")
    assert sparse["status"] == "ABSTAIN_INSUFFICIENT_EVENT_BRANCH_SUPPORT"
    assert "research_only_training_branch_centers" not in sparse
    assert sparse["conditional_portfolio_centers_usd"] is None
    assert sparse["heldout_metrics"] is None


def test_live_payroll_evidence_context_explicitly_reports_sparse_abort():
    from castle_risk.r107_evidence import payroll_context_for_positions
    reported = payroll_context_for_positions([{"symbol": "SPY", "type": "etf"}])
    screen = reported["binary_conditional_impact_evidence_gate"]
    assert reported["status"] == "HISTORICAL_CONTEXT_ONLY"
    assert screen["status"] == "ABSTAIN_INSUFFICIENT_EVENT_BRANCH_SUPPORT"
    assert screen["counts"]["training_yes"] == 1
    assert screen["historical_return_center_promoted"] is False
    assert reported["portfolio_pnl_updated"] is False
    assert reported["event_probability_updated"] is False


def test_generic_cpi_like_threshold_field_is_explicit_and_never_claims_real_evidence():
    # Synthetic guardrail: verifies applicability of mathematical method to
    # CPI/other event values, NOT a CPI backtest or source-authenticated surprise.
    data, cutoff = synthetic_clear_signal()
    derived = deepcopy(data)
    for row in derived:
        row["first_release_cpi_mom_pp"] = .4 if row["surprise_jobs"] < 0 else .2
        del row["surprise_jobs"]
    r = evaluate_binary_conditional_screen(
        derived, specifications(),
        threshold_value=.3, operator="gt", train_end=cutoff,
        analysis_as_of_utc="2027-01-01T00:00:00Z",
        evidence_class="SYNTHETIC_CPI_EVENT_UNIT_TEST_ONLY",
        observed_field="first_release_cpi_mom_pp",
        event_variable_id="BLS_HEADLINE_FIRST_RELEASE_CPI_MOM_SA",
        observed_unit="percentage_points",
    )
    assert r["event_question"]["unit"] == "percentage_points"
    assert r["event_question"]["threshold_value"] == .3
    assert r["event_question"]["threshold_jobs"] is None
    assert r["event_question"]["different_from_survey_consensus"] is False
    assert r["status"] == "RESEARCH_ONLY_RETRO_HOLDOUT_SCREEN_PASS_NOT_LIVE_VALIDATED"
    assert r["ex_ante_live_forecast_verified"] is False
    assert r["promoted_to_live_conditional_impact"] is False


def test_threshold_and_source_field_cannot_be_ambiguous():
    data, cutoff = synthetic_clear_signal()
    params = dict(event_observations=data, portfolios=specifications(),
                  operator="lt", train_end=cutoff,
                  analysis_as_of_utc="2027-01-01T00:00:00Z",
                  evidence_class="SYNTHETIC_REJECTION_TEST_ONLY")
    with pytest.raises(BinaryConditionalEvidenceError):
        evaluate_binary_conditional_screen(**params)
    with pytest.raises(BinaryConditionalEvidenceError):
        evaluate_binary_conditional_screen(**params, threshold_jobs=0, threshold_value=0)
    with pytest.raises(BinaryConditionalEvidenceError):
        evaluate_binary_conditional_screen(**params, threshold_value=0, observed_field="bad-field")
