"""R85 measured two-factor response: fitting, holdout admission and the outcome-table channel."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from castle_risk.fund_composition import portfolio_rate_sensitivity
from castle_risk.outcome_table import build_outcome_table
from castle_risk.portfolio import parse_portfolio
from castle_risk.two_factor_response import fit_two_factor

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "fresh_browser_20261023"


def _series(seed, b_spy, b_rate, noise=.001, n=760):
    rng = np.random.default_rng(seed)
    days = [(date(2023, 1, 2) + timedelta(days=i)).isoformat() for i in range(n)]
    spy_r, dy = rng.normal(0, .01, n), rng.normal(0, 6, n)
    fund_r = b_spy * spy_r + b_rate * dy + rng.normal(0, noise, n)
    def path(r):
        level, out = 100.0, []
        for d, x in zip(days, r):
            level *= 1 + x
            out.append({"date": d, "close": level})
        return out
    yields, level = [], 4.0
    for d, x in zip(days, dy):
        level += x / 100
        yields.append({"date": d, "yield_pct": level})
    return path(fund_r), path(spy_r), yields, days


def test_bond_like_fund_is_recovered_and_admitted():
    fund, spy, yields, days = _series(1, .08, -.0006)
    fit = fit_two_factor(fund, spy, yields, train_end=days[500], release_dates=days[520::12])
    assert fit["admitted"] is True and fit["status"] == "ADMITTED_MEASURED_TWO_FACTOR"
    assert fit["spy_loading"] == pytest.approx(.08, abs=.02)
    assert fit["rate_loading_per_bp"] == pytest.approx(-.0006, abs=.00005)
    assert fit["holdout_oos_r2"] > .5 and fit["holdout_event_days"] >= 8


def test_fund_unrelated_to_both_factors_is_not_admitted():
    fund, spy, yields, days = _series(2, 0.0, 0.0, noise=.01)
    fit = fit_two_factor(fund, spy, yields, train_end=days[500])
    assert fit["admitted"] is False and fit["status"] == "NOT_ADMITTED_HOLDOUT_FAILED"


def test_short_history_is_insufficient_not_zero():
    fund, spy, yields, days = _series(3, .5, 0, n=200)
    fit = fit_two_factor(fund, spy, yields, train_end=days[150])
    assert fit["status"] == "INSUFFICIENT_HISTORY" and "spy_loading" not in fit


def _mixed_table(two_factor_fits):
    mixed = json.loads((REPORTS / "custom_portfolio.json").read_text(encoding="utf-8"))
    distribution = next(e["distribution"] for e in mixed["default_analysis"]["events"]
                        if e["family"] == "US_INFLATION" and (e.get("distribution") or {}).get("atoms"))
    portfolio = parse_portfolio(mixed["portfolio"]["positions"], notional=mixed["portfolio"]["notional"])
    # QQQ's class is declared unresolved: without a measured response it blocks the table.
    exposures = {"positions": [{"symbol": "QQQ", "instrument_class": "unresolved_etf",
                                "provider_instrument_type": "ETF", "classification_status": "ETF_ASSET_CLASS_UNRESOLVED",
                                "holdings_composition": {"eligible_for_research": False}}]}
    return build_outcome_table(portfolio, distribution, mixed["empirical"]["daily_betas"],
        release_day_report=mixed["empirical"]["release_day_studies"]["CPI"],
        rate_sensitivity=portfolio_rate_sensitivity(portfolio, mixed["default_analysis"]["fund_signed_compositions"]),
        instrument_exposures=exposures, two_factor_fits=two_factor_fits)


def test_unresolved_fund_blocks_without_and_is_priced_with_an_admitted_fit():
    blocked = _mixed_table({})
    assert blocked["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY" and "QQQ" in blocked["missing_symbols"]
    fit = {"admitted": True, "status": "ADMITTED_MEASURED_TWO_FACTOR", "spy_loading": 1.2,
           "rate_loading_per_bp": -.0002, "covariance": [[1e-4, 0.0], [0.0, 1e-10]]}
    table = _mixed_table({"QQQ": fit})
    assert table["status"] == "COMPLETE_DECLARED_MODEL"
    assert table["instrument_identity_gate"]["two_factor_measured_symbols"] == ["QQQ"]
    row = next(p for p in table["positions"] if p["symbol"] == "QQQ")
    assert row["channel"] == "TWO_FACTOR"
    # Each outcome's total is the sum of its position contributions, and Q-centering holds.
    for outcome in table["outcomes"]:
        parts = sum(c["pnl_usd"] for c in outcome["position_contributions"])
        assert parts == pytest.approx(outcome["pnl_if_outcome_usd"], rel=1e-9, abs=1e-6)
    assert table["summary"]["q_weighted_pnl_usd"] == pytest.approx(0, abs=1e-3)


def test_failed_fit_does_not_unblock():
    fit = {"admitted": False, "status": "NOT_ADMITTED_HOLDOUT_FAILED", "spy_loading": 1.2,
           "rate_loading_per_bp": 0.0, "covariance": [[1e-4, 0], [0, 1e-10]]}
    assert _mixed_table({"QQQ": fit})["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"


def test_treasury_yield_attempt_keeps_acquisition_diagnostics_valid():
    from castle_risk.acquisition_status import massive_acquisition_stage
    inputs = {'attempts': [{'symbol': 'AGG', 'operation': 'daily_prices', 'status': 'OK'},
                           {'symbol': 'US10Y', 'operation': 'treasury_yields', 'status': 'UNAVAILABLE'}],
              'acquisition_quality': {'all_required_transport_operations_succeeded': True}}
    stage = massive_acquisition_stage(inputs, request_count=2)
    assert stage['attempt_diagnostics_valid'] is True
    assert stage['required_transport_complete'] is True
    assert stage['optional_metadata_failures'] == [{'symbol': 'US10Y', 'operation': 'treasury_yields', 'status': 'UNAVAILABLE'}]
