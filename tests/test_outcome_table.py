"""Outcome-table engine: loadings, rate channel, Q-centering and the recorded mixed portfolio."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from castle_risk.outcome_table import (CPI_RATE_PROXY_UPPER, HETEROGENEITY_Z, build_outcome_table,
                                       combine_loadings)
from castle_risk.fund_composition import portfolio_rate_sensitivity
from castle_risk.portfolio import parse_portfolio

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "fresh_browser_20261023"


def fit(beta, half_width, n_train, status="PASS"):
    return {"status": "ESTIMATED_INPUT_ASSOCIATION", "beta": beta,
            "beta_interval": [beta - half_width, beta + half_width], "n_train": n_train,
            "transfer_check": {"status": status}}


def test_precision_weighting_when_announcement_days_are_not_distinguishable():
    combined = combine_loadings(fit(2.085, .29, 502), fit(2.068, 3.36, 16))
    assert combined["basis"] == "PRECISION_WEIGHTED_DAILY_AND_ANNOUNCEMENT_DAY"
    assert abs(combined["heterogeneity_z"]) < HETEROGENEITY_Z
    assert combined["loading"] == pytest.approx(2.085, abs=.005)
    assert combined["se"] < .17  # sixteen announcement days no longer dominate the uncertainty
    assert combined["se_if_independent"] < combined["se"]  # dependence-robust, never the narrower pooled value


def test_pooled_standard_error_does_not_assume_independent_samples():
    """Two equally precise estimates from overlapping price history: no sqrt(2) gain is claimed."""
    combined = combine_loadings(fit(1.0, .2, 500), fit(1.0, .2, 500))
    assert combined["se_if_independent"] == pytest.approx(combined["daily_se"] / 2 ** .5)
    assert combined["se"] == pytest.approx(combined["daily_se"])


def test_heterogeneous_announcement_beta_is_used_only_with_a_passing_holdout():
    passing = combine_loadings(fit(.094, .10, 502), fit(.578, .47, 16, "PASS"))
    assert passing["basis"] == "ANNOUNCEMENT_DAY_HETEROGENEOUS_AND_HOLDOUT_PASS"
    assert passing["loading"] == pytest.approx(.578)
    failing = combine_loadings(fit(.094, .10, 502), fit(.664, .37, 23, "FAIL"))
    assert failing["basis"] == "HETEROGENEOUS_UNRESOLVED_POOLED_POINT_WIDENED"
    assert failing["se"] > (failing["event_day_beta"] - failing["daily_beta"]) / 2


def test_missing_daily_fit_is_missing_not_zero():
    assert combine_loadings(None, fit(1.0, .2, 16)) is None


@pytest.fixture(scope="module")
def recorded():
    mixed = json.loads((REPORTS / "custom_portfolio.json").read_text(encoding="utf-8"))
    nvda = json.loads((REPORTS / "nvidia_final.json").read_text(encoding="utf-8"))
    return mixed, nvda


def _distribution(report, family):
    return next(e["distribution"] for e in report["default_analysis"]["events"]
                if e["family"] == family and (e.get("distribution") or {}).get("atoms"))


def _table(mixed, distribution, study, **kwargs):
    portfolio = parse_portfolio(mixed["portfolio"]["positions"], notional=mixed["portfolio"]["notional"])
    return build_outcome_table(portfolio, distribution, mixed["empirical"]["daily_betas"],
                               release_day_report=mixed["empirical"]["release_day_studies"][study],
                               rate_sensitivity=portfolio_rate_sensitivity(portfolio,
                                   mixed["default_analysis"]["fund_signed_compositions"]), **kwargs)


def test_recorded_mixed_portfolio_fomc_table_has_determined_signs(recorded):
    """Falsifiable acceptance from the 2026-10-23 review: 20/50/30 at $250,000 on the
    recorded 2026-10-28 FOMC ladder (historical quotes, not current)."""
    mixed, nvda = recorded
    table = _table(mixed, _distribution(nvda, "FED_POLICY"), "FOMC")
    assert table["status"] == "COMPLETE_DECLARED_MODEL"
    rows = {r["outcome_id"]: r for r in table["outcomes"]}
    hike, cut, hold = rows["KXFEDDECISION-26OCT-H25"], rows["KXFEDDECISION-26OCT-C25"], rows["KXFEDDECISION-26OCT-H0"]
    assert hike["pnl_if_outcome_usd"] == pytest.approx(-3445, abs=5)
    assert hike["pnl_model_range_5_95_usd"][1] < 0 < cut["pnl_model_range_5_95_usd"][0]
    assert hike["sign_probability_under_model"] > .99
    assert 0 < hold["pnl_if_outcome_usd"] < 200
    assert abs(table["summary"]["q_weighted_pnl_usd"]) < 1e-6
    assert table["data_quality_gaps"] == []
    for row in table["outcomes"]:  # the displayed range spans every feasible Q, so it contains the reference-Q range
        assert row["pnl_model_range_5_95_usd"][0] <= row["pnl_model_range_5_95_at_reference_q_usd"][0] + 1e-9
        assert row["pnl_model_range_5_95_usd"][1] >= row["pnl_model_range_5_95_at_reference_q_usd"][1] - 1e-9
    low, high = table["reference_expectation_range_over_q_polytope"]
    assert low <= table["reference_expectation"] <= high and high > low
    assert hike["sign_determined_for_every_feasible_q"]
    channels = {p["symbol"]: p["channel"] for p in table["positions"]}
    assert channels == {"NVDA": "EQUITY", "QQQ": "EQUITY", "TLT": "RATES"}
    tlt = next(p for p in table["positions"] if p["symbol"] == "TLT")
    assert tlt["duration"]["duration"] == pytest.approx(14.77, abs=.01)


def test_recorded_mixed_portfolio_cpi_table_is_complete_and_hot_prints_lose(recorded):
    mixed, _ = recorded
    table = _table(mixed, _distribution(mixed, "US_INFLATION"), "CPI")
    assert table["status"] == "COMPLETE_DECLARED_MODEL"
    assert table["factors"]["rates"]["status"] == "DECLARED_PROXY_CORE_CPI_RESPONSE_BOUNDS_HEADLINE"
    assert table["factors"]["rates"]["draw"][2] == pytest.approx(CPI_RATE_PROXY_UPPER)
    values = [(r["state"], r["pnl_if_outcome_usd"]) for r in table["outcomes"]]
    assert all(b[1] < a[1] for a, b in zip(values, values[1:]))  # monotone: hotter CPI, lower P&L
    assert abs(sum(r["q_weighted_contribution_usd"] for r in table["outcomes"])) < 1e-6


def test_seeded_monte_carlo_is_reproducible(recorded):
    mixed, nvda = recorded
    first = _table(mixed, _distribution(nvda, "FED_POLICY"), "FOMC", n_draws=4000)
    second = _table(mixed, _distribution(nvda, "FED_POLICY"), "FOMC", n_draws=4000)
    assert first["outcomes"] == second["outcomes"]


def test_unknown_instrument_makes_the_table_partial_not_zero(recorded):
    mixed, nvda = recorded
    positions = [dict(p) for p in mixed["portfolio"]["positions"]]
    positions[0]["symbol"] = "ZZZZ"
    portfolio = parse_portfolio(positions, notional=mixed["portfolio"]["notional"])
    table = build_outcome_table(portfolio, _distribution(nvda, "FED_POLICY"), mixed["empirical"]["daily_betas"],
                                release_day_report=mixed["empirical"]["release_day_studies"]["FOMC"],
                                rate_sensitivity=portfolio_rate_sensitivity(portfolio,
                                    mixed["default_analysis"]["fund_signed_compositions"]))
    assert table["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert table["missing_symbols"] == ["ZZZZ"] and table["missing_weight"] == pytest.approx(.2)


def test_holdings_quality_gap_is_shown_in_the_table(recorded):
    """2026-10-23 live run: TLT's snapshot failed three quality checks; the table must say so."""
    from castle_risk.outcome_table import holdings_quality_gaps
    mixed, nvda = recorded
    inputs = {"holdings": {"TLT": {"reliable": False, "metadata": {"quality_issues": [
        "CONFLICTING_DUPLICATE_SECURITY", "NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT"]}},
        "QQQ": {"reliable": True, "metadata": {"quality_issues": []}}}}
    gaps = holdings_quality_gaps(inputs, {"TLT": {"gross_short_weight": -7.66e-05, "reported_weight_sum": 1.0000003}})
    assert list(gaps) == ["TLT"] and gaps["TLT"]["measured"]["gross_short_weight"] == -7.66e-05
    table = _table(mixed, _distribution(nvda, "FED_POLICY"), "FOMC", holdings_quality=gaps, n_draws=2000)
    assert [g["symbol"] for g in table["data_quality_gaps"]] == ["TLT"]
    tlt = next(p for p in table["positions"] if p["symbol"] == "TLT")
    assert tlt["data_quality"]["issues"][0] == "CONFLICTING_DUPLICATE_SECURITY"
