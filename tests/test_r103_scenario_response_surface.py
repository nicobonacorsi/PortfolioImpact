"""Scope and accounting tests; synthetic data are not economic validation."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import math

import numpy as np
import pytest

from castle_risk.portfolio import parse_portfolio
from castle_risk.scenario_response_surface import (
    attach_scenario_responses, build_response_model, evaluate_factor_shock, MODEL_SCHEMA,
)
from castle_risk.two_factor_response import fit_two_factor


def _portfolio():
    return parse_portfolio([
        {"symbol": "ABC", "weight": .4, "asset_type": "stock"},
        {"symbol": "FUND", "weight": .5, "asset_type": "etf"},
        {"symbol": "USD", "weight": .1, "asset_type": "cash"},
    ], notional=100000)


def _history():
    rng = np.random.default_rng(4103)
    day = datetime(2023, 1, 2)
    dates = []
    while len(dates) < 720:
        if day.weekday() < 5:
            dates.append(day.date().isoformat())
        day += timedelta(days=1)
    spy_returns = rng.normal(0, .008, len(dates))
    rate_changes = rng.normal(0, 3., len(dates))
    fund_returns = .7 * spy_returns - .0004 * rate_changes + rng.normal(0, .0001, len(dates))
    spy = 100 * np.cumprod(1 + spy_returns)
    fund = 100 * np.cumprod(1 + fund_returns)
    yields = 4 + np.cumsum(rate_changes) / 100
    return ([{"date": d, "close": float(v)} for d, v in zip(dates, fund)],
            [{"date": d, "close": float(v)} for d, v in zip(dates, spy)],
            [{"date": d, "yield_pct": float(v)} for d, v in zip(dates, yields)], dates[449])


def _model(missing=False):
    rows = [
        {"symbol": "ABC", "weight": .4, "status": "AVAILABLE", "coefficients":
         {"SPY_RETURN_PCT": .015, "US_10Y_YIELD_CHANGE_BP": -.0001}, "basis": "measured"},
        {"symbol": "FUND", "weight": .5, "status": "AVAILABLE", "coefficients":
         {"SPY_RETURN_PCT": .002, "US_10Y_YIELD_CHANGE_BP": -.0007}, "basis": "measured"},
        {"symbol": "USD", "weight": .1, "status": "AVAILABLE", "coefficients":
         {"SPY_RETURN_PCT": 0., "US_10Y_YIELD_CHANGE_BP": 0.}, "basis": "USD cash price fixed"},
    ]
    if missing:
        rows[1].update(status="UNAVAILABLE", reason="HOLDOUT_FAILED")
    return {"schema": MODEL_SCHEMA, "status": "COMPUTED", "notional": 100000,
            "positions": rows, "horizon": "One trading day", "scales": {
                "SPY_RETURN_PCT": {"value": 1., "basis": "TRAINING_SCALE", "training_span": ["2023-01-01", "2024-10-01"]}}}


def _risk(family="US_PAYROLLS", named=False):
    return {"id": "new-event", "family": family, "outcomes": [], "market_outcomes": [],
            "exposure_basis": "NAMED_SOURCE_CHANNELS" if named else "PORTFOLIO_CONTEXT_ONLY",
            "exposure": [{"symbol": "ABC", "weight": .45}] if named else [],
            "quality": {"definition_review": {"status": "DEFINED_FOR_REVIEW"}}}


def _report(model=None):
    return {"portfolio": _portfolio().to_dict(), "scenario_response_model": model or _model(),
            "evaluated_at_utc": "2026-10-31T19:00:00+00:00"}


def test_factor_shock_reconciles_without_probability_and_no_quote_requirement():
    result = evaluate_factor_shock(_model(), "US_10Y_YIELD_CHANGE_BP", 25)
    assert result["pnl"] == pytest.approx(-975)
    assert result["pnl"] == pytest.approx(sum(r["pnl"] for r in result["position_contributions"]))
    assert result["covered_weight"] == pytest.approx(1)
    assert result["probability"] is None
    risk = _risk()
    attach_scenario_responses([risk], _report())
    assert risk["scenario_response"]["level"] == "B"
    assert risk["outcomes"] == risk["market_outcomes"] == []
    assert risk["scenario_response"]["event_conditional_coverage_increment"] == 0


def test_missing_fit_is_partial_and_unknown_not_zero():
    result = evaluate_factor_shock(_model(missing=True), "SPY_RETURN_PCT", -2.)
    assert result["pnl"] is None
    assert result["known_pnl"] == pytest.approx(-1200)
    assert result["covered_weight"] == pytest.approx(.5)
    assert result["position_contributions"][1]["pnl"] is None


def test_linked_direct_and_fund_slices_conserve_original_nav():
    risk = _risk("COMPANY_CYBER", named=True)
    lookthrough = [{"symbol": "ABC", "direct": .4, "weight": .45, "via": {"FUND": .05},
                    "unresolved": False, "basis": "REVIEWED_LEDGER_DIRECT_AND_LOOK_THROUGH"}]
    attach_scenario_responses([risk], _report(), lookthrough=lookthrough)
    out = risk["scenario_response"]["scenarios"][0]
    assert out["pnl"] is None
    assert out["known_pnl"] == pytest.approx(-450)
    assert out["covered_weight"] == pytest.approx(.45)
    assert [r["weight"] for r in out["position_contributions"]] == [.4, .5, .1]
    assert out["position_contributions"][1]["pnl"] == pytest.approx(-50)
    assert out["position_contributions"][1]["response_weight"] == pytest.approx(.05)
    assert out["scope"] == "LINKED_PRICE_COMPONENT_ONLY"


def test_entire_fund_and_its_constituent_are_not_double_counted():
    risk = _risk("TRADE_TARIFFS", named=True)
    risk["exposure"].append({"symbol": "FUND", "weight": .5})
    lookthrough = [{"symbol": "ABC", "direct": .4, "via": {"FUND": .05},
                    "basis": "REVIEWED_LEDGER_DIRECT_AND_LOOK_THROUGH"}]
    attach_scenario_responses([risk], _report(), lookthrough=lookthrough)
    out = risk["scenario_response"]["scenarios"][0]
    assert out["covered_weight"] == pytest.approx(.9)
    assert out["known_pnl"] == pytest.approx(-900)


@pytest.mark.parametrize("defect", ["unknown_family", "definition", "identity", "source_link_absent"])
def test_unresolved_channels_fail_closed(defect):
    risk = _risk("NEW_FAMILY")
    if defect == "definition":
        risk = _risk()
        risk["quality"]["definition_review"]["status"] = "UNRESOLVED"
    if defect == "identity":
        risk = _risk()
        risk["quality"]["instrument_identity_gate"] = {"blocked_symbols": ["ABC"]}
    if defect == "source_link_absent":
        risk["exposure"] = [{"symbol": "ABC", "weight": .4}]
    attach_scenario_responses([risk], _report())
    assert risk["scenario_response"]["level"] == "C"
    assert risk["scenario_response"]["scenarios"] == []


def test_existing_outcomes_and_quotes_are_unchanged():
    risk = _risk()
    risk["outcomes"] = [{"pnl": 22., "q_reference": .3}]
    before = deepcopy(risk)
    attach_scenario_responses([risk], _report())
    assert {k: risk[k] for k in before} == before
    assert not risk["scenario_response"]["forecast_validated"]


def test_two_factor_regular_data_pass_and_future_fit_is_rejected():
    fund, spy, yields, split = _history()
    fit = fit_two_factor(fund, spy, yields, train_end=split)
    assert fit["admitted"]
    assert fit["holdout_oos_r2"] > .99
    portfolio = parse_portfolio([{"symbol": "ABC", "weight": 1., "asset_type": "stock"}])
    model = build_response_model(portfolio, prices_by_symbol={"SPY": spy, "ABC": fund},
        treasury_yields=yields, two_factor_responses={"ABC": fit},
        evaluated_at=datetime(2026, 10, 31, tzinfo=timezone.utc), train_end=split)
    assert model["positions"][0]["status"] == "AVAILABLE"
    assert all(v["training_span"][1] <= split for v in model["scales"].values())
    fit["holdout_span"][1] = "2027-01-01"
    model = build_response_model(portfolio, prices_by_symbol={"SPY": spy}, treasury_yields=yields,
        two_factor_responses={"ABC": fit}, evaluated_at=datetime(2026, 10, 31, tzinfo=timezone.utc), train_end=split)
    assert model["positions"][0]["status"] == "UNAVAILABLE"


def test_holdout_values_do_not_change_training_scales():
    fund, spy, yields, split = _history()
    portfolio = _portfolio()
    kw = dict(treasury_yields=yields, two_factor_responses={}, evaluated_at=datetime(2026, 10, 31, tzinfo=timezone.utc), train_end=split)
    a = build_response_model(portfolio, prices_by_symbol={"SPY": spy}, **kw)
    changed = [{**r, "close": r["close"] * 10 if r["date"] > split else r["close"]} for r in spy]
    b = build_response_model(portfolio, prices_by_symbol={"SPY": changed}, **kw)
    assert a["scales"] == b["scales"]


@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "constant", "misaligned"])
def test_two_factor_bad_inputs_are_diagnostics_not_run_crashes(bad):
    fund, spy, yields, split = _history()
    if bad == "duplicate":
        fund.append(fund[5])
    elif bad == "nonfinite":
        fund[12]["close"] = math.nan
    elif bad == "constant":
        yields = [{**r, "yield_pct": 4.} for r in yields]
    else:
        fund = fund[::2]
    result = fit_two_factor(fund, spy, yields, train_end=split)
    assert not result["admitted"]
    assert result["status"] in {"INVALID_SOURCE_SERIES", "RANK_DEFICIENT_DESIGN", "INSUFFICIENT_HISTORY"}


@pytest.mark.parametrize("value", [float("nan"), True, -101])
def test_invalid_factor_shocks_rejected(value):
    with pytest.raises(ValueError):
        evaluate_factor_shock(_model(), "SPY_RETURN_PCT", value)


def test_cash_zero_alone_does_not_create_a_numeric_response_channel():
    model = _model()
    for row in model["positions"][:2]:
        row.update(status="UNAVAILABLE", reason="HOLDOUT_FAILED")
    risk = _risk()
    attach_scenario_responses([risk], _report(model))
    assert risk["scenario_response"]["level"] == "C"
    assert risk["scenario_response"]["scenarios"] == []


@pytest.mark.parametrize("bad", [None, {"symbol": "ABC", "weight": 1.2}, {"symbol": "ABC", "weight": .4}])
def test_invalid_or_duplicate_named_exposure_is_not_priced(bad):
    risk = _risk("OPEC_SUPPLY", named=True)
    risk["exposure"].append(bad)
    attach_scenario_responses([risk], _report())
    assert risk["scenario_response"]["level"] == "C"
    assert risk["scenario_response"]["missing_channels"]


def test_nonfinite_linear_output_is_unknown_and_serializable():
    result = evaluate_factor_shock(_model(), "SPY_RETURN_PCT", 1e308)
    assert result["pnl"] is None
    assert all(row["pnl"] is None or math.isfinite(row["pnl"]) for row in result["position_contributions"])


def test_finite_components_cannot_overflow_the_scenario_aggregate():
    model = _model()
    model["notional"] = 1e308
    model["positions"] = [{"symbol": symbol, "weight": .5, "status": "AVAILABLE",
                            "coefficients": {"SPY_RETURN_PCT": .01}, "basis": "synthetic"}
                           for symbol in ("ONE", "TWO")]
    with pytest.raises(ValueError, match="aggregate"):
        evaluate_factor_shock(model, "SPY_RETURN_PCT", 200.)
