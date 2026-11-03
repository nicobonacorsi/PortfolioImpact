"""R57 frozen-grid BND/IEF 45-session price response with *paired* uncertainty.

The weights are initial portfolio-value weights; outcomes are conditional on
an EXOGENOUS scenario Treasury 10Y change, never its forecast. This code is
pure arithmetic with no provider calls or hidden data replacement.
"""
from __future__ import annotations

import math

from castle_risk.r55_treasury_ief import Window, model_audit
from castle_risk.r56_bnd_portfolio import economic_gate, same_window_bond_portfolio


WEIGHTS = tuple(k / 20 for k in range(21))
TRAIN_YEARS = tuple(range(2017, 2022))
CAL_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2025, 2026)


def _paired_windows(bnd: list[Window], ief: list[Window]) -> None:
    if len(bnd) != len(ief):
        raise ValueError("BND/IEF chronological window lengths differ")
    for a, b in zip(bnd, ief):
        if (a.year, a.start, a.end) != (b.year, b.start, b.end):
            raise ValueError("BND/IEF chronological window endpoints differ")
        if abs(a.delta_10_bp - b.delta_10_bp) > 1e-7 or abs(a.delta_7_bp - b.delta_7_bp) > 1e-7:
            raise ValueError("BND/IEF yield factors are not aligned")
    count = (sum(x.year in TRAIN_YEARS for x in bnd),
             sum(x.year in CAL_YEARS for x in bnd),
             sum(x.year in TEST_YEARS for x in bnd))
    if count != (25, 10, 14):
        raise ValueError(f"Different R56 chronological split: {count}")
    if any(x.year not in (*TRAIN_YEARS, *CAL_YEARS, *TEST_YEARS) for x in bnd):
        raise ValueError("Unexpected year")


def paired_residuals(bnd: list[Window], ief: list[Window],
                     bnd_model: dict, ief_model: dict, *, years=CAL_YEARS) -> list[dict]:
    """Return dated same-window paired price residuals in *percentage points*."""
    _paired_windows(bnd, ief)
    result = []
    for a, b in zip(bnd, ief):
        if a.year not in years:
            continue
        x = a.delta_10_bp
        result.append({
            "window_end": a.end.isoformat(),
            "BND_residual_pct": a.price_return_pct - bnd_model["intercept_return_pct"] -
                                bnd_model["slope_return_pct_per_yield_bp"] * x,
            "IEF_residual_pct": b.price_return_pct - ief_model["intercept_return_pct"] -
                                ief_model["slope_return_pct_per_yield_bp"] * x,
        })
    return result


def joint_marginal_radius(cal_residuals: list[dict], bnd_weight: float) -> float:
    """With exactly ten calibration blocks, rank-10 split residual quantile.

    The abs is taken AFTER combining the two errors on the same historical
    window. Combining separate marginal intervals would be invalid.
    """
    if len(cal_residuals) != 10 or not math.isfinite(bnd_weight) or not 0 <= bnd_weight <= 1:
        raise ValueError("Need ten paired calibration residuals and a feasible weight")
    values = []
    for row in cal_residuals:
        a, b = row["BND_residual_pct"], row["IEF_residual_pct"]
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Nonfinite paired calibration residual")
        values.append(abs(bnd_weight * a + (1 - bnd_weight) * b))
    return max(values)


def _model_at(bnd: list[Window], ief: list[Window], w: float) -> dict:
    if w == 0:
        return model_audit(ief)
    if w == 1:
        return model_audit(bnd)
    return model_audit(same_window_bond_portfolio(bnd, ief, w))


def yield_bands_at_weight(model: dict, source_receipt: dict,
                          component_models: tuple[dict, dict],
                          *, notional_usd: float = 100000) -> list[dict]:
    if model["tenor"] != "delta_10_bp" or model["outcome"] != "price_return_pct":
        raise ValueError("Model cannot be transferred to different yield or price basis")
    if (source_receipt.get("event_id") != "KXNOTE10Y-26DEC31" or
        source_receipt.get("settlement_date") != "2026-12-31" or
        source_receipt.get("treasury_baseline_asof") != "2026-10-27" or
        source_receipt.get("treasury_baseline_yield_10y_pct") != 4.99):
        raise ValueError("Archived Kalshi/official Treasury source definition mismatch")
    if not math.isfinite(notional_usd) or notional_usd <= 0:
        raise ValueError("Nonpositive notional")
    bnd, ief = component_models
    low = max(-100., model["training_min_delta_bp"],
              bnd["training_min_delta_bp"], ief["training_min_delta_bp"])
    high = min(100., model["training_max_delta_bp"],
               bnd["training_max_delta_bp"], ief["training_max_delta_bp"])
    radius = model["calibrated_marginal_abs_residual_halfwidth_pct"]
    alpha, beta = model["intercept_return_pct"], model["slope_return_pct_per_yield_bp"]
    if not all(map(math.isfinite, (low, high, radius, alpha, beta))) or low >= high or radius < 0:
        raise ValueError("Invalid support or coefficient")
    source_bands = source_receipt.get("source_defined_yield_bands", [])
    if len(source_bands) != 9:
        raise ValueError("Exact nine source-defined Kalshi bins are required")
    out = []
    for i, source in enumerate(source_bands):
        interval = source["source_treasury_10y_yield_interval_pct"]
        lo, hi = interval["lower_exclusive"], interval["upper_inclusive"]
        q = source["indicative_reference_Q_bin_mass_interval"]
        if len(q) != 2 or any(not math.isfinite(v) for v in q) or not 0 <= q[0] <= q[1] <= 1:
            raise ValueError("Malformed source quote")
        bounded = (lo is not None and hi is not None and
                   math.isfinite(lo) and math.isfinite(hi) and lo < hi)
        deltas = [100 * (lo - 4.99), 100 * (hi - 4.99)] if bounded else []
        supported = bool(bounded and deltas[0] >= low - 1e-9 and deltas[1] <= high + 1e-9)
        fitted = None
        outer = None
        if supported:
            endpoints = [(alpha + beta * x) * notional_usd / 100 for x in deltas]
            fitted = [min(endpoints), max(endpoints)]
            outer = [fitted[0] - radius * notional_usd / 100,
                     fitted[1] + radius * notional_usd / 100]
        out.append({
            "index": i,
            "yield_interval_pct": interval,
            "indicative_reference_Q_not_physical_P": q,
            "supported_without_extrapolation": supported,
            "fitted_price_pnl_usd_range_for_100k": fitted,
            "marginal_calibrated_price_pnl_usd_range_for_100k": outer,
            "joint_training_support_delta_10y_bp": [low, high],
        })
    return out


def evaluate_frozen_grid(bnd: list[Window], ief: list[Window],
                         source_receipt: dict) -> dict:
    _paired_windows(bnd, ief)
    bnd_model, ief_model = model_audit(bnd), model_audit(ief)
    paired = paired_residuals(bnd, ief, bnd_model, ief_model)
    results = []
    for index, w in enumerate(WEIGHTS):
        fit = _model_at(bnd, ief, w)
        alpha = w * bnd_model["intercept_return_pct"] + (1-w)*ief_model["intercept_return_pct"]
        beta = w * bnd_model["slope_return_pct_per_yield_bp"] + (1-w)*ief_model["slope_return_pct_per_yield_bp"]
        q = joint_marginal_radius(paired, w)
        if any(abs(a-b) > 1e-9 for a,b in (
            (alpha,fit["intercept_return_pct"]),
            (beta,fit["slope_return_pct_per_yield_bp"]),
            (q,fit["calibrated_marginal_abs_residual_halfwidth_pct"]))):
            raise ValueError("R57 weighted regression or joint uncertainty identity failed")
        gate = economic_gate(fit)
        results.append({
            "grid_index": index,
            "BND_weight": w, "IEF_weight": 1-w,
            "intercept_price_return_pct": alpha,
            "beta_price_return_pct_per_10y_bp": beta,
            "joint_nominal_90pct_marginal_radius_pct": q,
            "heldout": fit["heldout"],
            "economic_gate": gate,
            "promoted_for_conditional_scenarios": gate["passed"],
            "source_exact_yield_band_results": yield_bands_at_weight(
                fit, source_receipt, (bnd_model,ief_model)) if gate["passed"] else [],
        })
    if len(results) != 21 or len(paired) != 10:
        raise ValueError("R57 fixed-weight grid count changed")
    return {
        "BND": bnd_model, "IEF": ief_model,
        "paired_calibration_residuals_pct": paired,
        "weight_grid": results,
        "all_21_grid_weights_pass_economic_gate": all(r["promoted_for_conditional_scenarios"] for r in results),
        "passed_weight_count": sum(r["promoted_for_conditional_scenarios"] for r in results),
    }
