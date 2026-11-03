"""R56: frozen bond-portfolio and BND study using existing R55 rate engine.

The aggregate supports only exact future yield *outcome* scenario exposure,
never a causal prediction of yields or a physical settlement probability.
"""
from __future__ import annotations

from dataclasses import replace
import math

from castle_risk.r55_treasury_ief import Window, model_audit


def same_window_bond_portfolio(bnd: list[Window], ief: list[Window],
                               bnd_weight: float = .6) -> list[Window]:
    if not 0 < bnd_weight < 1 or len(bnd) != len(ief):
        raise ValueError("Invalid bond portfolio weights or unequal windows")
    out = []
    for a, b in zip(bnd, ief):
        if (a.year, a.start, a.end) != (b.year, b.start, b.end):
            raise ValueError("Cross-fund Treasury historical windows differ")
        if abs(a.delta_10_bp - b.delta_10_bp) > 1e-7:
            raise ValueError("Cross-fund Treasury 10Y changes differ")
        if abs(a.delta_7_bp - b.delta_7_bp) > 1e-7:
            raise ValueError("Cross-fund Treasury 7Y changes differ")
        out.append(replace(a,
            price_return_pct=bnd_weight*a.price_return_pct+(1-bnd_weight)*b.price_return_pct,
            adjusted_close_return_pct=bnd_weight*a.adjusted_close_return_pct+
                                      (1-bnd_weight)*b.adjusted_close_return_pct,
        ))
    return out


def economic_gate(study: dict, *, min_gain_pct: float = 30.0) -> dict:
    held = study["heldout"]
    # A 1bp move is 0.0001 in *decimal* yield and the response is recorded
    # in percentage points, so beta (% per bp) = -duration / 100.
    # Omitting this factor of 100 is a dimensional mistake, not a model fail.
    implied_duration = -100.0 * study["slope_return_pct_per_yield_bp"]
    tests = {
        "chronology_25_10_14": (study["n_train"], study["n_cal"], study["n_heldout"])
                                == (25, 10, 14),
        "negative_yield_beta_plausible_duration_2_to_10":
            2.0 <= implied_duration <= 10.0,
        "heldout_vs_zero_at_least_30pct": held["gain_vs_zero_pct"] >= min_gain_pct,
        "heldout_vs_training_mean_at_least_30pct": held["gain_vs_mean_pct"] >= min_gain_pct,
        "beats_zero_after_excluding_best_block": held["beats_zero_after_dropping_best"],
        "beats_mean_after_excluding_best_block": held["beats_mean_after_dropping_best"],
        "heldout_at_least_11_of_14_empirical_marginal_band_hits":
            held["nominal_90pct_coverage_count"] >= 11,
    }
    return {"passed": all(tests.values()), "criteria": tests,
            "slope_implied_duration_years": implied_duration,
            "method": "frozen_45_common_session_OLS_2017_21_cal_2022_23_holdout_2024_26"}


def exact_bond_portfolio_rate_bands(study: dict, source_receipt: dict,
                                     *, bnd: dict, ief: dict,
                                     notional: float = 100000.0) -> list[dict]:
    """Safely transfer the validated 60/40 *portfolio* model to exact yields."""
    if not (study["tenor"] == bnd["tenor"] == ief["tenor"] == "delta_10_bp"):
        raise ValueError("Source yield risk must be identically Treasury 10Y")
    if not (study["outcome"] == bnd["outcome"] == ief["outcome"] == "price_return_pct"):
        raise ValueError("Price basis mismatch")
    if (source_receipt.get("event_id") != "KXNOTE10Y-26DEC31" or
        source_receipt.get("settlement_date") != "2026-12-31" or
        source_receipt.get("treasury_baseline_asof") != "2026-10-27" or
        not math.isclose(source_receipt.get("treasury_baseline_yield_10y_pct",-1),4.99,
                         abs_tol=1e-9)):
        raise ValueError("Source contract settlement or baseline mismatch")
    if not math.isfinite(notional) or notional <= 0:
        raise ValueError("Invalid cash amount")
    if len(source_receipt.get("source_defined_yield_bands", [])) != 9:
        raise ValueError("Source quote ladder was not verified")
    alpha = study["intercept_return_pct"]
    slope = study["slope_return_pct_per_yield_bp"]
    radius = study["calibrated_marginal_abs_residual_halfwidth_pct"]
    if not all(math.isfinite(x) for x in (alpha,slope,radius)) or radius < 0:
        raise ValueError("Invalid portfolio model or uncertainty calibration")
    floor = max(-100., bnd["training_min_delta_bp"], ief["training_min_delta_bp"],
                study["training_min_delta_bp"])
    ceiling = min(100., bnd["training_max_delta_bp"], ief["training_max_delta_bp"],
                  study["training_max_delta_bp"])
    base = 4.99
    out = []
    for index, band in enumerate(source_receipt["source_defined_yield_bands"]):
        interval = band["source_treasury_10y_yield_interval_pct"]
        lo,hi = interval["lower_exclusive"], interval["upper_inclusive"]
        q = band["indicative_reference_Q_bin_mass_interval"]
        if not len(q) == 2 or not 0 <= q[0] <= q[1] <= 1:
            raise ValueError("Malformed market Q interval")
        admitted = lo is not None and hi is not None and lo < hi
        d0,d1 = ((lo-base)*100,(hi-base)*100) if admitted else (None,None)
        admitted = admitted and d0 >= floor-1e-9 and d1 <= ceiling+1e-9
        center,outer = None,None
        if admitted:
            v0,v1=alpha+slope*d0,alpha+slope*d1
            center=[notional*min(v0,v1)/100,notional*max(v0,v1)/100]
            outer=[center[0]-notional*radius/100,center[1]+notional*radius/100]
        out.append({
            "index":index,"yield_interval_pct":interval,
            "source_indicative_Q_not_physical_P":q,
            "full_bond_price_only_conditional_range_supported":bool(admitted),
            "full_60bnd_40ief_portfolio_center_pnl_usd_range":center,
            "full_60bnd_40ief_portfolio_calibrated_pnl_usd_range":outer,
            "model_applicability_delta_10y_bp":[floor,ceiling],
        })
    return out
