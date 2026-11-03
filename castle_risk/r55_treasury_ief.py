"""Auditable, horizon-matched Treasury 10Y -> IEF conditional price response.

Pure arithmetic: never requests provider access, never creates probabilities,
and never turns a partial IEF sleeve into an entire-portfolio prediction.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Window:
    year: int
    start: date
    end: date
    delta_10_bp: float
    delta_7_bp: float
    price_return_pct: float
    adjusted_close_return_pct: float


def digest(body: bytes) -> str:
    return sha256(body).hexdigest()


def read_treasury_year(body: bytes, year: int) -> dict[date, tuple[float, float]]:
    """Official year CSV, columns 7 Yr and 10 Yr, ascending or descending."""
    table = csv.DictReader(StringIO(body.decode("utf-8-sig")))
    if not {"Date", "7 Yr", "10 Yr"}.issubset(table.fieldnames or []):
        raise ValueError("Treasury curve header is missing required tenors")
    result: dict[date, tuple[float, float]] = {}
    for row in table:
        day = datetime.strptime(row["Date"], "%m/%d/%Y").date()
        if day.year != year or day in result:
            raise ValueError("Wrong-year or duplicate Treasury date")
        values = (float(row["7 Yr"]), float(row["10 Yr"]))
        if not all(math.isfinite(v) and 0.0 < v < 30.0 for v in values):
            raise ValueError("Invalid Treasury par yield")
        result[day] = values
    if not 150 <= len(result) <= 270:
        raise ValueError("Inadequate official Treasury annual observation count")
    return result


def read_yahoo_ief(body: bytes) -> dict[date, tuple[float, float]]:
    """Read split-adjusted daily cash closes and Yahoo adjusted-close proxy."""
    data = json.loads(body)["chart"]["result"]
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("Expected one Yahoo Finance IEF result")
    data = data[0]
    times = data["timestamp"]
    closes = data["indicators"]["quote"][0]["close"]
    adj = data["indicators"]["adjclose"][0]["adjclose"]
    if not len(times) == len(closes) == len(adj):
        raise ValueError("Inconsistent price time-series shapes")
    result: dict[date, tuple[float, float]] = {}
    for ts, close, adjusted in zip(times, closes, adj):
        if close is None or adjusted is None:
            continue
        day = datetime.fromtimestamp(int(ts), timezone.utc).date()
        vals = (float(close), float(adjusted))
        if day in result or not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError("Duplicate or nonpositive IEF market-price observation")
        result[day] = vals
    if len(result) < 500:
        raise ValueError("Insufficient IEF price history")
    return result


def join_windows(
    annual_treasury: Mapping[int, Mapping[date, tuple[float, float]]],
    older_prices: Mapping[date, tuple[float, float]],
    newer_prices: Mapping[date, tuple[float, float]],
    horizon: int,
) -> tuple[list[Window], dict]:
    if horizon not in (1, 10, 20, 45):
        raise ValueError("Unexpected unregistered horizon")
    overlap = set(older_prices) & set(newer_prices)
    if len(overlap) < 5:
        raise ValueError("Market source overlap cannot be verified")
    for day in overlap:
        a, b = older_prices[day], newer_prices[day]
        if abs(a[0] - b[0]) > 1e-5 or abs(a[1] - b[1]) > .001:
            raise ValueError("Inconsistent overlapping Yahoo price responses")
    combined = {k: v for k, v in older_prices.items() if k.year <= 2021}
    combined.update({k: v for k, v in newer_prices.items() if k.year >= 2022})
    years = sorted(annual_treasury)
    if years != list(range(2017, 2027)):
        raise ValueError("Annual source coverage 2017-2026 required")
    rows = []
    n_joint = 0
    for year in years:
        common = sorted(set(annual_treasury[year]) & set(combined))
        n_joint += len(common)
        for j in range(0, len(common) - horizon, horizon + 1):
            start, end = common[j], common[j + horizon]
            start_y7, start_y10 = annual_treasury[year][start]
            end_y7, end_y10 = annual_treasury[year][end]
            start_close, start_adjusted = combined[start]
            end_close, end_adjusted = combined[end]
            rows.append(Window(
                year=year, start=start, end=end,
                delta_10_bp=100 * (end_y10 - start_y10),
                delta_7_bp=100 * (end_y7 - start_y7),
                price_return_pct=100 * (end_close / start_close - 1),
                adjusted_close_return_pct=100 * (end_adjusted / start_adjusted - 1),
            ))
    return rows, {"joint_cash_sessions": n_joint, "overlap_verified_sessions": len(overlap)}


def model_audit(
    windows: list[Window], *, tenor: str = "delta_10_bp",
    outcome: str = "price_return_pct", calibration: float = .90,
) -> dict:
    if tenor not in ("delta_10_bp", "delta_7_bp"):
        raise ValueError("Unregistered Treasury tenor")
    if outcome not in ("price_return_pct", "adjusted_close_return_pct"):
        raise ValueError("Unregistered price basis")
    train = [r for r in windows if r.year <= 2021]
    cal = [r for r in windows if r.year in (2022, 2023)]
    test = [r for r in windows if r.year >= 2024]
    if not (len(train) >= 25 and len(cal) >= 10 and len(test) >= 14):
        raise ValueError("One or more required chronology windows missing")
    x = np.array([getattr(w, tenor) for w in train], dtype=float)
    y = np.array([getattr(w, outcome) for w in train], dtype=float)
    if np.std(x) < 1:
        raise ValueError("Treasury training variation inadequate")
    beta, alpha = np.polyfit(x, y, 1)
    yc = np.array([getattr(w, outcome) for w in cal])
    xc = np.array([getattr(w, tenor) for w in cal])
    residual = np.sort(np.abs(yc - alpha - beta * xc))
    rank = math.ceil((len(cal) + 1) * calibration)
    q = float(residual[rank - 1]) if rank <= len(cal) else None
    assert q is not None  # This protocol requires 10 independent calibration blocks.

    def evaluate(cohort: list[Window]) -> dict:
        yt = np.array([getattr(w, outcome) for w in cohort])
        xt = np.array([getattr(w, tenor) for w in cohort])
        p = alpha + beta * xt
        loss_model = np.square(yt - p)
        loss_zero = np.square(yt)
        loss_mean = np.square(yt - y.mean())
        rmse = lambda v: float(np.sqrt(np.mean(v)))
        # Drop the single event most favorable to the chosen model versus
        # each baseline separately; do not tune coefficients afterward.
        def leave_best(baseline: np.ndarray) -> bool:
            if len(cohort) < 3:
                return False
            advantage = baseline - loss_model
            keep = np.ones(len(cohort), dtype=bool)
            keep[int(np.argmax(advantage))] = False
            return rmse(loss_model[keep]) < rmse(baseline[keep])
        return {
            "n": len(cohort),
            "rmse_model_pct": rmse(loss_model),
            "rmse_zero_pct": rmse(loss_zero),
            "rmse_frozen_training_mean_pct": rmse(loss_mean),
            "gain_vs_zero_pct": 100 * (1 - rmse(loss_model) / rmse(loss_zero)),
            "gain_vs_mean_pct": 100 * (1 - rmse(loss_model) / rmse(loss_mean)),
            "nominal_90pct_coverage_count": int(np.sum(np.abs(yt - p) <= q)),
            "nominal_90pct_coverage_ratio": float(np.mean(np.abs(yt - p) <= q)),
            "beats_zero_after_dropping_best": leave_best(loss_zero),
            "beats_mean_after_dropping_best": leave_best(loss_mean),
        }

    return {
        "trained_years": [2017, 2018, 2019, 2020, 2021],
        "calibration_years": [2022, 2023],
        "heldout_years": [2024, 2025, 2026],
        "n_train": len(train), "n_cal": len(cal), "n_heldout": len(test),
        "tenor": tenor, "outcome": outcome,
        "intercept_return_pct": float(alpha),
        "slope_return_pct_per_yield_bp": float(beta),
        "training_min_delta_bp": float(min(x)),
        "training_max_delta_bp": float(max(x)),
        "calibrated_marginal_abs_residual_halfwidth_pct": q,
        "calibration_rank": rank,
        "heldout": evaluate(test),
        "holdout_by_year": {
            str(year): evaluate([w for w in test if w.year == year])
            for year in (2024, 2025, 2026)
        },
        "market_yield_change_is_observed_or_scenario_assumed": True,
        "future_yield_prediction": False,
        "causal_prediction": False,
    }


def admitted_yield_scenarios(*, study: dict, source_receipt: dict,
                             notional: float, ief_weight: float) -> list[dict]:
    """Convert *identical* 10Y predicates to identified-sleeve model ranges."""
    if study["tenor"] != "delta_10_bp" or study["outcome"] != "price_return_pct":
        raise ValueError("Source contract must match model tenors")
    if source_receipt.get("event_id") != "KXNOTE10Y-26DEC31":
        raise ValueError("Different source contract")
    if source_receipt.get("settlement_date") != "2026-12-31":
        raise ValueError("Different market settlement")
    if source_receipt.get("treasury_baseline_asof") != "2026-10-27":
        raise ValueError("Different original source-rate baseline")
    if not math.isfinite(notional) or notional <= 0 or not 0 <= ief_weight <= 1:
        raise ValueError("Invalid original instrument exposure")
    baseline = source_receipt["treasury_baseline_yield_10y_pct"]
    if not math.isclose(baseline, 4.99, abs_tol=1e-9):
        raise ValueError("Source yield baseline mismatch")
    slope = study["slope_return_pct_per_yield_bp"]
    alpha = study["intercept_return_pct"]
    radius = study["calibrated_marginal_abs_residual_halfwidth_pct"]
    floor = max(-100., study["training_min_delta_bp"])
    ceiling = min(100., study["training_max_delta_bp"])
    bands = source_receipt.get("source_defined_yield_bands", [])
    if len(bands) != 9 or not all(math.isfinite(v) for v in (alpha, slope, radius)):
        raise ValueError("Missing exact source distribution or invalid model")
    out = []
    for i, band in enumerate(bands):
        interval = band["source_treasury_10y_yield_interval_pct"]
        lo, hi = interval["lower_exclusive"], interval["upper_inclusive"]
        qbin = band["indicative_reference_Q_bin_mass_interval"]
        if not (len(qbin) == 2 and 0 <= qbin[0] <= qbin[1] <= 1):
            raise ValueError("Invalid indicative source market weight")
        admitted = lo is not None and hi is not None and lo < hi
        d0, d1 = (100 * (lo - baseline), 100 * (hi - baseline)) if admitted else (None, None)
        admitted = admitted and d0 >= floor - 1e-9 and d1 <= ceiling + 1e-9
        row = {
            "index": i, "yield_interval_pct": {"lower_exclusive": lo, "upper_inclusive": hi},
            "indicative_reference_market_Q_interval_not_physical_P": qbin,
            "full_ief_band_supported_without_extrapolation": bool(admitted),
            "source_matched_settlement": True,
            "historical_yield_change_support_bp": [floor, ceiling],
            "ief_sleeve_price_only_pnl_usd_range": None,
            "ief_sleeve_model_center_pnl_usd_range": None,
        }
        if admitted:
            points = [alpha + slope * d0, alpha + slope * d1]
            fraction = notional * ief_weight / 100
            row["ief_sleeve_model_center_pnl_usd_range"] = [
                float(min(points) * fraction), float(max(points) * fraction)]
            row["ief_sleeve_price_only_pnl_usd_range"] = [
                float((min(points) - radius) * fraction),
                float((max(points) + radius) * fraction),
            ]
        out.append(row)
    return out
