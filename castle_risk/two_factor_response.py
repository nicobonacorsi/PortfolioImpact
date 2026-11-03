"""Measured two-factor response for funds without a holdings-based channel.

A fund whose class or duration cannot be established from its holdings file
(a gold trust, a core bond fund with TBA rolls, a REIT or high-yield fund) is
still priced by its own history: daily returns are regressed on the S&P 500
(SPY) return and the change in the 10-year Treasury yield, both factors the
macro outcome model already moves. Loadings are fitted on the training period
and admitted only if they beat a zero forecast on the untouched holdout (out-of-
sample R2 of at least 0.02) and on the FOMC/CPI announcement days in it.

Documented retrospective evidence (R85 study, 19 funds, loadings fitted 2022-2023,
evaluated on 56 FOMC and CPI announcement days 2024-01 to 2026-10 with the
realized factor moves): the two-factor prediction beat zero for 19 of 19 funds
and the SPY-only beta for 18 of 19. Pooled RMSE 0.969% against 1.126% (zero)
and 1.018% (SPY only); AGG 0.096% against 0.426% (zero).
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "castle.option1.two_factor_response.v1"
MIN_TRAIN = 250
MIN_HOLDOUT = 120
MIN_EVENT_DAYS = 8
# This operational cutoff was chosen after exploratory fund diagnostics in R85.
# It MUST NOT be described as an independently pre-registered admission rule.
MIN_OOS_R2 = 0.02


def _changes(rows, field, *, price):
    """Dated intervals, not merely endpoints; reject malformed source series."""
    levels = {}
    for row in rows or []:
        if not isinstance(row, Mapping) or not isinstance(row.get("date"), str):
            raise ValueError("Malformed dated observation")
        day = date.fromisoformat(row["date"]).isoformat()
        value = row.get(field)
        if (day in levels or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (price and value <= 0)):
            raise ValueError("Duplicate, missing or invalid source observation")
        levels[day] = float(value)
    days = sorted(levels)
    changes = {}
    for a, b in zip(days, days[1:]):
        value = levels[b] / levels[a] - 1.0 if price else (levels[b] - levels[a]) * 100.
        if not math.isfinite(value):
            raise ValueError("Source interval has a nonfinite change")
        changes[b] = (a, value)
    return changes


def _returns(prices: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {b: value for b, (_, value) in _changes(prices, "close", price=True).items()}


def _yield_changes(yields: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {b: value for b, (_, value) in _changes(yields, "yield_pct", price=False).items()}


def fit_two_factor(fund_prices, benchmark_prices, yields, *, train_end: str,
                   release_dates: Sequence[str] = ()) -> dict[str, Any]:
    """OLS of daily fund return on SPY return and 10Y yield change (bp), HC3 covariance."""
    out = {"schema": SCHEMA, "train_end": train_end,
           "factors": ["SPY_RETURN_FRACTION", "US_10Y_YIELD_CHANGE_BP"], "admitted": False}
    try:
        date.fromisoformat(train_end)
        f = _changes(fund_prices, "close", price=True)
        s = _changes(benchmark_prices, "close", price=True)
        d = _changes(yields, "yield_pct", price=False)
    except (ValueError, TypeError, KeyError) as exc:
        return {**out, "status": "INVALID_SOURCE_SERIES", "reason": str(exc)}
    common = sorted(set(f) & set(s) & set(d))
    days = [day for day in common if f[day][0] == s[day][0] == d[day][0]
            and (date.fromisoformat(day) - date.fromisoformat(f[day][0])).days <= 4]
    fund = {day: f[day][1] for day in days}
    spy = {day: s[day][1] for day in days}
    dy = {day: d[day][1] for day in days}
    train = [d for d in days if d <= train_end]
    hold = [d for d in days if d > train_end]
    out.update(n_train=len(train), n_holdout=len(hold),
               excluded_misaligned_or_gapped_intervals=len(common)-len(days))
    if len(train) < MIN_TRAIN or len(hold) < MIN_HOLDOUT:
        out.update(status="INSUFFICIENT_HISTORY", reason="Too few aligned fund, SPY and 10Y yield observations.")
        return out
    X = np.array([[1.0, spy[d], dy[d]] for d in train])
    y = np.array([fund[d] for d in train])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        return {**out, "status": "INVALID_SOURCE_SERIES", "reason": "Nonfinite aligned changes."}
    std = np.std(X[:, 1:], axis=0)
    if np.any(std <= 0):
        return {**out, "status": "RANK_DEFICIENT_DESIGN", "reason": "A training factor has no variation."}
    scaled = np.column_stack((np.ones(len(X)), (X[:, 1:] - X[:, 1:].mean(axis=0))/std))
    if np.linalg.matrix_rank(scaled) != 3 or np.linalg.cond(scaled) > 1e10:
        return {**out, "status": "RANK_DEFICIENT_DESIGN", "reason": "Training factors do not identify separate loadings."}
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {**out, "status": "RANK_DEFICIENT_DESIGN", "reason": "Training design could not be inverted."}
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    if np.any(leverage >= 1 - 1e-10):
        return {**out, "status": "UNSTABLE_COVARIANCE", "reason": "HC3 covariance is undefined at unit leverage."}
    meat = (X * (resid / (1.0 - leverage))[:, None] ** 2).T @ X
    cov = xtx_inv @ meat @ xtx_inv
    if not np.isfinite(cov).all() or np.any(np.diag(cov) < -1e-12):
        return {**out, "status": "UNSTABLE_COVARIANCE", "reason": "Coefficient covariance is not finite and nonnegative."}
    slopes = beta[1:]

    def rmse(ds, model):
        actual = np.array([fund[d] for d in ds])
        pred = np.array([slopes[0] * spy[d] + slopes[1] * dy[d] for d in ds]) if model else 0.0
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    hold_model, hold_zero = rmse(hold, True), rmse(hold, False)
    events = sorted(set(release_dates) & set(hold))
    event_model = rmse(events, True) if events else None
    event_zero = rmse(events, False) if events else None
    oos_r2 = 1.0 - (hold_model / hold_zero) ** 2 if hold_zero > 0 else None
    event_ok = len(events) < MIN_EVENT_DAYS or (event_model is not None and event_model <= event_zero)
    admitted = bool(oos_r2 is not None and oos_r2 >= MIN_OOS_R2 and event_ok)
    out.update(
        status="ADMITTED_MEASURED_TWO_FACTOR" if admitted else "NOT_ADMITTED_HOLDOUT_FAILED",
        admitted=admitted,
        spy_loading=float(slopes[0]), rate_loading_per_bp=float(slopes[1]),
        covariance=[[float(cov[1, 1]), float(cov[1, 2])], [float(cov[2, 1]), float(cov[2, 2])]],
        spy_loading_se=math.sqrt(max(0., float(cov[1, 1]))), rate_loading_se_per_bp=math.sqrt(max(0., float(cov[2, 2]))),
        intercept=float(beta[0]), training_span=[train[0], train[-1]], holdout_span=[hold[0], hold[-1]],
        holdout_rmse=hold_model, holdout_zero_rmse=hold_zero, holdout_oos_r2=oos_r2,
        holdout_event_days=len(events), holdout_event_rmse=event_model, holdout_event_zero_rmse=event_zero,
        admission_rule=(f"Holdout out-of-sample R2 of at least {MIN_OOS_R2}, and holdout FOMC/CPI announcement-day RMSE no "
                        f"larger than a zero forecast when at least {MIN_EVENT_DAYS} such days exist."),
        scope=("Daily co-movement with the two factors the macro model moves. It is not a holdings-based class, "
               "duration or mandate, and it assumes the daily loadings hold on the announcement day."),
        price_basis="Split-adjusted closes; dividends are not added back.")
    return out


def two_factor_fits(symbols, prices_by_symbol, yields, *, benchmark: str, train_end: str,
                    release_dates: Sequence[str] = ()) -> dict[str, dict]:
    bench = (prices_by_symbol or {}).get(benchmark)
    if not bench or not yields:
        return {}
    return {s: fit_two_factor(prices_by_symbol[s], bench, yields, train_end=train_end, release_dates=release_dates)
            for s in symbols if s != benchmark and (prices_by_symbol or {}).get(s)}
