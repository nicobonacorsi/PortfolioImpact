"""R93: join real official FOMC rate surprises to exact stock/ETF daily returns.

No ex ante prediction, no binary rate-decision settlement equated to a
high-frequency futures surprise, no causal effect automatically identified.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import math

import numpy as np

from castle_risk.r92_factor_bridge import HOLDINGS, PORTFOLIOS, _finite, _fit_score, _rmse

SCHEMA = "castle.option1.r93_realized_event_holdings.v1"
EARLIEST_EVENT = "2017-01-01"
SPLIT = "2023-01-01"
HISTORICAL_BEGIN = "2016-11-01"
HISTORICAL_END = "2026-10-29"
MAX_EVENT_RETURN = 0.35


def _response_models(
    y: np.ndarray, ed4: np.ndarray, ff1: np.ndarray, train: np.ndarray,
) -> dict:
    def model(*features: np.ndarray) -> tuple[np.ndarray, float, float]:
        fit = np.column_stack((np.ones(int(np.sum(train))),
                               *(f[train] for f in features)))
        test = np.column_stack((np.ones(int(np.sum(~train))),
                                *(f[~train] for f in features)))
        if np.linalg.matrix_rank(fit) != fit.shape[1]:
            raise ValueError("Rank-deficient historical event factor model")
        beta = np.linalg.lstsq(fit, y[train], rcond=None)[0]
        return beta, _rmse(y[~train], test @ beta), _rmse(y[train], fit @ beta)

    b_mean = float(np.mean(y[train]))
    rmean = _rmse(y[~train], np.full(int(np.sum(~train)), b_mean))
    b_ed, r_ed, _ = model(ed4)
    b_ef, r_ef, _ = model(ed4, ff1)
    if not rmean > 0 or not r_ed > 0:
        raise ValueError("Zero residual baseline")
    return {
        "train_mean_return": b_mean,
        "train_ed4_coefficients": b_ed.tolist(),
        "train_ed4_ff1_coefficients": b_ef.tolist(),
        "holdout_mean_rmse": rmean,
        "holdout_ed4_rmse": r_ed,
        "holdout_ed4_ff1_rmse": r_ef,
        "gain_ed4_ff1_vs_mean": 1-r_ef/rmean,
        "gain_ed4_ff1_vs_ed4": 1-r_ef/r_ed,
        "passes_5pct_vs_mean": bool(1-r_ef/rmean >= 0.05),
        "passes_5pct_vs_ed4": bool(1-r_ef/r_ed >= 0.05),
        "last_factor_sign_diagnostic_not_pre_event": bool(
            b_ef[-1]*np.linalg.lstsq(
                np.column_stack((np.ones(int(np.sum(~train))),
                                 ed4[~train], ff1[~train])),
                y[~train], rcond=None,
            )[0][-1] > 0
        ),
    }


def evaluate_historical_event_holdings(
    prices: Mapping[str, Mapping[str, float]],
    portfolios: Mapping[str, Mapping[str, float]],
    monetary_event_rows: Sequence[Mapping],
) -> dict:
    if not isinstance(prices, Mapping) or set(prices) != HOLDINGS:
        raise ValueError("Exact 16 source securities required")
    if not isinstance(portfolios, Mapping) or set(portfolios) != PORTFOLIOS:
        raise ValueError("Four originally frozen portfolios required")
    for name, weights in portfolios.items():
        if (not isinstance(weights, Mapping) or
                not set(weights).issubset(HOLDINGS) or
                abs(math.fsum(_finite(w, "weight") for w in weights.values())-1)>1e-10 or
                any(_finite(w, "weight") <= 0 for w in weights.values())):
            raise ValueError("Bad frozen portfolio weights: "+name)
    if not isinstance(monetary_event_rows, Sequence):
        raise ValueError("USMPD rows must be a sequence")
    days = sorted(set.intersection(*(set(prices[t]) for t in sorted(HOLDINGS))))
    if min(days) > "2017-01-05" or max(days) < "2026-10-28":
        raise ValueError("Insufficient multi-year Nasdaq close history")
    past_day = {day: prior for prior, day in zip(days, days[1:])}
    aligned = []
    seen = set()
    count_off_calendar = count_incomplete = count_unscheduled = 0
    screened = []
    for event in monetary_event_rows:
        d = event.get("Date")
        if not isinstance(d, str):
            raise ValueError("Source event date must be ISO")
        date.fromisoformat(d[:10])
        if d in seen:
            raise ValueError("Duplicate FOMC source event date")
        seen.add(d)
        if d < EARLIEST_EVENT:
            continue
        if event.get("Unscheduled") not in (0, 0.0, 1, 1.0) or isinstance(event.get("Unscheduled"), bool):
            raise ValueError("Bad FOMC unscheduled indicator")
        if event["Unscheduled"] == 1:
            count_unscheduled += 1
            continue
        try:
            factors = tuple(_finite(event[key], key)
                            for key in ("ED4", "FF1", "SP500", "UST10Y"))
        except (ValueError, KeyError):
            count_incomplete += 1
            continue
        if d not in past_day:
            count_off_calendar += 1
            continue
        prior = past_day[d]
        returns = {
            ticker: _finite(prices[ticker][d], "daily Nasdaq close") /
                    _finite(prices[ticker][prior], "previous Nasdaq close") - 1
            for ticker in HOLDINGS
        }
        if any(prices[ticker][d] <= 0 or prices[ticker][prior] <= 0
               for ticker in HOLDINGS):
            raise ValueError("Nonpositive market close")
        if any(abs(r) > MAX_EVENT_RETURN for r in returns.values()):
            screened.append({
                "date": d,
                "extreme_tickers": sorted(t for t, r in returns.items()
                                          if abs(r) > MAX_EVENT_RETURN),
                "threshold_absolute_return": MAX_EVENT_RETURN,
            })
            continue
        aligned.append((d, factors, returns))
    aligned.sort(key=lambda r: r[0])
    if not aligned:
        raise ValueError("No aligned real event/holding pairs")
    training = np.array([row[0] < SPLIT for row in aligned], dtype=bool)
    count_train, count_test = int(np.sum(training)), int(np.sum(~training))
    if count_train < 35 or count_test < 20:
        raise ValueError(f"Event-holdings coverage below gate: {count_train} train, {count_test} OOS")
    vals = np.array([factors for _, factors, _ in aligned])
    ed4, ff1, sp, ust = vals.T
    outputs = {}
    for name, weights in portfolios.items():
        y = np.array([
            math.fsum(w * returns[t] for t, w in weights.items())
            for _, _, returns in aligned
        ])
        main = _response_models(y, ed4, ff1, training)
        observed_bridge = _fit_score(sp.reshape(-1,1), np.column_stack((sp, ust)), y, training)
        main["market_sp500_plus_ust10y_vs_sp500"] = observed_bridge
        placebo = {}
        for lag in (1, 5):
            shifted_ed4 = np.empty(len(aligned))
            shifted_ff1 = np.empty(len(aligned))
            shifted_ed4[training] = np.roll(ed4[training], lag)
            shifted_ed4[~training] = np.roll(ed4[~training], lag)
            shifted_ff1[training] = np.roll(ff1[training], lag)
            shifted_ff1[~training] = np.roll(ff1[~training], lag)
            shifted = _response_models(y, shifted_ed4, shifted_ff1, training)
            placebo[str(lag)] = {
                "wrong_meeting_shift": lag,
                "holdout_ed4_ff1_rmse": shifted["holdout_ed4_ff1_rmse"],
                "improvement_vs_true_aligned_model": (
                    1-main["holdout_ed4_ff1_rmse"]/shifted["holdout_ed4_ff1_rmse"]
                ),
            }
        main["wrong_meeting_placebo"] = placebo
        outputs[name] = main

    instruments = {}
    for ticker in sorted(HOLDINGS):
        y = np.array([returns[ticker] for _, _, returns in aligned])
        instruments[ticker] = _response_models(y, ed4, ff1, training)
    all_rate = all(item["passes_5pct_vs_mean"] for item in outputs.values())
    all_market = all(
        item["market_sp500_plus_ust10y_vs_sp500"]["clears_frozen_5pct_gate"]
        for item in outputs.values()
    )
    return {
        "schema": SCHEMA,
        "status": ("FOUR_PORTFOLIOS_SHOCK_HOLDOUT_PASSED" if all_rate
                   else "HISTORICAL_EVENT_RESPONSE_GATE_NOT_UNIFORMLY_PASSED"),
        "train_count": count_train, "holdout_count": count_test,
        "first_train_event": aligned[0][0],
        "last_test_event": aligned[-1][0],
        "split_date": SPLIT,
        "excluded_unmatched_market_calendar": count_off_calendar,
        "excluded_missing_event_factors": count_incomplete,
        "excluded_unscheduled_events": count_unscheduled,
        "excluded_gt35pct_move_event_days": screened,
        "portfolios": outputs,
        "individual_instruments": instruments,
        "all_four_ed4_ff1_vs_mean_pass_5pct": all_rate,
        "all_four_market_sp500_ust_vs_sp500_pass_5pct": all_market,
        "pnl_metric": "historical_price_only_same_day_rebalanced_simple_return",
        "event_state_conditioning": "observed_FOMC_futures_surprise_not_contract_yes",
        "source_window_mismatch": (
            "Nasdaq stock close-to-close includes earlier day and other news, "
            "whereas SF Fed factors are narrow intraday statement and presser windows."
        ),
        "historical_event_return_pairs_observed": True,
        "pre_event_scenario_point_prediction_validated": False,
        "conditional_Kalshi_yes_portfolio_pnl_identified": False,
        "risk_already_discounted_share_identified": False,
    }
