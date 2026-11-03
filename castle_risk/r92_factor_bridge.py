"""R92: frozen, chronological public-source factor-response audits.

Two distinct levels of evidence:
  1. Daily SPY and Treasury-factor accounting for exact stock/ETF holdings.
  2. High-frequency observed FOMC surprises associated with two market moves.

Neither observed contemporaneous response is the pre-event P&L conditional
on settlement of a prediction-market YES binary. No source is silently
promoted to causal evidence, event-contingent stock pricing, or a claim
about an exact fraction of current stock valuations already discounted.
"""
from __future__ import annotations

from datetime import date
import math
from collections.abc import Mapping, Sequence

import numpy as np

SCHEMA = "castle.option1.r92_factor_bridge.v1"
STOCKS = frozenset({"CAT", "DAL", "LMT", "MSFT", "NEE", "TSM", "XOM"})
ETFS = frozenset({"AGG", "GLD", "IEF", "JETS", "QQQ", "SPY", "TLT", "USO", "VWO"})
HOLDINGS = STOCKS | ETFS
PORTFOLIOS = frozenset({
    "energy_taiwan_recorded", "new_diversified",
    "new_energy_transport", "new_industrial_diversified",
})
SOURCE_BEGIN = "2026-04-01"
SOURCE_END = "2026-10-29"
HOLDOUT_START = "2026-09-01"
FOMC_SPLIT = "2020-01-01"
FOMC_BEGIN = "2011-01-01"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} requires a finite number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} requires a finite number")
    return val


def _rmse(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - predicted)**2)))


def _fit_score(
    x_baseline: np.ndarray, x_extended: np.ndarray,
    y: np.ndarray, train: np.ndarray, *,
    minimum_gain: float = 0.05,
) -> dict:
    """Train only on past observations; holdout slope is diagnostic only."""
    if x_baseline.ndim != 2 or x_extended.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected two feature matrices and a response vector")
    if len(x_baseline) != len(x_extended) or len(y) != len(train):
        raise ValueError("Mismatched train/test arrays")
    if len(y) != len(x_baseline):
        raise ValueError("Mismatched samples")
    if x_extended.shape[1] != x_baseline.shape[1] + 1:
        raise ValueError("Extended matrix must add one factor")
    if not np.isfinite(x_extended).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite model data")
    split = ~train
    if int(np.sum(train)) < 30 or int(np.sum(split)) < 15:
        raise ValueError("Insufficient data for chronological fitting")

    def design(x: np.ndarray, sample: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(int(np.sum(sample))), x[sample]))

    xb, xe = design(x_baseline, train), design(x_extended, train)
    if np.linalg.matrix_rank(xb) != xb.shape[1] or np.linalg.matrix_rank(xe) != xe.shape[1]:
        raise ValueError("Rank-deficient training design")
    fit_b = np.linalg.lstsq(xb, y[train], rcond=None)[0]
    fit_e = np.linalg.lstsq(xe, y[train], rcond=None)[0]
    predb = design(x_baseline, split) @ fit_b
    prede = design(x_extended, split) @ fit_e
    meanpred = np.full(int(np.sum(split)), float(np.mean(y[train])))
    rb, re, rmean = (_rmse(y[split], pred)
                     for pred in (predb, prede, meanpred))
    if rb <= 0:
        raise ValueError("Degenerate baseline holdout error")
    diag_e = np.linalg.lstsq(design(x_extended, split), y[split], rcond=None)[0]
    # Scale the non-intercept training features solely to diagnose numerical
    # conditioning; coefficients and predictions use unscaled source units.
    stds = np.std(x_extended[train], axis=0)
    if np.any(stds == 0):
        raise ValueError("Constant expanded feature")
    scaled = np.column_stack((
        np.ones(int(np.sum(train))),
        (x_extended[train] - np.mean(x_extended[train], axis=0)) / stds,
    ))
    gain = 1 - re / rb
    return {
        "baseline_holdout_rmse": rb,
        "extended_holdout_rmse": re,
        "train_mean_holdout_rmse": rmean,
        "extended_gain_vs_baseline": gain,
        "clears_frozen_5pct_gate": bool(gain >= minimum_gain),
        "train_baseline_coefficients": fit_b.tolist(),
        "train_extended_coefficients": fit_e.tolist(),
        "holdout_extended_last_factor_slope_diagnostic": float(diag_e[-1]),
        "last_factor_train_holdout_sign_stable": bool(fit_e[-1]*diag_e[-1] > 0),
        "train_scaled_design_condition_number": float(np.linalg.cond(scaled)),
        "train_count": int(np.sum(train)),
        "holdout_count": int(np.sum(split)),
    }


def evaluate_exact_holdings_daily(
    nasdaq_snapshot: Mapping, dgs10_by_day: Mapping[str, float],
) -> dict:
    """Evaluate all 16 exact stocks/ETFs and four portfolios on held-out days.

    Realized market factors are not ex-ante inputs. Period returns are simple
    price changes; the source does not independently certify total returns.
    """
    if not isinstance(nasdaq_snapshot, Mapping) or not isinstance(dgs10_by_day, Mapping):
        raise ValueError("Source snapshots must be mappings")
    spec = nasdaq_snapshot.get("specification", {})
    if spec.get("source_start") != SOURCE_BEGIN or spec.get("source_end") != SOURCE_END:
        raise ValueError("Frozen Nasdaq date window changed")
    price = nasdaq_snapshot.get("closes")
    portfolios = nasdaq_snapshot.get("r87_portfolios")
    if not isinstance(price, Mapping) or set(price) != HOLDINGS:
        raise ValueError("Exact 16-asset holdings universe changed")
    if not isinstance(portfolios, Mapping) or set(portfolios) != PORTFOLIOS:
        raise ValueError("Frozen four-portfolio universe changed")
    for weights in portfolios.values():
        if (not isinstance(weights, Mapping) or
                not set(weights).issubset(HOLDINGS) or
                abs(math.fsum(_finite(w, "portfolio weight") for w in weights.values())-1) > 1e-10 or
                not all(_finite(w, "weight") > 0 for w in weights.values())):
            raise ValueError("Portfolio weights must be original valid unit weights")
    dates = sorted(set.intersection(*(set(price[t]) for t in sorted(HOLDINGS))))
    if not dates or min(dates) != SOURCE_BEGIN or max(dates) != SOURCE_END:
        raise ValueError("Complete original source start/end trading coverage required")
    valid_yield = {}
    for day, value in dgs10_by_day.items():
        date.fromisoformat(day)
        valid_yield[day] = _finite(value, "DGS10 yield in percent")
    pairs = []
    rejected = 0
    for earlier, later in zip(dates, dates[1:]):
        if earlier not in valid_yield or later not in valid_yield:
            rejected += 1
            continue
        returns = {
            ticker: (_finite(price[ticker][later], "end price") /
                     _finite(price[ticker][earlier], "start price") - 1)
            for ticker in HOLDINGS
        }
        if any(not 0 < float(price[ticker][earlier]) for ticker in HOLDINGS):
            raise ValueError("Non-positive stock/ETF start price")
        dy_bp = (valid_yield[later] - valid_yield[earlier]) * 100.0
        pairs.append((later, returns["SPY"], dy_bp, returns))
    training = np.array([day < HOLDOUT_START for day, *_ in pairs], dtype=bool)
    ntrain, ntest = int(np.sum(training)), len(pairs)-int(np.sum(training))
    if ntrain < 90 or ntest < 30:
        raise ValueError(f"Coverage below preregistered threshold: {ntrain} train, {ntest} OOS")
    spy = np.array([sp for _, sp, _, _ in pairs])
    rate = np.array([dy for _, _, dy, _ in pairs])
    baseline = spy.reshape(-1, 1)
    extended = np.column_stack((spy, rate))
    results = {}
    for name, weights in portfolios.items():
        y = np.array([
            math.fsum(w*returns[t] for t, w in weights.items())
            for _, _, _, returns in pairs
        ])
        results[name] = _fit_score(baseline, extended, y, training)
    instruments = {}
    for ticker in sorted(HOLDINGS):
        y = np.array([returns[ticker] for _, _, _, returns in pairs])
        # SPY itself is the exact one-factor benchmark. Fitting it gives
        # an essentially zero OOS residual, so do not claim any gain.
        if ticker == "SPY":
            instruments[ticker] = {
                "status": "IDENTICAL_TO_BENCHMARK",
                "baseline_is_exact_spy_return": True,
                "eligible_for_two_factor_improvement": False,
            }
        else:
            instruments[ticker] = _fit_score(baseline, extended, y, training)
    accepted = all(result["clears_frozen_5pct_gate"] for result in results.values())
    return {
        "schema": SCHEMA,
        "status": ("VALIDATED_REALIZED_FACTOR_TRACKING" if accepted
                   else "REALIZED_FACTOR_TRACKING_NOT_UNIFORMLY_VALIDATED"),
        "train_count": ntrain, "holdout_count": ntest,
        "paired_sessions": len(pairs), "unusable_treasury_intervals": rejected,
        "first_return_date": pairs[0][0], "last_return_date": pairs[-1][0],
        "period_start": SOURCE_BEGIN, "split_test_start": HOLDOUT_START,
        "portfolios": results, "individual_holdings": instruments,
        "all_four_pass_5pct": accepted,
        "source_universe": sorted(HOLDINGS),
        "stock_price_return_type": "public_nasdaq_price_only_not_certified_total_return",
        "market_factor_time": "same_close_realized_spy_and_dgs10",
        "ex_ante_portfolio_event_yes_pnl_identified": False,
        "equity_risk_already_priced_fraction_identified": False,
        "interpretation": (
            "Holdout measures same-period factor tracking only. The factor "
            "shocks are unknown at pre-event trade time. Predictions "
            "conditional on arbitrary Kalshi YES contracts remain unidentified."
        ),
    }


def evaluate_fomc_dual_response(rows: Sequence[Mapping]) -> dict:
    """Frozen chronological bivariate-market responses to realized policy news."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Official FOMC rows must be a sequence")
    usable = []
    seen = set()
    excluded_early = excluded_unscheduled = excluded_missing = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed source event row")
        raw_day = row.get("Date")
        if not isinstance(raw_day, str):
            raise ValueError("Expected explicit ISO event date")
        d = date.fromisoformat(raw_day[:10])
        if d in seen:
            raise ValueError("Duplicate FOMC date")
        seen.add(d)
        if d.isoformat() < FOMC_BEGIN:
            excluded_early += 1
            continue
        unscheduled = row.get("Unscheduled")
        if isinstance(unscheduled, bool) or unscheduled not in (0, 0.0, 1, 1.0):
            raise ValueError("Unscheduled indicator must be numeric 0 or 1")
        if unscheduled == 1:
            excluded_unscheduled += 1
            continue
        try:
            vals = tuple(_finite(row.get(key), key)
                         for key in ("ED4", "FF1", "SP500", "UST10Y"))
        except ValueError:
            excluded_missing += 1
            continue
        usable.append((d.isoformat(), *vals))
    usable.sort(key=lambda x: x[0])
    training = np.array([r[0] < FOMC_SPLIT for r in usable], dtype=bool)
    ntrain = int(np.sum(training))
    ntest = len(usable)-ntrain
    if ntrain < 50 or ntest < 30:
        raise ValueError(f"FOMC below preregistered coverage: {ntrain} train, {ntest} OOS")
    vals = np.asarray([record[1:] for record in usable])
    ed4, ff1, sp, ust = vals.T
    baseline = ed4.reshape(-1, 1)
    extended = np.column_stack((ed4, ff1))
    results = {
        "SP500": _fit_score(baseline, extended, sp, training),
        "UST10Y": _fit_score(baseline, extended, ust, training),
    }
    accepted = all(result["clears_frozen_5pct_gate"] for result in results.values())
    return {
        "schema": SCHEMA,
        "status": ("VALIDATED_DUAL_MARKET_REALIZED_FOMC_ASSOCIATION" if accepted
                   else "DUAL_MARKET_FF1_EXTENSION_REJECTED"),
        "source": "https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/",
        "source_tab": "Monetary Events",
        "source_event_window": "statement plus press conference",
        "train_count": ntrain, "holdout_count": ntest,
        "first_event": usable[0][0], "last_event": usable[-1][0],
        "excluded_before_2011": excluded_early,
        "excluded_unscheduled": excluded_unscheduled,
        "excluded_missing_four_factors": excluded_missing,
        "predictors": ["ED4_event_surprise_percentage_points",
                       "FF1_event_surprise_percentage_points"],
        "responses": ["SP500_intraday_return_percent",
                      "UST10Y_intraday_yield_change_percentage_points"],
        "models": results, "both_markets_pass_5pct": accepted,
        "pre_event_known_inputs": False,
        "mapped_to_Kalshi_Dec_2026_hike_yes": False,
        "ex_ante_portfolio_event_yes_pnl_identified": False,
        "equity_risk_already_priced_fraction_identified": False,
        "interpretation": (
            "Only responses to realized FOMC futures surprises, including "
            "information effects. Cannot substitute rate-decision YES "
            "for ED4 or FF1 without a separately validated joint mapping."
        ),
    }
