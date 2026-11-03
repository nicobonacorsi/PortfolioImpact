"""Recorded macro reference classes and diagnostic return regressions.

The historical payroll surprise is *first-release jobs minus a Kalshi-ladder
expectation*, not the economist survey consensus. Massive adjusted=true
stock/fund bars are split-adjusted, not dividend-total-return adjusted.
Daily close-to-close returns mix the announcement with other news.
This module cannot admit a fitted
coefficient as a causal conditional price response or a forecast probability.
"""
from __future__ import annotations

from functools import lru_cache
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import NormalDist
from urllib.parse import urlsplit


SCHEMA = "castle.empirical_payroll_context.v1"
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "r107_payroll_context.json"
MARKET_SYMBOLS = ("DBC", "GLD", "HYG", "SPY", "USO", "UUP", "VWO")
SPLIT_DATE = "2026-01-01"
Z95 = NormalDist().inv_cdf(.975)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def wilson_95(successes, trials):
    """Frequentist Wilson interval; not a prediction for a different question."""
    if not isinstance(successes, int) or not isinstance(trials, int) or trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid event count")
    p, z2 = successes / trials, Z95 ** 2
    center = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    radius = Z95 * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials ** 2)) / (1 + z2 / trials)
    return [max(0., center - radius), min(1., center + radius)]


def empirical_miss_reference(tape):
    """Observed frequency for <=-100,000 versus *Kalshi-implied* expectation."""
    if not isinstance(tape, dict) or not isinstance(tape.get("events"), list):
        raise ValueError("Invalid payroll archive")
    records, dates = [], set()
    for row in tape["events"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid event row")
        surprise = row.get("surprise")
        if surprise is None:
            continue
        date = row.get("release_date")
        if not isinstance(date, str) or len(date) != 10 or date in dates:
            raise ValueError("Invalid or repeated payroll release date")
        if not all(_finite(row.get(field)) for field in ("actual", "expected")):
            raise ValueError("Missing actual or market-implied expectation")
        if not _finite(surprise) or abs((row["actual"] - row["expected"]) - surprise) > 1e-5:
            raise ValueError("Surprise is not reconciled with first release")
        records.append(row)
        dates.add(date)
    if not records:
        raise ValueError("Empty payroll reference sample")
    k = sum(row["surprise"] <= -100_000 for row in records)
    n = len(records)
    return {
        "status": "HISTORICAL_REFERENCE_CLASS_NOT_TARGET_PROBABILITY",
        "definition": "First-release payroll jobs at least 100,000 below the market expectation reconstructed from Kalshi pre-release ladders",
        "denominator": n, "count": k, "observed_frequency": k / n,
        "wilson_95": wilson_95(k, n),
        "not_equivalent_to": "A specific future payroll release at least 100,000 below economists' consensus",
        "source_expectation_method": "Kalshi traded-strike survival ladder, with explicit tail interpolation",
        "forecast_probability": None,
    }


def _ols_hc3_split(rows):
    """Two-variable OLS and a genuinely chronological, frozen 2026 holdout."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing regression rows")
    dates = [r["date"] for r in rows]
    if len(set(dates)) != len(dates):
        raise ValueError("Repeated release date in regression")
    ordered = sorted(rows, key=lambda r: r["date"])
    if any(not _finite(r.get("surprise_jobs")) or not _finite(r.get("return_fraction")) for r in ordered):
        raise ValueError("Non-finite regression input")
    train = [r for r in ordered if r["date"] < SPLIT_DATE]
    test = [r for r in ordered if r["date"] >= SPLIT_DATE]
    if len(train) < 20 or len(test) < 8:
        return {"status": "INSUFFICIENT_CHRONOLOGICAL_SAMPLE", "train_n": len(train),
                "test_n": len(test), "admitted_as_forecast": False}
    xs = [r["surprise_jobs"] / 100_000 for r in train]
    ys = [r["return_fraction"] for r in train]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x-mx)**2 for x in xs)
    if sxx <= 1e-12:
        return {"status": "UNIDENTIFIED_REGRESSION", "train_n": n,
                "test_n": len(test), "admitted_as_forecast": False}
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    # HC3 corrects each squared residual by its leave-one-out leverage.
    variance = 0.
    for x, y in zip(xs, ys):
        leverage = 1 / n + (x - mx)**2 / sxx
        if leverage >= 1 - 1e-12:
            raise ValueError("High-leverage observation makes HC3 undefined")
        residual = y - intercept - slope * x
        variance += (x-mx)**2 * (residual / (1-leverage))**2
    se = math.sqrt(variance) / sxx
    interval = [slope - Z95*se, slope + Z95*se]
    actual = [r["return_fraction"] for r in test]
    predicted = [intercept + slope*r["surprise_jobs"]/100_000 for r in test]
    rmse = lambda estimates: math.sqrt(sum((a-b)**2 for a,b in zip(actual, estimates)) / len(actual))
    benchmark = {"regression": rmse(predicted), "zero": rmse([0.]*len(actual)),
                 "training_mean": rmse([my]*len(actual))}
    sign_stable = interval[0] > 0 or interval[1] < 0
    beats_both = benchmark["regression"] < min(benchmark["zero"], benchmark["training_mean"])
    # An observed, multi-asset *exploratory* association is never admitted
    # as causal return prediction just because it passes these two diagnostics.
    return {"status": "DESCRIPTIVE_EVENT_DAY_REGRESSION",
            "train_n": len(train), "test_n": len(test),
            "train_first": train[0]["date"], "train_last": train[-1]["date"],
            "test_first": test[0]["date"], "test_last": test[-1]["date"],
            "intercept_fraction": intercept, "slope_fraction_per_100k_jobs": slope,
            "slope_hc3_normal_95": interval, "slope_hc3_standard_error": se,
            "test_rmse_fraction": benchmark, "slope_interval_excludes_zero": sign_stable,
            "beats_zero_and_training_mean_on_holdout": beats_both,
            "mechanical_screen_passed": bool(sign_stable and beats_both),
            "admitted_as_forecast": False,
            "limitation": "Daily returns contain unrelated news. Kalshi-ladder expected jobs differ from economist consensus. Neither the regression nor the holdout identifies a causal future outcome gap."}


def study_from_verified_prices(tape, price_closes, provenance):
    """Build only from externally verified, date-indexed adjusted daily closes."""
    reference = empirical_miss_reference(tape)
    if set(price_closes) != set(MARKET_SYMBOLS) or set(provenance) != set(MARKET_SYMBOLS):
        raise ValueError("Study requires all seven preregistered market instruments")
    daily = {}
    for symbol in MARKET_SYMBOLS:
        prices = price_closes[symbol]
        if not isinstance(prices, dict) or len(prices) < 250:
            raise ValueError("Market history too short")
        dts = sorted(prices)
        if any(not _finite(prices[dt]) or prices[dt] <= 0 for dt in dts):
            raise ValueError("Invalid market price")
        daily[symbol] = {b: prices[b]/prices[a]-1 for a,b in zip(dts,dts[1:])}
    rows = []
    for e in tape["events"]:
        date = e.get("release_date")
        if e.get("surprise") is None or not date:
            continue
        if not all(date in daily[s] for s in MARKET_SYMBOLS):
            continue
        rows.append({"date": date, "surprise_jobs": e["surprise"],
                     "return_fraction_by_symbol": {s: daily[s][date] for s in MARKET_SYMBOLS}})
    rows.sort(key=lambda r: r["date"])
    fits = {}
    for symbol in MARKET_SYMBOLS:
        fits[symbol] = _ols_hc3_split([
            {"date": r["date"], "surprise_jobs": r["surprise_jobs"],
             "return_fraction": r["return_fraction_by_symbol"][symbol]} for r in rows
        ])
    return {"schema": SCHEMA, "reference_class": reference,
            "event_study": {"status": "EXPLORATORY_HISTORICAL_ASSOCIATION_NOT_CAUSAL",
                            "close_to_close": True, "training_before": SPLIT_DATE,
                            "market_symbols": list(MARKET_SYMBOLS),
                            "event_dates_n": len(rows),
                            "observations": rows, "fits": fits,
                            "fit_predictor": "Kalshi-ladder first-release surprise, units 100,000 jobs",
                            "return_source": "Massive split-adjusted, not dividend-adjusted, one-day ETF closes on release dates",
                            "not_a_calibrated_outcome_return_gap": True},
            "provenance": provenance,
            "used_in_portfolio_pnl": False}


@lru_cache(maxsize=1)
def load_study():
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    reference, event_study = data.get("reference_class"), data.get("event_study")
    if not isinstance(reference, dict) or not isinstance(event_study, dict):
        return None
    if reference.get("forecast_probability") is not None or data.get("used_in_portfolio_pnl") is not False:
        return None
    if not isinstance(event_study.get("fits"), dict):
        return None
    return data


def payroll_context_for_positions(positions):
    """A historical footnote for a B scenario; zero effect on scenario math."""
    data = load_study()
    if data is None:
        return None
    positions = positions if isinstance(positions, list) else []
    symbols = {str(p.get("symbol") or "").upper() for p in positions if isinstance(p, dict)}
    evidence = {symbol: data["event_study"]["fits"][symbol] for symbol in sorted(symbols)
                if symbol in data["event_study"]["fits"]}
    # R121: validate whether a *binary conditional-impact* mean can even be
    # supported for the original historical target. This cannot overwrite the
    # scenario's YES/NO returns, forecast P(E), or original funded NAV.
    from .binary_conditional_oos import (
        BinaryConditionalEvidenceError, screen_archived_r107_payroll,
    )
    try:
        screen = screen_archived_r107_payroll()
        binary_gate = {
            "status": screen["status"], "counts": screen["counts"],
            "minimum_train_per_branch": screen["method"]["min_train_per_branch"],
            "minimum_test_per_branch": screen["method"]["min_heldout_per_branch"],
            "reason": screen.get("reason"),
            "historical_return_center_promoted": False,
            "source_price_return_basis": screen["return_basis"],
            "exact_event_not_economist_consensus": True,
        }
    except (BinaryConditionalEvidenceError, OSError, ValueError, TypeError, KeyError):
        binary_gate = {
            "status": "HISTORICAL_BINARY_IMPACT_SCREEN_UNAVAILABLE",
            "reason": "Archived event-return evidence failed verification or was unavailable.",
            "historical_return_center_promoted": False,
        }
    return {"status": "HISTORICAL_CONTEXT_ONLY", "reference_class": data["reference_class"],
            "holding_studies": evidence,
            "event_study_n": data["event_study"]["event_dates_n"],
            "observation_source": "Recorded Kalshi ladder plus Massive adjusted daily ETF closes",
            "binary_conditional_impact_evidence_gate": binary_gate,
            "event_probability_updated": False, "portfolio_pnl_updated": False,
            "source_document": "docs/R107_EMPIRICAL_EVIDENCE.md"}


def nested_threshold_probability_bounds(target, market):
    """Derive only logically certified bounds from *identical* settlement bases.

    The caller must give a source-marked verified market quote with a bracket.
    For identical metrics, measurement rules and settlement dates, stronger
    '>= threshold' events are subsets of weaker ones (reverse for '<=').
    Different time horizons or differently specified variables are rejected.
    """
    if not isinstance(target, dict) or not isinstance(market, dict):
        return None
    same = ("metric_id", "unit", "measurement_rule", "settlement_source", "settlement_date")
    if not all(isinstance(target.get(k), str) and target[k] and target[k] == market.get(k) for k in same):
        return None
    op = target.get("operator")
    if op not in ("ge", "le") or op != market.get("operator"):
        return None
    t, m = target.get("threshold"), market.get("threshold")
    quote = market.get("verified_quote_interval")
    if not _finite(t) or not _finite(m) or not isinstance(quote, list) or len(quote) != 2:
        return None
    if not all(_finite(v) for v in quote) or not 0 <= quote[0] <= quote[1] <= 1:
        return None
    if market.get("identity_verified") is not True or not isinstance(market.get("contract_id"), str) or not market["contract_id"]:
        return None
    if market.get("quote_status") != "OBSERVED_CONTRACT_SNAPSHOT":
        return None
    link = urlsplit(str(market.get("source_url") or ""))
    if (link.scheme != "https" or not link.hostname or
            not (link.hostname in ("kalshi.com", "polymarket.com") or
                 link.hostname.endswith(".kalshi.com") or link.hostname.endswith(".polymarket.com"))):
        return None
    try:
        quote_time = datetime.fromisoformat(str(market.get("quote_observed_at") or ""))
    except ValueError:
        return None
    if quote_time.tzinfo is None:
        return None
    if t == m:
        return {"status": "EXACT_THRESHOLD_IDENTITY", "interval": quote[:],
                "source_contract_id": market["contract_id"], "probability_type": "PRICE_IMPLIED"}
    target_subset = (t > m if op == "ge" else t < m)
    return {"status": "LOGICAL_SUBSET_BOUND" if target_subset else "LOGICAL_SUPERSET_BOUND",
            "interval": [0., quote[1]] if target_subset else [quote[0], 1.],
            "source_contract_id": market["contract_id"], "probability_type": "LOGICAL_BOUND_ON_PRICE_IMPLIED_EVENT_WEIGHT",
            "not_an_exact_market_quote_for_target": True}
