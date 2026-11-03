"""Date-indexed empirical associations and explicitly assumed event scenarios.

This module has no provider, credential, network, or archive-admission logic.
Daily prices are supplied as ``{symbol: [{"date": "YYYY-MM-DD", "close": ...}]}``.
The caller must supply completed, consistently adjusted US-session closes. Only
sessions strictly before the UTC evaluation date are used: an incomplete current
session must never enter a historical fit. Returns are aligned on BOTH their
start and end dates, rather than pairing one-day and missing-session returns.

``estimate_instrument_betas`` freezes an OLS fit at ``train_end`` and evaluates
later observations without refitting. ``fit_event_study`` does the same for
documented FOMC/CPI surprises. Neither function establishes causal effects,
historical public availability, a physical probability, or forecast validity.
In particular this module does not alter or satisfy the original P3 admission
gates. Its supplied-data event study is a separate, labelled association.

``beta_scaled_contrasts`` and ``event_study_contrasts`` require an explicit
YES-minus-NO shock; there is no invented default. A failed transfer diagnostic
remains visible when the caller uses a coefficient as a scenario assumption.
``build_binary_event_impact`` prices each original position once. The separate
Taiwan helper uses the existing authoritative exposure ledger for its NVDA
channel; it never adds a whole ETF to its underlying constituents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from scipy.stats import t as student_t

from .portfolio import ExposureReport, Portfolio, WEIGHT_TOLERANCE, _finite_number, _symbol


SCHEMA_VERSION = "CASTLE_EMPIRICAL_EVENT_IMPACT_V1"
EVENT_UNITS = {"FOMC": "policy_bp", "CPI": "cpi_pp"}
ASSOCIATION_LIMITATIONS = [
    "Supplied prices and source metadata are not independently authenticated by this mathematics module.",
    "Daily close-to-close price returns include other news and exclude cash dividends unless the caller supplies total-return prices.",
    "OLS associations are not causal effects or universal instrument sensitivities.",
    "HC3 intervals are approximate parameter intervals, not prediction intervals; serial dependence, regime changes and multiple comparisons are not covered.",
    "Held-out diagnostics use contemporaneously observed explanatory variables; they do not establish ex-ante forecast accuracy.",
]
PRICING_ASSUMPTIONS = [
    "The supplied binary pricing weight is transferred to the portfolio's economic branches as an explicit financial assumption.",
    "The quote does not identify a physical event probability or the pricing weight actually embedded in each asset.",
    "YES and NO are the complete complementary branches of the same exact event; their economic contrast must be supplied separately.",
    "Other news, risk premia, discounting and independent factor moves are held fixed in this scenario.",
    "Q-weighted zero mean follows algebraically from centering the two branches; it is not empirical proof that an event is fully priced in.",
]


def _number(value: Any, field: str) -> float:
    return _finite_number(value, field)


def _day(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        raise ValueError(f"{field} must be a session date, not a timestamp.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO session date.")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO session date.") from exc
    if result.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD.")
    return result


def _timestamp(value: Any, field: str) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO timestamp with a timezone.") from exc
    if result.tzinfo is None:
        raise ValueError(f"{field} must include a timezone.")
    return result.astimezone(timezone.utc)


def _interval(value: Any, point: float | None, field: str) -> list[float]:
    if value is None:
        if point is None:
            raise ValueError(f"{field} requires two endpoints.")
        return [point, point]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} requires two endpoints.")
    lo, hi = (_number(v, field) for v in value)
    if lo > hi or point is not None and not lo <= point <= hi:
        raise ValueError(f"{field} must be ordered and contain its point estimate.")
    return [lo, hi]


def _product_interval(left: Sequence[float], right: Sequence[float]) -> list[float]:
    values = [a * b for a in left for b in right]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Scenario multiplication overflowed; use finite economic inputs.")
    return [min(values), max(values)]


def _sample_minimum(value: Any, minimum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}.")
    return value


def _prices(prices_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]], as_of: datetime) -> tuple[dict, dict]:
    if not isinstance(prices_by_symbol, Mapping):
        raise ValueError("prices_by_symbol must map symbols to dated closes.")
    normalized, audits = {}, {}
    for raw_symbol, rows in prices_by_symbol.items():
        symbol = _symbol(raw_symbol)
        if symbol in normalized:
            raise ValueError("Duplicate normalized price symbols are not allowed.")
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"{symbol} prices must be a list of dated closes.")
        values, seen, excluded = {}, set(), 0
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{symbol} price rows must be mappings.")
            day = _day(row.get("date"), "Price date")
            close = _number(row.get("close"), "Close")
            if close <= 0:
                raise ValueError("Closing prices must be positive.")
            if day in seen:
                raise ValueError(f"Duplicate {symbol} session dates must be resolved upstream.")
            seen.add(day)
            if day >= as_of.date():
                excluded += 1
                continue
            if "available_at" in row and _timestamp(row["available_at"], "Price availability") > as_of:
                excluded += 1
                continue
            values[day] = close
        normalized[symbol] = dict(sorted(values.items()))
        content = [[d.isoformat(), p] for d, p in normalized[symbol].items()]
        audits[symbol] = {
            "input_bar_count": len(rows), "admitted_completed_session_count": len(values),
            "excluded_current_future_or_unavailable_count": excluded,
            "normalized_input_sha256": hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest(),
            "hash_scope": "Normalized supplied closes, not an independently authenticated provider response.",
        }
    return normalized, audits


def _returns(prices: Mapping[date, float]) -> dict[tuple[date, date], float]:
    ordered = list(prices.items())
    # Long gaps are not treated as ordinary daily observations. The seven-day
    # maximum is an explicit cleaning policy, not an inferred holiday calendar.
    result = {}
    for (d0, p0), (d1, p1) in zip(ordered, ordered[1:]):
        if (d1 - d0).days <= 7:
            value = p1 / p0 - 1.0
            if not math.isfinite(value):
                raise ValueError("Supplied prices produce a nonfinite return.")
            result[(d0, d1)] = value
    return result


def _span(rows: Sequence[tuple[date, float, float]]) -> list[str] | None:
    return [rows[0][0].isoformat(), rows[-1][0].isoformat()] if rows else None


def _fit_association(rows: list[tuple[date, float, float]], train_end: date, min_train: int, min_holdout: int) -> dict:
    rows = sorted(rows)
    training = [r for r in rows if r[0] <= train_end]
    held_out = [r for r in rows if r[0] > train_end]
    base = {
        "status": "NOT_ESTIMATED", "beta": None, "beta_interval": None,
        "intercept": None, "intercept_interval": None,
        "n_total": len(rows), "n_train": len(training), "n_holdout": len(held_out),
        "training_span": _span(training), "holdout_span": _span(held_out),
        "minimum_samples": {"train": min_train, "holdout": min_holdout},
        "train_end": train_end.isoformat(), "confidence_level": 0.95,
        "interval_method": "OLS HC3 heteroskedasticity-robust covariance; Student t with n_train-2 degrees of freedom.",
        "interval_scope": "Approximate parameter uncertainty conditional on the supplied sample and linear specification.",
        "transfer_check": {"status": "NOT_TESTED", "reason": "Insufficient chronological training or held-out observations."},
        "forward_prediction_certified": False,
    }
    if len(training) < min_train or len(held_out) < min_holdout:
        base["reason"] = "Insufficient observations; no coefficient or zero-valued substitute is estimated."
        return base
    x = np.array([r[1] for r in training], dtype=float)
    y = np.array([r[2] for r in training], dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    if np.linalg.matrix_rank(design) < 2:
        base["reason"] = "The training explanatory variable has no usable variation."
        return base
    try:
        inverse = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        base["reason"] = "The training design cannot be stably inverted."
        return base
    coefficients = inverse @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    if np.any(leverage >= 1 - 1e-10):
        base["reason"] = "A unit-leverage observation prevents finite HC3 parameter uncertainty."
        return base
    scaled = residuals / (1.0 - leverage)
    covariance = inverse @ (design.T @ (scaled[:, None] ** 2 * design)) @ inverse
    se = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    critical = float(student_t.ppf(0.975, len(training) - 2))
    intervals = [[float(c - critical * s), float(c + critical * s)] for c, s in zip(coefficients, se)]
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intervals)):
        base["reason"] = "The numerical regression did not produce finite coefficients and intervals."
        return base
    hx = np.array([r[1] for r in held_out], dtype=float)
    hy = np.array([r[2] for r in held_out], dtype=float)
    prediction = coefficients[0] + coefficients[1] * hx
    rmse = float(np.sqrt(np.mean((hy - prediction) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(hy ** 2)))
    mean_rmse = float(np.sqrt(np.mean((hy - np.mean(y)) ** 2)))
    # A null/constant series cannot earn a pass just by tying a null baseline.
    passes = rmse + 1e-12 < zero_rmse and rmse + 1e-12 < mean_rmse
    base.update({
        "status": "ESTIMATED_INPUT_ASSOCIATION",
        "beta": float(coefficients[1]), "beta_interval": intervals[1],
        "intercept": float(coefficients[0]), "intercept_interval": intervals[0],
        "training_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "training_x_range": [float(np.min(x)), float(np.max(x))],
        "transfer_check": {
            "status": "PASS" if passes else "FAIL", "held_out_rmse": rmse,
            "zero_return_baseline_rmse": zero_rmse, "frozen_training_mean_baseline_rmse": mean_rmse,
            "criterion": "Frozen model RMSE must be strictly below both zero-return and training-mean baselines on later observations.",
            "interpretation": "A limited chronological association diagnostic, not causal validation or ex-ante forecast certification.",
        },
        "reason": "Coefficients were fitted only on training observations and were not refitted on held-out data.",
    })
    return base


def _estimation_context(analysis_as_of: str | datetime, train_end: str | date, min_train: int, min_holdout: int) -> tuple:
    as_of = _timestamp(analysis_as_of, "analysis_as_of")
    cutoff = _day(train_end, "train_end")
    if cutoff >= as_of.date():
        raise ValueError("train_end must precede the evaluation date.")
    return as_of, cutoff, _sample_minimum(min_train, 4, "min_train"), _sample_minimum(min_holdout, 2, "min_holdout")


def estimate_instrument_betas(
    prices_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    benchmark_symbol: str,
    *, analysis_as_of: str | datetime, train_end: str | date,
    min_train: int = 60, min_holdout: int = 20,
) -> dict[str, Any]:
    """Estimate a distinct frozen daily-return beta for each supplied instrument.

``analysis_as_of`` requires a timezone. ``train_end`` is inclusive; all later
completed supplied sessions form one untouched diagnostic holdout. Prices are
split-adjusted price closes or total-return closes as explicitly chosen by the
caller; these bases must never be mixed within a series. No data are acquired.
    """
    as_of, cutoff, min_train, min_holdout = _estimation_context(analysis_as_of, train_end, min_train, min_holdout)
    prices, audits = _prices(prices_by_symbol, as_of)
    benchmark = _symbol(benchmark_symbol)
    benchmark_returns = _returns(prices.get(benchmark, {}))
    instruments = {}
    for symbol, values in prices.items():
        own_returns = _returns(values)
        pairs = sorted(set(own_returns) & set(benchmark_returns))
        rows = [(end, benchmark_returns[(start, end)], own_returns[(start, end)]) for start, end in pairs]
        fit = _fit_association(rows, cutoff, min_train, min_holdout)
        fit.update({
            "symbol": symbol, "benchmark_symbol": benchmark,
            "beta_unit": "instrument_return_fraction_per_benchmark_return_fraction",
            "alignment": "Identical start and end session dates; no forward fill or resampling.",
            "unmatched_or_long_gap_return_count": max(0, len(values) - 1 - len(pairs)),
            "source_status": "SUPPLIED_PRICES_NOT_INDEPENDENTLY_AUTHENTICATED_BY_THIS_MODULE",
        })
        if not benchmark_returns:
            fit["reason"] = "Benchmark prices are absent or have no usable completed return observations."
        instruments[symbol] = fit
    return {
        "schema": SCHEMA_VERSION, "kind": "INSTRUMENT_DAILY_BETAS",
        "analysis_as_of": as_of.isoformat(), "train_end": cutoff.isoformat(),
        "benchmark_symbol": benchmark, "instruments": instruments,
        "price_input_audit": audits, "limitations": list(ASSOCIATION_LIMITATIONS),
        "status": "ESTIMATED_INPUT_ASSOCIATIONS" if any(r["beta"] is not None for r in instruments.values()) else "NOT_ESTIMATED",
    }


def fit_event_study(
    prices_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    event_observations: Sequence[Mapping[str, Any]],
    family: str,
    *, analysis_as_of: str | datetime, train_end: str | date,
    min_train: int = 12, min_holdout: int = 4,
) -> dict[str, Any]:
    """Fit supplied, documented surprises to event-day close-to-close returns.

Each observation requires ``event_date``, ``actual``, ``expected``,
``surprise_unit`` (``policy_bp`` for FOMC, ``cpi_pp`` for headline SA MoM CPI),
``expectation_published_at``, ``actual_published_at``, ``expectation_source_sha256``
and ``actual_source_sha256``. An optional ``surprise`` must equal actual-expected.
Actuals must be initial releases and expectations must be as-published vintages;
the caller remains responsible for that source evidence. Hashes and timestamp
strings do not establish those facts, so the result is explicitly documentary
and does not pass the original P3 source-admission gate. No mandatory P3 research
or callback blocks an independently declared assessment scenario.
    """
    if family not in EVENT_UNITS:
        raise ValueError("Event family must be FOMC or CPI.")
    if not isinstance(event_observations, (list, tuple)):
        raise ValueError("event_observations must be a list of dated observations.")
    as_of, cutoff, min_train, min_holdout = _estimation_context(analysis_as_of, train_end, min_train, min_holdout)
    prices, audits = _prices(prices_by_symbol, as_of)
    duplicates = Counter(str(r.get("event_date")) for r in event_observations if isinstance(r, Mapping))
    accepted, rejected = {}, []
    try:
        new_york = ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        new_york = None
    for row in event_observations:
        label = str(row.get("event_date", "UNKNOWN")) if isinstance(row, Mapping) else "UNKNOWN"
        try:
            if not isinstance(row, Mapping):
                raise ValueError("Every event observation must be a mapping.")
            day = _day(row.get("event_date"), "Event date")
            if duplicates[day.isoformat()] > 1:
                raise ValueError("Duplicate event dates invalidate all records for that date.")
            if row.get("surprise_unit") != EVENT_UNITS[family]:
                raise ValueError("The surprise unit does not match the requested event family.")
            expected_at = _timestamp(row.get("expectation_published_at"), "Expectation publication")
            actual_at = _timestamp(row.get("actual_published_at"), "Actual publication")
            if not expected_at < actual_at <= as_of:
                raise ValueError("Expectation must precede the release, and the release must be observed by analysis_as_of.")
            if actual_at.date() != day or day >= as_of.date():
                raise ValueError("A completed event session matching the actual publication date is required.")
            if (actual_at - expected_at).total_seconds() > 35 * 86400:
                raise ValueError("Expectation is stale under the explicit 35-day policy.")
            if new_york is None:
                raise ValueError("America/New_York timezone data are required to verify the event-price window.")
            local_release = actual_at.astimezone(new_york)
            if local_release.date() != day or local_release.time().replace(tzinfo=None) >= time(16):
                raise ValueError("The event release must precede the US session close used for its return.")
            for key in ("expectation_source_sha256", "actual_source_sha256"):
                if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row[key]):
                    raise ValueError("Both supplied source records require SHA-256 identifiers.")
            surprise = _number(row.get("actual"), "Event actual") - _number(row.get("expected"), "Event expectation")
            if not math.isfinite(surprise):
                raise ValueError("The surprise difference must be finite.")
            if "surprise" in row and not math.isclose(_number(row["surprise"], "Surprise"), surprise, abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError("The supplied surprise disagrees with actual minus expected.")
            accepted[day] = surprise
        except (ValueError, TypeError) as exc:
            rejected.append({"event_date": label, "reason": str(exc)})
    instruments = {}
    for symbol, values in prices.items():
        returns_by_end = {end: ret for (_, end), ret in _returns(values).items()}
        rows = [(day, shock, returns_by_end[day]) for day, shock in accepted.items() if day in returns_by_end]
        fit = _fit_association(rows, cutoff, min_train, min_holdout)
        fit.update({
            "symbol": symbol, "beta_unit": f"instrument_return_fraction_per_{EVENT_UNITS[family]}",
            "missing_event_return_count": len(accepted) - len(rows),
            "source_status": "DOCUMENTARY_INPUTS_NOT_INDEPENDENTLY_ARCHIVE_VERIFIED",
            "original_p3_source_admission_satisfied": False,
            "source_authenticated_sample_count": None,
        })
        if fit["beta"] is not None:
            fit["status"] = "ESTIMATED_DOCUMENTARY_ASSOCIATION_ONLY"
        instruments[symbol] = fit
    return {
        "schema": SCHEMA_VERSION, "kind": "DOCUMENTARY_EVENT_STUDY", "family": family,
        "surprise_definition": "Actual initial release minus the documented pre-release expectation.",
        "surprise_unit": EVENT_UNITS[family], "analysis_as_of": as_of.isoformat(),
        "train_end": cutoff.isoformat(), "instruments": instruments,
        "input_event_count": len(event_observations), "documentary_candidate_count": len(accepted),
        "independently_admitted_event_count": 0, "rejected_events": rejected,
        "source_admission_status": "NOT_INDEPENDENTLY_VERIFIED_BY_THIS_MODULE",
        "original_p3_source_admission_satisfied": False,
        "price_input_audit": audits,
        "limitations": list(ASSOCIATION_LIMITATIONS) + [
            "Source hashes and caller-stated publication times are documentary evidence, not independent proof of an initial release or pre-release vintage.",
            "Any use of the supplied-data slope in an assessment is an explicitly assumed association; it does not certify a live forecast.",
        ],
        "status": "ESTIMATED_DOCUMENTARY_ASSOCIATIONS_ONLY" if any(r["beta"] is not None for r in instruments.values()) else "NOT_ESTIMATED",
    }


def fit_event_day_association(
    prices_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    event_dates: Sequence[str | Mapping[str, Any]],
    family: str, benchmark_symbol: str,
    *, analysis_as_of: str | datetime, train_end: str | date,
    min_train: int = 12, min_holdout: int = 4,
) -> dict[str, Any]:
    """Frozen instrument/benchmark association restricted to documented event days.

``event_dates`` contains ISO dates or records with ``event_date``. The caller
documents those dates against official release calendars. A calendar date does
not supply a policy/CPI surprise: this model regresses the instrument's observed
daily return on the benchmark's observed daily return on those dates. Both may
reflect other news. It is a usable, labelled scenario sensitivity when authentic
historical expectation vintages are absent, not a replacement P3 source gate.
    """
    if family not in EVENT_UNITS:
        raise ValueError("Event family must be FOMC or CPI.")
    if not isinstance(event_dates, (list, tuple)):
        raise ValueError("event_dates must be a list of documented session dates.")
    as_of, cutoff, min_train, min_holdout = _estimation_context(analysis_as_of, train_end, min_train, min_holdout)
    prices, audits = _prices(prices_by_symbol, as_of)
    benchmark = _symbol(benchmark_symbol)
    raw_days = [r.get("event_date") if isinstance(r, Mapping) else r for r in event_dates]
    counts = Counter(str(v) for v in raw_days)
    days, rejected = set(), []
    for value in raw_days:
        try:
            day = _day(value, "Event date")
            if counts[day.isoformat()] > 1:
                raise ValueError("Duplicate calendar dates invalidate all records for that date.")
            if day >= as_of.date():
                raise ValueError("Event session is not completed before the evaluation date.")
            days.add(day)
        except ValueError as exc:
            rejected.append({"event_date": str(value), "reason": str(exc)})
    benchmark_returns = _returns(prices.get(benchmark, {}))
    instruments = {}
    for symbol, values in prices.items():
        own_returns = _returns(values)
        pairs = sorted(pair for pair in set(own_returns) & set(benchmark_returns) if pair[1] in days)
        rows = [(end, benchmark_returns[(start, end)], own_returns[(start, end)]) for start, end in pairs]
        fit = _fit_association(rows, cutoff, min_train, min_holdout)
        fit.update({
            "symbol": symbol, "benchmark_symbol": benchmark,
            "beta_unit": "instrument_event_day_return_fraction_per_benchmark_event_day_return_fraction",
            "missing_event_return_count": len(days) - len(rows),
            "source_status": "SUPPLIED_PRICES_AND_DOCUMENTED_EVENT_DATES_NOT_INDEPENDENTLY_AUTHENTICATED_BY_THIS_MODULE",
            "explanatory_variable": "Observed same-session benchmark price return, not a policy/CPI surprise or advance forecast.",
            "original_p3_source_admission_satisfied": False,
        })
        if fit["beta"] is not None:
            fit["status"] = "ESTIMATED_EVENT_DAY_BENCHMARK_ASSOCIATION"
        instruments[symbol] = fit
    return {
        "schema": SCHEMA_VERSION, "kind": "EVENT_DAY_BENCHMARK_ASSOCIATION", "family": family,
        "benchmark_symbol": benchmark, "analysis_as_of": as_of.isoformat(), "train_end": cutoff.isoformat(),
        "instruments": instruments, "input_event_date_count": len(event_dates), "documented_event_date_count": len(days),
        "rejected_event_dates": rejected, "price_input_audit": audits,
        "surprise_estimated": False, "original_p3_source_admission_satisfied": False,
        "limitations": list(ASSOCIATION_LIMITATIONS) + [
            "Restricting dates to an official event calendar does not isolate that event from other news on the same session.",
            "Observed benchmark returns are the explanatory variable; no policy surprise, CPI surprise, historical consensus or physical probability is inferred.",
            "Applying this coefficient to a future benchmark branch contrast requires an explicit scenario assumption and preserves any held-out transfer FAIL.",
        ],
        "status": "ESTIMATED_EVENT_DAY_BENCHMARK_ASSOCIATIONS" if any(r["beta"] is not None for r in instruments.values()) else "NOT_ESTIMATED",
    }


def _scaled_contrasts(report: Mapping[str, Any], contrast: Any, interval: Any, basis: str, family: str) -> dict:
    point = _number(contrast, "Explicit YES-minus-NO explanatory-variable contrast")
    bounds = _interval(interval, point, "Declared shock contrast interval")
    result = {}
    for symbol, fit in report.get("instruments", {}).items():
        beta = fit.get("beta")
        transfer = fit.get("scenario_transfer_status", fit.get("transfer_check", {}).get("status", "NOT_TESTED"))
        common = {
            "basis": basis, "event_family": family, "scenario_only": True,
            "transfer_status": transfer, "coefficient_source_status": fit.get("source_status"),
            "declared_explanatory_contrast": point, "declared_explanatory_contrast_interval": bounds,
            "interval_type": "PARAMETER_INTERVAL_TIMES_DECLARED_SCENARIO_RANGE_NOT_A_PREDICTION_INTERVAL",
            "forward_prediction_certified": False,
        }
        if beta is None or fit.get("beta_interval") is None:
            result[symbol] = dict(common, return_contrast=None, return_contrast_interval=None, status="NOT_ESTIMATED", reason=fit.get("reason", "No estimated coefficient."))
            continue
        beta = _number(beta, "Estimated beta")
        beta_bounds = _interval(fit["beta_interval"], beta, "Beta parameter interval")
        value = beta * point
        if not math.isfinite(value):
            raise ValueError("Scenario return contrast overflowed.")
        result[symbol] = dict(common, return_contrast=value, return_contrast_interval=_product_interval(beta_bounds, bounds),
                              status="DECLARED_SCENARIO_TRANSFER_FAILED" if transfer == "FAIL" else "DECLARED_SCENARIO",
                              estimated_beta=beta, estimated_beta_interval=beta_bounds)
    return result


def beta_scaled_contrasts(beta_report: Mapping[str, Any], benchmark_return_contrast: float,
                          benchmark_contrast_interval: Sequence[float] | None = None,
                          *, event_family: str = "FOMC") -> dict:
    """Apply an explicitly declared benchmark YES-NO contrast to distinct betas.

The intercept cancels in a branch contrast under the assumed common linear
response. A FAIL is retained and does not become a forecast-validation pass.
    """
    kind = beta_report.get("kind")
    if kind not in {"INSTRUMENT_DAILY_BETAS", "EVENT_DAY_BENCHMARK_ASSOCIATION"}:
        raise ValueError("Expected an instrument daily-beta or event-day benchmark report.")
    if kind == "EVENT_DAY_BENCHMARK_ASSOCIATION" and beta_report.get("family") != event_family:
        raise ValueError("The requested scenario family must match its event-day calibration dates.")
    return _scaled_contrasts(beta_report, benchmark_return_contrast, benchmark_contrast_interval,
                             "DECLARED_BENCHMARK_SCENARIO_TIMES_EVENT_DAY_BETA" if kind == "EVENT_DAY_BENCHMARK_ASSOCIATION" else "DECLARED_BENCHMARK_SCENARIO_TIMES_ESTIMATED_DAILY_BETA", event_family)


def event_study_contrasts(study: Mapping[str, Any], yes_minus_no_surprise: float,
                         surprise_contrast_interval: Sequence[float] | None = None) -> dict:
    """Use a supplied-data event association as an explicitly declared scenario."""
    if study.get("kind") != "DOCUMENTARY_EVENT_STUDY":
        raise ValueError("Expected a documentary event-study report.")
    return _scaled_contrasts(study, yes_minus_no_surprise, surprise_contrast_interval,
                             "DECLARED_SCENARIO_TIMES_DOCUMENTARY_EVENT_ASSOCIATION", study.get("family"))


def _priced_in(delta: float | None, delta_interval: list[float] | None,
               q_interval: Sequence[float] | None, q_reference: float | None) -> dict:
    result = {
        "status": "NOT_ESTIMATED", "q_interval": None, "q_reference": None,
        "yes_pnl_usd": None, "no_pnl_usd": None,
        "yes_pnl_interval_usd": None, "no_pnl_interval_usd": None,
        "yes_formula": "(1 - q) * (P&L_YES - P&L_NO)",
        "no_formula": "-q * (P&L_YES - P&L_NO)",
        "assumptions": list(PRICING_ASSUMPTIONS),
        "interval_scope": "Joint endpoint envelope of the supplied quote range and economic contrast range; no physical confidence or coverage claim.",
    }
    if q_interval is None:
        if q_reference is not None:
            raise ValueError("q_reference requires a q_interval.")
        result["reason"] = "No binary quote-implied pricing-weight interval was supplied."
        return result
    bounds = _interval(q_interval, None, "q_interval")
    if bounds[0] < 0 or bounds[1] > 1:
        raise ValueError("Quote-implied pricing weights must lie in [0, 1].")
    q = (bounds[0] + bounds[1]) / 2 if q_reference is None else _number(q_reference, "q_reference")
    if not bounds[0] <= q <= bounds[1]:
        raise ValueError("q_reference must lie within q_interval.")
    result.update(q_interval=bounds, q_reference=q,
                  q_reference_basis="Arithmetic quote-range midpoint for display only; not a calibrated physical probability." if q_reference is None else "Explicit caller-selected point within the quote range.")
    if delta is None or delta_interval is None:
        result["reason"] = "The full economic branch contrast is not estimated."
        return result
    result.update(status="CONDITIONAL_PRICING_WEIGHT_SCENARIO", yes_pnl_usd=(1 - q) * delta, no_pnl_usd=-q * delta,
                  yes_pnl_interval_usd=_product_interval([1 - bounds[1], 1 - bounds[0]], delta_interval),
                  no_pnl_interval_usd=_product_interval([-bounds[1], -bounds[0]], delta_interval),
                  zero_q_mean_by_construction=True)
    return result


def _return_domain(contrast: float | None, bounds: list[float] | None,
                   q_interval: Sequence[float] | None, q_reference: float | None) -> dict:
    # The difference between two possible returns can exceed one. It is each
    # centred branch, rather than the raw difference, that has the -100% floor.
    centered = _priced_in(contrast, bounds, q_interval, q_reference)
    if centered["status"] != "CONDITIONAL_PRICING_WEIGHT_SCENARIO":
        return {"status": "NOT_ASSESSED", "reason": "Both a numeric contrast and a pricing weight are required for branch-price domain checks."}
    yes, no = centered["yes_pnl_usd"], centered["no_pnl_usd"]
    if min(yes, no) < -1.0 - 1e-12:
        raise ValueError("The declared point scenario implies a conditional instrument return below -100%; it violates the nonnegative-price domain.")
    yes_bounds, no_bounds = centered["yes_pnl_interval_usd"], centered["no_pnl_interval_usd"]
    valid_range = min(yes_bounds[0], no_bounds[0]) >= -1.0 - 1e-12
    return {
        "status": "WITHIN_NONNEGATIVE_PRICE_DOMAIN" if valid_range else "RANGE_EXTENDS_BELOW_MINUS_100_PERCENT",
        "yes_return_at_q_reference": yes, "no_return_at_q_reference": no,
        "yes_return_interval": yes_bounds, "no_return_interval": no_bounds,
        "range_within_nonnegative_price_domain": valid_range,
        "interpretation": "A domain check only; it does not establish forecast feasibility or accuracy." if valid_range else "The linear model's range includes negative-price branches and is not a physically feasible prediction envelope. No clipping or silent repair is applied.",
    }


def build_binary_event_impact(
    portfolio: Portfolio, instrument_contrasts: Mapping[str, Mapping[str, Any]],
    q_interval: Sequence[float] | None = None, *, q_reference: float | None = None,
) -> dict[str, Any]:
    """Aggregate original positions once, with unknown total if any is unmodeled.

Each contrast record requires ``return_contrast`` (fractional YES-minus-NO
return) and a nonempty ``basis``. ``return_contrast_interval`` is optional: an
omitted interval denotes an explicit point scenario, never zero uncertainty.
Cash is assigned a zero event-window contrast under an explicit unchanged-USD
cash assumption. Missing stock/ETF effects are never silently assigned zero.
    """
    if not isinstance(portfolio, Portfolio) or not isinstance(instrument_contrasts, Mapping):
        raise ValueError("Expected a validated Portfolio and instrument contrast mapping.")
    contrasts = {}
    for key, value in instrument_contrasts.items():
        symbol = _symbol(key)
        if symbol in contrasts:
            raise ValueError("Duplicate normalized contrast symbols are not allowed.")
        contrasts[symbol] = value
    positions, missing, contributions, lower, upper, transfers = [], [], [], [], [], []
    outside_domain = False
    known_weight = 0.0
    for position in portfolio.positions:
        row = {"symbol": position.symbol, "asset_type": position.asset_type, "weight": position.weight,
               "delta_pnl_usd": None, "delta_pnl_interval_usd": None}
        supplied = contrasts.get(position.symbol)
        if position.weight == 0:
            supplied = {"return_contrast": 0.0, "basis": "Zero portfolio weight; no dollar exposure."}
        elif position.asset_type == "cash" and supplied is None:
            supplied = {"return_contrast": 0.0, "basis": "Explicit unchanged USD cash balance over the event window; interest and FX excluded."}
        if supplied is None or isinstance(supplied, Mapping) and supplied.get("return_contrast") is None:
            row.update(status="NOT_ESTIMATED", reason="No numeric event contrast is supplied for this original position.")
            missing.append(position.symbol)
            positions.append(row)
            continue
        if not isinstance(supplied, Mapping) or not isinstance(supplied.get("basis"), str) or not supplied["basis"].strip():
            raise ValueError("Every numeric contrast requires an explicit assumption or estimation basis.")
        contrast = _number(supplied["return_contrast"], "Instrument return contrast")
        bounds = _interval(supplied.get("return_contrast_interval"), contrast, "Instrument contrast interval")
        domain = _return_domain(contrast, bounds, q_interval, q_reference)
        outside_domain = outside_domain or domain["status"] == "RANGE_EXTENDS_BELOW_MINUS_100_PERCENT"
        amount = portfolio.notional * position.weight
        pnl_bounds = _product_interval([amount, amount], bounds)
        pnl = amount * contrast
        if not math.isfinite(pnl):
            raise ValueError("Position scenario P&L overflowed.")
        transfer = supplied.get("transfer_status", "NOT_APPLICABLE")
        transfers.append(transfer)
        row.update(status="DECLARED_SCENARIO", return_contrast=contrast, return_contrast_interval=bounds,
                   basis=supplied["basis"], transfer_status=transfer, delta_pnl_usd=pnl, delta_pnl_interval_usd=pnl_bounds,
                   price_domain=domain,
                   interval_type=supplied.get("interval_type", "DECLARED_RANGE" if supplied.get("return_contrast_interval") is not None else "DECLARED_POINT_SCENARIO_NOT_AN_UNCERTAINTY_ESTIMATE"))
        contributions.append(pnl)
        lower.append(pnl_bounds[0])
        upper.append(pnl_bounds[1])
        known_weight += position.weight
        positions.append(row)
    known = math.fsum(contributions) if known_weight > 0 else None
    known_interval = [math.fsum(lower), math.fsum(upper)] if known_weight > 0 else None
    complete = not missing
    delta = known if complete else None
    bounds = known_interval if complete else None
    return {
        "schema": SCHEMA_VERSION, "status": "SCENARIO_ESTIMATED_RANGE_OUTSIDE_PRICE_DOMAIN" if complete and outside_domain else "SCENARIO_ESTIMATED_TRANSFER_FAILED" if complete and "FAIL" in transfers else "SCENARIO_ESTIMATED" if complete else "PARTIAL_SCENARIO" if known_weight > 0 else "NOT_ESTIMATED",
        "scope": "FULL_ORIGINAL_PORTFOLIO" if complete else "KNOWN_ORIGINAL_POSITIONS_ONLY",
        "notional_usd": portfolio.notional, "delta_pnl_usd": delta, "delta_pnl_interval_usd": bounds,
        "known_component_delta_pnl_usd": known, "known_component_delta_pnl_interval_usd": known_interval,
        "modeled_weight": known_weight, "missing_weight": math.fsum(p.weight for p in portfolio.positions if p.symbol in missing),
        "missing_symbols": missing, "positions": positions,
        "portfolio_accounting": "Each original position appears once; ETF constituents are not added to whole-ETF returns.",
        "interval_aggregation": "Sum of marginal economic contrast bounds is an outer range, without an independence or joint-attainability claim.",
        "priced_in": _priced_in(delta, bounds, q_interval, q_reference),
        "known_component_priced_in": _priced_in(known, known_interval, q_interval, q_reference),
        "transfer_status": "FAIL" if "FAIL" in transfers else "PASS" if transfers and all(v == "PASS" for v in transfers) else "NOT_UNIFORMLY_VALIDATED",
        "price_domain_status": "RANGE_EXTENDS_BELOW_MINUS_100_PERCENT" if outside_domain else "NO_ASSESSED_RANGE_VIOLATION",
        "scenario_only": True, "forward_prediction_certified": False,
    }


def build_taiwan_channel_scenario(
    portfolio: Portfolio, exposure_report: ExposureReport,
    nvda_return_contrast: float | None = None, q_interval: Sequence[float] | None = None,
    *, nvda_contrast_interval: Sequence[float] | None = None,
    remaining_return_contrast: float | None = None,
    remaining_contrast_interval: Sequence[float] | None = None,
    q_reference: float | None = None,
) -> dict[str, Any]:
    """Declared NVDA Taiwan channel and a break-even contrast on the remainder.

The verified channel is direct NVDA plus identified NVDA inside ETFs. It is
not all Taiwan exposure. An unidentified fund residual remains unresolved.
No shock is defaulted. Unless all portfolio mass is identified NVDA, a full
portfolio contrast also requires an explicit assumption for the remaining
weight (including any unresolved holdings). This is a sensitivity scenario,
not an optimizer, capital allocation rule, or loss-budget certificate.
    """
    if not isinstance(portfolio, Portfolio) or not isinstance(exposure_report, ExposureReport):
        raise ValueError("Expected the validated Portfolio and its authoritative ExposureReport.")
    mass, channel, ledger = defaultdict(list), defaultdict(list), []
    positions_by_symbol = {p.symbol: p for p in portfolio.positions}
    for entry in exposure_report.entries:
        if entry.original_symbol not in positions_by_symbol:
            raise ValueError("Exposure ledger belongs to a different portfolio.")
        weight = _number(entry.portfolio_weight, "Exposure weight")
        if weight < 0 or entry.status not in {"direct", "look_through", "unresolved_etf"}:
            raise ValueError("Exposure ledger contains unsupported status or negative mass.")
        mass[entry.original_symbol].append(weight)
        if entry.symbol == "NVDA" and entry.asset_type == "stock" and entry.status in {"direct", "look_through"}:
            channel[entry.original_symbol].append(weight)
            ledger.append({"symbol": "NVDA", "original_symbol": entry.original_symbol, "portfolio_weight": weight,
                           "status": entry.status, "as_of": entry.as_of, "retrieved_at": entry.retrieved_at, "source_url": entry.source_url})
    for position in portfolio.positions:
        if not math.isclose(math.fsum(mass[position.symbol]), position.weight, abs_tol=WEIGHT_TOLERANCE, rel_tol=0):
            raise ValueError("Exposure ledger does not preserve each original position's mass.")
    unresolved = math.fsum(e.portfolio_weight for e in exposure_report.entries if e.status == "unresolved_etf")
    if not math.isclose(unresolved, exposure_report.unresolved_weight, abs_tol=WEIGHT_TOLERANCE, rel_tol=0):
        raise ValueError("Exposure ledger unresolved mass disagrees with its report.")
    channel_by_symbol = {symbol: math.fsum(weights) for symbol, weights in channel.items()}
    channel_weight = math.fsum(channel_by_symbol.values())
    remaining_weight = max(0.0, 1.0 - channel_weight)
    shock = None if nvda_return_contrast is None else _number(nvda_return_contrast, "Explicit NVDA YES-minus-NO return contrast")
    shock_bounds = None if shock is None else _interval(nvda_contrast_interval, shock, "NVDA contrast interval")
    if shock is None and nvda_contrast_interval is not None:
        raise ValueError("An NVDA interval also requires an explicit point scenario.")
    rest = None if remaining_return_contrast is None else _number(remaining_return_contrast, "Remaining return contrast")
    rest_bounds = None if rest is None else _interval(remaining_contrast_interval, rest, "Remaining contrast interval")
    if rest is None and remaining_contrast_interval is not None:
        raise ValueError("A remainder interval also requires an explicit point scenario.")
    shock_domain = _return_domain(shock, shock_bounds, q_interval, q_reference)
    rest_domain = _return_domain(rest, rest_bounds, q_interval, q_reference)
    outside_domain = any(d["status"] == "RANGE_EXTENDS_BELOW_MINUS_100_PERCENT" for d in (shock_domain, rest_domain))
    complete = shock is not None and (remaining_weight <= WEIGHT_TOLERANCE or rest is not None)
    known = None if shock is None else portfolio.notional * channel_weight * shock
    known_bounds = None if shock_bounds is None else _product_interval([portfolio.notional * channel_weight] * 2, shock_bounds)
    delta, bounds = None, None
    if complete:
        rest_amount = portfolio.notional * remaining_weight
        rest_pnl_bounds = _product_interval([rest_amount] * 2, rest_bounds or [0.0, 0.0])
        delta = known + rest_amount * (rest or 0.0)
        bounds = [known_bounds[i] + rest_pnl_bounds[i] for i in (0, 1)]
    positions = []
    for position in portfolio.positions:
        own_channel = channel_by_symbol.get(position.symbol, 0.0)
        other = max(0.0, position.weight - own_channel)
        own_known = None if shock is None else portfolio.notional * own_channel * shock
        positions.append({
            "symbol": position.symbol, "weight": position.weight, "nvda_channel_weight": own_channel,
            "remaining_weight": other, "known_channel_delta_pnl_usd": own_known,
            "delta_pnl_usd": None if shock is None or other > WEIGHT_TOLERANCE and rest is None else own_known + portfolio.notional * other * (rest or 0.0),
        })
    break_even = {
        "status": "NOT_ESTIMATED", "remaining_return_contrast": None, "remaining_return_contrast_interval": None,
        "definition": "A common YES-minus-NO return contrast on all non-channel weight that offsets the declared NVDA-channel dollar contrast.",
        "formula": "-NVDA_channel_weight * NVDA_return_contrast / remaining_weight",
        "interpretation": "An algebraic sensitivity threshold, not an allocation recommendation or forecast.",
    }
    if shock is not None and remaining_weight > WEIGHT_TOLERANCE:
        scale = -channel_weight / remaining_weight
        break_even.update(status="DECLARED_SCENARIO_BREAK_EVEN", remaining_return_contrast=scale * shock,
                          remaining_return_contrast_interval=_product_interval([scale] * 2, shock_bounds))
    elif shock is not None:
        break_even["status"] = "NOT_APPLICABLE_NO_REMAINING_EXPOSURE" if shock == 0 else "IMPOSSIBLE_WITH_NO_REMAINING_EXPOSURE"
    return {
        "schema": SCHEMA_VERSION, "event_family": "TAIWAN", "scenario_only": True,
        "status": "NOT_ESTIMATED_NO_DECLARED_SHOCK" if shock is None else "DECLARED_PORTFOLIO_SCENARIO" if complete else "PARTIAL_CHANNEL_SCENARIO",
        "scope": "FULL_PORTFOLIO_UNDER_EXPLICIT_REMAINDER_ASSUMPTION" if complete and remaining_weight > WEIGHT_TOLERANCE else "FULL_IDENTIFIED_NVDA_PORTFOLIO" if complete else "IDENTIFIED_NVDA_CHANNEL_ONLY",
        "notional_usd": portfolio.notional, "channel_weight": channel_weight, "remaining_weight": remaining_weight,
        "unresolved_holdings_weight": unresolved, "channel_ledger": ledger, "positions": positions,
        "nvda_return_contrast": shock, "nvda_contrast_interval": shock_bounds,
        "remaining_return_contrast": rest, "remaining_contrast_interval": rest_bounds,
        "price_domain": {"nvda": shock_domain, "remainder": rest_domain},
        "price_domain_status": "RANGE_EXTENDS_BELOW_MINUS_100_PERCENT" if outside_domain else "NO_ASSESSED_RANGE_VIOLATION",
        "delta_pnl_usd": delta, "delta_pnl_interval_usd": bounds,
        "known_component_delta_pnl_usd": known, "known_component_delta_pnl_interval_usd": known_bounds,
        "missing_symbols": [] if complete else [p["symbol"] for p in positions if p["delta_pnl_usd"] is None and p["weight"] > 0],
        "priced_in": _priced_in(delta, bounds, q_interval, q_reference),
        "known_component_priced_in": _priced_in(known, known_bounds, q_interval, q_reference),
        "break_even": break_even, "forward_prediction_certified": False,
        "assumptions": [
            "The declared contrast concerns the exact Taiwan contract YES versus its full NO complement; a NO outcome need not mean de-escalation.",
            "Only identified NVDA stock weight is included in the semiconductor channel; other Taiwan channels are not estimated.",
            "ETF holdings are dated identities, not event sensitivities; fees, derivatives and tracking effects are excluded.",
            "An explicit remainder contrast, when supplied, applies to all remaining mass including unidentified holdings; it does not resolve their identities.",
        ],
    }
