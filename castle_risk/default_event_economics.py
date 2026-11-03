"""Declared macro announcement components from a source-constrained outcome set.

This is an explanatory scenario model, not a future-return predictor. A source
distribution supplies settlement probabilities; a separately declared transfer
model supplies economic state values and sensitivities. The two are never
silently identified. No network, clock read, refit, or allocation is performed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
import math

import numpy as np
from scipy.optimize import linprog

from .portfolio import Portfolio, ExposureReport, _finite_number
from .empirical_event_impact import build_taiwan_channel_scenario, _prices, _returns
from .event_impact import DurationEvidence, MAX_DURATION_AGE_DAYS, MAX_NET_YIELD_SHOCK_BPS
from .fund_composition import uncovered_bond_symbols
from .scheduled_event_timing import reviewed_macro_window


FOMC_SOURCE = {
    "title": "Bernanke and Kuttner (2004), What Explains the Stock Market's Reaction to Federal Reserve Policy?",
    "url": "https://www.federalreserve.gov/pubs/feds/2004/200416/200416pap.pdf",
    "locator": "Table 2, printed page 40, surprise-change row, columns (b) and (d)",
    "sample": "June 1989 to December 2002; 131 events, 125 excluding six influential observations",
    "published_point": -4.68,
    "published_absolute_t": 3.03,
    "robustness_point": -2.55,
    "robustness_absolute_t": 2.79,
    "published_unit": "percentage points of one-day broad-equity return per one percentage point policy-rate surprise",
}
CPI_SOURCE = {
    "title": "Knox and Timmer (2025), Stagflationary Stock Returns, FEDS 2025-056",
    "url": "https://www.federalreserve.gov/econres/feds/files/2025056pap.pdf",
    "locator": "Table I, Panel A, nominal 1-day Return column, printed page 58; metric definition in section 3.1",
    "sample": "1999 to 2022; 283 announcements",
    "published_point": -1.42,
    "published_robust_standard_error": 0.58,
    "published_unit": "percentage points of aggregate nominal stock return per one percentage point headline month-on-month CPI surprise",
}


def _number(value, name):
    return _finite_number(value, name)


def _range(value, name, *, point=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " requires two endpoints")
    lo, hi = (_number(x, name) for x in value)
    if lo > hi or point is not None and not lo <= point <= hi:
        raise ValueError(name + " is unordered or excludes its point")
    return [lo, hi]


def _product(*ranges):
    result = [1., 1.]
    for limits in ranges:
        products = [a * b for a in result for b in limits]
        if not all(math.isfinite(x) for x in products):
            raise ValueError("Economic arithmetic overflowed")
        result = [min(products), max(products)]
    return result


def published_response_model(family, metric_identity=None):
    """Return primary-source slopes with explicit, non-predictive uncertainty.

The FOMC envelope includes approximate 95% intervals from BOTH the reported
full-sample and no-outlier specifications, reconstructed from rounded t values.
No sign constraint or truncation is applied to either interval.
    """
    if family in {"FED_POLICY", "FOMC"}:
        intervals = [[point - 1.96 * abs(point) / t, point + 1.96 * abs(point) / t]
                     for point, t in [(-4.68, 3.03), (-2.55, 2.79)]]
        return {
            "family": "FED_POLICY", "status": "DECLARED_HISTORICAL_RESPONSE_TRANSFER",
            "coefficient": -.000468,
            "coefficient_interval": [min(x[0] for x in intervals) / 10000,
                                     max(x[1] for x in intervals) / 10000],
            "coefficient_unit": "fractional_benchmark_return_per_policy_basis_point",
            "state_unit": "policy_basis_points", "source": deepcopy(FOMC_SOURCE),
            "interval_method": "Envelope of two approximate 95% normal intervals reconstructed from rounded heteroskedasticity-robust t statistics. It is neither a joint confidence interval nor a future prediction interval.",
            "reference_benchmark": "SPY as a declared proxy for the historical broad-equity index",
        }
    if family not in {"US_INFLATION", "CPI"}:
        raise ValueError("Unsupported macro family")
    # A source settlement statistic still needs an explicitly disclosed mapping
    # to the BLS headline MoM first-release statistic used by the historical fit.
    text = str(metric_identity).lower()
    if any(word in text for word in ("core", "yoy", "year_on_year", "year-over-year", "annual")):
        raise ValueError("The headline MoM CPI coefficient cannot be applied to core or year-on-year inflation")
    if "headline" not in text or not any(word in text for word in ("month_on_month", "month_over_month", "month-on-month", "mom")):
        raise ValueError("An explicit headline MoM CPI metric identity is required")
    return {
        "family": "US_INFLATION", "status": "DECLARED_HISTORICAL_RESPONSE_TRANSFER",
        "coefficient": -.0142,
        "coefficient_interval": [(-1.42 - 1.96 * .58) / 100, (-1.42 + 1.96 * .58) / 100],
        "coefficient_unit": "fractional_benchmark_return_per_headline_monthly_cpi_percentage_point",
        "state_unit": "headline_monthly_cpi_percentage_points", "source": deepcopy(CPI_SOURCE),
        "interval_method": "Approximate 95% normal interval from the published robust standard error; no prediction-interval or future-transfer coverage claim.",
        "reference_benchmark": "SPY as a declared proxy for the historical aggregate nominal stock return",
    }


def _polytope(distribution, atoms):
    n = len(atoms)
    if not 2 <= n <= 64:
        raise ValueError("A bounded source partition requires between 2 and 64 atoms")
    identifiers = [str(a.get("outcome_id", a.get("atom_id", ""))) for a in atoms]
    if not all(identifiers) or len(set(identifiers)) != n:
        raise ValueError("Outcome identities must be nonempty and unique")
    constraints = distribution.get("constraints")
    if not isinstance(constraints, Mapping):
        raise ValueError("The joint source probability constraints are required")
    if constraints.get("atom_ids") not in (None, identifiers):
        raise ValueError("Probability constraint order differs from atom order")
    kwargs = {}
    for matrix_key, rhs_key in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        matrix, rhs = constraints.get(matrix_key), constraints.get(rhs_key)
        if matrix is None or len(matrix) == 0:
            if rhs not in (None, []):
                raise ValueError("Missing probability constraint matrix")
            continue
        if not isinstance(matrix, (list, tuple)) or len(matrix) != len(rhs):
            raise ValueError("Probability constraint dimensions differ")
        rows = []
        for row in matrix:
            if len(row) != n:
                raise ValueError("Probability constraint dimension differs from atoms")
            rows.append([_number(x, "Probability constraint") for x in row])
        kwargs[matrix_key] = rows
        kwargs[rhs_key] = [_number(x, "Probability constraint") for x in rhs]
    # Always impose the simplex; never normalize quotes or accept negative mass.
    kwargs.setdefault("A_eq", []).append([1.] * n)
    kwargs.setdefault("b_eq", []).append(1.)
    raw_bounds = constraints.get("bounds", [[0., 1.]] * n)
    if len(raw_bounds) != n:
        raise ValueError("Probability bound count differs from atoms")
    bounds = [_range(v, "Probability atom bound") for v in raw_bounds]
    if any(lo < 0 or hi > 1 for lo, hi in bounds):
        raise ValueError("Atom probabilities must remain in [0, 1]")
    kwargs.update(bounds=bounds, method="highs")
    zero = linprog([0.] * n, **kwargs)
    if not zero.success:
        raise ValueError("Source probability polytope is infeasible")
    q = [_number(a.get("q_reference"), "Joint reference probability") for a in atoms]
    if abs(math.fsum(q) - 1.) > 1e-8:
        raise ValueError("Reference probabilities must already sum to one; no normalization")
    if any(not lo - 1e-8 <= p <= hi + 1e-8 for p, (lo, hi) in zip(q, bounds)):
        raise ValueError("Reference probabilities violate atom bounds")
    for matrix, rhs, equality in ((kwargs.get("A_ub", []), kwargs.get("b_ub", []), False),
                                  (kwargs["A_eq"], kwargs["b_eq"], True)):
        for row, target in zip(matrix, rhs):
            residual = math.fsum(a * b for a, b in zip(row, q)) - target
            if (abs(residual) if equality else residual) > 1e-8:
                raise ValueError("Reference probabilities violate joint source constraints")
    return kwargs, q, identifiers


def _linear_bounds(low_coefficients, high_coefficients, kwargs):
    low = linprog(low_coefficients, **kwargs)
    high = linprog([-x for x in high_coefficients], **kwargs)
    if not low.success or not high.success:
        raise ValueError("Economic probability optimization failed")
    return [float(low.fun), float(-high.fun)]


def _conditional_mean_bounds(values_low, values_high, selected, kwargs):
    """Linear-fractional extrema via Charnes-Cooper; positive event mass only."""
    n = len(values_low)
    indicator = [1. if i in selected else 0. for i in range(n)]
    mass = _linear_bounds(indicator, indicator, kwargs)
    if mass[1] <= 1e-12:
        return None
    a_ub = [list(row) + [-rhs] for row, rhs in zip(kwargs.get("A_ub", []), kwargs.get("b_ub", []))]
    b_ub = [0.] * len(a_ub)
    for i, (lo, hi) in enumerate(kwargs["bounds"]):
        row = [0.] * n
        row[i] = 1.
        a_ub.append(row + [-hi]); b_ub.append(0.)
        a_ub.append([-v for v in row] + [lo]); b_ub.append(0.)
    a_eq = [list(row) + [-rhs] for row, rhs in zip(kwargs["A_eq"], kwargs["b_eq"])]
    a_eq.append(indicator + [0.])
    b_eq = [0.] * (len(a_eq) - 1) + [1.]
    common = dict(A_ub=a_ub or None, b_ub=b_ub or None, A_eq=a_eq, b_eq=b_eq,
                  bounds=[(0., None)] * (n + 1), method="highs")
    low = linprog([v * m for v, m in zip(values_low, indicator)] + [0.], **common)
    high = linprog([-v * m for v, m in zip(values_high, indicator)] + [0.], **common)
    if not low.success or not high.success:
        raise ValueError("Conditional mixture optimization failed")
    return [float(low.fun), float(-high.fun)]


from .event_state_mapping import _state_model


ANNOUNCEMENT_DAY_SPECIFICATION = "ANNOUNCEMENT_DAY_WHERE_CHRONOLOGICALLY_CHECKED"
# Source distribution family -> the only event-day study family admitted for it.
EVENT_DAY_STUDY_FAMILY = {"FED_POLICY": "FOMC", "US_INFLATION": "CPI"}
EVENT_DAY_STUDY_SCHEMA = "CASTLE_EMPIRICAL_EVENT_IMPACT_V1"
EVENT_DAY_BETA_UNIT = "instrument_event_day_return_fraction_per_benchmark_event_day_return_fraction"


def _aware(value, name):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(name + " is not an ISO timestamp") from exc
    if moment.utcoffset() is None:
        raise ValueError(name + " must be timezone-aware")
    return moment


def _day(value, name):
    try:
        return date.fromisoformat(str(value)[:10]) if len(str(value)) == 10 else datetime.fromisoformat(str(value)).date()
    except ValueError as exc:
        raise ValueError(name + " is not an ISO date") from exc


def validate_event_day_report(report, family, *, not_after=None):
    """Reject an announcement-day study that is not this family's, is malformed, or uses later data.

The check is structural and chronological: matching source family and SPY
benchmark, a timezone-aware analysis time no later than ``not_after`` (the
source distribution time), training strictly before analysis, and, for every
instrument fit, training before the frozen split, holdout after it and not
after analysis, the declared unit, finite parameters inside their interval,
sample sizes meeting the declared minimum and a well-formed holdout check. It
cannot authenticate the prices behind the fit; their recorded hashes are kept.
    """
    if not isinstance(report, Mapping):
        raise ValueError("Announcement-day study must be a mapping")
    expected = EVENT_DAY_STUDY_FAMILY.get(family)
    if expected is None:
        raise ValueError("No announcement-day study family is defined for " + str(family))
    if report.get("kind") != "EVENT_DAY_BENCHMARK_ASSOCIATION" or report.get("benchmark_symbol") != "SPY":
        raise ValueError("Announcement-day transfer requires the same-family SPY event-day association")
    if report.get("family") != expected:
        raise ValueError("Announcement-day study family " + repr(report.get("family")) + " does not match " + expected)
    if report.get("schema", EVENT_DAY_STUDY_SCHEMA) != EVENT_DAY_STUDY_SCHEMA:
        raise ValueError("Unsupported announcement-day study schema")
    analysis = _aware(report.get("analysis_as_of"), "Announcement-day analysis time")
    if not_after is not None and analysis > not_after:
        raise ValueError("Announcement-day analysis must not follow the source distribution evaluation")
    split = _day(report.get("train_end"), "Announcement-day train_end")
    if split >= analysis.date():
        raise ValueError("Announcement-day training must precede analysis")
    fits = report.get("instruments")
    if not isinstance(fits, Mapping) or not fits:
        raise ValueError("Announcement-day study has no instrument fits")
    for symbol, fit in fits.items():
        label = "Announcement-day fit " + str(symbol)
        if not isinstance(fit, Mapping):
            raise ValueError(label + " is malformed")
        if not str(fit.get("status", "")).startswith("ESTIMATED"):
            continue
        if fit.get("benchmark_symbol", "SPY") != "SPY" or fit.get("symbol", symbol) != symbol:
            raise ValueError(label + " has a mismatched benchmark or symbol")
        if fit.get("beta_unit") != EVENT_DAY_BETA_UNIT:
            raise ValueError(label + " has an undeclared or different beta unit")
        if _day(fit.get("train_end"), label + " train_end") != split:
            raise ValueError(label + " train_end differs from the study split")
        spans = [fit.get("training_span"), fit.get("holdout_span")]
        if not all(isinstance(x, (list, tuple)) and len(x) == 2 for x in spans):
            raise ValueError(label + " lacks training/holdout spans")
        (t0, t1), (h0, h1) = [[_day(v, label + " span") for v in x] for x in spans]
        if not (t0 <= t1 <= split < h0 <= h1 <= analysis.date()):
            raise ValueError(label + " chronology is not training <= split < holdout <= analysis")
        beta = _number(fit.get("beta"), label + " beta")
        _range(fit.get("beta_interval"), label + " beta interval", point=beta)
        minimum = fit.get("minimum_samples") or {}
        for count, floor in (("n_train", "train"), ("n_holdout", "holdout")):
            value = fit.get(count)
            if isinstance(value, bool) or not isinstance(value, int) or value < int(minimum.get(floor, 1)):
                raise ValueError(label + " " + count + " is missing or below the declared minimum")
        check = fit.get("transfer_check")
        if not isinstance(check, Mapping) or check.get("status") not in {"PASS", "FAIL"}:
            raise ValueError(label + " lacks a PASS/FAIL holdout check")
        rmse = [check.get(k) for k in ("held_out_rmse", "zero_return_baseline_rmse", "frozen_training_mean_baseline_rmse")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in rmse):
            raise ValueError(label + " holdout check has nonfinite or missing errors")
        if (check["status"] == "PASS") != (rmse[0] < rmse[1] and rmse[0] < rmse[2]):
            raise ValueError(label + " holdout status contradicts its recorded errors")
    return {"status": "ADMITTED", "family": expected, "analysis_as_of": analysis.isoformat(),
            "train_end": split.isoformat(), "instruments": sorted(fits)}


def _announcement_day_rows(rows, event_day_report):
    """Replace a daily beta by the same family's announcement-day beta when checked.

The default component is a SPY announcement-day move. On announcement days the
instrument/SPY co-movement differs from ordinary days (for Treasury ETFs the
all-day beta is near zero while the FOMC-day beta is clearly positive), so the
announcement-day OLS is the better-conditioned transfer when its frozen
chronological holdout check passed. Otherwise neither specification is
validated for this family: the point stays at the daily beta and the interval
becomes the hull of both parameter intervals, so the disagreement widens the
envelope instead of being resolved by choice.
    """
    if event_day_report.get("kind") != "EVENT_DAY_BENCHMARK_ASSOCIATION" or event_day_report.get("benchmark_symbol") != "SPY":
        raise ValueError("Announcement-day transfer requires the same-family SPY event-day association")
    fits = event_day_report.get("instruments", {})
    result = []
    for row in rows:
        row = deepcopy(row)
        fit = fits.get(row["symbol"], {})
        estimated = (str(fit.get("status", "")).startswith("ESTIMATED") and fit.get("beta") is not None
                     and fit.get("beta_interval") is not None)
        row["daily_beta"], row["daily_beta_interval"] = row.get("beta"), deepcopy(row.get("beta_interval"))
        # Keep the observed checks in their original domains. A daily PASS is
        # not a PASS for a family whose announcement-day check failed or could
        # not be estimated. The effective scenario status is separate from
        # these raw diagnostics and does not certify a future forecast.
        row["daily_transfer_check"] = deepcopy(row.get("transfer_check", {}))
        row["announcement_day_fit"] = ({"beta": fit.get("beta"), "beta_interval": deepcopy(fit.get("beta_interval")),
                                        "n_train": fit.get("n_train"), "n_holdout": fit.get("n_holdout"),
                                        "transfer_check": deepcopy(fit.get("transfer_check", {}))}
                                       if estimated else {"status": fit.get("status", "NOT_AVAILABLE")})
        if row.get("beta") is None or row["basis"] == "UNCHANGED_USD_CASH_OR_ZERO_WEIGHT":
            row["scenario_transfer_status"] = ("NOT_APPLICABLE" if row["basis"] ==
                "UNCHANGED_USD_CASH_OR_ZERO_WEIGHT" else "NOT_TESTED")
            result.append(row)
            continue
        if estimated and fit.get("transfer_check", {}).get("status") == "PASS":
            beta = _number(fit["beta"], "Announcement-day beta")
            row.update(beta=beta, beta_interval=_range(fit["beta_interval"], "Announcement-day beta interval", point=beta),
                       basis="ANNOUNCEMENT_DAY_OLS_FROZEN_CHRONOLOGICAL_CHECK_PASS",
                       transfer_check=deepcopy(fit["transfer_check"]),
                       scenario_transfer_status="PASS",
                       n_train=fit.get("n_train"), n_holdout=fit.get("n_holdout"))
        elif estimated:
            other = _range(fit["beta_interval"], "Announcement-day beta interval")
            row.update(beta_interval=[min(row["beta_interval"][0], other[0]), max(row["beta_interval"][1], other[1])],
                       basis="SPECIFICATION_UNRESOLVED_HULL_OF_DAILY_AND_ANNOUNCEMENT_DAY_INTERVALS",
                       scenario_transfer_status=("FAIL" if fit.get("transfer_check", {}).get("status") == "FAIL"
                                                 else "NOT_TESTED"))
        else:
            row["basis"] = "DAILY_BETA_NO_ANNOUNCEMENT_DAY_ESTIMATE"
            row["scenario_transfer_status"] = "NOT_TESTED"
        result.append(row)
    return result


def _betas(portfolio, report, event_day_report=None, *, rate_sensitivity=None):
    if report.get("kind") != "INSTRUMENT_DAILY_BETAS":
        raise ValueError("Default transfer uses the declared daily-beta specification, not a selected sparse event-day fit")
    if report.get("benchmark_symbol") != "SPY":
        raise ValueError("The default historical broad-equity transfer requires daily betas measured against SPY")
    positions, missing, fail = [], [], []
    blocked_bonds = uncovered_bond_symbols(rate_sensitivity)
    for position in portfolio.positions:
        fit = report.get("instruments", {}).get(position.symbol, {})
        row = {"symbol": position.symbol, "weight": position.weight, "asset_type": position.asset_type}
        if position.asset_type == "cash" or position.weight == 0:
            row.update(beta=0., beta_interval=[0., 0.], basis="UNCHANGED_USD_CASH_OR_ZERO_WEIGHT", transfer_check={"status": "NOT_APPLICABLE"})
        elif position.symbol in blocked_bonds:
            missing.append(position.symbol)
            row.update(beta=None, beta_interval=None, basis="KNOWN_BOND_WITHOUT_DURATION_NO_EQUITY_PROXY",
                       transfer_check={"status": "NOT_APPLICABLE"})
        elif fit.get("beta") is None or fit.get("beta_interval") is None:
            missing.append(position.symbol)
            row.update(beta=None, beta_interval=None, basis="MISSING_INSTRUMENT_BETA")
        else:
            beta = _number(fit["beta"], "Daily beta")
            bounds = _range(fit["beta_interval"], "Daily beta parameter interval", point=beta)
            check = deepcopy(fit.get("transfer_check", {}))
            if check.get("status") == "FAIL":
                fail.append(position.symbol)
            row.update(beta=beta, beta_interval=bounds, basis="FROZEN_DAILY_OLS_BETA_TRANSFERRED_TO_EVENT_COMPONENT",
                       transfer_check=check, n_train=fit.get("n_train"), n_holdout=fit.get("n_holdout"))
        row["scenario_transfer_status"] = row.get("transfer_check", {}).get("status", "NOT_TESTED")
        positions.append(row)
    if event_day_report is not None:
        positions = _announcement_day_rows(positions, event_day_report)
        # ``fail`` records the raw daily diagnostic. Effective family-scenario
        # failures are reported separately and never erase this history.
    known = [r for r in positions if r["beta"] is not None]
    point = math.fsum(r["weight"] * r["beta"] for r in known)
    bounds = [math.fsum(r["weight"] * r["beta_interval"][i] for r in known) for i in (0, 1)]
    return positions, missing, fail, point, bounds


def _historical_move_context(portfolio, report, prices):
    if not prices:
        return {"status": "NOT_COMPUTED_NO_RAW_PRICES", "prediction_interval": False}
    as_of = datetime.fromisoformat(report["analysis_as_of"].replace("Z", "+00:00"))
    series, _ = _prices(prices, as_of)
    returns = {symbol: _returns(rows) for symbol, rows in series.items()}
    noncash = [p for p in portfolio.positions if p.weight and p.asset_type != "cash"]
    if any(p.symbol not in returns for p in noncash):
        return {"status": "NOT_COMPUTED_MISSING_POSITION_PRICES", "prediction_interval": False}
    common = set.intersection(*(set(returns[p.symbol]) for p in noncash)) if noncash else set()
    train_end = datetime.fromisoformat(report["train_end"]).date()
    observations = [portfolio.notional * math.fsum(p.weight * returns[p.symbol][edge] for p in noncash)
                    for edge in sorted(common) if edge[1] > train_end]
    if len(observations) < 20:
        return {"status": "INSUFFICIENT_HELD_OUT_DAYS", "n": len(observations), "prediction_interval": False}
    return {"status": "HISTORICAL_UNCONDITIONAL_DAILY_MOVE_CONTEXT", "n": len(observations),
            "empirical_central_95_percent_range_usd": [float(x) for x in np.quantile(observations, [.025, .975])],
            "rms_daily_move_usd": float(np.sqrt(np.mean(np.square(observations)))),
            "prediction_interval": False,
            "interpretation": "Held-out observed whole-portfolio daily price moves, with all contemporaneous news. Neither conditioned on this event nor added to the component as a calibrated future prediction interval."}


def build_default_macro_economics(portfolio, distribution, daily_beta_report, *,
                                  release_day_report=None, prices_by_symbol=None,
                                  beta_specification="DAILY", rate_sensitivity=None):
    """Explain each settlement outcome with a joint-Q-centered dollar component.

Every numeric result is conditional on the displayed historical-response,
instrument-beta, source-statistic and asset-pricing-measure transfer assumptions.
The combined envelope covers the stated parameter/scenario rectangle only.
    """
    if beta_specification not in {"DAILY", ANNOUNCEMENT_DAY_SPECIFICATION}:
        raise ValueError("Unknown beta specification")
    if beta_specification != "DAILY" and not release_day_report:
        raise ValueError("The announcement-day specification needs the same-family release-day study")
    if not isinstance(portfolio, Portfolio):
        raise ValueError("A validated original-position portfolio is required")
    beta_time = datetime.fromisoformat(daily_beta_report["analysis_as_of"].replace("Z", "+00:00"))
    if beta_time.utcoffset() is None:
        raise ValueError("Beta analysis time must be timezone-aware")
    source_time = distribution.get("evaluated_at", distribution.get("evaluated_at_utc"))
    if source_time is not None:
        source_time = datetime.fromisoformat(source_time.replace("Z", "+00:00"))
        if source_time.utcoffset() is None or beta_time > source_time:
            raise ValueError("Economic input analysis must not follow the source distribution evaluation")
    cutoff = source_time or beta_time
    if not reviewed_macro_window(
            distribution.get("family"), distribution["event_date"], cutoff,
            source_close_at=distribution.get("source_market_close_at_utc"))["pending"]:
        raise ValueError("Default forward components require a future event at the declared evaluation time")
    if datetime.fromisoformat(daily_beta_report["train_end"]).date() >= beta_time.date():
        raise ValueError("Daily-beta training must precede analysis")
    family = distribution.get("family")
    response = published_response_model(family, distribution.get("metric_identity"))
    if beta_specification != "DAILY":
        validate_event_day_report(release_day_report, response["family"], not_after=source_time or beta_time)
    atoms = distribution.get("atoms") or distribution.get("outcomes")
    if not isinstance(atoms, list):
        raise ValueError("A source outcome partition is required")
    kwargs, q, identifiers = _polytope(distribution, atoms)
    state = _state_model(atoms, response["family"])
    x = [r["value"] for r in state]
    xl = [r["value_interval"][0] for r in state]
    xh = [r["value_interval"][1] for r in state]
    mean = math.fsum(p * value for p, value in zip(q, x))
    mean_range = _linear_bounds(xl, xh, kwargs)
    positions, missing, failed, aggregate_beta, beta_bounds = _betas(
        portfolio, daily_beta_report, release_day_report if beta_specification != "DAILY" else None,
        rate_sensitivity=rate_sensitivity)
    scenario_failed = [r["symbol"] for r in positions if r["scenario_transfer_status"] == "FAIL"]
    scenario_untested = [r["symbol"] for r in positions if r["scenario_transfer_status"] == "NOT_TESTED"]
    modeled_weight = math.fsum(row["weight"] for row in positions if row["beta"] is not None)
    coefficient, c_bounds = response["coefficient"], response["coefficient_interval"]
    scale = portfolio.notional * aggregate_beta * coefficient
    scale_bounds = _product([portfolio.notional] * 2, beta_bounds, c_bounds)
    all_cases = []
    unrestricted_atoms = []
    for i, atom in enumerate(atoms):
        source_bounds = atom.get("value_interval") or [None, None]
        if any(value is None for value in source_bounds):
            indicator = [float(j == i) for j in range(len(atoms))]
            source_mass = _linear_bounds(indicator, indicator, kwargs)
            if source_mass[1] > 1e-10:
                unrestricted_atoms.append(identifiers[i])
    for j, atom in enumerate(atoms):
        indicator = [float(i == j) for i in range(len(atoms))]
        q_bounds = _linear_bounds(indicator, indicator, kwargs)
        # x_j - sum q_i*x_i. Coefficients retain the same q_j on both
        # occurrences, so the own-state uncertainty is not double counted.
        low_coeff = [-xl[j] if i == j else -xh[i] for i in range(len(atoms))]
        high_coeff = [-xh[j] if i == j else -xl[i] for i in range(len(atoms))]
        raw_limits = _linear_bounds(low_coeff, high_coeff, kwargs)
        surprise_range = [xl[j] + raw_limits[0], xh[j] + raw_limits[1]]
        surprise = x[j] - mean
        point = scale * surprise
        full_range = _product(scale_bounds, surprise_range)
        quote_point_range = _linear_bounds([-v for v in x], [-v for v in x], kwargs)
        quote_only = _product([scale] * 2, [x[j] + quote_point_range[0], x[j] + quote_point_range[1]])
        other = [i for i in range(len(atoms)) if i != j]
        no_range = _conditional_mean_bounds(xl, xh, other, kwargs)
        no_value = math.fsum(q[i] * x[i] for i in other) / (1. - q[j]) if q[j] < 1. - 1e-12 else None
        delta_surprise = None if no_value is None else x[j] - no_value
        delta = None if delta_surprise is None else scale * delta_surprise
        delta_range = None if no_range is None else _product(scale_bounds, [xl[j] - no_range[1], xh[j] - no_range[0]])
        no_pnl = None if delta is None else -q[j] * delta
        no_pnl_range = None if delta_range is None else _product(delta_range, [-q_bounds[1], -q_bounds[0]])
        price_domain = all(min(_product(r["beta_interval"], c_bounds, surprise_range)) >= -1.
                           for r in positions if r["beta_interval"] is not None)
        case_positions = []
        for row in positions:
            contribution = deepcopy(row)
            if row["beta"] is not None:
                contribution.update(component_pnl_usd=portfolio.notional * row["weight"] * row["beta"] * coefficient * surprise,
                                    component_pnl_interval_usd=_product([portfolio.notional * row["weight"]] * 2, row["beta_interval"], c_bounds, surprise_range))
            else:
                contribution.update(component_pnl_usd=None, component_pnl_interval_usd=None)
            case_positions.append(contribution)
        complete = not missing and price_domain
        sign = "POSITIVE_UNDER_DECLARED_MODEL" if full_range[0] > 0 else "NEGATIVE_UNDER_DECLARED_MODEL" if full_range[1] < 0 else "DIRECTION_NOT_IDENTIFIED_WITHIN_DECLARED_MODEL"
        all_cases.append({
            "outcome_id": identifiers[j], "label": atom.get("label", identifiers[j]),
            "q_interval": q_bounds, "q_reference": q[j], "physical_probability_P": None,
            "state_model": state[j], "reference_expectation": mean,
            "surprise": surprise, "surprise_interval": surprise_range,
            "benchmark_component_return": coefficient * surprise,
            "component_pnl_usd": point if complete else None,
            "component_pnl_interval_usd": full_range if complete else None,
            "known_position_component_pnl_usd": point,
            "known_position_component_pnl_interval_usd": full_range,
            "finite_state_scenario_required_for_this_case": bool(unrestricted_atoms) or any(value is None for value in state[j]["source_value_interval"]),
            "delta_pnl_usd": delta if complete else None,
            "delta_pnl_interval_usd": delta_range if complete else None,
            "priced_in": {
                "status": "DECLARED_SAME_EVENT_Q_TO_ASSET_PRICING_TRANSFER" if complete else "INCOMPLETE_COMPONENT",
                "q_reference": q[j], "q_interval": q_bounds,
                "YES": point if complete else None, "NO": no_pnl if complete else None,
                "YES_interval": full_range if complete else None,
                "NO_interval": no_pnl_range if complete else None,
                "modeled_priced_component_usd": q[j] * delta if complete and delta is not None else None,
                "remaining_fraction_if_yes": 1. - q[j],
                "NO_mean_state": no_value, "NO_mean_state_interval": no_range,
                "NO_definition": "Conditional mixture of every other source atom under the same reference Q; not an opposite policy decision.",
                "asset_pricing_weight_empirically_identified": False,
            },
            "positions": case_positions, "scope": "FULL_ORIGINAL_PORTFOLIO_ANNOUNCEMENT_COMPONENT" if complete else "PARTIAL_COMPONENT",
            "sign_status": sign if complete else "TOTAL_UNAVAILABLE",
            "interval_layers": {
                "quote_only_fixed_model_usd": quote_only,
                "quote_state_coefficient_beta_envelope_usd": full_range,
                "sign_reversal_regime_sensitivity_usd": _product([portfolio.notional] * 2, beta_bounds, [-max(abs(c) for c in c_bounds), max(abs(c) for c in c_bounds)], surprise_range),
                "future_total_return_prediction_interval": None,
                "interpretation": "The main interval is a conditional scenario/parameter envelope, not a 95% future-return band. Sign reversal is a separate structural stress; unmodeled news can dominate either component.",
            },
            "unmodeled_news_break_even_usd": -point if complete else None,
            "price_domain_status": "NO_ASSESSED_VIOLATION" if price_domain else "COMPONENT_RANGE_BELOW_MINUS_100_PERCENT",
            "source_contract_ids": atom.get("source_contract_ids", []),
        })
    weighted_zero = math.fsum(row["q_reference"] * row["known_position_component_pnl_usd"] for row in all_cases)
    if abs(weighted_zero) > 1e-8 * max(1., portfolio.notional):
        raise ValueError("The centered component failed its joint-Q accounting identity")
    return {
        "schema": "castle.default_macro_economics.v1",
        "status": "DECLARED_DEFAULT_COMPONENT_MODEL" if not missing else "PARTIAL_DECLARED_COMPONENT_MODEL",
        **({} if beta_specification == "DAILY" else {"beta_specification": beta_specification}),
        "family": response["family"], "event_date": distribution.get("event_date"),
        "metric_identity": deepcopy(distribution.get("metric_identity")),
        "response_model": response, "state_models": state, "outcome_cases": all_cases,
        "portfolio_notional_usd": portfolio.notional, "reference_expectation": mean,
        "reference_expectation_interval": mean_range,
        "aggregate_daily_beta": aggregate_beta if not missing else None,
        "aggregate_daily_beta_interval": beta_bounds if not missing else None,
        "known_position_aggregate_daily_beta": aggregate_beta,
        "known_position_aggregate_daily_beta_interval": beta_bounds,
        "modeled_weight": modeled_weight,
        "unrestricted_support_review": {
            "status": "REFERENCE_EXPECTATION_NOT_FINITELY_BOUNDED_BY_SOURCE_SUPPORT" if unrestricted_atoms else "SOURCE_SUPPORTED_ACTIVE_MAGNITUDES_ARE_FINITE",
            "unbounded_support_atoms_with_positive_possible_mass": unrestricted_atoms,
            "finite_economic_domain_is_declared_assumption": bool(unrestricted_atoms),
            "explanation": "An open source tail with positive feasible mass prevents a finite source-only expectation bound. Displayed economic ranges condition on the declared finite tail domain; they do not certify a global bound.",
        },
        "missing_symbols": missing, "daily_transfer_failed_symbols": failed,
        "scenario_transfer_failed_symbols": scenario_failed,
        "scenario_transfer_untested_symbols": scenario_untested,
        "scenario_transfer_status": ("FAIL" if scenario_failed else "NOT_TESTED" if scenario_untested
            else "PASS" if any(r["scenario_transfer_status"] == "PASS" for r in positions) else "NOT_APPLICABLE"),
        "scenario_transfer_check_scope": "Chronological association check for the selected specification only; not causal identification or future forecast validation.",
        "release_day_diagnostics": {s: deepcopy(fit.get("transfer_check", {})) for s, fit in (release_day_report or {}).get("instruments", {}).items()},
        "prior_A6_forward_transfer_status": "FAIL_RETAINED_NOT_OVERRIDDEN",
        "q_weighted_component_sum_usd": weighted_zero,
        "historical_move_context": _historical_move_context(portfolio, daily_beta_report, prices_by_symbol),
        "physical_probability_P": None, "future_total_return_prediction_available": False,
        "assumptions": [
            "The historical broad-equity slope transfers to the SPY announcement component in this future event; daily instrument betas transfer to that component. These are declared model assumptions, not causal or future-transfer certifications.",
            "The source-contract Q is used as a proxy for the reference expectation embedded in this asset-price component, frozen until announcement. Its equality to physical probabilities or actual asset-specific pricing weights is not established.",
            "For CPI, the source settlement variable is mapped to headline month-on-month first-release CPI under the ordinary-release assumption. Core, year-on-year and annual statistics are rejected; source cancellation/revision exceptions are not estimated effects.",
            "Finite representatives and domains for open or opaque outcomes are explicit economic scenarios. They do not bound the source support. Their unrestricted-support sensitivity can be unbounded and is not concealed by the finite main case.",
            "The envelope combines source quote constraints with historical parameter intervals and beta intervals as a rectangle. It has no joint 95% coverage claim and omits model changes, information effects, other news, carry and idiosyncratic future shocks.",
            "Each original portfolio position is counted once. ETF constituents are not added to whole-ETF returns; holdings identify channel exposure elsewhere.",
            "Zero Q-weighted component is a centering identity, not empirical validation of asset prices or zero expected physical return.",
            *([] if beta_specification == "DAILY" else [
                "Instrument transfer uses the same family's announcement-day SPY beta where its frozen chronological holdout check passed; elsewhere the point stays at the daily beta and the interval is the hull of both specifications. Neither is a causal response to the surprise itself."]),
        ],
    }


def build_default_channel_sensitivity(portfolio, exposure_report, *, family,
                                      q_interval=None, event_definition=None):
    """Reviewed multiissuer sensitivities, retaining the legacy NVDA component.

Useful even when no matching probability exists: dollars per 1% NVDA shock and
the offsetting return required on the remainder. The stress sizes are explicitly
illustrative and never represented as historical event estimates.
    """
    if not isinstance(portfolio, Portfolio) or not isinstance(exposure_report, ExposureReport):
        raise ValueError("The original portfolio and authoritative exposure ledger are required")
    if family not in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}:
        raise ValueError("Unsupported identified channel")
    point, bounds = (-.20, [-.35, -.10]) if family == "TAIWAN_SUPPLY" else (-.10, [-.20, -.05])
    result = build_taiwan_channel_scenario(portfolio, exposure_report,
        nvda_return_contrast=point, nvda_contrast_interval=bounds, q_interval=q_interval)
    result.update(schema="castle.default_identified_channel.v1", family=family,
        event_definition=event_definition,
        shock_basis="ILLUSTRATIVE_DECLARED_NVDA_LOSS_STRESS_NOT_EMPIRICALLY_ESTIMATED_EVENT_EFFECT",
        dollars_per_one_percent_nvda_return=portfolio.notional * result["channel_weight"] * .01,
        unrelated_position_return_imputed=False,
        matched_probability_required_for_priced_in=True,
        physical_probability_P=None)
    from .geopolitical_channels import build_multiissuer_sensitivity
    result["multiissuer_sensitivity"] = build_multiissuer_sensitivity(
        portfolio, exposure_report,
        profile_name="TAIWAN_MAJOR_MILITARY_ACTION" if family == "TAIWAN_SUPPLY" else "US_AI_CHIP_EXPORT_CURB",
        nvda_spec={"point": point, "range": bounds, "unit": "NVDA residual return",
                   "basis": "ILLUSTRATIVE_DECLARED_NVDA_LOSS_STRESS_NOT_EMPIRICALLY_ESTIMATED_EVENT_EFFECT"})
    result["legacy_nvda_component_scope"] = "NVDA_ONLY_BACKWARD_COMPATIBLE_COMPONENT_SEE_MULTIISSUER_SENSITIVITY"
    return result


def build_tlt_duration_comparison(portfolio, macro_economics, duration_evidence, *, analysis_as_of):
    """Compare a source-backed TLT duration channel with its default beta channel.

This optional model REPLACES the TLT component; it is never added to the
whole-ETF component. Other instruments retain the declared default model.
The approximate yield-slope interval is a historical parameter interval, not
a prediction band. The caller must obtain evidence through the archive loader
if archive integrity is to be asserted; a dataclass alone is not a hash check.
    """
    if not isinstance(portfolio, Portfolio):
        raise ValueError("A validated original-position portfolio is required")
    tlt = [p for p in portfolio.positions if p.symbol == "TLT" and p.weight]
    if not tlt:
        return {"status": "NOT_APPLICABLE_NO_TLT_POSITION", "replaces_default": False}
    if macro_economics.get("family") != "FED_POLICY":
        raise ValueError("The reviewed TLT yield-response comparison supports only Fed policy surprises")
    if not isinstance(duration_evidence, DurationEvidence):
        raise ValueError("Reviewed TLT duration evidence is required")
    stamp = analysis_as_of if isinstance(analysis_as_of, datetime) else datetime.fromisoformat(analysis_as_of.replace("Z", "+00:00"))
    received = datetime.fromisoformat(duration_evidence.retrieved_at.replace("Z", "+00:00"))
    if stamp.utcoffset() is None or received > stamp:
        raise ValueError("Duration evidence must precede the timezone-aware analysis")
    age = (stamp.date() - date.fromisoformat(duration_evidence.data_as_of)).days
    if not 0 <= age <= MAX_DURATION_AGE_DAYS:
        raise ValueError("Duration evidence is outside the reviewed freshness window")
    if macro_economics.get("portfolio_notional_usd") != portfolio.notional:
        raise ValueError("Comparison notional must match the default model")
    # Kuttner (2001), Table 3: 19.4 yield bp per 100 policy bp; t = 2.3.
    slope = .194
    slope_range = [slope - 1.96 * slope / 2.3, slope + 1.96 * slope / 2.3]
    duration = duration_evidence.effective_duration_years
    coefficient = -duration * slope / 10000
    coefficient_range = [-duration * slope_range[1] / 10000, -duration * slope_range[0] / 10000]
    weight = tlt[0].weight
    expected_weights = {p.symbol: p.weight for p in portfolio.positions}
    cases = []
    for case in macro_economics.get("outcome_cases", []):
        positions = case["positions"]
        if len(positions) != len(expected_weights) or {p["symbol"]: p["weight"] for p in positions} != expected_weights:
            raise ValueError("Comparison positions must match the original portfolio exactly")
        original_tlt = next(p for p in positions if p["symbol"] == "TLT")
        other = [p for p in positions if p["symbol"] != "TLT"]
        surprise = _number(case["surprise"], "Fed policy surprise")
        surprise_range = _range(case["surprise_interval"], "Fed policy surprise interval", point=surprise)
        yield_range = _product(slope_range, surprise_range)
        return_range = _product(coefficient_range, surprise_range)
        in_domain = max(abs(v) for v in yield_range) <= MAX_NET_YIELD_SHOCK_BPS and return_range[0] >= -1.
        point = portfolio.notional * weight * coefficient * surprise
        bounds = _product([portfolio.notional * weight] * 2, coefficient_range, surprise_range)
        other_complete = all(p.get("component_pnl_usd") is not None for p in other)
        other_domain = other_complete and all(
            p["component_pnl_interval_usd"][0] >= -portfolio.notional * p["weight"] for p in other)
        complete = in_domain and other_domain
        total = point + math.fsum(p["component_pnl_usd"] for p in other) if complete else None
        total_range = [bounds[i] + math.fsum(p["component_pnl_interval_usd"][i] for p in other) for i in (0, 1)] if complete else None
        cases.append({
            "outcome_id": case["outcome_id"], "q_reference": case["q_reference"],
            "default_tlt_component_usd": original_tlt.get("component_pnl_usd"),
            "replacement_tlt_component_usd": point,
            "replacement_tlt_component_interval_usd": bounds,
            "replacement_portfolio_component_usd": total,
            "replacement_portfolio_component_interval_usd": total_range,
            "yield_surprise_bps": slope * surprise, "yield_surprise_interval_bps": yield_range,
            "linear_duration_domain_status": "WITHIN_REVIEWED_50BP_NET_YIELD_DOMAIN" if in_domain else "OUTSIDE_REVIEWED_DOMAIN_TOTAL_UNAVAILABLE",
            "other_position_price_domain_checked": bool(other_domain),
            "finite_state_scenario_required_for_this_case": case["finite_state_scenario_required_for_this_case"],
        })
    return {
        "schema": "castle.tlt_duration_model_comparison.v1",
        "status": "DECLARED_ALTERNATIVE_REPLACING_TLT_ONLY", "replaces_default": False,
        "replacement_rule": "Replace the TLT daily-beta component; never sum both TLT channels. All other original positions retain their default component.",
        "source": {
            "title": "Kuttner (2001), Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market",
            "url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr99.pdf",
            "locator": "Table 3, 30-year maturity, response to surprise change; printed page 11",
            "sample": "42 target-rate changes, June 6, 1989 to February 2, 2000",
            "published_point": 19.4, "published_absolute_t": 2.3,
            "published_unit": "30-year Treasury yield basis points per 100 policy-surprise basis points",
        },
        "duration_evidence": duration_evidence.to_dict(), "duration_age_days": age,
        "yield_response_per_policy_bp": slope, "yield_response_interval": slope_range,
        "tlt_fractional_return_per_policy_bp": coefficient,
        "tlt_fractional_return_coefficient_interval": coefficient_range,
        "outcome_cases": cases,
        "q_weighted_replacement_tlt_component_usd": math.fsum(c["q_reference"] * c["replacement_tlt_component_usd"] for c in cases),
        "future_total_return_prediction_available": False,
        "assumptions": [
            "Historical 30-year yield response transfers to the future shock and proxies a parallel shift of TLT's yield curve. Duration is frozen; convexity, carry, credit and curve-shape changes are omitted.",
            "The approximate normal parameter interval is reconstructed from a rounded t statistic. Summed position envelopes are conservative outer bounds, not sharp joint bounds or a joint 95% confidence band.",
            "This is a model comparison, not a selection based on a desired sign. The uniform daily-beta default remains unchanged. No BND or IEF duration is inferred from a ticker.",
        ],
    }
