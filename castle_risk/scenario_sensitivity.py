"""Deterministic structural sensitivity of an already-declared scenario model.

This is set arithmetic, not another fitted model.  It retains the original
outcome table, uses the hull of its recorded component parameter ranges, and
adds zero transfer as an explicit structural alternative.  No probability is
assigned to that alternative.  The resulting range is neither a confidence
interval nor a prediction interval and does not validate a failed model.

For outcome k, x_k - sum(q_j*x_j) uses the SAME state and probability vector.
With nonnegative simplex q, its extrema over bounded states and the source Q
polytope are two linear programs.  Shared equity/rates factors are applied to
aggregate portfolio exposures, not independently redrawn for each holding.
"""
from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping

from scipy.optimize import linprog
from scipy.stats import t as student_t

from .default_event_economics import _polytope


SCHEMA = "castle.scenario_sensitivity.r32.v1"
KIND = "DECLARED_STRUCTURAL_MODEL_SENSITIVITY"
_MACRO = {"FED_POLICY", "US_INFLATION"}
_SUPPORTED_STATUS = {"COMPLETE_DECLARED_MODEL", "COMPLETE_MEASURED_SURPRISE_RESPONSE",
                     "IDENTIFIED_NOMINAL_USD_CASH_ZERO"}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _interval(value, name, *, point=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " requires two finite endpoints")
    low, high = (_number(x, name) for x in value)
    if low > high or point is not None and not low <= point <= high:
        raise ValueError(name + " is unordered or excludes its point")
    return [low, high]


def _hull(*intervals):
    return [min(row[0] for row in intervals), max(row[1] for row in intervals)]


def _product(left, right):
    values = [x * y for x in left for y in right]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Sensitivity arithmetic overflowed")
    return [min(values), max(values)]


def _sum(intervals):
    result = [math.fsum(row[k] for row in intervals) for k in (0, 1)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Sensitivity arithmetic overflowed")
    return result


def _standard_error_range(point, se, n_train, name):
    point, se = _number(point, name + " point"), _number(se, name + " standard error")
    if se < 0:
        raise ValueError(name + " standard error cannot be negative")
    if n_train is not None:
        if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train <= 2:
            raise ValueError(name + " training count must exceed two")
        critical = float(student_t.ppf(.975, n_train - 2))
        basis = "RECORDED_SE_TIMES_STUDENT_T_975_WITH_RECORDED_TRAINING_DF"
    else:
        critical = 1.96
        basis = "RECORDED_SE_TIMES_1_96_TRAINING_DF_NOT_RETAINED"
    return _interval([point - critical * se, point + critical * se], name), basis


def _component_loading(position, prefix):
    point = position.get(prefix + "_beta")
    explicit = position.get(prefix + "_beta_interval")
    se = position.get(prefix + "_se")
    if point is None and explicit is None and se is None:
        return None
    point = _number(point, prefix + " loading")
    if explicit is not None:
        interval = _interval(explicit, prefix + " recorded parameter range", point=point)
        basis = "RECORDED_COMPONENT_PARAMETER_INTERVAL"
    else:
        interval, basis = _standard_error_range(point, se, position.get(prefix + "_n_train"), prefix)
    return {"component": prefix, "point": point, "interval": interval, "basis": basis}


def _equity_factor(table):
    factor = (table.get("factors") or {}).get("equity") or {}
    point = _number(factor.get("point"), "Equity response point")
    expected_unit = ("fractional_SPY_return_per_policy_bp" if table["family"] == "FED_POLICY"
                     else "fractional_SPY_return_per_headline_cpi_pp")
    if factor.get("unit") != expected_unit:
        raise ValueError("Equity response unit is missing or differs from the declared event state")
    source = factor.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Recorded equity-response source metadata are required")
    if factor.get("coefficient_interval") is not None:
        interval = _interval(factor["coefficient_interval"], "Recorded equity response interval", point=point)
        basis = "RECORDED_EQUITY_RESPONSE_PARAMETER_INTERVAL"
    elif table["family"] == "FED_POLICY":
        if source.get("published_unit") != "percentage points of one-day broad-equity return per one percentage point policy-rate surprise":
            raise ValueError("Published FOMC source coefficient unit is not the supported conversion")
        intervals = []
        for label in ("published", "robustness"):
            value = _number(source.get(label + "_point"), label + " FOMC coefficient")
            statistic = _number(source.get(label + "_absolute_t"), label + " FOMC absolute t")
            if statistic <= 0:
                raise ValueError("Published absolute t statistic must be positive")
            se = abs(value) / statistic
            intervals.append([(value - 1.96 * se) / 10_000, (value + 1.96 * se) / 10_000])
        interval = _interval(_hull(*intervals), "Published FOMC specification hull", point=point)
        basis = "HULL_OF_BOTH_RECORDED_PUBLISHED_APPROXIMATE_95PCT_SPECIFICATIONS"
    else:
        if source.get("published_unit") != "percentage points of aggregate nominal stock return per one percentage point headline month-on-month CPI surprise":
            raise ValueError("Published CPI source coefficient unit is not the supported conversion")
        value = _number(source.get("published_point"), "Published CPI coefficient")
        se = _number(source.get("published_robust_standard_error"), "Published CPI standard error")
        if se < 0:
            raise ValueError("Published CPI standard error cannot be negative")
        interval = _interval([(value - 1.96 * se) / 100, (value + 1.96 * se) / 100],
                             "Published CPI parameter interval", point=point)
        basis = "RECORDED_PUBLISHED_CPI_COEFFICIENT_PLUS_OR_MINUS_1_96_ROBUST_SE"
    return {"interval": interval, "with_no_transfer_alternative": _hull(interval, [0., 0.]),
            "basis": basis, "unit": expected_unit, "source": deepcopy(dict(source))}


def _rate_factor(table):
    factor = (table.get("factors") or {}).get("rates") or {}
    expected_unit = ("long_yield_bp_per_policy_bp" if table["family"] == "FED_POLICY"
                     else "long_yield_bp_per_headline_cpi_pp")
    if factor.get("unit") != expected_unit:
        raise ValueError("Rate response unit is missing or differs from the declared event state")
    point = _number(factor.get("point"), "Rate response point")
    source = factor.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Recorded rate-response source metadata are required")
    draw = factor.get("draw")
    if not isinstance(draw, (list, tuple)) or len(draw) != 3:
        raise ValueError("A recorded rate-response scale or bounded proxy interval is required")
    if table["family"] == "FED_POLICY" and draw[0] == "normal":
        mean, se = _number(draw[1], "Rate response mean"), _number(draw[2], "Rate response standard error")
        if se < 0 or not math.isclose(mean, point, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Recorded FOMC rate point and uncertainty disagree")
        interval = _interval([mean - 1.96 * se, mean + 1.96 * se], "Rate parameter interval")
        basis = "RECORDED_HISTORICAL_RATE_RESPONSE_PLUS_OR_MINUS_1_96_SE"
    elif table["family"] == "US_INFLATION" and draw[0] == "uniform":
        interval = _interval(draw[1:], "Declared headline CPI rate proxy domain", point=point)
        basis = "RECORDED_BOUNDED_CORE_TO_HEADLINE_CPI_PROXY_DOMAIN"
    else:
        raise ValueError("Unsupported rate-response uncertainty schema for this family")
    return {"interval": interval, "with_no_transfer_alternative": _hull(interval, [0., 0.]),
            "basis": basis, "unit": expected_unit, "source": deepcopy(dict(source))}


def _macro_response(table, positions):
    equity, expanded_equity, durations, records = [], [], [], []
    for position in positions:
        symbol, weight = position["symbol"], position["weight"]
        channel = position.get("channel")
        if not weight or channel == "NONE":
            if weight and position.get("asset_type") != "cash":
                raise ValueError("A nonzero NONE channel must be explicitly nominal USD cash")
            records.append({"symbol": symbol, "weight": weight, "channel": "NONE", "interval": [0., 0.]})
        elif channel == "EQUITY":
            components = [component for prefix in ("daily", "event_day")
                          if (component := _component_loading(position, prefix)) is not None]
            if not components:
                raise ValueError("No recorded component loading range for " + symbol)
            interval = _hull(*(c["interval"] for c in components))
            expanded = _hull(interval, [0., 0.])
            equity.append(_product([weight, weight], interval))
            expanded_equity.append(_product([weight, weight], expanded))
            records.append({"symbol": symbol, "weight": weight, "channel": channel,
                            "interval": interval, "with_no_transfer_alternative": expanded,
                            "components": components, "basis": "HULL_WITHOUT_HOLDOUT_SELECTION_OR_PRECISION_POOLING"})
        elif channel == "RATES":
            duration = position.get("duration") or {}
            point = _number(duration.get("duration"), "Source-derived duration for " + symbol)
            interval = _interval(duration.get("duration_range"), "Duration domain for " + symbol, point=point)
            if interval[0] < 0:
                raise ValueError("This duration channel requires a nonnegative declared duration domain")
            durations.append(_product([weight, weight], interval))
            records.append({"symbol": symbol, "weight": weight, "channel": channel,
                            "interval": interval, "basis": duration.get("basis"),
                            "assumption": duration.get("assumption"),
                            "covered_fund_nav_weight": duration.get("covered_fund_nav_weight")})
        else:
            raise ValueError("An unresolved holding channel cannot produce a full-portfolio sensitivity range")
    base, expanded, factors = [], [], {}
    if equity:
        gamma = _equity_factor(table)
        aggregate, aggregate_expanded = _sum(equity), _sum(expanded_equity)
        base.append(_product(aggregate, gamma["interval"]))
        expanded.append(_product(aggregate_expanded, gamma["with_no_transfer_alternative"]))
        factors["equity"] = {**gamma, "aggregate_loading_interval": aggregate,
                              "aggregate_loading_with_no_transfer_alternative": aggregate_expanded}
    if durations:
        kappa = _rate_factor(table)
        aggregate = _sum(durations)
        rate_exposure = _product(aggregate, [-1 / 10_000, -1 / 10_000])
        base.append(_product(rate_exposure, kappa["interval"]))
        expanded.append(_product(rate_exposure, kappa["with_no_transfer_alternative"]))
        factors["rates"] = {**kappa, "aggregate_duration_interval": aggregate}
    return _sum(base), _sum(expanded), {"positions": records, "shared_factors": factors}


def _payroll_response(table, positions):
    if table["status"] == "IDENTIFIED_NOMINAL_USD_CASH_ZERO":
        if any(p["weight"] and p.get("status") != "KNOWN_NOMINAL_USD_CASH_ZERO" for p in positions):
            raise ValueError("Identified payroll cash zero requires every active holding to be declared cash")
        return [0., 0.], [0., 0.], {"basis": "IDENTIFIED_NOMINAL_USD_CASH_ZERO"}
    fit = (table.get("factors") or {}).get("portfolio_response") or {}
    slope = _number(fit.get("slope"), "Portfolio payroll response")
    if fit.get("slope_interval_95") is not None:
        interval = _interval(fit["slope_interval_95"], "Recorded payroll response parameter interval", point=slope)
        basis = "RECORDED_PORTFOLIO_SLOPE_PARAMETER_INTERVAL"
    else:
        interval, basis = _standard_error_range(slope, fit.get("se"), fit.get("n_train"), "Payroll portfolio response")
    response = _product(interval, [1 / 100_000, 1 / 100_000])
    return response, _hull(response, [0., 0.]), {
        "basis": basis, "slope_interval_per_100k_jobs": interval,
        "portfolio_covariance_preserved": True,
        "training_span": deepcopy(fit.get("training_span")),
        "holdout_span": deepcopy(fit.get("holdout_span")),
        "original_transfer_check": deepcopy(fit.get("transfer_check")),
    }


def _positions(table):
    raw = table.get("positions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Original complete portfolio positions are required")
    positions, seen = [], set()
    for row in raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("symbol"), str) or not row["symbol"]:
            raise ValueError("Each position requires a nonempty symbol")
        if row["symbol"] in seen:
            raise ValueError("Duplicate original position symbol")
        seen.add(row["symbol"])
        weight = _number(row.get("weight"), "Position weight")
        if weight < 0:
            raise ValueError("Only the project's long-only portfolio scope is admitted")
        if weight:
            _respect_scope_guards(row)
            if row.get("asset_type") not in (None, "stock", "etf", "cash"):
                raise ValueError("Declared instrument type is outside the supported portfolio scope")
            classification = str(row.get("classification_status") or "").upper()
            if any(value in classification for value in ("UNSUPPORTED", "LEVERAGED", "INVERSE")):
                raise ValueError("An explicitly unsupported instrument classification cannot be admitted")
            instrument_class = str(row.get("instrument_class") or "").upper()
            if any(value in instrument_class for value in ("UNSUPPORTED", "LEVERAGED", "INVERSE")):
                raise ValueError("An explicitly unsupported underlying instrument class cannot be admitted")
        positions.append({**row, "weight": weight})
    if abs(math.fsum(row["weight"] for row in positions) - 1) > 1e-8:
        raise ValueError("Original position weights must already sum to one; no renormalization")
    return positions


def _respect_scope_guards(record):
    if record.get("legacy_fallback_allowed") is False:
        raise ValueError("A hard model-scope restriction cannot be bypassed by a sensitivity envelope")
    for scope in (record, record.get("instrument_identity_gate") or {}):
        if not isinstance(scope, Mapping):
            raise ValueError("Instrument-scope metadata are malformed")
        if any(scope.get(key) for key in ("blocked_symbols", "identity_blocked_symbols", "asset_class_blocked_symbols")):
            raise ValueError("Instrument identity or fund-class restrictions remain in force")
        if str(scope.get("status") or "").upper().startswith(("BLOCKED", "UNSUPPORTED")):
            raise ValueError("An explicit blocked instrument scope cannot be bypassed")


def _state_rows(table, atoms, identifiers, reference):
    raw = table.get("outcomes")
    if not isinstance(raw, list) or len(raw) != len(atoms):
        raise ValueError("A full modeled row is required for every source atom")
    by_id = {}
    for row in raw:
        if not isinstance(row, Mapping) or row.get("outcome_id") in by_id:
            raise ValueError("Modeled outcome identities must be unique")
        by_id[row.get("outcome_id")] = row
    if set(by_id) != set(identifiers):
        raise ValueError("Modeled outcomes and source probability atoms differ")
    rows = []
    for atom, identifier, q in zip(atoms, identifiers, reference):
        row = by_id[identifier]
        point = _number(row.get("state"), "Recorded outcome state")
        interval = _interval(row.get("state_interval"), "Recorded bounded scenario state", point=point)
        if row.get("q_reference") is not None and not math.isclose(
                _number(row["q_reference"], "Recorded outcome Q"), q, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Modeled outcome and source reference probabilities differ")
        source_range = atom.get("value_interval")
        if isinstance(source_range, (list, tuple)) and len(source_range) == 2 and all(x is not None for x in source_range):
            source_range = _interval(source_range, "Source state interval")
            if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(interval, source_range)):
                raise ValueError("Recorded bounded state and fully specified source state interval differ")
        rows.append({"outcome_id": identifier, "label": str(row.get("label", identifier)),
                     "state": point, "state_interval": interval,
                     "state_basis": str(row.get("state_basis") or "RECORDED_FINITE_SCENARIO_DOMAIN"),
                     "source_contract_ids": deepcopy(row.get("source_contract_ids") or atom.get("source_contract_ids") or [])})
    return rows


def _surprise_extrema(states, k, kwargs):
    lows = [r["state_interval"][0] for r in states]
    highs = [r["state_interval"][1] for r in states]
    low_coefficients = [-value for value in highs]
    low_coefficients[k] = -lows[k]
    high_coefficients = [-value for value in lows]
    high_coefficients[k] = -highs[k]
    lower = linprog(low_coefficients, **kwargs)
    upper = linprog([-value for value in high_coefficients], **kwargs)
    if not lower.success or not upper.success:
        raise ValueError("The source Q polytope does not admit sensitivity extrema")
    for optimum in (lower, upper):
        _checked_q_witness(optimum.x, kwargs)
    interval = _interval([lows[k] + float(lower.fun), highs[k] - float(upper.fun)], "Surprise envelope")
    lower_states, upper_states = highs[:], lows[:]
    lower_states[k], upper_states[k] = lows[k], highs[k]
    return interval, {
        "minimum": {"q": [float(v) for v in lower.x], "joint_states": lower_states},
        "maximum": {"q": [float(v) for v in upper.x], "joint_states": upper_states},
    }


def _checked_q_witness(witness, kwargs):
    values = [_number(float(value), "Q extremum witness") for value in witness]
    if len(values) != len(kwargs["bounds"]):
        raise ValueError("Q extremum witness has the wrong dimension")
    if any(not low - 1e-8 <= value <= high + 1e-8 for value, (low, high) in zip(values, kwargs["bounds"])):
        raise ValueError("Q extremum witness violates a probability bound")
    for row, rhs in zip(kwargs.get("A_eq", []), kwargs.get("b_eq", [])):
        if abs(math.fsum(a * value for a, value in zip(row, values)) - rhs) > 1e-8:
            raise ValueError("Q extremum witness violates an equality")
    for row, rhs in zip(kwargs.get("A_ub", []), kwargs.get("b_ub", [])):
        if math.fsum(a * value for a, value in zip(row, values)) - rhs > 1e-8:
            raise ValueError("Q extremum witness violates a quote constraint")


def build_sensitivity_envelope(table, distribution):
    """Return a deterministic assumption-set envelope, or an explicit reason.

    The caller's table and distribution are never modified.  This function
    neither fits coefficients nor changes diagnostics/admission.  All finite
    state domains must already be present in the declared outcome table.
    """
    base = {"schema": SCHEMA, "kind": KIND, "status": "UNAVAILABLE", "outcomes": [],
            "confidence_level": None, "prediction_interval": False,
            "independent_validation": False, "no_response_alternative_included": True,
            "original_model_unchanged": True, "original_diagnostics_unchanged": True,
            "method": "EXACT_LINEAR_PROGRAMS_FOR_JOINT_STATE_Q_SURPRISE_AND_SHARED_FACTOR_INTERVAL_ARITHMETIC"}
    try:
        if not isinstance(table, Mapping) or not isinstance(distribution, Mapping):
            raise ValueError("A recorded outcome table and source distribution are required")
        family = table.get("family")
        base["family"] = family
        if family not in _MACRO | {"US_PAYROLLS"} or distribution.get("family") != family:
            raise ValueError("Only matching FOMC, headline CPI or payroll scenario families are supported")
        if distribution.get("status") != "CONDITIONAL_SOURCE_DISTRIBUTION":
            raise ValueError("The source distribution must have been admitted before sensitivity is calculated")
        expected_state_unit = {"FED_POLICY": "policy_basis_points",
                               "US_INFLATION": "headline_monthly_cpi_percentage_points",
                               "US_PAYROLLS": "jobs, first-release change in total nonfarm payrolls"}[family]
        if table.get("state_unit") != expected_state_unit:
            raise ValueError("Recorded state unit differs from the supported scenario family")
        if table.get("status") not in _SUPPORTED_STATUS:
            raise ValueError("A complete declared model is required; partial or blocked models are not filled")
        if table.get("missing_symbols") or _number(table.get("missing_weight", 0), "Missing weight") != 0:
            raise ValueError("Missing holdings prohibit a complete-portfolio sensitivity envelope")
        _respect_scope_guards(table)
        notional = _number(table.get("notional_usd"), "Portfolio notional")
        if notional <= 0:
            raise ValueError("Portfolio notional must be positive")
        atoms = distribution.get("atoms")
        if not isinstance(atoms, list):
            raise ValueError("Source probability atoms are required")
        kwargs, reference, identifiers = _polytope(distribution, atoms)
        positions = _positions(table)
        states = _state_rows(table, atoms, identifiers, reference)
        response, expanded, parameters = (_macro_response(table, positions) if family in _MACRO
                                           else _payroll_response(table, positions))
        response, expanded = _interval(response, "Portfolio response"), _interval(expanded, "Expanded portfolio response")
        rows = []
        for k, row in enumerate(states):
            surprise, witnesses = _surprise_extrema(states, k, kwargs)
            original = _product(_product(response, surprise), [notional, notional])
            structural = _product(_product(expanded, surprise), [notional, notional])
            rows.append({**row, "surprise_interval": surprise,
                         "pnl_component_parameter_set_range_usd": original,
                         "pnl_model_sensitivity_range_usd": structural,
                         "direction_robust_to_declared_set": bool(structural[0] > 0 or structural[1] < 0),
                         "zero_effect_in_declared_set": structural[0] <= 0 <= structural[1],
                         "surprise_extrema_witnesses": witnesses})
        base.update(status="AVAILABLE", notional_usd=notional, state_unit=table.get("state_unit"),
                    response_interval_fraction_per_state_unit=expanded,
                    component_response_interval_fraction_per_state_unit=response,
                    outcomes=rows, parameter_sets=parameters,
                    bounds_scope="CONSERVATIVE_OUTER_ENVELOPE_OF_THE_EXPLICIT_CARTESIAN_ASSUMPTION_SET",
                    surprise_extrema_scope="SHARP_PER_OUTCOME_FOR_RECORDED_BOUNDED_STATES_AND_SOURCE_Q_POLYTOPE",
                    numerical_feasibility_tolerance=1e-8,
                    joint_outcome_extrema_simultaneously_attainable=False,
                    assumptions=[
                        "Each state stays in the finite scenario domain already recorded in the outcome table; open contract tails are not thereby bounded in reality.",
                        "The same feasible Q and joint state vector define both the realized state and the priced-in expectation.",
                        "Recorded component parameter intervals are combined by their hull, never selected for a favorable held-out result.",
                        "Zero transfer is added as a declared structural alternative for empirical/historical responses; no probability or statistical confidence is assigned to it.",
                        "All holdings share the same equity factor and rate factor; factors are applied after aggregating the original weighted exposures.",
                        "Cross-component restrictions beyond the recorded source-Q constraints are unknown; the Cartesian parameter set is deliberately conservative.",
                    ],
                    limitations=[
                        "This is not a confidence interval, a simultaneous confidence region, a prediction interval, a causal bound, or a claim that the true response lies inside the set.",
                        "Component parameter endpoints use their recorded scales; 1.96 times SE is explicitly a diagnostic scale when training degrees of freedom were not retained.",
                        "No new historical fit, holdout, validation pass, forecast-skill claim or probability calibration is produced.",
                        "Original failed and untested diagnostics remain unchanged. Their failure is not repaired by widening a range.",
                        "Different outcomes' extrema may require different joint states, Q or parameters and cannot generally occur simultaneously.",
                        "Other news, nonlinear responses, changes beyond this declared set and future total-return noise remain outside this calculation.",
                    ])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        base.update(status="UNAVAILABLE", reason=str(exc), outcomes=[])
    return base
