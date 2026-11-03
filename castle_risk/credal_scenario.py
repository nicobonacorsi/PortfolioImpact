"""Coherent probability bounds for a finite structural event partition.

The model supplies a point forecast and epistemic intervals for scenario masses
and event-conditional probabilities. Bounds are *assumption-conditional*, not
frequentist confidence intervals or guarantees of empirical calibration.

With q_i = P(S_i), r_i = P(E | S_i), the credal model is

    sum_i q_i = 1, a_i <= q_i <= b_i, l_i <= r_i <= u_i.

Because q_i >= 0, the extrema of sum_i q_i r_i occur at r_i = l_i
or r_i = u_i. The remaining optimization is a bounded-simplex linear program.
The greedy mass-allocation solver below is exact for this rectangular model.
Cross-scenario dependence restrictions require a different joint model and
must never be silently assumed away.
"""
from __future__ import annotations

from math import fsum, isfinite

SCHEMA = "castle.structural_partition_credal.v1"
MAX_SCENARIOS = 7
_TOL = 1e-9


class CredalInputError(ValueError):
    """The proposed partition or its probability bounds are not admissible."""


def _probability(value: object, label: str) -> float:
    if type(value) not in (float, int) or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise CredalInputError(f"{label} must be a finite probability in [0, 1].")
    return float(value)


def _triple(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, dict) or set(value) != {"low", "center", "high"}:
        raise CredalInputError(f"{label} must have low, center and high.")
    low, center, high = (_probability(value[k], f"{label}.{k}")
                         for k in ("low", "center", "high"))
    if not low <= center <= high:
        raise CredalInputError(f"{label} bounds must contain its central value.")
    return low, center, high


def _weighted_extremum(lower: list[float], upper: list[float],
                       coefficients: list[float], *, maximize: bool) -> float:
    """Exact bounded-simplex LP optimum by allocating residual mass in cost order.

    Starts at q_i = lower_i; then moves the remaining probability mass to
    available capacities in ascending/descending objective-coefficient order.
    The exchange argument shows any inverted allocation can be improved by
    transferring mass to the cheaper/more expensive branch.
    """
    mass = list(lower)
    remaining = 1.0 - fsum(mass)
    if remaining < -_TOL or fsum(upper) < 1.0 - _TOL:
        raise CredalInputError("Scenario probability intervals have an empty simplex intersection.")
    remaining = max(0.0, remaining)
    order = sorted(range(len(mass)), key=lambda i: ((-coefficients[i] if maximize else coefficients[i]), i))
    for i in order:
        increment = min(remaining, upper[i] - mass[i])
        if increment > 0.0:
            mass[i] += increment
            remaining -= increment
        if remaining <= _TOL:
            break
    if remaining > _TOL:
        raise CredalInputError("Scenario probability intervals cannot sum to one.")
    return min(1.0, max(0.0, fsum(q * c for q, c in zip(mass, coefficients))))


def evaluate_structural_partition(scenarios: object) -> dict:
    """Return the model point forecast and tight coherent credal bounds.

    Scenarios are *asserted* to be a mutually exclusive and exhaustive partition.
    This routine proves numerical coherence conditional on that structural
    assertion; it cannot certify the natural-language predicates themselves.
    """
    if not isinstance(scenarios, list) or not 2 <= len(scenarios) <= MAX_SCENARIOS:
        raise CredalInputError("Supply between two and seven ordered scenarios, including a residual branch.")
    identifiers = set()
    low_mass, central_mass, high_mass = [], [], []
    low_cond, central_cond, high_cond = [], [], []
    residual_count = 0
    for i, item in enumerate(scenarios):
        if not isinstance(item, dict):
            raise CredalInputError("Each scenario must be an object.")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise CredalInputError("Scenario IDs must be distinct, nonempty strings.")
        identifiers.add(identifier)
        if item.get("residual") is True:
            residual_count += 1
            if i != len(scenarios) - 1:
                raise CredalInputError("The residual scenario must be last.")
        elif item.get("residual") is not False:
            raise CredalInputError("Each scenario must declare its residual flag.")
        a, q, b = _triple(item.get("scenario_probability"), f"{identifier}.scenario_probability")
        l, r, u = _triple(item.get("event_conditional_probability"), f"{identifier}.event_conditional_probability")
        low_mass.append(a)
        central_mass.append(q)
        high_mass.append(b)
        low_cond.append(l)
        central_cond.append(r)
        high_cond.append(u)
    if residual_count != 1:
        raise CredalInputError("Exactly one final residual branch is required for a complete ordered partition.")
    if abs(fsum(central_mass) - 1.0) > _TOL:
        raise CredalInputError("Central scenario probabilities must sum to one; no implicit renormalization.")
    if fsum(low_mass) > 1.0 + _TOL or fsum(high_mass) < 1.0 - _TOL:
        raise CredalInputError("The scenario interval constraints are infeasible.")
    center = fsum(q * r for q, r in zip(central_mass, central_cond))
    low = _weighted_extremum(low_mass, high_mass, low_cond, maximize=False)
    high = _weighted_extremum(low_mass, high_mass, high_cond, maximize=True)
    if low > center + _TOL or center > high + _TOL:
        raise CredalInputError("The proposed central forecast is outside the coherent credal set.")
    low = min(low, center)
    high = max(high, center)
    return {
        "schema": SCHEMA,
        "probability_center": center,
        "credal_interval": [low, high],
        "interval_width": high - low,
        "scenario_count": len(scenarios),
        "coherence_status": "NONEMPTY_BOUNDED_SIMPLEX_AND_CONDITIONAL_CONSTRAINTS",
        "partition_semantics": "ORDERED_FIRST_MATCH_WITH_FINAL_RESIDUAL_ASSERTED_NOT_LOGICALLY_VERIFIED",
        "uncertainty_semantics": "RECTANGULAR_EPISTEMIC_MODEL_JUDGMENT_NOT_FREQUENTIST_COVERAGE",
        "physical_probability_empirically_identified": False,
        "forecast_calibrated": False,
    }


def evaluate_structural_partition_lp(scenarios: object) -> dict:
    """Independent cross-check of the greedy extrema with a generic LP solver.

    Atomic representation: x_i = P(E and S_i), y_i = P(not E and S_i),
    sum_i (x_i + y_i) = 1, a_i <= x_i + y_i <= b_i, l_i (x_i+y_i) <= x_i <= u_i (x_i+y_i).
    Minimizing and maximizing sum_i x_i are two linear programs solved with
    scipy's HiGHS backend. This function shares no code path with the greedy
    allocation and exists so an acceptance script can prove both agree.
    """
    from scipy.optimize import linprog
    import numpy as np
    verified = evaluate_structural_partition(scenarios)
    n = verified["scenario_count"]
    a = [float(item["scenario_probability"]["low"]) for item in scenarios]
    b = [float(item["scenario_probability"]["high"]) for item in scenarios]
    lo = [float(item["event_conditional_probability"]["low"]) for item in scenarios]
    hi = [float(item["event_conditional_probability"]["high"]) for item in scenarios]
    # Variables: x_0..x_{n-1}, y_0..y_{n-1}
    A_ub, b_ub = [], []
    for i in range(n):
        mass = np.zeros(2 * n)
        mass[i] = mass[n + i] = 1.0
        A_ub.append(mass.copy()); b_ub.append(b[i])          # x+y <= b
        A_ub.append(-mass.copy()); b_ub.append(-a[i])        # x+y >= a
        row = np.zeros(2 * n); row[i] = lo[i] - 1.0; row[n + i] = lo[i]
        A_ub.append(row); b_ub.append(0.0)                   # l(x+y) - x <= 0
        row = np.zeros(2 * n); row[i] = 1.0 - hi[i]; row[n + i] = -hi[i]
        A_ub.append(row); b_ub.append(0.0)                   # x - u(x+y) <= 0
    A_eq = [np.ones(2 * n)]
    b_eq = [1.0]
    c = np.concatenate([np.ones(n), np.zeros(n)])
    results = []
    for sign in (1.0, -1.0):
        solution = linprog(sign * c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq),
                           b_eq=np.array(b_eq), bounds=[(0.0, 1.0)] * (2 * n), method="highs")
        if solution.status != 0:
            raise CredalInputError("Independent LP did not solve: " + str(solution.message))
        results.append(float(sign * solution.fun))
    low, high = min(results), max(results)
    low = min(max(0.0, low), verified["probability_center"])
    high = max(min(1.0, high), verified["probability_center"])
    greedy_low, greedy_high = verified["credal_interval"]
    return {
        "schema": SCHEMA + ".independent_lp_crosscheck",
        "lp_credal_interval": [low, high],
        "greedy_credal_interval": [greedy_low, greedy_high],
        "max_abs_difference": max(abs(low - greedy_low), abs(high - greedy_high)),
        "agrees_within_1e_9": max(abs(low - greedy_low), abs(high - greedy_high)) <= 1e-9,
        "probability_center": verified["probability_center"],
        "solver": "scipy.optimize.linprog(method=highs)",
    }
