"""Sharp, explicitly assumed event-conditioned portfolio stress envelopes.

This engine deliberately separates a *user-specified what-if* from a forecast.
An original event need not have a prediction-market counterpart to be used as
a scenario condition. Every original portfolio leg needs a declared response.
No event probability, observed causal stock response or priced-in adjustment
is inferred. All factors are percent asset-return shocks over one declared
horizon and every numerical assumption is visible in the output.

The robust bounds solve two linear programs over a compact shock polytope.
They are sharp for the stated affine response model, *not* confidence bounds
for future asset prices or real-world conditional P&L.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math

from scipy.optimize import linprog

from .portfolio import parse_portfolio, _symbol


class EventStressError(ValueError):
    """Incomplete or inconsistent evidence for the declared stress calculation."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EventStressError(name + " must be a finite number")
    return float(value)


def _label(value, name, *, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise EventStressError(name + " must be nonempty bounded text")
    return value.strip()


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def _factors(records):
    if not isinstance(records, list) or not 1 <= len(records) <= 16:
        raise EventStressError("Between 1 and 16 explicit stress factors required")
    result = []
    names = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise EventStressError("Invalid stress factor")
        name = _label(row.get("name"), "Factor name", limit=80)
        if name in names:
            raise EventStressError("Repeated factor name")
        names.add(name)
        bounds = row.get("return_range_pct")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise EventStressError("Each factor needs explicit lower/upper percent returns")
        low = _finite(bounds[0], "Lower factor shock")
        high = _finite(bounds[1], "Upper factor shock")
        if not -100 <= low <= high:
            raise EventStressError("Factor returns must be ordered and at least -100%")
        basis = _label(row.get("assumption_basis"), "Factor assumption basis")
        if basis == "MARKET_PROBABILITY" or basis == "PROVEN_CAUSAL_EVENT_RESPONSE":
            raise EventStressError("A quote or unsupported causal label is not a return assumption")
        result.append({"name": name, "return_range_pct": [low, high],
                       "assumption_basis": basis,
                       "economic_channel": _label(row.get("economic_channel"),
                                                  "Factor economic channel")})
    return result


def _response_positions(rows, portfolio, factors):
    if not isinstance(rows, Mapping):
        raise EventStressError("A complete original-position response mapping is required")
    names = [r["name"] for r in factors]
    expected = {p.symbol for p in portfolio.positions if p.weight > 0}
    asset_types = {p.symbol: p.asset_type for p in portfolio.positions}
    given = {}
    for raw, entry in rows.items():
        symbol = _symbol(raw)
        if symbol in given or symbol not in expected or not isinstance(entry, Mapping):
            raise EventStressError("Response must refer to each original owned position exactly once")
        given[symbol] = entry
    if set(given) != expected:
        raise EventStressError("An absent original position response is unknown, not zero")
    out = {}
    for symbol, entry in given.items():
        source = _label(entry.get("response_basis"), "Position response basis")
        if source == "INFERRED_FROM_PREDICTION_MARKET_PRICE":
            raise EventStressError("Prediction-market quote does not infer stock return")
        if "fixed_return_pct" not in entry:
            raise EventStressError("Position intercept is unknown until explicitly supplied")
        intercept = _finite(entry["fixed_return_pct"], "Explicit position intercept")
        if not isinstance(entry.get("factor_loadings"), Mapping):
            raise EventStressError("Position factor loadings required; empty is explicit zero")
        if set(entry["factor_loadings"]) - set(names):
            raise EventStressError("Response refers to an undeclared factor")
        loads = {name: _finite(entry["factor_loadings"].get(name, 0.0),
                               "Position factor loading") for name in names}
        if asset_types[symbol] == "cash" and (abs(intercept) > 1e-12 or any(abs(c) > 1e-12 for c in loads.values())):
            raise EventStressError("Nominal USD cash cannot have fabricated asset shocks")
        out[symbol] = {"fixed_return_pct": intercept, "factor_loadings": loads,
                       "response_basis": source}
    return out


def _constraints(rows, names):
    if rows is None:
        rows = []
    if not isinstance(rows, (list, tuple)) or len(rows) > 48:
        raise EventStressError("A bounded list of additional linear constraints is required")
    matrix, rhs, equal_matrix, equal_rhs, documented = [], [], [], [], []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("coefficients"), Mapping):
            raise EventStressError("Malformed factor relationship")
        if row.get("sense") not in ("<=", ">=", "="):
            raise EventStressError("Only closed linear inequalities/equalities supported")
        if set(row["coefficients"]) - set(names):
            raise EventStressError("Constraint refers to an undeclared factor")
        vector = [_finite(row["coefficients"].get(n, 0.0), "Constraint coefficient") for n in names]
        target = _finite(row.get("rhs_pct"), "Constraint right-hand side")
        reason = _label(row.get("assumption_basis"), "Constraint assumption basis")
        sense = row["sense"]
        if sense == ">=":
            vector, target = [-v for v in vector], -target
        if sense == "=":
            equal_matrix.append(vector)
            equal_rhs.append(target)
        else:
            matrix.append(vector)
            rhs.append(target)
        documented.append({"coefficients": dict(row["coefficients"]),
                           "sense": sense, "rhs_pct": _finite(row["rhs_pct"], "Constraint rhs"),
                           "assumption_basis": reason})
    return matrix, rhs, equal_matrix, equal_rhs, documented


def evaluate_original_event_robust_stress(
    *, portfolio_rows, notional, original_event_id, original_event_question,
    common_return_horizon, factors, position_responses, linear_constraints=None,
):
    """Compute exact dollar extrema over explicitly assumed shock states.

    This calculation is about the *original event definition*, independent of
    whether a related prediction-market contract settles a different question.
    Results are what-if envelopes, never empirical event-effect estimates.
    """
    identifier = _label(original_event_id, "Original event identifier", limit=160)
    original_question = _label(original_event_question, "Original event question")
    horizon = _label(common_return_horizon, "Common return horizon", limit=400)
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    fs = _factors(factors)
    names = [r["name"] for r in fs]
    responses = _response_positions(position_responses, portfolio, fs)
    matrix, rhs, equal_matrix, equal_rhs, constraints = _constraints(linear_constraints, names)
    constant = math.fsum(p.weight * responses[p.symbol]["fixed_return_pct"]
                         for p in portfolio.positions if p.weight > 0)
    slopes = [math.fsum(p.weight * responses[p.symbol]["factor_loadings"][name]
                        for p in portfolio.positions if p.weight > 0) for name in names]
    bounds = [tuple(r["return_range_pct"]) for r in fs]
    common = dict(bounds=bounds, A_ub=matrix or None, b_ub=rhs or None,
                  A_eq=equal_matrix or None, b_eq=equal_rhs or None,
                  method="highs")
    min_result = linprog(slopes, **common)
    max_result = linprog([-c for c in slopes], **common)
    if not min_result.success or not max_result.success:
        raise EventStressError("No jointly admissible finite factor scenarios: "
                               + str(min_result.message if not min_result.success else max_result.message))
    # A factor may be a valid percentage but a leveraged combination of
    # loadings could imply physically impossible losses on an unlevered share.
    for position in portfolio.positions:
        if position.weight <= 0:
            continue
        response = responses[position.symbol]
        leg_slopes = [response["factor_loadings"][n] for n in names]
        leg_min = linprog(leg_slopes, **common)
        if not leg_min.success or response["fixed_return_pct"] + leg_min.fun < -100 - 1e-7:
            raise EventStressError("A declared original-position return can fall below -100%")

    def extremum(solution):
        shock = {name: float(z) for name, z in zip(names, solution.x)}
        legs = []
        for position in portfolio.positions:
            if position.weight == 0:
                continue
            response = responses[position.symbol]
            return_pct = response["fixed_return_pct"] + math.fsum(
                response["factor_loadings"][name] * shock[name] for name in names)
            legs.append({"symbol": position.symbol, "original_weight": position.weight,
                         "assumed_return_pct": return_pct,
                         "assumed_pnl_usd": portfolio.notional * position.weight * return_pct / 100})
        total = math.fsum(leg["assumed_pnl_usd"] for leg in legs)
        formula = portfolio.notional * (constant + math.fsum(
            coeff * shock[name] for coeff, name in zip(slopes, names))) / 100
        if not math.isclose(total, formula, abs_tol=1e-7, rel_tol=0):
            raise EventStressError("Portfolio valuation reconciliation failed")
        return {"pnl_usd": total, "portfolio_return_pct": 100 * total / portfolio.notional,
                "factor_shocks_pct": shock, "original_position_attribution": legs}

    downside, upside = extremum(min_result), extremum(max_result)
    signature_data = {"original_event_id": identifier,
                      "original_event_question": original_question,
                      "common_return_horizon": horizon,
                      "portfolio": portfolio.to_dict(),
                      "factors": fs, "position_responses": responses,
                      "linear_constraints": constraints}
    return {
        "schema": "castle.original_event_robust_stress.v1",
        "status": "COMPLETE_EXPLICIT_EVENT_DEFINED_STRESS_NOT_A_FORECAST",
        "original_event_id": identifier,
        "original_event_question": original_question,
        "scenario_condition": "IF_ORIGINAL_EVENT_AND_ALL_DECLARED_FACTOR_RESPONSE_ASSUMPTIONS",
        "common_return_horizon": horizon,
        "original_positions_only_no_etf_double_count": True,
        "original_portfolio_notional_usd": portfolio.notional,
        "full_portfolio_arithmetic_under_assumptions": True,
        "factors": fs,
        "position_responses": responses,
        "linear_constraints": constraints,
        "weighted_factor_dollar_sensitivities_per_1pp": {
            name: portfolio.notional * coeff / 100 for name, coeff in zip(names, slopes)},
        "sharp_assumption_conditional_pnl_range_usd": [downside["pnl_usd"], upside["pnl_usd"]],
        "downside_stress": downside,
        "upside_stress": upside,
        "scenario_assumptions_sha256": sha256(_canonical(signature_data)).hexdigest(),
        "mathematical_method": "Two linear programs; affine P&L over a compact bounded factor-shock polytope",
        "original_event_probability_P": None,
        "prediction_market_pricing_Q": None,
        "already_priced_in_reference": None,
        "identified_real_world_event_conditional_pnl_usd": None,
        "causally_validated_or_out_of_sample": False,
        "economic_ranking_allowed": False,
        "not_a_future_return_prediction": True,
        "limitations": [
            "Factor shock ranges and position response loadings are supplied assumptions, not event-estimated effects.",
            "Sharpness refers only to the declared affine model and joint shock constraints.",
            "With no independently measured event response, P&L conditional on the event remains unidentified.",
            "The original prediction-market contract and physical event probability may be unavailable.",
            "The current asset prices' already-anticipated event component is not inferred.",
            "ETF underlying constituents may inform a loading, but the owned ETF is valued exactly once.",
        ],
    }
