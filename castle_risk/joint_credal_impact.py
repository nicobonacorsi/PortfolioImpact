"""Exact, assumption-conditional bounds for joint portfolio event stresses.

This module never estimates event effects from market quotes or prices. An agent
may propose three *joint* holding-return vectors per branch, but their provenance
remains uncalibrated model judgment. The LP optimizes over a bounded simplex
for P(branch | E), not P(E), and over the convex hull of the three supplied
joint vectors within each branch. No independent per-holding extrema are
combined into economically impossible simultaneous moves.

The central event-vs-no-event difference is NOT absolute future P&L from today's
NAV: the no-event counterfactual and the asset-specific priced-in shares are
unobserved. ETF holdings count once at their original fund weight.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from math import fsum

SCHEMA = "castle.joint_credal_portfolio_impact.v1"
MAX_POSITIONS = 24
MAX_BRANCHES = 4
MAX_ABS_GAP_PCT = 100.0  # Operational proposal guard, not a theoretical return bound.
_TOL = 1e-8
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,11}\Z")


class JointImpactInputError(ValueError):
    """One fail-closed mathematical or original-position accounting check failed."""


def _finite(value, name, *, minimum=None, maximum=None):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise JointImpactInputError(f"{name} must be finite numeric (not boolean).")
    number = float(value)
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        raise JointImpactInputError(f"{name} is outside the declared protocol.")
    return number


def _original_positions(positions, notional):
    notional = _finite(notional, "notional", minimum=0.01, maximum=1_000_000_000)
    if not isinstance(positions, list) or not 1 <= len(positions) <= MAX_POSITIONS:
        raise JointImpactInputError("One to 24 original holdings are required.")
    unique, rows = set(), []
    for position in positions:
        if not isinstance(position, dict):
            raise JointImpactInputError("A holding must be an object.")
        symbol = position.get("symbol")
        kind = position.get("type", position.get("asset_type"))
        if (not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol) or
                symbol in unique or kind not in ("stock", "etf", "cash")):
            raise JointImpactInputError("Unique, resolved stock/ETF/cash identity is required.")
        unique.add(symbol)
        weight = _finite(position.get("weight"), f"{symbol}.weight", minimum=0.0, maximum=1.0)
        if kind == "cash" and symbol not in ("USD", "CASH"):
            raise JointImpactInputError("Nominal cash must use USD or CASH.")
        rows.append({"symbol": symbol, "type": kind, "weight": weight})
    if abs(fsum(row["weight"] for row in rows) - 1.0) > _TOL:
        raise JointImpactInputError("Original position weights must sum to one, without renormalization.")
    return sorted(rows, key=lambda row: row["symbol"]), notional


def portfolio_signature(positions, notional, horizon_sessions=20):
    """A public consistency fingerprint, not a cryptographic provenance signature."""
    rows, nav = _original_positions(positions, notional)
    if type(horizon_sessions) is not int or not 1 <= horizon_sessions <= 60:
        raise JointImpactInputError("Horizon must be between 1 and 60 trading sessions.")
    canonical = {"notional": round(nav, 8), "horizon_sessions": horizon_sessions,
                 "positions": [{**row, "weight": round(row["weight"], 12)} for row in rows]}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _weight_triple(value, name):
    if not isinstance(value, dict) or set(value) != {"low", "center", "high"}:
        raise JointImpactInputError(f"{name} needs low/center/high.")
    a, c, b = (_finite(value[k], f"{name}.{k}", minimum=0.0, maximum=1.0)
               for k in ("low", "center", "high"))
    if not a <= c <= b:
        raise JointImpactInputError(f"{name} has unordered weights.")
    return (a, c, b)


def _joint_vector(value, positions, name):
    if not isinstance(value, list) or len(value) != len(positions):
        raise JointImpactInputError(f"{name} must contain every original holding exactly once.")
    expected = {row["symbol"]: row for row in positions}
    observed = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"symbol", "gap_pct"}:
            raise JointImpactInputError(f"{name} has malformed holding entries.")
        sym = item.get("symbol")
        if sym not in expected or sym in observed:
            raise JointImpactInputError(f"{name} has an unknown or repeated holding.")
        gap = _finite(item.get("gap_pct"), f"{name}.{sym}",
                      minimum=-MAX_ABS_GAP_PCT, maximum=MAX_ABS_GAP_PCT)
        if expected[sym]["type"] == "cash" and gap != 0.0:
            raise JointImpactInputError("Nominal USD cash must have zero event gap.")
        observed[sym] = gap
    return {"portfolio_gap_pct": fsum(row["weight"] * observed[row["symbol"]] for row in positions),
            "holding_gap_pct": {row["symbol"]: observed[row["symbol"]] for row in positions}}


def _bounded_simplex_extreme(lower, upper, coefficients, *, maximize):
    """Exact signed linear-program optimum by greedy residual-mass allocation."""
    if (not lower or len(lower) != len(upper) or len(lower) != len(coefficients) or
            fsum(lower) > 1 + _TOL or fsum(upper) < 1 - _TOL):
        raise JointImpactInputError("No feasible conditional scenario mixture.")
    mass = list(lower)
    remainder = max(0.0, 1.0 - fsum(mass))
    order = sorted(range(len(mass)),
                   key=lambda i: ((-coefficients[i] if maximize else coefficients[i]), i))
    for i in order:
        grant = min(remainder, upper[i] - mass[i])
        if grant > 0:
            mass[i] += grant
            remainder -= grant
        if remainder <= _TOL:
            break
    if remainder > _TOL:
        raise JointImpactInputError("Probability simplex capacity exhausted.")
    return fsum(mass[i] * coefficients[i] for i in range(len(mass)))


def evaluate_joint_impact(positions, notional, branches, *, horizon_sessions=20):
    """Recompute point and coherent model-set bounds, or reject every defect.

    The path-level uncertainty set is a *convex hull of complete vectors*. The
    simultaneous extrema in different assets are never selected independently.
    These bounds are conditional on model-supplied assumptions, not 95% coverage.
    """
    rows, nav = _original_positions(positions, notional)
    signature = portfolio_signature(positions, notional, horizon_sessions)
    if not isinstance(branches, list) or not 2 <= len(branches) <= MAX_BRANCHES:
        raise JointImpactInputError("Two to four ordered conditional-on-E branches are required.")
    ids, sums, p_low, p_mid, p_high = set(), [], [], [], []
    residual_count = 0
    for i, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise JointImpactInputError("Every branch must be an object.")
        ident = branch.get("id")
        if not isinstance(ident, str) or not 1 <= len(ident) <= 64 or ident in ids:
            raise JointImpactInputError("Branch identifiers must be unique.")
        ids.add(ident)
        if branch.get("residual") is True:
            residual_count += 1
            if i != len(branches)-1:
                raise JointImpactInputError("Only the last branch may be residual.")
        elif branch.get("residual") is not False:
            raise JointImpactInputError("Branch residual flags must be explicit.")
        trigger = branch.get("trigger")
        if not isinstance(trigger, str) or not 8 <= len(trigger.strip()) <= 650:
            raise JointImpactInputError("Each branch needs an operational trigger.")
        a, c, b = _weight_triple(branch.get("weight_given_event"), ident)
        p_low.append(a)
        p_mid.append(c)
        p_high.append(b)
        paths = branch.get("joint_paths")
        if not isinstance(paths, dict) or set(paths) != {"adverse", "central", "favorable"}:
            raise JointImpactInputError("Three coupled holding vectors are mandatory.")
        joint = {path: _joint_vector(paths[path], rows, f"{ident}.{path}")
                 for path in ("adverse", "central", "favorable")}
        vals = [joint[p]["portfolio_gap_pct"] for p in ("adverse", "central", "favorable")]
        if vals[0] > vals[1] + _TOL or vals[1] > vals[2] + _TOL:
            raise JointImpactInputError("Coupled path labels must be ordered by full-portfolio impact.")
        sums.append({"id": ident, "residual": branch["residual"],
                     "weight_given_event": [a, c, b], "joint_paths": joint})
    if residual_count != 1:
        raise JointImpactInputError("Exactly one final residual branch is required.")
    if abs(fsum(p_mid) - 1.0) > _TOL:
        raise JointImpactInputError("Conditional-on-E central weights must sum to one.")
    if fsum(p_low) > 1 + _TOL or fsum(p_high) < 1 - _TOL:
        raise JointImpactInputError("Conditional scenario credal weights are infeasible.")
    a_values = [row["joint_paths"]["adverse"]["portfolio_gap_pct"] for row in sums]
    c_values = [row["joint_paths"]["central"]["portfolio_gap_pct"] for row in sums]
    f_values = [row["joint_paths"]["favorable"]["portfolio_gap_pct"] for row in sums]
    central = fsum(p_mid[i] * c_values[i] for i in range(len(sums)))
    lower = _bounded_simplex_extreme(p_low, p_high, a_values, maximize=False)
    upper = _bounded_simplex_extreme(p_low, p_high, f_values, maximize=True)
    if lower > central + _TOL or upper + _TOL < central:
        raise JointImpactInputError("Central estimate is outside the model credal set.")
    if not all(math.isfinite(x) for x in (central, lower, upper)):
        raise JointImpactInputError("Nonfinite portfolio impact.")
    convert = nav / 100.0
    return {
        "schema": SCHEMA,
        "portfolio_signature": signature,
        "notional_usd": nav,
        "horizon_trading_sessions": horizon_sessions,
        "gap_center_pct": central,
        "gap_model_set_pct": [lower, upper],
        "gap_center_usd": convert * central,
        "gap_model_set_usd": [convert * lower, convert * upper],
        "branch_summaries": sums,
        "evidence_class": "UNCALIBRATED_HYPOTHETICAL_JOINT_STRESS",
        "credal_math": "EXACT_BOUNDED_SIMPLEX_LP_OVER_COUPLED_HOLDING_PATH_CONVEX_HULLS",
        "credal_limits": "MODEL_PROPOSAL_BOUNDS_NOT_FREQUENTIST_CONFIDENCE_INTERVALS",
        "estimated_causal_event_impact": False,
        "absolute_event_yes_pnl_from_current_nav_identified": False,
        "market_event_probability_used": False,
        "equity_pricing_q_used": False,
        "original_etf_nav_counted_once": True,
    }
