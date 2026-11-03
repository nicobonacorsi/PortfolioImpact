"""Shared R10 state representatives for production and R11 chain evaluation.

This module is deliberately standard-library-only. Moving the state function
here does not change its representatives, finite-tail assumptions, or errors.
Finite tail intervals are declared scenario domains, never source support.
"""
from __future__ import annotations

import math

from .portfolio import _finite_number


def _number(value, name):
    return _finite_number(value, name)


def _range(value, name, *, point=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " requires two endpoints")
    lo, hi = (_number(x, name) for x in value)
    if lo > hi or point is not None and not lo <= point <= hi:
        raise ValueError(name + " is unordered or excludes its point")
    return [lo, hi]


def _state_model(atoms, family):
    rows = []
    finite_cpi = [float(x) for atom in atoms for x in (atom.get("value_interval") or [])
                  if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    for atom in atoms:
        value = atom.get("value")
        source_bounds = atom.get("value_interval") or [None, None]
        if not isinstance(source_bounds, (list, tuple)) or len(source_bounds) != 2:
            raise ValueError("Source state interval requires two endpoints")
        lo, hi = source_bounds
        basis = "SOURCE_VALUE_OR_BOUNDED_BIN_WITH_DECLARED_MIDPOINT"
        if value is not None:
            value = _number(value, "Source state")
            if lo is None and hi is None:
                lo = hi = value
        if lo is not None and hi is not None:
            bounds = _range([lo, hi], "Source state interval", point=value)
            value = math.fsum(bounds) / 2 if value is None else value
        elif family == "FED_POLICY":
            direction = str(atom.get("direction", "")).lower()
            if direction == "cut":
                value, bounds = -50., [-100., -25.]
            elif direction == "hike":
                value, bounds = 50., [25., 100.]
            elif direction == "hold":
                value, bounds = 0., [0., 0.]
            else:
                value, bounds = 0., [-100., 100.]
            basis = "DECLARED_ORDINARY_POLICY_MAGNITUDE_DOMAIN_NOT_SOURCE_CERTIFIED"
        else:
            if lo is None and hi is not None:
                hi = _number(hi, "CPI tail boundary")
                value, bounds = hi - .1, [hi - .5, hi]
            elif lo is not None and hi is None:
                lo = _number(lo, "CPI tail boundary")
                value, bounds = lo + .1, [lo, lo + .5]
            else:
                lower = min(finite_cpi, default=-.5) - .5
                upper = max(finite_cpi, default=.5) + .5
                value, bounds = (lower + upper) / 2, [lower, upper]
            basis = "DECLARED_FINITE_CPI_TAIL_SCENARIO_NOT_A_BOUND_ON_THE_SOURCE_SUPPORT"
        if not all(math.isfinite(x) for x in [value, *bounds]):
            raise ValueError("Nonfinite economic state")
        rows.append({"outcome_id": atom.get("outcome_id", atom.get("atom_id")),
                     "value": value, "value_interval": bounds,
                     "source_value_interval": list(source_bounds), "basis": basis,
                     "numeric_mapping_status": atom.get("numeric_mapping_status", "UNKNOWN"),
                     "not_a_source_support_claim": "DECLARED" in basis})
    return rows


state_model = _state_model
