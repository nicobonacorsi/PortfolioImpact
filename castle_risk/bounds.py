"""Finite Boolean coherence bounds, independent of language-model behavior.

Inputs are *normalized pricing-measure* intervals, not physical probabilities.
The calling semantic verifier is responsible for setting ``verified`` correctly;
this module proves consequences of supplied constraints, not their truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any
import math

import numpy as np
from scipy.optimize import linprog


RELATION_KINDS = frozenset({"EQUIVALENT", "IMPLIES", "IMPLIED_BY", "EXCLUDES", "NONE"})
MAX_SUPPORTED_EVENTS = 16
NUMERICAL_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Relation:
    """``IMPLIES`` means left => right; ``IMPLIED_BY`` means right => left."""

    left: str
    kind: str
    right: str
    evidence: str = ""
    verified: bool = True


@dataclass(frozen=True)
class QuoteInterval:
    """A caller-normalized bid/ask constraint under one named pricing measure.

    A stale, halted, missing-sided, or otherwise excluded observation can be
    retained for audit with ``usable=False`` and an explicit exclusion reason.
    Freshness itself is assessed by the market-data adapter, not this LP.
    """

    event_id: str
    lower: float
    upper: float
    measure_id: str = "kalshi:USD:common-settlement"
    source_id: str = ""
    usable: bool = True
    exclusion_reason: str = ""


@dataclass(frozen=True)
class BoundsResult:
    status: str
    lower: float | None
    upper: float | None
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable output, including a sparse optimal witness."""
        return asdict(self)


def _failure(status: str, message: str, diagnostics: dict[str, Any]) -> BoundsResult:
    return BoundsResult(status, None, None, message, diagnostics)


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _probability(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _witness(atoms: np.ndarray, masses: np.ndarray, event_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {"mass": float(mass), "true_events": [name for name, yes in zip(event_ids, atom) if yes]}
        for atom, mass in zip(atoms, masses)
        if mass > NUMERICAL_TOLERANCE
    ]


def coherent_bounds(
    target: str,
    relations: list[Relation],
    quotes: list[QuoteInterval],
    max_events: int = 12,
) -> BoundsResult:
    """Find sharp lower/upper marginal bounds on a finite Boolean event.

    Only verified non-NONE relations are logical constraints. Only quotes in
    the target's connected logical component are admitted; disconnected inputs
    are exposed in diagnostics. Thus an unrelated inconsistent family cannot
    invalidate a target bound. All admitted intervals must share a measure ID.

    Sharpness is conditional on supplied logical truth and on existence of a
    common normalized pricing measure with expectations in the input intervals.
    HiGHS solves the finite LP to a numerical tolerance, not exact arithmetic.
    """
    diagnostics: dict[str, Any] = {
        "target": target,
        "interpretation": "Conditional normalized pricing-measure bounds; not confidence intervals or physical-probability estimates.",
        "constraint_scope": "target_verified_semantic_component",
        "numerical_tolerance": NUMERICAL_TOLERANCE,
    }
    if not _valid_id(target):
        return _failure("INVALID_INPUT", "Target must be a nonempty event identifier.", diagnostics)
    if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= MAX_SUPPORTED_EVENTS:
        return _failure("INVALID_INPUT", f"max_events must be an integer between 1 and {MAX_SUPPORTED_EVENTS}.", diagnostics)
    if not isinstance(relations, (list, tuple)) or not isinstance(quotes, (list, tuple)):
        return _failure("INVALID_INPUT", "Relations and quotes must be lists or tuples of their schema objects.", diagnostics)

    for index, relation in enumerate(relations):
        if not isinstance(relation, Relation):
            return _failure("INVALID_INPUT", f"Relation {index} does not satisfy the Relation schema.", diagnostics)
        if not _valid_id(relation.left) or not _valid_id(relation.right) or not isinstance(relation.kind, str) or relation.kind not in RELATION_KINDS:
            return _failure("INVALID_INPUT", f"Relation {index} has invalid event IDs or an unsupported relation kind.", diagnostics)
        if not isinstance(relation.verified, bool) or not isinstance(relation.evidence, str):
            return _failure("INVALID_INPUT", f"Relation {index} has invalid verification metadata.", diagnostics)
    for index, quote in enumerate(quotes):
        if not isinstance(quote, QuoteInterval):
            return _failure("INVALID_INPUT", f"Quote {index} does not satisfy the QuoteInterval schema.", diagnostics)
        if not _valid_id(quote.event_id) or not _valid_id(quote.measure_id):
            return _failure("INVALID_INPUT", f"Quote {index} has invalid event or pricing-measure identifiers.", diagnostics)
        if not _probability(quote.lower) or not _probability(quote.upper) or quote.lower > quote.upper:
            return _failure("INVALID_INPUT", f"Quote {index} must satisfy finite 0 <= lower <= upper <= 1 after normalization.", diagnostics)
        if not isinstance(quote.usable, bool) or not isinstance(quote.exclusion_reason, str) or not isinstance(quote.source_id, str):
            return _failure("INVALID_INPUT", f"Quote {index} has invalid eligibility or source metadata.", diagnostics)

    verified = [r for r in relations if r.verified and r.kind != "NONE"]
    adjacency: dict[str, set[str]] = {}
    for relation in verified:
        adjacency.setdefault(relation.left, set()).add(relation.right)
        adjacency.setdefault(relation.right, set()).add(relation.left)
    component = {target}
    pending = [target]
    while pending:
        for neighbor in adjacency.get(pending.pop(), set()):
            if neighbor not in component:
                component.add(neighbor)
                pending.append(neighbor)

    active_relations = [r for r in verified if r.left in component and r.right in component]
    relevant_quotes = [q for q in quotes if q.event_id in component]
    active_quotes = [q for q in relevant_quotes if q.usable]
    diagnostics.update({
        "events": sorted(component),
        "admitted_relations": [asdict(r) for r in active_relations],
        "ignored_unverified_relations": [asdict(r) for r in relations if not r.verified],
        "semantic_nonmatches": [asdict(r) for r in relations if r.kind == "NONE"],
        "disconnected_relations": [asdict(r) for r in verified if r.left not in component],
        "disconnected_quotes": [asdict(q) for q in quotes if q.event_id not in component],
        "excluded_quotes": [asdict(q) for q in relevant_quotes if not q.usable],
        "admitted_quotes": [asdict(q) for q in active_quotes],
    })
    # A direct observation on the target requires no cross-event relationship.
    # A direct but unusable observation is price insufficiency, not a nonmatch.
    if not active_relations and not relevant_quotes:
        return _failure("NO_SEMANTIC_MATCH", "No verified logical relationship or direct target observation is available.", diagnostics)
    if not active_quotes:
        return _failure("INSUFFICIENT_PRICE_EVIDENCE", "A semantic link exists, but no usable normalized quotation constrains its component.", diagnostics)
    measures = sorted({q.measure_id for q in active_quotes})
    if len(measures) != 1:
        diagnostics["measure_ids"] = measures
        return _failure("INVALID_INPUT", "Relevant quotations name different pricing measures; cross-measure pooling is not permitted.", diagnostics)
    diagnostics["measure_id"] = measures[0]
    if len(component) > max_events:
        diagnostics["event_limit"] = max_events
        return _failure("MODEL_TOO_LARGE", "The semantic component exceeds the configured Boolean-enumeration limit.", diagnostics)

    event_ids = sorted(component)
    event_index = {name: index for index, name in enumerate(event_ids)}
    bit_numbers = np.arange(1 << len(event_ids), dtype=np.uint32)
    atoms = ((bit_numbers[:, None] >> np.arange(len(event_ids), dtype=np.uint32)) & 1).astype(np.uint8)
    valid = np.ones(len(atoms), dtype=bool)
    for relation in active_relations:
        left = atoms[:, event_index[relation.left]].astype(bool)
        right = atoms[:, event_index[relation.right]].astype(bool)
        if relation.kind == "EQUIVALENT":
            valid &= left == right
        elif relation.kind == "IMPLIES":
            valid &= ~left | right
        elif relation.kind == "IMPLIED_BY":
            valid &= ~right | left
        elif relation.kind == "EXCLUDES":
            valid &= ~(left & right)
    atoms = atoms[valid]
    diagnostics["unrestricted_atom_count"] = int(len(valid))
    diagnostics["admissible_atom_count"] = int(len(atoms))

    # Duplicate observations mean intersecting constraints, never averaging.
    interval_by_event: dict[str, tuple[float, float]] = {}
    for quote in active_quotes:
        previous = interval_by_event.get(quote.event_id, (0.0, 1.0))
        interval_by_event[quote.event_id] = (max(previous[0], float(quote.lower)), min(previous[1], float(quote.upper)))
    diagnostics["aggregated_intervals"] = {name: list(interval) for name, interval in interval_by_event.items()}
    for event_id, (lower, upper) in interval_by_event.items():
        if lower > upper:
            return _failure("INFEASIBLE", f"Separately valid quotation intervals have empty intersection for {event_id}. This does not establish executable arbitrage.", diagnostics)
    if len(atoms) == 0:
        return _failure("INFEASIBLE", "No Boolean atom satisfies the admitted logical constraints.", diagnostics)

    inequalities: list[np.ndarray] = []
    limits: list[float] = []
    for event_id, (lower, upper) in interval_by_event.items():
        indicator = atoms[:, event_index[event_id]].astype(float)
        inequalities.extend([indicator, -indicator])
        limits.extend([upper, -lower])
    objective = atoms[:, event_index[target]].astype(float)
    common = {
        "A_ub": np.asarray(inequalities),
        "b_ub": np.asarray(limits),
        "A_eq": np.ones((1, len(atoms))),
        "b_eq": np.array([1.0]),
        "bounds": (0, None),
        "method": "highs",
        "options": {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    }
    minimum = linprog(objective, **common)
    diagnostics["min_solver_status"] = int(minimum.status)
    diagnostics["min_solver_message"] = str(minimum.message)
    if minimum.status == 2:
        return _failure("INFEASIBLE", "No common pricing measure satisfies the supplied logic and quote intervals. Check rule verification, settlement assumptions, timestamps, and prices; this is not an arbitrage certificate.", diagnostics)
    if not minimum.success:
        return _failure("SOLVER_ERROR", "The LP solver did not establish a feasible optimum.", diagnostics)
    maximum = linprog(-objective, **common)
    diagnostics["max_solver_status"] = int(maximum.status)
    diagnostics["max_solver_message"] = str(maximum.message)
    if not maximum.success:
        return _failure("SOLVER_ERROR", "The LP solver did not establish the upper optimum.", diagnostics)
    # Bounds are clipped only for numerical roundoff, never to repair invalid input.
    raw_lower, raw_upper = float(minimum.fun), float(-maximum.fun)
    if raw_lower < -NUMERICAL_TOLERANCE or raw_upper > 1 + NUMERICAL_TOLERANCE or raw_lower > raw_upper + NUMERICAL_TOLERANCE:
        return _failure("SOLVER_ERROR", "The solver returned numerically invalid probability extrema.", diagnostics)
    lower, upper = max(0.0, min(1.0, raw_lower)), max(0.0, min(1.0, raw_upper))
    diagnostics["lower_witness"] = _witness(atoms, minimum.x, event_ids)
    diagnostics["upper_witness"] = _witness(atoms, maximum.x, event_ids)
    diagnostics["width"] = upper - lower
    if lower <= NUMERICAL_TOLERANCE and upper >= 1 - NUMERICAL_TOLERANCE:
        return BoundsResult("UNINFORMATIVE", lower, upper, "The model is feasible, but its bounds are uninformative to numerical tolerance.", diagnostics)
    return BoundsResult("BOUNDED", lower, upper, "Sharp bounds under the admitted logical and common pricing-measure assumptions, subject to numerical solver tolerance.", diagnostics)
