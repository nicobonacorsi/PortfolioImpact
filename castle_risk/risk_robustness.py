"""Auditable robustness of the *modeled* event ranking, without inventing P.

The incumbent UI ranks an event by the reference-Q standard deviation of its
conditional, not-yet-priced dollar P&L.  This module tests how that ranking
changes when source quote weights and *declared* impact severity change.

For each payout vector x, the source's marginal bid/ask bounds are intersected
with the unit simplex. They are an OUTER relaxation of any source joint quote
constraints. Therefore, a non-overlapping ranking interval is a valid
conditional certificate; overlapping intervals alone prove NO rank reversal.

Var_q(x) = E_q[x*x] - E_q[x]^2. Linear extrema for both moments give a safe
outer bound without assuming independence of market outcomes. Translating x
by any common priced-in expectation leaves Var_q(x) invariant, so this metric
does not separately identify the asset market's embedded expectation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


class RobustnessInputError(ValueError):
    """Invalid marginal interval or modeled scenario; never silently repair."""


def _real(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RobustnessInputError(name + " must be finite")
    return float(value)


def _q_and_pnl(outcomes: Sequence[Mapping]):
    if not isinstance(outcomes, list) or len(outcomes) < 2:
        raise RobustnessInputError("At least two complete outcomes required")
    q, limits, pnl = [], [], []
    identities = set()
    for row in outcomes:
        key = row.get("outcome_id")
        if not isinstance(key, str) or not key or key in identities:
            raise RobustnessInputError("Outcome IDs must be distinct")
        identities.add(key)
        reference = _real(row.get("q_reference"), "Q reference")
        interval = row.get("q_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise RobustnessInputError("Missing source Q interval")
        low, high = (_real(v, "Q bound") for v in interval)
        if not 0 <= low <= reference + 1e-9 or not reference <= high + 1e-9 or high > 1 + 1e-9:
            raise RobustnessInputError("Q interval is inconsistent")
        limits.append((low, high))
        q.append(reference)
        pnl.append(_real(row.get("pnl_if_outcome_usd"), "Complete scenario P&L"))
    if not math.isclose(math.fsum(q), 1, abs_tol=1e-6, rel_tol=0):
        raise RobustnessInputError("Q reference is not normalized")
    if math.fsum(lo for lo, _ in limits) > 1 + 1e-7 or math.fsum(hi for _, hi in limits) < 1 - 1e-7:
        raise RobustnessInputError("Quote interval relaxation has empty simplex")
    # A nonzero Q-weighted residual means these are not the priced-in P&L rows
    # we claim to audit (or the source artifact was corrupted).
    residual = math.fsum(w * x for w, x in zip(q, pnl))
    if abs(residual) > 1e-6 * max(1, *(abs(x) for x in pnl)):
        raise RobustnessInputError("Not-yet-priced payout vector fails Q-centering")
    return q, limits, pnl


def linear_extreme(values, limits, *, maximize=False):
    """Exact LP over a box intersected with sum(q)=1 (greedy exchange)."""
    if len(values) != len(limits) or not values:
        raise RobustnessInputError("LP dimensions differ")
    base = math.fsum(lo * val for val, (lo, _) in zip(values, limits))
    remaining = 1 - math.fsum(lo for lo, _ in limits)
    if remaining < -1e-7 or remaining > math.fsum(hi-lo for lo, hi in limits) + 1e-7:
        raise RobustnessInputError("Marginal bounds have no simplex intersection")
    for index in sorted(range(len(values)), key=lambda i: values[i], reverse=maximize):
        lo, hi = limits[index]
        add = min(max(0., remaining), max(0., hi - lo))
        base += add * values[index]
        remaining -= add
    if remaining > 1e-7:
        raise RobustnessInputError("Insufficient quote capacity")
    return base


def q_std_bounds(outcomes, *, pnl_override=None):
    """Conservative moment-relaxation certificate for source Q uncertainty."""
    q, limits, original = _q_and_pnl(outcomes)
    x = original if pnl_override is None else [_real(v, "Alternate modeled P&L") for v in pnl_override]
    if len(x) != len(q):
        raise RobustnessInputError("Alternate calibration changed the outcome partition")
    # Alternative calibrations must be centered on the SAME frozen Q.
    residual = math.fsum(a*b for a, b in zip(q, x))
    if abs(residual) > 1e-6 * max(1, *(abs(v) for v in x)):
        raise RobustnessInputError("Alternate calibration breaks reference Q-centering")
    mu_lo = linear_extreme(x, limits)
    mu_hi = linear_extreme(x, limits, maximize=True)
    second = [v*v for v in x]
    m2_lo = linear_extreme(second, limits)
    m2_hi = linear_extreme(second, limits, maximize=True)
    farthest_squared = max(mu_lo**2, mu_hi**2)
    nearest_squared = 0 if mu_lo <= 0 <= mu_hi else min(mu_lo**2, mu_hi**2)
    variance_lo = max(0., m2_lo - farthest_squared)
    variance_hi = max(0., m2_hi - nearest_squared)
    reference = math.sqrt(max(0., math.fsum(w * v*v for w, v in zip(q, x)) - residual**2))
    lo, hi = math.sqrt(variance_lo), math.sqrt(variance_hi)
    if lo > reference + 1e-5 or reference > hi + 1e-5:
        raise RobustnessInputError("Variance outer bound excluded the reference Q")
    return {"reference_q_std_usd": reference, "quote_only_lower_usd": lo,
            "quote_only_upper_usd": hi, "mean_pnl_under_marginal_q_bounds_usd": [mu_lo, mu_hi],
            "bounding_method": "MOMENT_LP_OUTER_RELAXATION_NOT_EXACT_Q_STD_EXTREMA"}


def _scenarios(event, alternative_vectors):
    table = event.get("outcome_table") or {}
    if isinstance(table, Mapping) and str(table.get("status", "")).startswith("COMPLETE_") and table.get("outcomes"):
        yield "source_outcome_table", table["outcomes"], (alternative_vectors or {}).get(event["event_id"], [])
    for scenario in ((event.get("contract_first") or {}).get("contract_scenarios") or []):
        if scenario.get("status") == "DECLARED_CONTRACT_SCENARIO" and scenario.get("outcomes"):
            yield scenario.get("contract", {}).get("contract_id", "contract_scenario"), scenario["outcomes"], []


def analyze_ranking(events, *, severity_domain=(.5, 1.5), alternative_vectors=None):
    """One analysis of all priced cards; unpriceable risks remain a first-class list.

    Severity is an *independently declared stress multiplier*, not a confidence
    region or identified parameter. Contract scenarios are *not added*; an
    event's baseline ranking uses the largest individually priced scenario.
    """
    if not isinstance(events, list):
        raise RobustnessInputError("An ordered event card list is required")
    if not isinstance(severity_domain, (tuple, list)) or len(severity_domain) != 2:
        raise RobustnessInputError("Severity domain needs two endpoints")
    low_scale, high_scale = (_real(v, "Severity multiplier") for v in severity_domain)
    if not 0 <= low_scale <= 1 <= high_scale or high_scale > 10:
        raise RobustnessInputError("Stress domain must contain 1 and remain within [0,10]")
    modeled, watchlist = [], []
    for event in events:
        event_id, title = event["event_id"], event["title"]
        admitted, exclusions = [], []
        for label, outcome_rows, alternatives in _scenarios(event, alternative_vectors):
            try:
                candidates = [("reference_model", q_std_bounds(outcome_rows))]
                for alternate in alternatives:
                    candidates.append((alternate["name"], q_std_bounds(outcome_rows,
                                                                         pnl_override=alternate["pnl_if_outcome_usd"])))
            except (RobustnessInputError, KeyError, TypeError) as exc:
                exclusions.append({"scenario": label, "reason": str(exc)})
                continue
            reference = candidates[0][1]["reference_q_std_usd"]
            q_lower = min(row["quote_only_lower_usd"] for _, row in candidates)
            q_upper = max(row["quote_only_upper_usd"] for _, row in candidates)
            admitted.append({"scenario": label, "reference_q_std_usd": reference,
                             "quote_and_calibration_interval_usd": [q_lower, q_upper],
                             "calibrations": [name for name, _ in candidates],
                             "calibration_scores_at_reference_q": [
                                 {"calibration": name, "q_std_usd": score["reference_q_std_usd"]}
                                 for name, score in candidates],
                             "stress_interval_usd": [low_scale*q_lower, high_scale*q_upper],
                             "quote_only_interval_reference_calibration_usd": [
                                 candidates[0][1]["quote_only_lower_usd"],
                                 candidates[0][1]["quote_only_upper_usd"]]})
        if not admitted:
            watchlist.append({"event_id": event_id, "title": title, "family": event.get("family"),
                              "reason": "NO_COMPLETE_QUOTE_AND_PORTFOLIO_COMPONENT" if not exclusions
                              else "INPUT_CERTIFICATION_FAILED", "excluded_scenarios": exclusions})
            continue
        # A constructive witness uses the *feasible reference Q* for each
        # scenario, an actually enumerated historical calibration (or the
        # reference model), and one allowed severity factor PER EVENT. This
        # is stronger than overlapping conservative outer intervals: every
        # selected stress configuration can be written down explicitly.
        low_config, upper_choices = [], []
        for scenario in admitted:
            ordered = sorted(scenario["calibration_scores_at_reference_q"],
                             key=lambda row: (row["q_std_usd"], row["calibration"]))
            low_config.append({"scenario": scenario["scenario"], **ordered[0]})
            upper_choices.append({"scenario": scenario["scenario"], **ordered[-1]})
        ref_minimum = low_scale * max(row["q_std_usd"] for row in low_config)
        best_upper = max(upper_choices, key=lambda row: (row["q_std_usd"], row["scenario"]))
        ref_maximum = high_scale * best_upper["q_std_usd"]
        modeled.append({"event_id": event_id, "title": title, "family": event.get("family"),
                        "reference_risk_usd": max(row["reference_q_std_usd"] for row in admitted),
                        "feasible_min_at_reference_q_usd": ref_minimum,
                        "feasible_max_at_reference_q_usd": ref_maximum,
                        "witness_min_configuration": {"severity_multiplier": low_scale,
                                                      "scenario_calibrations": low_config},
                        "witness_max_dominant_scenario": {"severity_multiplier": high_scale,
                                                          **best_upper},
                        "quote_and_calibration_interval_usd": [
                            max(row["quote_and_calibration_interval_usd"][0] for row in admitted),
                            max(row["quote_and_calibration_interval_usd"][1] for row in admitted)],
                        "stress_interval_usd": [
                            max(row["stress_interval_usd"][0] for row in admitted),
                            max(row["stress_interval_usd"][1] for row in admitted)],
                        "priced_scenarios": admitted, "excluded_scenarios": exclusions,
                        "broad_event_probability_certified": False if event.get("family") in
                          {"TAIWAN_SUPPLY", "EXPORT_CONTROLS", "HORMUZ_ENERGY", "OPEC_SUPPLY", "RUSSIA_ENERGY_SANCTIONS"}
                          else None})
    modeled.sort(key=lambda row: (-row["reference_risk_usd"], row["event_id"]))
    leader = modeled[0] if modeled else None
    challengers = []
    witnessed_reversals = []
    if leader:
        for item in modeled[1:]:
            reference = item["reference_risk_usd"]
            threshold = leader["reference_risk_usd"] / reference if reference > 0 else None
            challengers.append({"event_id": item["event_id"], "title": item["title"],
                                "severity_multiplier_to_tie_reference_leader": threshold,
                                "quote_or_calibration_only_can_be_ruled_out": (
                                    leader["quote_and_calibration_interval_usd"][0] >
                                    item["quote_and_calibration_interval_usd"][1]),
                                "severity_only_can_be_ruled_out": (
                                    low_scale * leader["reference_risk_usd"] >
                                    high_scale * reference),
                                "stress_intervals_overlap": (
                                    leader["stress_interval_usd"][0] <= item["stress_interval_usd"][1])})
            other_minima = [row for row in modeled if row["event_id"] != item["event_id"]]
            most_resistant = max(other_minima, key=lambda row: row["feasible_min_at_reference_q_usd"])
            if item["feasible_max_at_reference_q_usd"] > (
                    most_resistant["feasible_min_at_reference_q_usd"] + 1e-8):
                witnessed_reversals.append({
                    "new_leader_event_id": item["event_id"], "new_leader_title": item["title"],
                    "original_leader_event_id": leader["event_id"],
                    "new_leader_stressed_q_std_usd": item["feasible_max_at_reference_q_usd"],
                    "highest_other_event_id": most_resistant["event_id"],
                    "highest_other_stressed_q_std_usd": most_resistant["feasible_min_at_reference_q_usd"],
                    "strict_margin_usd": item["feasible_max_at_reference_q_usd"] -
                                         most_resistant["feasible_min_at_reference_q_usd"],
                    "new_leader_dominant_calibration": item["witness_max_dominant_scenario"],
                    "other_event_settings": [
                        {"event_id": row["event_id"], "severity_multiplier": low_scale,
                         "scenario_calibrations": row["witness_min_configuration"]["scenario_calibrations"]}
                        for row in other_minima],
                    "quote_vector_selection": "SOURCE_REFERENCE_Q_FOR_EACH_MODELED_SCENARIO",
                    "scope": "ATTAINABLE_ONLY_WITHIN_INDEPENDENT_DECLARED_MODEL_STRESSES",
                })
    certified = bool(leader and all(leader["stress_interval_usd"][0] > row["stress_interval_usd"][1]
                                    for row in modeled[1:]))
    if certified and witnessed_reversals:
        raise RobustnessInputError("A constructive stress witness contradicted the outer certificate")
    return {"schema": "castle.option1.risk_robustness.v1",
            "status": ("NO_COMPLETE_PRICED_SCENARIO" if not leader else
                       "RANK_REVERSAL_WITNESSED_IN_DECLARED_STRESS_MODEL" if witnessed_reversals else
                       "CONDITIONALLY_ROBUST_PRICED_SCENARIO_LEADER" if certified else
                       "RANK_UNRESOLVED_IN_DECLARED_STRESS_DOMAIN"),
            "leader_event_id": None if not leader else leader["event_id"],
            "severity_domain": [low_scale, high_scale], "priced_event_count": len(modeled),
            "unquantifiable_event_count": len(watchlist), "modeled_rank": modeled,
            "visible_unquantifiable_risks": watchlist, "challenger_thresholds": challengers,
            "ranking_reversal_witnesses": witnessed_reversals,
            "proof_scope": "ONLY_COMPLETE_PRICED_SCENARIOS_AND_DECLARED_MULTIPLIER_DOMAIN",
            "unknown_risk_excluded_from_certificate": bool(watchlist),
            "not_proven": ["Physical event P", "The asset pricing measure equals market Q",
                           "Model severity coverage", "Missing-event P&L", "Causal identification",
                           "A ranking reversal when conservative intervals overlap",
                           "Economic realism of independent stress choices in a constructed ranking reversal",
                           "Additivity of overlapping contract scenarios"]}
