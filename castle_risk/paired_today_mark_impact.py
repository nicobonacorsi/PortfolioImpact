"""Paired, mathematically coupled event-YES / no-event future mark scenarios.

R118 point 3: both modeled state returns start at the ACTUAL marked NAV as of
capture and finish 20 US trading sessions after the explicitly ASSUMED event
onset date. This is a conditional hypothetical scenario estimate, not an
identified causal response, observed future return, or inferred equity priced-in
fraction. Markets' P(E)/Q never enter these equations.

Each branch/regime pairs a complete model-proposed no-event holding return
vector with the SAME complete R116 event-versus-no-event gap vector. Event YES
returns are computed, never independently hallucinated: Y_i = N_i + G_i.
The LP acts on complete paired full-portfolio paths and branch probabilities.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from math import fsum, isfinite

from .joint_credal_impact import (
    JointImpactInputError, _original_positions, _finite, _weight_triple,
    _joint_vector, _bounded_simplex_extreme, evaluate_joint_impact,
    portfolio_signature,
)

SCHEMA = "castle.paired_today_mark_impact.v1"
HORIZON_TRADING_SESSIONS = 20
LABELS = ("adverse", "central", "favorable")
_TOL = 1e-7


class PairedTodayMarkError(JointImpactInputError):
    """Refuse a numerical YES-from-today P&L on any missing coupled premise."""


def _iso_day(value, label):
    if not isinstance(value, str):
        raise PairedTodayMarkError(label + " must be ISO date.")
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise PairedTodayMarkError(label + " must be ISO date.") from None
    if day.isoformat() != value:
        raise PairedTodayMarkError(label + " must be canonical ISO date.")
    return day


def _complete_no_event_path(value, positions, label):
    if not isinstance(value, list) or len(value) != len(positions):
        raise PairedTodayMarkError(label + " must enumerate every ORIGINAL holding.")
    weights = {row["symbol"]: row for row in positions}
    result = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"symbol", "no_event_return_pct"}:
            raise PairedTodayMarkError(label + " has a malformed model return.")
        symbol = entry.get("symbol")
        if symbol not in weights or symbol in result:
            raise PairedTodayMarkError(label + " repeats or substitutes a holding.")
        move = _finite(entry["no_event_return_pct"], label + "." + symbol,
                       minimum=-100., maximum=200.)
        if weights[symbol]["type"] == "cash" and move != 0:
            raise PairedTodayMarkError("Nominal USD cash has zero modeled mark return.")
        result[symbol] = move
    return result


def _extreme(branches, lows, highs, name, *, maximize):
    vals = [max(b["joint_paths"][label][name] for label in LABELS)
            if maximize else min(b["joint_paths"][label][name] for label in LABELS)
            for b in branches]
    return _bounded_simplex_extreme(lows, highs, vals, maximize=maximize)


def evaluate_paired_today_mark(positions, notional, branches, *,
                               as_of_utc, deadline, observed_mark_context):
    """Recompute coupled model YES/NO dollars, exact LP and all provenance gates.

    No bounds are formed by subtracting separate YES and NO extrema: the event
    contrast is the difference of each PAIRED path before optimization.
    """
    originals, nav = _original_positions(positions, notional)
    signature = portfolio_signature(originals, nav, HORIZON_TRADING_SESSIONS)
    if (not isinstance(observed_mark_context, dict)
            or observed_mark_context.get("status") !=
               "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
            or observed_mark_context.get("portfolio_signature") != signature
            or observed_mark_context.get("as_of_utc") != as_of_utc):
        raise PairedTodayMarkError("No matching real completed-price mark context.")
    try:
        snapshot = datetime.fromisoformat(str(as_of_utc).replace("Z", "+00:00"))
        if snapshot.utcoffset() is None:
            raise ValueError
        capture_day = snapshot.astimezone(timezone.utc).date()
    except (ValueError, TypeError):
        raise PairedTodayMarkError("Missing timezone-aware capture.") from None
    deadline_day = _iso_day(deadline, "original deadline")
    if capture_day >= deadline_day:
        raise PairedTodayMarkError("The exact event deadline is not in the future.")

    # Original relative-gap math must already pass every simplex and joint
    # vector check; do not silently change older point-3 event assumptions.
    reference = evaluate_joint_impact(originals, nav, branches,
                                      horizon_sessions=HORIZON_TRADING_SESSIONS)
    summaries, lows, mid, highs, onset_dates = [], [], [], [], []
    for branch in branches:
        ident = branch["id"]
        day = _iso_day(branch.get("assumed_onset_date"),
                       ident + ".assumed_onset_date")
        if not capture_day < day <= deadline_day:
            raise PairedTodayMarkError("Assumed onset must occur after capture and by the ORIGINAL deadline.")
        rationale = branch.get("no_event_baseline_rationale")
        if not isinstance(rationale, str) or not 20 <= len(rationale.strip()) <= 550:
            raise PairedTodayMarkError("Future no-event baseline needs explicit, bounded rationale.")
        baseline = branch.get("no_event_current_mark_paths")
        if not isinstance(baseline, dict) or set(baseline) != set(LABELS):
            raise PairedTodayMarkError("Three paired current-mark NO-event paths are required.")
        a, c, b = _weight_triple(branch["weight_given_event"], ident)
        lows.append(a)
        mid.append(c)
        highs.append(b)
        onset_dates.append(day.isoformat())
        paired_paths = {}
        for label in LABELS:
            proposed_gap = _joint_vector(branch["joint_paths"][label], originals,
                                         ident + "." + label + ".gap")
            modeled_no = _complete_no_event_path(baseline[label], originals,
                                                  ident + "." + label + ".no")
            return_yes, return_gap = {}, {}
            for holding in originals:
                symbol = holding["symbol"]
                gap = proposed_gap["holding_gap_pct"][symbol]
                yes = modeled_no[symbol] + gap
                if not isfinite(yes) or yes < -100 - _TOL or yes > 200 + _TOL:
                    raise PairedTodayMarkError("Conditional YES return violates unlevered operational guard.")
                if holding["type"] == "cash" and yes != 0:
                    raise PairedTodayMarkError("Nominal USD cash may not reprice.")
                return_yes[symbol] = yes
                return_gap[symbol] = gap
            no_pct = fsum(row["weight"] * modeled_no[row["symbol"]] for row in originals)
            yes_pct = fsum(row["weight"] * return_yes[row["symbol"]] for row in originals)
            gap_pct = fsum(row["weight"] * return_gap[row["symbol"]] for row in originals)
            if abs((yes_pct - no_pct) - gap_pct) > _TOL:
                raise PairedTodayMarkError("Paired full-portfolio YES-NO-gap arithmetic failed.")
            paired_paths[label] = {
                "event_yes_from_today_pct": yes_pct,
                "event_no_from_today_pct": no_pct,
                "event_gap_pct": gap_pct,
                "holding_yes_pct": return_yes,
                "holding_no_pct": modeled_no,
                "holding_gap_pct": return_gap,
            }
        summaries.append({"id": ident, "residual": branch["residual"],
                          "assumed_onset_date": day.isoformat(),
                          "no_event_baseline_rationale": rationale,
                          "weight_given_event": [a,c,b],
                          "joint_paths": paired_paths})
    if abs(fsum(mid) - 1) > _TOL or fsum(lows) > 1 + _TOL or fsum(highs) < 1 - _TOL:
        raise PairedTodayMarkError("Infeasible conditional scenario probabilities.")

    def center(key):
        return fsum(mid[i] * row["joint_paths"]["central"][key]
                    for i, row in enumerate(summaries))

    def set_bounds(key):
        return [_extreme(summaries, lows, highs, key, maximize=False),
                _extreme(summaries, lows, highs, key, maximize=True)]

    yc = center("event_yes_from_today_pct")
    nc = center("event_no_from_today_pct")
    gc = center("event_gap_pct")
    yb = set_bounds("event_yes_from_today_pct")
    nb = set_bounds("event_no_from_today_pct")
    gb = set_bounds("event_gap_pct")
    if (abs((yc - nc) - gc) > _TOL
            or abs(gc - reference["gap_center_pct"]) > _TOL
            or any(abs(a-b) > _TOL for a,b in zip(gb, reference["gap_model_set_pct"]))):
        raise PairedTodayMarkError("Paired stress gap disagrees with original joint-stress model.")
    if not yb[0] - _TOL <= yc <= yb[1] + _TOL:
        raise PairedTodayMarkError("Central conditional YES return is outside credal set.")
    if not nb[0] - _TOL <= nc <= nb[1] + _TOL:
        raise PairedTodayMarkError("Central conditional NO return is outside credal set.")

    convert = nav / 100.
    return {
        "schema": SCHEMA, "status": "MODEL_TODAY_MARK_STRESS_UNCALIBRATED",
        "portfolio_signature": signature,
        "as_of_utc": as_of_utc, "deadline": deadline,
        "return_horizon_trading_sessions_after_assumed_onset": HORIZON_TRADING_SESSIONS,
        "horizon_calendar_end_is_common_across_branches": False,
        "assumed_onset_dates": onset_dates,
        "market_context_last_completed_session": observed_mark_context["last_completed_session"],
        "event_yes_from_today_center_pct": yc,
        "event_yes_from_today_model_set_pct": yb,
        "event_yes_from_today_center_usd": convert * yc,
        "event_yes_from_today_model_set_usd": [convert * x for x in yb],
        "event_no_from_today_center_pct": nc,
        "event_no_from_today_model_set_pct": nb,
        "event_no_from_today_center_usd": convert * nc,
        "event_no_from_today_model_set_usd": [convert * x for x in nb],
        "paired_event_gap_center_pct": gc,
        "paired_event_gap_model_set_pct": gb,
        "conditional_branch_summaries": summaries,
        "pricing_q_estimated": False,
        "prediction_market_Q_used_as_equity_q": False,
        "physical_event_probability_used": False,
        "event_return_caused_empirically": False,
        "forward_returns_calibrated": False,
        "future_pnl_observed_or_identified": False,
        "branch_timing_assumed_not_forecast_calibrated": True,
        "bounds_are_statistical_confidence_intervals": False,
        "original_etf_constituents_added_to_nav": False,
        "model_class": "JOINT_STRUCTURAL_HYPOTHETICAL_TODAY_MARK_PAIRS",
        "math_assurance": "SIGNED_EXACT_SIMPLEX_LP_OVER_FULL_COUPLED_YES_NO_EVENT_PATHS",
        "assurance_limit": "ASSUMPTION_CONDITIONAL_NO_CAUSAL_OR_OOS_FORECAST_VALIDATION",
    }


def verified_saved_paired_today_mark(saved, positions, notional, branches, *,
                                     as_of_utc, deadline, observed_mark_context):
    """Full deterministic recheck; rejects altered P&L, weights or extra flags."""
    if not isinstance(saved, dict):
        return False
    try:
        expected = evaluate_paired_today_mark(
            positions, notional, branches, as_of_utc=as_of_utc, deadline=deadline,
            observed_mark_context=observed_mark_context)
    except (PairedTodayMarkError, JointImpactInputError, KeyError, TypeError, ValueError):
        return False
    return saved == expected
