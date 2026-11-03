"""Common-endpoint paired event-YES / NO-event portfolio scenarios (R122, v2).

Versioned successor to the R118 paired today-mark scenario. Every YES branch and
the single NO-event state are marked at ONE fixed future session T* chosen by a
deterministic calendar rule before the model is called. Mixing branches is then
a mixture of portfolio values at the same date, and gain/loss from today is
V(T*) - V0 for the same original funded NAV.

The NO-event state is one hypothetical world with three coupled regimes
(adverse / central / favorable). Each YES branch supplies, per regime, the
event-minus-no-event gap vector at T*. The verifier computes
    YES_{b,r,i} = NO_{r,i} + GAP_{b,r,i}
holding by holding and never accepts an independently stated YES vector.

Bounds are exact: for each regime r a bounded-simplex linear program over the
branch weights p_b in [l_b, u_b], sum p_b = 1, then the extreme over regimes.
They are assumption-conditional model sets, not confidence intervals. No P(E),
no venue quote Q and no equity priced-in q enter these equations. The saved
R118 v1 scenarios and the frozen R120 fixed-endpoint locks are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from math import fsum, isfinite

from .joint_credal_impact import (
    JointImpactInputError, _original_positions, _finite, _weight_triple, _joint_vector,
    _bounded_simplex_extreme,
)
from .us_trading_calendar import (
    ENDPOINT_RULES, TradingCalendarError, common_endpoint_for_event, is_trading_session,
    sessions_strictly_between,
)

SCHEMA = "castle.common_endpoint_paired_impact.v2"
REFERENCE_SCHEMA = "castle.historical_horizon_reference.v1"
LABELS = ("adverse", "central", "favorable")
_TOL = 1e-7
_MIN_REFERENCE_WINDOWS = 20


class CommonEndpointError(JointImpactInputError):
    """Refuse a numerical common-endpoint P&L on any missing or inconsistent premise."""


def _iso_day(value, label):
    if not isinstance(value, str):
        raise CommonEndpointError(label + " must be an ISO date.")
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise CommonEndpointError(label + " must be an ISO date.") from None
    if day.isoformat() != value:
        raise CommonEndpointError(label + " must be a canonical ISO date.")
    return day


def _capture_day(as_of_utc):
    try:
        snapshot = datetime.fromisoformat(str(as_of_utc).replace("Z", "+00:00"))
        if snapshot.utcoffset() is None:
            raise ValueError
    except (ValueError, TypeError):
        raise CommonEndpointError("Missing timezone-aware capture.") from None
    return snapshot.astimezone(timezone.utc).date()


def endpoint_signature(positions, notional, endpoint_session_date):
    """Consistency fingerprint of original funded holdings and the common endpoint."""
    rows, nav = _original_positions(positions, notional)
    day = _iso_day(endpoint_session_date, "common endpoint")
    canonical = {"notional": round(nav, 8), "common_endpoint_session_date": day.isoformat(),
                 "positions": [{**row, "weight": round(row["weight"], 12)} for row in rows]}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _no_event_vector(value, positions, label):
    if not isinstance(value, list) or len(value) != len(positions):
        raise CommonEndpointError(label + " must enumerate every ORIGINAL holding.")
    rows = {row["symbol"]: row for row in positions}
    result = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"symbol", "no_event_return_pct"}:
            raise CommonEndpointError(label + " has a malformed model return.")
        symbol = entry.get("symbol")
        if symbol not in rows or symbol in result:
            raise CommonEndpointError(label + " repeats or substitutes a holding.")
        move = _finite(entry["no_event_return_pct"], label + "." + symbol, minimum=-100., maximum=200.)
        if rows[symbol]["type"] == "cash" and move != 0:
            raise CommonEndpointError("Nominal USD cash has zero modeled return.")
        result[symbol] = move
    return result


def canonicalize_regimes(positions, no_event_paths, branches):
    """Sort the three NO-event whole vectors by portfolio return; transport branch gaps.

    Regime labels are economic states shared by the NO vector and every branch
    gap vector, so one permutation is applied to all of them together. No
    holding entry is ever moved independently of its complete vector.
    """
    originals, _ = _original_positions(positions, 100000.)
    if not isinstance(no_event_paths, dict) or set(no_event_paths) != set(LABELS):
        raise CommonEndpointError("Three NO-event regime vectors are required.")
    totals = []
    for i, label in enumerate(LABELS):
        vector = _no_event_vector(no_event_paths[label], originals, "no_event." + label)
        totals.append((fsum(row["weight"] * vector[row["symbol"]] for row in originals), i, label))
    totals.sort(key=lambda t: (t[0], t[1]))
    order = [t[2] for t in totals]
    if order == list(LABELS):
        return deepcopy(no_event_paths), deepcopy(branches), []
    relabeled_no = {label: deepcopy(no_event_paths[source]) for label, source in zip(LABELS, order)}
    relabeled_branches = deepcopy(branches)
    for branch in relabeled_branches:
        paths = branch.get("joint_paths")
        if not isinstance(paths, dict) or set(paths) != set(LABELS):
            raise CommonEndpointError("Every branch needs three regime gap vectors.")
        branch["joint_paths"] = {label: deepcopy(paths[source]) for label, source in zip(LABELS, order)}
    return relabeled_no, relabeled_branches, [{"rule": "SORT_WHOLE_NO_EVENT_VECTORS_AND_TRANSPORT_BRANCH_GAPS",
                                              "original_label_for_canonical": dict(zip(LABELS, order))}]


def evaluate_common_endpoint_impact(positions, notional, no_event_paths, branches, *,
                                    as_of_utc, deadline, endpoint_rule,
                                    last_completed_session, historical_reference=None):
    """Recompute coupled YES/NO dollars at one common endpoint, exact LP bounds and gates."""
    try:
        return _evaluate(positions, notional, no_event_paths, branches, as_of_utc=as_of_utc,
                         deadline=deadline, endpoint_rule=endpoint_rule,
                         last_completed_session=last_completed_session,
                         historical_reference=historical_reference)
    except CommonEndpointError:
        raise
    except JointImpactInputError as exc:
        # Shared original-position and vector guards surface under one fail-closed type.
        raise CommonEndpointError(str(exc)) from None


def _evaluate(positions, notional, no_event_paths, branches, *, as_of_utc, deadline, endpoint_rule,
              last_completed_session, historical_reference):
    originals, nav = _original_positions(positions, notional)
    if endpoint_rule not in ENDPOINT_RULES:
        raise CommonEndpointError("Unknown common endpoint rule.")
    capture_day = _capture_day(as_of_utc)
    deadline_day = _iso_day(deadline, "original deadline")
    if capture_day >= deadline_day:
        raise CommonEndpointError("The exact event deadline is not in the future.")
    mark_day = _iso_day(last_completed_session, "last completed session")
    if mark_day >= capture_day:
        raise CommonEndpointError("The last completed session must precede the capture date.")
    try:
        endpoint = common_endpoint_for_event(deadline_day, endpoint_rule)
        if not is_trading_session(endpoint):
            raise CommonEndpointError("Common endpoint is not a regular session.")
        sessions = sessions_strictly_between(mark_day, endpoint)
    except TradingCalendarError as exc:
        raise CommonEndpointError("Calendar: " + str(exc)) from None
    if sessions < 1:
        raise CommonEndpointError("Common endpoint precedes the last completed mark.")
    signature = endpoint_signature(originals, nav, endpoint.isoformat())
    no_paths, canonical_branches, relabelings = canonicalize_regimes(originals, no_event_paths, branches)
    no_vectors = {label: _no_event_vector(no_paths[label], originals, "no_event." + label) for label in LABELS}
    no_pct = {label: fsum(row["weight"] * no_vectors[label][row["symbol"]] for row in originals) for label in LABELS}
    if not no_pct["adverse"] <= no_pct["central"] + _TOL and no_pct["central"] <= no_pct["favorable"] + _TOL:
        raise CommonEndpointError("NO-event regimes are not ordered after canonicalization.")
    if not isinstance(canonical_branches, list) or not 2 <= len(canonical_branches) <= 4:
        raise CommonEndpointError("Two to four ordered conditional-on-E branches are required.")
    ids, lows, mids, highs, summaries, residual_count = set(), [], [], [], [], 0
    for i, branch in enumerate(canonical_branches):
        if not isinstance(branch, dict):
            raise CommonEndpointError("Every branch must be an object.")
        ident = branch.get("id")
        if not isinstance(ident, str) or not 1 <= len(ident) <= 64 or ident in ids:
            raise CommonEndpointError("Branch identifiers must be unique.")
        ids.add(ident)
        if branch.get("residual") is True:
            residual_count += 1
            if i != len(canonical_branches) - 1:
                raise CommonEndpointError("Only the last branch may be residual.")
        elif branch.get("residual") is not False:
            raise CommonEndpointError("Branch residual flags must be explicit.")
        trigger = branch.get("trigger")
        if not isinstance(trigger, str) or not 8 <= len(trigger.strip()) <= 650:
            raise CommonEndpointError("Each branch needs an operational trigger.")
        a, c, b = _weight_triple(branch.get("weight_given_event"), ident)
        lows.append(a)
        mids.append(c)
        highs.append(b)
        paths = branch.get("joint_paths")
        if not isinstance(paths, dict) or set(paths) != set(LABELS):
            raise CommonEndpointError("Three coupled gap vectors are mandatory.")
        regimes = {}
        for label in LABELS:
            gap = _joint_vector(paths[label], originals, ident + "." + label + ".gap")["holding_gap_pct"]
            yes = {}
            for row in originals:
                symbol = row["symbol"]
                value = no_vectors[label][symbol] + gap[symbol]
                if not isfinite(value) or value < -100 - _TOL or value > 200 + _TOL:
                    raise CommonEndpointError("Conditional YES return violates unlevered operational guard.")
                if row["type"] == "cash" and value != 0:
                    raise CommonEndpointError("Nominal USD cash may not reprice.")
                yes[symbol] = value
            yes_pct = fsum(row["weight"] * yes[row["symbol"]] for row in originals)
            gap_pct = fsum(row["weight"] * gap[row["symbol"]] for row in originals)
            if abs((yes_pct - no_pct[label]) - gap_pct) > _TOL:
                raise CommonEndpointError("Paired full-portfolio YES-NO-gap arithmetic failed.")
            regimes[label] = {"event_yes_from_today_pct": yes_pct, "event_no_from_today_pct": no_pct[label],
                              "event_gap_pct": gap_pct, "holding_yes_pct": yes,
                              "holding_no_pct": dict(no_vectors[label]), "holding_gap_pct": gap}
        summaries.append({"id": ident, "residual": branch["residual"], "weight_given_event": [a, c, b],
                          "regimes": regimes})
    if residual_count != 1:
        raise CommonEndpointError("Exactly one final residual branch is required.")
    if abs(fsum(mids) - 1) > _TOL or fsum(lows) > 1 + _TOL or fsum(highs) < 1 - _TOL:
        raise CommonEndpointError("Infeasible conditional scenario probabilities.")

    def regime_extreme(key, label, *, maximize):
        values = [s["regimes"][label][key] for s in summaries]
        return _bounded_simplex_extreme(lows, highs, values, maximize=maximize)

    gap_center = fsum(mids[i] * s["regimes"]["central"]["event_gap_pct"] for i, s in enumerate(summaries))
    yes_center = no_pct["central"] + gap_center
    yes_bounds = [min(regime_extreme("event_yes_from_today_pct", label, maximize=False) for label in LABELS),
                  max(regime_extreme("event_yes_from_today_pct", label, maximize=True) for label in LABELS)]
    gap_bounds = [min(regime_extreme("event_gap_pct", label, maximize=False) for label in LABELS),
                  max(regime_extreme("event_gap_pct", label, maximize=True) for label in LABELS)]
    no_bounds = [min(no_pct.values()), max(no_pct.values())]
    center_yes_check = fsum(mids[i] * s["regimes"]["central"]["event_yes_from_today_pct"] for i, s in enumerate(summaries))
    if abs(center_yes_check - yes_center) > _TOL:
        raise CommonEndpointError("Central YES mixture disagrees with NO plus central gap.")
    if not yes_bounds[0] - _TOL <= yes_center <= yes_bounds[1] + _TOL:
        raise CommonEndpointError("Central conditional YES return is outside its model set.")
    if not gap_bounds[0] - _TOL <= gap_center <= gap_bounds[1] + _TOL:
        raise CommonEndpointError("Central gap is outside its model set.")
    reference = None
    if historical_reference is not None:
        reference = _accept_reference(historical_reference, signature, sessions, mark_day)
    convert = nav / 100.
    return {
        "schema": SCHEMA, "status": "MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED",
        "portfolio_signature": signature, "as_of_utc": as_of_utc, "deadline": deadline,
        "endpoint_rule": endpoint_rule,
        "common_endpoint_session_date": endpoint.isoformat(),
        "last_completed_session": mark_day.isoformat(),
        "sessions_from_last_completed_mark_to_endpoint": sessions,
        "horizon_calendar_end_is_common_across_branches": True,
        "event_yes_from_today_center_pct": yes_center,
        "event_yes_from_today_model_set_pct": yes_bounds,
        "event_yes_from_today_center_usd": convert * yes_center,
        "event_yes_from_today_model_set_usd": [convert * x for x in yes_bounds],
        "event_no_from_today_center_pct": no_pct["central"],
        "event_no_from_today_model_set_pct": no_bounds,
        "event_no_from_today_center_usd": convert * no_pct["central"],
        "event_no_from_today_model_set_usd": [convert * x for x in no_bounds],
        "paired_event_gap_center_pct": gap_center,
        "paired_event_gap_model_set_pct": gap_bounds,
        "paired_event_gap_center_usd": convert * gap_center,
        "paired_event_gap_model_set_usd": [convert * x for x in gap_bounds],
        "no_event_regime_portfolio_pct": dict(no_pct),
        "conditional_branch_summaries": summaries,
        "regime_relabelings": relabelings,
        "historical_no_event_horizon_reference": reference,
        "pricing_q_estimated": False,
        "prediction_market_Q_used_as_equity_q": False,
        "physical_event_probability_used": False,
        "event_return_caused_empirically": False,
        "forward_returns_calibrated": False,
        "future_pnl_observed_or_identified": False,
        "bounds_are_statistical_confidence_intervals": False,
        "original_etf_constituents_added_to_nav": False,
        "model_class": "JOINT_STRUCTURAL_HYPOTHETICAL_COMMON_ENDPOINT_PAIRS",
        "math_assurance": "SIGNED_EXACT_SIMPLEX_LP_PER_REGIME_OVER_COUPLED_YES_NO_PATHS_AT_ONE_ENDPOINT",
        "assurance_limit": "ASSUMPTION_CONDITIONAL_NO_CAUSAL_OR_OOS_FORECAST_VALIDATION",
    }


def _accept_reference(reference, signature, sessions, mark_day):
    """Admit a historical horizon reference only when it matches this exact scenario."""
    if not isinstance(reference, dict) or reference.get("schema") != REFERENCE_SCHEMA:
        raise CommonEndpointError("Historical horizon reference has an unknown schema.")
    if (reference.get("status") != "HISTORICAL_REFERENCE_NOT_A_FORECAST"
            or reference.get("portfolio_signature") != signature
            or reference.get("horizon_sessions") != sessions
            or reference.get("last_completed_session") != mark_day.isoformat()):
        raise CommonEndpointError("Historical horizon reference does not match this scenario.")
    keys = ("min_pct", "p05_pct", "median_pct", "mean_pct", "p95_pct", "max_pct")
    values = [reference.get(k) for k in keys]
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise CommonEndpointError("Historical horizon reference has invalid statistics.")
    lo, p05, med, _mean, p95, hi = values
    if not lo <= p05 <= med <= p95 <= hi:
        raise CommonEndpointError("Historical horizon reference statistics are unordered.")
    if type(reference.get("windows")) is not int or reference["windows"] < _MIN_REFERENCE_WINDOWS:
        raise CommonEndpointError("Historical horizon reference has too few windows.")
    if reference.get("is_forward_forecast") is not False or reference.get("event_attribution") is not False:
        raise CommonEndpointError("Historical horizon reference flags are not explicit.")
    return deepcopy(reference)


def _quantile(values, p):
    k = (len(values) - 1) * p
    low = int(k)
    return values[low] * (1 - (k - low)) + values[min(low + 1, len(values) - 1)] * (k - low)


def historical_horizon_reference(positions, notional, prices_by_symbol, *, horizon_sessions,
                                 as_of_utc, endpoint_session_date):
    """Buy-and-hold k-session price returns of the ORIGINAL weights over past windows.

    Start weights are today's weights applied at each historical window start;
    cash earns zero. This is an empirical reference distribution for the size
    of ordinary k-session moves. It is not a forecast of the NO-event path,
    not an event attribution, and prices exclude dividends.
    """
    rows, nav = _original_positions(list(positions), notional)
    if type(horizon_sessions) is not int or not 1 <= horizon_sessions <= 260:
        raise CommonEndpointError("Horizon sessions must be an integer between 1 and 260.")
    today = _capture_day(as_of_utc)
    signature = endpoint_signature(rows, nav, endpoint_session_date)
    base = {"schema": REFERENCE_SCHEMA, "portfolio_signature": signature,
            "horizon_sessions": horizon_sessions, "as_of_utc": as_of_utc,
            "weight_convention": "ORIGINAL_WEIGHTS_AT_WINDOW_START_BUY_AND_HOLD",
            "source_adjustment": "SPLIT_ADJUSTED_CLOSE_EXCLUDES_CASH_DIVIDENDS",
            "is_forward_forecast": False, "event_attribution": False,
            "windows_overlap_and_are_not_independent": True}
    active = [row for row in rows if row["type"] != "cash" and row["weight"] > 0]
    if not active:
        return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "NO_ACTIVE_MARKET_POSITIONS"}
    if not isinstance(prices_by_symbol, dict):
        return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "PRICE_SESSIONS_NOT_AVAILABLE"}
    by_symbol = {}
    for holding in active:
        series = prices_by_symbol.get(holding["symbol"])
        if not isinstance(series, list):
            return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "MISSING_PRICE_HISTORY: " + holding["symbol"]}
        dated = {}
        for record in series:
            if not isinstance(record, dict) or "date" not in record or "close" not in record:
                return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "INVALID_DAILY_BAR: " + holding["symbol"]}
            try:
                day = _iso_day(record["date"], "session")
            except CommonEndpointError:
                return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "INVALID_SESSION_DATE: " + holding["symbol"]}
            close = record["close"]
            if day >= today or type(close) not in (int, float) or not isfinite(close) or close <= 0 or day in dated:
                return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "INVALID_OR_INCOMPLETE_SESSION: " + holding["symbol"]}
            dated[day] = float(close)
        by_symbol[holding["symbol"]] = dated
    common = sorted(set.intersection(*(set(v) for v in by_symbol.values())))
    if len(common) < horizon_sessions + _MIN_REFERENCE_WINDOWS:
        return {**base, "status": "INSUFFICIENT_HISTORY", "reason": "INSUFFICIENT_SYNCHRONIZED_PRICE_SESSIONS"}
    windows = []
    for start_index in range(0, len(common) - horizon_sessions):
        start, end = common[start_index], common[start_index + horizon_sessions]
        # A missing bar in even one holding must not silently turn a
        # k-observation window into a (k+1)- or longer NYSE-session window.
        # The explicitly transcribed calendar is the authority for horizon
        # length. Outside its supported years we abstain from this OPTIONAL
        # historical size reference, without affecting the event P&L.
        try:
            if sessions_strictly_between(start, end) != horizon_sessions:
                continue
        except TradingCalendarError:
            continue
        windows.append(100 * fsum(row["weight"] * (by_symbol[row["symbol"]][end] / by_symbol[row["symbol"]][start] - 1)
                                  for row in active))
    if len(windows) < _MIN_REFERENCE_WINDOWS:
        return {**base, "status": "INSUFFICIENT_HISTORY",
                "reason": "FEWER_THAN_20_CALENDAR_VERIFIED_EXACT_HORIZON_WINDOWS",
                "candidate_synchronized_windows": len(common) - horizon_sessions,
                "verified_exact_session_windows": len(windows)}
    ordered = sorted(windows)
    return {**base, "status": "HISTORICAL_REFERENCE_NOT_A_FORECAST",
            "last_completed_session": common[-1].isoformat(),
            "history_start": common[0].isoformat(), "windows": len(windows),
            "calendar_verified_exact_session_horizon": True,
            "candidate_synchronized_windows": len(common) - horizon_sessions,
            "min_pct": ordered[0], "p05_pct": _quantile(ordered, .05), "median_pct": _quantile(ordered, .5),
            "mean_pct": fsum(ordered) / len(ordered), "p95_pct": _quantile(ordered, .95), "max_pct": ordered[-1],
            "interpretation": ("Distribution of past %d-session price returns of today's weights; a size reference for "
                               "ordinary moves over the scenario horizon, not the no-event forecast." % horizon_sessions)}


def verified_saved_common_endpoint(saved, positions, notional, no_event_paths, branches, *,
                                   as_of_utc, deadline, endpoint_rule, last_completed_session):
    """Full deterministic recheck; the saved reference is accepted only if consistent."""
    if not isinstance(saved, dict):
        return False
    try:
        expected = evaluate_common_endpoint_impact(
            positions, notional, no_event_paths, branches, as_of_utc=as_of_utc, deadline=deadline,
            endpoint_rule=endpoint_rule, last_completed_session=last_completed_session,
            historical_reference=saved.get("historical_no_event_horizon_reference"))
    except (CommonEndpointError, JointImpactInputError, KeyError, TypeError, ValueError):
        return False
    return saved == expected
