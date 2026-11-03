"""Deterministic R11 market-reference probability -> return/P&L engine.

Only the standard library is used. Source admission and production state mapping
are upstream responsibilities; this module checks their normalized contracts,
chronology, probability arithmetic, and fixed fit/calibration split again.
Historical evaluation and the live adapter call the same fit/predict functions.

An event supplies event_id, family, metric, state_unit, release_time_utc,
cutoff_time_utc, states [{id,value,q,lower,upper,settlement_bounds?}], and, for
training/evaluation, available_time_utc, settled_state_id, settlement_value,
asset_returns {symbol: fractional_return}, eligible, exclusion_reasons and
source_hashes. ``lower/upper`` are the declared finite production support;
``settlement_bounds`` hold the ORIGINAL classification boundaries (including
infinity via null and explicit lower_inclusive/upper_inclusive flags).

Portfolio specifications contain id, weights, notional_usd and optionally
declaration_time_utc. No weights are normalized and no ETF look-through is added.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import math
from statistics import median
from typing import Mapping, Sequence


MODEL_SCHEMA = "castle.market_reference_response.r11.v1"
PREDICTION_SCHEMA = "castle.market_reference_predictive_distribution.r11.v1"
REPORT_SCHEMA = "castle.historical_probability_pnl_chain.r11.v1"
MIN_PRIOR = 24
CALIBRATION_COUNT = 12
MASS_TOLERANCE = 1e-9
SIGN_TOLERANCE = 1e-12
SCOPE_FIELDS = ("cohort", "model_domain_id", "state_mapping_sha256", "horizon_id")


class ChainInputError(ValueError):
    """An invalid or incompatible input, never a favorable missing score."""


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ChainInputError(f"{label} must be a finite number")
    return float(value)


def _time(value, label):
    if not isinstance(value, str):
        raise ChainInputError(f"{label} must be an ISO timestamp with timezone")
    try:
        answer = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ChainInputError(f"Invalid {label}") from exc
    if answer.tzinfo is None or answer.utcoffset() is None:
        raise ChainInputError(f"{label} must carry a timezone")
    return answer.astimezone(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _protocol(protocol):
    answer = dict(protocol or {})
    if answer.get("min_prior", MIN_PRIOR) != MIN_PRIOR or answer.get("calibration_count", CALIBRATION_COUNT) != CALIBRATION_COUNT:
        raise ChainInputError("R11 frozen protocol requires 24 prior and 12 calibration events")
    if answer.get("model_type", "OLS_INTERCEPT") != "OLS_INTERCEPT":
        raise ChainInputError("R11 frozen protocol requires OLS_INTERCEPT")
    answer.update(min_prior=MIN_PRIOR, calibration_count=CALIBRATION_COUNT, model_type="OLS_INTERCEPT")
    return answer


def _checked_states(states):
    if not isinstance(states, (list, tuple)) or not states:
        raise ChainInputError("At least one state is required")
    checked, seen = [], set()
    for state in states:
        row = dict(state)
        sid = row.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ChainInputError("State IDs must be nonempty and unique")
        seen.add(sid)
        row["value"] = _finite(row.get("value"), "state value")
        row["q"] = _finite(row.get("q"), "state q")
        if not 0 <= row["q"] <= 1:
            raise ChainInputError("State probabilities must be in [0, 1]")
        for bound in ("lower", "upper"):
            if row.get(bound) is not None:
                row[bound] = _finite(row[bound], f"state {bound}")
        if row.get("lower") is not None and row.get("upper") is not None and row["lower"] > row["upper"]:
            raise ChainInputError("Reversed finite state support")
        if row.get("lower") is not None and row["value"] < row["lower"] - 1e-12:
            raise ChainInputError("Representative below declared state support")
        if row.get("upper") is not None and row["value"] > row["upper"] + 1e-12:
            raise ChainInputError("Representative above declared state support")
        checked.append(row)
    mass = math.fsum(row["q"] for row in checked)
    if abs(mass - 1.0) > MASS_TOLERANCE:
        raise ChainInputError("State probabilities must sum to one; no normalization is performed")
    return checked


def _checked_event(event, *, outcome_required):
    row = dict(event)
    for key in ("event_id", "family", "metric", "state_unit"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ChainInputError(f"Missing {key}")
    for key in SCOPE_FIELDS:
        if row.get(key) is not None and (not isinstance(row[key], str) or not row[key]):
            raise ChainInputError(f"{key} must be a nonempty scope identifier when supplied")
    cutoff = _time(row.get("cutoff_time_utc"), "forecast cutoff")
    release = _time(row.get("release_time_utc"), "release time")
    if not cutoff < release:
        raise ChainInputError("Forecast cutoff must precede release")
    row["cutoff_time_utc"], row["release_time_utc"] = _iso(cutoff), _iso(release)
    row["states"] = _checked_states(row.get("states"))
    row["reference_expectation"] = math.fsum(s["q"] * s["value"] for s in row["states"])
    if outcome_required:
        available = _time(row.get("available_time_utc"), "training availability")
        if available < release:
            raise ChainInputError("Outcome availability cannot precede release")
        row["available_time_utc"] = _iso(available)
        matches = [s for s in row["states"] if s["id"] == row.get("settled_state_id")]
        if len(matches) != 1:
            raise ChainInputError("Settlement must identify exactly one state")
        row["settled_state_value"] = matches[0]["value"]
        row["realized_reference_surprise"] = matches[0]["value"] - row["reference_expectation"]
        if row.get("settlement_value") is not None:
            _decimal(row["settlement_value"])
            if all("settlement_bounds" in s for s in row["states"]):
                mapped = map_settlement_to_state(row["states"], row["settlement_value"])
                if mapped != row["settled_state_id"]:
                    raise ChainInputError("Settlement value disagrees with exact state classification")
        returns = row.get("asset_returns")
        if not isinstance(returns, Mapping):
            raise ChainInputError("asset_returns must map original symbols to fractional returns")
        # Missing/unavailable symbols belong in the target exclusion ledger.
        # A supplied nonfinite value is invalid, rather than a zero observation.
        row["asset_returns"] = {str(k): _finite(v, f"return for {k}") for k, v in returns.items() if v is not None}
    return row


def _decimal(value):
    if isinstance(value, bool):
        raise ChainInputError("A Boolean is not a settlement statistic")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ChainInputError("Invalid decimal settlement statistic") from exc
    if not result.is_finite():
        raise ChainInputError("Settlement statistic must be finite")
    return result


def map_settlement_to_state(states, value):
    """Map a decimal statistic using explicit original comparator boundaries.

    No fallback to finite production-tail endpoints: these are model assumptions,
    not the original contract's settlement boundaries. Ambiguity is an error.
    """
    x, matched = _decimal(value), []
    for state in states:
        bounds = state.get("settlement_bounds")
        if not isinstance(bounds, Mapping):
            raise ChainInputError("Original settlement_bounds are required for frequency baseline")
        if not isinstance(bounds.get("lower_inclusive"), bool) or not isinstance(bounds.get("upper_inclusive"), bool):
            raise ChainInputError("Comparator inclusivity must be explicit")
        lower = None if bounds.get("lower") is None else _decimal(bounds["lower"])
        upper = None if bounds.get("upper") is None else _decimal(bounds["upper"])
        if lower is not None and upper is not None and lower > upper:
            raise ChainInputError("Reversed settlement interval")
        lower_ok = lower is None or x > lower or (bounds["lower_inclusive"] and x == lower)
        upper_ok = upper is None or x < upper or (bounds["upper_inclusive"] and x == upper)
        if lower_ok and upper_ok:
            matched.append(state["id"])
    if len(matched) != 1:
        raise ChainInputError("Statistic must map to exactly one original settlement interval")
    return matched[0]


def _distribution(values, masses):
    if len(values) != len(masses) or not values:
        raise ChainInputError("Distribution values and masses must have equal positive length")
    points = [(_finite(x, "distribution value"), _finite(w, "distribution mass")) for x, w in zip(values, masses)]
    if any(w < 0 for _, w in points) or abs(math.fsum(w for _, w in points) - 1) > MASS_TOLERANCE:
        raise ChainInputError("Distribution masses must be nonnegative and sum to one")
    return sorted((x, w) for x, w in points if w > 0)


def exact_crps(values, masses, observed):
    """Exact weighted finite-distribution CRPS in O(n log n), without sampling."""
    points, y = _distribution(values, masses), _finite(observed, "observed return")
    first = math.fsum(w * abs(x - y) for x, w in points)
    cumulative_mass = cumulative_value = pair_half = 0.0
    for x, w in points:
        pair_half += w * (x * cumulative_mass - cumulative_value)
        cumulative_mass += w
        cumulative_value += w * x
    value = first - pair_half
    if value < -1e-12 * max(1.0, first):
        raise ChainInputError("Negative CRPS beyond numerical tolerance")
    return max(0.0, value)


def weighted_quantile(values, masses, probability):
    """Left-continuous generalized inverse: first x with cumulative mass >= p."""
    if not 0 <= probability <= 1:
        raise ChainInputError("Quantile probability must be in [0, 1]")
    points, cumulative = _distribution(values, masses), 0.0
    for x, w in points:
        cumulative += w
        if cumulative >= probability - 1e-15:
            return x
    return points[-1][0]


def _interval(values, masses, observed=None):
    lower = weighted_quantile(values, masses, 0.05)
    upper = weighted_quantile(values, masses, 0.95)
    answer = {"lower_05": lower, "upper_95": upper, "width_90": upper - lower}
    if observed is not None:
        y = _finite(observed, "observed return")
        answer.update(covered_90=lower <= y <= upper,
                      interval_score_90=upper - lower + 20 * max(0, lower - y) + 20 * max(0, y - upper))
    return answer


def _weights(weights, symbols):
    answer = {s: 1.0 for s in symbols} if weights is None else {str(k): _finite(v, "portfolio weight") for k, v in weights.items()}
    if not answer or any(s not in symbols for s in answer):
        raise ChainInputError("Portfolio weights require modeled original symbols")
    return answer


def _dot(weights, values):
    return math.fsum(w * values[s] for s, w in weights.items())


def fit_response_model(events, symbols, *, cutoff_time_utc, family, metric, state_unit, protocol=None,
                       cohort=None, model_domain_id=None, state_mapping_sha256=None, horizon_id=None):
    """Fit frozen expanding OLS + last-12 joint residual calibration.

    Only eligible, complete matching rows whose outcome availability STRICTLY
    precedes cutoff enter. Rejected rows are retained in model.exclusions.
    No response from the prospective event is required or accepted implicitly.
    """
    protocol, cutoff = _protocol(protocol), _time(cutoff_time_utc, "fit cutoff")
    expected_cohort = protocol.get("quote_cohorts", {}).get(family)
    if cohort is not None and expected_cohort is not None and cohort != expected_cohort:
        raise ChainInputError("Requested cohort conflicts with frozen protocol quote cohort")
    scope = dict(cohort=cohort if cohort is not None else expected_cohort,
                 model_domain_id=model_domain_id, state_mapping_sha256=state_mapping_sha256, horizon_id=horizon_id)
    matching = [e for e in events if e.get("eligible") is not False and
                (e.get("family"), e.get("metric"), e.get("state_unit")) == (family, metric, state_unit) and
                all(value is None or e.get(field) == value for field, value in scope.items())]
    for field in SCOPE_FIELDS:
        if scope[field] is None:
            values = {e.get(field) for e in matching}
            if len(values) > 1:
                raise ChainInputError(f"Mixed {field} requires separate models")
            scope[field] = next(iter(values), None)
    symbols = sorted(set(symbols))
    if not symbols or any(not isinstance(s, str) or not s for s in symbols):
        raise ChainInputError("A nonempty original symbol set is required")
    eligible, exclusions, seen = [], [], set()
    for event in events:
        eid = event.get("event_id")
        if eid in seen:
            raise ChainInputError("Duplicate event ID in fit inputs")
        seen.add(eid)
        if (event.get("family"), event.get("metric"), event.get("state_unit")) != (family, metric, state_unit):
            exclusions.append({"event_id": eid, "reasons": ["DIFFERENT_FAMILY_METRIC_OR_UNIT"]})
            continue
        if event.get("eligible") is False:
            exclusions.append({"event_id": eid, "reasons": list(event.get("exclusion_reasons") or ["SOURCE_INELIGIBLE"])})
            continue
        mismatched_scope = [field for field in SCOPE_FIELDS if event.get(field) != scope[field]]
        if mismatched_scope:
            reasons = [{"cohort": "DIFFERENT_QUOTE_COHORT", "horizon_id": "DIFFERENT_RETURN_HORIZON"}.get(field, "DIFFERENT_MODEL_DOMAIN_OR_MAPPING")
                       for field in mismatched_scope]
            exclusions.append({"event_id": eid, "reasons": sorted(set(reasons)), "scope_fields": mismatched_scope})
            continue
        row = _checked_event(event, outcome_required=True)
        if _time(row["release_time_utc"], "release") >= cutoff or _time(row["available_time_utc"], "availability") >= cutoff:
            exclusions.append({"event_id": eid, "reasons": ["NOT_AVAILABLE_BEFORE_FORECAST"]})
            continue
        missing = [s for s in symbols if s not in row["asset_returns"]]
        if missing:
            exclusions.append({"event_id": eid, "reasons": ["INCOMPLETE_TARGET_SYMBOLS"], "missing_symbols": missing})
            continue
        eligible.append(row)
    eligible.sort(key=lambda e: (_time(e["release_time_utc"], "release"), e["event_id"]))
    releases = [row["release_time_utc"] for row in eligible]
    if len(releases) != len(set(releases)):
        raise ChainInputError("Duplicate release in one model scope; select the source series upstream")
    answer = {"schema": MODEL_SCHEMA, "model_type": "OLS_INTERCEPT_JOINT_EMPIRICAL_RESIDUALS",
              "family": family, "metric": metric, "state_unit": state_unit, "symbols": symbols,
              **scope,
              "fit_cutoff_time_utc": _iso(cutoff), "protocol": protocol,
              "state_model_sha256": protocol.get("state_model_sha256"),
              "prior_complete_n": len(eligible), "exclusions": exclusions,
              "status": "INSUFFICIENT_HISTORY"}
    if len(eligible) < MIN_PRIOR:
        answer["required_prior_n"] = MIN_PRIOR
        return answer
    fit, calibration = eligible[:-CALIBRATION_COUNT], eligible[-CALIBRATION_COUNT:]
    xs = [e["realized_reference_surprise"] for e in fit]
    xmean = math.fsum(xs) / len(xs)
    centered = [x - xmean for x in xs]
    sxx = math.fsum(x * x for x in centered)
    # Fixed relative rank threshold, not estimated from forecast errors.
    rank_threshold = len(xs) * (1e-12 * max(1.0, max(abs(x) for x in xs))) ** 2
    answer.update(fit_n=len(fit), calibration_n=len(calibration),
                  fit_event_ids=[e["event_id"] for e in fit],
                  calibration_event_ids=[e["event_id"] for e in calibration],
                  latest_training_available_time_utc=_iso(max(_time(e["available_time_utc"], "availability") for e in eligible)),
                  design={"rank": 2 if sxx > rank_threshold else 1, "centered_sum_squares": sxx,
                          "rank_threshold": rank_threshold, "regressor_mean": xmean},
                  source_hashes={e["event_id"]: e.get("source_hashes", {}) for e in eligible},
                  fit_regressors=[{"event_id": e["event_id"], "reference_expectation": e["reference_expectation"],
                                   "settled_state_value": e["settled_state_value"], "regressor": e["realized_reference_surprise"]} for e in fit])
    if sxx <= rank_threshold:
        answer["status"] = "INSUFFICIENT_IDENTIFICATION"
        return answer
    coefficients = {}
    for symbol in symbols:
        ys = [e["asset_returns"][symbol] for e in fit]
        ymean = math.fsum(ys) / len(ys)
        slope = math.fsum(dx * (y - ymean) for dx, y in zip(centered, ys)) / sxx
        coefficients[symbol] = {"intercept": ymean - slope * xmean, "slope": slope}
    residuals, calibration_returns = [], []
    for event in calibration:
        z = event["realized_reference_surprise"]
        residuals.append({"event_id": event["event_id"],
                          "returns": {s: event["asset_returns"][s] - coefficients[s]["intercept"] - coefficients[s]["slope"] * z for s in symbols}})
        calibration_returns.append({"event_id": event["event_id"], "returns": {s: event["asset_returns"][s] for s in symbols},
                                    "settlement_value": event.get("settlement_value"),
                                    "available_time_utc": event["available_time_utc"]})
    answer.update(status="READY", coefficients=coefficients, residual_vectors=residuals,
                  calibration_returns=calibration_returns,
                  residual_means={s: math.fsum(e["returns"][s] for e in residuals) / CALIBRATION_COUNT for s in symbols},
                  calibration_regressors=[{"event_id": e["event_id"], "reference_expectation": e["reference_expectation"],
                                           "settled_state_value": e["settled_state_value"], "regressor": e["realized_reference_surprise"]} for e in calibration],
                  assumptions=["Reference Q is not asserted to be a calibrated physical distribution.",
                               "Response is OLS with an intercept, fitted before disjoint residual calibration.",
                               "All assets reuse the same whole residual vectors.",
                               "Additive residual invariance across states and time is assumed.",
                               "Twelve residual vectors give coarse predictive tails; coverage is not guaranteed."])
    return answer


def predict_distribution(model, event, *, weights=None, notional_usd=1.0):
    """Production and historical shared path, requiring no realized event data.

    Returns scenario_rows with event-only and full central estimates, and exact
    finite atoms with asset_returns, portfolio_return, pnl_usd and probability.
    A missing explicit weight map means one unit of each modeled symbol, NOT an
    implicitly normalized portfolio. Callers should always specify real weights.
    """
    if model.get("schema") != MODEL_SCHEMA or model.get("status") != "READY":
        raise ChainInputError("A ready versioned R11 response artifact is required")
    _protocol(model.get("protocol"))
    if model.get("model_type") != "OLS_INTERCEPT_JOINT_EMPIRICAL_RESIDUALS" or model.get("design", {}).get("rank") != 2:
        raise ChainInputError("A fully identified R11 OLS response artifact is required")
    row = _checked_event(event, outcome_required=False)
    if (row["family"], row["metric"], row["state_unit"]) != (model["family"], model["metric"], model["state_unit"]):
        raise ChainInputError("Response artifact family, metric or unit mismatch")
    if any(row.get(field) != model.get(field) for field in SCOPE_FIELDS):
        raise ChainInputError("Response artifact quote cohort, model domain, state mapping or return horizon mismatch")
    expected_cohort = model.get("protocol", {}).get("quote_cohorts", {}).get(row["family"])
    if expected_cohort is not None and (row.get("cohort") != expected_cohort or model.get("cohort") != expected_cohort):
        raise ChainInputError("Forecast quote cohort conflicts with frozen protocol")
    cutoff = _time(row["cutoff_time_utc"], "forecast cutoff")
    if _time(model["fit_cutoff_time_utc"], "artifact fit cutoff") > cutoff:
        raise ChainInputError("Model artifact was fitted after forecast cutoff")
    if _time(model["latest_training_available_time_utc"], "latest training availability") >= cutoff:
        raise ChainInputError("Training or calibration outcome was not available strictly before forecast")
    fit_ids, calibration_ids = model["fit_event_ids"], model["calibration_event_ids"]
    if (len(fit_ids) < MIN_PRIOR - CALIBRATION_COUNT or len(calibration_ids) != CALIBRATION_COUNT or
            len(set(fit_ids)) != len(fit_ids) or len(set(calibration_ids)) != len(calibration_ids) or
            model.get("fit_n") != len(fit_ids) or model.get("calibration_n") != len(calibration_ids)):
        raise ChainInputError("Artifact fit/calibration counts or unique IDs are invalid")
    if set(fit_ids) & set(calibration_ids) or row["event_id"] in set(fit_ids) | set(calibration_ids):
        raise ChainInputError("Forecast, fit and calibration event IDs must be disjoint")
    symbols = model["symbols"]
    weights, nav = _weights(weights, symbols), _finite(notional_usd, "portfolio notional")
    if nav <= 0:
        raise ChainInputError("Notional must be positive")
    residuals = model["residual_vectors"]
    if len(residuals) != CALIBRATION_COUNT or [r["event_id"] for r in residuals] != calibration_ids:
        raise ChainInputError("Artifact must retain all 12 aligned calibration residual vectors")
    for symbol in symbols:
        for key in ("intercept", "slope"):
            _finite(model["coefficients"][symbol][key], f"model {symbol} {key}")
        for residual in residuals:
            _finite(residual["returns"][symbol], f"residual for {symbol}")
    scenarios, atoms = [], []
    for state in row["states"]:
        z = state["value"] - row["reference_expectation"]
        event_component = {s: model["coefficients"][s]["slope"] * z for s in symbols}
        central = {s: model["coefficients"][s]["intercept"] + event_component[s] for s in symbols}
        central_portfolio = _dot(weights, central)
        scenarios.append({"state_id": state["id"], "q": state["q"], "state_value": state["value"], "surprise": z,
                          "event_component_returns": event_component, "conditional_returns": central,
                          "event_component_portfolio_return": _dot(weights, event_component),
                          "event_component_pnl_usd": nav * _dot(weights, event_component),
                          "conditional_portfolio_return": central_portfolio, "conditional_pnl_usd": nav * central_portfolio})
        for residual in residuals:
            returns = {s: central[s] + residual["returns"][s] for s in symbols}
            portfolio_return = _dot(weights, returns)
            atoms.append({"state_id": state["id"], "calibration_event_id": residual["event_id"],
                          "probability": state["q"] / CALIBRATION_COUNT, "asset_returns": returns,
                          "portfolio_return": portfolio_return, "pnl_usd": nav * portfolio_return})
    values, masses = [a["portfolio_return"] for a in atoms], [a["probability"] for a in atoms]
    _distribution(values, masses)
    mean = math.fsum(x * w for x, w in zip(values, masses))
    variance = math.fsum(w * (x - mean) ** 2 for x, w in zip(values, masses))
    event_mean = math.fsum(s["q"] * s["event_component_portfolio_return"] for s in scenarios)
    residual_mean = {s: math.fsum(r["returns"][s] for r in residuals) / CALIBRATION_COUNT for s in symbols}
    intercept_portfolio = _dot(weights, {s: model["coefficients"][s]["intercept"] for s in symbols})
    mean_residual_portfolio = _dot(weights, residual_mean)
    if abs(event_mean) > 1e-10 * max(1.0, max(abs(s["event_component_portfolio_return"]) for s in scenarios)):
        raise ChainInputError("Q-centered event component failed the centering identity")
    summary = {"mean_return": mean, "mean_pnl_usd": nav * mean, "standard_deviation_return": math.sqrt(max(0.0, variance)),
               "standard_deviation_pnl_usd": nav * math.sqrt(max(0.0, variance)),
               "q_weighted_event_component_return": event_mean, "intercept_portfolio_return": intercept_portfolio,
               "mean_residual_portfolio_return": mean_residual_portfolio, **_interval(values, masses)}
    summary.update(lower_05_pnl_usd=nav * summary["lower_05"], upper_95_pnl_usd=nav * summary["upper_95"])
    return {"schema": PREDICTION_SCHEMA, "status": "READY", "event_id": row["event_id"],
            "family": row["family"], "metric": row["metric"], "state_unit": row["state_unit"],
            **{field: row.get(field) for field in SCOPE_FIELDS},
            "release_time_utc": row["release_time_utc"], "cutoff_time_utc": row["cutoff_time_utc"],
            "reference_expectation": row["reference_expectation"], "weights": weights, "notional_usd": nav,
            "symbols": symbols, "scenario_rows": scenarios, "atoms": atoms, "distribution_summary": summary,
            "physical_probability_P": None, "distribution_interpretation": "Q-state reference mixture with joint empirical residuals; physical calibration is not assumed",
            "event_component_interpretation": "Q-centered surprise contribution; intercept and residual mean are separately disclosed"}


def _sign(value):
    return 0 if abs(value) <= SIGN_TOLERANCE else (1 if value > 0 else -1)


def _probability_scores(event, calibration):
    states, settled = event["states"], event["settled_state_id"]
    brier = math.fsum((s["q"] - int(s["id"] == settled)) ** 2 for s in states)
    answer = {"native_atom_brier": brier,
              "realized_atom_probability": next(s["q"] for s in states if s["id"] == settled),
              "atom_count": len(states), "baseline_status": "UNAVAILABLE_MAPPING",
              "native_atom_observations": [{"state_id": s["id"], "q": s["q"], "occurred": s["id"] == settled} for s in states]}
    try:
        prior_states = [map_settlement_to_state(states, e["settlement_value"]) for e in calibration]
    except (ChainInputError, KeyError, TypeError) as exc:
        answer["baseline_unavailable_reason"] = str(exc)
        return answer
    counts = Counter(prior_states)
    masses = {s["id"]: counts[s["id"]] / len(prior_states) for s in states}
    baseline_brier = math.fsum((masses[s["id"]] - int(s["id"] == settled)) ** 2 for s in states)
    answer.update(baseline_status="AVAILABLE_SAME_PARTITION", frequency_baseline_probabilities=masses,
                  frequency_baseline_brier=baseline_brier, paired_brier_difference=brier - baseline_brier)
    return answer


def _score_forecast(event, model, prediction, weights):
    y = _dot(weights, event["asset_returns"])
    actual_state = next(s for s in prediction["scenario_rows"] if s["state_id"] == event["settled_state_id"])
    conditional = actual_state["conditional_portfolio_return"]
    model_values = [a["portfolio_return"] for a in prediction["atoms"]]
    masses = [a["probability"] for a in prediction["atoms"]]
    baseline_values = [_dot(weights, e["returns"]) for e in model["calibration_returns"]]
    baseline_masses = [1.0 / CALIBRATION_COUNT] * CALIBRATION_COUNT
    prior_mean = math.fsum(baseline_values) / CALIBRATION_COUNT
    model_crps = exact_crps(model_values, masses, y)
    baseline_crps = exact_crps(baseline_values, baseline_masses, y)
    score = {"observed_return": y, "observed_pnl_usd": prediction["notional_usd"] * y,
             "observed_state_conditional_return": conditional,
             "prior_mean_return": prior_mean, "crps_model": model_crps, "crps_qblind": baseline_crps,
             "crps_zero": abs(y), "crps_paired_difference": model_crps - baseline_crps,
             "crps_model_bps": 10000 * model_crps, "crps_qblind_bps": 10000 * baseline_crps,
             "conditional_error": conditional - y, "conditional_squared_error": (conditional - y) ** 2,
             "conditional_absolute_error": abs(conditional - y),
             "prior_mean_squared_error": (prior_mean - y) ** 2, "prior_mean_absolute_error": abs(prior_mean - y),
             "zero_squared_error": y ** 2, "zero_absolute_error": abs(y),
             "conditional_sign_correct": _sign(conditional) == _sign(y),
             "conditional_sign_evaluable_nonzero": _sign(conditional) != 0 and _sign(y) != 0,
             "model_interval": _interval(model_values, masses, y),
             "qblind_interval": _interval(baseline_values, baseline_masses, y),
             "qblind_atoms": [{"return": v, "probability": 1.0 / CALIBRATION_COUNT,
                               "event_id": e["event_id"]} for v, e in zip(baseline_values, model["calibration_returns"])]}
    state = next(s for s in event["states"] if s["id"] == event["settled_state_id"])
    violation = None
    if event.get("settlement_value") is not None:
        value = _decimal(event["settlement_value"])
        # Finite supports are production floating-point model endpoints, unlike
        # the exact original decimal classification bounds used above.
        tolerance = Decimal("1e-12")
        violation = ((state.get("lower") is not None and value < _decimal(state["lower"]) - tolerance) or
                     (state.get("upper") is not None and value > _decimal(state["upper"]) + tolerance))
    score["realization_outside_declared_finite_state_support"] = violation
    score["finite_support_numeric_tolerance"] = 1e-12
    return score


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _aggregate(forecasts):
    scored = [f for f in forecasts if f["status"] == "EVALUATED"]
    if not scored:
        return {"n": 0, "status": "NO_ELIGIBLE_FORECASTS", "empirical_skill_supported": False,
                "descriptive_sample_skill_supported": False, "broad_empirical_endorsement": False}
    scores = [f["scores"] for f in scored]
    differences = [s["crps_paired_difference"] for s in scores]
    mean_difference = _mean(differences)
    nonzero = [s for s in scores if s["conditional_sign_evaluable_nonzero"]]
    probability = [f["probability_scores"] for f in scored]
    paired_probability = [p for p in probability if p["baseline_status"] == "AVAILABLE_SAME_PARTITION"]
    # A bin is a descriptive native-atom diagnostic, not independent replications.
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        observations = [o for p in probability for o in p["native_atom_observations"]
                        if lower <= o["q"] and (o["q"] < upper or (index == 9 and o["q"] <= upper))]
        bins.append({"lower": lower, "upper": upper, "upper_inclusive": index == 9,
                     "n": len(observations), "mean_q": _mean([o["q"] for o in observations]),
                     "observed_frequency": _mean([int(o["occurred"]) for o in observations])})
    return {"n": len(scored), "status": "DESCRIPTIVE_PILOT" if len(scored) < 12 else "RETROSPECTIVE_CHRONOLOGICAL",
            "first_event": scored[0]["event_id"], "last_event": scored[-1]["event_id"],
            "mean_crps_model": _mean([s["crps_model"] for s in scores]),
            "mean_crps_qblind": _mean([s["crps_qblind"] for s in scores]),
            "mean_crps_zero": _mean([s["crps_zero"] for s in scores]),
            "mean_crps_model_bps": 10000 * _mean([s["crps_model"] for s in scores]),
            "mean_crps_qblind_bps": 10000 * _mean([s["crps_qblind"] for s in scores]),
            "paired_crps_difference_mean": mean_difference,
            "paired_crps_difference_median": median(differences),
            "fraction_model_crps_better": _mean([int(d < 0) for d in differences]),
            "empirical_skill_supported": mean_difference < 0,
            "descriptive_sample_skill_supported": mean_difference < 0,
            "broad_empirical_endorsement": False,
            "skill_claim_scope": "Lower average CRPS in this named retrospective sample only" if mean_difference < 0 else "Incremental CRPS skill over Q-blind baseline is not supported in this sample",
            "conditional_rmse": math.sqrt(_mean([s["conditional_squared_error"] for s in scores])),
            "conditional_mae": _mean([s["conditional_absolute_error"] for s in scores]),
            "prior_mean_rmse": math.sqrt(_mean([s["prior_mean_squared_error"] for s in scores])),
            "prior_mean_mae": _mean([s["prior_mean_absolute_error"] for s in scores]),
            "zero_rmse": math.sqrt(_mean([s["zero_squared_error"] for s in scores])),
            "zero_mae": _mean([s["zero_absolute_error"] for s in scores]),
            "conditional_sign_accuracy_all": _mean([int(s["conditional_sign_correct"]) for s in scores]),
            "conditional_sign_accuracy_nonzero": _mean([int(s["conditional_sign_correct"]) for s in nonzero]),
            "conditional_sign_nonzero_n": len(nonzero),
            "sign_rule": "abs(value)<=1e-12 is zero; exact sign match includes zero/zero; nonzero accuracy excludes either zero",
            "model_coverage_90": _mean([int(s["model_interval"]["covered_90"]) for s in scores]),
            "model_mean_width_90": _mean([s["model_interval"]["width_90"] for s in scores]),
            "model_mean_interval_score_90": _mean([s["model_interval"]["interval_score_90"] for s in scores]),
            "qblind_coverage_90": _mean([int(s["qblind_interval"]["covered_90"]) for s in scores]),
            "qblind_mean_width_90": _mean([s["qblind_interval"]["width_90"] for s in scores]),
            "qblind_mean_interval_score_90": _mean([s["qblind_interval"]["interval_score_90"] for s in scores]),
            "finite_support_violation_n": sum(s["realization_outside_declared_finite_state_support"] is True for s in scores),
            "probability": {"mean_native_atom_brier": _mean([p["native_atom_brier"] for p in probability]),
                            "paired_frequency_baseline_n": len(paired_probability),
                            "mean_paired_brier_difference": _mean([p["paired_brier_difference"] for p in paired_probability]),
                            "native_atom_calibration_bins": bins,
                            "interpretation": "Changing native partitions and dependent atoms limit aggregate Brier/calibration interpretation"}}


def evaluate_historical_chain(events, portfolio_specs, protocol=None):
    """Chronological forecasts for every symbol and every declared portfolio.

    Optional protocol.asof_time_utc creates latest_model_artifacts for production
    using only information available before that timestamp. Source-ineligible,
    missing-symbol, warmup and unidentified rows remain in explicit ledgers.
    """
    protocol = _protocol(protocol)
    checked, source_exclusions, seen = [], [], set()
    for event in events:
        eid = event.get("event_id")
        if eid in seen:
            raise ChainInputError("Duplicate historical event ID")
        seen.add(eid)
        if event.get("eligible") is False:
            source_exclusions.append({"event_id": eid, "family": event.get("family"),
                                      "reasons": list(event.get("exclusion_reasons") or ["SOURCE_INELIGIBLE"]),
                                      "source_receipt": dict(event)})
            continue
        try:
            checked.append(_checked_event(event, outcome_required=True))
        except (ChainInputError, KeyError, TypeError) as exc:
            source_exclusions.append({"event_id": eid, "family": event.get("family"), "reasons": ["INVALID_NORMALIZED_EVENT", str(exc)],
                                      "source_receipt": dict(event)})
    checked.sort(key=lambda e: (_time(e["release_time_utc"], "release"), e["event_id"]))
    def group_key(event):
        return (event["family"], event["metric"], event["state_unit"], *(event.get(field) for field in SCOPE_FIELDS))

    groups = sorted(set(group_key(e) for e in checked), key=lambda key: tuple("" if value is None else value for value in key))
    all_symbols = sorted(set(s for event in checked for s in event["asset_returns"]))
    targets = [{"id": f"symbol:{s}", "kind": "symbol", "weights": {s: 1.0}, "notional_usd": 100000.0,
                "primary_diagnostic": s == "SPY"} for s in all_symbols]
    target_ids = set(t["id"] for t in targets)
    for spec in portfolio_specs:
        target = dict(spec)
        if not isinstance(target.get("id"), str) or not target["id"] or target["id"] in target_ids:
            raise ChainInputError("Portfolio IDs must be nonempty and unique across targets")
        target_ids.add(target["id"])
        if not isinstance(target.get("weights"), Mapping) or not target["weights"]:
            raise ChainInputError("Each portfolio must retain explicit weights")
        target["weights"] = {str(s): _finite(w, "portfolio weight") for s, w in target["weights"].items() if w != 0}
        if not target["weights"]:
            raise ChainInputError("At least one nonzero portfolio position is required")
        target["notional_usd"] = _finite(target.get("notional_usd"), "notional")
        if target["notional_usd"] <= 0:
            raise ChainInputError("Portfolio notional must be positive")
        target.update(kind="portfolio", primary_diagnostic=False)
        targets.append(target)
    results, latest = [], []
    for family, metric, unit, cohort, domain_id, mapping_sha, horizon_id in groups:
        family_events = [e for e in checked if group_key(e) == (family, metric, unit, cohort, domain_id, mapping_sha, horizon_id)]
        scope = dict(cohort=cohort, model_domain_id=domain_id, state_mapping_sha256=mapping_sha, horizon_id=horizon_id)
        releases = [row["release_time_utc"] for row in family_events]
        if len(releases) != len(set(releases)):
            raise ChainInputError("Duplicate release in one model scope; select the source series upstream")
        for target in targets:
            symbols, forecasts = sorted(target["weights"]), []
            for event in family_events:
                forecast = {"event_id": event["event_id"], "family": family, "release_time_utc": event["release_time_utc"],
                            "cutoff_time_utc": event["cutoff_time_utc"], "source_hashes": event.get("source_hashes", {})}
                missing = [s for s in symbols if s not in event["asset_returns"]]
                if missing:
                    forecast.update(status="INCOMPLETE_TARGET_SYMBOLS", missing_symbols=missing)
                    forecasts.append(forecast)
                    continue
                # Future rows do not even reach the fitter for this forecast.
                past = [e for e in family_events if _time(e["release_time_utc"], "past release") < _time(event["cutoff_time_utc"], "cutoff")]
                model = fit_response_model(past, symbols, cutoff_time_utc=event["cutoff_time_utc"],
                                           family=family, metric=metric, state_unit=unit, protocol=protocol, **scope)
                forecast["model"] = model
                if model["status"] != "READY":
                    forecast["status"] = model["status"]
                    forecasts.append(forecast)
                    continue
                prediction = predict_distribution(model, event, weights=target["weights"], notional_usd=target["notional_usd"])
                forecast.update(status="EVALUATED", prediction=prediction,
                                scores=_score_forecast(event, model, prediction, target["weights"]),
                                probability_scores=_probability_scores(event, model["calibration_returns"]),
                                settled_state_id=event["settled_state_id"], settlement_value=event.get("settlement_value"))
                forecasts.append(forecast)
            results.append({"target": target, "family": family, "metric": metric, "state_unit": unit, **scope,
                            "candidate_complete_source_n": len(family_events), "forecasts": forecasts,
                            "status_counts": dict(Counter(f["status"] for f in forecasts)),
                            "summary": _aggregate(forecasts),
                            "summary_2025_2026": _aggregate([f for f in forecasts if "2025-01-01" <= f["release_time_utc"] < "2027-01-01"])})
            if protocol.get("asof_time_utc"):
                model = fit_response_model(family_events, symbols, cutoff_time_utc=protocol["asof_time_utc"],
                                           family=family, metric=metric, state_unit=unit, protocol=protocol, **scope)
                latest.append({"target_id": target["id"], "weights": target["weights"], "notional_usd": target["notional_usd"], "model": model})
    n_scored = sum(result["summary"]["n"] for result in results)
    return {"schema": REPORT_SCHEMA, "protocol": protocol,
            "evaluation_type": "RETROSPECTIVE_CHRONOLOGICAL_NOT_BLIND",
            "input_event_n": len(events), "normalized_eligible_event_n": len(checked),
            "source_exclusions": source_exclusions, "target_results": results, "latest_model_artifacts": latest,
            "conclusions": {"pipeline_arithmetic_checked": True, "historical_chain_evaluated": n_scored > 0,
                            "source_admission_verified_by_this_module": False,
                            "universal_empirical_skill_claim": False,
                            "note": "Skill is reported per target; overlapping targets are not independent evidence. Source admission requires upstream provenance receipts."},
            "event_receipts": [dict(event) for event in checked],
            "limitations": ["Historical outcomes were previously available; this is not a blind or prospective test.",
                            "Source-rule vintage, finite tails, quote feasibility and minute-price admission are separate upstream checks.",
                            "A lower sample CRPS does not establish trading alpha, causal identification or calibrated physical probabilities.",
                            "No model, fit window, symbol or portfolio is selected using these forecast scores."]}
