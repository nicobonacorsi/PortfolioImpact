"""Guarded production adapter for the R11 historical market-reference response.

The same deterministic ``predict_distribution`` function is used by historical
evaluation and this table. There is no refit, provider call, wall-clock read,
portfolio normalization, or substitution of a published response here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .event_state_mapping import _state_model
from .portfolio import Portfolio


SCHEMA = "castle.option1.empirical_outcome_table.r11.v1"
REQUEST_SCHEMA = "castle.option1.empirical_response_request.r11.v1"
MODEL_SCHEMA = "castle.market_reference_response.r11.v1"
FAMILIES = {"CPI": "US_INFLATION", "FOMC": "FED_POLICY"}
MODES = {"LIVE_AS_OF", "RETROSPECTIVE_REPLAY"}
WINDOWS = {
    "CPI": {"release_local": "08:30", "quote_cutoff_local": "08:20",
            "start_minute_local": "08:29", "end_minute_local": "08:44"},
    "FOMC": {"release_local": "14:00", "quote_cutoff_local": "13:50",
             "start_minute_local": "13:59", "end_minute_local": "14:14"},
}
COHORTS = {
    "CPI_EXACT_0820_REFERENCE": ("CPI", 0, "PRIMARY_EXACT_CUTOFF_REFERENCE"),
    "FOMC_PREANNOUNCEMENT_OBSERVED_REFERENCE_900S": ("FOMC", 900, "AMENDED_OBSERVED_REFERENCE"),
    "CPI_LAGGED_REFERENCE_24H": ("CPI", 86400, "EXPLORATORY_LAGGED_OBSERVED_REFERENCE"),
    "FOMC_LAGGED_REFERENCE_24H": ("FOMC", 86400, "EXPLORATORY_LAGGED_OBSERVED_REFERENCE"),
}
HORIZON_IDS = {"CPI": "CPI_0829_CLOSE_0844_CLOSE", "FOMC": "FOMC_1359_CLOSE_1414_CLOSE"}
PRODUCTION_METRICS = {
    ("US_INFLATION", "HEADLINE_CPI_U_SEASONALLY_ADJUSTED_MONTH_OVER_MONTH_SINGLE_DECIMAL_SETTLEMENT_PERCENT"):
        ("headline_CPI_SA_MoM_percentage_points", "percentage_points"),
    ("FED_POLICY", "SCHEDULED_FED_TARGET_RATE_CHANGE_BP_WITH_CANCELLATION_AS_ZERO"):
        ("scheduled_FOMC_target_rate_change", "basis_points"),
}


def _aware(value, name):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be an ISO timestamp") from exc
    if result.utcoffset() is None:
        raise ValueError(name + " must include a timezone")
    return result.astimezone(timezone.utc)


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(name + " must be a finite number")
    return result


def state_mapping_sha256():
    """Hash the shared executable mapping, not an independently copied model."""
    return hashlib.sha256(Path(__file__).with_name("event_state_mapping.py").read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode("utf-8")).hexdigest()


def _new_york_time(moment):
    """US Eastern civil-time rule in force since 2007, without a tzdata import.

    Python's Windows stdlib has no bundled IANA database. This explicit rule is
    sufficient for the supported historical sample; other years are rejected.
    It is not a general timezone implementation.
    """
    year = moment.year
    if not 2007 <= year <= 2099:
        raise ValueError("The explicit New York schedule check supports 2007-2099")
    march = datetime(year, 3, 1, 7, tzinfo=timezone.utc)
    november = datetime(year, 11, 1, 6, tzinfo=timezone.utc)
    start = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    end = november + timedelta(days=(6 - november.weekday()) % 7)
    offset = -4 if start <= moment < end else -5
    return moment.astimezone(timezone(timedelta(hours=offset)))


def _window(event):
    family = event.get("family")
    if family not in WINDOWS:
        raise ValueError("The empirical response supports only CPI or FOMC")
    if event.get("horizon_id") != HORIZON_IDS[family]:
        raise ValueError("The event must identify the exact frozen fifteen-minute return horizon")
    release = _aware(event.get("release_time_utc"), "Release time")
    cutoff = _aware(event.get("cutoff_time_utc"), "Quote cutoff")
    spec = WINDOWS[family]
    local = _new_york_time(release)
    if local.strftime("%H:%M") != spec["release_local"] or local.second or local.microsecond:
        raise ValueError("Release time is outside the frozen standard announcement window")
    if cutoff != release - timedelta(minutes=10):
        raise ValueError("Quote cutoff differs from the frozen ten-minute pre-announcement cutoff")
    quote_times = event.get("quote_timestamps_utc")
    if not isinstance(quote_times, (list, tuple)) or not quote_times:
        raise ValueError("The event must retain its actual historical quote timestamps")
    cohort = event.get("cohort")
    if cohort not in COHORTS or COHORTS[cohort][0] != family:
        raise ValueError("The event requires an explicit compatible frozen quote cohort")
    max_age = COHORTS[cohort][1]
    ages = [(cutoff - _aware(t, "Provider quote timestamp")).total_seconds() for t in quote_times]
    if any(age < 0 or age > max_age for age in ages):
        raise ValueError("Actual quote age is outside the declared empirical cohort")
    start, end = release - timedelta(minutes=1), release + timedelta(minutes=14)
    for key, expected in (("window_start_minute_utc", start), ("window_end_minute_utc", end)):
        if event.get(key) is not None and _aware(event[key], key) != expected:
            raise ValueError("Event return window differs from the frozen empirical horizon")
    return {**spec, "timezone": "America/New_York", "price_convention": "MINUTE_CLOSE_TO_MINUTE_CLOSE",
            "price_return_excludes_cash_dividends": True,
            "start_minute_utc": start.isoformat(), "end_minute_utc": end.isoformat(),
            "window_end_utc": (end + timedelta(minutes=1)).isoformat(),
            "horizon_minutes": 15, "cutoff_time_utc": cutoff.isoformat(),
            "release_time_utc": release.isoformat(), "quote_cohort": cohort,
            "quote_cohort_scope": COHORTS[cohort][2], "quote_ages_seconds": ages,
            "maximum_quote_age_seconds": max(ages),
            "quote_interpretation": ("Exact-cutoff reference snapshots." if max_age == 0 else
                "Earlier observed quote references; no claim that the book remained unchanged or executable at the cutoff.")}


def _state_identity(event, model):
    expected_hash = state_mapping_sha256()
    event_hash = event.get("state_mapping_sha256", event.get("state_model_sha256"))
    model_hash = (model.get("state_model_sha256") or model.get("state_mapping_sha256")
                  or (model.get("protocol") or {}).get("state_model_sha256"))
    if event_hash != expected_hash or model_hash != expected_hash:
        raise ValueError("Event and empirical artifact must name the current shared state-mapping hash")
    atoms = event.get("source_atoms")
    states = event.get("states")
    if not isinstance(atoms, list) or not atoms or not isinstance(states, list) or len(atoms) != len(states):
        raise ValueError("Source atoms are required to reproduce the production state mapping")
    mapped = _state_model(atoms, FAMILIES[event["family"]])
    for state, result in zip(states, mapped):
        if str(state.get("id")) != str(result["outcome_id"]):
            raise ValueError("Normalized state order or identity differs from the source atoms")
        if abs(_finite(state.get("value"), "State value") - result["value"]) > 1e-12:
            raise ValueError("Normalized representative differs from production state mapping")
        bounds = [state.get("lower"), state.get("upper")]
        if any(v is None for v in bounds):
            raise ValueError("Normalized states must retain the declared finite model interval")
        if any(abs(_finite(x, "State interval") - y) > 1e-12
               for x, y in zip(bounds, result["value_interval"])):
            raise ValueError("Normalized finite interval differs from production state mapping")
    return mapped


def _validate_hashes(hashes, label):
    values = []
    def collect(value):
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        else:
            values.append(value)
    collect(hashes)
    if not values or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", v) for v in values):
        raise ValueError(label + " must retain nonempty SHA-256 source hashes")


def _weighted_quantile(values, probabilities, level):
    cumulative = 0.0
    ordered = sorted(zip(values, probabilities))
    for value, probability in ordered:
        cumulative += probability
        if cumulative >= level - 1e-14:
            return value
    return ordered[-1][0]


def build_empirical_chain_table(portfolio, event, model, *, generated_at_utc,
                                mode="LIVE_AS_OF", evaluated_at_utc=None,
                                forecast_receipt=None, stability_artifact=None):
    """Render the exact evaluated algorithm as a separately labelled table.

    ``event`` has the normalized historical-chain fields plus ``source_atoms``,
    ``state_mapping_sha256`` and ``quote_timestamps_utc``. The source atoms are
    rerun through the shared production mapping. ``generated_at_utc`` is the
    real artifact-generation time; retrospective replay does not rewrite it.

    A retrospective receipt must identify the same event and exact fit and
    calibration IDs. An optional ``prediction_sha256`` also reproduces the
    archived engine output byte-independently through canonical JSON.
    """
    from .historical_chain import predict_distribution

    if not isinstance(portfolio, Portfolio):
        raise ValueError("A validated original-position portfolio is required")
    if not isinstance(event, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Normalized event and response artifact must be mappings")
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("Unsupported empirical response artifact schema")
    if mode not in MODES:
        raise ValueError("Unknown empirical response mode")
    generated = _aware(generated_at_utc, "Actual artifact generation time")
    if evaluated_at_utc is None:
        raise ValueError("The actual evaluation time must be explicit")
    evaluated = _aware(evaluated_at_utc, "Actual evaluation time")
    if generated > evaluated:
        raise ValueError("The artifact did not exist at the actual evaluation time")
    if model.get("status") != "READY":
        return {"schema": SCHEMA, "status": "UNAVAILABLE_EMPIRICAL_RESPONSE", "mode": mode,
                "reason": model.get("status", "MISSING_MODEL_STATUS"),
                "family": event.get("family"), "event_id": event.get("event_id"),
                "outcomes": [], "fallback_used": False, "model_artifact": deepcopy(model)}
    window = _window(event)
    cutoff = _aware(event["cutoff_time_utc"], "Quote cutoff")
    release = _aware(event["release_time_utc"], "Release time")
    if model.get("family") != event.get("family") or model.get("metric") != event.get("metric") \
            or model.get("state_unit") != event.get("state_unit"):
        raise ValueError("Empirical family, metric and unit must match the event exactly")
    if model.get("horizon_id") != event.get("horizon_id"):
        raise ValueError("Artifact horizon identity differs from the event")
    if not event.get("model_domain_id") or model.get("model_domain_id") != event.get("model_domain_id"):
        raise ValueError("Artifact and event require the same explicit model domain")
    protocol = model.get("protocol") or {}
    declared_window = (protocol.get("return_windows") or {}).get(event["family"], {})
    if any(declared_window.get(key) != WINDOWS[event["family"]][key]
           for key in ("start_minute_local", "end_minute_local")):
        raise ValueError("Artifact does not bind the same frozen fifteen-minute return horizon")
    if (protocol.get("quote_cohorts") or {}).get(event["family"]) != event.get("cohort"):
        raise ValueError("Artifact quote cohort differs from the event's declared empirical scope")
    if event.get("eligible") is not True:
        raise ValueError("Only an explicitly eligible source event can enter the empirical response")
    if _aware(model.get("fit_cutoff_time_utc"), "Model fit cutoff") > cutoff:
        raise ValueError("Model fit cutoff follows the forecast cutoff")
    if _aware(model.get("latest_training_available_time_utc"), "Last available training observation") >= cutoff:
        raise ValueError("Every fitted and calibration observation must be strictly before the forecast cutoff")
    fit_ids = model.get("fit_event_ids")
    calibration_ids = model.get("calibration_event_ids")
    if not isinstance(fit_ids, list) or not isinstance(calibration_ids, list):
        raise ValueError("Fit and calibration event identities are required")
    if len(fit_ids) < 12 or len(calibration_ids) != 12 or set(fit_ids) & set(calibration_ids):
        raise ValueError("The empirical artifact does not use disjoint fit and twelve-event calibration samples")
    if len(set(fit_ids)) != len(fit_ids) or len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError("Repeated training or calibration event identity")
    if event.get("event_id") in set(fit_ids) | set(calibration_ids):
        raise ValueError("The forecast event appears in its own fit or calibration sample")
    if mode == "LIVE_AS_OF":
        if evaluated != cutoff or generated > cutoff or evaluated >= release:
            raise ValueError("Live empirical use requires the exact source cutoff and an already available artifact")
        receipt_status = "LIVE_SOURCE_CUTOFF_CHECKED"
    else:
        if evaluated < release:
            raise ValueError("Retrospective replay must disclose an actual evaluation after the release")
        if not isinstance(forecast_receipt, Mapping):
            raise ValueError("Retrospective replay requires the archived forecast identity receipt")
        for key, value in (("event_id", event.get("event_id")), ("fit_event_ids", fit_ids),
                           ("calibration_event_ids", calibration_ids)):
            if forecast_receipt.get(key) != value:
                raise ValueError("Retrospective forecast identity does not match the archived receipt: " + key)
        receipt_status = "RETROSPECTIVE_EVENT_AND_SAMPLE_IDENTITIES_CHECKED"
    mapped = _state_identity(event, model)
    _validate_hashes(event.get("source_hashes"), "Event")
    _validate_hashes(model.get("source_hashes"), "Empirical artifact")
    needed = {p.symbol for p in portfolio.positions if p.weight != 0 and p.asset_type != "cash"}
    available = set(model.get("symbols") or [])
    missing = sorted(needed - available)
    if missing:
        return {"schema": SCHEMA, "status": "UNSUPPORTED_PORTFOLIO_SCOPE", "mode": mode,
                "family": event["family"], "event_id": event["event_id"],
                "missing_symbols": missing, "missing_weight": math.fsum(p.weight for p in portfolio.positions if p.symbol in missing),
                "outcomes": [], "fallback_used": False, "horizon": window,
                "reason": "No historical direct response for every original noncash position; no renormalization or factor substitution."}
    weights = {p.symbol: p.weight for p in portfolio.positions if p.symbol in needed}
    # Zero weights for additional model instruments preserve the model's common
    # residual sample while making the requested original portfolio explicit.
    predict_weights = {symbol: weights.get(symbol, 0.0) for symbol in model["symbols"]}
    prediction = predict_distribution(model, event, weights=predict_weights, notional_usd=portfolio.notional)
    if prediction.get("status") != "READY":
        raise ValueError("The shared prediction engine did not produce a ready distribution")
    prediction_hash = canonical_sha256(prediction)
    if mode == "RETROSPECTIVE_REPLAY" and forecast_receipt.get("prediction_sha256") is not None:
        if forecast_receipt["prediction_sha256"] != prediction_hash:
            raise ValueError("Shared prediction output differs from the archived forecast receipt")
        receipt_status = "RETROSPECTIVE_PREDICTION_HASH_AND_IDENTITIES_CHECKED"
    coefficients = model["coefficients"]
    intercept_pnl = portfolio.notional * math.fsum(weights.get(s, 0.0) * coefficients[s]["intercept"] for s in model["symbols"])
    residual_pnl = [portfolio.notional * math.fsum(weights.get(s, 0.0) * row["returns"][s] for s in model["symbols"])
                    for row in model["residual_vectors"]]
    mean_residual_pnl = math.fsum(residual_pnl) / len(residual_pnl)
    mapped_by_id = {str(row["outcome_id"]): row for row in mapped}
    outcomes = []
    for scenario in prediction["scenario_rows"]:
        state_id = str(scenario["state_id"])
        event_pnl = portfolio.notional * math.fsum(weights.get(s, 0.0) * scenario["event_component_returns"][s] for s in model["symbols"])
        conditional_pnl = portfolio.notional * math.fsum(weights.get(s, 0.0) * scenario["conditional_returns"][s] for s in model["symbols"])
        conditional_atoms = [conditional_pnl + value for value in residual_pnl]
        probability = [1.0 / len(conditional_atoms)] * len(conditional_atoms)
        position_rows = []
        for position in portfolio.positions:
            event_return = scenario["event_component_returns"].get(position.symbol, 0.0) if position.symbol in needed else 0.0
            central_return = scenario["conditional_returns"].get(position.symbol, 0.0) if position.symbol in needed else 0.0
            position_rows.append({"symbol": position.symbol, "weight": position.weight,
                                  "asset_type": position.asset_type,
                                  "event_component_pnl_usd": portfolio.notional * position.weight * event_return,
                                  "conditional_central_pnl_usd": portfolio.notional * position.weight * central_return})
        outcomes.append({"outcome_id": state_id, "q_reference": scenario["q"],
                         "state": scenario["state_value"], "state_interval": mapped_by_id[state_id]["value_interval"],
                         "state_mapping": mapped_by_id[state_id], "surprise": scenario["surprise"],
                         "event_component_pnl_usd": event_pnl, "intercept_pnl_usd": intercept_pnl,
                         "conditional_central_pnl_usd": conditional_pnl,
                         "conditional_predictive_mean_pnl_usd": conditional_pnl + mean_residual_pnl,
                         "conditional_predictive_5_95_pnl_usd": [_weighted_quantile(conditional_atoms, probability, x) for x in (.05, .95)],
                         "position_contributions": position_rows})
    weighted_event_pnl = math.fsum(row["q_reference"] * row["event_component_pnl_usd"] for row in outcomes)
    if abs(weighted_event_pnl) > max(1e-8, portfolio.notional * 1e-11):
        raise ValueError("The Q-centered event component failed its accounting identity")
    table = {"schema": SCHEMA, "status": "COMPLETE_DECLARED_EMPIRICAL_MODEL", "mode": mode,
            "model_kind": "OLS_INTERCEPT_MARKET_REFERENCE_SURPRISE_WITH_12_JOINT_RESIDUAL_VECTORS",
            "family": event["family"], "metric": event["metric"], "state_unit": event["state_unit"],
            "cohort": event["cohort"], "horizon_id": event["horizon_id"], "model_domain_id": event["model_domain_id"],
            "domain_assumptions": deepcopy(event.get("domain_assumptions", [])),
            "event_id": event["event_id"], "notional_usd": portfolio.notional,
            "horizon": window, "outcomes": outcomes, "reference_expectation": prediction["reference_expectation"],
            "summary": {**deepcopy(prediction["distribution_summary"]),
                        "q_weighted_event_component_pnl_usd": weighted_event_pnl,
                        "intercept_pnl_usd": intercept_pnl,
                        "calibration_residual_mean_pnl_usd": mean_residual_pnl,
                        "full_predictive_mean_pnl_usd": intercept_pnl + mean_residual_pnl},
            "predictive_distribution": prediction,
            "original_positions": [{"symbol": p.symbol, "weight": p.weight, "asset_type": p.asset_type} for p in portfolio.positions],
            "model_symbols": list(model["symbols"]), "fit_event_ids": fit_ids, "calibration_event_ids": calibration_ids,
            "coefficients": deepcopy(coefficients), "residual_vectors": deepcopy(model["residual_vectors"]),
            "artifact_generated_at_utc": generated.isoformat(), "actual_evaluation_time_utc": evaluated.isoformat(),
            "forecast_cutoff_time_utc": cutoff.isoformat(), "model_artifact_sha256": canonical_sha256(model),
            "prediction_sha256": prediction_hash, "state_mapping_sha256": state_mapping_sha256(),
            "source_hashes": deepcopy(event["source_hashes"]), "model_source_hashes": deepcopy(model["source_hashes"]),
            "receipt_status": receipt_status, "fallback_used": False, "physical_probability_P": None,
            "interpretation": "The event component is Q-centered. The full 15-minute return distribution separately includes the fitted intercept and twelve aligned empirical residual vectors. Q is tested as a reference distribution, not certified as physical probability.",
            "limitations": ["Retrospective scores are not prospective certification.",
                            "The fitted 15-minute horizon is not an all-day price response.",
                            "Twelve calibration residual vectors imply coarse tails and no guaranteed coverage.",
                            "Finite state tail domains are assumptions, not source support or loss bounds.",
                            "No executable price, causal response, trading profit, or universal portfolio claim."]}
    if stability_artifact is not None:
        from .historical_chain_stability import predict_stabilized_distribution
        stabilized = predict_stabilized_distribution(model, event, weights=predict_weights,
            notional_usd=portfolio.notional, stability_artifact=stability_artifact)
        if stabilized.get("status") != "READY_EXPERIMENTAL":
            raise ValueError("The shared stability engine did not produce a ready distribution")
        stability_hash = canonical_sha256(stabilized)
        stability_receipt_status = "LIVE_STABILITY_ARTIFACT_AND_TARGET_CHECKED"
        if mode == "RETROSPECTIVE_REPLAY":
            if forecast_receipt.get("stability_prediction_sha256") != stability_hash:
                raise ValueError("Shared stability prediction differs from the archived forecast receipt")
            stability_receipt_status = "RETROSPECTIVE_STABILITY_PREDICTION_HASH_CHECKED"
        table["stability"] = {"scope": "EXPERIMENTAL_PREQUENTIAL_POOL_RETROSPECTIVE_EVALUATION",
                              "artifact": deepcopy(stability_artifact), "prediction": stabilized,
                              "prediction_sha256": stability_hash, "receipt_status": stability_receipt_status,
                              "v1_result_preserved": True,
                              "interpretation": "An explicitly separate joint-distribution pool with the Q-blind historical baseline. Its portfolio-specific weight is learned only from preceding completed forecast scores; this is not an endorsement of V1 or a prospective certification."}
    return table


def build_empirical_outcome_table(portfolio, distribution, empirical_response):
    """Entry point used by ``build_outcome_table(..., empirical_response=...)``.

    The explicit request contains ``model``, normalized ``event``, generation
    and actual evaluation timestamps, and the mode/optional replay receipt.
    The distribution's production atoms must match the normalized event exactly.
    """
    if not isinstance(empirical_response, Mapping) or empirical_response.get("schema") != REQUEST_SCHEMA:
        raise ValueError("An explicit versioned empirical response request is required")
    event = deepcopy(empirical_response.get("event"))
    if not isinstance(event, dict):
        raise ValueError("Empirical response request lacks its normalized source event")
    if distribution.get("family") != FAMILIES.get(event.get("family")):
        raise ValueError("Production distribution family differs from the empirical event")
    metric = distribution.get("metric_identity")
    mapped_metric = PRODUCTION_METRICS.get((distribution["family"], metric))
    if mapped_metric is None:
        if not isinstance(metric, str) or metric != event.get("metric"):
            raise ValueError("Production settlement metric differs from the empirical event")
    elif mapped_metric != (event.get("metric"), event.get("state_unit")):
        raise ValueError("Production settlement metric or unit differs from the empirical event")
    if metric == "SCHEDULED_FED_TARGET_RATE_CHANGE_BP_WITH_CANCELLATION_AS_ZERO" \
            and event.get("model_domain_id") != "HELD_SCHEDULED_FOMC_QUARTER_POINT_ABS100_V1":
        raise ValueError("FOMC cancellation semantics require the explicit held-meeting empirical domain")
    if distribution.get("atoms") != event.get("source_atoms"):
        raise ValueError("Empirical event source atoms differ from the actual production distribution")
    source_q = [atom.get("q_reference") for atom in distribution["atoms"]]
    event_q = [state.get("q") for state in event.get("states", [])]
    if len(source_q) != len(event_q) or any(abs(_finite(a, "Source Q") - _finite(b, "Event Q")) > 1e-12 for a, b in zip(source_q, event_q)):
        raise ValueError("The empirical response must use the actual source distribution Q")
    if str(distribution.get("event_ticker", distribution.get("event_id", ""))) != str(event.get("event_id")):
        raise ValueError("Production event identity differs from the normalized empirical event")
    source_time = distribution.get("evaluated_at", distribution.get("evaluated_at_utc"))
    if _aware(source_time, "Production source evaluation time") != _aware(event.get("cutoff_time_utc"), "Event quote cutoff"):
        raise ValueError("Production Q is outside the empirically evaluated quote lead time")
    return build_empirical_chain_table(portfolio, event, empirical_response["model"],
        generated_at_utc=empirical_response["generated_at_utc"],
        mode=empirical_response.get("mode", "LIVE_AS_OF"),
        evaluated_at_utc=empirical_response.get("evaluated_at_utc"),
        forecast_receipt=empirical_response.get("forecast_receipt"),
        stability_artifact=empirical_response.get("stability_artifact"))


def build_published_chain_table(portfolio, event, model, *, generated_at_utc,
                                evaluated_at_utc, forecast_receipt):
    """Separate retrospective published-FOMC pilot; never empirical V1.

    This entry point deliberately has no live-certification mode. The original
    published-response function remains callable by application code, while the
    reviewed product feature is an exact, guarded historical replay.
    """
    from .published_chain import (MODEL_SCHEMA as PUBLISHED_MODEL_SCHEMA,
                                  predict_published_distribution,
                                  canonical_sha256 as published_sha256)
    if not isinstance(portfolio, Portfolio) or not isinstance(event, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Validated portfolio, normalized event and published-response model are required")
    if model.get("schema") != PUBLISHED_MODEL_SCHEMA or model.get("status") != "READY":
        raise ValueError("A ready separate published-response artifact is required")
    if event.get("eligible") is not True:
        raise ValueError("Published replay requires an explicitly eligible source event")
    generated = _aware(generated_at_utc, "Actual published replay generation time")
    evaluated = _aware(evaluated_at_utc, "Actual published replay evaluation time")
    if generated > evaluated or evaluated < _aware(event.get("release_time_utc"), "Release time"):
        raise ValueError("Published historical replay timestamps are inconsistent")
    window = _window(event)
    mapped = _state_identity(event, model)
    _validate_hashes(event.get("source_hashes"), "Published event")
    daily = model.get("daily_context") or {}
    _validate_hashes(daily.get("source_archive_sha256"), "Daily loading archive")
    _validate_hashes(daily.get("source_member_sha256"), "Daily loading source members")
    needed = {p.symbol for p in portfolio.positions if p.weight and p.asset_type != "cash"}
    missing = needed - set(model.get("symbols") or [])
    if missing:
        raise ValueError("Published replay has no original-position response for: " + ", ".join(sorted(missing)))
    weights = {p.symbol: p.weight for p in portfolio.positions if p.symbol in needed}
    if not weights:
        raise ValueError("Published replay requires at least one original noncash position")
    if not isinstance(forecast_receipt, Mapping) or forecast_receipt.get("event_id") != event.get("event_id"):
        raise ValueError("Published forecast event receipt mismatch")
    if forecast_receipt.get("calibration_event_ids") != model.get("calibration_event_ids"):
        raise ValueError("Published forecast calibration receipt mismatch")
    prediction = predict_published_distribution(model, event, weights=weights, notional_usd=portfolio.notional)
    digest = published_sha256(prediction)
    if forecast_receipt.get("prediction_sha256") != digest:
        raise ValueError("Published shared prediction differs from the archived forecast hash")
    return {"schema": "castle.option1.published_outcome_table.r11.v1",
            "status": "REPRODUCED_PUBLISHED_RESPONSE_PILOT", "mode": "RETROSPECTIVE_PUBLISHED_RESPONSE_PILOT",
            "event_id": event["event_id"], "family": event["family"], "metric": event["metric"],
            "state_unit": event["state_unit"], "cohort": event["cohort"], "horizon_id": event["horizon_id"],
            "horizon": window, "model_domain_id": event["model_domain_id"],
            "domain_assumptions": deepcopy(event.get("domain_assumptions", [])),
            "state_mapping": mapped, "state_mapping_sha256": state_mapping_sha256(),
            "predictive_distribution": prediction, "summary": deepcopy(prediction["distribution_summary"]),
            "original_positions": [{"symbol": p.symbol, "weight": p.weight, "asset_type": p.asset_type} for p in portfolio.positions],
            "notional_usd": portfolio.notional, "prediction_sha256": digest,
            "model_artifact_sha256": published_sha256(model), "source_hashes": deepcopy(event["source_hashes"]),
            "fixed_response": deepcopy(model["fixed_response"]), "daily_context": deepcopy(daily),
            "calibration_event_ids": list(model["calibration_event_ids"]),
            "artifact_generated_at_utc": generated.isoformat(), "actual_evaluation_time_utc": evaluated.isoformat(),
            "receipt_status": "RETROSPECTIVE_PUBLISHED_PREDICTION_HASH_AND_IDENTITIES_CHECKED",
            "physical_probability_P": None, "fallback_used": False,
            "empirical_v1_certified": False, "assumptions": deepcopy(prediction["assumptions"]),
            "interpretation": "Separate fixed published-response pilot with lagged historical Q, historical daily loadings, assumed TLT duration and twelve joint residual vectors. Source chronology and numerical replay do not certify the one-day-to-fifteen-minute response transfer."}
