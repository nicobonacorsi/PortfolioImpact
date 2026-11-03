"""Experimental prequential joint-distribution stability layer for frozen R11.

The v1 engine is unchanged. This layer was designed after v1's unfavorable CPI
scores were seen. It is retrospective development evidence, never a blind test.
Exactly one scale-free projected AdaGrad update, frozen in the cited protocol,
learns a target-bound mixture of v1 and its own Q-blind calibration baseline.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

from . import historical_chain as chain


ARTIFACT_SCHEMA = "castle.prequential_stability_artifact.r11.v1"
PREDICTION_SCHEMA = "castle.prequential_stability_prediction.r11.v1"
PROTOCOL_SHA256 = "9f37408e2a61245dddde8030258672271b7c38839d2dc2188d8e05d9db1dd47a7"
ALGORITHM = "PROJECTED_SCALE_FREE_ADAGRAD_CRPS_JOINT_CDF_POOL"
EVALUATION_LABEL = "RETROSPECTIVE_EXPERIMENTAL_POST_V1_DIAGNOSTIC"
_SCOPE = ("family", "metric", "state_unit", *chain.SCOPE_FIELDS)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _code_hashes():
    return {"v1_engine_sha256": hashlib.sha256(Path(chain.__file__).read_bytes()).hexdigest(),
            "stability_engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _binding(source, weights):
    normalized = {str(symbol): chain._finite(weight, "binding weight") for symbol, weight in sorted(weights.items())}
    if not normalized:
        raise chain.ChainInputError("A stability learner requires exact original portfolio weights")
    return {**{key: source.get(key) for key in _SCOPE}, "weights": normalized}


def _same_number(actual, expected):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual):
        return False
    return abs(float(actual) - expected) <= max(math.ulp(0.0), 64 * sys.float_info.epsilon * (abs(actual) + abs(expected)))


def _cancel(terms):
    raw = math.fsum(terms)
    tolerance = max(math.ulp(0.0), 64 * sys.float_info.epsilon * math.fsum(abs(term) for term in terms))
    return raw, 0.0 if abs(raw) <= tolerance else raw, tolerance


def _absolute_pair(values_a, masses_a, values_b, masses_b):
    a, b = chain._distribution(values_a, masses_a), chain._distribution(values_b, masses_b)
    return math.fsum(pa * pb * abs(xa - xb) for xa, pa in a for xb, pb in b)


def crps_pool_gradient(f_values, f_masses, g_values, g_masses, observed, weight):
    """Exact CRPS quadratic and gradient; no data-selected optimization settings."""
    f, g = chain._distribution(f_values, f_masses), chain._distribution(g_values, g_masses)
    y, weight = chain._finite(observed, "observed return"), chain._finite(weight, "event-model weight")
    if not 0 <= weight <= 1:
        raise chain.ChainInputError("Event-model weight must be in [0, 1]")
    af, ag = math.fsum(p * abs(x - y) for x, p in f), math.fsum(p * abs(x - y) for x, p in g)
    bf = _absolute_pair(f_values, f_masses, f_values, f_masses)
    bg = _absolute_pair(g_values, g_masses, g_values, g_masses)
    bfg = _absolute_pair(f_values, f_masses, g_values, g_masses)
    d_raw, d, d_tolerance = _cancel([2 * bfg, -bf, -bg])
    if d < 0:
        raise chain.ChainInputError("Negative CDF-distance curvature beyond floating-point roundoff")
    h_raw, h, h_tolerance = _cancel([af, -ag, bg, -bfg])
    gradient_raw, gradient, gradient_tolerance = _cancel([h, weight * d])
    return {"A_F": af, "A_G": ag, "B_F": bf, "B_G": bg, "B_FG": bfg,
            "D_raw": d_raw, "D": d, "D_tolerance": d_tolerance,
            "h_raw": h_raw, "h": h, "h_tolerance": h_tolerance,
            "gradient_raw": gradient_raw, "gradient": gradient, "gradient_tolerance": gradient_tolerance,
            "crps_F": af - bf / 2, "crps_G": ag - bg / 2,
            "crps_pool_quadratic": ag - bg / 2 + weight * h + weight * weight * d / 2}


def _new_state():
    return {"weight": 0.0, "sum_squared_gradients": 0.0, "ledger": []}


def _apply_feedback(state, feedback):
    """Apply one completed outcome in availability order, at its original weight."""
    record = deepcopy(feedback)
    eid = record["event_id"]
    if eid in {row["event_id"] for row in state["ledger"]}:
        raise chain.ChainInputError("A forecast outcome cannot update a learner twice")
    available = chain._time(record["available_time_utc"], "feedback availability")
    cutoff = chain._time(record["cutoff_time_utc"], "original prediction cutoff")
    if available <= cutoff:
        raise chain.ChainInputError("Feedback cannot be available before its original prediction")
    if state["ledger"]:
        last = state["ledger"][-1]
        if (available, eid) < (chain._time(last["available_time_utc"], "last availability"), last["event_id"]):
            raise chain.ChainInputError("Feedback must be processed by availability then event ID")
    known_at_prediction = [row for row in state["ledger"] if chain._time(row["available_time_utc"], "earlier availability") < cutoff]
    expected_weight = known_at_prediction[-1]["weight_after"] if known_at_prediction else 0.0
    expected_ids = [row["event_id"] for row in known_at_prediction]
    if not _same_number(record["forecast_weight"], expected_weight) or record["prior_completed_event_ids"] != expected_ids:
        raise chain.ChainInputError("Stored forecast weight or history contains unavailable feedback")
    terms = crps_pool_gradient(record["F_values"], record["F_masses"], record["G_values"], record["G_masses"],
                               record["observed_return"], record["forecast_weight"])
    before_weight, before_s = state["weight"], state["sum_squared_gradients"]
    after_s = before_s + terms["gradient"] ** 2
    if not math.isfinite(after_s):
        raise chain.ChainInputError("Nonfinite accumulated gradient magnitude")
    after_weight = max(0.0, min(1.0, before_weight - terms["gradient"] / math.sqrt(after_s))) if after_s > 0 else before_weight
    record.update(terms=terms, weight_before=before_weight, weight_after=after_weight,
                  sum_squared_gradients_before=before_s, sum_squared_gradients_after=after_s,
                  delayed_feedback=len(known_at_prediction) != len(state["ledger"]),
                  cumulative_completed_event_ids=[row["event_id"] for row in state["ledger"]] + [eid])
    state["weight"], state["sum_squared_gradients"] = after_weight, after_s
    state["ledger"].append(record)
    return record


def _artifact(state, binding, target_id, asof_time_utc, *, source_report_protocol=None):
    asof = chain._time(asof_time_utc, "stability artifact asof")
    if any(chain._time(record["available_time_utc"], "feedback availability") >= asof for record in state["ledger"]):
        raise chain.ChainInputError("Artifact includes feedback unavailable strictly before asof")
    result = {"schema": ARTIFACT_SCHEMA, "algorithm": ALGORITHM, "evaluation_label": EVALUATION_LABEL,
              "protocol_sha256": PROTOCOL_SHA256, "target_id": target_id,
              "binding": deepcopy(binding), "binding_sha256": _digest(binding),
              "asof_time_utc": chain._iso(asof), "event_model_weight": state["weight"],
              "baseline_weight": 1 - state["weight"], "sum_squared_gradients": state["sum_squared_gradients"],
              "completed_feedback_n": len(state["ledger"]),
              "completed_event_ids": [record["event_id"] for record in state["ledger"]],
              "latest_processed_available_time_utc": state["ledger"][-1]["available_time_utc"] if state["ledger"] else None,
              "feedback_ledger": deepcopy(state["ledger"]), "source_report_protocol": deepcopy(source_report_protocol or {}),
              "initialization": "BASELINE_FIRST_WEIGHT_ZERO", "nav_invariant_learning": True,
              "negative_v1_results_retained": True, "physical_probability_P": None, **_code_hashes()}
    result["artifact_sha256"] = _digest(result)
    return result


def _verify_artifact(artifact, model, event, weights):
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("algorithm") != ALGORITHM or artifact.get("protocol_sha256") != PROTOCOL_SHA256:
        raise chain.ChainInputError("A matching frozen stability artifact is required")
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _digest(unsigned):
        raise chain.ChainInputError("Stability artifact checksum mismatch")
    binding = _binding(model, weights)
    if artifact.get("binding") != binding or artifact.get("binding_sha256") != _digest(binding):
        raise chain.ChainInputError("Stability learner belongs to different original weights or model scope")
    if _binding(event, weights) != binding:
        raise chain.ChainInputError("Future event and stability learner scope differ")
    if any(artifact.get(key) != value for key, value in _code_hashes().items()):
        raise chain.ChainInputError("Stability artifact requires its recorded engine implementations")
    cutoff = chain._time(event["cutoff_time_utc"], "future cutoff")
    asof = chain._time(artifact["asof_time_utc"], "artifact asof")
    if asof > cutoff:
        raise chain.ChainInputError("Stability artifact was built after forecast cutoff")
    state = _new_state()
    for record in artifact["feedback_ledger"]:
        if record.get("binding_sha256") != artifact["binding_sha256"]:
            raise chain.ChainInputError("Feedback belongs to another portfolio or scope")
        available = chain._time(record["available_time_utc"], "record availability")
        if available >= cutoff or available >= asof:
            raise chain.ChainInputError("Current or future feedback cannot enter a stability prediction")
        if record["event_id"] == event["event_id"]:
            raise chain.ChainInputError("The predicted event already updated this learner")
        rebuilt = _apply_feedback(state, record)
        for key in ("weight_before", "weight_after", "sum_squared_gradients_before", "sum_squared_gradients_after"):
            if not _same_number(record.get(key), rebuilt[key]):
                raise chain.ChainInputError("Stability feedback state does not replay exactly")
        if record.get("delayed_feedback") != rebuilt["delayed_feedback"] or record.get("cumulative_completed_event_ids") != rebuilt["cumulative_completed_event_ids"]:
            raise chain.ChainInputError("Stability feedback chronology does not replay")
        if set(record.get("terms", {})) != set(rebuilt["terms"]) or any(not _same_number(record["terms"][key], value) for key, value in rebuilt["terms"].items()):
            raise chain.ChainInputError("Stability feedback score gradient does not replay")
    if (not _same_number(artifact["event_model_weight"], state["weight"]) or
            not _same_number(artifact["baseline_weight"], 1 - state["weight"]) or
            not _same_number(artifact["sum_squared_gradients"], state["sum_squared_gradients"]) or
            artifact["completed_feedback_n"] != len(state["ledger"]) or
            artifact["completed_event_ids"] != [record["event_id"] for record in state["ledger"]]):
        raise chain.ChainInputError("Stability artifact summary differs from its complete score ledger")
    latest = state["ledger"][-1]["available_time_utc"] if state["ledger"] else None
    if artifact.get("latest_processed_available_time_utc") != latest:
        raise chain.ChainInputError("Stability artifact latest-availability receipt differs")
    return state["weight"]


def predict_stabilized_distribution(model, event, *, weights, notional_usd, stability_artifact):
    """Shared live and historical joint CDF pool, with strict target/asof binding.

    The new conditional_returns are predictive conditional MEANS, including v1's
    raw residual mean. Original regression-central values are retained separately.
    """
    weight = _verify_artifact(stability_artifact, model, event, weights)
    original = chain.predict_distribution(model, event, weights=weights, notional_usd=notional_usd)
    symbols, nav = original["symbols"], original["notional_usd"]
    calibration = model["calibration_returns"]
    if len(calibration) != chain.CALIBRATION_COUNT:
        raise chain.ChainInputError("Stability baseline requires the same 12 joint calibration vectors")
    baseline_mean = {symbol: math.fsum(row["returns"][symbol] for row in calibration) / len(calibration) for symbol in symbols}
    residual_mean = {symbol: math.fsum(row["returns"][symbol] for row in model["residual_vectors"]) / len(calibration) for symbol in symbols}
    scenarios, atoms = [], []
    for scenario in original["scenario_rows"]:
        predictive_v1_mean = {symbol: scenario["conditional_returns"][symbol] + residual_mean[symbol] for symbol in symbols}
        conditional_mean = {symbol: (1 - weight) * baseline_mean[symbol] + weight * predictive_v1_mean[symbol] for symbol in symbols}
        contribution = {symbol: weight * scenario["event_component_returns"][symbol] for symbol in symbols}
        conditional_portfolio = chain._dot(weights, conditional_mean)
        scenarios.append({"state_id": scenario["state_id"], "q": scenario["q"], "state_value": scenario["state_value"],
                          "surprise": scenario["surprise"], "conditional_returns": conditional_mean,
                          "conditional_predictive_mean_returns": conditional_mean,
                          "conditional_portfolio_return": conditional_portfolio, "conditional_pnl_usd": nav * conditional_portfolio,
                          "event_component_returns": contribution, "event_component_portfolio_return": chain._dot(weights, contribution),
                          "event_component_pnl_usd": nav * chain._dot(weights, contribution),
                          "v1_conditional_regression_returns": scenario["conditional_returns"],
                          "v1_conditional_predictive_mean_returns": predictive_v1_mean,
                          "qblind_mean_returns": baseline_mean,
                          "conditional_estimate_kind": "FULL_PREDICTIVE_CONDITIONAL_MEAN_INCLUDING_RAW_RESIDUAL_MEAN"})
        for atom in original["atoms"]:
            if atom["state_id"] == scenario["state_id"]:
                atoms.append({**atom, "probability": weight * atom["probability"], "expert": "V1_EVENT_RESPONSE"})
        for prior in calibration:
            returns = {symbol: prior["returns"][symbol] for symbol in symbols}
            value = chain._dot(weights, returns)
            atoms.append({"state_id": scenario["state_id"], "calibration_event_id": prior["event_id"],
                          "probability": (1 - weight) * scenario["q"] / len(calibration),
                          "asset_returns": returns, "portfolio_return": value, "pnl_usd": nav * value,
                          "expert": "Q_BLIND_JOINT_BASELINE"})
    values, masses = [atom["portfolio_return"] for atom in atoms], [atom["probability"] for atom in atoms]
    chain._distribution(values, masses)
    mean = math.fsum(value * mass for value, mass in zip(values, masses))
    variance = math.fsum(mass * (value - mean) ** 2 for value, mass in zip(values, masses))
    interval = chain._interval(values, masses)
    for scenario in scenarios:
        state_atoms = [atom for atom in atoms if atom["state_id"] == scenario["state_id"]]
        state_mass = math.fsum(atom["probability"] for atom in state_atoms)
        if not _same_number(state_mass, scenario["q"]):
            raise chain.ChainInputError("Pooling changed the market-reference Q marginal")
    summary = {"mean_return": mean, "mean_pnl_usd": nav * mean,
               "standard_deviation_return": math.sqrt(max(0, variance)), "standard_deviation_pnl_usd": nav * math.sqrt(max(0, variance)),
               "q_weighted_event_component_return": math.fsum(s["q"] * s["event_component_portfolio_return"] for s in scenarios),
               "qblind_mean_contribution_return": (1 - weight) * chain._dot(weights, baseline_mean),
               "v1_intercept_contribution_return": weight * chain._dot(weights, {s: model["coefficients"][s]["intercept"] for s in symbols}),
               "v1_residual_mean_contribution_return": weight * chain._dot(weights, residual_mean),
               **interval, "lower_05_pnl_usd": nav * interval["lower_05"], "upper_95_pnl_usd": nav * interval["upper_95"]}
    result = {"schema": PREDICTION_SCHEMA, "status": "READY_EXPERIMENTAL", "algorithm": ALGORITHM,
              "evaluation_label": EVALUATION_LABEL, "event_id": original["event_id"],
              **{key: original.get(key) for key in _SCOPE}, "weights": dict(weights), "notional_usd": nav,
              "release_time_utc": original["release_time_utc"], "cutoff_time_utc": original["cutoff_time_utc"],
              "reference_expectation": original["reference_expectation"], "symbols": symbols,
              "event_model_weight": weight, "qblind_baseline_weight": 1 - weight,
              "stability_artifact_sha256": stability_artifact["artifact_sha256"], "binding_sha256": stability_artifact["binding_sha256"],
              "prior_completed_event_ids": list(stability_artifact["completed_event_ids"]),
              "v1_prediction_sha256": _digest(original), "v1_model_sha256": _digest(model),
              "qblind_joint_vectors_sha256": _digest(calibration), "scenario_rows": scenarios, "atoms": atoms,
              "distribution_summary": summary, "physical_probability_P": None,
              "assumptions": ["Q-blind calibration return vectors are independent of event state in the baseline branch.",
                              "Each branch preserves complete original joint asset vectors.",
                              "Q remains a market-reference marginal, not an assumed calibrated physical probability.",
                              "Attenuation does not erase the original unattenuated scenario stress table.",
                              "This mechanism was developed after v1 CPI failures; new scores are retrospective experimental evidence."]}
    return result


def _feedback(forecast, event, target, prediction, artifact):
    original, model = forecast["prediction"], forecast["model"]
    weights = target["weights"]
    observed = chain._dot(weights, event["asset_returns"])
    if not _same_number(observed, forecast["scores"]["observed_return"]):
        raise chain.ChainInputError("Historical observed return disagrees with original-position aggregation")
    baseline = [chain._dot(weights, row["returns"]) for row in model["calibration_returns"]]
    return {"event_id": forecast["event_id"], "cutoff_time_utc": forecast["cutoff_time_utc"],
            "available_time_utc": event["available_time_utc"], "forecast_weight": artifact["event_model_weight"],
            "prior_completed_event_ids": list(artifact["completed_event_ids"]), "binding_sha256": artifact["binding_sha256"],
            "F_values": [atom["portfolio_return"] for atom in original["atoms"]],
            "F_masses": [atom["probability"] for atom in original["atoms"]],
            "G_values": baseline, "G_masses": [1 / len(baseline)] * len(baseline), "observed_return": observed,
            "v1_prediction_sha256": _digest(original), "stability_prediction_sha256": _digest(prediction),
            "v1_model_sha256": _digest(model), "source_event_receipt_sha256": _digest(event),
            "source_hashes": deepcopy(event.get("source_hashes", {})),
            "fit_event_ids": list(model["fit_event_ids"]), "calibration_event_ids": list(model["calibration_event_ids"])}


def _scores(prediction, event, original_scores):
    values, masses = [atom["portfolio_return"] for atom in prediction["atoms"]], [atom["probability"] for atom in prediction["atoms"]]
    y = original_scores["observed_return"]
    conditional = next(row["conditional_portfolio_return"] for row in prediction["scenario_rows"] if row["state_id"] == event["settled_state_id"])
    crps = chain.exact_crps(values, masses, y)
    baseline_raw, baseline_difference, _ = _cancel([crps, -original_scores["crps_qblind"]])
    v1_raw, v1_difference, _ = _cancel([crps, -original_scores["crps_model"]])
    return {"observed_return": y, "observed_pnl_usd": prediction["notional_usd"] * y,
            "crps_stability": crps, "crps_stability_bps": crps * 10000,
            "crps_v1": original_scores["crps_model"], "crps_qblind": original_scores["crps_qblind"],
            "crps_zero": original_scores["crps_zero"],
            "paired_crps_difference_vs_qblind": baseline_difference,
            "paired_crps_difference_vs_qblind_raw": baseline_raw,
            "paired_crps_difference_vs_v1": v1_difference,
            "paired_crps_difference_vs_v1_raw": v1_raw,
            "conditional_predictive_mean_return": conditional, "conditional_mean_error": conditional - y,
            "conditional_mean_squared_error": (conditional - y) ** 2, "conditional_mean_absolute_error": abs(conditional - y),
            "v1_regression_central_squared_error": original_scores["conditional_squared_error"],
            "prior_mean_squared_error": original_scores["prior_mean_squared_error"],
            "stability_interval": chain._interval(values, masses, y),
            "v1_interval": deepcopy(original_scores["model_interval"]), "qblind_interval": deepcopy(original_scores["qblind_interval"])}


def _summary(forecasts):
    scored = [row for row in forecasts if "stability_scores" in row]
    if not scored:
        return {"n": 0, "status": "NO_ELIGIBLE_V1_FORECASTS", "descriptive_sample_skill_supported": False,
                "broad_empirical_endorsement": False, "evaluation_label": EVALUATION_LABEL}
    scores = [row["stability_scores"] for row in scored]
    difference = [row["paired_crps_difference_vs_qblind"] for row in scores]
    from statistics import median
    return {"n": len(scored), "status": "EXPERIMENTAL_PILOT" if len(scored) < 12 else "RETROSPECTIVE_EXPERIMENTAL",
            "evaluation_label": EVALUATION_LABEL, "first_event": scored[0]["event_id"], "last_event": scored[-1]["event_id"],
            "mean_crps_stability_bps": 10000 * chain._mean([row["crps_stability"] for row in scores]),
            "mean_crps_v1_bps": 10000 * chain._mean([row["crps_v1"] for row in scores]),
            "mean_crps_qblind_bps": 10000 * chain._mean([row["crps_qblind"] for row in scores]),
            "mean_crps_zero_bps": 10000 * chain._mean([row["crps_zero"] for row in scores]),
            "paired_crps_difference_vs_qblind_mean": chain._mean(difference),
            "paired_crps_difference_vs_qblind_median": median(difference),
            "fraction_crps_better_than_qblind": chain._mean([int(value < 0) for value in difference]),
            "fraction_crps_worse_than_qblind": chain._mean([int(value > 0) for value in difference]),
            "descriptive_sample_skill_supported": chain._mean(difference) < 0,
            "broad_empirical_endorsement": False, "blind_evaluation": False,
            "conditional_mean_rmse": math.sqrt(chain._mean([row["conditional_mean_squared_error"] for row in scores])),
            "conditional_mean_mae": chain._mean([row["conditional_mean_absolute_error"] for row in scores]),
            "v1_regression_central_rmse": math.sqrt(chain._mean([row["v1_regression_central_squared_error"] for row in scores])),
            "prior_mean_rmse": math.sqrt(chain._mean([row["prior_mean_squared_error"] for row in scores])),
            "coverage_90": chain._mean([int(row["stability_interval"]["covered_90"]) for row in scores]),
            "mean_width_90": chain._mean([row["stability_interval"]["width_90"] for row in scores]),
            "mean_interval_score_90": chain._mean([row["stability_interval"]["interval_score_90"] for row in scores]),
            "mean_event_model_weight": chain._mean([row["stability_prediction"]["event_model_weight"] for row in scored]),
            "baseline_only_forecast_n": sum(row["stability_prediction"]["event_model_weight"] == 0 for row in scored),
            "mean_v1_difference_vs_qblind_retained": chain._mean([row["crps_v1"] - row["crps_qblind"] for row in scores]),
            "claim_scope": "Named target, identical historical forecast events, one frozen post-v1 development algorithm; no universal improvement claim"}


def augment_report_with_stability(engine_report):
    """Return a deep copy with stability forecasts, exact updates and live artifacts.

    All v1 predictions/scores/summaries remain byte-equivalent JSON subobjects.
    For every forecast, stability_artifact is its PRE-event learner state, so the
    same shared production helper can reproduce the historical mixture exactly.
    """
    if engine_report.get("schema") != chain.REPORT_SCHEMA:
        raise chain.ChainInputError("Expected the frozen historical-chain report schema")
    result = deepcopy(engine_report)
    events = {event["event_id"]: event for event in result["event_receipts"]}
    if len(events) != len(result["event_receipts"]):
        raise chain.ChainInputError("Duplicate event receipts in the original report")
    asof = result.get("protocol", {}).get("asof_time_utc")
    latest_artifacts = []
    for target_result in result["target_results"]:
        target, state, pending = target_result["target"], _new_state(), []
        binding = _binding(target_result, target["weights"])
        forecasts = sorted(target_result["forecasts"], key=lambda row: (chain._time(row["cutoff_time_utc"], "forecast cutoff"), row["event_id"]))
        for forecast in forecasts:
            if forecast["status"] != "EVALUATED":
                forecast["stability_status"] = "NO_V1_FORECAST"
                continue
            cutoff = chain._time(forecast["cutoff_time_utc"], "forecast cutoff")
            completed = sorted([row for row in pending if chain._time(row["available_time_utc"], "feedback availability") < cutoff],
                               key=lambda row: (chain._time(row["available_time_utc"], "feedback availability"), row["event_id"]))
            for feedback in completed:
                _apply_feedback(state, feedback)
                pending.remove(feedback)
            event = events[forecast["event_id"]]
            if _binding(event, target["weights"]) != binding or _binding(forecast["model"], target["weights"]) != binding:
                raise chain.ChainInputError("Historical event or v1 model crosses stability learner scope")
            artifact = _artifact(state, binding, target["id"], forecast["cutoff_time_utc"], source_report_protocol=result.get("protocol"))
            prediction = predict_stabilized_distribution(forecast["model"], event, weights=target["weights"],
                                                         notional_usd=target["notional_usd"], stability_artifact=artifact)
            if prediction["v1_prediction_sha256"] != _digest(forecast["prediction"]):
                raise chain.ChainInputError("Stored v1 forecast differs from the shared predictor's expert distribution")
            feedback = _feedback(forecast, event, target, prediction, artifact)
            forecast.update(stability_status="EVALUATED_EXPERIMENTAL", stability_artifact=artifact,
                            stability_prediction=prediction, stability_scores=_scores(prediction, event, forecast["scores"]),
                            stability_receipt={"protocol_sha256": PROTOCOL_SHA256, "binding_sha256": artifact["binding_sha256"],
                                               "artifact_sha256": artifact["artifact_sha256"], "prediction_sha256": _digest(prediction),
                                               "prior_completed_event_ids": list(artifact["completed_event_ids"]),
                                               "event_model_weight": artifact["event_model_weight"],
                                               "sum_squared_gradients": artifact["sum_squared_gradients"],
                                               "v1_prediction_sha256": _digest(forecast["prediction"])})
            pending.append(feedback)
        if asof:
            cutoff = chain._time(asof, "report asof")
            completed = sorted([row for row in pending if chain._time(row["available_time_utc"], "feedback availability") < cutoff],
                               key=lambda row: (chain._time(row["available_time_utc"], "feedback availability"), row["event_id"]))
            for feedback in completed:
                _apply_feedback(state, feedback)
                pending.remove(feedback)
            artifact = _artifact(state, binding, target["id"], asof, source_report_protocol=result.get("protocol"))
            latest_artifacts.append({"target_id": target["id"], "weights": deepcopy(target["weights"]),
                                     "reference_notional_usd": target["notional_usd"], "artifact": artifact})
        target_result.update(stability_summary=_summary(forecasts),
                             stability_summary_2025_2026=_summary([row for row in forecasts if "2025-01-01" <= row["release_time_utc"] < "2027-01-01"]),
                             stability_feedback_ledger=deepcopy(state["ledger"]),
                             stability_unprocessed_feedback_ids=[row["event_id"] for row in pending])
    result["latest_stability_artifacts"] = latest_artifacts
    result["stability_layer"] = {"algorithm": ALGORITHM, "protocol_sha256": PROTOCOL_SHA256,
                                 "evaluation_label": EVALUATION_LABEL, "original_v1_report_sha256": _digest(engine_report),
                                 "original_v1_preserved": True, "blind_evaluation": False,
                                 "universal_improvement_claim": False, **_code_hashes(),
                                 "scope": "Target-bound joint distribution pool, trained only on completed genuine v1 chronological forecasts"}
    return result
