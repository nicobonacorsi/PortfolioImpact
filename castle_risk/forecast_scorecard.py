"""Chronology-gated evaluation of locked resolved event forecasts.

This is an offline scoring instrument, NOT proof that a model did not know a
future event during pretraining or that every cited source was actually
published before the frozen forecast. Independently archived provider receipts
and prospectively resolved targets are required for a real holdout claim.
"""
from __future__ import annotations

from datetime import datetime
from math import fsum, isfinite, log, log1p

SCHEMA = "castle.structural_forecast_scorecard.v1"


class ScorecardInputError(ValueError):
    """An archived evaluation row is invalid or vulnerable to obvious leakage."""


def _when(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ScorecardInputError(f"{label} must be an ISO timestamp.")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ScorecardInputError(f"{label} is not a valid ISO timestamp.") from None
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ScorecardInputError(f"{label} must contain a time zone.")
    return instant


def _prob(value: object, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ScorecardInputError(f"{label} must be finite in [0,1].")
    return float(value)


def score_locked_forecasts(records: object, *, min_sample: int = 30) -> dict:
    """Compute central Brier/log-loss and a descriptive epistemic worst-case score.

    Every row includes ALL eligible resolved targets, even abstentions: otherwise
    selective reporting would inflate measured coverage. Each row requires
    a record-creation timestamp, a source cutoff, an outcome resolution time,
    an explicit disjoint holdout flag, and one frozen model version.

    A chronology gate detects simple post-resolution leaks; it cannot certify
    model pretraining isolation, source content or independent event adjudication.
    """
    if not isinstance(records, list) or not records or len(records) > 10000:
        raise ScorecardInputError("Supply 1 to 10,000 resolved holdout event rows.")
    if type(min_sample) is not int or not 2 <= min_sample <= 10000:
        raise ScorecardInputError("min_sample must be an integer from 2 to 10,000.")
    seen = set()
    versions = set()
    scored = []
    abstained = 0
    for item in records:
        if not isinstance(item, dict):
            raise ScorecardInputError("Each archived event row must be an object.")
        ident = item.get("event_id")
        condition = item.get("target_condition")
        version = item.get("model_version")
        if (not isinstance(ident, str) or not ident.strip() or ident in seen or
                not isinstance(condition, str) or len(condition.strip()) < 10 or
                not isinstance(version, str) or not version.strip()):
            raise ScorecardInputError("Use unique event IDs, original target questions, and one model version.")
        seen.add(ident)
        versions.add(version)
        observed = item.get("outcome")
        if type(observed) is not int or observed not in (0, 1):
            raise ScorecardInputError("Resolved binary outcome must be exactly integer 0 or 1.")
        if item.get("held_out_before_forecast") is not True:
            raise ScorecardInputError("Holdout status was not declared before forecasting.")
        forecast_time = _when(item.get("forecast_recorded_at_utc"), "forecast_recorded_at_utc")
        source_time = _when(item.get("latest_evidence_at_utc"), "latest_evidence_at_utc")
        resolved_time = _when(item.get("resolved_at_utc"), "resolved_at_utc")
        training_time = _when(item.get("model_training_cutoff_utc"), "model_training_cutoff_utc")
        if not (training_time <= forecast_time and source_time <= forecast_time < resolved_time):
            raise ScorecardInputError("Training, source or resolution chronology is not compatible with a locked forecast.")
        status = item.get("status")
        if status == "ABSTAIN":
            if item.get("probability_center") is not None or item.get("credal_interval") is not None:
                raise ScorecardInputError("Abstentions must not report fabricated probabilities.")
            abstained += 1
            continue
        if status != "MODEL_FORECAST_UNCALIBRATED":
            raise ScorecardInputError("Unknown or empirically unverified forecast status.")
        p = _prob(item.get("probability_center"), "probability_center")
        bounds = item.get("credal_interval")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ScorecardInputError("Credal bounds must be an ordered pair.")
        lo, hi = [_prob(v, "credal bound") for v in bounds]
        if not lo <= p <= hi:
            raise ScorecardInputError("Reported central probability lies outside its credal bounds.")
        brier = (p - observed) ** 2
        # Exact log loss: a confidently wrong probability 0 or 1 has
        # unbounded loss. Never silently clip it to manufacture a finite mean.
        infinite = (p == 0.0 and observed == 1) or (p == 1.0 and observed == 0)
        log_loss = None if infinite else (-log(p) if observed else -log1p(-p)) if 0 < p < 1 else 0.0
        worst_brier = (1 - lo) ** 2 if observed else hi ** 2
        scored.append((p, observed, brier, log_loss, worst_brier, hi - lo))
    if len(versions) != 1:
        raise ScorecardInputError("Do not aggregate incompatible model versions without stratification.")
    count = len(scored)
    bins = []
    for i in range(10):
        group = [(p, y) for p, y, *_ in scored
                 if (min(9, int(p * 10)) == i)]
        if group:
            bins.append({"bin": i, "count": len(group),
                         "mean_forecast": fsum(v[0] for v in group) / len(group),
                         "observed_event_rate": fsum(v[1] for v in group) / len(group)})
    return {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_SCORES_NOT_EXTERNAL_CALIBRATION_CERTIFICATION" if count >= min_sample
                  else "INSUFFICIENT_SCORED_EVENTS_FOR_CALIBRATION_DIAGNOSTICS",
        "model_version": next(iter(versions)),
        "eligible_resolved_events": len(records),
        "scored_forecasts": count,
        "abstentions": abstained,
        "forecast_coverage": count / len(records),
        "brier_score": fsum(row[2] for row in scored) / count if count else None,
        "log_loss": (fsum(row[3] for row in scored) / count
                     if count and all(row[3] is not None for row in scored) else None),
        "log_loss_infinite_due_to_confident_error": any(row[3] is None for row in scored),
        "worst_case_brier_over_credal_bounds": (fsum(row[4] for row in scored) / count if count else None),
        "mean_credal_width": fsum(row[5] for row in scored) / count if count else None,
        "reliability_deciles": bins,
        "quality_scope": "CHRONOLOGY_AND_DECLARED_HOLDOUT_ONLY_SOURCES_OUTCOMES_AND_MODEL_KNOWLEDGE_NOT_AUDITED",
        "forecast_empirically_calibrated": False,
        "credal_bounds_are_not_confidence_intervals": True,
        "worst_case_score_is_descriptive_not_A_PROPER_IMPRECISE_SCORING_RULE": True,
    }
