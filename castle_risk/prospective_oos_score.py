"""Prospectively locked event-probability scoring, with exact-market comparator.

Runs ONLY after new binary outcomes have actually resolved. Historical
backfill of a language model is not treated as new OOS evidence; a GitHub
commit and provider message receipt must predate the observable event result.
Even correct local chronology does not cryptographically prove source truth
or that a proprietary model's pretraining excluded the answer.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import fsum, isfinite, log, log1p
from random import Random

LOCK_SCHEMA = "castle.prospective_event_lock.v1"
SCORE_SCHEMA = "castle.prospective_oos_probability_score.v1"


class ProspectiveScoreError(ValueError):
    """Invalid lock, ambiguous resolved event, or simple post-outcome leakage."""


def _time(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProspectiveScoreError(label + " must be an ISO UTC timestamp.")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ProspectiveScoreError(label + " must be an ISO UTC timestamp.") from None
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ProspectiveScoreError(label + " must include a timezone.")
    return dt


def _p(value: object, name: str) -> float:
    if type(value) not in (float, int) or not isfinite(value) or not 0 <= value <= 1:
        raise ProspectiveScoreError(name + " must be finite in [0,1].")
    return float(value)


def _logloss(probability: float, outcome: int) -> float:
    if probability == 0.0:
        return float("inf") if outcome else 0.0
    if probability == 1.0:
        return 0.0 if outcome else float("inf")
    return -log(probability) if outcome else -log1p(-probability)


def _mean_loss(vals: list[float]) -> float | None:
    # Never emit invalid JSON Infinity for a confidently wrong 0/1 forecast.
    return (None if not vals or any(v == float("inf") for v in vals)
            else fsum(vals) / len(vals))


def evaluate_prospective_lock(lock: object, adjudications: object, *,
                              min_independent_families: int = 30,
                              bootstrap_draws: int = 2000,
                              seed: int = 119,
                              audit_time_utc: str | None = None) -> dict:
    """Score only prelocked model P and matched market Q after independent resolution.

    Missing outcomes are PENDING, not a guessed NO. All locked rows count
    toward planned coverage; all resolved rows count toward scored coverage.
    Market vs model Brier comparisons are paired on the exact same resolved
    binary target. A clustered bootstrap resamples independent event families,
    NOT sibling strikes on one CPI release as independent observations.
    """
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
        raise ProspectiveScoreError("Expected a frozen prospective lock.")
    events = lock.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 10000:
        raise ProspectiveScoreError("Expected 1-10000 originally locked events.")
    if not isinstance(adjudications, list) or len(adjudications) > len(events):
        raise ProspectiveScoreError("Adjudications must be a list no larger than the lock.")
    if (type(min_independent_families) is not int or min_independent_families < 2
            or type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 10000
            or type(seed) is not int):
        raise ProspectiveScoreError("Invalid prespecified diagnostics or bootstrap settings.")
    now = (_time(audit_time_utc, "audit_time_utc")
           if audit_time_utc else datetime.now(timezone.utc))
    locked_at = _time(lock.get("locked_at_utc"), "locked_at_utc")
    if locked_at > now:
        raise ProspectiveScoreError("An ex post audit cannot precede lock creation.")
    originals = {}
    forecast_versions = set()
    for e in events:
        if not isinstance(e, dict):
            raise ProspectiveScoreError("Locked event must be an object.")
        ident, target, family = e.get("event_id"), e.get("target_condition"), e.get("event_family_id")
        if (not isinstance(ident, str) or not ident.strip() or ident in originals
                or not isinstance(target, str) or len(target.strip()) < 10
                or not isinstance(family, str) or not family.strip()):
            raise ProspectiveScoreError("Unique original event, predicate, and family required.")
        originals[ident] = e
        forecast = e.get("model_forecast")
        if not isinstance(forecast, dict) or forecast.get("status") not in (
                "MODEL_FORECAST_UNCALIBRATED", "ABSTAIN"):
            raise ProspectiveScoreError("Lock every model forecast or explicit abstention.")
        model_version = forecast.get("model_version")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ProspectiveScoreError("Every forecast requires a frozen prompt/model version.")
        forecast_versions.add(model_version)
        issued = _time(forecast.get("provider_response_received_at_utc"),
                       "provider_response_received_at_utc")
        evidence = _time(forecast.get("last_evidence_retrieved_at_utc"),
                         "last_evidence_retrieved_at_utc")
        if not evidence <= issued <= locked_at:
            raise ProspectiveScoreError("Model evidence, response, and commit lock chronology invalid.")
        if forecast["status"] == "MODEL_FORECAST_UNCALIBRATED":
            p = _p(forecast.get("probability_center"), "model P")
            bounds = forecast.get("credal_interval")
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ProspectiveScoreError("Model credal bounds missing.")
            lo, hi = (_p(x, "model credal bound") for x in bounds)
            if not lo <= p <= hi:
                raise ProspectiveScoreError("Model point outside credal bounds.")
            if not isinstance(forecast.get("provider_message_id"), str) or not forecast["provider_message_id"]:
                raise ProspectiveScoreError("Missing independently returned provider message ID.")
        elif forecast.get("probability_center") is not None:
            raise ProspectiveScoreError("An abstention cannot have a model probability.")
        quote = e.get("market_quote")
        if quote is not None:
            if not isinstance(quote, dict):
                raise ProspectiveScoreError("Quote must be an object.")
            if not _time(quote.get("captured_at_utc"), "quote captured_at_utc") <= issued:
                raise ProspectiveScoreError("Market benchmark must predate the model forecast receipt.")
            b, a = _p(quote.get("yes_bid"), "YES bid"), _p(quote.get("yes_ask"), "YES ask")
            if b > a:
                raise ProspectiveScoreError("Inverted market bid/ask.")
            if quote.get("match_status") not in (
                    "EXACT_CONTRACT_RULES_AUDITED", "PROVISIONAL_MATCH_NOT_SCORED"):
                raise ProspectiveScoreError("Market match status not audited.")
            if not isinstance(quote.get("ticker"), str) or not quote["ticker"]:
                raise ProspectiveScoreError("Missing exact venue market identifier.")
    if len(forecast_versions) != 1:
        raise ProspectiveScoreError("Mixed prospective model versions require separate scorecards.")
    observed = {}
    for row in adjudications:
        if not isinstance(row, dict):
            raise ProspectiveScoreError("Adjudication must be an object.")
        ident = row.get("event_id")
        if ident not in originals or ident in observed:
            raise ProspectiveScoreError("Unexpected or duplicate event adjudication.")
        outcome = row.get("outcome")
        if type(outcome) is not int or outcome not in (0, 1):
            raise ProspectiveScoreError("Outcome must be exactly integer 0 or 1.")
        if row.get("target_condition") != originals[ident]["target_condition"]:
            raise ProspectiveScoreError("Adjudicated predicate differs from frozen original.")
        if row.get("resolution_type") not in (
                "VENUE_SETTLEMENT", "OFFICIAL_DATA_RELEASE", "INDEPENDENT_GEO_ADJUDICATION"):
            raise ProspectiveScoreError("Independent resolution method required.")
        if not isinstance(row.get("resolution_source_url"), str) or not row["resolution_source_url"].startswith("https://"):
            raise ProspectiveScoreError("A publicly resolvable source URL is required.")
        settled = _time(row.get("resolved_at_utc"), "resolved_at_utc")
        if not locked_at < settled <= now:
            raise ProspectiveScoreError("Outcome must occur after lock and before audit.")
        observed[ident] = row
    score_rows, paired = [], []
    for ident, e in originals.items():
        if ident not in observed:
            continue
        y = observed[ident]["outcome"]
        forecast = e["model_forecast"]
        if forecast["status"] == "ABSTAIN":
            continue
        p = forecast["probability_center"]
        model_ll = _logloss(p, y)
        item = {"event_id": ident, "event_family_id": e["event_family_id"],
                "model_p": p, "outcome": y, "model_brier": (p-y)**2,
                "model_log_loss": model_ll if isfinite(model_ll) else None,
                "model_log_loss_is_infinite": not isfinite(model_ll)}
        quote = e.get("market_quote")
        if quote is not None and quote["match_status"] == "EXACT_CONTRACT_RULES_AUDITED":
            q = (quote["yes_bid"] + quote["yes_ask"]) / 2
            market_ll = _logloss(q, y)
            item.update({"market_q_mid": q, "market_brier": (q-y)**2,
                         "market_log_loss": market_ll if isfinite(market_ll) else None,
                         "market_log_loss_is_infinite": not isfinite(market_ll),
                         "brier_improvement_over_market": (q-y)**2-(p-y)**2})
            paired.append(item)
        score_rows.append(item)
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        by_family[row["event_family_id"]].append(row["brier_improvement_over_market"])
    fam_means = [_mean_loss(v) for _, v in sorted(by_family.items())]
    mean_delta = _mean_loss(fam_means)
    ci = None
    if len(fam_means) >= 10:
        rng = Random(seed)
        b = []
        for _ in range(bootstrap_draws):
            b.append(fsum(fam_means[rng.randrange(len(fam_means))]
                          for _ in fam_means)/len(fam_means))
        b.sort()
        ci = [b[int(.025 * len(b))], b[min(len(b)-1, int(.975 * len(b)))]]
    enough = len(set(row["event_family_id"] for row in score_rows)) >= min_independent_families
    return {
        "schema": SCORE_SCHEMA,
        "status": ("NO_RESOLVED_HOLDOUT_EVENTS_YET" if not observed
                   else "INSUFFICIENT_INDEPENDENT_FAMILIES" if not enough
                   else "DESCRIPTIVE_PROSPECTIVE_SCORES_REQUIRING_EXTERNAL_AUDIT"),
        "locked_event_count": len(events),
        "resolved_event_count": len(observed),
        "unresolved_event_count": len(events)-len(observed),
        "scored_model_count": len(score_rows),
        "frozen_model_version": next(iter(forecast_versions)),
        "resolved_abstention_count": len(observed)-len(score_rows),
        "independent_scored_families": len(set(r["event_family_id"] for r in score_rows)),
        "model_brier": _mean_loss([r["model_brier"] for r in score_rows]),
        "model_log_loss": (_mean_loss([r["model_log_loss"] for r in score_rows])
                           if not any(r["model_log_loss_is_infinite"] for r in score_rows)
                           else None),
        "model_log_loss_infinite_due_to_confident_error": any(
            r["model_log_loss_is_infinite"] for r in score_rows),
        "paired_exact_market_count": len(paired),
        "independent_paired_market_families": len(by_family),
        "market_midpoint_brier_on_paired_subset": _mean_loss([r["market_brier"] for r in paired]),
        "model_brier_on_paired_subset": _mean_loss([r["model_brier"] for r in paired]),
        "market_minus_model_brier_by_family": mean_delta,
        "cluster_bootstrap_95_percent_difference_interval": ci,
        "model_score_coverage_of_resolved": len(score_rows)/len(observed) if observed else None,
        "forecast_empirically_calibrated": False,
        "external_outcome_authenticity_verified_by_this_module": False,
        "credible_bounds_are_epistemic_not_95_percent_coverage_intervals": True,
        "market_quote_is_benchmark_not_physical_probability_identification": True,
        "score_rows": score_rows,
    }
