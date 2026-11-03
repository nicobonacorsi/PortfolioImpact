"""R125 identity and coherence audit for every displayed point-2 number.

Castle requirement 12 of the R125 brief: every probability exposed in the
interface must belong to the same event, with identical definition, deadline
and capture time. This module re-derives those identities from the public
case (what the browser receives) against the private derived report (what the
engine produced), recomputes every credal interval with the greedy solver AND
the independent LP, and classifies abstentions. It changes nothing.
"""
from __future__ import annotations

from math import isfinite

from .credal_scenario import CredalInputError, evaluate_structural_partition, evaluate_structural_partition_lp

FORECAST_SCHEMA = "castle.autonomous_structural_event_forecast.v1"
REVIEW_SCHEMA = "castle.event_prediction_market_review.v1"
CONTEXT_SCHEMA = "castle.event_probability_assumptions.v1"


def _same_identity(row: dict, context: dict) -> bool:
    return all(row.get(key) == context.get(key)
               for key in ("event_id", "target_condition", "deadline", "as_of_utc"))


def audit_public_case(case: dict, report: dict) -> dict:
    """Return per-event audit rows and aggregate pass/fail for the identity gate."""
    if not isinstance(case, dict) or not isinstance(case.get("risks"), list):
        raise ValueError("A public case with risks is required.")
    forecasts = (report or {}).get("autonomous_event_forecasts") or {}
    reviews = (report or {}).get("event_market_reviews") or {}
    rows = []
    for risk in case["risks"]:
        if not isinstance(risk, dict):
            continue
        context = risk.get("event_probability_question")
        if not isinstance(context, dict) or context.get("schema") != CONTEXT_SCHEMA:
            continue
        if context.get("status") != "OPEN_FOR_DECLARED_EVENT_FORECAST":
            continue
        row = {"event_id": risk.get("id"), "title": risk.get("title"), "date": risk.get("date"),
               "context_matches_risk": (context.get("event_id") == risk.get("id")
                                        and context.get("target_condition") == risk.get("title")
                                        and context.get("deadline") == risk.get("date")
                                        and context.get("as_of_utc") == case.get("capture")),
               "forecast_displayed": False, "forecast_identity_ok": None, "forecast_status": None,
               "forecast_matches_report": None, "credal_recomputation_ok": None, "lp_crosscheck_ok": None,
               "center_inside_interval": None, "masses_sum_to_one": None, "single_final_residual": None,
               "abstention_class": None, "market_review_displayed": False, "market_review_identity_ok": None,
               "market_review_matches_report": None, "exact_quotes_displayed": 0,
               "exact_quotes_identity_ok": None, "q_substituted_flag_false": None}
        forecast = risk.get("autonomous_event_forecast")
        if isinstance(forecast, dict):
            row["forecast_displayed"] = True
            row["forecast_status"] = forecast.get("status")
            row["forecast_identity_ok"] = (forecast.get("schema") == FORECAST_SCHEMA
                                           and _same_identity(forecast, context))
            source = forecasts.get(risk.get("id")) if isinstance(forecasts, dict) else None
            row["forecast_matches_report"] = (isinstance(source, dict) and _same_identity(source, context)
                                              and source.get("status") == forecast.get("status")
                                              and source.get("probability_center") == forecast.get("probability_center")
                                              and source.get("credal_interval") == forecast.get("credal_interval"))
            row["q_substituted_flag_false"] = forecast.get("prediction_market_Q_substituted") is False or forecast.get("status") == "ABSTAIN"
            if forecast.get("status") == "ABSTAIN":
                row["abstention_class"] = forecast.get("abstention_class")
            elif forecast.get("status") == "MODEL_FORECAST_UNCALIBRATED":
                branches = forecast.get("scenario_branches")
                try:
                    greedy = evaluate_structural_partition(branches)
                    lp = evaluate_structural_partition_lp(branches)
                    center, interval = forecast.get("probability_center"), forecast.get("credal_interval")
                    row["credal_recomputation_ok"] = (
                        isinstance(interval, list) and len(interval) == 2 and type(center) in (int, float)
                        and isfinite(center) and abs(center - greedy["probability_center"]) <= 1e-9
                        and all(abs(a - b) <= 1e-9 for a, b in zip(interval, greedy["credal_interval"])))
                    row["lp_crosscheck_ok"] = bool(lp["agrees_within_1e_9"])
                    row["center_inside_interval"] = bool(interval[0] - 1e-12 <= center <= interval[1] + 1e-12)
                    row["masses_sum_to_one"] = abs(sum(b["scenario_probability"]["center"] for b in branches) - 1) <= 1e-9
                    row["single_final_residual"] = (sum(1 for b in branches if b.get("residual") is True) == 1
                                                    and branches[-1].get("residual") is True)
                except (CredalInputError, TypeError, KeyError, ValueError, IndexError):
                    row["credal_recomputation_ok"] = False
                    row["lp_crosscheck_ok"] = False
                quotes = forecast.get("exact_market_quotes") or []
                row["exact_quotes_displayed"] = len(quotes)
                if quotes:
                    review = reviews.get(risk.get("id")) if isinstance(reviews, dict) else None
                    exact_ids = {(c.get("venue"), c.get("contract_id"))
                                 for c in ((review or {}).get("decisions") or {}).get("exact") or []}
                    row["exact_quotes_identity_ok"] = all(
                        (q.get("venue"), q.get("contract_id")) in exact_ids and q.get("quote_is_market_Q_not_event_P") is True
                        for q in quotes)
        review_public = risk.get("prediction_market_review")
        if isinstance(review_public, dict):
            row["market_review_displayed"] = True
            row["market_review_identity_ok"] = (review_public.get("schema") == REVIEW_SCHEMA
                                                and _same_identity(review_public, context)
                                                and review_public.get("quotes_are_market_Q_not_event_P") is True)
            source = reviews.get(risk.get("id")) if isinstance(reviews, dict) else None
            row["market_review_matches_report"] = (isinstance(source, dict) and _same_identity(source, context)
                                                   and source.get("counts") == review_public.get("counts"))
        rows.append(row)
    displayed = [r for r in rows if r["forecast_displayed"]]
    numeric = [r for r in displayed if r["forecast_status"] == "MODEL_FORECAST_UNCALIBRATED"]
    market_rows = [r for r in rows if r["market_review_displayed"]]
    checks = {
        "every_displayed_forecast_identity_ok": all(r["forecast_identity_ok"] for r in displayed),
        "every_displayed_forecast_matches_report": all(r["forecast_matches_report"] for r in displayed),
        "every_numeric_forecast_recomputes": all(r["credal_recomputation_ok"] for r in numeric),
        "every_numeric_forecast_passes_lp_crosscheck": all(r["lp_crosscheck_ok"] for r in numeric),
        "every_numeric_center_inside_interval": all(r["center_inside_interval"] for r in numeric),
        "every_numeric_masses_sum_to_one": all(r["masses_sum_to_one"] for r in numeric),
        "every_numeric_single_final_residual": all(r["single_final_residual"] for r in numeric),
        "no_quote_substituted_as_probability": all(r["q_substituted_flag_false"] for r in displayed),
        "every_displayed_market_review_identity_ok": all(r["market_review_identity_ok"] for r in market_rows),
        "every_displayed_market_review_matches_report": all(r["market_review_matches_report"] for r in market_rows),
        "every_displayed_exact_quote_identity_ok": all(r["exact_quotes_identity_ok"] in (None, True) for r in numeric),
        "every_abstention_classified": all(bool(r["abstention_class"]) for r in displayed
                                           if r["forecast_status"] == "ABSTAIN"),
        "every_context_matches_its_risk": all(r["context_matches_risk"] for r in rows),
    }
    return {
        "schema": "castle.r125.point2_identity_audit.v1",
        "eligible_contexts": len(rows), "forecasts_displayed": len(displayed),
        "numeric_forecasts_displayed": len(numeric),
        "abstentions_displayed": sum(1 for r in displayed if r["forecast_status"] == "ABSTAIN"),
        "market_reviews_displayed": len(market_rows),
        "checks": checks, "all_checks_pass": all(checks.values()), "rows": rows,
    }
