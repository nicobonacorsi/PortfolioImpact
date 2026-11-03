"""Small reviewer-facing readout of the unchanged Option 1 mathematical engine.

This module is intentionally pure: it does not alter event identity, return
models, vendor receipts, source probabilities, or ranking. The full audit is
retained in the existing inspector behind one optional disclosure control.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _finite(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value))


def _worst_case(event):
    """Use only actual admitted outcomes with recorded reference Q >= 1%."""
    scenarios = (event.get("contract_first") or {}).get("contract_scenarios") or []
    if scenarios:
        rows = [row for scenario in scenarios for row in scenario.get("outcomes", [])
                if _finite(row.get("pnl_if_outcome_usd"))
                and _finite(row.get("q_reference")) and row["q_reference"] >= .01
                and not scenario.get("summary", {}).get("no_portfolio_channel")]
        if not rows:
            return None
        return min(row["pnl_if_outcome_usd"] for row in rows)
    summary = ((event.get("outcome_table") or {}).get("summary") or {})
    row = summary.get("largest_loss_outcome_reference_q_at_least_1pct") or {}
    return row.get("pnl_if_outcome_usd") if _finite(row.get("pnl_if_outcome_usd")) else None


def _quote_state(event):
    first = event.get("contract_first") or {}
    if first.get("contract_scenarios"):
        bounds = ((first.get("broad_risk") or {}).get("probability") or {}).get("lower_bounds_by_venue") or []
        return "Contract scenarios; broad bound" if bounds else "Contract scenarios; broad Q unknown"
    if (event.get("distribution") or {}).get("atoms"):
        return "Outcome quotes Q"
    return "No admitted event quote"


def _short_family(event):
    return {
        "FED_POLICY": "Fed", "US_INFLATION": "Inflation", "US_PAYROLLS": "Employment",
        "TAIWAN_SUPPLY": "Taiwan", "EXPORT_CONTROLS": "Export controls",
        "TRADE_TARIFFS": "Tariffs", "HORMUZ_ENERGY": "Energy supply",
        "OPEC_SUPPLY": "Oil supply", "RUSSIA_ENERGY_SANCTIONS": "Sanctions",
        "US_DRUG_PRICING": "Drug pricing", "FDA_DRUG_APPROVAL": "FDA",
        "BANK_CREDIT_STRESS": "Credit stress", "BANK_CAPITAL_RULES": "Bank rules",
    }.get(event.get("family"), "Other")


def build_reviewer_overview(report: Mapping) -> dict:
    """A concise view, with explicit distinction between modeled and unknown.

    Q-standard deviation is *not* physical-event probability or the loss
    conditional on the broad research hypothesis; it measures variation across
    modeled source-defined outcomes under a declared economic specification.
    """
    analysis = report.get("default_analysis") or {}
    events = analysis.get("events") or []
    ranked = sorted(events, key=lambda e: ((e.get("ranking") or {}).get("display_order", 10**9),
                                          str(e.get("event_id", ""))))
    rows, unquantified = [], []
    for event in ranked:
        value = (event.get("risk_ranking") or {}).get("q_standard_deviation_usd")
        modeled = _finite(value)
        impact = _worst_case(event)
        item = {
            "event_id": event.get("event_id"), "title": str(event.get("title") or "Unspecified event"),
            "family": _short_family(event), "event_date": event.get("event_date"),
            "modeled_q_std_usd": round(value, 2) if modeled else None,
            "worst_priced_outcome_usd": round(impact, 2) if impact is not None else None,
            "market_evidence": _quote_state(event),
            "full_outcome_model_available": modeled,
            "source_event": event,
        }
        rows.append(item)
        if not modeled:
            unquantified.append(item)
    stages = report.get("stages") or {}
    source_quality_partial = any(isinstance(stage, Mapping) and stage.get("status") in {
        "PARTIAL", "FAILED", "BLOCKED_MISSING_CREDENTIAL"} for stage in stages.values())
    return {
        "rows": rows,
        "modeled_count": sum(row["full_outcome_model_available"] for row in rows),
        "unquantified": unquantified,
        "source_quality_partial": source_quality_partial,
        "portfolio_value_usd": (report.get("portfolio") or {}).get("notional"),
        "evaluated_at_utc": analysis.get("evaluated_at_utc"),
        "interpretation": ("Q-std describes variability of modeled, priced settlement scenarios, "
                           "not physical event probability or worst possible loss. "
                           "Values depend on explicit transmission assumptions."),
    }
