"""Small, read-only views of existing event outcomes for the main app.

Nothing here estimates a probability, recalibrates an impact, or changes the
admission rules. A displayed probability always belongs to its named outcome.
"""
from __future__ import annotations

import math
from datetime import datetime


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def money(value):
    if not _finite(value):
        return "Unknown"
    return ("-" if value < 0 else "+" if value > 0 else "") + f"${abs(value):,.0f}"


def impact(row):
    # Some archived contract scenarios store zero when no transmission channel
    # was modeled. That is not a quantified zero loss for the portfolio.
    if row.get("no_portfolio_channel"):
        return "Unknown — no modeled channel"
    if row.get("partial_portfolio_coverage"):
        value = row.get("known_positions_only_pnl_usd", row.get("pnl_if_outcome_usd"))
        return money(value) + " subtotal; total unknown" if _finite(value) else "Unknown total"
    return money(row.get("pnl_if_outcome_usd"))


def has_partial_impact(event):
    table = event.get("outcome_table") or {}
    return bool(str(table.get("status", "")).startswith("PARTIAL")
                or table.get("missing_symbols")
                or (_finite(table.get("missing_weight")) and table["missing_weight"] > 0))


def probability(row):
    bounds = row.get("q_interval")
    if (isinstance(bounds, (list, tuple)) and len(bounds) == 2
            and all(_finite(value) for value in bounds)
            and 0 <= bounds[0] <= bounds[1] <= 1):
        low, high = bounds
        # A narrow quoted interval is still an interval, even if both displayed
        # endpoints happen to round to the same tenth of a percent.
        return f"{low:.1%}–{high:.1%}"
    if bounds is not None:
        return "Unknown"
    reference = row.get("q_reference")
    return f"{reference:.1%} reference" if _finite(reference) and 0 <= reference <= 1 else "Unknown"


def title(event):
    return {
        "FED_POLICY": "Fed interest-rate decision",
        "US_INFLATION": "US inflation release",
        "US_PAYROLLS": "US jobs report",
        "TAIWAN_SUPPLY": "Taiwan supply disruption",
        "EXPORT_CONTROLS": "Chip export restrictions",
    }.get(event.get("family"), str(event.get("title") or "Portfolio risk"))


def date_label(event):
    value = event.get("event_date")
    if not value:
        return "Date not established"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return str(value)


def outcome_rows(event):
    """Retain all source outcomes and unknown impacts, with exact contract scope."""
    first = event.get("contract_first") or {}
    scenarios = first.get("contract_scenarios") or []
    if first:
        rows = []
        for scenario in scenarios:
            contract = scenario.get("contract") or {}
            question = contract.get("market_question") or contract.get("contract_id") or "Contract"
            for outcome in scenario.get("outcomes") or []:
                outcome_id = str(outcome.get("outcome_id", ""))
                side = ("YES" if outcome_id.endswith(":YES") else "NO" if outcome_id.endswith(":NO")
                        else str(outcome.get("label") or "Unspecified outcome"))
                rows.append(dict(outcome, display_label=f"{side} · {question}",
                                 contract_question=question,
                                 quote_scope=contract.get("quote_scope"),
                                 partial_portfolio_coverage=bool(scenario.get("missing_symbols")),
                                 no_portfolio_channel=bool((scenario.get("summary") or {}).get("no_portfolio_channel"))))
        return rows
    outcomes = (event.get("outcome_table") or {}).get("outcomes")
    if not outcomes:
        outcomes = (event.get("distribution") or {}).get("atoms") or []
    rows = []
    for outcome in outcomes:
        label = str(outcome.get("label") or outcome.get("outcome_id") or "Outcome")
        for direction in ("Hike", "Cut"):
            if label == direction + " source bucket; magnitude ambiguous":
                label = f"{direction} · size assumed by the model"
        rows.append(dict(outcome, display_label=label,
                         partial_portfolio_coverage=has_partial_impact(event)))
    return rows


def downside(event):
    """Same Q >= 1% selection as the existing reviewer; never a max-loss bound."""
    rows = [row for row in outcome_rows(event)
            if _finite(row.get("pnl_if_outcome_usd"))
            and _finite(row.get("q_reference")) and row["q_reference"] >= .01
            and not row.get("no_portfolio_channel") and not row.get("partial_portfolio_coverage")]
    return min(rows, key=lambda row: row["pnl_if_outcome_usd"]) if rows else None


def probability_scope(event):
    first = event.get("contract_first") or {}
    if first:
        broad = (first.get("broad_risk") or {}).get("probability") or {}
        if broad.get("lower_bounds_by_venue"):
            return "Contract odds; the broad risk has only a lower bound."
        if broad.get("conditional_lower_bounds_by_venue"):
            return "Contract odds; the broad risk has only a conditional bound."
        return "Contract odds; the broad-risk probability is unknown."
    return "Odds apply to the named outcome."


def why_it_matters(event):
    """Explain the recorded channel, without inventing portfolio sensitivities."""
    links = event.get("exposure_basis") or []
    if links:
        statements = [str(row.get("statement")) for row in links if row.get("statement")]
        if statements:
            return statements[0]
    family = event.get("family")
    return {
        "FED_POLICY": "An unexpected rate decision can move share prices and the value of bond holdings.",
        "US_INFLATION": "Inflation surprises can change interest-rate expectations and reprice stocks and bonds.",
        "US_PAYROLLS": "An employment surprise can change expectations for growth and interest rates.",
        "TAIWAN_SUPPLY": "Disruption in Taiwan can affect companies linked to semiconductor production and trade.",
        "EXPORT_CONTROLS": "Export restrictions can affect chip suppliers and companies that depend on their products.",
    }.get(family, "The investigation linked this event to the portfolio. The evidence below shows that link.")
