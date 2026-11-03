"""Point-3 identification boundary: an event contrast cannot determine today's P&L.

For a common future horizon, the event-vs-no-event gap G obeys the
accounting identity PNL_YES(today) = PNL_NO(today) + G. Neither the
no-event mark-to-horizon return nor the part already priced in today's
stock/ETF mark is recovered from G, P(E), or a binary contract's quote.
In particular, there is no hidden 50% or zero-background default.
"""
from __future__ import annotations

from math import isfinite

SCHEMA = "castle.conditional_today_mark_identification.v2"
_TOLERANCE = 1e-7


class ConditionalMarkError(ValueError):
    """Invalid conditional scenario gap; no today-mark claim may be published."""


def _number(value, label):
    if type(value) not in (int, float) or not isfinite(value):
        raise ConditionalMarkError(f"{label} must be finite numeric.")
    return float(value)


def conditional_today_sensitivity(gap_center_usd, gap_model_set_usd):
    """Retained call name, now a STRICT non-identification diagnostic.

    No B=0, no common q (whether 0%, 50% or 100%), no event-YES mark
    amounts, and no false bounded envelope. Even with a bounded G, the
    unknown no-event future mark leaves the event-YES P&L unbounded unless
    further economic constraints are independently supplied and verified.
    """
    center = _number(gap_center_usd, "gap_center_usd")
    raw = gap_model_set_usd
    if not isinstance(raw, list) or len(raw) != 2:
        raise ConditionalMarkError("An ordered two-sided model gap set is required.")
    lower = _number(raw[0], "gap_lower_usd")
    upper = _number(raw[1], "gap_upper_usd")
    if lower > center + _TOLERANCE or center > upper + _TOLERANCE or lower > upper:
        raise ConditionalMarkError("The central gap must lie inside its model-set bounds.")
    return {
        "schema": SCHEMA,
        "status": "EVENT_YES_PNL_FROM_TODAY_NOT_IDENTIFIED",
        "return_horizon_trading_sessions": 20,
        "event_minus_no_event_gap_center_usd": center,
        "event_minus_no_event_gap_model_set_usd": [lower, upper],
        "identity": "EVENT_YES_PNL_FROM_TODAY = NO_EVENT_PNL_FROM_TODAY + EVENT_MINUS_NO_EVENT_GAP",
        "event_yes_pnl_from_today_usd": None,
        "event_yes_pnl_from_today_model_set_usd": None,
        "no_event_future_mark_identified": False,
        "equity_priced_in_fraction_identified": False,
        "zero_background_assumed": False,
        "arbitrary_common_priced_in_fraction_assumed": False,
        "physical_event_probability_used": False,
        "prediction_market_quote_used_as_equity_q": False,
        "confidence_interval_claimed": False,
    }


def verified_saved_today_sensitivity(saved, gap_center_usd, gap_model_set_usd):
    """Full comparison to deterministic output rejects invented mark numbers."""
    if not isinstance(saved, dict):
        return False
    try:
        expected = conditional_today_sensitivity(gap_center_usd, gap_model_set_usd)
    except ConditionalMarkError:
        return False
    return saved == expected
