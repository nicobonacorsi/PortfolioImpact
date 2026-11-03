"""Sharp *conditional* binary-state pricing-weight sensitivity, not a forecast.

An admitted YES/NO contract may supply a quote Q and a declared economic
contrast Delta.  Q prices a prediction-market claim; it does NOT identify the
state-pricing weight pi relevant to the holdings.  Conditional on a common
two-state, same-horizon, discounted-value decomposition,

    YES repricing = (1-pi) Delta; NO repricing = -pi Delta.

Without an independently justified pi, only scenario sensitivity and ranges
under declared pi bounds are available.  A risk-neutral pi need not equal a
physical event probability or the price of a prediction-market contract.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SCHEMA = "castle.option1.priced_in_sensitivity.v1"
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _range(value: Sequence[object], name: str, *, unit: bool = False) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lo, hi = (_number(v, name) for v in value)
    if lo > hi or (unit and not 0 <= lo <= hi <= 1):
        raise ValueError(f"{name} is inverted or out of domain")
    return lo, hi


def _product_bounds(a: tuple[float, float], b: tuple[float, float]) -> list[float]:
    vertices = [x * y for x in a for y in b]
    return [min(vertices), max(vertices)]


def binary_weight_bounds(
    contrast_usd: float,
    *,
    contrast_interval_usd: Sequence[float] | None = None,
    equity_pricing_weight_interval: Sequence[float] = (0.0, 1.0),
) -> dict:
    """Return sharp bounds *relative to the declared rectangular input set*.

    The bounds are marginal: YES and NO extremes may occur at different pi and
    Delta.  This is not a confidence interval and cannot certify coverage.
    """
    delta = _number(contrast_usd, "contrast_usd")
    d = _range(contrast_interval_usd if contrast_interval_usd is not None else (delta, delta),
               "contrast_interval_usd")
    if not d[0] <= delta <= d[1]:
        raise ValueError("Point contrast must be inside declared contrast interval")
    pi = _range(equity_pricing_weight_interval, "equity_pricing_weight_interval", unit=True)
    yes_interval = _product_bounds((1-pi[1], 1-pi[0]), d)
    no_interval = _product_bounds((-pi[1], -pi[0]), d)
    return {
        "schema": SCHEMA, "status": "DECLARED_TWO_STATE_PARTIAL_IDENTIFICATION",
        "equity_pricing_weight_identified": False,
        "equity_pricing_weight_interval": list(pi), "contrast_usd": delta,
        "contrast_interval_usd": list(d),
        "yes_pnl_usd_range": yes_interval, "no_pnl_usd_range": no_interval,
        "point_grid": [{"assumed_pi": p, "yes_pnl_usd": (1-p)*delta,
                        "no_pnl_usd": -p*delta} for p in GRID],
        "statement": "Sharp marginal vertex bounds conditional on the independent, explicitly declared pi and contrast rectangles; inputs are assumptions, not empirical coverage claims.",
        "assumptions": [
            "Exactly two mutually exclusive settlement states at one horizon and a common discounted-value decomposition for the full original portfolio.",
            "Delta is a declared YES-minus-NO same-horizon portfolio value contrast, not an inferred event-causal equity move.",
            "The effective equity state-pricing weight pi lies in the declared range; it is not measured from the event-contract quote Q.",
            "No risk premia, time evolution, discounting differences or concurrent news are identified by this decomposition.",
        ],
        "cannot_infer": ["physical_event_probability", "equity_pricing_weight",
                         "future_event_return_accuracy", "causal_event_effect"],
    }


def admitted_binary_contract_sensitivity(event: Mapping) -> dict | None:
    """Fail closed on source-specific, complete, Q-centered binary stress cards.

    Market Q is checked solely to verify the already stored *assumed pi=Q*
    display. It is not used to identify pi or tighten the robust bounds.
    """
    if (not isinstance(event, Mapping) or event.get("source_defined_scenario") is not True
            or (event.get("quality") or {}).get("model_kind") != "DECLARED_CONTRACT_SCENARIO"):
        return None
    rows = event.get("outcomes")
    if not isinstance(rows, list) or len(rows) != 2:
        return None
    by_polarity = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        key = str(row.get("id") or "")
        suffix = key.rsplit(":", 1)[-1]
        if suffix not in {"YES", "NO"} or suffix in by_polarity:
            return None
        by_polarity[suffix] = row
    if set(by_polarity) != {"YES", "NO"}:
        return None
    y, n = by_polarity["YES"], by_polarity["NO"]
    if (not y.get("contract_id") or y.get("contract_id") != n.get("contract_id")
            or str(y.get("id")) != str(y["contract_id"]) + ":YES"
            or str(n.get("id")) != str(n["contract_id"]) + ":NO"):
        return None
    try:
        q, complement = _number(y.get("q_reference"), "q_yes"), _number(n.get("q_reference"), "q_no")
        yes, no = _number(y.get("pnl"), "yes_pnl"), _number(n.get("pnl"), "no_pnl")
        yq, nq = _range(y.get("q_interval"), "yes_quote", unit=True), _range(n.get("q_interval"), "no_quote", unit=True)
    except ValueError:
        return None
    delta = yes - no
    scale = max(1.0, abs(delta))
    if (not 0 < q < 1 or abs(q + complement - 1) > 1e-8
            or not yq[0] <= q <= yq[1] or not nq[0] <= complement <= nq[1]
            or abs(yq[0] + nq[1] - 1) > 1e-8 or abs(yq[1] + nq[0] - 1) > 1e-8
            or abs(q * yes + complement * no) > 1e-7 * scale):
        return None
    result = binary_weight_bounds(delta)
    result.update(contract_id=str(y["contract_id"]), venue=str(y.get("venue") or ""),
                  contract_quote_q_reference=q, quote_is_equity_weight=False,
                  stored_yes_pnl_under_assumed_pi_equals_q=yes,
                  stored_no_pnl_under_assumed_pi_equals_q=no,
                  original_contract_event_only=True)
    return result
