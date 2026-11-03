"""Auditable user-declared portfolio scenarios, independent of language models.

There are two different valuations:

* ``today``: a declared change from current position prices. No priced-in
  adjustment is made because it would subtract expectations twice.
* ``binary_gap``: a *difference* between event and no-event values. A separately
  supplied equity-pricing weight q defines P0 = q*V_yes + (1-q)*V_no. This
  algebra is conditional on a two-state pricing assumption, not an estimate
  of an equity pricing weight or a physical event probability.

Factor returns use jointly fitted SPY/rate coefficients if available. Explicit
position-return inputs override factors for that original holding; an ETF's
underlying constituents are never added to its original NAV a second time.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re


SCHEMA = "castle.scenario_lab.v1"
MODEL_SCHEMA = "castle.scenario_lab.model.v1"
FACTOR_UNITS = {"SPY_RETURN_PCT": "percent", "US_10Y_YIELD_CHANGE_BP": "basis_points"}
MAX_TEXT = 480
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,11}\Z")


class ScenarioInputError(ValueError):
    """A user-facing invalid assumption; no values should be silently fixed."""


def _real(value, *, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ScenarioInputError(f"{label} must be a finite number.")
    return float(value)


def interpret_hypothesis(value):
    """Identify only an economic topic, never invent an event-to-return shock."""
    if not isinstance(value, str):
        raise ScenarioInputError("Describe the event in plain language.")
    text = " ".join(value.split())
    if not 3 <= len(text) <= MAX_TEXT or any(ord(c) < 32 for c in text):
        raise ScenarioInputError("Use a short, readable event description (up to 480 characters).")
    low = text.casefold()
    categories = (
        ("INFLATION", r"\b(cpi|inflation|consumer prices?)\b", ["SPY_RETURN_PCT", "US_10Y_YIELD_CHANGE_BP"]),
        ("RATES", r"\b(fomc|federal reserve|fed rate|interest rates?|treasur(?:y|ies)|bond yields?)\b",
         ["US_10Y_YIELD_CHANGE_BP", "SPY_RETURN_PCT"]),
        ("LABOR", r"\b(payrolls?|jobs report|unemployment|employment|nonfarm)\b",
         ["SPY_RETURN_PCT", "US_10Y_YIELD_CHANGE_BP"]),
        ("TRADE", r"\b(tariffs?|import duties|trade war|sanctions?)\b", ["SPY_RETURN_PCT"]),
        ("ENERGY", r"\b(oil|crude|gas prices?|opec|energy supply)\b", ["SPY_RETURN_PCT"]),
    )
    family, factors = "OTHER", []
    for name, pattern, candidates in categories:
        if re.search(pattern, low):
            family, factors = name, candidates
            break
    claims = []
    for match in re.finditer(r"(?<![\w.])[+-]?\d{1,4}(?:\.\d{1,3})?\s*(?:percentage points?|pp|basis points?|bps?|%)(?!\w)", low):
        claims.append(match.group(0).strip())
        if len(claims) == 8:
            break
    return {"description": text, "topic": family, "suggested_factors": factors,
            "user_stated_magnitudes": claims, "event_to_market_shock_identified": False,
            "interpretation": "Text organizes a hypothesis; it is not evidence for a price move."}


def public_factor_model(report, positions, *, notional=None):
    """Minimal, receipt-free projection of the separately admitted factor model."""
    model = report.get("scenario_response_model") or {}
    if model.get("schema") != "castle.scenario_response_model.v1" or model.get("status") != "COMPUTED":
        return {"schema": MODEL_SCHEMA, "status": "UNAVAILABLE", "positions": [], "scales": {}}
    if notional is not None and (
        isinstance(model.get("notional"), bool) or
        not isinstance(model.get("notional"), (int, float)) or
        not math.isfinite(model["notional"]) or
        abs(model["notional"] - notional) > .01
    ):
        return {"schema": MODEL_SCHEMA, "status": "UNAVAILABLE", "positions": [], "scales": {}}
    originals = {row["symbol"]: row for row in positions}
    projected = []
    for row in model.get("positions") or []:
        symbol = row.get("symbol")
        if symbol not in originals or row.get("status") not in {"AVAILABLE", "UNAVAILABLE"}:
            continue
        coeffs = {}
        if row["status"] == "AVAILABLE":
            for factor in FACTOR_UNITS:
                v = (row.get("coefficients") or {}).get(factor)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    coeffs[factor] = float(v)
        projected.append({"symbol": symbol, "status": row["status"], "coefficients": coeffs,
                          "basis": str(row.get("basis") or "UNAVAILABLE"),
                          "training_span": deepcopy(row.get("training_span")),
                          "holdout_span": deepcopy(row.get("holdout_span")),
                          "holdout_oos_r2": row.get("holdout_oos_r2") if isinstance(row.get("holdout_oos_r2"), (int, float))
                          and math.isfinite(row["holdout_oos_r2"]) else None})
    scales = {}
    for factor in FACTOR_UNITS:
        info = (model.get("scales") or {}).get(factor) or {}
        v = info.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
            scales[factor] = {"value": float(v), "unit": FACTOR_UNITS[factor],
                              "training_span": deepcopy(info.get("training_span")),
                              "basis": "ONE_TRAINING_DAILY_STANDARD_DEVIATION_NOT_EVENT_SEVERITY"}
    return {"schema": MODEL_SCHEMA, "status": "COMPUTED", "as_of": model.get("as_of"),
            "horizon": "ONE_TRADING_DAY", "positions": projected, "scales": scales,
            "scope": "Joint historical factor associations; not event effects or proven forecasts."}


def _positions(case):
    if not isinstance(case, Mapping):
        raise ScenarioInputError("A valid portfolio analysis is required.")
    notional = _real(case.get("notional"), label="Portfolio value")
    if not 0 < notional <= 1_000_000_000:
        raise ScenarioInputError("Portfolio value is outside the supported range.")
    entries = case.get("positions")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 24:
        raise ScenarioInputError("A portfolio with 1 to 24 original holdings is required.")
    symbols, weights, positions = set(), [], []
    for row in entries:
        if not isinstance(row, Mapping):
            raise ScenarioInputError("An original holding is malformed.")
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol) or symbol in symbols:
            raise ScenarioInputError("Holdings must have distinct verified ticker symbols.")
        weight = _real(row.get("weight"), label="Original holding weight")
        if not 0 < weight <= 1:
            raise ScenarioInputError("Each original holding needs a positive portfolio weight.")
        symbols.add(symbol)
        weights.append(weight)
        positions.append({"symbol": symbol, "weight": weight,
                          "asset_type": row.get("type") or row.get("asset_type") or "stock"})
    if abs(math.fsum(weights) - 1) > 1e-8:
        raise ScenarioInputError("Original holdings do not reconcile to 100% of portfolio value.")
    return notional, positions


def _shocks(value, allowed, *, label, low, high):
    if not isinstance(value, Mapping) or len(value) > len(allowed):
        raise ScenarioInputError(f"{label} must name supported exposures only.")
    result = {}
    for k, v in value.items():
        if not isinstance(k, str) or k not in allowed:
            raise ScenarioInputError(f"{label} contains an unsupported name.")
        x = _real(v, label=f"{k} shock")
        minimum, maximum = low[k] if isinstance(low, Mapping) else low, high[k] if isinstance(high, Mapping) else high
        if not minimum <= x <= maximum:
            raise ScenarioInputError(f"{k} shock is outside the supported stress range.")
        result[k] = x
    return result


def evaluate_scenario(case, request):
    """Deterministic P&L with missing exposure and pricing assumptions visible."""
    if not isinstance(request, Mapping) or set(request) - {
        "hypothesis", "mode", "position_shocks", "factor_shocks", "pricing_weight", "pricing_weight_range"
    }:
        raise ScenarioInputError("Unrecognized scenario fields.")
    interpretation = interpret_hypothesis(request.get("hypothesis"))
    notional, positions = _positions(case)
    original = {p["symbol"]: p for p in positions}
    direct = _shocks(request.get("position_shocks", {}), original, label="Position returns", low=-100, high=500)
    factors = _shocks(request.get("factor_shocks", {}), FACTOR_UNITS, label="Factor shocks",
                     low={"SPY_RETURN_PCT": -100, "US_10Y_YIELD_CHANGE_BP": -1000},
                     high={"SPY_RETURN_PCT": 300, "US_10Y_YIELD_CHANGE_BP": 1000})
    if not direct and not factors:
        raise ScenarioInputError("Enter a return assumption for a holding or a factor shock.")
    mode = request.get("mode", "today")
    if mode not in {"today", "binary_gap"}:
        raise ScenarioInputError("Choose a supported pricing interpretation.")
    if mode == "today":
        if request.get("pricing_weight") is not None or request.get("pricing_weight_range") is not None:
            raise ScenarioInputError("Already-priced-in weights cannot be applied twice: use an event-vs-no-event gap, not a move from today's price.")
        q = q_range = None
    else:
        q = _real(request.get("pricing_weight"), label="Equity pricing weight")
        if not 0 <= q <= 1:
            raise ScenarioInputError("The equity pricing weight must be between zero and one.")
        q_range = request.get("pricing_weight_range", [q, q])
        if not isinstance(q_range, list) or len(q_range) != 2:
            raise ScenarioInputError("The pricing range must have two endpoints.")
        q_range = [_real(v, label="Pricing-range endpoint") for v in q_range]
        if not 0 <= q_range[0] <= q <= q_range[1] <= 1:
            raise ScenarioInputError("The pricing range must include the selected weight and stay in [0,1].")
    model = case.get("scenario_lab_model") or {}
    by_symbol = {r.get("symbol"): r for r in model.get("positions") or [] if isinstance(r, Mapping)}
    if factors and (model.get("schema") != MODEL_SCHEMA or model.get("status") != "COMPUTED"):
        raise ScenarioInputError("This portfolio has no admitted dated factor responses. Use explicit holding returns.")
    rows, covered, known_gap, yes_pnl, no_pnl = [], [], [], [], []
    for pos in positions:
        symbol, weight = pos["symbol"], pos["weight"]
        value = notional * weight
        source = by_symbol.get(symbol) or {}
        if symbol in direct:
            if pos["asset_type"] == "cash" and direct[symbol] != 0:
                raise ScenarioInputError("USD cash is held fixed; give it a zero return.")
            ret, basis = direct[symbol] / 100, "USER_DECLARED_ORIGINAL_HOLDING_RETURN"
        elif pos["asset_type"] == "cash":
            ret, basis = 0., "USD_CASH_HELD_FIXED"
        elif factors and source.get("status") == "AVAILABLE" and all(
            isinstance((source.get("coefficients") or {}).get(factor), (int, float))
            and math.isfinite(source["coefficients"][factor]) for factor in factors
        ):
            ret = math.fsum(source["coefficients"][factor] * shock for factor, shock in factors.items())
            basis = str(source.get("basis") or "JOINT_HISTORICAL_FACTOR_ASSOCIATION")
        else:
            ret, basis = None, "NO_DECLARED_POSITION_RETURN_OR_ADMITTED_FACTOR_RESPONSE"
        if ret is not None and (not math.isfinite(ret) or ret < -1):
            raise ScenarioInputError(f"The {symbol} stress implies an invalid negative asset price.")
        if ret is not None and mode == "binary_gap":
            # Non-negative prices must hold throughout the stated q sensitivity.
            if any((1 - z) * ret < -1 or -z * ret < -1 for z in q_range):
                raise ScenarioInputError(f"The {symbol} event gap implies a negative state value at this pricing weight.")
        modeled = ret is not None
        if modeled:
            covered.append(weight)
            known_gap.append(value * ret)
            if mode == "binary_gap":
                yes_pnl.append(value * (1 - q) * ret)
                no_pnl.append(-value * q * ret)
        rows.append({"symbol": symbol, "weight": weight, "original_value_usd": value,
                     "return_gap_pct": ret * 100 if modeled else None,
                     "pnl_usd": value * ret if modeled and mode == "today" else None,
                     "yes_pnl_usd": value * (1 - q) * ret if modeled and mode == "binary_gap" else None,
                     "no_pnl_usd": -value * q * ret if modeled and mode == "binary_gap" else None,
                     "basis": basis, "modeled": modeled, "training_span": source.get("training_span") if basis == source.get("basis") else None,
                     "holdout_span": source.get("holdout_span") if basis == source.get("basis") else None})
    covered_weight = math.fsum(covered)
    complete = len(covered) == len(positions)
    gap = math.fsum(known_gap)
    yes_total = math.fsum(yes_pnl) if mode == "binary_gap" else None
    no_total = math.fsum(no_pnl) if mode == "binary_gap" else None
    sensitivity = None
    if mode == "binary_gap":
        lo, hi = q_range
        sensitivity = {
            "yes_pnl_usd": sorted([(1 - lo) * gap, (1 - hi) * gap]),
            "no_pnl_usd": sorted([-lo * gap, -hi * gap]),
            "interpretation": "Endpoints across the user-declared equity pricing-weight range; not confidence limits.",
        }
    return {"schema": SCHEMA, "case_id": case.get("id"), "capture": case.get("capture"),
            "notional": notional, "interpretation": interpretation, "mode": mode,
            "assumptions": {"position_return_pct": direct, "factor_shocks": factors,
                            "horizon": "ONE_TRADING_DAY_FACTOR_SHOCK" if factors else "USER_DECLARED_POSITION_RETURN_HORIZON",
                            "position_shocks_override_factor_model": True},
            "pricing": {"equity_weight_q": q, "equity_weight_range": q_range,
                        "prediction_market_quote_used_as_equity_weight": False,
                        "q_empirically_identified": False,
                        "identity": "P0=q*V_yes+(1-q)*V_no (two-state model, other premia held fixed)" if mode == "binary_gap" else None},
            "covered_weight": covered_weight, "unmodeled_weight": max(0., 1 - covered_weight),
            "complete": complete, "positions": rows,
            "known_change_usd": gap if mode == "today" else None,
            "portfolio_change_usd": gap if mode == "today" and complete else None,
            "known_event_gap_usd": gap if mode == "binary_gap" else None,
            "yes_pnl_usd": yes_total if mode == "binary_gap" and complete else None,
            "no_pnl_usd": no_total if mode == "binary_gap" and complete else None,
            "known_yes_pnl_usd": yes_total, "known_no_pnl_usd": no_total,
            "q_sensitivity": sensitivity, "event_effect_estimated": False,
            "financial_forecast_validated": False,
            "scope": "Original-capital arithmetic for user-specified moves or historical factor association; no identified event severity."}
