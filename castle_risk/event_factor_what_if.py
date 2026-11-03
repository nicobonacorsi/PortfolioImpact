"""Event-linked *assumption* stress, with an explicit unknown-event-effect gap.

This module answers a useful but deliberately narrower question: if an
originally discovered event were accompanied by independently *assumed*
equity, curve and gold shocks, what would the dated original holdings do?
It does not claim that the event will cause those shocks.  A source quote,
Claude proposal, or daily SPY beta cannot identify a real conditional return.

The complete branch reuses the existing sharp affine/polytope stress engine.
The partial branch reports a known-factor *subtotal*, never a zero-price
assumption for missing holdings.  Failed out-of-sample stock betas do not
enter the conservative branch.  A clearly labeled exploratory variant can
be requested independently; it never upgrades the market-price model.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any, Mapping

from .event_defined_robust_stress import evaluate_original_event_robust_stress
from .issuer_factor_evidence import review_issuer_factor
from .portfolio import Portfolio


_FACTORS = ("SPY_INDEX_RETURN", "DURATION_ONE_BOND_PRICE_RETURN", "GOLD_SPOT_RETURN")
# Source-free, declared stress domains for simultaneous factors.  These are
# not empirically calibrated conditional event shocks or prediction bands.
_REGIMES = {
    "BROAD_RISK_OFF": {
        "SPY_INDEX_RETURN": (-3.0, -1.0),
        "DURATION_ONE_BOND_PRICE_RETURN": (-0.25, 0.10),  # yields +25 to -10 bp
        "GOLD_SPOT_RETURN": (0.0, 5.0),
    },
    "RATE_SPIKE": {
        "SPY_INDEX_RETURN": (-2.0, 0.0),
        "DURATION_ONE_BOND_PRICE_RETURN": (-0.40, -0.10), # yields +40 to +10 bp
        "GOLD_SPOT_RETURN": (-2.0, 3.0),
    },
    "FLIGHT_TO_SAFETY": {
        "SPY_INDEX_RETURN": (-4.0, -2.0),
        "DURATION_ONE_BOND_PRICE_RETURN": (0.10, 0.40),  # yields -10 to -40 bp
        "GOLD_SPOT_RETURN": (2.0, 6.0),
    },
}
_MECHANISM_LIMITATIONS = {
    "HORMUZ_ENERGY": "Oil-supply shock to crude prices, company margins and cash flows is not estimated.",
    "OPEC_SUPPLY": "Oil prices to each producer/utility cash-flow channel are not identified.",
    "RUSSIA_ENERGY_SANCTIONS": "Sanctions to energy prices, quantities and firm-level cash flows are not identified.",
    "US_DRUG_PRICING": "Government reimbursement rules to insurer/pharma cash flows are not identified.",
    "FDA_DRUG_APPROVAL": "Approval outcomes to drug revenues and issuer equity values are not identified.",
    "BANK_CREDIT_STRESS": "Credit losses, loan volumes and interchange economics are not identified.",
    "BANK_CAPITAL_RULES": "Regulatory capital rules to bank valuations are not identified.",
    "TRADE_TARIFFS": "Country/product-specific tariff changes to issuer margins are not identified by generic factors.",
    "EXPORT_CONTROLS": "Restricted chip volumes and substitution to issuer margins are not identified.",
    "TAIWAN_SUPPLY": "Taiwan manufacturing disruption to issuer production and supplier cash flows is not identified.",
}


def _now(value: datetime | str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("A valid ISO as-of timestamp is required") from exc
    if parsed.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return parsed.astimezone(timezone.utc)


def _finite_beta(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _review_beta(symbol: str, beta_report: Mapping[str, Any], as_of: datetime) -> dict[str, Any]:
    """Return only a reviewed historical SPY association, not a causal effect."""
    failed = {"symbol": symbol, "status": "UNAVAILABLE", "basis": "NO_ADMISSIBLE_HISTORICAL_SPY_ASSOCIATION"}
    if beta_report.get("kind") != "INSTRUMENT_DAILY_BETAS" or beta_report.get("benchmark_symbol") != "SPY":
        return failed
    try:
        observed = _now(beta_report["analysis_as_of"])
    except (KeyError, TypeError, ValueError):
        return failed
    if observed > as_of:
        return {**failed, "reason": "SPY beta was measured after the stress as-of"}
    fit = (beta_report.get("instruments") or {}).get(symbol) or {}
    value = fit.get("beta")
    interval = fit.get("beta_interval")
    if (not _finite_beta(value) or not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(not _finite_beta(v) for v in interval) or not interval[0] <= value <= interval[1]):
        return failed
    transfer = (fit.get("transfer_check") or {}).get("status", "NOT_TESTED")
    return {"symbol": symbol, "status": "ESTIMATED_HISTORICAL_ASSOCIATION", "beta": float(value),
            "beta_interval": [float(v) for v in interval], "holdout_status": transfer,
            "analysis_as_of": observed.isoformat(),
            "basis": "FROZEN_SPY_DAILY_BETA_ASSOCIATION_NOT_CAUSAL_EVENT_RESPONSE"}


def _factor_rows(regime: str) -> list[dict]:
    if regime not in _REGIMES:
        raise ValueError("Unknown declared stress regime")
    details = {
        "SPY_INDEX_RETURN": "Assumed broad U.S. equity-index return, percentage points; not caused by the selected event in the model.",
        "DURATION_ONE_BOND_PRICE_RETURN": "Assumed duration-one bond price move; minus the assumed parallel yield shift in percentage points, not a Fed decision.",
        "GOLD_SPOT_RETURN": "Assumed gold-spot price change, percentage points; not inferred from the selected event.",
    }
    return [{"name": name, "return_range_pct": list(_REGIMES[regime][name]),
             "assumption_basis": "EXPLICIT_PREDECLARED_OPERATOR_WHAT_IF_NOT_OBSERVED_EVENT_RESPONSE",
             "economic_channel": details[name]} for name in _FACTORS]


def _partial_known_interval(portfolio: Portfolio, factors: list[dict], responses: dict) -> list[float] | None:
    """Exact bounds for the known factor subtotal over an independent box."""
    if not responses:
        return None
    low = high = 0.0
    for factor in factors:
        coefficient = math.fsum(p.weight * responses[p.symbol]["factor_loadings"].get(factor["name"], 0.)
                                for p in portfolio.positions if p.symbol in responses)
        left, right = factor["return_range_pct"]
        low += min(coefficient * left, coefficient * right)
        high += max(coefficient * left, coefficient * right)
    constant = math.fsum(p.weight * responses[p.symbol]["fixed_return_pct"]
                         for p in portfolio.positions if p.symbol in responses)
    return [portfolio.notional * (low + constant) / 100.,
            portfolio.notional * (high + constant) / 100.]


def build_factor_what_if(portfolio: Portfolio, event: Mapping[str, Any],
                         daily_beta_report: Mapping[str, Any],
                         instrument_metadata: Mapping[str, Any], *,
                         evaluated_at: datetime | str,
                         allow_failed_beta_as_explicit_assumption: bool = False,
                         regimes: tuple[str, ...] = tuple(_REGIMES)) -> dict[str, Any]:
    """Attach optional what-if calculations to an original discovered event.

    The usual economic admission and ranking remain untouched.  A failed OOS
    beta is absent from the conservative version; allowing it is an explicit
    experimental stress *assumption*, with the failed test visible in output.
    """
    if not isinstance(portfolio, Portfolio):
        raise ValueError("Original portfolio object required")
    now = _now(evaluated_at)
    identifier = str(event.get("event_id") or "").strip()
    question = str(event.get("question_we_care_about") or event.get("description")
                   or event.get("title") or "").strip()
    if not identifier or not question:
        raise ValueError("Original event identity and question required")
    if not isinstance(allow_failed_beta_as_explicit_assumption, bool):
        raise ValueError("Explicit failed-beta assumption must be boolean")
    if not regimes or len(set(regimes)) != len(regimes):
        raise ValueError("At least one nonrepeated stress regime is required")
    known: dict[str, dict] = {}
    instruments: list[dict[str, Any]] = []
    issuer_evidence: list[dict[str, Any]] = []
    for position in portfolio.positions:
        if not position.weight:
            continue
        symbol, meta = position.symbol, (instrument_metadata or {}).get(position.symbol) or {}
        entry = {"symbol": symbol, "original_weight": position.weight,
                 "original_asset_type": position.asset_type}
        loadings: dict[str, float] | None = None
        basis = None
        if position.asset_type == "cash":
            loadings, basis = {}, "DECLARED_USD_CASH_INSTANTANEOUS_ZERO_PRICE_MOVE"
        elif position.asset_type == "etf" and symbol in {"AGG", "BND", "GLD"}:
            fact = review_issuer_factor(symbol, meta, now)
            issuer_evidence.append(fact)
            if fact["status"] == "ISSUER_FACT_AND_PROVIDER_IDENTITY_REVIEWED":
                if fact["kind"].startswith("BOND_"):
                    loadings = {"DURATION_ONE_BOND_PRICE_RETURN": fact["duration_years"]}
                    basis = "ORIGINAL_FUND_ISSUER_DURATION_TIMES_ASSUMED_PARALLEL_YIELD_MOVE"
                else:
                    loadings = {"GOLD_SPOT_RETURN": fact["spot_loading"]}
                    basis = "OFFICIAL_GOLD_TRUST_OBJECTIVE_WITH_ASSUMED_SPOT_TRACKING"
            else:
                entry["reason"] = fact["reason"]
        elif (position.asset_type == "stock" and meta.get("ticker") == symbol
              and meta.get("type") in ("CS", "ADRC") and meta.get("active") is True):
            fit = _review_beta(symbol, daily_beta_report, now)
            entry["historical_beta_review"] = fit
            if fit["status"] == "ESTIMATED_HISTORICAL_ASSOCIATION":
                if fit["holdout_status"] == "PASS":
                    loadings = {"SPY_INDEX_RETURN": fit["beta"]}
                    basis = "HISTORICAL_SPY_BETA_HOLDOUT_PASS_NOT_CAUSAL_EVENT_EFFECT"
                elif allow_failed_beta_as_explicit_assumption:
                    loadings = {"SPY_INDEX_RETURN": fit["beta"]}
                    basis = "EXPLICIT_EXPLORATORY_FAILED_OR_UNTESTED_SPY_BETA_ASSUMPTION"
                    entry["failed_beta_explicitly_assumed"] = True
                else:
                    entry["reason"] = "Historical SPY beta did not pass the chronological holdout."
            else:
                entry["reason"] = "No valid measured SPY loading for this equity position."
        else:
            entry["reason"] = "No approved original-position factor response; ETF composition is not a zero-return proxy."
        if loadings is not None:
            known[symbol] = {"fixed_return_pct": 0.0, "factor_loadings": loadings,
                             "response_basis": basis}
            entry.update(status="KNOWN_ASSUMED_FACTOR_RESPONSE", factor_loadings=deepcopy(loadings),
                         response_basis=basis)
        else:
            entry["status"] = "UNMODELED_OR_INVALID_FACTOR_RESPONSE"
        instruments.append(entry)
    active = [p for p in portfolio.positions if p.weight]
    covered = math.fsum(p.weight for p in active if p.symbol in known)
    uncovered = math.fsum(p.weight for p in active if p.symbol not in known)
    complete = not any(p.symbol not in known for p in active)
    cases = []
    for regime in regimes:
        factors = _factor_rows(regime)
        if complete:
            exact = evaluate_original_event_robust_stress(
                portfolio_rows=portfolio.to_dict()["positions"], notional=portfolio.notional,
                original_event_id=identifier, original_event_question=question,
                common_return_horizon="Instantaneous repricing at the hypothetical shock, no carry or terminal prediction",
                factors=factors, position_responses=known)
            cases.append({"regime": regime, "status": "COMPLETE_ARITHMETIC_ONLY_UNDER_DECLARED_FACTORS",
                          "sharp_assumed_factor_pnl_range_usd": exact["sharp_assumption_conditional_pnl_range_usd"],
                          "full_portfolio_arithmetic_under_factor_only_assumptions": True,
                          "factor_shock_assumptions": factors,
                          "sharp_lp_certificate": exact})
        else:
            cases.append({"regime": regime, "status": "PARTIAL_KNOWN_FACTOR_CONTRIBUTION_ONLY",
                          "known_position_pnl_range_usd": _partial_known_interval(portfolio, factors, known),
                          "full_portfolio_pnl_range_usd": None,
                          "full_portfolio_arithmetic_under_factor_only_assumptions": False,
                          "factor_shock_assumptions": factors,
                          "unmodeled_weight": uncovered})
    return {
        "schema": "castle.event_factor_what_if.v1",
        "status": "COMPLETE_FACTOR_ONLY_WHAT_IF_NOT_EVENT_EFFECT" if complete else "PARTIAL_FACTOR_ONLY_WHAT_IF_UNKNOWN_REMAINDER",
        "event_id": identifier,
        "original_question": question,
        "source_as_of": now.isoformat(),
        "scenario_condition": "IF_SELECTED_EVENT_AND_EXPLICIT_UNMEASURED_MARKET_FACTOR_SHOCKS",
        "original_position_response_coverage_weight": covered,
        "unmodeled_original_position_weight": uncovered,
        "positions": instruments,
        "issuer_documentary_factor_evidence": issuer_evidence,
        "assumption_stress_regimes": cases,
        "failed_historical_betas_can_only_enter_by_explicit_assumption": True,
        "failed_historical_beta_assumed": allow_failed_beta_as_explicit_assumption,
        "source_quote_used_as_event_probability": False,
        "event_to_factor_shocks_identified": False,
        "direct_event_to_issuer_cash_flows_identified": False,
        "identified_event_conditional_portfolio_pnl": None,
        "original_event_probability_P": None,
        "prediction_market_probability_Q": None,
        "already_priced_in_component": None,
        "economic_ranking_allowed": False,
        "primary_economic_admission_allowed": False,
        "missing_direct_event_mechanism": _MECHANISM_LIMITATIONS.get(
            str(event.get("family") or ""), "The event-specific effect on issuer cash flows and market factors is not estimated."),
        "interpretation": (
            "Conditional-on-assumed-factor response surface, not conditional-on-event price discovery. "
            "Documented fund duration and gold objective are first-order exposure facts. "
            "All factor shock intervals are operator hypotheses, not confidence or prediction intervals. "
            "Unmeasured direct issuer effects remain unknown even if factor-only arithmetic covers the full portfolio."
        ),
    }
