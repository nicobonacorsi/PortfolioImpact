"""Source-bound FOMC announcement-component scenarios, with explicit transfer assumptions.

This module does not estimate physical probabilities, conditional total returns,
alpha, or portfolio allocations. It joins the existing reviewed contract/Q engine
and issuer-duration evidence to published historical *surprise* sensitivities.
Unknown outcome magnitudes, instrument transfer and market-to-asset expectation
transfer remain visible assumptions. No provider or network call is made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Sequence

from scipy.optimize import linprog

from .adapters import Market
from .contract_questions import source_defined_yes_question
from .event_impact import load_duration_evidence, MAX_DURATION_AGE_DAYS
from .fed_outcome_pricing import evaluate_fed_outcome_quotes
from .portfolio import Portfolio, _finite_number
from .workflow import load_real_snapshot

REVIEW_AS_OF = "2026-10-21T04:00:00+00:00"
DEFAULT_CONTRACT = "KXFEDDECISION-26OCT-H25"
OTHER = "OTHER_UNLISTED_OUTCOMES"
EQUITY_REFERENCE = {
    "title": "Bernanke and Kuttner, What Explains the Stock Market's Reaction to Federal Reserve Policy? (2004 working paper)",
    "url": "https://www.federalreserve.gov/pubs/feds/2004/200416/200416pap.pdf",
    "locator": "Table 2, column (b), printed page 40; robustness column (d)",
    "source_estimate": -4.68,
    "source_unit": "percentage points of one-day CRSP value-weighted return per +100bp surprise in the policy target",
    "fractional_return_per_policy_bp": -0.000468,
    "robustness_excluding_outliers_fractional_return_per_policy_bp": -0.000255,
    "sample": "June 1989 to December 2002; 131 event observations, 125 after excluding six outliers",
    "transfer_status": "Historical broad-equity response; applying it to 2026 SPY/QQQ/NVDA is an explicit proxy assumption, not a fitted instrument coefficient.",
}
YIELD_REFERENCE = {
    "title": "Kuttner, Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market (2000)",
    "url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr99.pdf",
    "locator": "Table 3, 30-year row, printed page 11; sample described in Table 1",
    "source_estimate": 19.4,
    "source_unit": "basis points of one-day 30-year on-the-run Treasury yield response per +100bp policy surprise",
    "yield_bps_per_policy_bp": 0.194,
    "sample": "42 target-rate changes, June 6 1989 to February 2 2000",
    "transfer_status": "Transferring a historical 30-year yield coefficient to a parallel shift of TLT's relevant long-yield curve in 2026 is a scenario assumption.",
}


class EconomicBridgeError(ValueError):
    """An invalid assumption or source cannot enter a numerical scenario."""


def _number(value: Any, label: str) -> float:
    try:
        return _finite_number(value, label)
    except ValueError as exc:
        raise EconomicBridgeError(str(exc)) from exc


def _interval(value: Any, point: float, label: str, limit: float) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EconomicBridgeError(f"{label} requires exactly two finite endpoints")
    low, high = (_number(x, label) for x in value)
    if not -limit <= low <= point <= high <= limit:
        raise EconomicBridgeError(f"{label} must contain its point and stay within +/-{limit:g}")
    return low, high


@dataclass(frozen=True)
class FomcEconomicAssumptions:
    """Deliberate model settings; ranges are stress domains, never confidence bands.

    The symmetric coefficient ranges deliberately include no transmission and
    sign reversal. Their endpoints are chosen scenarios, not fitted uncertainty.
    Unknown contract magnitudes are not inferred from garbled source rules.
    """
    opaque_policy_change_bps: float = 0.0
    opaque_policy_change_range_bps: tuple[float, float] = (-100.0, 100.0)
    equity_return_per_policy_bp: float = -0.000468
    equity_return_per_policy_bp_range: tuple[float, float] = (-0.000468, 0.000468)
    long_yield_bps_per_policy_bp: float = 0.194
    long_yield_bps_per_policy_bp_range: tuple[float, float] = (-0.194, 0.194)

    def __post_init__(self) -> None:
        for name, limit in (("opaque_policy_change_bps", 100.0),
                            ("equity_return_per_policy_bp", 0.005),
                            ("long_yield_bps_per_policy_bp", 1.0)):
            point = _number(getattr(self, name), name)
            bounds_name = name + "_range"
            # Public field names put the unit suffix last for policy magnitude.
            if name == "opaque_policy_change_bps":
                bounds_name = "opaque_policy_change_range_bps"
            bounds = _interval(getattr(self, bounds_name), point, bounds_name, limit)
            object.__setattr__(self, name, point)
            object.__setattr__(self, bounds_name, bounds)


def _stamp(value: str | datetime) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise EconomicBridgeError("A valid timezone-aware analysis timestamp is required") from exc
    if result.utcoffset() is None:
        raise EconomicBridgeError("A timezone-aware analysis timestamp is required")
    return result.astimezone(timezone.utc)


def _product(first: Sequence[float], second: Sequence[float]) -> list[float]:
    values = [a * b for a in first for b in second]
    return [min(values), max(values)]


def _linear_bounds(coefficients_low: Sequence[float], coefficients_high: Sequence[float],
                   constraints: Sequence[tuple[float, float]]) -> dict[str, Any]:
    n = len(constraints)
    settings = {"A_eq": [[1.0] * n], "b_eq": [1.0], "bounds": constraints, "method": "highs"}
    lower = linprog(list(coefficients_low), **settings)
    upper = linprog([-x for x in coefficients_high], **settings)
    if not (lower.success and upper.success):
        raise EconomicBridgeError("Source-compatible Q simplex is infeasible")
    for witness in (lower.x, upper.x):
        if abs(math.fsum(witness) - 1.0) > 1e-8 or any(
                not low - 1e-8 <= x <= high + 1e-8
                for x, (low, high) in zip(witness, constraints)):
            raise EconomicBridgeError("Economic LP returned an invalid source-compatible witness")
    return {"range": [float(lower.fun), float(-upper.fun)],
            "lower_witness": [float(x) for x in lower.x],
            "upper_witness": [float(x) for x in upper.x]}


def build_fomc_economic_case(
    portfolio: Portfolio, *, root: Path | str | None = None,
    selected_contract_id: str = DEFAULT_CONTRACT,
    analysis_as_of: str | datetime = REVIEW_AS_OF,
    assumptions: FomcEconomicAssumptions | None = None,
    markets: Sequence[Market] | None = None,
) -> dict[str, Any]:
    """Model one exact FOMC contract's YES and its full NO mixture, offline.

    The default review time is deliberately frozen and returned prominently.
    It is an historical-reference scenario using archived quotes, not a current
    forecast. Every source must have been available by the requested review
    time, which must precede the announcement. The default source archive is
    read from this module's repository; ``root`` locates issuer evidence only.
    """
    if not isinstance(portfolio, Portfolio):
        raise EconomicBridgeError("A validated original-position Portfolio is required")
    settings = FomcEconomicAssumptions() if assumptions is None else assumptions
    if not isinstance(settings, FomcEconomicAssumptions):
        raise EconomicBridgeError("Use validated FomcEconomicAssumptions")
    now = _stamp(analysis_as_of)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "UNAVAILABLE",
        "basis": "HISTORICAL_SOURCE_BOUND_FOMC_ANNOUNCEMENT_COMPONENT_MODEL",
        "analysis_as_of": now.isoformat(),
        "live_or_current_forecast": False,
        "selected_contract_id": selected_contract_id,
        "portfolio_notional_usd": portfolio.notional,
        "horizon": "Same FOMC announcement-day repricing component under frozen reference expectations; not a return from the quote/review date to the meeting or to December.",
        "physical_probability_P": None,
        "physical_expected_portfolio_pnl": None,
        "future_empirical_transfer_validated": False,
        "model_settings": asdict(settings),
        "market_probability": None,
        "reference_expectation": None,
        "branches": [],
        "sources": [dict(EQUITY_REFERENCE), dict(YIELD_REFERENCE)],
        "assumptions": [
            "The archived Kalshi single-Q contract weights are used as an explicit proxy for the reference policy expectations embedded in the asset-price component. Neither equality with asset pricing weights nor equality with physical P is established.",
            "Reference weights and state mappings are frozen from the archived quote time until immediately before the announcement; intervening news is excluded. These old quotes are not today's expectation estimate.",
            "The historical papers use one-day surprises measured from fed-funds futures. Substituting the modeled outcome minus the archived-Q expectation and transferring the historical coefficients to 2026 are assumptions.",
            "Only the modeled surprise component is reported. Intercepts, carry, other news, information effects, within-state risk, nonlinearities, transaction costs and full holding-period returns are excluded.",
            "NVDA, QQQ and SPY use one common broad-equity proxy slope. No instrument beta or technology multiplier is estimated or invented; individual and sector responses can differ substantially.",
            "TLT uses verified dated issuer duration with a historical 30-year Treasury surprise slope transferred to a parallel long-yield move. Curve shape, convexity and term-premium effects are omitted.",
            "Opaque numeric contracts and unlisted OTHER outcomes retain all their feasible mass. Their representative policy change and bounded magnitude are explicit model assumptions, not source-certified outcomes.",
            "NO is the conditional mixture of every other outcome under the same feasible reference Q, including opaque and unlisted mass. It is not an opposite decision or a claim that a hike did not affect prices.",
            "All P&L is computed on original positions once. ETF constituents inform exposure discovery and are never added a second time to P&L.",
            "Zero Q-weighted modeled component follows from centering by construction. It is not empirical validation, a no-arbitrage test on observed asset prices, or zero physical expected return.",
            "The quote/opaque-magnitude envelope and coefficient stress envelope are different objects. The combined enclosure includes deliberately sign-reversed transmission; it has no statistical coverage guarantee and does not bound all future returns.",
        ],
    }

    def unavailable(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    source_markets = list(load_real_snapshot("oct26")["markets"] if markets is None else markets)
    try:
        pricing = evaluate_fed_outcome_quotes(source_markets, as_of=now)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        return unavailable("Original contract source/quote verification failed: " + str(exc))
    quote_time = _stamp(pricing["as_of_source_receipt_utc"])
    if now < quote_time:
        return unavailable("The archived market source was not available by the requested analysis time")
    by_id = {market.ticker: market for market in source_markets}
    if selected_contract_id not in by_id:
        return unavailable("The selected contract is absent from the reviewed co-observed market set")
    questions = {key: source_defined_yes_question(value) for key, value in by_id.items()}
    predicate = questions[selected_contract_id]["typed_predicate"]
    if not predicate or predicate.get("metric") != "fed_target_rate_change" or predicate.get("operator") != "eq":
        return unavailable("Selected contract has no independently certified exact policy-change magnitude; numeric impact is unavailable")
    meeting_day = datetime.fromisoformat(predicate["period"]).date()
    if not 1 <= (meeting_day - now.date()).days <= 90:
        return unavailable("This frozen-expectation scenario requires a future announcement within 90 days of the declared review time")
    result["event"] = {
        "date": meeting_day.isoformat(), "question": questions[selected_contract_id]["yes_question"],
        "definition": "THIS_EXACT_CONTRACT_SETTLES_YES_UNDER_FULL_SOURCE_RULES",
        "numeric_predicate": predicate,
        "numeric_review_status": questions[selected_contract_id]["numeric_rule_status"],
        "rules_sha256": questions[selected_contract_id]["verified_rules_sha256"],
        "ordinary_announcement_assumption": "Policy-change mappings are used as a rate-news component; cancellation treated as zero under reviewed terms is not modeled as an ordinary announcement with unchanged other news.",
    }
    atoms = pricing["source_outcome_bounds_with_lp_witnesses"]
    labels = [a["source_outcome_id"] for a in atoms]
    limits = [(q["lower"], q["upper"]) for q in pricing["source_contract_quote_constraints"]] + [(0.0, 1.0)]
    endpoints = [a[key] for a in atoms for key in ("lower_attaining_joint_witness", "upper_attaining_joint_witness")]
    # Convexity preserves every quote interval and the simplex without dividing
    # incoherent quote midpoints by their sum or deleting residual mass.
    reference_q = [math.fsum(w[i] for w in endpoints) / len(endpoints) for i in range(len(labels))]
    if abs(math.fsum(reference_q) - 1.0) > 1e-8 or any(
            not low - 1e-8 <= p <= high + 1e-8 for p, (low, high) in zip(reference_q, limits)):
        raise EconomicBridgeError("The reference Q is not feasible")
    selected_index = labels.index(selected_contract_id)
    q_yes = reference_q[selected_index]
    q_interval = list(atoms[selected_index]["pricing_measure_Q_interval"])
    if not 0 < q_yes < 1 or q_interval[1] >= 1:
        return unavailable("Both YES and NO need positive reference mass and a defined conditional NO envelope")
    outcomes = []
    changes, lows, highs, opaque = [], [], [], []
    for index, label in enumerate(labels):
        typed = questions[label]["typed_predicate"] if label in questions else None
        exact = bool(typed and typed.get("metric") == "fed_target_rate_change" and typed.get("operator") == "eq")
        if exact:
            value = _number(typed["threshold"], "Source policy-change magnitude")
            if abs(value) > 100:
                return unavailable("Exact policy magnitude exceeds the reviewed +/-100bp scenario domain")
            low = high = value
        else:
            value = settings.opaque_policy_change_bps
            low, high = settings.opaque_policy_change_range_bps
        changes.append(value); lows.append(low); highs.append(high); opaque.append(0.0 if exact else 1.0)
        outcomes.append({
            "source_outcome_id": label, "reference_Q_weight": reference_q[index],
            "source_Q_interval": atoms[index]["pricing_measure_Q_interval"],
            "numeric_mapping_basis": "SOURCE_CERTIFIED_EXACT_CHANGE" if exact else "OPAQUE_MAGNITUDE_EXPLICIT_SCENARIO_ASSUMPTION",
            "policy_change_bps": value, "policy_change_range_bps": [low, high],
            "numeric_predicate": typed if exact else None,
        })
    mean = math.fsum(p * change for p, change in zip(reference_q, changes))
    mean_bounds = _linear_bounds(lows, highs, limits)
    unknown_mass = _linear_bounds(opaque, opaque, limits)
    selected_change = changes[selected_index]
    yes_surprise = selected_change - mean
    yes_range = [selected_change - mean_bounds["range"][1], selected_change - mean_bounds["range"][0]]
    no_mean = math.fsum(reference_q[i] * changes[i] for i in range(len(labels)) if i != selected_index) / (1.0 - q_yes)
    no_surprise = no_mean - mean
    # The q ratio and YES enclosure may be dependent; interval multiplication
    # deliberately returns a conservative enclosure, never a sharp claim.
    ratios = [-q_interval[1] / (1.0 - q_interval[1]), -q_interval[0] / (1.0 - q_interval[0])]
    no_range = _product(ratios, yes_range)
    result["market_probability"] = {
        "status": "HISTORICAL_MARKET_IMPLIED_Q_REFERENCE_NOT_PHYSICAL_PROBABILITY",
        "contract_yes_Q_interval": q_interval,
        "quote_as_of_utc": pricing["as_of_source_receipt_utc"],
        "source_response_sha256": pricing["original_local_source_sha256"],
        "source_url": source_markets[0].metadata.get("source_url", "https://external-api.kalshi.com/trade-api/v2/events/" + source_markets[0].event_ticker),
        "exchange_quote_update_time_verified": False,
        "physical_probability_P": None,
        "current_quote_available": False,
    }
    result["reference_expectation"] = {
        "mean_policy_change_bps": mean,
        "mean_policy_change_range_bps": mean_bounds["range"],
        "mean_lower_witness": mean_bounds["lower_witness"],
        "mean_upper_witness": mean_bounds["upper_witness"],
        "mean_lower_outcome_policy_changes_bps": lows,
        "mean_upper_outcome_policy_changes_bps": highs,
        "selected_reference_Q_weight": q_yes,
        "witness_selection": "Arithmetic mean of all validated lower/upper marginal LP witnesses; feasible illustrative reference, not a probability point estimate or a normalized midpoint.",
        "outcomes": outcomes,
        "opaque_reference_mass": math.fsum(p * flag for p, flag in zip(reference_q, opaque)),
        "opaque_mass_Q_interval": unknown_mass["range"],
        "NO_mixture": [{"source_outcome_id": labels[i], "conditional_reference_Q_weight": reference_q[i] / (1.0 - q_yes), "policy_change_bps": changes[i]} for i in range(len(labels)) if i != selected_index],
        "range_interpretation": "Mean bounds are LP-sharp for the stated Q and opaque-magnitude model. Branch NO and combined sensitivity ranges are conservative enclosures, not confidence or predictive intervals.",
    }
    duration = None
    duration_reason = "No TLT position"
    if any(p.symbol == "TLT" and p.asset_type == "etf" and p.weight > 0 for p in portfolio.positions):
        try:
            duration = load_duration_evidence(root)
            age = (now.date() - datetime.fromisoformat(duration.data_as_of).date()).days
            if _stamp(duration.retrieved_at) > now or not 0 <= age <= MAX_DURATION_AGE_DAYS:
                duration_reason = "Issuer duration is future-dated or older than the reviewed 30-day scope"
                duration = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            duration_reason = "Issuer duration unavailable: " + str(exc)
    if duration is not None:
        result["sources"].append(duration.to_dict())
    max_surprise = max(abs(x) for x in (*yes_range, *no_range))
    if duration is not None and max_surprise * max(abs(x) for x in settings.long_yield_bps_per_policy_bp_range) > 50.0:
        duration = None
        duration_reason = "Assumed long-yield shift exceeds the reviewed first-order +/-50bp duration domain"
    rows = []
    for position in portfolio.positions:
        coefficient = coefficient_range = None
        basis = "UNMODELED_INSTRUMENT"
        reason = "No reviewed economic proxy for this symbol and asset type"
        if position.weight == 0 or position.asset_type == "cash":
            coefficient, coefficient_range = 0.0, [0.0, 0.0]
            basis = "ZERO_WEIGHT" if position.weight == 0 else "USD_CASH_PRICE_COMPONENT_ZERO_BY_MODEL"
            reason = "No event price component; cash carry is outside this announcement-component model"
        elif ((position.symbol == "NVDA" and position.asset_type == "stock") or
              (position.symbol in {"QQQ", "SPY"} and position.asset_type == "etf")):
            coefficient = settings.equity_return_per_policy_bp
            coefficient_range = list(settings.equity_return_per_policy_bp_range)
            basis = "HISTORICAL_BROAD_EQUITY_PROXY_TRANSFER_ASSUMPTION"
            reason = "Published broad-equity surprise slope; not a fitted sensitivity of this instrument"
        elif position.symbol == "TLT" and position.asset_type == "etf":
            if duration is not None:
                coefficient = -duration.effective_duration_years * settings.long_yield_bps_per_policy_bp / 10000.0
                coefficient_range = sorted(-duration.effective_duration_years * x / 10000.0 for x in settings.long_yield_bps_per_policy_bp_range)
                basis = "ISSUER_DURATION_WITH_HISTORICAL_LONG_YIELD_TRANSFER_ASSUMPTION"
                reason = "Verified issuer duration; the event-to-long-yield mapping is a historical proxy assumption"
            else:
                reason = duration_reason
        rows.append({"symbol": position.symbol, "asset_type": position.asset_type,
                     "original_weight": position.weight, "position_value_usd": portfolio.notional * position.weight,
                     "model_basis": basis, "reason": reason,
                     "return_coefficient_per_policy_bp": coefficient,
                     "return_coefficient_stress_range": coefficient_range})
    unmodeled_weight = math.fsum(row["original_weight"] for row in rows if row["return_coefficient_per_policy_bp"] is None)
    modeled_weight = math.fsum(row["original_weight"] for row in rows if row["return_coefficient_per_policy_bp"] is not None)
    if modeled_weight == 0:
        result.update(modeled_weight=0.0, unmodeled_weight=unmodeled_weight, position_models=rows)
        return unavailable("No positive-weight original position has a supported economic model")
    total_coefficient = math.fsum(row["position_value_usd"] * row["return_coefficient_per_policy_bp"] for row in rows if row["return_coefficient_per_policy_bp"] is not None)
    total_coefficient_range = [math.fsum(row["position_value_usd"] * row["return_coefficient_stress_range"][i] for row in rows if row["return_coefficient_per_policy_bp"] is not None) for i in (0, 1)]
    branches = []
    for state, mean_change, surprise, surprise_range in (("YES", selected_change, yes_surprise, yes_range), ("NO", no_mean, no_surprise, no_range)):
        contributions = []
        for row in rows:
            coefficient = row["return_coefficient_per_policy_bp"]
            supported = coefficient is not None
            quote_range = _product([coefficient, coefficient], surprise_range) if supported else None
            all_range = _product(row["return_coefficient_stress_range"], surprise_range) if supported else None
            component = coefficient * surprise if supported else None
            if component is not None and (component < -1 or all_range[0] < -1):
                raise EconomicBridgeError("Scenario exceeds the reviewed unlevered arithmetic-return domain")
            contributions.append({**row,
                "return_component": component,
                "pnl_component_usd": row["position_value_usd"] * component if supported else None,
                "quote_and_opaque_magnitude_pnl_range_usd": [row["position_value_usd"] * x for x in quote_range] if supported else None,
                "combined_model_sensitivity_pnl_range_usd": [row["position_value_usd"] * x for x in all_range] if supported else None})
        pnl = total_coefficient * surprise
        quote_range = _product([total_coefficient, total_coefficient], surprise_range)
        sensitivity_range = _product(total_coefficient_range, surprise_range)
        branches.append({
            "state": state, "conditional_policy_change_bps": mean_change,
            "policy_surprise_bps": surprise, "policy_surprise_range_bps": surprise_range,
            "known_pnl_component_usd": pnl,
            "known_quote_and_opaque_magnitude_pnl_range_usd": quote_range,
            "known_combined_model_sensitivity_pnl_range_usd": sensitivity_range,
            "total_pnl_component_usd": pnl if unmodeled_weight == 0 else None,
            "total_combined_model_sensitivity_pnl_range_usd": sensitivity_range if unmodeled_weight == 0 else None,
            "contributions": contributions,
        })
    centered = q_yes * branches[0]["known_pnl_component_usd"] + (1 - q_yes) * branches[1]["known_pnl_component_usd"]
    result.update(
        status="AVAILABLE_CONDITIONAL_ECONOMIC_SCENARIO" if unmodeled_weight == 0 else "PARTIAL_CONDITIONAL_ECONOMIC_SCENARIO",
        reason="Published historical surprise mechanism plus visible reference-expectation, instrument-transfer and residual-outcome assumptions; not a calibrated future forecast.",
        modeled_weight=modeled_weight, unmodeled_weight=unmodeled_weight,
        coverage_status="ALL_ORIGINAL_POSITIONS_HAVE_SCENARIO_COMPONENTS" if unmodeled_weight == 0 else "KNOWN_SUBTOTAL_ONLY_UNMODELED_POSITIONS_RETAINED",
        branches=branches,
        reference_Q_weighted_component_usd=centered,
        consistency_check_interpretation="Zero by construction of the centered component, not empirical validation or a statement about physical expected P&L.",
    )
    return result
