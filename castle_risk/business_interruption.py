"""Source-anchored, explicitly conditional business-interruption valuation.

This is a deterministic cash-flow mechanism, not a fitted stock-return model.
The caller supplies a separately verified revenue anchor and *declares* which
fraction of that run rate is interrupted, its contribution margin, catch-up,
tax shield, timing and discount rate.  No operating margin is silently used as
a contribution margin.  No broad rail-strike contract is equated to an outage
at this issuer, and no provider quote supplies an asset-pricing probability.

For a d-calendar-day interruption, each missed shipment's baseline contribution
is paid at the midpoint of its original day. A fraction c is instead earned L
days later.  When c can be positive, L >= d ensures all recovered shipments
occur after the interruption. With a single declared contribution margin m,
affected revenue fraction a, daily revenue R and annual effective discount r,

  loss(d) = (1-t) [R*a*m*sum_j DF(start+j+1/2)*(1-c*DF(L))
                    + K*DF(start+d)].

K is an explicitly supplied additional restart/penalty cost, or is EXCLUDED
from this component; exclusion is not an estimate of zero total extra cost.
Tax shield t is an assumed immediately usable cash shield on the entire
specified component, not a statutory tax rate inferred from a filing.

The five rectangular domains a,m,c,t,K have mathematically sharp extrema:
all factors are nonnegative and their monotonicity is known for fixed timing
and r>=0. Endpoint witnesses retain shared inputs rather than separately
minimizing lost and recovered revenue. These are assumption-set bounds, not
confidence bounds, prediction intervals or guaranteed bounds on actual loss.

The optional equity normalization holds the market value denominator fixed.
It models only this cash-flow component, excluding changes in discount rates,
terminal value, wages, financing, general-market news and other channels. A
provided equity embedded weight q gives immediate valuation changes on learning
the scenario: YES=-(1-q)*loss and NO=q*loss. q is explicitly an assumption,
never a prediction-market Q, a physical P or an estimated asset-price weight.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
import json
import math
import re


SCHEMA = "castle.business_interruption.v1"
MAX_DURATION_DAYS = 366
MAX_TIMING_DAYS = 3650
MAX_CASES = 32
_DOMAINS = ("affected_fraction", "contribution_margin", "catchup_fraction", "tax_shield_fraction")


class BusinessInterruptionError(ValueError):
    """Malformed observations or undeclared economic assumptions."""


def _real(value, name, *, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessInterruptionError(name + " must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise BusinessInterruptionError(name + " must be finite") from exc
    if not math.isfinite(result):
        raise BusinessInterruptionError(name + " must be finite")
    if lower is not None and result < lower or upper is not None and result > upper:
        raise BusinessInterruptionError(name + " is outside its declared numeric domain")
    return result


def _text(value, name, maximum=400):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise BusinessInterruptionError(name + " requires bounded nonempty text")
    return value.strip()


def _symbol(value, name):
    value = _text(value, name, 32)
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^/-]{0,31}", value):
        raise BusinessInterruptionError(name + " must be a canonical instrument symbol")
    return value


def _day(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise BusinessInterruptionError(name + " must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BusinessInterruptionError(name + " is not a valid calendar date") from exc


def _instant(value, name):
    if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)", value):
        raise BusinessInterruptionError(name + " must be a timezone-aware ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None:
            raise ValueError
        return result
    except (ValueError, OverflowError) as exc:
        raise BusinessInterruptionError(name + " is not a valid timestamp") from exc


def _integer(value, name, *, lower=0, upper=MAX_TIMING_DAYS):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise BusinessInterruptionError(f"{name} must be an integer in [{lower},{upper}]")
    return value


def _domain(value, name, *, unit_interval=True):
    if not isinstance(value, Mapping) or set(value) - {"lower", "upper", "illustrative_point"}:
        raise BusinessInterruptionError(name + " requires lower/upper and an optional illustrative_point")
    lo = _real(value.get("lower"), name + " lower", lower=0, upper=1 if unit_interval else None)
    hi = _real(value.get("upper"), name + " upper", lower=0, upper=1 if unit_interval else None)
    if lo > hi:
        raise BusinessInterruptionError(name + " lower exceeds upper")
    point = value.get("illustrative_point")
    if point is not None:
        point = _real(point, name + " illustrative point", lower=lo, upper=hi)
        point_basis = "EXPLICIT_ILLUSTRATIVE_ASSUMPTION_NOT_ESTIMATE"
    elif lo == hi:
        point, point_basis = lo, "SINGLETON_DECLARED_ASSUMPTION"
    else:
        point_basis = "NO_POINT_SELECTED_FROM_ASSUMPTION_INTERVAL"
    return {"lower": lo, "upper": hi, "illustrative_point": point, "point_basis": point_basis}


def _anchor(value):
    if not isinstance(value, Mapping):
        raise BusinessInterruptionError("Revenue anchor must be a mapping")
    issuer = _symbol(value.get("issuer"), "Anchor issuer")
    start, end = (_day(value.get(key), key) for key in ("period_start", "period_end"))
    if start > end:
        raise BusinessInterruptionError("Revenue period is reversed")
    revenue = _real(value.get("revenue_usd"), "Reported period revenue", lower=0)
    days = (end - start).days + 1
    return {"issuer": issuer, "revenue_usd": revenue,
            "revenue_metric": _text(value.get("revenue_metric"), "Revenue metric"),
            "period_start": start.isoformat(), "period_end": end.isoformat(),
            "calendar_days_in_reported_period": days,
            "flat_daily_revenue_usd": revenue / days,
            "source_record_id": _text(value.get("source_record_id"), "Revenue source record", 1000),
            "source_authority": "CALLER_SUPPLIED_ANCHOR_REQUIRES_SEPARATE_ORIGINAL_SOURCE_VERIFICATION",
            "run_rate_basis": "DECLARED_FLAT_CALENDAR_DAY_RUN_RATE_NOT_A_SEASONAL_OR_FORWARD_REVENUE_ESTIMATE"}


def _assumptions(value, durations):
    if not isinstance(value, Mapping):
        raise BusinessInterruptionError("Explicit economic assumptions are required")
    domains = {key: _domain(value.get(key), key) for key in _DOMAINS}
    timing = {key: _integer(value.get(key), key) for key in ("start_delay_days", "catchup_delay_days")}
    if domains["catchup_fraction"]["upper"] > 0 and timing["catchup_delay_days"] < max(durations):
        raise BusinessInterruptionError("Each recovered shipment must occur after restart: catchup_delay_days must cover every tested duration")
    rate = _real(value.get("annual_discount_rate"), "Annual effective discount rate", lower=0)
    mode = value.get("additional_cost_mode")
    if mode == "DECLARED_ADDITIONAL_COST":
        domains["restart_penalty_cost_usd"] = _domain(value.get("restart_penalty_cost_usd"),
                                                      "restart_penalty_cost_usd", unit_interval=False)
    elif mode == "EXCLUDED_FROM_COMPONENT":
        if value.get("restart_penalty_cost_usd") is not None:
            raise BusinessInterruptionError("Excluded additional costs cannot also have a numeric assumption")
        # This mathematical zero applies only to the enumerated component.
        # The output explicitly preserves additional total costs as unmodeled.
        domains["restart_penalty_cost_usd"] = {
            "lower": 0., "upper": 0., "illustrative_point": 0.,
            "point_basis": "EXCLUDED_FROM_COMPONENT_NOT_ESTIMATED_ZERO_TOTAL_COST"}
    else:
        raise BusinessInterruptionError("Additional costs must be declared or explicitly excluded from the component")
    return {**domains, **timing, "annual_discount_rate": rate, "additional_cost_mode": mode,
            "calendar_day_discount_basis": 365,
            "additional_cost_payment_time": "END_OF_INTERRUPTION",
            "revenue_payment_time": "MIDPOINT_OF_EACH_INTERRUPTED_CALENDAR_DAY",
            "catchup_timing": "EACH_ORIGINAL_SHIPMENT_DELAYED_BY_THE_SAME_DECLARED_NUMBER_OF_CALENDAR_DAYS",
            "catchup_contribution_margin": "SAME_AS_BASELINE_DECLARED_MARGIN; INCREMENTAL_RESTART_OR_CATCHUP_COSTS_REQUIRE_SEPARATE_INPUT",
            "tax_shield_timing": "SAME_TIME_AS_EACH_COMPONENT_CASH_FLOW; IMMEDIATE_USABILITY_ASSUMED"}


def _exposure(value, issuer):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise BusinessInterruptionError("Original portfolio exposure must be a mapping")
    notional = _real(value.get("notional_usd"), "Portfolio notional", lower=0)
    if notional == 0:
        raise BusinessInterruptionError("Portfolio notional must be positive")
    raw = value.get("positions")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 500:
        raise BusinessInterruptionError("Between 1 and 500 original portfolio positions are required")
    positions, seen = [], set()
    for original in raw:
        if not isinstance(original, Mapping):
            raise BusinessInterruptionError("Each original portfolio position must be a mapping")
        symbol = _symbol(original.get("symbol"), "Position symbol")
        if symbol in seen:
            raise BusinessInterruptionError("Duplicate original position; ETF constituents must not be added as synthetic positions")
        seen.add(symbol)
        weight = _real(original.get("weight"), "Position weight", lower=0, upper=1)
        asset_type = original.get("asset_type")
        if asset_type not in {"stock", "etf", "cash"}:
            raise BusinessInterruptionError("Unsupported original instrument type")
        fraction = original.get("issuer_fraction")
        unresolved = original.get("unresolved_issuer_fraction", 0.)
        ref = None
        if asset_type == "etf":
            if symbol == issuer:
                raise BusinessInterruptionError("Issuer identity contradicts the declared fund type")
            if fraction is not None:
                fraction = _real(fraction, "ETF issuer fraction", lower=0, upper=1)
                if weight and "unresolved_issuer_fraction" not in original:
                    raise BusinessInterruptionError("A positive-weight fund requires an explicit unresolved issuer fraction; omission cannot certify complete exposure")
                unresolved = _real(unresolved, "Unresolved ETF issuer fraction", lower=0, upper=1)
                if fraction + unresolved > 1 + 1e-12:
                    raise BusinessInterruptionError("Known and unresolved fund NAV fractions exceed the original fund")
                if weight:
                    ref = _text(original.get("holdings_source_record_id"), "Fund holdings source record", 1000)
            else:
                if "unresolved_issuer_fraction" in original and _real(unresolved,
                        "Unresolved ETF issuer fraction", lower=0, upper=1) != 1.:
                    raise BusinessInterruptionError("A wholly unknown fund fraction cannot claim a smaller unresolved remainder")
                unresolved = 1.
        else:
            expected = 1. if asset_type == "stock" and symbol == issuer else 0.
            if symbol == issuer and asset_type != "stock":
                raise BusinessInterruptionError("Issuer identity contradicts a cash classification")
            if fraction is None or _real(fraction, "Non-fund issuer fraction", lower=0, upper=1) != expected:
                raise BusinessInterruptionError("Direct-stock and cash fractions must agree with the declared issuer identity")
            if _real(unresolved, "Unresolved non-fund issuer fraction", lower=0, upper=1) != 0.:
                raise BusinessInterruptionError("A verified direct identity cannot also have an unresolved fund fraction")
            fraction = expected
        positions.append({"symbol": symbol, "asset_type": asset_type, "weight": weight,
                          "issuer_fraction": fraction,
                          "unresolved_issuer_fraction": unresolved,
                          "issuer_portfolio_weight": weight * fraction if fraction is not None else None,
                          "unresolved_issuer_portfolio_weight": weight * unresolved,
                          "holdings_source_record_id": ref})
    if not math.isclose(math.fsum(row["weight"] for row in positions), 1., rel_tol=0, abs_tol=1e-10):
        raise BusinessInterruptionError("Original weights must sum to one; no exposure renormalization")
    known = math.fsum(row["issuer_portfolio_weight"] for row in positions if row["issuer_portfolio_weight"] is not None)
    unresolved = math.fsum(row["unresolved_issuer_portfolio_weight"] for row in positions)
    return {"notional_usd": notional, "issuer": issuer, "original_positions": positions,
            "known_issuer_portfolio_weight": known,
            "unresolved_fund_portfolio_weight": unresolved,
            "issuer_portfolio_weight_outer_range": [known, known + unresolved],
            "complete_issuer_exposure": unresolved == 0,
            "scope": "ORIGINAL_POSITIONS_ONLY_DIRECT_COMMON_EQUITY_PLUS_FUND_COMMON_EQUITY_ISSUER_SLEEVES_COUNTED_ONCE",
            "source_authority": "CALLER_SUPPLIED_IDENTITY_AND_HOLDINGS_REQUIRE_SEPARATE_VERIFICATION"}


def _equity_value(value, issuer):
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("issuer") != issuer:
        raise BusinessInterruptionError("Equity market value must identify the same issuer")
    capitalization = _real(value.get("market_cap_usd"), "Equity market capitalization", lower=0)
    if capitalization == 0:
        raise BusinessInterruptionError("Equity market capitalization must be positive")
    observed = _instant(value.get("as_of"), "Equity market-value as_of")
    return {"issuer": issuer, "market_cap_usd": capitalization, "as_of": observed.isoformat(),
            "source_record_id": _text(value.get("source_record_id"), "Equity value source record", 1000),
            "scope": "FIXED_PROVIDED_EQUITY_VALUE_DENOMINATOR_NOT_REVALIDATED_OR_FRESHNESS_CERTIFIED"}


def _df(days, rate):
    return math.exp(-math.log1p(rate) * (days / 365.))


def _loss(daily_revenue, duration, params, values):
    rate, start = params["annual_discount_rate"], params["start_delay_days"]
    discounted_days = math.fsum(_df(start + day + .5, rate) for day in range(duration))
    recovery_discount = _df(params["catchup_delay_days"], rate)
    before_catchup = daily_revenue * values["affected_fraction"] * values["contribution_margin"] * discounted_days
    recovered = before_catchup * values["catchup_fraction"] * recovery_discount
    additional = values["restart_penalty_cost_usd"] * _df(start + duration, rate)
    shield_multiplier = 1. - values["tax_shield_fraction"]
    # Keep an arbitrarily small but positive delay cost when c=1 and the rate
    # is small. Subtracting two large, rounded PV numbers would erase it.
    delayed_fraction_loss = -math.expm1(-math.log1p(rate) * (params["catchup_delay_days"] / 365.))
    lost_fraction = (1. - values["catchup_fraction"]) + values["catchup_fraction"] * delayed_fraction_loss
    contribution_loss = before_catchup * lost_fraction
    loss = shield_multiplier * (contribution_loss + additional)
    components = {"baseline_contribution_pv_usd": before_catchup,
                  "recovered_contribution_pv_usd": recovered,
                  "net_contribution_loss_pv_usd": contribution_loss,
                  "declared_additional_cost_pv_usd": additional,
                  "immediately_usable_tax_shield_pv_usd": (contribution_loss + additional) * values["tax_shield_fraction"],
                  "after_tax_cashflow_component_loss_pv_usd": loss}
    if not all(math.isfinite(number) for number in components.values()) or loss < 0:
        raise BusinessInterruptionError("Cash-flow arithmetic overflowed or violated the nonnegative-loss mechanism")
    return components


def _corners(params):
    low, high = {}, {}
    for key in (*_DOMAINS, "restart_penalty_cost_usd"):
        reversed_sign = key in {"catchup_fraction", "tax_shield_fraction"}
        low[key] = params[key]["upper" if reversed_sign else "lower"]
        high[key] = params[key]["lower" if reversed_sign else "upper"]
    return low, high


def _point(params):
    values = {key: params[key]["illustrative_point"] for key in (*_DOMAINS, "restart_penalty_cost_usd")}
    return None if any(value is None for value in values.values()) else values


def _scaled_interval(interval, scale):
    result = sorted(scale * value for value in interval)
    if not all(math.isfinite(value) for value in result):
        raise BusinessInterruptionError("Valuation normalization overflowed")
    return result


def _normalization(loss_interval, exposure, equity):
    if equity is None:
        return {"status": "EQUITY_VALUE_DENOMINATOR_NOT_PROVIDED",
                "issuer_equity_value_contrast_fraction": None,
                "known_issuer_portfolio_component_contrast_usd": None,
                "complete_issuer_component_contrast_usd": None, "original_position_components": []}
    if loss_interval[1] > equity["market_cap_usd"]:
        return {"status": "FIRST_ORDER_COMPONENT_EXCEEDS_EQUITY_VALUE_CAPITAL_STRUCTURE_NOT_MODELED",
                "issuer_equity_value_contrast_fraction": None,
                "known_issuer_portfolio_component_contrast_usd": None,
                "complete_issuer_component_contrast_usd": None, "original_position_components": []}
    fractional = _scaled_interval(loss_interval, -1 / equity["market_cap_usd"])
    rows, known, complete = [], None, None
    if exposure is not None:
        for row in exposure["original_positions"]:
            dollars = (None if row["issuer_portfolio_weight"] is None else
                       _scaled_interval(fractional, exposure["notional_usd"] * row["issuer_portfolio_weight"]))
            rows.append({"symbol": row["symbol"], "issuer_portfolio_weight": row["issuer_portfolio_weight"],
                         "issuer_component_contrast_usd": dollars})
        known = _scaled_interval(fractional, exposure["notional_usd"] * exposure["known_issuer_portfolio_weight"])
        complete = known if exposure["complete_issuer_exposure"] else None
    return {"status": "FIXED_DENOMINATOR_CASHFLOW_COMPONENT_ONLY",
            "issuer_equity_value_contrast_fraction": fractional,
            "known_issuer_portfolio_component_contrast_usd": known,
            "complete_issuer_component_contrast_usd": complete,
            "original_position_components": rows,
            "total_issuer_stock_return_forecast": None, "total_portfolio_return_forecast": None}


def evaluate_business_interruption(anchor, *, duration_days, assumptions,
                                   exposure=None, equity_value=None, equity_pricing_weights=None):
    """Evaluate an explicitly declared interruption mechanism without API calls.

    ``anchor`` contains issuer, revenue_usd, period_start/end, revenue_metric
    and an opaque source_record_id. It is not itself proof of original source
    authenticity. ``assumptions`` supplies four domains {lower, upper,
    illustrative_point?}, fixed start/catch-up timing, effective annual rate,
    and an explicit extra-cost inclusion/exclusion choice. No midpoint, tax
    rate, margin, catch-up amount or pricing weight is chosen by this function.

    Optional exposure includes the *original* portfolio and each original
    fund's separately reviewed common-equity issuer fraction. A positive-weight
    fund with a numeric issuer fraction must also explicitly declare its
    ``unresolved_issuer_fraction`` (including explicit zero for complete source
    coverage); this preserves its unclassified remaining NAV
    without erasing a separately observed issuer slice. A missing fund fraction
    means the whole fund is unknown. These fractions must not count the issuer's
    bonds or synthetic leveraged exposures as common equity. All output remains
    a company cash-flow component even if
    the portfolio consists solely of the affected issuer.
    """
    if not isinstance(duration_days, list) or not 1 <= len(duration_days) <= MAX_CASES:
        raise BusinessInterruptionError("A bounded, explicit duration scenario list is required")
    durations = [_integer(value, "Interruption duration", lower=1, upper=MAX_DURATION_DAYS) for value in duration_days]
    if durations != sorted(set(durations)):
        raise BusinessInterruptionError("Duration scenarios must be distinct and increasing")
    source = _anchor(anchor)
    params = _assumptions(assumptions, durations)
    original_exposure = _exposure(exposure, source["issuer"])
    denominator = _equity_value(equity_value, source["issuer"])
    if denominator and _day(source["period_end"], "Revenue period end") > _instant(denominator["as_of"], "Equity valuation time").date():
        raise BusinessInterruptionError("Revenue reporting period cannot end after the valuation date")
    if equity_pricing_weights is None:
        pricing = []
    else:
        if not isinstance(equity_pricing_weights, list) or not 1 <= len(equity_pricing_weights) <= 11:
            raise BusinessInterruptionError("Explicit equity-pricing weight scenarios must be a bounded list")
        pricing = [_real(q, "Declared equity embedded weight", lower=0, upper=1) for q in equity_pricing_weights]
        if len(set(pricing)) != len(pricing):
            raise BusinessInterruptionError("Duplicate equity-pricing assumptions")
    corners = _corners(params)
    point_inputs = _point(params)
    results = []
    for duration in durations:
        low, high = [_loss(source["flat_daily_revenue_usd"], duration, params, values) for values in corners]
        interval = [low["after_tax_cashflow_component_loss_pv_usd"], high["after_tax_cashflow_component_loss_pv_usd"]]
        if interval[0] > interval[1]:
            raise BusinessInterruptionError("The sharp conditional envelope is reversed")
        point = (_loss(source["flat_daily_revenue_usd"], duration, params, point_inputs)
                 if point_inputs is not None else None)
        if point and not interval[0] - 1e-8 <= point["after_tax_cashflow_component_loss_pv_usd"] <= interval[1] + 1e-8:
            raise BusinessInterruptionError("Illustrative point lies outside its declared envelope")
        normalized = _normalization(interval, original_exposure, denominator)
        capital_structure_blocked = (normalized["status"] ==
            "FIRST_ORDER_COMPONENT_EXCEEDS_EQUITY_VALUE_CAPITAL_STRUCTURE_NOT_MODELED")
        priced = []
        for q in pricing:
            company_yes = None if capital_structure_blocked else _scaled_interval(interval, -(1 - q))
            company_no = None if capital_structure_blocked else _scaled_interval(interval, q)
            known = normalized["known_issuer_portfolio_component_contrast_usd"]
            priced.append({"assumed_equity_embedded_weight": q,
                "probability_basis": "EXPLICIT_ASSET_PRICE_SCENARIO_WEIGHT_NOT_ESTIMATED_P_OR_MARKET_Q",
                "equity_projection_status": (normalized["status"] if capital_structure_blocked else
                    "DECLARED_DOLLAR_FOR_DOLLAR_CASHFLOW_COMPONENT_EQUITY_PROJECTION"),
                "company_yes_immediate_valuation_change_usd": company_yes,
                "company_no_immediate_valuation_change_usd": company_no,
                "known_issuer_portfolio_yes_component_usd": None if known is None else _scaled_interval(known, 1 - q),
                "known_issuer_portfolio_no_component_usd": None if known is None else _scaled_interval(known, -q),
                "same_underlying_cashflow_loss_in_yes_and_no": True,
                "zero_weighted_valuation_change_identity": (None if capital_structure_blocked else
                    "q*[-(1-q)*L]+(1-q)*[q*L]=0 FOR THE SAME L; DO NOT MIX DIFFERENT BOUND ENDPOINTS")})
        results.append({"duration_calendar_days": duration,
            "company_after_tax_cashflow_component_loss_pv_usd": interval,
            "company_yes_minus_no_equity_value_component_usd": (None if capital_structure_blocked else
                [-interval[1], -interval[0]]),
            "company_no_interruption_counterfactual_component_usd": 0.,
            "minimum_witness": {"assumptions": dict(corners[0]), "cashflows": low},
            "maximum_witness": {"assumptions": dict(corners[1]), "cashflows": high},
            "illustrative_point": point,
            "illustrative_point_basis": ("EXPLICIT_DECLARED_OR_SINGLETON_ASSUMPTIONS_NOT_EMPIRICAL_ESTIMATE"
                                          if point is not None else "NO_POINT_CHOSEN"),
            "equity_normalization": normalized,
            "equity_pricing_sensitivity": priced})
    result = {"schema": SCHEMA,
        "status": "CONDITIONAL_BUSINESS_INTERRUPTION_CASHFLOW_COMPONENT",
        "source_anchor": source, "declared_assumptions": params,
        "original_portfolio_exposure": original_exposure,
        "equity_value_denominator": denominator,
        "duration_scenarios": results,
        "bounds_method": "SHARP_MONOTONE_CORNER_EXTREMA_FOR_FIXED_TIMING_AND_DISCOUNT_WITH_SHARED_RECTANGULAR_INPUTS",
        "confidence_level": None, "prediction_interval": False,
        "event_probability_P": None, "prediction_market_Q": None,
        "equity_priced_in_weight_identified": False,
        "event_to_issuer_interruption_verified": False,
        "source_authenticity_verified_by_this_function": False,
        "full_portfolio_conditional_pnl_identified": False,
        "total_issuer_return_identified": False,
        "source_or_model_calls": 0,
        "interpretation": "Present-value contrast of a declared issuer interruption versus no interruption, with all unmodeled channels held outside this component; not an event-conditional stock or portfolio return forecast.",
        "assumptions": [
            "Reported period revenue is spread uniformly over its actual calendar-day count; this run rate is a declared scaling assumption, not a future revenue estimate.",
            "The selected issuer's stated affected fraction actually stops earning the modeled revenue for the declared duration; a broader Class I strike or a contract YES does not establish this condition.",
            "Contribution margin is revenue less avoidable cash costs; fixed costs continue and an accounting operating margin is not substituted.",
            "Recovered shipments earn the same contribution margin at their delayed payment times; additional catch-up, restart and penalty costs require separate declared inputs.",
            "Each component cash flow receives the explicitly assumed immediately usable tax shield and fixed effective annual discount rate.",
            "Equity normalization assumes dollar-for-dollar change in the specified cash-flow present value, with unchanged financing, share count, discount-rate expectations and all other valuation channels.",
            "Direct shares and issuer sleeves inside original ETF positions are counted once; ETF returns are not added to constituent components.",
            "Any displayed pricing weights are separate hypothetical asset-embedded weights, not inferred from prediction-market quotes or physical frequencies.",
        ],
        "unmodeled_channels": [
            "The likelihood, actual network involvement, timing, duration and future revenue run rate of a real interruption.",
            "Long-run customer loss, future labor agreements, capital expenditure, financing or terminal-value effects.",
            "Macro spillovers and contemporaneous price changes of other holdings or the issuer itself.",
            "Actual asset-price embedded expectations, risk-premium changes and the mapping from a broad event contract to these cash flows.",
        ]}
    if params["additional_cost_mode"] == "EXCLUDED_FROM_COMPONENT":
        result["unmodeled_channels"].append("All additional restart, catch-up, legal and penalty costs: excluded, not estimated to be zero.")
    result["mechanism_fingerprint_sha256"] = sha256(json.dumps(result, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    return result
