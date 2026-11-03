"""Traceable arithmetic-return portfolio stresses, separate from market bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .portfolio import Portfolio, _finite_number, _symbol


def pricing_in_sensitivity(*, payoff_gap: float, equity_pricing_weight: float) -> dict[str, Any]:
    """Illustrate a separate assumed two-state, unit-discount valuation model.

    ``payoff_gap`` is the dollar difference V_yes - V_no for the same asset or
    portfolio, not a percentage return. The user explicitly supplies the equity
    pricing weight pi. With V_today = pi*V_yes + (1-pi)*V_no and discount factor
    one, resolution changes value by (1-pi)*gap in YES and -pi*gap in NO.

    This is an algebraic sensitivity, not an estimated model. No prediction-
    market probability, physical probability, stock price, or return is inferred.
    """
    gap = _finite_number(payoff_gap, "Dollar YES-minus-NO payoff gap")
    pi = _finite_number(equity_pricing_weight, "Assumed equity pricing weight")
    if not 0 <= pi <= 1:
        raise ValueError("Assumed equity pricing weight must lie between zero and one.")
    return {
        "payoff_gap": gap,
        "equity_pricing_weight": pi,
        "yes_change": (1 - pi) * gap,
        "no_change": -pi * gap,
        "unit": "USD value change, not percentage return",
        "basis": "user_assumed_two_state_valuation",
        "assumptions": [
            "The user specifies the dollar YES-minus-NO terminal value gap for the same asset or portfolio.",
            "The user specifies a separate equity pricing weight pi; no Kalshi probability is substituted or inferred.",
            "Today's value is assumed to equal pi times the YES value plus (1-pi) times the NO value, with discount factor one.",
            "Only two exhaustive terminal states are modeled; state values are held fixed and other price changes are excluded.",
            "These are algebraic dollar changes under assumptions, not forecasts, empirical estimates, percentage returns, or recommendations.",
        ],
    }


@dataclass(frozen=True)
class ScenarioContribution:
    symbol: str
    asset_type: str
    weight: float
    position_value: float
    assumed_return: float | None
    portfolio_return_contribution: float | None
    pnl: float | None
    lower_return_contribution: float | None
    upper_return_contribution: float | None
    status: str


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    basis: str
    contributions: tuple[ScenarioContribution, ...]
    total_return: float | None
    total_pnl: float | None
    known_return_subtotal: float
    known_pnl_subtotal: float
    missing_stress_weight: float
    return_lower: float | None
    return_upper: float | None
    pnl_lower: float | None
    pnl_upper: float | None
    status: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_scenario(
    portfolio: Portfolio,
    position_returns: Mapping[str, float],
    *,
    name: str = "User-specified portfolio stress",
    missing_return_bounds: tuple[float, float] | None = None,
    assumption_note: str = "",
) -> ScenarioResult:
    """Apply explicit stresses once to original positions, including ETF shares.

    Each stress is an arithmetic total return over a common assumed horizon.
    Missing stresses are never silently zero-filled. A finite interval for them
    is used only when the caller explicitly supplies the two endpoints. Those
    endpoints are user assumptions, not probability or confidence bounds.
    """
    if not isinstance(position_returns, Mapping):
        raise ValueError("position_returns must map original portfolio symbols to return assumptions.")
    stresses: dict[str, float] = {}
    portfolio_symbols = {p.symbol for p in portfolio.positions}
    for raw_symbol, value in position_returns.items():
        symbol = _symbol(raw_symbol)
        if symbol in stresses:
            raise ValueError("Duplicate stress symbols after symbol normalization.")
        if symbol not in portfolio_symbols:
            raise ValueError(f"Stress symbol {symbol} is not an original portfolio position; look-through assets are not added to ETF P&L.")
        stress = _finite_number(value, f"Return stress for {symbol}")
        if stress < -1:
            raise ValueError("Arithmetic returns below -100% are invalid for this unlevered long-only position model.")
        stresses[symbol] = stress
    bounds = None
    if missing_return_bounds is not None:
        if len(missing_return_bounds) != 2:
            raise ValueError("missing_return_bounds must contain exactly two endpoints.")
        low = _finite_number(missing_return_bounds[0], "Missing-return lower bound")
        high = _finite_number(missing_return_bounds[1], "Missing-return upper bound")
        if low < -1 or low > high:
            raise ValueError("Missing-return bounds must satisfy -1 <= lower <= upper.")
        bounds = (low, high)
    contributions = []
    for position in portfolio.positions:
        stress = stresses.get(position.symbol)
        # A zero-weight row has identically zero impact even if no stress exists.
        known = stress is not None or position.weight == 0
        contribution = position.weight * (stress or 0.0) if known else None
        if known:
            lower = upper = contribution
            status = "assumed_return" if stress is not None else "zero_weight"
        elif bounds is not None:
            lower, upper = (position.weight * bound for bound in bounds)
            status = "missing_return_with_assumed_range"
        else:
            lower = upper = None
            status = "missing_return"
        contributions.append(ScenarioContribution(
            position.symbol, position.asset_type, position.weight,
            portfolio.notional * position.weight, stress, contribution,
            None if contribution is None else portfolio.notional * contribution,
            lower, upper, status,
        ))
    missing_weight = math.fsum(c.weight for c in contributions if c.portfolio_return_contribution is None)
    known_subtotal = math.fsum(c.portfolio_return_contribution for c in contributions if c.portfolio_return_contribution is not None)
    complete = all(c.portfolio_return_contribution is not None for c in contributions)
    bounded = all(c.lower_return_contribution is not None for c in contributions)
    lower = math.fsum(c.lower_return_contribution for c in contributions) if bounded else None
    upper = math.fsum(c.upper_return_contribution for c in contributions) if bounded else None
    assumptions = [
        "Returns are user-specified scenario assumptions, not empirical estimates or forecasts.",
        "Position weights are the initial portfolio weights; holdings are static over a common scenario horizon.",
        "Returns apply to original positions only: an ETF and its constituents are never both charged to portfolio P&L.",
        "Prediction-market probability bounds are displayed separately and are not used as equity pricing weights.",
        "USD arithmetic total returns are assumed; taxes, trading costs, cash flows, financing, and FX are excluded.",
    ]
    if bounds is not None:
        assumptions.append(f"For unstressed positions only, returns are assumed to lie in [{bounds[0]:.8g}, {bounds[1]:.8g}]. This is not a statistical confidence interval.")
    if assumption_note:
        assumptions.append(str(assumption_note))
    return ScenarioResult(
        name=str(name), basis="user_specified_stress", contributions=tuple(contributions),
        total_return=known_subtotal if complete else None,
        total_pnl=portfolio.notional * known_subtotal if complete else None,
        known_return_subtotal=known_subtotal, known_pnl_subtotal=portfolio.notional * known_subtotal,
        missing_stress_weight=missing_weight, return_lower=lower, return_upper=upper,
        pnl_lower=None if lower is None else portfolio.notional * lower,
        pnl_upper=None if upper is None else portfolio.notional * upper,
        status="complete_assumed_scenario" if complete else "incomplete_bounded_by_user_assumption" if bounded else "incomplete_missing_stresses",
        assumptions=tuple(assumptions),
    )
