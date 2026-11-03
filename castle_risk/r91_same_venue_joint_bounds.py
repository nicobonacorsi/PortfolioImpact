"""R91: sharp *conditional index-payoff* bounds from two Kalshi quote ladders.

NO equity P&L is claimed to be point-identified. In particular, a single
venue is not proof that prices at two expiries use a common stochastic
discount factor. The common-Q and zero/known discount factor assumptions
must be declared explicitly before these deterministic bounds are shown.

Given an index ladder with bid/ask quote intervals ``[b_i,a_i]`` and a
binary event quote ``[b,a]``, let ``p_i=Q(B_i)`` and ``q=Q(A)`` under ONE
*stipulated* common measure. For index-bin payoff bounds ``l_i<=g<=u_i``,
the sharp conditional YES lower bound solves

    min sum_i l_i z_i / q
    s.t. b_i <= p_i <= a_i, sum_i p_i=1,
         b <= q <= a, 0<=z_i<=p_i, sum_i z_i=q.

Charnes-Cooper: ``t=1/q``, ``y_i=t p_i``, ``w_i=t z_i``. This is an LP:

    min sum_i l_i w_i, sum_i y_i=t, sum_i w_i=1,
    b_i t<=y_i<=a_i t, 0<=w_i<=y_i, 1/a<=t<=1/b.

The upper bound uses ``u_i`` and maximization. Conversely, every LP
solution gives a genuine joint coupling by ``p_i=y_i/t,z_i=w_i/t``;
therefore bounds are sharp conditional on the stipulated assumptions.

This is NOT an empirical price of event/stock joint claims. Quote prices
are treated as normalized digital state prices only under a disclosed
discount-factor convention, ignoring trading fees, timing drift, and
possible market incompleteness. Cross-maturity pricing kernels generally
need not coincide.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np
from scipy.optimize import linprog


SCHEMA = "castle.option1.r91_quote_aware_coupling.v1"
PROVENANCE = "STIPULATED_SINGLE_MEASURE_SAME_VENUE_TWO_EXPIRIES"


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: a finite number is required")
    try:
        out = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: a finite number is required") from exc
    if not isfinite(out):
        raise ValueError(f"{name}: a finite number is required")
    return out


def _quote(raw_bid: Any, raw_ask: Any, *, discount_factor: float = 1) -> tuple[float, float]:
    bid, ask = _real(raw_bid, "YES bid"), _real(raw_ask, "YES ask")
    discount = _real(discount_factor, "discount factor")
    if not 0 < discount <= 1 or not 0 <= bid <= ask <= 1:
        raise ValueError("Invalid YES bid/ask or discount factor")
    out = bid / discount, ask / discount
    if out[0] > 1 + 1e-10:
        raise ValueError("A bid exceeds the assumed discount-normalized state price")
    return max(0.0, out[0]), min(1.0, out[1])


def parse_sp500_year_end_ladder(
    markets: Sequence[Mapping[str, Any]],
    *,
    ticker_prefix: str = "KXINXY-26DEC31H1600-",
    discount_factor: float = 1,
) -> list[dict[str, Any]]:
    """Strictly parse a penny-grid, exhaustive Kalshi *range* partition.

    The source's ``T4000`` lower tail is '3999.99 or below' and the
    ``T9000`` upper tail is '9000.01 or above'. Those offsets differ from
    their raw strike fields and must not create an overlapping boundary.
    """
    if not isinstance(ticker_prefix, str) or not ticker_prefix.endswith("-"):
        raise ValueError("Expected a fully qualified market-family prefix")
    if not isinstance(markets, Sequence) or isinstance(markets, (str, bytes)):
        raise ValueError("Expected source market records")
    found = []
    for row in markets:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed market record")
        ticker = row.get("ticker")
        if not isinstance(ticker, str) or not ticker.startswith(ticker_prefix):
            continue
        suffix = ticker.removeprefix(ticker_prefix)
        floor_raw, cap_raw = row.get("floor_strike"), row.get("cap_strike")
        if len(suffix) < 2 or suffix[0] not in "BT" or not suffix[1:].isdigit():
            raise ValueError("Unexpected index market strike identity")
        if suffix.startswith("B"):
            if floor_raw is None or cap_raw is None:
                raise ValueError("Missing bounded-bin strikes")
            floor = _real(floor_raw, "bin lower strike")
            cap = _real(cap_raw, "bin upper strike")
            if round(floor * 100) != floor * 100 or abs(round(cap * 100)-cap * 100)>1e-6:
                raise ValueError("Index market strikes must lie on a cent grid")
            if cap < floor:
                raise ValueError("Reversed index market interval")
            interval = (round(floor, 2), round(cap, 2))
        elif floor_raw is None and cap_raw is not None:
            upper = _real(cap_raw, "lower tail cap")
            interval = (None, round(upper - .01, 2))
            if "or below" not in str(row.get("subtitle", "")).lower():
                raise ValueError("The lower-tail source contract must say 'or below'")
        elif cap_raw is None and floor_raw is not None:
            lower = _real(floor_raw, "upper tail floor")
            interval = (round(lower + .01, 2), None)
            if "or above" not in str(row.get("subtitle", "")).lower():
                raise ValueError("The upper-tail source contract must say 'or above'")
        else:
            raise ValueError("Unsupported open-tail strike geometry")
        bid, ask = _quote(row.get("yes_bid_dollars"), row.get("yes_ask_dollars"),
                          discount_factor=discount_factor)
        found.append({"ticker": ticker, "lower_index": interval[0],
                      "upper_index": interval[1], "q_bid": bid, "q_ask": ask})
    if len(found) < 3 or len({x["ticker"] for x in found}) != len(found):
        raise ValueError("Missing or duplicate index-ladder contracts")
    found.sort(key=lambda x: float("-inf") if x["lower_index"] is None else x["lower_index"])
    if found[0]["lower_index"] is not None or found[-1]["upper_index"] is not None:
        raise ValueError("Index bins must cover BOTH unbounded tails")
    for before, after in zip(found, found[1:]):
        if before["upper_index"] is None or after["lower_index"] is None:
            raise ValueError("Nonterminal open tail")
        if round((after["lower_index"]-before["upper_index"])*100) != 1:
            raise ValueError("Index bins contain a gap or overlap")
    if sum(r["q_bid"] for r in found) > 1+1e-9 or sum(r["q_ask"] for r in found) < 1-1e-9:
        raise ValueError("Quotes inconsistent with any normalized index marginal")
    return found


def sp500_below_threshold_payoffs(
    bins: Sequence[Mapping[str, Any]], *, threshold: float
) -> list[tuple[float, float]]:
    """Exact binary tail payoff only if the threshold respects market bins."""
    cutoff = _real(threshold, "index cutoff")
    result = []
    for row in bins:
        lower, upper = row["lower_index"], row["upper_index"]
        if upper is not None and upper < cutoff:
            result.append((1., 1.))
        elif lower is not None and lower >= cutoff:
            result.append((0., 0.))
        else:
            raise ValueError("Threshold slices an observed quote bin: cannot assign exact payoff")
    return result


def capped_index_return_payoffs(
    bins: Sequence[Mapping[str, Any]],
    *,
    reference_index: float,
    loss_cap: float = -.2,
    gain_cap: float = .2,
) -> list[tuple[float, float]]:
    """Range of an explicitly capped *synthetic* index return in each bin."""
    reference = _real(reference_index, "reference index")
    low, high = _real(loss_cap, "loss cap"), _real(gain_cap, "gain cap")
    if reference <= 0 or not -1 <= low < 0 < high or high > 2:
        raise ValueError("Invalid reference index or payoff clipping")
    def payoff(s: float) -> float:
        return max(low, min(high, s/reference-1))
    ranges = []
    for row in bins:
        left, right = row["lower_index"], row["upper_index"]
        ranges.append((payoff(0 if left is None else left),
                       high if right is None else payoff(right)))
    return ranges


def _conditional_lp(
    bid: np.ndarray,
    ask: np.ndarray,
    q_bounds: tuple[float, float],
    payoff_ranges: np.ndarray,
) -> dict[str, Any]:
    n = bid.size
    q_low, q_high = q_bounds
    if not 0 < q_low <= q_high < 1:
        raise ValueError("Strictly positive, non-certain conditional event required")
    # variables: [y_0..y_(n-1), w_0..w_(n-1), t]
    t_pos = 2*n
    eq = np.zeros((2, 2*n+1))
    eq[0, :n], eq[0, t_pos] = 1, -1
    eq[1, n:2*n] = 1
    upper = np.zeros((3*n, 2*n+1))
    for i in range(n):
        upper[3*i, i], upper[3*i, t_pos] = 1, -ask[i]
        upper[3*i+1, i], upper[3*i+1, t_pos] = -1, bid[i]
        upper[3*i+2, n+i], upper[3*i+2, i] = 1, -1
    bounds = [(0, None)]*(2*n) + [(1/q_high, 1/q_low)]
    extremes = []
    for which, sign in [(0, 1), (1, -1)]:
        obj = np.zeros(2*n+1)
        obj[n:2*n] = sign*payoff_ranges[:, which]
        opt = linprog(obj, A_ub=upper, b_ub=np.zeros(3*n),
                      A_eq=eq, b_eq=np.array([0, 1]), bounds=bounds,
                      method="highs")
        if not opt.success:
            raise ValueError("No coherent joint law within the observed quote bands: " + opt.message)
        y, w, t = opt.x[:n], opt.x[n:2*n], float(opt.x[t_pos])
        p, z = y/t, w/t
        if (abs(sum(p)-1) > 1e-7 or abs(sum(z)-1/t) > 1e-7
                or np.any(z < -1e-7) or np.any(z-p > 1e-7)
                or np.any(p < bid-1e-7) or np.any(p > ask+1e-7)):
            raise ArithmeticError("Solver witness fails coupling feasibility verification")
        extremes.append({"value": float(sign * opt.fun),
                         "event_q": 1/t,
                         "marginal_bin_masses": [float(x) for x in p],
                         "event_bin_joint_masses": [float(x) for x in z]})
    return {"lower": extremes[0]["value"], "upper": extremes[1]["value"],
            "lower_witness": extremes[0], "upper_witness": extremes[1]}


def _unconditional_lp(
    bid: np.ndarray, ask: np.ndarray, payoff_ranges: np.ndarray
) -> dict[str, Any]:
    """Sharp marginal price range under per-bin quote intervals."""
    solutions = []
    for which, sign in [(0, 1), (1, -1)]:
        opt = linprog(sign * payoff_ranges[:, which],
                      A_eq=np.ones((1, bid.size)), b_eq=np.array([1.]),
                      bounds=list(zip(bid.tolist(), ask.tolist())), method="highs")
        if not opt.success:
            raise ValueError("No normalized index marginal in those quotes: " + opt.message)
        solutions.append({"value": float(sign * opt.fun),
                          "marginal_bin_masses": [float(v) for v in opt.x]})
    return {"lower": solutions[0]["value"], "upper": solutions[1]["value"],
            "lower_witness": solutions[0], "upper_witness": solutions[1]}


def sharp_quote_aware_conditional_bounds(
    bins: Sequence[Mapping[str, Any]],
    *,
    event_yes_quote: tuple[float, float],
    payoff_intervals: Sequence[tuple[float, float]],
    provenance: str,
    event_discount_factor: float = 1,
    event_label: str = "",
    payoff_label: str = "",
) -> dict[str, Any]:
    """Sharp event-YES/NO expectation bands, NOT identified equity stock P&L.

    All inputs are real quote intervals. The same-Q hypothesis is *not*
    established by venue identity; the caller must provide an explicit
    provenance token to prevent accidental admission as a market fact.
    """
    if provenance != PROVENANCE:
        return {"schema": SCHEMA, "status": "ABSTAIN_NO_OBSERVED_COMMON_Q",
                "sharp_conditional_bounds": None,
                "portfolio_equity_pnl_identified": False,
                "already_priced_in_equity_fraction_identified": False}
    if len(bins) < 2 or len(bins) != len(payoff_intervals):
        raise ValueError("Complete bin-aligned payoff schedule required")
    bid, ask = [], []
    for row in bins:
        lo, hi = _real(row.get("q_bid"), "index bid"), _real(row.get("q_ask"), "index ask")
        if not 0 <= lo <= hi <= 1:
            raise ValueError("Invalid bin quote interval")
        bid.append(lo)
        ask.append(hi)
    bid_arr, ask_arr = np.array(bid, dtype=float), np.array(ask, dtype=float)
    if sum(bid_arr) > 1+1e-9 or sum(ask_arr) < 1-1e-9:
        raise ValueError("No feasible normalized index marginal")
    payoff = np.array([(_real(x[0], "payoff lower"), _real(x[1], "payoff upper"))
                       for x in payoff_intervals], dtype=float)
    if np.any(payoff[:, 0] > payoff[:, 1]):
        raise ValueError("Inverted bounded-payoff interval")
    event_q = _quote(*event_yes_quote, discount_factor=event_discount_factor)
    if not 0 < event_q[0] <= event_q[1] < 1:
        raise ValueError("Nondegenerate event probability quote interval required")
    yes = _conditional_lp(bid_arr, ask_arr, event_q, payoff)
    no = _conditional_lp(bid_arr, ask_arr, (1-event_q[1], 1-event_q[0]), payoff)
    unconditioned = _unconditional_lp(bid_arr, ask_arr, payoff)
    return {
        "schema": SCHEMA,
        "status": "SHARP_BOUNDS_UNDER_STIPULATED_COMMON_Q_NOT_STOCK_PNL",
        "event_label": event_label,
        "payoff_label": payoff_label,
        "event_q_interval_assumed": list(event_q),
        "same_venue": "Kalshi",
        "single_measure_is_observed": False,
        "common_pricing_kernel_verified": False,
        "portfolio_equity_pnl_identified": False,
        "already_priced_in_equity_fraction_identified": False,
        "yes_conditional": yes,
        "no_conditional": no,
        "unconditional_index_payoff": unconditioned,
        "payoff_exactly_binary": bool(np.all(payoff[:, 0] == payoff[:, 1])
                                      and np.all((payoff[:, 0] == 0) | (payoff[:, 0] == 1))),
        "index_quoted_bin_count": len(bins),
        "assumptions": [
            "One stipulated risk-neutral measure Q common to BOTH Kalshi contract maturities.",
            "Each observed tradable YES bid/ask interval is treated as a normalized digital state-price band under the declared discount factors; fees and capacity ignored.",
            "The index market ranges form an exclusive and exhaustive terminal partition.",
            "Nothing identifies the event-index dependency or the portfolio's individual stock response.",
            "A conditional pricing-measure expectation is neither a physical expectation nor an actual trading P&L.",
        ],
    }


def sharp_binary_event_contribution_band(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Sharp algebraic interval for a *binary index claim's* priced event term.

    Under an assumed common Q and a two-state, zero-rate identity:

        Q(B) = Q(B|not A) + q * [Q(B|A)-Q(B|not A)].

    We identify a sharp band for the *second term* from separate bid/asks,
    not a percentage of a STOCK PRICE or a causal repricing of an issuer.
    The formula is valid when the input result comes from an exact 0/1
    payoff partition. The caller must pass the binary-coupling result.
    """
    if (result.get("status") != "SHARP_BOUNDS_UNDER_STIPULATED_COMMON_Q_NOT_STOCK_PNL"
            or result.get("payoff_exactly_binary") is not True):
        raise ValueError("A coherent conditional-index quote report is required")
    p_low = _real(result["unconditional_index_payoff"]["lower"], "binary index marginal low")
    p_high = _real(result["unconditional_index_payoff"]["upper"], "binary index marginal high")
    q_high = _real(result["event_q_interval_assumed"][1], "event quote high")
    if not 0 <= p_low <= p_high <= 1 or not 0 < q_high < 1:
        raise ValueError("Invalid binary marginal and event weights")
    # With q fixed, z=Q(B,A) is allowed exactly on the Fréchet segment
    # [max(0,p+q-1), min(p,q)]. Lower contribution is the V-shaped
    # piecewise function in p with minimum p=clip(1-q); upper has maximum
    # p=clip(q). Both objective endpoints are nondecreasing in q on the
    # upper side / nonincreasing on the lower side; q_max is extremal.
    p_worst = max(p_low, min(p_high, 1-q_high))
    p_best = max(p_low, min(p_high, q_high))
    z_min = max(0., p_worst+q_high-1)
    z_max = min(p_best, q_high)
    low = (z_min-q_high*p_worst)/(1-q_high)
    high = (z_max-q_high*p_best)/(1-q_high)
    return {
        "status": "SHARP_COMMON_Q_BINARY_CLAIM_EVENT_CONTRIBUTION_BAND",
        "market_state_price_difference_from_hypothetical_no_event_baseline": [low, high],
        "unit": "index-binary state price probability points (not stock valuation percent)",
        "counterfactual_no_event_baseline_observed": False,
        "fraction_of_equity_market_cap_already_priced_identified": False,
        "independence_zero_contribution_also_feasible": low <= 0 <= high,
        "worst_state_witness": {"index_down_marginal": p_worst,
                                 "event_q": q_high, "joint_index_down_and_yes": z_min},
        "best_state_witness": {"index_down_marginal": p_best,
                                "event_q": q_high, "joint_index_down_and_yes": z_max},
        "warning": "Only for exact binary outcomes. Not an observed causal contribution or the 'priced-in fraction' of any stock.",
    }
