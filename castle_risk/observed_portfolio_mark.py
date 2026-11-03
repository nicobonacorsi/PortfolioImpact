"""Observed portfolio repricing context; NEVER an estimated equity priced-in probability.

Today's original weights are current-mark weights. Each holding's current
shares are marked backward for 20 completed sessions (no intervening trades
assumed), and all same-date price moves are aggregated. Historical rolling
windows are synthetic as-if-current-weight reference marks, not trade histories.
None can establish why a price moved or an equity priced-in probability.
No prediction-market quote, physical P(E), causal event effect or asset q
is inferred. Cash counts once; ETF constituents do not add to original NAV.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
from math import fsum, isfinite
import json

from .joint_credal_impact import JointImpactInputError, _original_positions, portfolio_signature

SCHEMA = "castle.observed_portfolio_mark_context.v1"
HORIZON_SESSIONS = 20
MIN_WINDOWS = 20
MAX_DAYS_STALE = 7
MAX_INTERSESSION_GAP_DAYS = 7
_TOL = 1e-7


def _day(value):
    if not isinstance(value, str):
        raise ValueError("Session date must be an ISO date.")
    try:
        result = date.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid session ISO date.") from None
    if value != result.isoformat():
        raise ValueError("Noncanonical session ISO date.")
    return result


def _number(value):
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError("Close must be a finite strictly positive number.")
    return float(value)


def _quantile(values, p):
    """Linear empirical sample quantile; never a predictive confidence bound."""
    k = (len(values) - 1) * p
    low = int(k)
    return values[low] * (1 - (k - low)) + values[min(low + 1, len(values) - 1)] * (k - low)


def observed_portfolio_mark(positions, notional, prices_by_symbol, *, as_of_utc):
    """Read-only, fail-closed derivation from already acquired completed daily bars."""
    rows, nav = _original_positions(list(positions), notional)
    signature = portfolio_signature(rows, nav, HORIZON_SESSIONS)
    capture = datetime.fromisoformat(str(as_of_utc).replace("Z", "+00:00"))
    if capture.utcoffset() is None:
        raise ValueError("The assessment capture needs a timezone.")
    today = capture.astimezone(timezone.utc).date()
    base = {
        "schema": SCHEMA, "as_of_utc": as_of_utc,
        "portfolio_signature": signature, "horizon_trading_sessions": HORIZON_SESSIONS,
        "source_adjustment": "SPLIT_ADJUSTED_CLOSE_EXCLUDES_CASH_DIVIDENDS",
        "mark_convention": "CURRENT_END_MARK_SHARES_VALUED_AT_20_SESSION_PRIOR_CLOSE",
        "historical_portfolio_is_synthetic_reference": True,
        "empirical_event_attribution": False, "equity_priced_in_fraction_identified": False,
        "prediction_market_probability_used_as_equity_q": False,
        "historical_distribution_is_forward_forecast": False,
        "original_etf_constituents_added_to_nav": False,
    }

    def abstain(reason):
        return {**base, "status": "INSUFFICIENT_COMPLETED_MARK_DATA",
                "reason": reason, "recent_20_session_mark_change_pct_current_nav": None,
                "recent_20_session_pnl_usd": None}

    active = [row for row in rows if row["type"] != "cash" and row["weight"] > 0]
    if not active:
        return abstain("NO_ACTIVE_MARKET_POSITIONS")
    if not isinstance(prices_by_symbol, dict):
        return abstain("PRICE_SESSIONS_NOT_AVAILABLE")
    by_symbol = {}
    for holding in active:
        symbol = holding["symbol"]
        series = prices_by_symbol.get(symbol)
        if not isinstance(series, list) or len(series) < HORIZON_SESSIONS + MIN_WINDOWS:
            return abstain("MISSING_OR_SHORT_SPLIT_ADJUSTED_PRICE_HISTORY: " + symbol)
        dated = {}
        try:
            for record in series:
                if not isinstance(record, dict) or set(record) < {"date", "close"}:
                    raise ValueError("Invalid daily bar.")
                day = _day(record["date"])
                if day >= today:
                    raise ValueError("Incomplete or future session not admissible.")
                close = _number(record["close"])
                if day in dated:
                    raise ValueError("Duplicate completed date.")
                dated[day] = close
        except ValueError:
            return abstain("INVALID_OR_DUPLICATE_COMPLETED_PRICE_SESSION: " + symbol)
        by_symbol[symbol] = dated
    common_days = sorted(set.intersection(*(set(pts) for pts in by_symbol.values())))
    if len(common_days) < HORIZON_SESSIONS + MIN_WINDOWS:
        return abstain("INSUFFICIENT_SYNCHRONIZED_PRICE_SESSIONS")
    if (today - common_days[-1]).days > MAX_DAYS_STALE:
        return abstain("STALENESS_NOT_A_CURRENT_MARK")
    if any((b - a).days > MAX_INTERSESSION_GAP_DAYS
           for a, b in zip(common_days, common_days[1:])):
        return abstain("DISCONTINUOUS_CROSS_HOLDING_SESSIONS")
    synchronized_windows = []
    for stop in range(HORIZON_SESSIONS, len(common_days)):
        start = common_days[stop - HORIZON_SESSIONS]
        end = common_days[stop]
        # Current-end weights represent the holdings' dollar shares at END.
        # Backward-mark the SAME share counts to the prior completed mark:
        # PNL / NAV_end = sum(w_i * [1 - price_i(start)/price_i(end)]).
        # Earlier overlapping windows are synthetic reference portfolios,
        # with current model weights applied at EACH historical end mark.
        total = fsum(row["weight"] * (1 - by_symbol[row["symbol"]][start] /
                                    by_symbol[row["symbol"]][end])
                     for row in active)
        synchronized_windows.append(total * 100)
    recent_by_symbol = {
        row["symbol"]: (0.0 if row["type"] == "cash" or row["weight"] == 0 else
                        100 * (1 - by_symbol[row["symbol"]][common_days[-1 - HORIZON_SESSIONS]] /
                               by_symbol[row["symbol"]][common_days[-1]]))
        for row in rows
    }
    # The most recent synchronous window is exactly the sum of holding moves.
    recent = fsum(row["weight"] * recent_by_symbol[row["symbol"]] for row in rows)
    if abs(recent - synchronized_windows[-1]) > _TOL:
        return abstain("INTERNAL_SYNCHRONIZED_WINDOW_MISMATCH")
    sorted_windows = sorted(synchronized_windows)
    sample_digest = sha256(json.dumps({
        s: [(d.isoformat(), by_symbol[s][d]) for d in common_days]
        for s in sorted(by_symbol)
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**base, "status": "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION",
            "last_completed_session": common_days[-1].isoformat(),
            "recent_window_start": common_days[-1 - HORIZON_SESSIONS].isoformat(),
            "synchronized_sessions": len(common_days),
            "overlapping_historical_windows": len(synchronized_windows),
            "history_start": common_days[0].isoformat(),
            "history_end": common_days[-1].isoformat(),
            "recent_20_session_mark_change_pct_current_nav": recent,
            "recent_20_session_pnl_usd": nav * recent / 100,
            "holding_recent_20_session_mark_change_pct_current_nav": recent_by_symbol,
            "historical_portfolio_20_session_mark_change_range_pct_current_nav": [sorted_windows[0], sorted_windows[-1]],
            "historical_portfolio_20_session_mark_change_p05_p95_pct_current_nav": [
                _quantile(sorted_windows, .05), _quantile(sorted_windows, .95)],
            "historical_portfolio_20_session_mark_change_median_pct_current_nav": _quantile(sorted_windows, .5),
            "private_price_session_fingerprint_sha256": sample_digest,
            "sample_windows_overlap_and_are_not_independent": True,
            "interpretation": "Retrospective mark change of current modeled share counts if held without trades; historical windows apply today's modeled weights at past window ends. Neither an actual transaction history nor an event attribution or a priced-in share.",
    }


def review_observed_mark(case):
    """Check public identity and accounting, not unavailable private-provider bytes.

    There is NO authenticated proof of the privately derived quantiles inside
    the public export. Their published provenance is explicitly limited.
    """
    row = case.get("observed_mark_context")
    if not isinstance(row, dict):
        case.pop("observed_mark_context", None)
        return False
    try:
        positions, nav = _original_positions(case["positions"], case["notional"])
        expected_sig = portfolio_signature(positions, nav, HORIZON_SESSIONS)
        if (row.get("schema") != SCHEMA or row.get("status") !=
                "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
                or row.get("portfolio_signature") != expected_sig
                or row.get("as_of_utc") != case.get("capture")
                or row.get("horizon_trading_sessions") != HORIZON_SESSIONS
                or row.get("source_adjustment") != "SPLIT_ADJUSTED_CLOSE_EXCLUDES_CASH_DIVIDENDS"
                or row.get("mark_convention") != "CURRENT_END_MARK_SHARES_VALUED_AT_20_SESSION_PRIOR_CLOSE"
                or row.get("historical_portfolio_is_synthetic_reference") is not True
                or any(row.get(k) is not False for k in (
                    "empirical_event_attribution", "equity_priced_in_fraction_identified",
                    "prediction_market_probability_used_as_equity_q",
                    "historical_distribution_is_forward_forecast",
                    "original_etf_constituents_added_to_nav"))
                or row.get("sample_windows_overlap_and_are_not_independent") is not True):
            raise ValueError("Unsupported price attribution or mismatched capture.")
        completed = _day(row["last_completed_session"])
        first = _day(row["history_start"])
        start = _day(row["recent_window_start"])
        end = _day(row["history_end"])
        today = datetime.fromisoformat(case["capture"].replace("Z", "+00:00")).astimezone(timezone.utc).date()
        sessions = row["synchronized_sessions"]
        windows = row["overlapping_historical_windows"]
        if (not first <= start < completed == end < today
                or (today - completed).days > MAX_DAYS_STALE
                or type(sessions) is not int or type(windows) is not int
                or sessions < HORIZON_SESSIONS + MIN_WINDOWS
                or windows != sessions - HORIZON_SESSIONS):
            raise ValueError("Inconsistent price-window chronology.")
        moves = row["holding_recent_20_session_mark_change_pct_current_nav"]
        if not isinstance(moves, dict) or set(moves) != {p["symbol"] for p in positions}:
            raise ValueError("Incomplete original-holding repricing.")
        if any(type(v) not in (int, float) or not isfinite(v) or v >= 100
               for v in moves.values()):
            raise ValueError("Invalid holding return.")
        if any(p["type"] == "cash" and moves[p["symbol"]] != 0 for p in positions):
            raise ValueError("Nominal USD cash cannot reprice.")
        aggregate = fsum(p["weight"] * moves[p["symbol"]] for p in positions)
        observed_pct = row["recent_20_session_mark_change_pct_current_nav"]
        observed_usd = row["recent_20_session_pnl_usd"]
        if any(type(x) not in (int, float) or not isfinite(x) for x in
               (observed_pct, observed_usd)):
            raise ValueError("Nonfinite observed repricing.")
        if (abs(aggregate - observed_pct) > 1e-7
                or abs(observed_usd - nav * observed_pct / 100) > max(1e-6, abs(nav) * 1e-10)):
            raise ValueError("Original NAV repricing mismatch.")
        r = row["historical_portfolio_20_session_mark_change_range_pct_current_nav"]
        q = row["historical_portfolio_20_session_mark_change_p05_p95_pct_current_nav"]
        med = row["historical_portfolio_20_session_mark_change_median_pct_current_nav"]
        if (not isinstance(r, list) or not isinstance(q, list)
                or len(r) != 2 or len(q) != 2
                or any(type(x) not in (int, float) or not isfinite(x) for x in r + q + [med])
                or not r[0] <= q[0] <= med <= q[1] <= r[1]
                or not r[0] - 1e-7 <= observed_pct <= r[1] + 1e-7):
            raise ValueError("Malformed historical range.")
        if (not isinstance(row.get("private_price_session_fingerprint_sha256"), str)
                or len(row["private_price_session_fingerprint_sha256"]) != 64):
            raise ValueError("Missing private input fingerprint.")
    except (ValueError, TypeError, KeyError, JointImpactInputError, OverflowError):
        case.pop("observed_mark_context", None)
        return False
    return True
