"""Price-in observability tests: synchronized original-NAV marks, not event inference."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from math import isclose

import pytest

from castle_risk.observed_portfolio_mark import (
    observed_portfolio_mark, review_observed_mark, SCHEMA,
)

CAPTURE = "2026-11-01T09:00:00+00:00"
PORT = [
    {"symbol": "AAA", "type": "stock", "weight": .40},
    {"symbol": "BBB", "type": "etf", "weight": .35},
    {"symbol": "USD", "type": "cash", "weight": .25},
]


def _history(*, n=90):
    days = []
    day = date(2026, 7, 1)
    while day <= date(2026, 10, 30):
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    days = days[-n:]
    return {
        "AAA": [{"date": d.isoformat(), "close": 100 * 1.01 ** i}
                for i, d in enumerate(days)],
        "BBB": [{"date": d.isoformat(), "close": 100 * .995 ** i}
                for i, d in enumerate(days)],
    }


def _good():
    return observed_portfolio_mark(PORT, 100000., _history(), as_of_utc=CAPTURE)


def test_live_price_math_replays_synchronous_original_holdings_without_q():
    x = _good()
    assert x["schema"] == SCHEMA
    assert x["status"] == "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
    assert x["last_completed_session"] == "2026-10-30"
    assert x["synchronized_sessions"] >= 80
    assert x["overlapping_historical_windows"] == x["synchronized_sessions"] - 20
    expected = .40 * (1 - 1 / 1.01 ** 20) + .35 * (1 - 1 / .995 ** 20)
    assert isclose(x["recent_20_session_mark_change_pct_current_nav"], expected * 100, abs_tol=1e-8)
    assert isclose(x["recent_20_session_pnl_usd"], expected * 100000, abs_tol=1e-7)
    assert x["holding_recent_20_session_mark_change_pct_current_nav"]["USD"] == 0
    assert x["historical_portfolio_20_session_mark_change_range_pct_current_nav"][0] <= x["recent_20_session_mark_change_pct_current_nav"]
    assert x["historical_portfolio_20_session_mark_change_range_pct_current_nav"][1] >= x["recent_20_session_mark_change_pct_current_nav"]
    assert x["equity_priced_in_fraction_identified"] is False
    assert x["empirical_event_attribution"] is False
    assert x["historical_distribution_is_forward_forecast"] is False
    assert x["prediction_market_probability_used_as_equity_q"] is False
    assert x["source_adjustment"] == "SPLIT_ADJUSTED_CLOSE_EXCLUDES_CASH_DIVIDENDS"
    assert x["mark_convention"] == "CURRENT_END_MARK_SHARES_VALUED_AT_20_SESSION_PRIOR_CLOSE"
    assert x["historical_portfolio_is_synthetic_reference"] is True
    assert "current_price" not in x
    case = {"capture": CAPTURE, "positions": deepcopy(PORT), "notional": 100000.,
            "observed_mark_context": deepcopy(x)}
    assert review_observed_mark(case) is True


def test_reordering_equivalent_original_positions_preserves_result_and_signature():
    h = _history()
    a = observed_portfolio_mark(PORT, 100000., h, as_of_utc=CAPTURE)
    b = observed_portfolio_mark(list(reversed(PORT)), 100000., h, as_of_utc=CAPTURE)
    assert a == b


def test_missing_and_stale_data_abstain_not_fill_zero_or_call_it_priced_in():
    missing = observed_portfolio_mark(PORT, 100000.,
                                      {"AAA": _history()["AAA"]}, as_of_utc=CAPTURE)
    assert missing["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"
    assert missing["recent_20_session_pnl_usd"] is None
    assert "BBB" in missing["reason"]
    old = observed_portfolio_mark(PORT, 100000., _history(),
                                   as_of_utc="2026-11-20T09:00:00+00:00")
    assert old["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"
    assert old["reason"] == "STALENESS_NOT_A_CURRENT_MARK"


def test_per_holding_last_prices_cannot_be_mixed_on_nonoverlapping_dates():
    h = _history()
    # Non-overlap of the last 60 sessions: never mix asynchronous latest closes.
    h["BBB"] = h["BBB"][:25]
    out = observed_portfolio_mark(PORT, 100000., h, as_of_utc=CAPTURE)
    assert out["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"
    assert out["recent_20_session_mark_change_pct_current_nav"] is None


@pytest.mark.parametrize("mutation", [
    lambda h: h["AAA"].append(deepcopy(h["AAA"][-1])),
    lambda h: h["AAA"][-1].update(close=-3.),
    lambda h: h["AAA"][-1].update(close=float("nan")),
    lambda h: h["BBB"][-1].update(date="2026-11-01"),
    lambda h: h["BBB"][-1].update(date="2026-10-99"),
    lambda h: h["AAA"][-1].update(close=True),
])
def test_untrusted_prices_with_duplication_future_or_invalid_values_abstain(mutation):
    h = _history()
    mutation(h)
    x = observed_portfolio_mark(PORT, 100000., h, as_of_utc=CAPTURE)
    assert x["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"


@pytest.mark.parametrize("change", [
    lambda x: x.update(recent_20_session_pnl_usd=0),
    lambda x: x.update(recent_20_session_mark_change_pct_current_nav=500.),
    lambda x: x.update(portfolio_signature="0" * 64),
    lambda x: x.update(equity_priced_in_fraction_identified=True),
    lambda x: x.update(empirical_event_attribution=True),
    lambda x: x["holding_recent_20_session_mark_change_pct_current_nav"].update(USD=1),
    lambda x: x["historical_portfolio_20_session_mark_change_p05_p95_pct_current_nav"].__setitem__(0,1000.),
    lambda x: x.update(as_of_utc="2026-09-10T00:00:00+00:00"),
    lambda x: x.update(overlapping_historical_windows=999),
])
def test_public_export_accounting_claims_and_mark_mismatches_fail_closed(change):
    x = deepcopy(_good())
    change(x)
    case = {"capture": CAPTURE, "positions": deepcopy(PORT), "notional": 100000.,
            "observed_mark_context": x}
    assert review_observed_mark(case) is False
    assert "observed_mark_context" not in case


def test_wrong_held_symbol_is_not_replaced_by_a_related_etf_or_benchmark():
    h = _history()
    h["QQQ"] = h.pop("BBB")
    out = observed_portfolio_mark(PORT, 100000., h, as_of_utc=CAPTURE)
    assert out["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"
    assert "BBB" in out["reason"]


def test_no_intraday_today_bar_may_be_passed_as_complete():
    h = _history()
    h["AAA"][-1]["date"] = "2026-11-01"
    out = observed_portfolio_mark(PORT, 100000., h, as_of_utc=CAPTURE)
    assert out["status"] == "INSUFFICIENT_COMPLETED_MARK_DATA"
