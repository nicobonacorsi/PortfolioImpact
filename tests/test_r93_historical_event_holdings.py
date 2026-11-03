"""R93: source-event matching, 16-holder calculation and falsification gates."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from castle_risk.r92_factor_bridge import HOLDINGS
from castle_risk.r93_historical_event_holdings import (
    SCHEMA, evaluate_historical_event_holdings,
)
from tests.test_r92_factor_bridge import synthetic_holdings


def long_event_mock() -> tuple[dict, dict, list]:
    rng = np.random.default_rng(9301)
    portfolios = synthetic_holdings()[0]["r87_portfolios"]
    prices = {t: {"2016-11-01": 100.0, "2026-10-29": 100.0}
              for t in HOLDINGS}
    source_events = []
    for year in range(2017, 2027):
        for month in range(1, 9):
            event, prior = f"{year}-{month:02d}-15", f"{year}-{month:02d}-14"
            ed4, ff1 = float(rng.normal(0, .02)), float(rng.normal(0, .013))
            sp, ust = 3*ed4+float(rng.normal(0, .015)), .8*ed4+ff1
            source_events.append({"Date": event, "Unscheduled": 0,
                                  "ED4": ed4, "FF1": ff1,
                                  "SP500": sp, "UST10Y": ust})
            for ticker in HOLDINGS:
                response = 1.2*ed4 + 2.5*ff1 + float(rng.normal(0, .001))
                prices[ticker][prior] = 100.0
                prices[ticker][event] = 100*(1+response)
    return prices, portfolios, source_events


def test_synthetic_80_events_all_exact_securities():
    prices, portfolios, events = long_event_mock()
    result = evaluate_historical_event_holdings(prices, portfolios, events)
    assert result["schema"] == SCHEMA
    assert result["train_count"] == 48 and result["holdout_count"] == 32
    assert len(result["individual_instruments"]) == 16
    assert len(result["portfolios"]) == 4
    assert result["all_four_ed4_ff1_vs_mean_pass_5pct"]
    assert not result["conditional_Kalshi_yes_portfolio_pnl_identified"]
    assert not result["risk_already_discounted_share_identified"]
    assert set(result["portfolios"]["new_diversified"]["wrong_meeting_placebo"]) == {"1", "5"}


def test_unpaired_event_is_excluded_without_imputation():
    prices, portfolios, events = long_event_mock()
    events.append({"Date": "2026-09-16", "Unscheduled": 0,
                   "ED4": .03, "FF1": -.02, "SP500": .1, "UST10Y": .01})
    result = evaluate_historical_event_holdings(prices, portfolios, events)
    assert result["excluded_unmatched_market_calendar"] == 1
    assert result["holdout_count"] == 32


def test_events_with_large_unadjusted_split_like_price_jump_are_screened():
    prices, portfolios, events = long_event_mock()
    prices["USO"]["2025-08-15"] = 240
    outcome = evaluate_historical_event_holdings(prices, portfolios, events)
    assert len(outcome["excluded_gt35pct_move_event_days"]) == 1
    assert outcome["excluded_gt35pct_move_event_days"][0]["date"] == "2025-08-15"
    assert "USO" in outcome["excluded_gt35pct_move_event_days"][0]["extreme_tickers"]
    assert outcome["holdout_count"] == 31


def test_unscheduled_and_missing_factor_exclusions():
    prices, portfolios, events = long_event_mock()
    events[0]["Unscheduled"] = 1
    events[1]["ED4"] = None
    outcome = evaluate_historical_event_holdings(prices, portfolios, events)
    assert outcome["excluded_unscheduled_events"] == 1
    assert outcome["excluded_missing_event_factors"] == 1


def test_duplicate_event_rejected():
    prices, portfolios, events = long_event_mock()
    events.append(deepcopy(events[-1]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_historical_event_holdings(prices, portfolios, events)


def test_missing_held_instrument_rejected():
    prices, portfolios, events = long_event_mock()
    del prices["AGG"]
    with pytest.raises(ValueError, match="16"):
        evaluate_historical_event_holdings(prices, portfolios, events)


def test_bad_or_too_few_events_rejected():
    prices, portfolios, events = long_event_mock()
    with pytest.raises(ValueError, match="coverage"):
        evaluate_historical_event_holdings(prices, portfolios, events[:37])
