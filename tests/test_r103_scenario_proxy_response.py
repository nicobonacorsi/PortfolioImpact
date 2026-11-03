"""Synthetic accounting and frozen-admission checks, not economic backtests."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pytest

from castle_risk.portfolio import parse_portfolio
from castle_risk.scenario_proxy_response import fit_proxy_response, build_proxy_models
from castle_risk.scenario_response_surface import attach_scenario_responses, build_response_model
from castle_risk.live_assessment import public_assessment_view
from tests.test_r103_scenario_response_surface import _history


def _proxy_data(signal=True):
    _, spy, yields, split = _history()
    rng = np.random.default_rng(92031)
    x = rng.normal(0, .015, len(spy))
    proxy = 100 * np.cumprod(1+x)
    market = np.array([r["close"] for r in spy])
    s = np.r_[0., market[1:]/market[:-1]-1]
    yld = np.array([r["yield_pct"] for r in yields])
    r = np.r_[0., np.diff(yld)*100]
    instrument_return = .8*s - .0002*r + (1.2*x if signal else 0.) + rng.normal(0, .00001, len(spy))
    instrument = 100*np.cumprod(1+instrument_return)
    prices = [{"date": row["date"], "close": float(v)} for row,v in zip(spy,instrument)]
    proxy = [{"date": row["date"], "close": float(v)} for row,v in zip(spy,proxy)]
    return prices, spy, yields, proxy, split


def test_incremental_proxy_signal_passes_frozen_comparators():
    prices, spy, yields, proxy, split = _proxy_data()
    fit = fit_proxy_response(prices, spy, yields, proxy, train_end=split)
    assert fit["admitted"]
    assert fit["proxy_loading"] == pytest.approx(1.2, abs=.001)
    assert fit["holdout_rmse"] < .001 * fit["holdout_two_factor_rmse"]
    assert fit["admission_period_is_not_an_independent_post_selection_test"]
    assert not fit["forecast_validated"]


def test_malformed_proxy_observation_is_rejected_without_bridging():
    prices, spy, yields, proxy, split = _proxy_data()
    proxy[12] = None
    portfolio = parse_portfolio([{"symbol": "NEW", "weight": 1., "asset_type": "stock"}])
    result = build_proxy_models(portfolio, {"NEW": prices, "SPY": spy}, {"USO": proxy}, yields,
        train_end=split, evaluated_at=datetime(2026, 10, 31, tzinfo=timezone.utc))
    assert result["USO"]["positions"]["NEW"]["status"] == "INVALID_SOURCE_SERIES"
    assert result["USO"]["scale_pct"] is None


def test_missing_rates_returns_a_diagnostic_not_a_build_failure():
    prices, spy, _, proxy, split = _proxy_data()
    result = fit_proxy_response(prices, spy, [], proxy, train_end=split)
    assert result["status"] == "INSUFFICIENT_HISTORY"
    assert not result["admitted"]
    assert result["n_train"] == result["n_holdout"] == 0


def test_unhelpful_proxy_rejected_and_metrics_retained():
    prices, spy, yields, proxy, split = _proxy_data(signal=False)
    fit = fit_proxy_response(prices, spy, yields, proxy, train_end=split)
    assert not fit["admitted"]
    assert fit["status"] == "PROXY_HOLDOUT_FAILED"
    assert fit["holdout_two_factor_rmse"] > 0
    assert fit["holdout_training_mean_rmse"] > 0


def test_collinear_proxy_does_not_crash_or_invent_loading():
    prices, spy, yields, _, split = _proxy_data()
    fit = fit_proxy_response(prices, spy, yields, spy, train_end=split)
    assert fit["status"] == "RANK_DEFICIENT_DESIGN"
    assert not fit["admitted"]


def test_own_proxy_identity_and_unknown_positions_remain_distinct():
    prices, spy, yields, proxy, split = _proxy_data()
    p = parse_portfolio([{"symbol": "USO", "weight": .6, "asset_type": "etf"},
                         {"symbol": "UNKNOWN", "weight": .4, "asset_type": "stock"}])
    fits = build_proxy_models(p, {"SPY": spy}, {"USO": proxy}, yields,
        train_end=split, evaluated_at=datetime(2026,10,31,tzinfo=timezone.utc))
    assert fits["USO"]["positions"]["USO"]["proxy_loading"] == 1
    assert not fits["USO"]["positions"]["UNKNOWN"]["admitted"]
    assert fits["GLD"]["status"] == "PROXY_PRICES_UNAVAILABLE"


def test_proxy_event_routing_and_original_position_accounting():
    prices, spy, yields, proxy, split = _proxy_data()
    p = parse_portfolio([{"symbol": "NEW", "weight": .4, "asset_type": "stock"},
                         {"symbol": "USO", "weight": .6, "asset_type": "etf"}])
    source = {"adjustment": "split_adjusted_close_not_dividend_total_return",
              "receipts": [{"source_url": "https://example.invalid/synthetic-prices", "source_response_sha256": "a"*64,
                            "retrieved_at": "2026-10-31T00:00:00+00:00"}],
              "raw_rows": [{"close": 99.}]}
    model = build_response_model(p, prices_by_symbol={"SPY": spy,"NEW":prices,"USO":proxy},
        treasury_yields=yields, two_factor_responses={}, evaluated_at=datetime(2026,10,31,tzinfo=timezone.utc),
        train_end=split, price_sources={"SPY": source, "NEW": source, "USO": source},
        proxy_prices={"USO": spy}, proxy_sources={"USO": {"receipts": [{"source_response_sha256": "b"*64}]}},
        treasury_source={"series": "US_TREASURY_10Y_CONSTANT_MATURITY_PERCENT", "receipts": source["receipts"]})
    risk = {"id":"oil-supply", "family":"OPEC_SUPPLY", "exposure_basis":"NAMED_SOURCE_CHANNELS",
            "exposure":[{"symbol":"USO","weight":.6}],
            "quality":{"definition_review":{"status":"DEFINED_FOR_REVIEW"}}, "outcomes":[]}
    report = {"portfolio":p.to_dict(),"scenario_response_model":model,"evaluated_at_utc":"2026-10-31T00:00:00+00:00"}
    attach_scenario_responses([risk],report)
    response = risk["scenario_response"]
    assert response["level"] == "B"
    assert response["linkage"]["status"] == "DOCUMENTED_EVENT_PROXY_HYPOTHESIS"
    evidence = response["response_evidence"]
    assert evidence["proxy_sources"]["USO"]["receipts"] == source["receipts"]
    assert evidence["factor_sources"]["US10Y"]["receipts"] == source["receipts"]
    assert evidence["position_sources"]["NEW"]["adjustment"] == source["adjustment"]
    assert "raw_rows" not in json.dumps(evidence)
    assert len(response["scenarios"]) == 2
    case = response["scenarios"][0]
    assert case["shock"]["factor"] == "USO_RETURN_PCT"
    assert case["covered_weight"] == pytest.approx(1.)
    assert case["pnl"] == pytest.approx(sum(r["pnl"] for r in case["position_contributions"]))
    assert not any(row["reason"] == "NO_SOURCE_LINKED_POSITION_TRANSMISSION" for row in response["missing_channels"])
    risk2 = {**deepcopy(risk),"family":"UNKNOWN"}
    attach_scenario_responses([risk2],report)
    assert risk2["scenario_response"]["linkage"]["status"] == "DOCUMENTED_NAMED_HOLDING_COMPONENT"
    assert all(c["shock"]["factor"] == "LINKED_HOLDING_RETURN_PCT" for c in risk2["scenario_response"]["scenarios"])
    # A valid measured proxy does not manufacture the event's portfolio link.
    absent = {**deepcopy(risk), "exposure": [{"symbol": "ABSENT", "weight": 1.}]}
    attach_scenario_responses([absent], report)
    assert absent["scenario_response"]["level"] == "C"
    assert absent["scenario_response"]["scenarios"] == []
    understated = {**deepcopy(risk), "exposure": [{"symbol": "USO", "weight": .2}]}
    attach_scenario_responses([understated], report)
    assert understated["scenario_response"]["level"] == "C"


def test_holdout_never_refits_extension_coefficients():
    prices, spy, yields, proxy, split = _proxy_data()
    first = fit_proxy_response(prices,spy,yields,proxy,train_end=split)
    changed = [{**r,"close":r["close"]*5 if r["date"]>split else r["close"]} for r in prices]
    second = fit_proxy_response(changed,spy,yields,proxy,train_end=split)
    assert first["proxy_loading"] == second["proxy_loading"]
    assert first["covariance"] == second["covariance"]


def test_private_proxy_price_rows_are_not_exported():
    report = {"exposures":{},"events":[],"scenario_proxy_prices":{"USO":[{"date":"2025-01-01","close":50.}]}}
    public = public_assessment_view(report)
    assert "scenario_proxy_prices" not in public
