"""Synthetic forward-impact OOS math and leakage tests; no empirical skill claim."""
from __future__ import annotations

from copy import deepcopy

import pytest

from castle_risk.impact_oos_score import (
    ImpactOosInputError, score_forward_portfolio_impacts,
)


def _case():
    return {
        "case_id": "case-1", "event_family_id": "release_2027_01",
        "event_condition": "US central bank raises the target rate at scheduled date",
        "status": "MODEL_ASSUMPTION_UNCALIBRATED", "model_version": "frozen-fixture-v1",
        "model_response_received_at_utc": "2026-11-02T00:00:00Z",
        "opening_mark_recorded_at_utc": "2026-11-01T20:00:00Z",
        "fixed_price_return_end_session_date": "2027-02-10",
        "original_notional_usd": 100000,
        "positions": [
            {"symbol":"AAPL","weight":.6,"opening_split_adjusted_close":100.0,
             "predicted_yes_return_pct":5.0,"predicted_no_return_pct":0.0},
            {"symbol":"TLT","weight":.25,"opening_split_adjusted_close":50.0,
             "predicted_yes_return_pct":-4.0,"predicted_no_return_pct":0.0},
            {"symbol":"USD","weight":.15,"opening_split_adjusted_close":1.0,
             "predicted_yes_return_pct":0.0,"predicted_no_return_pct":0.0},
        ],
    }


def _lock(events):
    return {"schema": "castle.prospective_impact_lock.v1",
            "locked_at_utc": "2026-11-02T00:01:00Z", "events": events}


def _outcome(event, y=1):
    return {
        "case_id":event["case_id"],"event_condition":event["event_condition"],
        "outcome":y,
        "resolved_at_utc":"2027-01-30T00:00:00Z",
        "resolution_source_kind":"OFFICIAL_EVENT_ADJUDICATION",
        "resolution_source_url":"https://www.federalreserve.gov/monetarypolicy/",
        "end_close_session_date":"2027-02-10",
        "end_prices_observed_at_utc":"2027-02-11T01:00:00Z",
        "price_data_source":"REAL_SPLIT_ADJUSTED_PROVIDER_RECEIPT_FIXTURE_ONLY",
        "observed_closing_prices":[{"symbol":"AAPL","split_adjusted_close":105.0},
                                   {"symbol":"TLT","split_adjusted_close":48.0}],
    }


def _score(lock, actual):
    return score_forward_portfolio_impacts(
        lock, actual, audit_time_utc="2027-02-12T00:00:00Z",
        minimum_independent_event_families=2)


def test_pending_impact_has_no_out_of_sample_skill():
    result=_score(_lock([_case()]),[])
    assert result["status"]=="PENDING_NO_RESOLVED_FIXED_HORIZONS"
    assert result["model_rmse_pct_nav"] is None
    assert result["unobserved_causal_counterfactual_identified"] is False


def test_realized_yes_price_pnl_and_zero_return_baseline_arithmetic():
    event=_case()
    out=_score(_lock([event]),[_outcome(event)])
    assert out["scored_cases"]==1
    row=out["score_rows"][0]
    assert row["observed_price_return_pct"]==pytest.approx(2.0)
    assert row["observed_pnl_usd"]==pytest.approx(2000.0)
    assert row["predicted_price_return_pct"]==pytest.approx(2.0)
    assert row["predicted_pnl_usd"]==pytest.approx(2000.0)
    assert out["model_mae_pct_nav"]==pytest.approx(0.0)
    assert out["zero_change_mae_pct_nav"]==pytest.approx(2.0)
    assert out["event_yes_minus_no_causal_effect_validated"] is False
    assert out["realized_branch_only_is_scored"] is True


def test_no_branch_scoring_uses_only_locked_no_vector_not_yes_gap():
    event=_case()
    out=_score(_lock([event]),[_outcome(event,y=0)])
    row=out["score_rows"][0]
    assert row["realized_event_state"]==0
    assert row["predicted_pnl_usd"]==0
    assert row["observed_pnl_usd"]==pytest.approx(2000)
    assert out["model_mae_pct_nav"]==pytest.approx(2)


@pytest.mark.parametrize("edit",[
    lambda e,r:e["positions"][0].update(weight=.5),
    lambda e,r:e["positions"][0].update(opening_split_adjusted_close=0),
    lambda e,r:e["positions"][2].update(predicted_yes_return_pct=1),
    lambda e,r:e.update(model_response_received_at_utc="2027-02-01T00:00:00Z"),
    lambda e,r:e.update(fixed_price_return_end_session_date="2026-10-31"),
    lambda e,r:r.update(outcome=True),
    lambda e,r:r.update(event_condition="A different central bank action"),
    lambda e,r:r.update(end_close_session_date="2027-02-09"),
    lambda e,r:r.update(end_prices_observed_at_utc="2027-01-01T00:00:00Z"),
    lambda e,r:r["observed_closing_prices"].pop(1),
    lambda e,r:r["observed_closing_prices"][0].update(split_adjusted_close=float("nan")),
])
def test_incomplete_or_leaky_impact_rows_are_rejected(edit):
    event=_case()
    actual=_outcome(event)
    edit(event,actual)
    with pytest.raises(ImpactOosInputError):
        _score(_lock([event]),[actual])


def test_duplicate_case_ids_and_mixed_model_versions_rejected():
    a=_case()
    with pytest.raises(ImpactOosInputError):
        _score(_lock([a,deepcopy(a)]),[])
    b=deepcopy(a)
    b["case_id"]="case-2"
    b["model_version"]="new-after-results"
    with pytest.raises(ImpactOosInputError):
        _score(_lock([a,b]),[])


def test_abstention_is_in_denominator_and_not_backfilled():
    a=_case()
    b=deepcopy(a)
    b["case_id"]="case-2"
    b["status"]="ABSTAIN"
    for p in b["positions"]:
        p.pop("predicted_yes_return_pct")
        p.pop("predicted_no_return_pct")
    out=_score(_lock([a,b]),[_outcome(a),_outcome(b)])
    assert out["resolved_cases"]==2
    assert out["scored_cases"]==1
    assert out["resolved_abstentions"]==1
