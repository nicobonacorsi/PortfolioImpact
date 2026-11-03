"""R118 exact paired YES/NO current-mark mechanics and public-export safety.

All economic returns below are declared TEST assumptions. A real portfolio
observation or source is not being claimed by synthetic fixtures.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from itertools import product
from math import isclose

import pytest

from castle_risk.joint_credal_impact import portfolio_signature
from castle_risk.observed_portfolio_mark import observed_portfolio_mark, review_observed_mark
from castle_risk.paired_today_mark_impact import (
    PairedTodayMarkError, evaluate_paired_today_mark, verified_saved_paired_today_mark)
from castle_risk.autonomous_event_impact import (
    forecast_portfolio_impacts, attach_autonomous_impacts, recheck_saved_autonomous_impacts)
from test_r116_joint_credal_impact import POSITIONS, _branches
from test_r116_autonomous_event_impact import FakeClient, SOURCE, _context, _case, EVENT, CAPTURE

NAV = 100000.


def _bars():
    day = date(2026, 6, 1)
    days = []
    while day <= date(2026, 10, 30):
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return {row["symbol"]: [
        {"date": d, "close": (200 if row["symbol"]=="TSM" else 100) *
                              (1 + .0005 * (i+1)) ** (1 if row["symbol"] in ("TSM","IEF") else 2)}
        for i,d in enumerate(days)] for row in POSITIONS}


def _mark():
    m = observed_portfolio_mark(POSITIONS, NAV, _bars(), as_of_utc=CAPTURE)
    assert m["status"] == "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
    return m


def _paired_branches():
    branches = _branches()
    for i, b in enumerate(branches):
        b["rationale"]="Explicit hypothesis of energy and geopolitical repricing channels."
        b["counterevidence"]="A different policy and supply regime could reverse the assumed shocks."
        b["source_ids"]=["W01"]
        b["assumed_onset_date"] = ["2026-11-20", "2026-12-15", "2027-01-25"][i]
        b["no_event_baseline_rationale"] = (
            "Synthetic no-event continuation with uncertain pre-onset drift. "
            "Past price sessions are not used as a predictive event coefficient.")
        b["no_event_current_mark_paths"] = {
            label: [
                {"symbol": p["symbol"],
                 "no_event_return_pct": float(
                     {"XOM": 2., "TSM": -1., "LMT": 3., "IEF": .5}[p["symbol"]] +
                     {"adverse": -3., "central": 0., "favorable": 3.}[label] +
                     i)}
                for p in POSITIONS
            ] for label in ("adverse","central","favorable")
        }
    return branches


def _evaluation():
    return evaluate_paired_today_mark(
        POSITIONS, NAV, _paired_branches(), as_of_utc=CAPTURE,
        deadline=EVENT["date"], observed_mark_context=_mark())


def test_paired_state_identity_and_exact_portfolio_weight_once():
    x=_evaluation()
    assert x["status"] == "MODEL_TODAY_MARK_STRESS_UNCALIBRATED"
    assert x["future_pnl_observed_or_identified"] is False
    assert x["pricing_q_estimated"] is False
    assert x["prediction_market_Q_used_as_equity_q"] is False
    assert x["forward_returns_calibrated"] is False
    assert x["physical_event_probability_used"] is False
    assert x["horizon_calendar_end_is_common_across_branches"] is False
    assert x["assumed_onset_dates"] == ["2026-11-20", "2026-12-15", "2027-01-25"]
    assert x["portfolio_signature"] == portfolio_signature(POSITIONS, NAV)
    assert x["paired_event_gap_center_pct"] == pytest.approx(-1.7)
    assert x["paired_event_gap_model_set_pct"] == pytest.approx([-4.645, .990])
    assert x["event_yes_from_today_center_pct"] - x["event_no_from_today_center_pct"] == pytest.approx(-1.7)
    assert x["event_yes_from_today_center_usd"] - x["event_no_from_today_center_usd"] == pytest.approx(-1700.)
    assert x["event_yes_from_today_center_usd"] != x["paired_event_gap_center_pct"]*NAV/100.
    assert x["event_yes_from_today_model_set_usd"][0] < x["event_yes_from_today_center_usd"]
    assert x["event_yes_from_today_center_usd"] < x["event_yes_from_today_model_set_usd"][1]
    assert verified_saved_paired_today_mark(x,POSITIONS,NAV,_paired_branches(),
        as_of_utc=CAPTURE,deadline=EVENT["date"],observed_mark_context=_mark())


def test_three_regime_gap_is_always_paired_before_LP_not_subtracted_extreme_bands():
    x=_evaluation()
    for b in x["conditional_branch_summaries"]:
        for label in ("adverse","central","favorable"):
            path=b["joint_paths"][label]
            assert path["event_yes_from_today_pct"] - path["event_no_from_today_pct"] == pytest.approx(path["event_gap_pct"])
            for sym in path["holding_yes_pct"]:
                assert path["holding_yes_pct"][sym] == pytest.approx(
                    path["holding_no_pct"][sym] + path["holding_gap_pct"][sym])
    # Differencing independently optimized marginal YES and NO bounds
    # would generally create an impossible or excessively wide event gap.
    y=x["event_yes_from_today_model_set_pct"]
    n=x["event_no_from_today_model_set_pct"]
    g=x["paired_event_gap_model_set_pct"]
    assert not (isclose(y[0]-n[1],g[0],abs_tol=1e-8) and
                isclose(y[1]-n[0],g[1],abs_tol=1e-8))


@pytest.mark.parametrize("corrupt", [
    lambda b: b[0]["no_event_current_mark_paths"]["central"].pop(),
    lambda b: b[0]["no_event_current_mark_paths"]["central"].append({"symbol":"XOM","no_event_return_pct":5}),
    lambda b: b[0]["no_event_current_mark_paths"]["central"][0].update(symbol="IEF"),
    lambda b: b[1]["no_event_current_mark_paths"]["central"][0].update(no_event_return_pct=float("nan")),
    lambda b: b[0]["no_event_current_mark_paths"]["central"][0].update(no_event_return_pct=True),
    lambda b: b[0]["no_event_current_mark_paths"]["central"][0].update(no_event_return_pct=-101),
    lambda b: b[0]["no_event_current_mark_paths"]["central"][0].update(no_event_return_pct=201),
    lambda b: b[0].update(assumed_onset_date="2026-10-30"),
    lambda b: b[0].update(assumed_onset_date="2027-01-31"),
    lambda b: b[0].update(assumed_onset_date="not a date"),
    lambda b: b[0].update(no_event_baseline_rationale="guess"),
    lambda b: b[0]["no_event_current_mark_paths"].pop("adverse"),
])
def test_fail_closed_for_unbounded_false_or_missing_future_premises(corrupt):
    branches = _paired_branches()
    corrupt(branches)
    with pytest.raises((PairedTodayMarkError,ValueError)):
        evaluate_paired_today_mark(POSITIONS,NAV,branches,
            as_of_utc=CAPTURE,deadline=EVENT["date"],observed_mark_context=_mark())


@pytest.mark.parametrize("corrupt", [
    lambda x: x.update(event_yes_from_today_center_usd=99999),
    lambda x: x["event_yes_from_today_model_set_usd"].__setitem__(0,-1),
    lambda x: x.update(event_no_from_today_center_pct=0),
    lambda x: x.update(pricing_q_estimated=True),
    lambda x: x.update(forward_returns_calibrated=True),
    lambda x: x.update(future_pnl_observed_or_identified=True),
    lambda x: x.update(event_return_caused_empirically=True),
    lambda x: x["conditional_branch_summaries"][0]["joint_paths"]["central"].update(event_yes_from_today_pct=999),
    lambda x: x["assumed_onset_dates"].__setitem__(1,"2027-01-30"),
    lambda x: x.update(market_context_last_completed_session="2026-09-01"),
])
def test_public_pnl_tampering_and_spurious_q_or_causal_claims_rejected(corrupt):
    x=deepcopy(_evaluation())
    corrupt(x)
    assert not verified_saved_paired_today_mark(x,POSITIONS,NAV,_paired_branches(),
        as_of_utc=CAPTURE,deadline=EVENT["date"],observed_mark_context=_mark())


def test_real_client_style_proposal_attaches_and_replays_both_marks_without_impact_A_change():
    model={"event_id":EVENT["id"],"status":"PROPOSED",
           "abstain_reason":"Not applicable","branches":_paired_branches()}
    client=FakeClient([model])
    mark=_mark()
    row=forecast_portfolio_impacts([_context()],POSITIONS,NAV,SOURCE,client,
        observed_mark_context=mark)[EVENT["id"]]
    assert row["status"]=="MODEL_STRESS_UNCALIBRATED"
    assert row["paired_today_mark"]["status"]=="MODEL_TODAY_MARK_STRESS_UNCALIBRATED"
    assert client.calls[0][1]["observed_portfolio_mark_not_event_attribution"][
        "equity_priced_in_fraction_identified"] is False
    case=_case()
    case["observed_mark_context"]=mark
    assert review_observed_mark(case)
    previous=deepcopy(case["risks"][0]["outcomes"])
    attach_autonomous_impacts(case,{EVENT["id"]:row})
    assert case["risks"][0]["outcomes"]==previous
    assert "paired_today_mark" in case["risks"][0]["autonomous_event_impact"]
    recheck_saved_autonomous_impacts(case)
    assert "paired_today_mark" in case["risks"][0]["autonomous_event_impact"]
    forged=deepcopy(case)
    forged["risks"][0]["autonomous_event_impact"]["paired_today_mark"][
        "event_yes_from_today_center_usd"]+=100
    recheck_saved_autonomous_impacts(forged)
    assert "autonomous_event_impact" not in forged["risks"][0]


def test_model_gap_retained_but_no_absolute_pnl_if_mark_missing():
    model={"event_id":EVENT["id"],"status":"PROPOSED",
           "abstain_reason":"Not applicable","branches":_paired_branches()}
    row=forecast_portfolio_impacts([_context()],POSITIONS,NAV,SOURCE,
        FakeClient([model]))[EVENT["id"]]
    assert row["status"]=="MODEL_STRESS_UNCALIBRATED"
    assert "paired_today_mark" not in row
    assert row["today_mark_sensitivity"]["event_yes_pnl_from_today_usd"] is None


def test_relabel_must_keep_baseline_and_gap_vectors_coupled():
    branches=_paired_branches()
    base=evaluate_paired_today_mark(POSITIONS,NAV,branches,as_of_utc=CAPTURE,
        deadline=EVENT["date"],observed_mark_context=_mark())
    b=branches[0]
    for field in ("joint_paths","no_event_current_mark_paths"):
        b[field]["adverse"],b[field]["central"]=b[field]["central"],b[field]["adverse"]
    model={"event_id":EVENT["id"],"status":"PROPOSED",
           "abstain_reason":"Not applicable","branches":branches}
    row=forecast_portfolio_impacts([_context()],POSITIONS,NAV,SOURCE,
        FakeClient([model]),observed_mark_context=_mark())[EVENT["id"]]
    assert row["status"]=="MODEL_STRESS_UNCALIBRATED"
    assert row["joint_path_relabelings"][0]["branch_id"]=="S1"
    assert row["paired_today_mark"]["event_yes_from_today_center_pct"] == pytest.approx(
        base["event_yes_from_today_center_pct"])
    assert row["paired_today_mark"]["paired_event_gap_center_pct"] == pytest.approx(-1.7)


def test_all_three_paired_model_bounds_match_independent_vertex_enumeration():
    """Independently enumerate simplex vertices AND all complete regimes.

    The production optimizer uses greedy signed LP filling. This proof test
    uses explicit active-constraint vertices without calling that algorithm.
    """
    x = _evaluation()
    summaries = x["conditional_branch_summaries"]
    lows = [r["weight_given_event"][0] for r in summaries]
    highs = [r["weight_given_event"][2] for r in summaries]
    vertices = []
    for free in range(len(summaries)):
        bound_indices = [i for i in range(len(summaries)) if i != free]
        for values in product((0, 1), repeat=len(bound_indices)):
            point = [0.] * len(summaries)
            for i, choice in zip(bound_indices, values):
                point[i] = [lows[i], highs[i]][choice]
            point[free] = 1. - sum(point)
            if lows[free]-1e-10 <= point[free] <= highs[free]+1e-10:
                vertices.append(point)
    assert len(vertices) >= 3
    labels = ("adverse", "central", "favorable")
    for field, bounds in (
        ("event_yes_from_today_pct", x["event_yes_from_today_model_set_pct"]),
        ("event_no_from_today_pct", x["event_no_from_today_model_set_pct"]),
        ("event_gap_pct", x["paired_event_gap_model_set_pct"]),
    ):
        outcomes = []
        for weights in vertices:
            for regimes in product(labels, repeat=len(summaries)):
                outcomes.append(sum(
                    weights[i] * summaries[i]["joint_paths"][regimes[i]][field]
                    for i in range(len(summaries))))
        assert min(outcomes) == pytest.approx(bounds[0], abs=1e-9)
        assert max(outcomes) == pytest.approx(bounds[1], abs=1e-9)


def test_paired_model_with_cash_and_one_etf_counts_original_nav_once():
    """Independent cash zero and ETF one-fund-NAV safety for absolute P&L."""
    positions = [
        {"symbol":"USD", "type":"cash", "weight":.2},
        {"symbol":"QQQ", "type":"etf", "weight":.5},
        {"symbol":"NVDA", "type":"stock", "weight":.3},
    ]
    mock_bars = _bars()
    marks = observed_portfolio_mark(
        positions, NAV, {"QQQ":mock_bars["XOM"], "NVDA":mock_bars["TSM"]},
        as_of_utc=CAPTURE)
    assert marks["status"] == "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION"
    assert marks["holding_recent_20_session_mark_change_pct_current_nav"]["USD"] == 0.
    branches = []
    weights = [(.3,.55,.8), (.2,.45,.7)]
    for idx in range(2):
        gaps = ({"QQQ":(-8.,-5.,2.),"NVDA":(-15.,-7.,5.)} if idx==0
                else {"QQQ":(-3.,1.,4.),"NVDA":(-5.,2.,8.)})
        gap_vec, no_vec = {}, {}
        for j, label in enumerate(("adverse","central","favorable")):
            gap_vec[label] = [
                {"symbol":sym, "gap_pct":float(gaps[sym][j]) if sym!="USD" else 0.}
                for sym in ("USD","QQQ","NVDA")]
            no_vec[label] = [
                {"symbol":sym, "no_event_return_pct":
                 (float((0.,2.,4.)[j] + idx) if sym=="QQQ" else
                  float((-2.,1.,3.)[j] + idx) if sym=="NVDA" else 0.)}
                for sym in ("USD","QQQ","NVDA")]
        branches.append({
            "id":"A" if idx==0 else "RESIDUAL",
            "trigger":"Hypothetical complete branch "+str(idx),
            "residual":idx==1,
            "weight_given_event":dict(zip(("low","center","high"),weights[idx])),
            "joint_paths":gap_vec,
            "assumed_onset_date":"2026-12-01" if idx==0 else "2027-01-15",
            "no_event_baseline_rationale":
                "A hypothetical no-event market path including uncertain forward drift.",
            "no_event_current_mark_paths":no_vec,
        })
    x = evaluate_paired_today_mark(
        positions,NAV,branches,as_of_utc=CAPTURE,
        deadline=EVENT["date"],observed_mark_context=marks)
    assert x["status"]=="MODEL_TODAY_MARK_STRESS_UNCALIBRATED"
    assert x["original_etf_constituents_added_to_nav"] is False
    for row in x["conditional_branch_summaries"]:
        for label in ("adverse","central","favorable"):
            joined = row["joint_paths"][label]
            assert joined["holding_yes_pct"]["USD"] == 0
            assert joined["holding_no_pct"]["USD"] == 0
            assert joined["holding_gap_pct"]["USD"] == 0
            assert joined["event_yes_from_today_pct"] == pytest.approx(
                .5*joined["holding_yes_pct"]["QQQ"]+
                .3*joined["holding_yes_pct"]["NVDA"])
    bad = deepcopy(branches)
    bad[0]["no_event_current_mark_paths"]["central"][0]["no_event_return_pct"]=1.
    with pytest.raises(PairedTodayMarkError):
        evaluate_paired_today_mark(positions,NAV,bad,
            as_of_utc=CAPTURE,deadline=EVENT["date"],observed_mark_context=marks)


def test_expanded_R118_response_budget_uses_one_original_event_per_api_call():
    """A two-event R118 schema response can exhaust high-effort output tokens.

    The larger coupled YES/NO vectors must be requested in separate complete
    bounded calls and must never silently borrow another event's model.
    """
    from castle_risk.probability_assumptions import probability_question_context

    second={"id":"SECOND_HORMUZ_EVENT","date":"2027-01-30",
            "title":"Hormuz tanker transit interrupted for at least seven days by 2027-01-30"}
    second_context=probability_question_context({"capture":CAPTURE},second)
    assert second_context is not None
    class OneAtATime:
        model="injected-one-at-a-time"
        def __init__(self):
            self.calls=[]
        def structured(self,name,schema,payload,system=""):
            targets=payload["original_event_targets"]
            assert len(targets)==1
            self.calls.append(targets[0]["event_id"])
            return {"assessments":[{
                "event_id":targets[0]["event_id"],"status":"ABSTAIN",
                "abstain_reason":"No sound conditional shock magnitudes are available.",
                "branches":[]}]}
    client=OneAtATime()
    outcome=forecast_portfolio_impacts([_context(),second_context],
        POSITIONS,NAV,SOURCE,client,observed_mark_context=_mark())
    assert client.calls==[EVENT["id"],second["id"]]
    assert all(outcome[event_id]["status"]=="ABSTAIN" for event_id in client.calls)
