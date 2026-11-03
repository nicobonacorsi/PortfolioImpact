"""R122 common-endpoint paired YES/NO scenarios: exact arithmetic, LP, calendar, fail-closed."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from itertools import product
from math import fsum, isclose

import pytest

from castle_risk.common_endpoint_impact import (
    CommonEndpointError, REFERENCE_SCHEMA, SCHEMA, canonicalize_regimes,
    endpoint_signature, evaluate_common_endpoint_impact, historical_horizon_reference,
    verified_saved_common_endpoint,
)
from castle_risk.us_trading_calendar import (
    TradingCalendarError, common_endpoint_for_event, is_trading_session, sessions_after,
    sessions_strictly_between,
)

POSITIONS = [{"symbol": "XOM", "weight": .30, "type": "stock"}, {"symbol": "TSM", "weight": .25, "type": "stock"},
             {"symbol": "LMT", "weight": .25, "type": "stock"}, {"symbol": "IEF", "weight": .20, "type": "etf"}]
CASH_POSITIONS = [{"symbol": "NVDA", "weight": .35, "type": "stock"}, {"symbol": "QQQ", "weight": .30, "type": "etf"},
                  {"symbol": "TLT", "weight": .20, "type": "etf"}, {"symbol": "USD", "weight": .15, "type": "cash"}]
AS_OF = "2026-11-02T03:30:00+00:00"
DEADLINE = "2027-01-30"
MARK = "2026-10-30"


def no_paths(symbols, central, spread=3.):
    return {"adverse": [{"symbol": s, "no_event_return_pct": (0. if s == "USD" else central[s] - spread)} for s in symbols],
            "central": [{"symbol": s, "no_event_return_pct": central[s]} for s in symbols],
            "favorable": [{"symbol": s, "no_event_return_pct": (0. if s == "USD" else central[s] + spread)} for s in symbols]}


def gap_paths(symbols, central, spread=2.):
    return {"adverse": [{"symbol": s, "gap_pct": (0. if s == "USD" else central[s] - spread)} for s in symbols],
            "central": [{"symbol": s, "gap_pct": central[s]} for s in symbols],
            "favorable": [{"symbol": s, "gap_pct": (0. if s == "USD" else central[s] + spread)} for s in symbols]}


def fixture():
    symbols = [p["symbol"] for p in POSITIONS]
    nop = no_paths(symbols, {"XOM": 2., "TSM": 4., "LMT": 1., "IEF": .5})
    branches = [
        {"id": "S1", "trigger": "Short interruption, resolved within two weeks", "residual": False,
         "weight_given_event": {"low": .2, "center": .35, "high": .5},
         "joint_paths": gap_paths(symbols, {"XOM": 3., "TSM": -8., "LMT": 4., "IEF": 1.})},
        {"id": "S2", "trigger": "Prolonged interruption with naval escort regime", "residual": False,
         "weight_given_event": {"low": .3, "center": .45, "high": .6},
         "joint_paths": gap_paths(symbols, {"XOM": 8., "TSM": -25., "LMT": 9., "IEF": 2.5})},
        {"id": "S3", "trigger": "Residual: all other realizations of the event", "residual": True,
         "weight_given_event": {"low": .1, "center": .2, "high": .35},
         "joint_paths": gap_paths(symbols, {"XOM": 5., "TSM": -40., "LMT": 6., "IEF": 4.})},
    ]
    return nop, branches


def evaluate(positions=POSITIONS, nop=None, branches=None, **overrides):
    base_no, base_branches = fixture()
    kwargs = dict(as_of_utc=AS_OF, deadline=DEADLINE, endpoint_rule="WINDOW_DEADLINE_PLUS_20_SESSIONS",
                  last_completed_session=MARK)
    kwargs.update(overrides)
    return evaluate_common_endpoint_impact(positions, 100_000., nop or base_no, branches or base_branches, **kwargs)


def brute_force(positions, nop, branches, metric):
    """Independent extrema: enumerate all vertices of the bounded simplex per regime."""
    weights = {p["symbol"]: p["weight"] for p in positions}
    lows = [b["weight_given_event"]["low"] for b in branches]
    highs = [b["weight_given_event"]["high"] for b in branches]
    n = len(branches)
    values = {}
    for label in ("adverse", "central", "favorable"):
        no_total = fsum(weights[e["symbol"]] * e["no_event_return_pct"] for e in nop[label])
        gaps = [fsum(weights[e["symbol"]] * e["gap_pct"] for e in b["joint_paths"][label]) for b in branches]
        values[label] = (no_total, gaps)
    candidates = []
    for label, (no_total, gaps) in values.items():
        # Vertices of {l<=p<=u, sum p=1}: all but one coordinate at a bound, the remaining one determined.
        for free in range(n):
            for corner in product(*[(lows[i], highs[i]) for i in range(n) if i != free]):
                p = list(corner)
                p.insert(free, 1 - fsum(corner))
                if not lows[free] - 1e-12 <= p[free] <= highs[free] + 1e-12:
                    continue
                mix = fsum(pi * g for pi, g in zip(p, gaps))
                candidates.append(mix + (no_total if metric == "yes" else 0.))
    return min(candidates), max(candidates)


def test_calendar_matches_official_nyse_dates_and_r120_scheduled_convention():
    assert common_endpoint_for_event(date(2026, 11, 10), "SCHEDULED_RELEASE_PLUS_2_SESSIONS") == date(2026, 11, 12)
    assert common_endpoint_for_event(date(2026, 12, 9), "SCHEDULED_RELEASE_PLUS_2_SESSIONS") == date(2026, 12, 11)
    # Twenty regular sessions after Saturday 2027-01-30, with Washington's Birthday (2027-02-15) closed.
    assert common_endpoint_for_event(date(2027, 1, 30), "WINDOW_DEADLINE_PLUS_20_SESSIONS") == date(2027, 3, 1)
    assert sessions_strictly_between(date(2027, 1, 30), date(2027, 2, 26)) == 19
    assert not is_trading_session(date(2027, 2, 15)) and not is_trading_session(date(2026, 11, 26))
    assert is_trading_session(date(2026, 11, 27))  # early close is still a session
    assert sessions_after(date(2026, 12, 24), 1) == date(2026, 12, 28)
    with pytest.raises(TradingCalendarError):
        is_trading_session(date(2025, 12, 31))
    with pytest.raises(TradingCalendarError):
        common_endpoint_for_event(date(2027, 1, 30), "SOMETHING_ELSE")


def test_yes_equals_no_plus_gap_for_every_holding_regime_and_branch_and_dollars_reconcile():
    r = evaluate()
    assert r["schema"] == SCHEMA and r["status"] == "MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED"
    assert r["common_endpoint_session_date"] == "2027-03-01"
    assert r["sessions_from_last_completed_mark_to_endpoint"] == 81
    assert r["horizon_calendar_end_is_common_across_branches"] is True
    weights = {p["symbol"]: p["weight"] for p in POSITIONS}
    for summary in r["conditional_branch_summaries"]:
        for label, regime in summary["regimes"].items():
            for symbol in weights:
                assert isclose(regime["holding_yes_pct"][symbol],
                               regime["holding_no_pct"][symbol] + regime["holding_gap_pct"][symbol], abs_tol=1e-12)
            assert isclose(regime["event_yes_from_today_pct"] - regime["event_no_from_today_pct"],
                           regime["event_gap_pct"], abs_tol=1e-9)
            assert isclose(regime["event_no_from_today_pct"], r["no_event_regime_portfolio_pct"][label], abs_tol=1e-12)
    assert isclose(r["event_yes_from_today_center_pct"] - r["event_no_from_today_center_pct"],
                   r["paired_event_gap_center_pct"], abs_tol=1e-9)
    assert isclose(r["event_yes_from_today_center_usd"], 1000 * r["event_yes_from_today_center_pct"], abs_tol=1e-6)
    # Independent hand computation: central NO = .3*2+.25*4+.25*1+.2*.5 = 1.95%; gaps per branch central
    assert isclose(r["event_no_from_today_center_pct"], 1.95, abs_tol=1e-12)
    g1 = .3 * 3 + .25 * -8 + .25 * 4 + .2 * 1
    g2 = .3 * 8 + .25 * -25 + .25 * 9 + .2 * 2.5
    g3 = .3 * 5 + .25 * -40 + .25 * 6 + .2 * 4
    assert isclose(r["paired_event_gap_center_pct"], .35 * g1 + .45 * g2 + .2 * g3, abs_tol=1e-12)
    assert isclose(r["event_yes_from_today_center_pct"], 1.95 + .35 * g1 + .45 * g2 + .2 * g3, abs_tol=1e-12)


def test_exact_lp_bounds_match_independent_vertex_enumeration():
    nop, branches = fixture()
    r = evaluate()
    lo, hi = brute_force(POSITIONS, nop, branches, "yes")
    assert isclose(r["event_yes_from_today_model_set_pct"][0], lo, abs_tol=1e-9)
    assert isclose(r["event_yes_from_today_model_set_pct"][1], hi, abs_tol=1e-9)
    lo, hi = brute_force(POSITIONS, nop, branches, "gap")
    assert isclose(r["paired_event_gap_model_set_pct"][0], lo, abs_tol=1e-9)
    assert isclose(r["paired_event_gap_model_set_pct"][1], hi, abs_tol=1e-9)
    assert r["event_yes_from_today_model_set_pct"][0] <= r["event_yes_from_today_center_pct"] <= r["event_yes_from_today_model_set_pct"][1]
    assert r["event_no_from_today_model_set_pct"] == [r["no_event_regime_portfolio_pct"]["adverse"],
                                                      r["no_event_regime_portfolio_pct"]["favorable"]]


def test_regime_relabeling_transports_branch_gaps_with_the_no_vector():
    nop, branches = fixture()
    swapped_no = {"adverse": nop["favorable"], "central": nop["central"], "favorable": nop["adverse"]}
    swapped_branches = deepcopy(branches)
    for b in swapped_branches:
        b["joint_paths"] = {"adverse": b["joint_paths"]["favorable"], "central": b["joint_paths"]["central"],
                            "favorable": b["joint_paths"]["adverse"]}
    canonical_no, canonical_branches, audit = canonicalize_regimes(POSITIONS, swapped_no, swapped_branches)
    assert canonical_no == nop and canonical_branches == branches
    assert audit[0]["original_label_for_canonical"] == {"adverse": "favorable", "central": "central", "favorable": "adverse"}
    assert evaluate(nop=swapped_no, branches=swapped_branches)["event_yes_from_today_model_set_pct"] == \
        evaluate()["event_yes_from_today_model_set_pct"]


def test_cash_is_zero_in_both_states_and_nonzero_cash_fails_closed():
    symbols = [p["symbol"] for p in CASH_POSITIONS]
    nop = no_paths(symbols, {"NVDA": 5., "QQQ": 3., "TLT": 1., "USD": 0.})
    branches = [
        {"id": "A", "trigger": "Enforced quarantine lasting under a month", "residual": False,
         "weight_given_event": {"low": .4, "center": .6, "high": .8},
         "joint_paths": gap_paths(symbols, {"NVDA": -20., "QQQ": -10., "TLT": 4., "USD": 0.})},
        {"id": "B", "trigger": "Residual: all other realizations", "residual": True,
         "weight_given_event": {"low": .2, "center": .4, "high": .6},
         "joint_paths": gap_paths(symbols, {"NVDA": -40., "QQQ": -22., "TLT": 8., "USD": 0.})},
    ]
    r = evaluate(CASH_POSITIONS, nop, branches)
    for s in r["conditional_branch_summaries"]:
        for regime in s["regimes"].values():
            assert regime["holding_yes_pct"]["USD"] == 0 and regime["holding_no_pct"]["USD"] == 0
    bad = deepcopy(nop)
    bad["central"][3]["no_event_return_pct"] = 1.
    with pytest.raises(CommonEndpointError):
        evaluate(CASH_POSITIONS, bad, branches)
    bad_gap = deepcopy(branches)
    bad_gap[0]["joint_paths"]["central"][3]["gap_pct"] = -1.
    with pytest.raises(CommonEndpointError):
        evaluate(CASH_POSITIONS, nop, bad_gap)


def test_chronology_calendar_and_simplex_defects_fail_closed():
    with pytest.raises(CommonEndpointError):
        evaluate(as_of_utc="2027-02-01T00:00:00+00:00")  # capture after deadline
    with pytest.raises(CommonEndpointError):
        evaluate(last_completed_session="2026-11-03")  # mark after capture
    with pytest.raises(CommonEndpointError):
        evaluate(endpoint_rule="UNKNOWN_RULE")
    with pytest.raises(CommonEndpointError):
        evaluate(as_of_utc="2026-11-02T03:30:00")  # naive timestamp
    nop, branches = fixture()
    infeasible = deepcopy(branches)
    infeasible[0]["weight_given_event"] = {"low": .6, "center": .7, "high": .9}
    with pytest.raises(CommonEndpointError):
        evaluate(nop=nop, branches=infeasible)
    no_residual = deepcopy(branches)
    no_residual[-1]["residual"] = False
    with pytest.raises(CommonEndpointError):
        evaluate(nop=nop, branches=no_residual)
    missing_holding = deepcopy(nop)
    missing_holding["central"] = missing_holding["central"][:-1]
    with pytest.raises(CommonEndpointError):
        evaluate(nop=missing_holding, branches=branches)
    too_low = deepcopy(branches)
    too_low[2]["joint_paths"]["adverse"][1]["gap_pct"] = -100.
    negative_no = deepcopy(nop)
    negative_no["adverse"][1]["no_event_return_pct"] = -.5
    with pytest.raises(CommonEndpointError):  # TSM YES = -0.5 - 100 < -100%
        evaluate(nop=negative_no, branches=too_low)


def test_saved_recheck_rejects_tampered_dollars_or_endpoint():
    nop, branches = fixture()
    r = evaluate()
    kw = dict(as_of_utc=AS_OF, deadline=DEADLINE, endpoint_rule="WINDOW_DEADLINE_PLUS_20_SESSIONS",
              last_completed_session=MARK)
    assert verified_saved_common_endpoint(r, POSITIONS, 100_000., nop, branches, **kw)
    tampered = deepcopy(r)
    tampered["event_yes_from_today_center_usd"] += 1.
    assert not verified_saved_common_endpoint(tampered, POSITIONS, 100_000., nop, branches, **kw)
    moved = deepcopy(r)
    moved["common_endpoint_session_date"] = "2027-02-26"
    assert not verified_saved_common_endpoint(moved, POSITIONS, 100_000., nop, branches, **kw)
    assert not verified_saved_common_endpoint(r, POSITIONS, 100_000., nop, branches,
                                              **{**kw, "last_completed_session": "2026-10-29"})


def _prices(symbols, n=140, start=date(2026, 4, 1)):
    from datetime import timedelta
    series = {}
    day, out = start, {s: [] for s in symbols}
    i = 0
    while i < n:
        if is_trading_session(day):
            for k, s in enumerate(symbols):
                out[s].append({"date": day.isoformat(), "close": 100. * (1 + .001 * (k + 1)) ** i + (i % 7) * .1})
            i += 1
        day += timedelta(days=1)
    return out


def test_historical_horizon_reference_is_buy_and_hold_and_gated_by_signature():
    symbols = [p["symbol"] for p in POSITIONS]
    prices = _prices(symbols)
    ref = historical_horizon_reference(POSITIONS, 100_000., prices, horizon_sessions=81,
                                       as_of_utc=AS_OF, endpoint_session_date="2027-03-01")
    assert ref["schema"] == REFERENCE_SCHEMA and ref["status"] == "HISTORICAL_REFERENCE_NOT_A_FORECAST"
    assert ref["windows"] == 140 - 81
    assert ref["calendar_verified_exact_session_horizon"] is True
    assert ref["min_pct"] <= ref["p05_pct"] <= ref["median_pct"] <= ref["p95_pct"] <= ref["max_pct"]
    # Independent recomputation of the first window with start weights.
    first = fsum(p["weight"] * (prices[p["symbol"]][81]["close"] / prices[p["symbol"]][0]["close"] - 1) for p in POSITIONS)
    assert isclose(min(ref["min_pct"], 100 * first), ref["min_pct"], abs_tol=1e-9)
    assert ref["is_forward_forecast"] is False and ref["event_attribution"] is False
    assert ref["portfolio_signature"] == endpoint_signature(POSITIONS, 100_000., "2027-03-01")
    # The reference is admitted into the scenario only when it matches the mark and horizon.
    ref["last_completed_session"] = MARK
    r = evaluate(historical_reference=ref)
    assert r["historical_no_event_horizon_reference"]["windows"] == ref["windows"]
    wrong = {**ref, "horizon_sessions": 80}
    with pytest.raises(CommonEndpointError):
        evaluate(historical_reference=wrong)
    short = historical_horizon_reference(POSITIONS, 100_000., _prices(symbols, n=60), horizon_sessions=81,
                                         as_of_utc=AS_OF, endpoint_session_date="2027-03-01")
    assert short["status"] == "INSUFFICIENT_HISTORY"


def test_historical_reference_never_compresses_missing_market_sessions():
    """One missing daily bar cannot masquerade as an exact 20-session window."""
    symbols = [p["symbol"] for p in POSITIONS]
    prices = _prices(symbols, n=140)
    missing_session = prices[symbols[0]][30]["date"]
    prices[symbols[0]] = [r for r in prices[symbols[0]] if r["date"] != missing_session]
    ref = historical_horizon_reference(POSITIONS, 100_000., prices, horizon_sessions=20,
                                       as_of_utc=AS_OF, endpoint_session_date="2027-03-01")
    assert ref["status"] == "HISTORICAL_REFERENCE_NOT_A_FORECAST"
    assert ref["calendar_verified_exact_session_horizon"] is True
    dates = sorted(set.intersection(*(set(r["date"] for r in prices[s]) for s in symbols)))
    exact = sum(sessions_strictly_between(date.fromisoformat(dates[i]),
                                          date.fromisoformat(dates[i + 20])) == 20
                for i in range(len(dates) - 20))
    assert ref["windows"] == exact
    assert ref["windows"] < ref["candidate_synchronized_windows"]
