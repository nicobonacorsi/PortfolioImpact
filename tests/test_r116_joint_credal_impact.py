"""R116 deterministic numerical proofs and adversarial original-NAV invariants.

No model API, market provider, credentials or synthetic-to-live promotion.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest

from castle_risk.joint_credal_impact import (
    JointImpactInputError, evaluate_joint_impact, portfolio_signature,
)


POSITIONS = [
    {"symbol": "XOM", "type": "stock", "weight": .30},
    {"symbol": "TSM", "type": "stock", "weight": .25},
    {"symbol": "LMT", "type": "stock", "weight": .25},
    {"symbol": "IEF", "type": "etf", "weight": .20},
]
SCENARIOS = [
    ("S1", (.20, .35, .50), {"XOM": 3., "TSM": -8., "LMT": 4., "IEF": 1.}),
    ("S2", (.30, .45, .60), {"XOM": 8., "TSM": -25., "LMT": 9., "IEF": 2.5}),
    ("S3", (.10, .20, .35), {"XOM": 5., "TSM": -40., "LMT": 6., "IEF": 4.}),
]


def _vector(returns, shift=0.):
    return [{"symbol": s, "gap_pct": val + shift} for s, val in returns.items()]


def _branches():
    return [
        {"id": label, "trigger": f"Operational branch for {label}",
         "residual": i == 2,
         "weight_given_event": dict(zip(("low", "center", "high"), mass)),
         "joint_paths": {"adverse": _vector(moves, -2.),
                         "central": _vector(moves),
                         "favorable": _vector(moves, 2.)}}
        for i, (label, mass, moves) in enumerate(SCENARIOS)
    ]


def test_full_portfolio_center_and_coupled_interval_are_exact():
    result = evaluate_joint_impact(POSITIONS, 100_000, _branches())
    assert result["gap_center_usd"] == pytest.approx(-1700.)
    assert result["gap_model_set_usd"] == pytest.approx([-4645., 990.])
    assert result["gap_model_set_pct"] == pytest.approx([-4.645, .990])
    assert result["estimated_causal_event_impact"] is False
    assert result["absolute_event_yes_pnl_from_current_nav_identified"] is False
    assert result["market_event_probability_used"] is False
    assert result["equity_pricing_q_used"] is False
    assert result["original_etf_nav_counted_once"] is True
    assert sum(p["weight"] for p in POSITIONS) == pytest.approx(1.)


def test_cash_is_zero_and_etf_is_single_original_nav():
    positions = [{"symbol": "CASH", "type": "cash", "weight": .4},
                 {"symbol": "QQQ", "type": "etf", "weight": .6}]
    branches = [
        {"id": "t1", "trigger": "First market regime", "residual": False,
         "weight_given_event": {"low": .3, "center": .5, "high": .7},
         "joint_paths": {k: [{"symbol": "CASH", "gap_pct": 0.},
                             {"symbol": "QQQ", "gap_pct": v}]
                         for k, v in (("adverse", -10), ("central", -5), ("favorable", 2))}},
        {"id": "rest", "trigger": "All remaining conditions", "residual": True,
         "weight_given_event": {"low": .3, "center": .5, "high": .7},
         "joint_paths": {k: [{"symbol": "CASH", "gap_pct": 0.},
                             {"symbol": "QQQ", "gap_pct": v}]
                         for k, v in (("adverse", -3), ("central", 0), ("favorable", 3))}},
    ]
    result = evaluate_joint_impact(positions, 100_000, branches)
    assert result["gap_center_usd"] == pytest.approx(-1500.)
    assert result["gap_model_set_usd"] == pytest.approx([-4740., 1620.])
    assert all(len(r["joint_paths"]["central"]["holding_gap_pct"]) == 2
               for r in result["branch_summaries"])
    assert result["portfolio_signature"] == portfolio_signature(
        list(reversed(positions)), 100000)


@pytest.mark.parametrize("corruption", [
    lambda b, p: b[0]["joint_paths"]["central"].pop(),
    lambda b, p: b[0]["joint_paths"]["central"].append(
        {"symbol": "QQQ", "gap_pct": 10.}),
    lambda b, p: b[1]["joint_paths"]["adverse"][0].update(symbol="TSM"),
    lambda b, p: b[0]["joint_paths"]["adverse"][0].update(gap_pct=float("nan")),
    lambda b, p: b[0]["joint_paths"]["adverse"][0].update(gap_pct=float("inf")),
    lambda b, p: b[0]["joint_paths"]["adverse"][0].update(gap_pct=True),
    lambda b, p: b[0]["joint_paths"]["adverse"][0].update(gap_pct=-101.),
    lambda b, p: b[0]["weight_given_event"].update(center=.45),
    lambda b, p: b[0]["weight_given_event"].update(low=.5),
    lambda b, p: b[0]["weight_given_event"].update(high=.2),
    lambda b, p: b[0]["weight_given_event"].update(high=float("nan")),
    lambda b, p: b[0].update(residual=True),
    lambda b, p: b[2].update(residual=False),
    lambda b, p: b[2].update(id="S2"),
    lambda b, p: b[0].update(trigger="no"),
    lambda b, p: b[0]["joint_paths"].update(adverse=b[0]["joint_paths"]["favorable"]),
    lambda b, p: p[0].update(weight=.4),
    lambda b, p: p[0].update(type="auto"),
    lambda b, p: p[0].update(symbol="IEF"),
])
def test_adversarial_numeric_or_semantic_scope_change_fails_closed(corruption):
    branches, positions = _branches(), deepcopy(POSITIONS)
    corruption(branches, positions)
    with pytest.raises(JointImpactInputError):
        evaluate_joint_impact(positions, 100_000, branches)


def test_missing_cash_zero_is_rejected_without_silent_market_exposure():
    positions = deepcopy(POSITIONS)
    positions[0].update(symbol="USD", type="cash")
    branches = _branches()
    for branch in branches:
        for path in branch["joint_paths"].values():
            path[0]["symbol"] = "USD"
    with pytest.raises(JointImpactInputError, match="Nominal USD cash"):
        evaluate_joint_impact(positions, 100000, branches)


def test_weight_feasibility_is_not_replaced_by_normalization():
    branches = _branches()
    for branch in branches:
        branch["weight_given_event"].update(low=.5, center=1/3, high=.8)
    with pytest.raises(JointImpactInputError):
        evaluate_joint_impact(POSITIONS, 100000, branches)
    branches = _branches()
    for branch in branches:
        branch["weight_given_event"].update(high=.25)
    with pytest.raises(JointImpactInputError):
        evaluate_joint_impact(POSITIONS, 100000, branches)


def test_signed_lp_against_independent_three_branch_vertex_enumeration():
    """A numerical independent check; no SciPy dependency or optimizer reuse."""
    from castle_risk.joint_credal_impact import _bounded_simplex_extreme

    for lo, hi in [
        ([.0, .2, .0], [.4, .9, .6]),
        ([.1, .1, .1], [.8, .8, .8]),
        ([.2, .3, .1], [.5, .6, .35]),
    ]:
        for costs in ([.1, -1.1, -6.2], [5.0, -100.0, 3.2],
                      [-1.0, -1.0, -1.0], [-9., 9., 0.]):
            vertices = []
            for i in range(3):
                j, k = [t for t in range(3) if t != i]
                for x, y in product((lo[j], hi[j]), (lo[k], hi[k])):
                    z = 1-x-y
                    if lo[i]-1e-9 <= z <= hi[i]+1e-9:
                        p = [None]*3
                        p[j], p[k], p[i] = x, y, z
                        vertices.append(sum(a*b for a, b in zip(p, costs)))
            assert vertices
            assert _bounded_simplex_extreme(lo, hi, costs, maximize=False) == pytest.approx(min(vertices))
            assert _bounded_simplex_extreme(lo, hi, costs, maximize=True) == pytest.approx(max(vertices))


def test_horizon_signature_changes_with_time_and_holding_type():
    sig = portfolio_signature(POSITIONS, 100000, 20)
    assert sig != portfolio_signature(POSITIONS, 100000, 5)
    changed = deepcopy(POSITIONS)
    changed[-1]["type"] = "stock"
    assert sig != portfolio_signature(changed, 100000, 20)
