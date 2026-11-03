"""R115: exact rectangular credal-partition bounds and fail-closed admission.

All examples are synthetic mathematics tests, not real event forecasts.
"""
from __future__ import annotations

from copy import deepcopy
import random

import pytest
from scipy.optimize import linprog

from castle_risk.credal_scenario import (
    CredalInputError, evaluate_structural_partition,
)


def _row(i, q, a, b, r, low, high, n):
    return {
        "id": f"s{i}", "residual": i == n - 1,
        "scenario_probability": {"low": a, "center": q, "high": b},
        "event_conditional_probability": {"low": low, "center": r, "high": high},
    }


def _example():
    return [
        _row(0, .25, .1, .4, .2, .1, .3, 3),
        _row(1, .45, .2, .6, .8, .7, .9, 3),
        _row(2, .30, .0, .5, .3, .2, .4, 3),
    ]


def test_exact_structural_credal_bounds_not_midpoint_or_confidence_interval():
    result = evaluate_structural_partition(_example())
    assert result["probability_center"] == pytest.approx(.5)
    assert result["credal_interval"] == pytest.approx([.26, .69])
    assert result["probability_center"] != pytest.approx(sum(result["credal_interval"]) / 2)
    assert result["forecast_calibrated"] is False
    assert result["physical_probability_empirically_identified"] is False
    assert "NOT_FREQUENTIST" in result["uncertainty_semantics"]


@pytest.mark.parametrize("mutate", [
    lambda b: b[0].update(residual=True),
    lambda b: b[2].update(residual=False),
    lambda b: b[0]["scenario_probability"].update(center=.12),
    lambda b: b[0]["scenario_probability"].update(high=.15),
    lambda b: b[1]["event_conditional_probability"].update(low=.9),
    lambda b: b[1]["event_conditional_probability"].update(center=True),
    lambda b: b[2]["event_conditional_probability"].update(high=float("nan")),
    lambda b: b[1].update(id="s0"),
    lambda b: b[1].update(scenario_probability={"low": .8, "center": .8, "high": .9}),
])
def test_incoherent_intervals_rejected_without_silent_repairs(mutate):
    branches = deepcopy(_example())
    mutate(branches)
    with pytest.raises(CredalInputError):
        evaluate_structural_partition(branches)


def test_vacuous_conditional_information_remains_vacuous():
    branches = [
        _row(0, .4, 0, 1, .5, 0, 1, 2),
        _row(1, .6, 0, 1, .5, 0, 1, 2),
    ]
    assert evaluate_structural_partition(branches)["credal_interval"] == [0, 1]


def test_greedy_extrema_match_independent_linear_program_solver():
    rng = random.Random(115)
    for n in range(2, 8):
        for _ in range(75):
            raw = [rng.random() for i in range(n)]
            q = [v / sum(raw) for v in raw]
            a = [rng.uniform(0, v) for v in q]
            b = [rng.uniform(v, 1) for v in q]
            lo = [rng.random() * .5 for _ in q]
            hi = [rng.uniform(x, 1) for x in lo]
            r = [rng.uniform(x, y) for x, y in zip(lo, hi)]
            scenarios = [_row(i, q[i], a[i], b[i], r[i], lo[i], hi[i], n)
                         for i in range(n)]
            evaluated = evaluate_structural_partition(scenarios)
            q_bounds = list(zip(a, b))
            lower = linprog(lo, A_eq=[[1.] * n], b_eq=[1.],
                            bounds=q_bounds, method="highs")
            upper = linprog([-v for v in hi], A_eq=[[1.] * n], b_eq=[1.],
                            bounds=q_bounds, method="highs")
            assert lower.success and upper.success
            assert evaluated["credal_interval"][0] == pytest.approx(lower.fun, abs=1e-9)
            assert evaluated["credal_interval"][1] == pytest.approx(-upper.fun, abs=1e-9)
            assert (evaluated["credal_interval"][0] - 1e-9 <=
                    evaluated["probability_center"] <=
                    evaluated["credal_interval"][1] + 1e-9)
