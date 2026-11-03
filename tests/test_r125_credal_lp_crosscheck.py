"""Independent LP recomputation of the greedy bounded-simplex credal bounds (R125)."""
from __future__ import annotations

import random

import pytest

from castle_risk.credal_scenario import (
    CredalInputError, evaluate_structural_partition, evaluate_structural_partition_lp,
)


def _random_partition(rng, n):
    centers = [rng.random() for _ in range(n)]
    total = sum(centers)
    centers = [c / total for c in centers]
    rows = []
    for i, q in enumerate(centers):
        a = max(0.0, q - rng.random() * q)
        b = min(1.0, q + rng.random() * (1 - q))
        r = rng.random()
        lo = max(0.0, r - rng.random() * r)
        hi = min(1.0, r + rng.random() * (1 - r))
        rows.append({"id": f"s{i}", "residual": i == n - 1,
                     "scenario_probability": {"low": a, "center": q, "high": b},
                     "event_conditional_probability": {"low": lo, "center": r, "high": hi},
                     "trigger": "t", "rationale": "r", "counterevidence": "c", "source_ids": []})
    return rows


@pytest.mark.parametrize("seed", range(40))
def test_greedy_and_lp_bounds_agree_on_random_feasible_partitions(seed):
    rng = random.Random(seed)
    rows = _random_partition(rng, rng.randint(2, 7))
    greedy = evaluate_structural_partition(rows)
    lp = evaluate_structural_partition_lp(rows)
    assert lp["agrees_within_1e_9"], (greedy["credal_interval"], lp["lp_credal_interval"])
    assert greedy["credal_interval"][0] <= greedy["probability_center"] <= greedy["credal_interval"][1]


def test_lp_crosscheck_rejects_what_the_greedy_solver_rejects():
    bad = [{"id": "a", "residual": False, "scenario_probability": {"low": .6, "center": .7, "high": .8},
            "event_conditional_probability": {"low": .1, "center": .2, "high": .3}},
           {"id": "b", "residual": True, "scenario_probability": {"low": .5, "center": .3, "high": .6},
            "event_conditional_probability": {"low": .1, "center": .2, "high": .3}}]
    with pytest.raises(CredalInputError):
        evaluate_structural_partition_lp(bad)
