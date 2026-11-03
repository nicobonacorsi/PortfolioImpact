"""The point-3 today-mark calculation must abstain without an observed baseline."""
from copy import deepcopy
import pytest

from castle_risk.conditional_today_mark import (
    ConditionalMarkError, conditional_today_sensitivity,
    verified_saved_today_sensitivity,
)


def test_no_arbitrary_q_or_background_in_the_assessment():
    x = conditional_today_sensitivity(-1837.5, [-6750., 677.5])
    assert x["schema"] == "castle.conditional_today_mark_identification.v2"
    assert x["status"] == "EVENT_YES_PNL_FROM_TODAY_NOT_IDENTIFIED"
    assert x["event_minus_no_event_gap_center_usd"] == -1837.5
    assert x["event_minus_no_event_gap_model_set_usd"] == [-6750., 677.5]
    assert x["event_yes_pnl_from_today_usd"] is None
    assert x["event_yes_pnl_from_today_model_set_usd"] is None
    assert x["no_event_future_mark_identified"] is False
    assert x["equity_priced_in_fraction_identified"] is False
    assert x["zero_background_assumed"] is False
    assert x["arbitrary_common_priced_in_fraction_assumed"] is False
    assert x["prediction_market_quote_used_as_equity_q"] is False
    assert x["physical_event_probability_used"] is False
    assert x["confidence_interval_claimed"] is False
    assert "conditional_today_mark_scenarios" not in x
    assert "flat_background_unknown_q_envelope_usd" not in x
    assert verified_saved_today_sensitivity(x, -1837.5, [-6750., 677.5])


@pytest.mark.parametrize("c,bounds", [
    (-100., [-150., -50.]),
    (100., [50., 150.]),
    (0., [-10., 10.]),
    (0., [0., 0.]),
])
def test_no_absolute_yes_pnl_is_inferred_even_with_a_bounded_gap(c, bounds):
    row = conditional_today_sensitivity(c, bounds)
    assert row["event_yes_pnl_from_today_usd"] is None
    assert row["event_yes_pnl_from_today_model_set_usd"] is None
    assert row["event_minus_no_event_gap_model_set_usd"] == bounds


@pytest.mark.parametrize("center,bounds", [
    (True, [-1.,1.]), (0., [False,1.]),
    (float("nan"),[-1.,1.]), (0., [float("inf"),1.]),
    (0., [1.,2.]), (10., [-2.,4.]),
    (0., [0.,float("nan")]), (0., [0.]),
    (0., "not numbers"), (0., [0.,1.,2.]),
])
def test_invalid_pricing_bridge_fails_closed(center,bounds):
    with pytest.raises(ConditionalMarkError):
        conditional_today_sensitivity(center,bounds)


@pytest.mark.parametrize("change", [
    lambda r: r.update(event_yes_pnl_from_today_usd=-500.),
    lambda r: r.update(equity_priced_in_fraction_identified=True),
    lambda r: r.update(zero_background_assumed=True),
    lambda r: r.update(prediction_market_quote_used_as_equity_q=True),
    lambda r: r.update(arbitrary_common_priced_in_fraction_assumed=True),
    lambda r: r["event_minus_no_event_gap_model_set_usd"].append(0.),
    lambda r: r.update(status="IDENTIFIED"),
])
def test_forged_abs_pnl_and_q_claims_are_rejected(change):
    original = conditional_today_sensitivity(-1700., [-4645., 990.])
    changed = deepcopy(original)
    change(changed)
    assert not verified_saved_today_sensitivity(changed, -1700., [-4645., 990.])


def test_two_different_unobserved_no_event_baselines_yield_different_yes_pnl():
    gap = -1837.5
    yes_if_no_event_would_be_zero = gap + 0
    yes_if_no_event_would_be_plus_5000 = gap + 5000
    assert yes_if_no_event_would_be_zero == -1837.5
    assert yes_if_no_event_would_be_plus_5000 == 3162.5
    assert conditional_today_sensitivity(gap, [-6750., 677.5])[
        "event_yes_pnl_from_today_usd"] is None
