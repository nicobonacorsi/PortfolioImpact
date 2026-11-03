"""R88: exact algebra, scope gates and real recorded contract-scenario checks."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from castle_risk.priced_in_sensitivity import (
    admitted_binary_contract_sensitivity, binary_weight_bounds,
)
from castle_risk.portfolio_impact_export import make_report_zip

ROOT = Path(__file__).resolve().parents[1]


def _recorded():
    return json.loads((ROOT / 'examples/recorded_analysis/case.json').read_text())


def test_exact_vertex_bounds_negative_contrast():
    result = binary_weight_bounds(-12_000)
    assert result['equity_pricing_weight_identified'] is False
    assert result['yes_pnl_usd_range'] == [-12_000, 0]
    assert result['no_pnl_usd_range'] == [0, 12_000]
    assert result['point_grid'][2] == {
        'assumed_pi': .5, 'yes_pnl_usd': -6_000, 'no_pnl_usd': 6_000,
    }
    assert result['point_grid'][0]['yes_pnl_usd'] == -12_000
    assert result['point_grid'][-1]['yes_pnl_usd'] == 0


def test_two_observationally_identical_prices_can_have_different_conditional_pnl():
    # A proof by counterexample: Q and the current portfolio price alone do
    # not identify an event-caused state contrast even if pi were known.
    pi, quote, price = .1, .1, 100.
    economy_a = (100., 100.)  # YES, NO values
    economy_b = (55., 105.)
    assert quote == pi
    for yes, no in (economy_a, economy_b):
        assert pi * yes + (1-pi)*no == pytest.approx(price)
    assert economy_a[0] - price == 0
    assert economy_b[0] - price == -45


def test_vertex_bounds_with_intervals_cover_entire_declared_rectangle():
    for contrast, band, weights in (
        (-12, [-20, -2], [.2, .6]),
        (4, [-8, 12], [.1, .9]),
        (6, [2, 8], [0, 1]),
    ):
        result = binary_weight_bounds(contrast, contrast_interval_usd=band,
                                      equity_pricing_weight_interval=weights)
        for k in range(41):
            pi = weights[0] + k/40*(weights[1]-weights[0])
            for j in range(41):
                d = band[0] + j/40*(band[1]-band[0])
                yes, no = (1-pi)*d, -pi*d
                assert result['yes_pnl_usd_range'][0]-1e-10 <= yes <= result['yes_pnl_usd_range'][1]+1e-10
                assert result['no_pnl_usd_range'][0]-1e-10 <= no <= result['no_pnl_usd_range'][1]+1e-10
        for pi in weights:
            for d in band:
                assert result['yes_pnl_usd_range'][0] <= (1-pi)*d <= result['yes_pnl_usd_range'][1]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, None])
def test_reject_nonfinite_contrast(bad):
    with pytest.raises(ValueError):
        binary_weight_bounds(bad)


def test_reject_reversed_or_invalid_weights_and_contrasts():
    for kwargs in (
        {'equity_pricing_weight_interval': [.8, .2]},
        {'equity_pricing_weight_interval': [-.1, .8]},
        {'equity_pricing_weight_interval': [0, 1.2]},
        {'contrast_interval_usd': [2, -2]},
        {'contrast_interval_usd': [2, 4]},
    ):
        with pytest.raises(ValueError):
            binary_weight_bounds(-3, **kwargs)


def test_recorded_real_source_card_quote_not_equity_weight():
    case = _recorded()
    cards = [r for r in case['risks'] if r.get('source_defined_scenario')]
    studies = [admitted_binary_contract_sensitivity(r) for r in cards]
    valid = [r for r in studies if r]
    assert len(valid) >= 4
    taiwan = next(r for r in studies if r and r['contract_id'] == '2382819')
    assert taiwan['quote_is_equity_weight'] is False
    assert taiwan['equity_pricing_weight_identified'] is False
    assert taiwan['contract_quote_q_reference'] == pytest.approx(.0255)
    d = taiwan['contrast_usd']
    assert d < -10_000
    assert taiwan['yes_pnl_usd_range'] == pytest.approx([d, 0])
    assert taiwan['no_pnl_usd_range'] == pytest.approx([0, -d])
    assert taiwan['stored_yes_pnl_under_assumed_pi_equals_q'] == pytest.approx((1-.0255)*d)
    assert taiwan['point_grid'][2]['yes_pnl_usd'] == pytest.approx(.5*d)


def test_fail_closed_on_mismatched_quotes_and_broader_parent():
    case = _recorded()
    original = next(r for r in case['risks'] if r['id'] == 'taiwan_blockade_declared_by_20270128')
    assert admitted_binary_contract_sensitivity(original) is None
    card = next(r for r in case['risks'] if r.get('source_defined_scenario')
                and r['title'].startswith('Polymarket: Will China blockade'))
    copy = json.loads(json.dumps(card))
    copy['outcomes'][0]['q_reference'] = .50
    assert admitted_binary_contract_sensitivity(copy) is None
    copy = json.loads(json.dumps(card))
    copy['outcomes'][0]['pnl'] *= 2
    assert admitted_binary_contract_sensitivity(copy) is None
    copy = json.loads(json.dumps(card))
    copy['source_defined_scenario'] = False
    assert admitted_binary_contract_sensitivity(copy) is None


def test_actual_html_export_contains_sensitivity_not_falsely_on_parent():
    case = _recorded()
    with ZipFile(BytesIO(make_report_zip(case))) as archive:
        html = archive.read('report.html').decode()
        text = archive.read('README.txt').decode()
    assert 'Assumed effective equity pricing weight' in html
    assert 'does not identify π for the portfolio' in html
    assert 'sharp' in html
    assert 'historical' in text.lower()
    assert html.count('class="pricing-weight-sensitivity"') >= 4
