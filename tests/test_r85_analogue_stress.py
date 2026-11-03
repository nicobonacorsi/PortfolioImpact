"""Past-episode replay: arithmetic, missing data, baselines and adapter wiring."""
from datetime import date, timedelta
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from castle_risk.analogue_stress import (EPISODE_SETS, analogue_stress, analogue_stress_for_events,
                                         episode_set_for)


def _sessions(start='2018-01-02', end='2026-10-29'):
    day, last, out = date.fromisoformat(start), date.fromisoformat(end), []
    while day <= last:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _series(drift, shocks=None):
    shocks = shocks or {}
    closes, level = [], 100.0
    for i, day in enumerate(_sessions()):
        level *= 1 + drift * (1 if i % 2 else -1) + shocks.get(day, 0.0)
        closes.append({'date': day, 'close': level})
    return {'prices': closes}


def test_episode_window_return_is_weighted_close_to_close():
    onset = '2022-02-24'
    prices = {'AAA': _series(.001, {onset: -.10}), 'BBB': _series(.001)}
    positions = [{'symbol': 'AAA', 'weight': .5, 'asset_type': 'stock'},
                 {'symbol': 'BBB', 'weight': .3, 'asset_type': 'etf'},
                 {'symbol': 'USD', 'weight': .2, 'asset_type': 'cash'}]
    result = analogue_stress(positions, prices, notional=100_000, set_name='OIL_SUPPLY_SHOCK')
    row = next(e for e in result['episodes'] if e['onset_session'] == onset)
    closes = {s: {p['date']: p['close'] for p in prices[s]['prices']} for s in prices}
    expected = sum(w * (closes[s][row['end_session']] / closes[s][row['base_session']] - 1)
                   for s, w in (('AAA', .5), ('BBB', .3)))
    assert row['portfolio_return'] == pytest.approx(expected)
    assert row['pnl_usd'] == pytest.approx(100_000 * expected)
    assert row['base_session'] < onset <= row['end_session']
    assert result['status'] == 'MEASURED_RANGE'
    assert result['pnl_range_usd'][0] <= row['pnl_usd'] <= result['pnl_range_usd'][1]
    assert result['leave_one_out_total'] == result['measured_count']
    assert result['leave_one_out_coverage_is_order_statistic_identity'] is True
    assert result['ordinary_baseline_excludes_full_episode_windows'] is True


def test_historical_range_is_outlier_sensitive_not_validated_by_loo_containment():
    """For distinct observations LOO min/max containment always equals n-2.

    Unlike that tautology, endpoint influence measures whether one extreme
    episode accounts for most of the apparently reassuring historical range.
    """
    onset = '2025-04-01'
    positions = [{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}]
    prices = {'AAA': _series(0., {
        '2022-08-02': .001, '2023-04-10': .002,
        '2024-05-23': -.001, '2024-10-14': .003, onset: -.30})}
    result = analogue_stress(positions, prices, notional=100_000, set_name='TAIWAN_STRAIT')
    assert result['status'] == 'MEASURED_RANGE'
    assert result['leave_one_out_covered'] == result['measured_count'] - 2
    assert result['leave_one_out_coverage_is_order_statistic_identity'] is True
    assert result['most_influential_episode'] == 'PLA "Strait Thunder-2025A" drills'
    assert result['max_envelope_endpoint_influence_usd'] > 29_000
    assert result['ordinary_window_5_95'] == pytest.approx([0., 0.], abs=1e-12)
    assert result['episodes_outside_ordinary_range'] == 5
    shock = next(e for e in result['episodes'] if e['onset_session'] == onset)
    assert shock['ordinary_window_two_sided_tail_rank'] < .01


def test_ordinary_control_excludes_rolling_windows_overlapping_event():
    """The event impulse must not contaminate the ordinary control distribution."""
    onset = '2025-04-01'
    prices = {'AAA': _series(0., {onset: -.15})}
    result = analogue_stress([{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}], prices,
                             notional=1, set_name='TAIWAN_STRAIT')
    assert result['ordinary_window_5_95'] == pytest.approx([0., 0.], abs=1e-12)
    assert result['ordinary_baseline_excludes_full_episode_windows']


def test_downloaded_recorded_case_recomputes_informative_taiwan_sensitivity():
    """R85's frozen public data should get the corrected R86 export framing."""
    from castle_risk.portfolio_impact_export import make_report_zip
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / 'examples/recorded_analysis/case.json').read_text())
    taiwan = next(e['analogue_stress'] for e in case['risks']
                  if 'blockade or quarantine of Taiwan' in e.get('title', ''))
    clean = [x['pnl_usd'] for x in taiwan['episodes']
             if x['status'] == 'MEASURED' and not x['overlapping_episodes']]
    assert len(clean) == 4
    assert min(clean) == pytest.approx(-1694, abs=1.)
    assert max(clean) == pytest.approx(1157, abs=1.)
    with ZipFile(BytesIO(make_report_zip(case))) as archive:
        html = archive.read('report.html').decode('utf-8')
    assert 'Excluding catalogued overlaps' in html
    assert '$1,694' in html and '$1,157' in html
    assert '$7,135' in html
    assert 'Leave-one-episode-out coverage' not in html


def test_missing_history_is_reported_not_filled():
    late = {'prices': [p for p in _series(.001)['prices'] if p['date'] >= '2024-01-01']}
    prices = {'AAA': _series(.001), 'NEW': late}
    positions = [{'symbol': 'AAA', 'weight': .5, 'asset_type': 'stock'},
                 {'symbol': 'NEW', 'weight': .5, 'asset_type': 'stock'}]
    result = analogue_stress(positions, prices, notional=1, set_name='TAIWAN_STRAIT')
    old = [e for e in result['episodes'] if e['onset_session'] < '2024-01-01']
    assert old and all(e['status'] == 'OUTSIDE_COMMON_PRICE_HISTORY' and 'NEW' in e['reason'] for e in old)
    assert result['measured_count'] == 3


def test_too_few_episodes_gives_no_range():
    prices = {'AAA': _series(.001)}
    result = analogue_stress([{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}], prices,
                             notional=1, set_name='US_SOVEREIGN_DOWNGRADE')
    assert result['status'] == 'TOO_FEW_EPISODES'
    assert 'pnl_range_usd' not in result


def test_overlapping_episode_windows_are_flagged():
    prices = {'AAA': _series(.001)}
    result = analogue_stress([{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}], prices,
                             notional=1, set_name='TAIWAN_STRAIT')
    april = next(e for e in result['episodes'] if e['onset_session'] == '2025-04-01')
    assert any('2025-04-03' in text for text in april['overlapping_episodes'])
    assert result['catalogued_overlap_excluded_count'] >= 1
    assert result['unconfounded_by_catalogued_events_count'] + result['catalogued_overlap_excluded_count'] == result['measured_count']
    isolated = [e['portfolio_return'] for e in result['episodes']
                if e['status'] == 'MEASURED' and not e['overlapping_episodes']]
    if len(isolated) >= 3:
        assert result['no_catalogued_overlap_return_range'] == pytest.approx([min(isolated), max(isolated)])


def test_family_and_wording_mapping():
    assert episode_set_for('HORMUZ_ENERGY') == 'HORMUZ_TRANSIT_SHOCK'
    assert episode_set_for('TAIWAN_SUPPLY') == 'TAIWAN_STRAIT'
    assert episode_set_for('OTHER', 'A lapse in federal appropriations causing a government shutdown') == 'US_GOVERNMENT_SHUTDOWN'
    assert episode_set_for('RUSSIA_ENERGY_SANCTIONS', 'Russia and Ukraine agree a ceasefire') is None
    assert episode_set_for('FED_POLICY') is None
    events = [{'event_id': 'a', 'family': 'TAIWAN_SUPPLY'}, {'event_id': 'b', 'family': 'FED_POLICY'}]
    out = analogue_stress_for_events(events, [{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}],
                                     {'AAA': _series(.001)}, notional=1)
    assert set(out) == {'a'}


def test_every_episode_onset_is_a_weekday_iso_date():
    for spec in EPISODE_SETS.values():
        for _, onset, _ in spec['episodes']:
            assert date.fromisoformat(onset).weekday() < 5


def test_adapter_attaches_only_schema_tagged_results():
    from castle_risk import portfolio_impact_adapter as adapter
    import inspect
    assert 'castle.option1.historical_analogue_stress.v1' in inspect.getsource(adapter.live_report_to_case)


@pytest.mark.parametrize('family,text,expected', [
    ('EXPORT_CONTROLS', 'China bans rare-earth and critical mineral exports to the US', 'CHINA_CRITICAL_MINERALS'),
    ('EXPORT_CONTROLS', 'A formal BIS rule newly restricts exports of advanced AI chips to China', 'US_CHIP_EXPORT_CONTROLS'),
    ('TAIWAN_SUPPLY', 'USGS lists a magnitude 7.0 earthquake near Taiwan', None),
    ('TRADE_TARIFFS', 'Section 232 proclamation imposes import tariffs on semiconductors', 'US_CHINA_TARIFFS'),
    ('OTHER', 'Iran closes the Strait of Hormuz', 'HORMUZ_TRANSIT_SHOCK'),
])
def test_wording_takes_precedence_over_coarse_family(family, text, expected):
    assert episode_set_for(family, text) == expected
