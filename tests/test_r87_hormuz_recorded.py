"""Exact recorded R87 addition survives actual exported report; source fields not faked."""
from io import BytesIO
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from castle_risk.portfolio_impact_export import make_report_zip


def test_recorded_hormuz_severe_stress_export_and_original_distribution_untouched():
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / 'examples/recorded_analysis/case.json').read_text())
    event = next(r for r in case['risks'] if r['id'] == 'hormuz_closure_declared_by_20270128')
    opec = next(r for r in case['risks'] if r['id'] == 'opec_plus_production_cut_by_20270128')
    a = event['analogue_stress']
    direct = a['direct_disruption_2026']
    assert a['measured_count'] == 8
    assert direct['schema'] == 'castle.r87.realized_hormuz_reference.v1'
    assert not direct['equivalent_to_2027_original_settlement_predicate']
    assert direct['probability_of_new_2027_predicate'] is None
    assert opec['analogue_stress'].get('direct_disruption_2026') is None
    assert abs(sum(c['pnl_usd'] for c in direct['contributions']) - direct['pnl_usd']) < 1e-7
    assert abs(direct['pnl_usd'] + 3124.917844433878) < .01
    with ZipFile(BytesIO(make_report_zip(case))) as archive:
        html = archive.read('report.html').decode()
        assert 'Directly observed 2026 Hormuz/Gulf disruption' in html
        assert '$3,125' in html
        csv_file = next(n for n in archive.namelist() if n.endswith('past_episode_replay.csv'))
        csv_rows = list(csv.DictReader(archive.read(csv_file).decode('utf-8-sig').splitlines()))
    matched = [r for r in csv_rows if r['episode_set'] == 'DIRECT_2026_HORMUZ_OBSERVATION']
    assert len(matched) == 1
    assert matched[0]['event_id'] == event['id']
    assert matched[0]['base_session'] == '2026-02-27'
    assert matched[0]['end_session'] == '2026-03-06'
