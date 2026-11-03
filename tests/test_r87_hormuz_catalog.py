"""R87: the 2026 genuine disruption is distinct from an ordinary OPEC cut."""
from datetime import date, timedelta
import math
import unittest

from castle_risk.analogue_stress import EPISODE_SETS, analogue_stress, episode_set_for


def _prices():
    day = date(2018, 1, 2)
    end = date(2026, 10, 30)
    result = []
    level = 100.0
    while day <= end:
        if day.weekday() < 5:
            key = day.isoformat()
            # Isolate one large real-event-shaped shock for deterministic math.
            level *= 1.25 if key == '2026-03-02' else 1.0
            result.append({'date': key, 'close': level})
        day += timedelta(days=1)
    return {'AAA': {'prices': result}}


class TestR87Catalog(unittest.TestCase):
    def test_shock_is_a_ninth_hormuz_only_episode_with_source(self):
        self.assertEqual(len(EPISODE_SETS['OIL_SUPPLY_SHOCK']['episodes']), 8)
        self.assertEqual(len(EPISODE_SETS['HORMUZ_TRANSIT_SHOCK']['episodes']), 9)
        self.assertEqual(episode_set_for('HORMUZ_ENERGY'), 'HORMUZ_TRANSIT_SHOCK')
        self.assertEqual(episode_set_for('OPEC_SUPPLY'), 'OIL_SUPPLY_SHOCK')
        self.assertEqual(episode_set_for(None, 'Iran closes the Strait of Hormuz'),
                         'HORMUZ_TRANSIT_SHOCK')
        self.assertEqual(episode_set_for(None, 'OPEC+ cuts production'),
                         'OIL_SUPPLY_SHOCK')
        portfolio = [{'symbol': 'AAA', 'weight': 1., 'asset_type': 'stock'}]
        actual = analogue_stress(portfolio, _prices(), notional=100000,
                                 set_name='HORMUZ_TRANSIT_SHOCK')
        ordinary = analogue_stress(portfolio, _prices(), notional=100000,
                                   set_name='OIL_SUPPLY_SHOCK')
        self.assertEqual(actual['status'], 'MEASURED_RANGE')
        self.assertEqual(actual['measured_count'], 9)
        self.assertEqual(actual['direct_disruption_measured_count'], 1)
        self.assertEqual(ordinary['measured_count'], 8)
        self.assertNotIn('direct_disruption_measured_count', ordinary)
        dated = next(x for x in actual['episodes'] if x['onset_session'] == '2026-03-02')
        self.assertEqual(dated['base_session'], '2026-02-27')
        self.assertEqual(dated['end_session'], '2026-03-06')
        self.assertTrue(dated['direct_disruption_observation_not_causal_estimate'])
        self.assertTrue(dated['source_reference'].startswith('https://www.federalreserve.gov/'))
        self.assertAlmostEqual(dated['pnl_usd'], 25000., places=6)
        self.assertEqual(dated['overlapping_episodes'], [])
        self.assertEqual(actual['most_influential_episode'], dated['episode'])

    def test_duplicate_comparator_names_are_not_false_confounds(self):
        historical = analogue_stress([{'symbol': 'AAA', 'weight': 1.,
                                      'asset_type': 'stock'}], _prices(),
                                     notional=1, set_name='OIL_SUPPLY_SHOCK')
        self.assertTrue(all(not any('Drone attack on Saudi' in overlap
                                    for overlap in row.get('overlapping_episodes', []))
                            for row in historical['episodes']))


if __name__ == '__main__':
    unittest.main()
