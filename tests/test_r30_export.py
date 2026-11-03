"""R30 interval-first export regressions; pure in-memory, no providers or writes."""
from __future__ import annotations

import csv
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from castle_risk.portfolio_impact_adapter import live_report_to_case
from castle_risk.portfolio_impact_export import make_report_zip


ROOT = Path(__file__).resolve().parents[1]
RECORDED_R21 = ROOT / 'reports' / 'r21_validation' / 'user_live_assessment_20261027.json'


def example():
    return {
        'id': 'r30-offline-synthetic', 'user_supplied': True, 'notional': 100000,
        'capture': '2026-10-27T21:00:00Z', 'verification_status': 'SYNTHETIC_TEST_ONLY',
        'positions': [{'symbol': 'CAT', 'type': 'stock', 'weight': .3},
                      {'symbol': 'UNP', 'type': 'stock', 'weight': .25},
                      {'symbol': 'NEE', 'type': 'stock', 'weight': .25},
                      {'symbol': 'IEF', 'type': 'etf', 'weight': .2}],
        'lookthrough': [], 'fund_details': [],
        'risks': [{
            'id': 'test', 'title': 'Synthetic payroll scenario', 'family': 'US_PAYROLLS',
            'date': '2026-12-04',
            'outcomes': [{
                'id': 'band', 'label': 'Synthetic jobs band', 'q_reference': .25,
                'q_interval': [.2, .3], 'pnl': -8.98790543003924,
                'pnl_interval': [-1530.8315780178095, 1491.4640519034551],
                'pnl_interval_kind': 'SCENARIO_PARAMETER_ENVELOPE',
                'pnl_interval_interpretation': 'Recorded parameter envelope, not a prediction interval.',
                'source_contract_ids': ['TEST-LOWER', 'TEST-UPPER'],
            }],
            'admitted': [{'id': 'TEST-LOWER', 'venue': 'Test', 'question': 'Synthetic threshold'},
                         {'id': 'TEST-UPPER', 'venue': 'Test', 'question': 'Synthetic threshold'},
                         {'id': 'TEST-LOWER', 'venue': 'Test', 'question': 'Duplicate record'}],
            'rejected': [{'id': 'Unmatched question', 'decision': 'NO_EXACT_FIT',
                          'reason': 'No source-compatible quote established in this search.'}],
            'review_notes': [],
            'exposure': [], 'exposure_basis': 'PORTFOLIO_CONTEXT_ONLY',
            'quality': {
                'model_kind': 'MEASURED_PAYROLL', 'transfer_status': 'NOT_REPORTED',
                'failed_portfolio_weight': None,
                'model_comparisons': [{
                    'id': 'economics', 'label': 'Daily-beta comparison', 'transfer_status': 'FAIL',
                    'failed_symbols': ['UNP', 'NEE'], 'failed_portfolio_weight': .5,
                    'outcomes': [],
                }, {
                    'id': 'announcement_day_economics', 'label': 'Announcement-day-beta comparison',
                    'transfer_status': 'NOT_REPORTED', 'failed_symbols': [],
                    'failed_portfolio_weight': None, 'outcomes': [],
                }],
            },
        }],
    }


def decoded(case):
    with ZipFile(BytesIO(make_report_zip(case))) as bundle:
        return {
            'names': set(bundle.namelist()),
            'html': bundle.read('report.html').decode('utf-8'),
            'rows': list(csv.DictReader(StringIO(bundle.read('scenario_results.csv').decode('utf-8-sig')))),
            'case': json.loads(bundle.read('portfolio_data.json')),
            'readme': bundle.read('README.txt').decode('utf-8'),
        }


class R30ExportTests(unittest.TestCase):
    def test_range_first_and_direction_uncertain_in_report(self):
        result = decoded(example())
        html, row = result['html'], result['rows'][0]
        self.assertIn('−$1,531 to +$1,491', html)
        self.assertIn('class="neutral"', html)
        self.assertIn('Direction uncertain', html)
        self.assertLess(html.index('<th>Model range</th>'), html.index('<th>Point estimate</th>'))
        self.assertAlmostEqual(float(row['modeled_change_usd']), -8.98790543003924)
        self.assertAlmostEqual(float(row['model_range_low_usd']), -1530.8315780178095)
        self.assertAlmostEqual(float(row['model_range_high_usd']), 1491.4640519034551)
        self.assertEqual(row['direction'], 'DIRECTION_UNCERTAIN')
        self.assertEqual(row['model_range_kind'], 'SCENARIO_PARAMETER_ENVELOPE')
        self.assertEqual(row['model_range_interpretation'],
                         'Recorded parameter envelope, not a prediction interval.')

    def test_unknown_status_and_weight_are_never_pass_or_zero(self):
        case = example()
        del case['risks'][0]['quality']['transfer_status']
        case['risks'][0]['quality']['failed_portfolio_weight'] = 0
        row = decoded(case)['rows'][0]
        self.assertEqual(row['verification'], 'NOT_REPORTED')
        self.assertEqual(row['unvalidated_portfolio_share'], '')
        self.assertEqual(row['daily_transfer_status'], 'FAIL')
        self.assertEqual(row['daily_failed_symbols'], 'UNP; NEE')
        self.assertEqual(row['daily_failed_portfolio_share'], '0.5')
        self.assertEqual(row['announcement_transfer_status'], 'NOT_REPORTED')
        self.assertEqual(row['announcement_failed_portfolio_share'], '')

    def test_composite_not_tested_does_not_erase_daily_fail(self):
        case = example()
        case['risks'][0]['quality'].update(model_kind='CHANNEL_AWARE', transfer_status='NOT_TESTED')
        result = decoded(case)
        self.assertEqual(result['rows'][0]['verification'], 'NOT_TESTED')
        self.assertEqual(result['rows'][0]['unvalidated_portfolio_share'], '')
        self.assertEqual(result['rows'][0]['daily_transfer_status'], 'FAIL')
        self.assertIn('not independently tested', result['html'])
        self.assertIn('Model reliability warning', result['html'])
        self.assertIn('Daily-beta comparison', result['html'])
        self.assertIn('50% of portfolio flagged', result['html'])

    def test_portfolio_only_failure_does_not_invent_failing_holdings(self):
        case = example()
        case['risks'][0]['quality'].update(
            transfer_status='FAIL', portfolio_transfer_status='FAIL',
            failure_scope='PORTFOLIO_ONLY', failed_symbols=[], failed_portfolio_weight=0)
        result = decoded(case)
        row = result['rows'][0]
        self.assertEqual(row['portfolio_transfer_status'], 'FAIL')
        self.assertEqual(row['failure_scope'], 'PORTFOLIO_ONLY')
        self.assertEqual(row['failed_symbols'], '')
        self.assertEqual(row['unvalidated_portfolio_share'], '')
        self.assertIn('Portfolio-level chronological check failed.', result['html'])
        self.assertNotIn('some holdings', result['html'])
        self.assertNotIn('(0% of portfolio)', result['html'])

    def test_fixed_nominal_cash_identity_is_not_a_missing_estimate(self):
        case = example()
        quality = case['risks'][0]['quality']
        quality.update(model_kind='NOMINAL_CASH_IDENTITY', model_comparisons=[])
        case['risks'][0]['outcomes'][0].update(
            pnl=0, pnl_interval=[0, 0], pnl_interval_kind='EXACT_NOMINAL_CASH_IDENTITY',
            pnl_interval_interpretation='Exact zero nominal price change for fixed USD cash.')
        result = decoded(case)
        self.assertIn('Fixed USD cash identity', result['html'])
        self.assertIn('Fixed nominal USD cash value', result['html'])
        self.assertIn('class="neutral"><strong>$0 to $0', result['html'])
        self.assertNotIn('Direction uncertain', result['html'])
        self.assertNotIn('not independently verified', result['html'])
        self.assertEqual(result['rows'][0]['direction'], 'ZERO_UNDER_STATED_MODEL')
        self.assertEqual(result['rows'][0]['model_range_low_usd'], '0')
        self.assertEqual(result['rows'][0]['model_range_high_usd'], '0')

    def test_search_notes_are_not_contracts_and_actual_ids_remain(self):
        result = decoded(example())
        row = result['rows'][0]
        self.assertEqual(row['contracts_used'], '2')
        self.assertEqual(row['contracts_excluded'], '0')
        self.assertIn('No source-compatible quote', row['review_notes'])
        self.assertIn('Prediction-market evidence (2 contracts)', result['html'])
        self.assertNotIn('Unmatched question', result['html'])
        self.assertIn('TEST-LOWER; TEST-UPPER', row['source_contract_ids'])
        self.assertIn('TEST-LOWER; TEST-UPPER', result['html'])
        self.assertIn('<h3>Search notes</h3>', result['html'])

    def test_absent_or_invalid_range_is_not_invented(self):
        for bounds in (None, [-4], [4, -4], ['-4', 4], [True, 4]):
            with self.subTest(bounds=bounds):
                case = example()
                case['risks'][0]['outcomes'][0]['pnl_interval'] = bounds
                result = decoded(case)
                row = result['rows'][0]
                self.assertEqual(row['model_range_low_usd'], '')
                self.assertEqual(row['model_range_high_usd'], '')
                self.assertEqual(row['direction'], 'NOT_ESTABLISHED_WITHOUT_MODEL_RANGE')
                self.assertIn('Range unavailable', result['html'])
                self.assertNotIn('Direction uncertain', result['html'])

    def test_direction_uses_range_not_point_or_rounding(self):
        tests = [([-30, -2], 'NONPOSITIVE_UNDER_STATED_MODEL'),
                 ([0, 8], 'NONNEGATIVE_UNDER_STATED_MODEL'),
                 ([0, 0], 'ZERO_UNDER_STATED_MODEL'),
                 ([-.01, .01], 'DIRECTION_UNCERTAIN')]
        for bounds, expected in tests:
            with self.subTest(bounds=bounds):
                case = example()
                case['risks'][0]['outcomes'][0]['pnl_interval'] = bounds
                self.assertEqual(decoded(case)['rows'][0]['direction'], expected)

    def test_empty_outcome_includes_source_blocker_without_fake_values(self):
        case = example()
        case['risks'][0]['outcomes'] = []
        case['risks'][0]['quality']['reason'] = 'Source boundary quotes do not identify this event.'
        result = decoded(case)
        self.assertIn('Source boundary quotes do not identify this event.', result['html'])
        row = result['rows'][0]
        for field in ('modeled_change_usd', 'model_range_low_usd', 'model_range_high_usd', 'direction'):
            self.assertEqual(row[field], '')
        self.assertIn('Source boundary quotes', row['unavailable_reason'])

    def test_html_and_csv_injections_stay_inert(self):
        case = example()
        case['risks'][0]['title'] = '=HYPERLINK("https://unsafe.example")<script>x()</script>'
        case['risks'][0]['review_notes'] = [{'reason': '<img onerror="x()" src="x">'}]
        result = decoded(case)
        self.assertNotIn('<script>', result['html'])
        self.assertNotIn('<img onerror=', result['html'])
        self.assertIn('&lt;script&gt;', result['html'])
        self.assertTrue(result['rows'][0]['event'].startswith("'=HYPERLINK"))
        self.assertEqual(result['names'],
                         {'report.html', 'portfolio_data.json', 'scenario_results.csv', 'past_episode_replay.csv', 'README.txt'})
        self.assertEqual(result['case'], case)

    def test_offline_replay_never_implies_fresh_quotes(self):
        for status, label in (
            ('OFFLINE_SAVED_INPUTS_RECOMPUTED_R30', 'Saved analysis recalculated locally'),
            ('ORIGINAL_RESULTS_RETAINED', 'Original saved results retained'),
        ):
            with self.subTest(status=status):
                case = example()
                case['offline_replay'] = {'status': status, 'new_provider_calls': 0}
                result = decoded(case)
                self.assertIn(label, result['html'])
                self.assertIn('quotes remain dated 2026-10-27T21:00:00Z', result['html'])
                self.assertIn('No new source requests.', result['html'])
                self.assertEqual(result['case']['capture'], case['capture'])

    def test_actual_archived_report_exports_every_selected_model_range(self):
        report = json.loads(RECORDED_R21.read_text(encoding='utf-8'))
        case = live_report_to_case(report)
        result = decoded(case)
        index = {(row['event'], row['outcome']): row for row in result['rows'] if row['outcome']}
        outcomes = [o for risk in case['risks'] for o in risk['outcomes']]
        self.assertGreater(len(outcomes), 40)
        for risk in case['risks']:
            for outcome in risk['outcomes']:
                with self.subTest(event=risk['id'], outcome=outcome['id']):
                    row = index[(risk['title'], outcome['label'])]
                    self.assertEqual(float(row['modeled_change_usd']), outcome['pnl'])
                    if outcome.get('pnl_interval'):
                        self.assertEqual(float(row['model_range_low_usd']), outcome['pnl_interval'][0])
                        self.assertEqual(float(row['model_range_high_usd']), outcome['pnl_interval'][1])
        self.assertTrue(any(row['daily_transfer_status'] == 'FAIL' for row in result['rows']))
        self.assertIn('NOT_TESTED', result['html'])


if __name__ == '__main__':
    unittest.main()
