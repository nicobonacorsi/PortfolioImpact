"""Record a single live assessment or the required three-portfolio acceptance.

Keys are read only from the process environment, never command-line arguments.
Raw Massive inputs stay under runtime/; --output-dir receives derived reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from castle_risk.live_assessment import public_assessment_view, run_live_assessment
from castle_risk.economic_binding import complete_bound_case

PORTFOLIOS = {
    'technology_and_treasury': [
        {'symbol': 'NVDA', 'weight': .35, 'asset_type': 'stock'},
        {'symbol': 'QQQ', 'weight': .45, 'asset_type': 'etf'},
        {'symbol': 'TLT', 'weight': .20, 'asset_type': 'etf'}],
    'bonds': [
        {'symbol': 'BND', 'weight': .6, 'asset_type': 'etf'},
        {'symbol': 'IEF', 'weight': .4, 'asset_type': 'etf'}],
    'outside_old_registry': [
        {'symbol': 'AAPL', 'weight': .35, 'asset_type': 'stock'},
        {'symbol': 'JPM', 'weight': .35, 'asset_type': 'stock'},
        {'symbol': 'GLD', 'weight': .30, 'asset_type': 'etf'}],
}
SCENARIOS = {
    'FED_POLICY': {'benchmark_return_contrast': -.01, 'benchmark_contrast_interval': [-.02, .005]},
    'US_INFLATION': {'benchmark_return_contrast': -.0075, 'benchmark_contrast_interval': [-.015, .005]},
    'TAIWAN_SUPPLY': {'nvda_return_contrast': -.20, 'nvda_contrast_interval': [-.35, .05]},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portfolio-json', type=Path, help='JSON list of symbol, weight and asset_type rows')
    parser.add_argument('--three-portfolios', action='store_true', help='Run all three distinct acceptance portfolios')
    parser.add_argument('--notional', type=float, default=100000.)
    parser.add_argument('--scenarios-json', type=Path, help='Explicit branch contrast assumptions; overrides documented illustrations')
    parser.add_argument('--event-observations-json', type=Path, help='Dated actual/expected surprise records with source references')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'live_acceptance')
    parser.add_argument('--public-only', action='store_true', help='Inspect public contracts; authenticated acceptance remains incomplete')
    args = parser.parse_args(argv)
    if args.three_portfolios and args.portfolio_json:
        parser.error('Choose --three-portfolios or --portfolio-json')
    cases = PORTFOLIOS if args.three_portfolios else {
        'custom': json.loads(args.portfolio_json.read_text(encoding='utf-8'))} if args.portfolio_json else {
        'technology_and_treasury': PORTFOLIOS['technology_and_treasury']}
    scenarios = json.loads(args.scenarios_json.read_text(encoding='utf-8')) if args.scenarios_json else SCENARIOS
    observations = json.loads(args.event_observations_json.read_text(encoding='utf-8')) if args.event_observations_json else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.public_only:
        # This opt-in mode disables authenticated stages without exposing values.
        import os
        for key in ('MASSIVE_API_KEY', 'POLYGON_API_KEY', 'ANTHROPIC_API_KEY'):
            os.environ.pop(key, None)
    receipt = {'schema': 'castle.option1.three_portfolio_live_acceptance.v1',
               'started_at_utc': datetime.now(timezone.utc).isoformat(), 'cases': {},
               'public_only_requested': args.public_only, 'windows_cold_start_verified': False,
               'declared_scenario_basis': 'Primary default: source-defined multi-outcome distributions with published-response and estimated-beta announcement components. Legacy family stresses remain separate; no causal or total-return forecast certification.',
               'authenticated_acceptance_completed': False}
    for name, portfolio in cases.items():
        print(name + ': starting read-only source acquisition', flush=True)
        try:
            report = run_live_assessment(portfolio_rows=portfolio, notional=args.notional, root=ROOT,
                                         scenario_settings=scenarios, event_observations=observations,
                                         include_geopolitical_search=True, include_payrolls=True)
            path = args.output_dir / (name + '.json')
            path.write_text(json.dumps(public_assessment_view(report), indent=2, allow_nan=False), encoding='utf-8')
            massive = report['stages']['Massive']
            claude = report['stages']['Claude']
            receipt['cases'][name] = {
                'status': report['verification_status'], 'evaluated_at_utc': report['evaluated_at_utc'],
                'report': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'stages': report['stages'], 'events': len(report['events']),
                'estimated_total_impacts': sum(e['impact']['delta_pnl_usd'] is not None for e in report['events']),
                'economic_binding_policy': report.get('economic_binding_policy', 'LEGACY_REPORT_NUMERICAL_COMPATIBILITY_ONLY'),
                'family_stress_figures': sum(e.get('family_stress', {}).get('delta_pnl_usd') is not None for e in report['events']),
                'source_defined_binary_questions': len(report.get('source_question_refinement', {}).get('refinements', [])),
                'explicitly_bound_complete_cases': sum(complete_bound_case(e) for e in report['events']),
                'default_component_outcomes': sum(
                    case.get('component_pnl_usd') is not None
                    for event in report.get('default_analysis', {}).get('events', [])
                    for case in (event.get('economics') or {}).get('outcome_cases', [])),
                'default_component_scope': 'DECLARED_ANNOUNCEMENT_COMPONENT_NOT_TOTAL_RETURN_PREDICTION',
                'all_saved_provider_hashes_match': bool(report['provider_receipts']) and all(r['matches_receipt'] for r in report['provider_receipts']),
                'authenticated_calls_executed': massive.get('request_count', 0) > 0 and claude.get('request_count', 0) > 0,
                'claude_source_review_succeeded': claude.get('status') == 'COMPLETE_SOURCE_BACKED_REVIEW',
                'kalshi_reads_succeeded': report['stages']['Kalshi'].get('status') == 'READS_COMPLETED',
                'geopolitical_search_failed_or_partial': report['stages'].get('GeopoliticalSearch', {}).get('status') in {'FAILED', 'PARTIAL'},
                'matched_complete_economic_cases': sum(
                    complete_bound_case(e) if report.get('schema') == 'castle.option1.live_assessment.v2' else
                    e['probability']['pricing_measure_Q_interval'] is not None and
                    e['impact']['delta_pnl_usd'] is not None and
                    e['impact']['priced_in']['yes_pnl_usd'] is not None for e in report['events']),
                'all_acquisitions_succeeded': bool(report['acquisition_attempts']) and all(a['status'] == 'OK' for a in report['acquisition_attempts']),
                'required_acquisitions_succeeded': required_acquisitions_succeeded(report, massive),
                'all_required_holdings_quality_passed': massive.get('holdings_quality_complete', True),
                'required_transport_complete': massive.get('required_transport_complete', False),
                'optional_metadata_failures': massive.get('optional_metadata_failures', []),
                'acquisition_status_scope': 'The gates count the operations the assessment needs; optional metadata the provider did not return is listed above, not counted.',
            }
            print(name + ': ' + report['verification_status'], flush=True)
        except Exception:
            receipt['cases'][name] = {'status': 'RUN_FAILED', 'reason': 'Assessment failed; inspect sanitized runtime receipts.'}
            print(name + ': RUN_FAILED (details suppressed)', flush=True)
    every_case_complete = not args.public_only and all(case_complete(c) for c in receipt['cases'].values())
    receipt['authenticated_acceptance_completed'] = args.three_portfolios and len(cases) >= 3 and every_case_complete
    if args.three_portfolios:
        receipt['status'] = 'PASS_WITH_DISPLAYED_MODEL_LIMITATIONS' if receipt['authenticated_acceptance_completed'] else 'INCOMPLETE_AUTHENTICATED_ACCEPTANCE'
    else:
        # One portfolio is an assessment, not the three-portfolio acceptance test: it reports its own outcome.
        receipt['scope'] = 'SINGLE_ASSESSMENT_THREE_PORTFOLIO_ACCEPTANCE_NOT_RUN'
        receipt['status'] = 'ASSESSMENT_COMPLETE_WITH_DISPLAYED_MODEL_LIMITATIONS' if every_case_complete else 'ASSESSMENT_INCOMPLETE'
    receipt['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / 'acceptance_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(receipt['status'])
    return 0 if receipt['authenticated_acceptance_completed'] or (not args.three_portfolios and every_case_complete) else 2


def required_acquisitions_succeeded(report, massive):
    """Every acquisition the assessment needs returned OK; optional metadata may be missing."""
    optional = set(massive.get('acquisition_quality', {}).get('optional_metadata_operations', []))
    attempts = report['acquisition_attempts']
    return bool(attempts) and all(a['status'] == 'OK' for a in attempts if a.get('operation') not in optional)


def case_complete(case):
    return bool(case.get('authenticated_calls_executed') and case.get('required_acquisitions_succeeded') and
                case.get('all_required_holdings_quality_passed') and
                case.get('claude_source_review_succeeded') and case.get('kalshi_reads_succeeded') and
                not case.get('geopolitical_search_failed_or_partial', False) and
                case.get('all_saved_provider_hashes_match') and case.get('matched_complete_economic_cases', 0) > 0)


if __name__ == '__main__':
    raise SystemExit(main())
