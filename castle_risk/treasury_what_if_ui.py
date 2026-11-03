"""R45 read-only, source-replayed bridge for one historical Castle demo.

An original-event quote + explicit stock scenario is a diagnostic component,
NOT event-conditional investment P&L. Never attaches to arbitrary portfolios.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from .event_plus_equity_scenario import event_plus_equity_scenario

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = '20261028T154706240078Z'
ORIGINAL_ASSESSMENT_SHA256 = (
    '3ec52572e2c78bde8376fdb44ee07ac1230453f8c5576fcb2ed028e1b1d997cd'
)
R42_RECEIPT_SHA256 = (
    '1bb4ab5ba6234c22ac925d0381467873fb9ca41d00eb2a249d67e026700fe93e'
)
R43_RECEIPT_SHA256 = (
    '6964d3981b8264ae80cee455d6fbb10e46b20e6a7e2fe689f2b810efe9991b22'
)
EVIDENCE_DIR = ROOT / 'data' / 'r45_source_verified_what_if'
EVENT_ID = 'source-kxnote10y-26dec31-what-if'
PUBLIC_SCHEMA = 'castle.r45_read_only_treasury_what_if.v1'


class ResearchWhatIfUnavailable(ValueError):
    """Original saved inputs or source evidence fail the pinned integrity gate."""


def _pinned_json(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ResearchWhatIfUnavailable('Required original source receipt is missing')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ResearchWhatIfUnavailable('Original source receipt was modified')
    return json.loads(raw)


def verify_original_context(assessment_path, *, evidence_dir=EVIDENCE_DIR):
    """Independently replay original HTTP bytes and 9 years of Treasury CSVs."""
    assessment_path = Path(assessment_path)
    if assessment_path.is_symlink() or not assessment_path.is_file():
        raise ResearchWhatIfUnavailable('Original portfolio record is unavailable')
    assessment = _pinned_json(assessment_path, ORIGINAL_ASSESSMENT_SHA256)
    evidence_dir = Path(evidence_dir)
    r42dir, r43dir = evidence_dir / 'r42', evidence_dir / 'r43'
    r42 = _pinned_json(r42dir / 'direct_yield_bridge_receipt.json', R42_RECEIPT_SHA256)
    r43 = _pinned_json(r43dir / 'r43_curve_basis_audit.json', R43_RECEIPT_SHA256)
    # Recompute from the original source CSV/JSON bytes, never merely trust
    # the pinned output receipt. This is an offline deterministic operation.
    from scripts.r42_verify_direct_yield_capture import verify as verify_r42
    from scripts.r43_verify_treasury_curve_basis import verify as verify_r43
    a = verify_r42(r42dir, assessment_path)
    b = verify_r43(r43dir, assessment_path)
    if (a['status'] != 'R42_SOURCE_BYTES_AND_INDEPENDENT_ANALYTIC_AUDIT_PASS'
            or b['status'] != 'R43_SOURCE_BYTES_AND_INDEPENDENT_DECIMAL_HOLDOUT_REPLAY_PASS'
            or b['calibration_windows'] != 25 or b['heldout_windows'] != 20
            or b['covered_out_of_time'] != 18 or b['release_allowed'] is not False):
        raise ResearchWhatIfUnavailable('Frozen independent source replay failed')
    portfolio = assessment['portfolio']
    if portfolio['notional'] != 100000 or r42['event_id'] != 'KXNOTE10Y-26DEC31':
        raise ResearchWhatIfUnavailable('Original market or portfolio identity changed')
    return {'r42': r42, 'r43': r43, 'positions': portfolio['positions'],
            'assessment_sha256': ORIGINAL_ASSESSMENT_SHA256,
            'source_asof': r42['source_retrieved_at_utc'],
            'source_yield_baseline_asof': r42['treasury_baseline_asof']}


def append_original_research_event(case, *, context):
    """Add one plainly labeled event to the original case without ranking it."""
    if (not isinstance(case, dict) or case.get('user_supplied') is not True
            or case.get('notional') != 100000):
        raise ResearchWhatIfUnavailable('The what-if is not a generic portfolio estimate')
    expected = {'CAT': ('stock', .30), 'UNP': ('stock', .25),
                'NEE': ('stock', .25), 'IEF': ('etf', .20)}
    positions = case.get('positions', [])
    if (not isinstance(positions, list) or len(positions) != 4
            or {p.get('symbol'): (p.get('type'), p.get('weight')) for p in positions} != expected):
        raise ResearchWhatIfUnavailable('Unverified portfolio holdings')
    if any(r.get('id') == EVENT_ID for r in case.get('risks', [])):
        raise ResearchWhatIfUnavailable('Duplicated source-event scenario')
    research = {
        'id': EVENT_ID,
        'title': '10-year Treasury yield on Dec 31, 2026',
        'family': 'OTHER',
        'date': '2026-12-31',
        'source_defined_scenario': True,
        'outcomes': [],
        'market_outcomes': [],
        'exposure': [],
        'admitted': [],
        'rejected': [],
        'quality': {
            'model_admission': {
                'state': 'DIAGNOSTIC_ONLY',
                'primary_impact_allowed': False,
                'economic_ranking_allowed': False,
                'reason': 'Treasury source market plus an explicit user stock assumption; not a causal event-conditional portfolio forecast.'}},
        'what_if_explorer': {
            'schema': PUBLIC_SCHEMA,
            'source_event_id': context['r42']['event_id'],
            'quote_captured_at': context['source_asof'],
            'official_baseline_date': context['source_yield_baseline_asof'],
            'original_case_only': True,
            'user_assumption_required': True,
            'full_portfolio_pnl_identified': False}}
    new = dict(case)
    new['risks'] = [*case['risks'], research]
    return new


def evaluate_original_what_if(context, *, equity_return_pct, notional_usd):
    """Run the actual guarded R44 engine for a user-selected return, read-only."""
    if (type(notional_usd) not in (int, float)
            or not isfinite(notional_usd) or not 0 < notional_usd <= 1e9):
        raise ValueError('Select a finite portfolio value between $0 and $1 billion')
    result = event_plus_equity_scenario(
        r42=context['r42'], r43=context['r43'],
        positions=context['positions'], notional_usd=float(notional_usd),
        user_selected_common_equity_return_pct=equity_return_pct)
    if (result['market_event_id'] != 'KXNOTE10Y-26DEC31'
            or result['release_allowed'] is not False
            or result['conditional_expected_portfolio_return'] is not None
            or result['realized_total_portfolio_event_PNL'] is not None
            or result['already_priced_in_amount'] is not None
            or len(result['source_yield_band_scenarios']) != 9):
        raise ResearchWhatIfUnavailable('Source-scope or PNL honesty gate failed')
    return {
        'schema': PUBLIC_SCHEMA,
        'market_event_id': result['market_event_id'],
        'market_close_date': '2026-12-31',
        'source_captured_at': context['source_asof'],
        'source_yield_baseline_asof': context['source_yield_baseline_asof'],
        'source_10y_baseline_pct': result['source_yield_baseline_pct'],
        'portfolio_notional_usd': result['portfolio_notional_usd'],
        'explicit_user_equity_return_pct': float(equity_return_pct),
        'equity_dollar_component': result['equity_position_value_usd'] * float(equity_return_pct) / 100.,
        'yield_bands': [
            {'index': i,
             'yield_interval_pct': row['treasury_10y_settlement_yield_interval'],
             'market_Q_indicative_bin_interval_not_physical_P':
                 row['reference_Q_not_physical_event_probability'],
             'covered_ief_proxy_stress_usd_range':
                 row['modeled_ief_covered_proxy_dollar_range'],
             'assumed_stock_plus_covered_bond_stress_usd_range':
                 row['modeled_portfolio_components_dollar_range'],
             'stock_break_even_pct_transition_interval':
                 row['equity_return_break_even_transition_pct_interval'],
             'model_admission': row['covered_bond_admission']}
            for i, row in enumerate(result['source_yield_band_scenarios'])
        ],
        'excluded_ief_fund_nav_fraction': result['excluded_ief_fund_nav_weight'],
        'actual_full_portfolio_event_pnl': None,
        'physical_event_probability': None,
        'already_priced_in_amount': None,
        'source_quotes_are_not_fresh': True,
        'release_allowed': False}
