"""r9: bond funds and single-line vehicles accepted only within declared, measured tolerances.

The structures come from the 2026-10-23 live captures (BND: netted TBA rows, repeated
lots, a ticker-field collision, unidentified short-term cash and 1.55% of NAV not
listed; GLD: one unidentified line at 100.025% of NAV). The fixtures are synthetic.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from castle_risk.acquisition_status import massive_acquisition_stage
from castle_risk.adapters import _normalize_constituent
from castle_risk.default_assessment_ui import _fund_reviews_caption
from castle_risk.fund_composition import _isin_check
from castle_risk.outcome_table import holdings_quality_gaps
from castle_risk.source_class_quality import review_source_class_snapshot


def isin(body):
    """A check-digit-valid ISIN from an 11-character body."""
    return next(body + d for d in "0123456789" if _isin_check(body + d))


def snapshot(tmp_path, fund, rows):
    rows = deepcopy(rows)
    for rank, row in enumerate(rows, start=1):
        row.update(composite_ticker=fund, constituent_rank=rank,
                   effective_date='2026-09-30', processed_date='2026-10-22')
    body = json.dumps({'status': 'OK', 'results': rows}, sort_keys=True).encode()
    raw = tmp_path / 'page.json'
    raw.write_bytes(body)
    meta = {'url': 'https://api.massive.com/etf-global/v1/constituents?composite_ticker=' + fund,
            'sha256': sha256(body).hexdigest(), 'status': 200, 'fixture': False,
            'retrieved_at': '2026-10-23T15:40:00+00:00'}
    (tmp_path / 'page.meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return {'fund_symbol': fund, 'weight_basis': 'fraction_of_nav', 'reliable': False, 'complete': True,
            'as_of': '2026-09-30', 'processed_date': '2026-10-22',
            'holdings': [_normalize_constituent(row, fund) for row in rows],
            'metadata': {'selection_provisional': False, 'selected_batch': ['2026-09-30', '2026-10-22'],
                         'pages': [{'raw_path': str(raw), 'sha256': meta['sha256'], 'retrieved_at': meta['retrieved_at']}],
                         'quality_issues': ['NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT'],
                         'selected_raw_row_count': len(rows), 'exact_duplicate_rows_removed': 0}}


def bond(ticker, code, weight, security='US GOVERNMENT', name='BOND'):
    return {'constituent_ticker': ticker, 'isin': code, 'weight': weight, 'asset_class': 'Fixed Income',
            'security_type': security, 'constituent_name': name, 'currency_traded': 'USD'}


def bond_fund(**change):
    tba = isin('US01F020600')
    lot = isin('US05593MAH6')
    rows = [bond('T', isin('US91282CAA0'), .55), bond('T', isin('US91282CAB0'), .30),
            bond('XYZ', isin('US98765432A'), .105, 'Corp'),
            bond('FNCL', tba, .03, 'MBS 30yr', 'FNMA TBA 30 YR 2'), bond('FNCL', tba, -.02, 'MBS 30yr', 'FNMA TBA 30 YR 2'),
            bond('BMO', lot, .0003, 'CMBS', 'BMO MORTGAGE TRUST'), bond('BMO', lot, .0002, 'CMBS', 'BMO MORTGAGE TRUST'),
            bond('TBND', isin('US3137FNAE0'), .0004, 'Agncy CMBS', 'FREDDIE MAC CMBS'),
            {'asset_class': 'SHORT_TERM', 'weight': .02},
            {'constituent_ticker': 'USD', 'asset_class': 'Cash', 'security_type': 'Cash', 'currency_traded': 'USD', 'weight': -.0005}]
    if change.get('short') is not None:
        rows[4]['weight'] = change['short']
    if change.get('short_isin'):
        rows[4]['isin'] = isin('US01F099900')
    if change.get('corporate') is not None:
        rows[2]['weight'] = change['corporate']
    if change.get('collision') is not None:
        rows[7]['weight'] = change['collision']
    if change.get('lots') is not None:
        rows[5]['weight'] = rows[6]['weight'] = change['lots']
    if change.get('cash_like') is not None:
        rows[8]['weight'] = change['cash_like']
        rows[0]['weight'] -= change['cash_like'] - .02
    if change.get('unidentified_bond'):
        rows[2]['weight'] -= .01
        rows.append({'asset_class': 'Fixed Income', 'weight': .01})
    return rows


def test_bond_fund_with_the_five_recorded_structures_is_accepted_and_every_tolerance_is_measured(tmp_path):
    result = review_source_class_snapshot(snapshot(tmp_path, 'TBND', bond_fund()), symbol='TBND')
    assert result['status'] == 'SIGNED_FIXED_INCOME_ACCEPTED_WITH_DECLARED_TOLERANCES'
    assert result['holdings_accepted_for_assessment'] is True
    assert result['eligible_for_source_class_attribution'] is True
    assert result['eligible_for_equity_issuer_lookthrough'] is False
    assert result['weights_reweighted_or_clipped'] is False
    used = {t['tolerance']: t['measured'] for t in result['declared_tolerances_used']}
    assert used['SHORT_ROWS_NETTED_BY_SAME_IDENTIFIER_LONGS'] == {'rows': 1, 'gross_short_nav': pytest.approx(-.02)}
    assert used['REPEATED_ROWS_OF_ONE_IDENTIFIER_SUMMED']['nav'] == pytest.approx(.0005)
    assert used['TICKER_FIELD_EQUALS_FUND_KEPT_UNRESOLVED'] == {'rows': 1, 'nav': pytest.approx(.0004)}
    assert used['UNIDENTIFIED_CASH_OR_SHORT_TERM_ROWS']['nav'] == pytest.approx(.02)
    assert used['LISTED_NAV_SHORTFALL']['unlisted_nav'] == pytest.approx(1 - .9854)


@pytest.mark.parametrize('change', [
    {'short': -.04},                 # the TBA group is net short
    {'short_isin': True},            # a short row with no same-identifier long
    {'corporate': .094},             # 2.46% of NAV not listed
    {'collision': .002},             # ticker-field collision above 0.1% of NAV
    {'lots': .001},                  # repeated lots of one security above 0.1% of NAV
    {'cash_like': .06},              # unidentified short-term rows above 5% of NAV
    {'unidentified_bond': True},     # an unidentified row the provider calls a bond
])
def test_each_tolerance_breach_keeps_the_structural_gap(tmp_path, change):
    result = review_source_class_snapshot(snapshot(tmp_path, 'TBND', bond_fund(**change)), symbol='TBND')
    assert result['status'] == 'SOURCE_CLASS_STRUCTURAL_GAPS_RETAINED'
    assert result['holdings_accepted_for_assessment'] is False
    assert result['eligible_for_source_class_attribution'] is False


def vehicle(**change):
    rows = [{'weight': change.get('weight', 1.00025)}, {'weight': 0.0}]
    if change.get('ticker'):
        rows[0]['constituent_ticker'] = 'XYZ'
    if change.get('two_lines'):
        rows[0]['weight'], rows[1]['weight'] = .6, .4
    if change.get('liability_row'):
        rows[1]['weight'] = -.0005
    return rows


def test_single_line_exchange_traded_vehicle_is_priced_as_a_whole(tmp_path):
    result = review_source_class_snapshot(snapshot(tmp_path, 'TGLD', vehicle()), symbol='TGLD',
                                          instrument={'type': 'ETV', 'name': 'Test Gold Trust'})
    assert result['status'] == 'SINGLE_LINE_EXCHANGE_TRADED_VEHICLE_PRICED_AS_A_WHOLE'
    assert result['holdings_accepted_for_assessment'] is True
    assert result['eligible_for_source_class_attribution'] is False
    assert result['eligible_for_equity_issuer_lookthrough'] is False
    assert result['declared_tolerances_used'][0]['measured'] == {'line_nav': pytest.approx(1.00025)}
    assert 'not inferred' in result['reason']


@pytest.mark.parametrize('instrument, change', [
    ({'type': 'ETF'}, {}),                      # an investment company could hide a master fund
    (None, {}),                                 # no provider type
    ({'type': 'ETV'}, {'weight': 1.002}),       # outside 0.1% of NAV
    ({'type': 'ETV'}, {'ticker': True}),        # a named holding would need look-through
    ({'type': 'ETV'}, {'two_lines': True}),
    ({'type': 'ETV'}, {'liability_row': True}),
])
def test_other_single_line_shapes_are_not_accepted(tmp_path, instrument, change):
    result = review_source_class_snapshot(snapshot(tmp_path, 'TGLD', vehicle(**change)), symbol='TGLD', instrument=instrument)
    assert result['holdings_accepted_for_assessment'] is False


def test_accepted_review_passes_the_acquisition_gate_and_is_not_a_table_gap():
    attempts = [{'symbol': 'BND', 'operation': op, 'status': 'OK'} for op in ('ticker_details', 'daily_prices')]
    attempts.append({'symbol': 'BND', 'operation': 'holdings', 'status': 'OK',
                     'data_quality_status': 'ACCEPTED_BY_DECLARED_HOLDINGS_REVIEW'})
    quality = {'all_required_transport_operations_succeeded': True, 'all_required_holdings_quality_passed': True}
    assert massive_acquisition_stage({'attempts': attempts, 'acquisition_quality': quality}, request_count=3)['status'] == 'ACQUIRED'
    attempts[-1]['data_quality_status'] = 'UNRELIABLE_OR_INCOMPLETE_SNAPSHOT'
    assert massive_acquisition_stage({'attempts': attempts, 'acquisition_quality': quality}, request_count=3)['status'] == 'PARTIAL'
    inputs = {'holdings': {'BND': {'reliable': False, 'metadata': {'quality_issues': ['SELF_REFERENTIAL_FUND_HOLDING']}},
                           'GLD': {'reliable': False, 'metadata': {'quality_issues': ['REPORTED_NAV_WEIGHT_EXCEEDS_ONE']}}}}
    gaps = holdings_quality_gaps(inputs, {}, {'BND': {'holdings_accepted_for_assessment': True},
                                              'GLD': {'holdings_accepted_for_assessment': False}})
    assert list(gaps) == ['GLD']


class Captions:
    def __init__(self):
        self.text = []

    def caption(self, text):
        self.text.append(text)


def test_caption_states_each_measured_tolerance_in_words():
    st = Captions()
    _fund_reviews_caption(st, {'fund_holdings_reviews': [
        {'fund_symbol': 'TLT', 'status': 'SIGNED_FIXED_INCOME_SOURCE_CLASS_ACCEPTED', 'holdings_accepted_for_assessment': True},
        {'fund_symbol': 'BND', 'status': 'SIGNED_FIXED_INCOME_ACCEPTED_WITH_DECLARED_TOLERANCES',
         'holdings_accepted_for_assessment': True, 'declared_tolerances_used': [
             {'tolerance': 'LISTED_NAV_SHORTFALL', 'measured': {'unlisted_nav': .01547}},
             {'tolerance': 'SHORT_ROWS_NETTED_BY_SAME_IDENTIFIER_LONGS', 'measured': {'rows': 11, 'gross_short_nav': -.0022212}}]},
        {'fund_symbol': 'GLD', 'status': 'SINGLE_LINE_EXCHANGE_TRADED_VEHICLE_PRICED_AS_A_WHOLE',
         'holdings_accepted_for_assessment': True, 'declared_tolerances_used': [
             {'tolerance': 'SINGLE_LINE_WITHIN_NAV_BAND', 'measured': {'line_nav': 1.0002499644}}]},
        {'fund_symbol': 'QQQ', 'status': 'PLAIN_EQUITY_HOLDINGS_ALREADY_ADMITTED', 'holdings_accepted_for_assessment': True}]})
    text = st.text[0]
    assert 'TLT: bond holdings accepted' in text
    assert 'BND: 1.55% of NAV not listed, not renormalized; 11 short rows (0.22% of NAV)' in text
    assert 'GLD: one holding line at 100.025% of NAV' in text
    assert 'QQQ' not in text
