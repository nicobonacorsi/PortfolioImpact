"""Issuer holdings that sum slightly above 100% keep the fund's portfolio weight."""
from datetime import datetime, timezone

import pytest

from castle_risk.adapters import _normalize_constituent
from castle_risk.portfolio import recover_exposures

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _snapshot(weights):
    return {'fund_symbol': 'FND', 'as_of': '2026-10-28', 'retrieved_at': '2026-10-30T00:00:00+00:00',
            'source_url': 'https://example.invalid/holdings', 'reliable': True, 'complete': True, 'weight_basis': 'fraction_of_nav',
            'holdings': [{'symbol': f'S{i}', 'weight': w, 'asset_type': 'stock'} for i, w in enumerate(weights)]}



def test_small_excess_is_scaled_to_conserve_weight():
    from castle_risk.portfolio import parse_portfolio
    portfolio = parse_portfolio([{'symbol': 'FND', 'weight': .4, 'asset_type': 'etf'},
                                 {'symbol': 'XYZ', 'weight': .6, 'asset_type': 'stock'}])
    report = recover_exposures(portfolio, {'FND': _snapshot([.6, .4022])}, now=NOW)
    assert report.total_weight == pytest.approx(1.0, abs=1e-12)
    assert report.unresolved_weight == pytest.approx(0.0, abs=1e-12)


def test_large_excess_still_rejects_snapshot():
    from castle_risk.portfolio import parse_portfolio
    portfolio = parse_portfolio([{'symbol': 'FND', 'weight': 1., 'asset_type': 'etf'}])
    report = recover_exposures(portfolio, {'FND': _snapshot([.6, .45])}, now=NOW)
    assert report.unresolved_weight == pytest.approx(1.0)


@pytest.mark.parametrize('row,expected', [
    ({'constituent_ticker': 'AMT', 'security_type': 'REIT', 'asset_class': 'Real Estate', 'isin': 'US03027X1000'}, 'stock'),
    ({'constituent_ticker': 'ADI', 'security_type': None, 'asset_class': 'Equity', 'isin': 'US0326541051'}, 'stock'),
    ({'constituent_ticker': 'ADI', 'security_type': None, 'asset_class': 'Equity'}, 'other'),
    ({'constituent_ticker': '', 'security_type': 'Currency', 'asset_class': None, 'currency_traded': 'USD'}, 'cash'),
    ({'constituent_ticker': 'ES', 'security_type': 'Future', 'asset_class': 'Equity', 'isin': 'US0000000000'}, 'other'),
])
def test_constituent_typing(row, expected):
    assert _normalize_constituent({'weight': .01, **row}, 'FND')['asset_type'] == expected


def test_named_channels_drop_numerical_residue():
    from castle_risk.portfolio_impact_adapter import _exposure_for_event
    event = {'named_exposure_sensitivity': {'per_symbol': [
        {'symbol': 'TSM', 'portfolio_weight': 0.25}, {'symbol': 'SPY', 'portfolio_weight': 7.9e-9}]}}
    rows, basis = _exposure_for_event(event, [], {'model': {}, 'source': None})
    assert basis == 'NAMED_SOURCE_CHANNELS'
    assert [r['symbol'] for r in rows] == ['TSM']


def test_equity_receipts_and_preferred_are_issuer_equity():
    from castle_risk.adapters import _normalize_constituent
    base = {'constituent_ticker': 'X', 'weight': 0.01, 'asset_class': 'Equity', 'isin': 'US0000000000'}
    for kind in ('Preference', 'ADR', 'NVDR', 'Unit', 'Right'):
        assert _normalize_constituent({**base, 'security_type': kind}, 'EEM')['asset_type'] == 'stock'
    assert _normalize_constituent({**base, 'security_type': 'Warrant'}, 'EEM')['asset_type'] == 'other'
    assert _normalize_constituent({**base, 'asset_class': 'Fixed Income', 'security_type': 'Unit'}, 'EEM')['asset_type'] == 'other'


def test_trace_fixed_income_does_not_make_an_equity_fund_a_bond_fund():
    from castle_risk.fund_composition import portfolio_rate_sensitivity, uncovered_bond_symbols
    from castle_risk.portfolio import parse_portfolio
    portfolio = parse_portfolio([{"symbol": "EMFUND", "weight": .5, "asset_type": "etf"},
                                 {"symbol": "BONDISH", "weight": .5, "asset_type": "etf"}])
    cov = lambda fi: {"fixed_income_duration_coverage": {"fixed_income_net_nav_weight": fi,
                                                         "unresolved_classification_gross_nav_weight": 0.0}}
    rates = portfolio_rate_sensitivity(portfolio, {"EMFUND": cov(1.6e-6), "BONDISH": cov(.4)})
    assert uncovered_bond_symbols(rates) == {"BONDISH"}
    assert rates["portfolio_weight_with_unknown_bond_duration"] == pytest.approx(.5 * 1.6e-6 + .5 * .4)
