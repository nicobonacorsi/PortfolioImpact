"""Twenty frozen portfolios for structural discovery and accounting checks.

These are *offline invariants*, not a blind estimate of live risk recall or
forecast accuracy. The XOM/TSM/LMT/IEF case was separately preregistered for
live provider evaluation before the R111 model-tuning experiment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest

from castle_risk.agent_budgeted_discovery import concise_exposures
from castle_risk.portfolio import parse_portfolio, recover_exposures
from castle_risk.portfolio_impact_adapter import _risk_watchlist


# Each ticker appears with its declared instrument type. Asset identity must
# still be independently verified by the real provider at live evaluation.
FROZEN_PORTFOLIOS = (
    ('MSFT', 'JPM', 'TLT', 'GLD'),
    ('XOM', 'TSM', 'LMT', 'IEF'),
    ('NVDA', 'AAPL', 'AMZN', 'QQQ'),
    ('JPM', 'BAC', 'WFC', 'AGG'),
    ('UNH', 'LLY', 'PFE', 'XLV'),
    ('NEE', 'DUK', 'SO', 'XLU'),
    ('XOM', 'CVX', 'SLB', 'XLE'),
    ('GLD', 'SLV', 'USO', 'DBC'),
    ('TLT', 'IEF', 'SHY', 'AGG'),
    ('HYG', 'LQD', 'IEF', 'GLD'),
    ('VWO', 'EEM', 'FXI', 'EFA'),
    ('CAT', 'UNP', 'DE', 'XLI'),
    ('LMT', 'RTX', 'NOC', 'ITA'),
    ('TSM', 'AMD', 'ASML', 'SOXX'),
    ('WMT', 'COST', 'HD', 'XLY'),
    ('VNQ', 'O', 'PLD', 'IYR'),
    ('V', 'MA', 'PYPL', 'QQQ'),
    ('MU', 'NVDA', 'AMD', 'XLE'),
    ('SPY', 'TLT', 'GLD', 'JPM'),
    ('EFA', 'LQD', 'XOM', 'MSFT'),
)
ETF_SYMBOLS = frozenset(('QQQ', 'AGG', 'XLV', 'XLU', 'XLE', 'GLD', 'SLV', 'USO',
                         'DBC', 'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'VWO', 'EEM',
                         'FXI', 'EFA', 'XLI', 'ITA', 'SOXX', 'XLY', 'VNQ', 'IYR', 'SPY'))
WEIGHTS = (.4, .3, .2, .1)


@pytest.mark.parametrize('symbols', FROZEN_PORTFOLIOS, ids=lambda x: '-'.join(x))
def test_new_portfolio_exposures_and_research_inspection_have_no_implicit_pnl(symbols):
    portfolio_input = [{'symbol': symbol, 'weight': weight,
                        'asset_type': 'etf' if symbol in ETF_SYMBOLS else 'stock'}
                       for symbol, weight in zip(symbols, WEIGHTS)]
    portfolio = parse_portfolio(portfolio_input, notional=100000.)
    ledger = recover_exposures(portfolio, {}, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    context = concise_exposures(portfolio, ledger)
    positions = context['portfolio']['positions']
    assert {row['symbol'] for row in positions} == set(symbols)
    assert math.isclose(math.fsum(row['weight'] for row in positions), 1., abs_tol=1e-12)
    assert math.isclose(context['portfolio']['omitted_positions_weight'], 0., abs_tol=1e-12)
    assert context['portfolio']['weights_not_renormalized'] is True
    assert ledger.unresolved_weight >= 0.

    # A model hypothesis is matched to its actual holding by an explicit
    # symbol, not by a guessed issuer beta or synthetic news event.
    hypotheses = [{'risk_id': f'risk-{symbol.lower()}',
                   'risk_title': f'{symbol} stress test hypothesis',
                   'exposure_rationale': f'{symbol} named original holding'}
                  for symbol in symbols]
    watch = _risk_watchlist({'risk_watchlist': hypotheses}, portfolio_input)
    assert len(watch) == 4
    assert [row['exposed_positions'][0]['symbol'] for row in watch] == list(symbols)
    assert [row['linked_direct_portfolio_weight'] for row in watch] == list(WEIGHTS)
    assert all(row['pnl'] is row['probability'] is row['portfolio_impact_usd'] is None for row in watch)
