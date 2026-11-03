"""Transparent two-factor *stress transmission*, not an event-P&L forecast.

For the independently source-reviewed CAT/UNP/NEE/IEF demonstration portfolio,
we can decompose an EXOGENOUS same-window SPY move and a same-window
10-year Treasury yield move into a modeled position-level value change.

Neither those factor moves nor the event-to-factor link is inferred from a
Kalshi contract, and the result is not a conditional mean given event YES.
The model is intentionally narrow: the bond coefficient uses actual
source-reviewed IEF constituents; the equity beta=1 convention is a
transparent assumption whose cross-regime accuracy must be reported.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class FactorTransmissionError(ValueError):
    """Untrusted or unsupported portfolio/factor inputs cannot be priced."""


DEMO_HOLDINGS = {'CAT': ('stock', .30), 'UNP': ('stock', .25),
                 'NEE': ('stock', .25), 'IEF': ('etf', .20)}
SCHEMA = 'castle.conditional_factor_transmission.v1'
PRESETS = (
    ('RATES_UP_EQUITIES_DOWN', -1.0, 10.0),
    ('RATES_DOWN_EQUITIES_UP', 1.0, -10.0),
    ('NO_MARKET_FACTOR_MOVE', 0.0, 0.0),
)


def _real(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise FactorTransmissionError(name + ' must be a finite numeric value')
    return float(value)


def _portfolio_weights(positions):
    if (not isinstance(positions, Sequence) or isinstance(positions, (str, bytes))
            or len(positions) != len(DEMO_HOLDINGS)):
        raise FactorTransmissionError('Only the source-verified industrial portfolio is supported')
    found = {}
    for p in positions:
        if not isinstance(p, Mapping):
            raise FactorTransmissionError('Each position must be an object')
        symbol, kind = p.get('symbol'), p.get('type', p.get('asset_type'))
        if symbol not in DEMO_HOLDINGS or symbol in found or kind != DEMO_HOLDINGS[symbol][0]:
            raise FactorTransmissionError('Missing, duplicate or misclassified original position')
        weight = _real(p.get('weight'), 'Position weight')
        if not math.isclose(weight, DEMO_HOLDINGS[symbol][1], abs_tol=1e-12, rel_tol=0):
            raise FactorTransmissionError('Original source-verified position weights changed')
        found[symbol] = weight
    if set(found) != set(DEMO_HOLDINGS) or not math.isclose(sum(found.values()), 1., abs_tol=1e-12):
        raise FactorTransmissionError('The original portfolio must contain all four positions')
    return found


def _bond_per_bp(rate_sensitivity, notional):
    if not isinstance(rate_sensitivity, Mapping):
        raise FactorTransmissionError('Missing source-reviewed bond duration')
    positions = rate_sensitivity.get('positions')
    if not isinstance(positions, list) or len(positions) != 1 or positions[0].get('symbol') != 'IEF':
        raise FactorTransmissionError('The source IEF duration must be unique')
    position = positions[0]
    if position.get('basis') != 'SOURCE_PRICE_IMPLIED':
        raise FactorTransmissionError('Bond duration source basis is unverified')
    rate = _real(position.get('usd_per_bp_parallel_rise'), 'Bond dollars per bp')
    combined = _real(rate_sensitivity.get('usd_per_bp_parallel_rise_covered'),
                     'Combined bond dollars per bp')
    if rate >= 0 or not math.isclose(rate, combined, rel_tol=1e-12, abs_tol=1e-9):
        raise FactorTransmissionError('Bond sensitivity does not reconcile')
    covered = _real(position.get('covered_fund_nav_weight'), 'Covered ETF NAV')
    uncovered = _real(position.get('uncovered_fixed_income_fund_nav_weight'),
                      'Uncovered ETF NAV')
    unknown_portfolio = _real(rate_sensitivity.get('portfolio_weight_with_unknown_bond_duration'),
                              'Unknown-duration portfolio share')
    if (not 0 < covered <= 1 or uncovered < 0 or covered + uncovered > 1 + 0.0005
            or unknown_portfolio < 0
            or not math.isclose(unknown_portfolio, uncovered * .2, abs_tol=1e-5)):
        raise FactorTransmissionError('Uncovered duration cannot be silently set to zero')
    # The source amount was for the original $100,000 portfolio. Scale only
    # by a user-selected total notional, never by changing issuer weights.
    return rate * notional / 100000., covered, uncovered


def conditional_factor_stress(*, positions, notional_usd, rate_sensitivity,
                              spy_return_pct, ust10y_yield_change_bps):
    """Compute source-anchored, EXOGENOUS factor-shock scenario dollars.

    This function is deliberately an intermediate economic model. It does
    not accept an event outcome, prediction probability, or historical
    regression coefficient, so callers cannot turn it into an automatic
    event YES/NO forecast without a separately identified bridge.
    """
    weights = _portfolio_weights(positions)
    notional = _real(notional_usd, 'Portfolio notional')
    if not 0 < notional <= 1e10:
        raise FactorTransmissionError('Portfolio notional must be positive and bounded')
    spy = _real(spy_return_pct, 'SPY scenario return percent')
    bps = _real(ust10y_yield_change_bps, '10Y Treasury scenario yield bps')
    if spy < -10 or spy > 10 or abs(bps) > 100:
        raise FactorTransmissionError('Stress outside declared [-10%,10%] and [-100,100] bp domain')
    rate, covered, uncovered = _bond_per_bp(rate_sensitivity, notional)
    equity = {symbol: notional * weights[symbol] * spy / 100.
              for symbol in ('CAT', 'UNP', 'NEE')}
    bond = rate * bps
    total = math.fsum(equity.values()) + bond
    if not math.isclose(total, notional*.8*spy/100 + rate*bps, rel_tol=1e-12, abs_tol=1e-8):
        raise FactorTransmissionError('Portfolio accounting identity failed')
    if not all(math.isfinite(v) for v in (*equity.values(), bond, total)):
        raise FactorTransmissionError('Nonfinite factor-stress output')
    return {
        'schema': SCHEMA,
        'status': 'EXPLICIT_FACTOR_STRESS_NOT_EVENT_CONDITIONAL_PNL',
        'factor_inputs': {'spy_return_pct': spy, 'ust10y_yield_change_bps': bps},
        'portfolio_notional_usd': notional,
        'equity_beta_policy': 'ALL_THREE_NAMED_STOCKS_UNIT_BETA_STRESS_ASSUMPTION',
        'stock_component_usd': equity,
        'ief_covered_bond_component_usd': bond,
        'ief_usd_per_bp_covered': rate,
        'ief_fund_nav_weight_with_known_duration': covered,
        'ief_fund_nav_weight_without_known_duration': uncovered,
        'modeled_portfolio_change_usd': total,
        'modeled_portfolio_return_pct': 100. * total/notional,
        'physical_event_probability_P': None,
        'matched_event_contract_Q': None,
        'event_to_factor_mapping_identified': False,
        'priced_in_identified': False,
        'whole_portfolio_future_return_predicted': False,
        'conditional_on': 'Specified contemporaneous SPY return and UST10Y yield change, not event settlement',
        'outside_the_model': ['Stock-specific returns beyond assumed unit SPY sensitivity',
                              'IEF uncovered duration and ETF tracking / basis risk',
                              'Convexity, curve twists, liquidity, news and changing risk premia'],
        'source_scope': 'Source-reviewed 2026 IEF bond constituents and original CAT/UNP/NEE/IEF positions',
        'calibrated_future_prediction_band_pct': None,
    }


def standard_factor_stresses(*, positions, notional_usd, rate_sensitivity):
    """Three inspectable hypothetical scenarios, no probability or ranking."""
    return [{'preset': name,
             **conditional_factor_stress(positions=positions, notional_usd=notional_usd,
                 rate_sensitivity=rate_sensitivity,
                 spy_return_pct=spy, ust10y_yield_change_bps=bps)}
            for name, spy, bps in PRESETS]
