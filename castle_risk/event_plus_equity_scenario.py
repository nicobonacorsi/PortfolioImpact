"""R44 direct market-defined Treasury state + *explicit* stock what-if.

This computes transparent portfolio-component stresses only when a user
chooses an exogenous common CAT/UNP/NEE return hypothesis. Nothing here
estimates conditional stock returns, probabilities or priced-in fractions.
"""
from __future__ import annotations

from math import isfinite
from collections.abc import Mapping

from .conditional_factor_transmission import _portfolio_weights
from .treasury_curve_basis import hypothetically_widen_r42_ief_proxy_bands


SCHEMA='castle.market_defined_yield_plus_explicit_equity_stress.v1'


class EventEquityScenarioError(ValueError):
    pass


def event_plus_equity_scenario(*,r42,r43,positions,notional_usd,
                               user_selected_common_equity_return_pct):
    """Join independently verified fixed bond stress and user-chosen stocks.

    `R42`/`R43` audit and full source hashes must have been independently
    verified by the external source-byte scripts before calling this map.
    """
    weights=_portfolio_weights(positions)
    if (type(notional_usd) not in (int,float) or not isfinite(notional_usd)
            or not 0<notional_usd<=1e10):
        raise EventEquityScenarioError('Original portfolio notional is unverified')
    s=user_selected_common_equity_return_pct
    if type(s) not in (int,float) or not isfinite(s) or not -10<=s<=10:
        raise EventEquityScenarioError('A finite explicitly selected stock return in [-10%, 10%] is mandatory')
    if not isinstance(r42,Mapping) or not isinstance(r43,Mapping):
        raise EventEquityScenarioError('The source-verified R42/R43 inputs are required')
    # This source-reconciled function fails closed on market identity, quote
    # states, original IEF sensitivity and experimental basis audit status.
    widened=hypothetically_widen_r42_ief_proxy_bands(r42=r42,r43=r43)
    original_source=r42['source_defined_yield_bands']
    scalar=notional_usd/100000.
    equity={symbol:notional_usd*weights[symbol]*s/100.
            for symbol in ('CAT','UNP','NEE')}
    eq_total=sum(equity.values())
    equity_dollars_per_percentage_point=notional_usd*sum(
        weights[symbol] for symbol in ('CAT','UNP','NEE'))/100.
    if not isfinite(eq_total):
        raise EventEquityScenarioError('Unbounded equity scenario arithmetic')
    rows=[]
    for i, source in enumerate(widened['states']):
        base=original_source[i]
        original_q=base.get('indicative_reference_Q_bin_mass_interval')
        original=source['7y_proxy_ief_stress_usd_interval']
        r={'treasury_10y_settlement_yield_interval':source['source_treasury_10y_yield_interval_pct'],
           'reference_Q_not_physical_event_probability':original_q,
           'explicit_user_stock_return_assumption_pct':float(s),
           'equity_assumption_dollar_components':equity,
           'equity_assumption_dollar_total':eq_total,
           'modeled_ief_covered_proxy_dollar_range':None,
           'modeled_portfolio_components_dollar_range':None,
           'equity_return_break_even_transition_pct_interval':None,
           'covered_bond_admission':source['admission']}
        if original is not None:
            bond=[scalar*float(x) for x in original]
            total=[eq_total+x for x in bond]
            if not all(isfinite(x) for x in (*bond,*total)) or total[0]>total[1]:
                raise EventEquityScenarioError('Invalid original-bond stress interval')
            r['modeled_ief_covered_proxy_dollar_range']=bond
            r['modeled_portfolio_components_dollar_range']=total
            r['stress_range_endpoint_flags']=['CLOSED_MINIMUM','OPEN_SUPREMUM']
            # Sharp within *model* break-even range: if stock s is below
            # its left edge both modeled bounds are negative; if above its
            # right edge they are positive. This says nothing about the
            # actual event-caused stock return or omitted portfolio risk.
            r['equity_return_break_even_transition_pct_interval']=[
                -bond[1]/equity_dollars_per_percentage_point,
                -bond[0]/equity_dollars_per_percentage_point]
        rows.append(r)
    if len(rows)!=9 or sum(r['modeled_portfolio_components_dollar_range'] is not None for r in rows)!=6:
        raise EventEquityScenarioError('R42/R43 exact source eligibility changed')
    covered_nav=float(r43['covered_ief_fund_nav_weight'])
    unresolved_nav=float(r43['unresolved_ief_fund_nav_weight'])
    if not (0<covered_nav<=1 and 0<=unresolved_nav<=1
            and abs(unresolved_nav-.084843)<1e-5):
        raise EventEquityScenarioError('Unreconciled original covered IEF holdings')
    return {
        'schema':SCHEMA,
        'status':'USER_CHOSEN_COMMON_STOCK_RETURN_PLUS_TREASURY_CONTRACT_BOND_PROXY_STRESS',
        'portfolio_tickers':['CAT','UNP','NEE','IEF'],
        'portfolio_notional_usd':float(notional_usd),
        'market_event_id':widened['market_event_id'],
        'source_yield_baseline_pct':widened['official_10y_baseline_pct'],
        'user_must_explicitly_select_the_stock_scenario':True,
        'user_assumption_not_event_estimate':{
            'stock_tickers':['CAT','UNP','NEE'],
            'all_three_share_the_selected_return_pct':float(s),
            'no_causal_estimate_of_stock_response':True},
        'covered_ief_fund_nav_weight':covered_nav,
        'equity_position_value_usd':notional_usd*.8,
        'equity_dollars_per_user_return_percentage_point':equity_dollars_per_percentage_point,
        'excluded_ief_fund_nav_weight':unresolved_nav,
        'unmodeled_ief_portfolio_weight':weights['IEF']*unresolved_nav,
        'source_yield_band_scenarios':rows,
        'quote_reference_is_physical_probability':False,
        'conditional_expected_portfolio_return':None,
        'realized_total_portfolio_event_PNL':None,
        'already_priced_in_amount':None,
        'future_stress_coverage_guaranteed':False,
        'release_allowed':False,
        'caveats':[
            'All three stock returns are a USER assumption, not inferred from the market YES event.',
            'The covered IEF stress assumes a 7Y CMT proxy and an empirically chosen historical basis tolerance, not a future bound.',
            'The other IEF NAV, carry, convexity and stock-specific news remain outside the modeled components.',
            'Quote-based Q-bin ranges cannot be multiplied by modeled stresses to infer a physical expected return.',
        ]}
