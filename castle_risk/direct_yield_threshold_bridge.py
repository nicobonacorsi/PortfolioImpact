"""Source-exact 10Y Treasury predicate -> covered IEF *parallel-rate stress*.

The event's settlement variable is the US Treasury 10-year par yield. That
provides a direct mapping to an exogenous rate stress, **not** to an observed
IEF return, physical probability, priced-in fraction or total portfolio P&L.
All results are explicitly conditional on the parallel-curve, first-order,
source-vintage assumptions described in R42_DIRECT_YIELD_PREDICATE_PROTOCOL.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
import json
import math
import re

from scipy.optimize import linprog

from .conditional_factor_transmission import _bond_per_bp, _portfolio_weights


class YieldPredicateBridgeError(ValueError):
    """A source, model-domain, quote or accounting requirement failed."""


SCHEMA = 'castle.direct_yield_predicate_to_bond_stress.v1'
EVENT = 'KXNOTE10Y-26DEC31'
SOURCE_DATE = date(2026, 12, 31)
STRIKES = tuple(Decimal(x) for x in
                ('4.74', '4.99', '5.09', '5.14', '5.24', '5.49', '5.74', '5.99'))
SECONDARY_RULE = ('The market will expire at the sooner of the first 7:00 PM ET '
                  'following the data release for Dec 31, 2026, or one week '
                  'following Dec 31, 2026.')
RULE = re.compile(r'If the 10Y US Treasury Yield is above '
                  r'([0-9]+(?:\.[0-9]+)?)% on Dec 31, 2026, '
                  r'then the market resolves to Yes\.')


def _data_bytes(payload, label):
    if isinstance(payload, bytes):
        value = payload
    elif isinstance(payload, str):
        value = payload.encode('utf-8')
    else:
        raise YieldPredicateBridgeError(label + ' must contain original source bytes')
    if not 1 <= len(value) <= 3_000_000:
        raise YieldPredicateBridgeError(label + ' exceeds the bounded source size')
    return value


def _source_sha(payload, expected, label):
    data = _data_bytes(payload, label)
    if (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
            or sha256(data).hexdigest() != expected):
        raise YieldPredicateBridgeError(label + ' source SHA256 mismatch')
    return data


def _decimal(value, label):
    if isinstance(value, bool) or type(value) not in (str, int, float, Decimal):
        raise YieldPredicateBridgeError(label + ' is not a finite decimal')
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise YieldPredicateBridgeError(label + ' is not a finite decimal') from exc
    if not d.is_finite():
        raise YieldPredicateBridgeError(label + ' is not finite')
    return d


def _treasury_yield(csv_bytes, asof):
    try:
        rows = csv.DictReader(StringIO(csv_bytes.decode('utf-8-sig')))
        if not rows.fieldnames or rows.fieldnames.count('Date') != 1 or rows.fieldnames.count('10 Yr') != 1:
            raise YieldPredicateBridgeError('Official Treasury CSV lacks unique Date and 10 Yr columns')
        matches = [r for r in rows if datetime.strptime(r['Date'], '%m/%d/%Y').date() == asof]
    except (UnicodeError, TypeError, ValueError, KeyError) as exc:
        raise YieldPredicateBridgeError('Malformed official Treasury daily CSV') from exc
    if len(matches) != 1:
        raise YieldPredicateBridgeError('Requested dated 10Y par yield is absent or duplicated')
    y = _decimal(matches[0]['10 Yr'], 'Official 10Y yield')
    if not Decimal('0') <= y <= Decimal('30'):
        raise YieldPredicateBridgeError('Treasury par-yield source outside declared unit domain')
    return y


def _parse_markets(raw):
    try:
        data = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise YieldPredicateBridgeError('Invalid Kalshi source JSON') from exc
    if not isinstance(data, dict) or data.get('cursor') not in ('', None):
        raise YieldPredicateBridgeError('Incomplete Kalshi source-market pagination')
    markets = data.get('markets')
    if not isinstance(markets, list) or len(markets) != len(STRIKES):
        raise YieldPredicateBridgeError('Incomplete/expanded unreviewed 2026 yield threshold ladder')
    parsed = {}
    for m in markets:
        if not isinstance(m, dict):
            raise YieldPredicateBridgeError('Nonobject market in Kalshi source')
        strike = _decimal(m.get('floor_strike'), 'Kalshi floor strike')
        if strike not in STRIKES or strike in parsed:
            raise YieldPredicateBridgeError('Missing, duplicate or out-of-scope yield strike')
        exact_ticker = EVENT + '-T' + format(strike, '.2f')
        rule = m.get('rules_primary')
        parsed_rule = RULE.fullmatch(rule) if isinstance(rule, str) else None
        if (m.get('ticker') != exact_ticker or m.get('event_ticker') != EVENT
                or m.get('strike_type') != 'greater'
                or m.get('market_type') != 'binary' or m.get('status') != 'active'
                or m.get('rules_secondary') != SECONDARY_RULE
                or not parsed_rule or _decimal(parsed_rule.group(1), 'Rule yield') != strike):
            raise YieldPredicateBridgeError('Unverified event, target date, units or YES predicate')
        try:
            close = datetime.fromisoformat(m['close_time'].replace('Z', '+00:00'))
        except (KeyError, ValueError, AttributeError) as exc:
            raise YieldPredicateBridgeError('Missing exact-market close clock') from exc
        if close.tzinfo is None or close.astimezone(timezone.utc).date() != SOURCE_DATE:
            raise YieldPredicateBridgeError('Market close incompatible with source target day')
        bid = _decimal(m.get('yes_bid_dollars'), 'Source YES bid')
        ask = _decimal(m.get('yes_ask_dollars'), 'Source YES ask')
        if not Decimal('0') < bid <= ask < Decimal('1'):
            raise YieldPredicateBridgeError('Invalid or unquoted source bid/ask')
        parsed[strike] = {'ticker':exact_ticker, 'yes_bid':float(bid),
                          'yes_ask':float(ask),'rule_primary':rule,
                          'contract_close_utc':close.isoformat()}
    if tuple(sorted(parsed)) != STRIKES:
        raise YieldPredicateBridgeError('Source yield threshold ladder is not complete')
    return [dict(threshold_yield_pct=float(s),**parsed[s]) for s in STRIKES]


def _coherent_quote_bins(markets):
    """Sharp individual bin-mass LP ranges under all quoted nested tails."""
    n = len(markets)
    bounds = [(m['yes_bid'],m['yes_ask']) for m in markets]
    inequalities = [[0.] * n for _ in range(n-1)]
    for i, row in enumerate(inequalities):
        row[i] = -1.
        row[i+1] = 1.
    feasibility = linprog([0.]*n,A_ub=inequalities,b_ub=[0.]*(n-1),
                          bounds=bounds,method='highs')
    if not feasibility.success:
        raise YieldPredicateBridgeError('Infeasible simultaneous Kalshi nested bid/ask constraints')
    out=[]
    for index in range(n+1):
        c=[0.]*n
        constant=0.
        if index==0:
            constant=1.
            c[0]=-1.
        elif index==n:
            c[-1]=1.
        else:
            c[index-1]=1.
            c[index]=-1.
        low=linprog(c,A_ub=inequalities,b_ub=[0.]*(n-1),bounds=bounds,method='highs')
        high=linprog([-v for v in c],A_ub=inequalities,b_ub=[0.]*(n-1),bounds=bounds,method='highs')
        if not low.success or not high.success:
            raise YieldPredicateBridgeError('One-sided quote polytope extremum failed')
        p_low = constant+sum(v*x for v,x in zip(c,low.x))
        p_high = constant+sum(v*x for v,x in zip(c,high.x))
        if not -1e-9 <= p_low <= p_high+1e-9 <= 1+1e-9:
            raise YieldPredicateBridgeError('Unreconciled nested quote probabilities')
        out.append({'indicative_reference_Q_bin_mass_interval':
                    [max(0.,p_low),min(1.,p_high)],
                    'attaining_reference_Q_tail_vectors':{
                        'minimum_mass':[float(x) for x in low.x],
                        'maximum_mass':[float(x) for x in high.x]}})
    return out


def direct_yield_threshold_stress(*, positions, notional_usd, rate_sensitivity,
                                  kalshi_markets_raw, kalshi_expected_sha256,
                                  treasury_csv_raw, treasury_expected_sha256,
                                  baseline_date, retrieval_time_utc):
    """Reconcile original source bytes and map exact yield-band states to IEF.

    No API calls. The output says nothing about equity conditional event P&L.
    """
    _portfolio_weights(positions)
    if type(notional_usd) not in (int,float) or not math.isfinite(notional_usd) or not 0<notional_usd<=1e10:
        raise YieldPredicateBridgeError('Invalid original-portfolio notional')
    if isinstance(baseline_date, str):
        try:
            asof=date.fromisoformat(baseline_date)
        except ValueError as exc:
            raise YieldPredicateBridgeError('Invalid official baseline date') from exc
    else:
        raise YieldPredicateBridgeError('Official baseline must identify a dated source')
    try:
        captured=datetime.fromisoformat(retrieval_time_utc.replace('Z','+00:00'))
    except (TypeError,ValueError,AttributeError) as exc:
        raise YieldPredicateBridgeError('Source retrieval timestamp is unverified') from exc
    if (captured.tzinfo is None or captured.utcoffset() != timezone.utc.utcoffset(captured)
            or not asof < captured.date() < SOURCE_DATE):
        raise YieldPredicateBridgeError('Incorrect historical baseline/source/settlement time order')
    kalshi=_source_sha(kalshi_markets_raw,kalshi_expected_sha256,'Kalshi catalogue')
    treasury=_source_sha(treasury_csv_raw,treasury_expected_sha256,'US Treasury CSV')
    baseline=_treasury_yield(treasury,asof)
    markets=_parse_markets(kalshi)
    bins=_coherent_quote_bins(markets)
    per_bp,covered,uncovered=_bond_per_bp(rate_sensitivity,notional_usd)
    result=[]
    for i, b in enumerate(bins):
        lower=STRIKES[i-1] if i else None
        upper=STRIKES[i] if i<len(STRIKES) else None
        output={
            'state':('LOWER_OPEN_TAIL' if lower is None else
                     'UPPER_OPEN_TAIL' if upper is None else 'BETWEEN_EXACT_STRIKES'),
            'source_treasury_10y_yield_interval_pct':
                {'lower_exclusive':float(lower) if lower is not None else None,
                 'upper_inclusive':float(upper) if upper is not None else None},
            'reference_Q_cumulative_quotes_are_not_physical_P':True,
            **b,
        }
        if lower is not None and upper is not None:
            low_bps=100.*float(lower-baseline)
            high_bps=100.*float(upper-baseline)
            if low_bps >= -100.-1e-9 and high_bps <= 100.+1e-9:
                # per_bp < 0 reverses the range: closed low, open high.
                low_pnl=per_bp*high_bps
                high_pnl=per_bp*low_bps
                if low_pnl > high_pnl:
                    raise YieldPredicateBridgeError('Inverted source bond stress')
                output['covered_ief_linear_parallel_stress_usd_interval'] = [low_pnl,high_pnl]
                output['covered_stress_interval_endpoints']=['CLOSED_MINIMUM','OPEN_SUPREMUM']
                output['admission']='FINITE_SOURCE_INTERVAL_WITHIN_DECLARED_LINEAR_DOMAIN'
            else:
                output['covered_ief_linear_parallel_stress_usd_interval']=None
                output['admission']='YIELD_INTERVAL_OUTSIDE_LOCAL_PARALLEL_STRESS_DOMAIN'
        else:
            output['covered_ief_linear_parallel_stress_usd_interval']=None
            output['admission']='UNBOUNDED_TAIL_OUTSIDE_LOCAL_PARALLEL_STRESS_DOMAIN'
        result.append(output)
    return {
        'schema':SCHEMA,
        'status':'SOURCE_EXACT_YIELD_BAND_TO_COVERED_BOND_STRESS_NOT_EVENT_PNL',
        'event_id':EVENT,'settlement_date':SOURCE_DATE.isoformat(),
        'treasury_baseline_asof':asof.isoformat(),
        'treasury_baseline_yield_10y_pct':float(baseline),
        'source_retrieved_at_utc':captured.isoformat(),
        'original_source_hashes':{'kalshi_markets':sha256(kalshi).hexdigest(),
                                  'treasury_official_csv':sha256(treasury).hexdigest()},
        'exact_original_market_threshold_count':len(markets),
        'original_market_thresholds':markets,
        'strictly_nested_quotes_feasible':True,
        'reference_quote_model':'INDICATIVE_NESTED_TAIL_BID_ASK_POLYTOPE_NOT_JOINT_EXECUTABLE_MARKET',
        'source_defined_yield_bands':result,
        'source_price_implied_covered_ief_usd_per_parallel_bp':per_bp,
        'ief_covered_fund_nav_weight':covered,
        'ief_fund_nav_without_verified_duration_weight':uncovered,
        'first_order_yield_scenario_bps_domain':[-100.,100.],
        'source_contract_Q_is_equity_pricing_measure':False,
        'physical_probability_P':None,
        'event_yes_no_full_portfolio_pnl':None,
        'already_priced_in_fraction':None,
        'future_ief_return_forecast':None,
        'release_allowed':False,
        'critical_assumptions':['A source par Treasury 10Y move is mapped to covered IEF bonds as a PARALLEL yield-curve shift.',
                                'The fixed 2026 holding-level first-order sensitivity is used unchanged to Dec 31; carry, convexity and curve twists excluded.',
                                'Only the covered IEF bond sleeve is priced; all stocks and unresolved IEF NAV are excluded.',
                                'Market quotes are API-observed bid/ask values with unknown per-order freshness, not validated contemporaneous executable trades.'],
    }
