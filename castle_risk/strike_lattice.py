"""Certified *conditional* settlement strike lattice from co-observed Kalshi bytes.

A narrow, fail-closed grammar recognizes otherwise-identical CPI strike YES
contracts. Their payoffs are nested; under the explicitly assumed common
unit-payout pricing measure Q, quoted marginal intervals constrain a simplex
of disjoint settlement-value bins. SciPy HiGHS computes sharp interval limits
and feasible witness distributions. This is NOT BLS first-release event matching,
physical probability P, equity priced-in q, or identified asset returns.

No live provider calls. Admission requires an independently verified original
local raw-response receipt for EVERY normalized contract and the SAME source
response body for simultaneous sampling. A source receipt is not an exchange
quote-update timestamp and cannot prove contemporaneous executable liquidity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import math
import re
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog

from .adapters import Market, normalize_quote
from .agentic_bridge import AgenticEvidenceError, _receipt

CPI_STRIKE_GRAMMAR = re.compile(
    r'^If the Consumer Price Index \(CPI\) increases by more than '
    r'(?P<strike>-?\d+(?:\.\d+)?)% \(single-decimal\) in '
    r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) '
    r'(?P<year>20\d{2}), then the market resolves to Yes\.$'
)
# Kalshi's December 2026 CPI event words the same one-decimal threshold differently; the
# settlement sentence of its secondary terms is unchanged. One event never mixes the two wordings.
CPI_STRIKE_GRAMMAR_AS_REPORTED = re.compile(
    r'^If the Consumer Price Index \(CPI\) increases by more than '
    r'(?P<strike>-?\d+(?:\.\d+)?)% in '
    r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) '
    r'(?P<year>20\d{2}), as reported to one decimal place, then the market resolves to Yes\.$'
)
_TICKER_GRAMMAR = re.compile(r'^KXCPI-\d{2}[A-Z]{3}-T(?P<strike>-?\d+(?:\.\d+)?)$')


class StrikeLatticeError(ValueError):
    """The provider bytes cannot support a common settlement partition."""


def _assert_same_float(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-9


def _parse_source(market: Market) -> tuple[float, dict[str, Any]]:
    if not isinstance(market, Market):
        raise StrikeLatticeError('Every input must be an original normalized provider Market object.')
    try:
        receipt = _receipt(market)
    except AgenticEvidenceError as exc:
        raise StrikeLatticeError('A complete locally verified source receipt is required: ' + str(exc)) from None
    raw = market.metadata.get('raw', {})
    if (not isinstance(raw, dict) or raw.get('status') not in ('active', 'open')
            or raw.get('market_type') != 'binary'
            or market.series_ticker != 'KXCPI'
            or not isinstance(raw.get('rules_secondary'), str)
            or not raw['rules_secondary'].strip()):
        raise StrikeLatticeError('Contract not verified as open binary CPI with full secondary terms.')
    match = CPI_STRIKE_GRAMMAR.fullmatch(market.rules_primary)
    wording = 'SINGLE_DECIMAL_PARENTHETICAL'
    if not match:
        match = CPI_STRIKE_GRAMMAR_AS_REPORTED.fullmatch(market.rules_primary)
        wording = 'AS_REPORTED_TO_ONE_DECIMAL_PLACE'
    ticker_match = _TICKER_GRAMMAR.fullmatch(market.ticker)
    if not match or not ticker_match:
        raise StrikeLatticeError('Primary rules or ticker are outside the narrow identical CPI strike grammar.')
    strike = float(match['strike'])
    if not _assert_same_float(strike, float(ticker_match['strike'])):
        raise StrikeLatticeError('Ticker strike contradicts the original primary settlement rule.')
    # No optional exceptions are stripped or silently equated; the entire
    # secondary text and key venue settlement fields must agree byte-for-byte.
    identity = {
        'event_ticker': market.event_ticker,
        'primary_wording': wording,
        'reference_month': match['month'],
        'reference_year': match['year'],
        'raw_secondary_sha256': sha256(market.rules_secondary.encode('utf-8')).hexdigest(),
        'close_time': market.close_time,
        'expiration_time': market.expiration_time,
        'expected_expiration_time': raw.get('expected_expiration_time'),
        'settlement_timer_seconds': raw.get('settlement_timer_seconds'),
        'notional_value_dollars': raw.get('notional_value_dollars'),
        'status': market.status,
        'market_type': raw['market_type'],
        'source_response_sha256': receipt['source_response_sha256'],
        'source_request_url': receipt['request_url'],
        'retrieved_at_utc': receipt['retrieved_at_utc'],
    }
    if not re.fullmatch(r'KXCPI-\d{2}[A-Z]{3}', market.event_ticker):
        raise StrikeLatticeError('An unexpected CPI event identifier was supplied.')
    if str(raw.get('notional_value_dollars', '')) not in ('1.0000', '1.00', '1.0', '1'):
        raise StrikeLatticeError('The contracts do not establish the same USD unit payout.')
    for field in ('close_time', 'expiration_time', 'expected_expiration_time'):
        if not isinstance(identity[field], str) or not identity[field]:
            raise StrikeLatticeError('Missing exact venue settlement timing field: ' + field)
    return strike, identity


def evaluate_strike_lattice(
    markets: Sequence[Market], *, observed_at: datetime | str | None = None,
    max_quote_receipt_age_seconds: int = 300,
) -> dict[str, Any]:
    """Return sharp LP bounds for each mutually exclusive interval of CPI settlement values.

    For sorted thresholds t[0] < ... < t[n-1], define bin masses p[0],...,p[n]
    on (-inf,t0], (t0,t1],...,(tn-1,inf). Each event E_i={X>t_i}
    has Q(E_i)=sum_{j>i} p[j]. Quoted intervals impose linear constraints.
    Endpoint witnesses prove attainable bounds under these stated constraints.
    """
    if not isinstance(markets, (tuple, list)) or not 2 <= len(markets) <= 12:
        raise StrikeLatticeError('Between two and twelve source-backed strikes are required.')
    if isinstance(observed_at, str):
        try:
            observed_at = datetime.fromisoformat(observed_at.replace('Z', '+00:00'))
        except ValueError:
            raise StrikeLatticeError('Observation timestamp could not be parsed.') from None
    if observed_at is not None and (observed_at.tzinfo is None or observed_at.utcoffset() is None):
        raise StrikeLatticeError('Observation timestamp must include timezone information.')
    current = observed_at or datetime.now(timezone.utc)
    specs = []
    same_scope = None
    seen_tickers = set()
    for market in markets:
        strike, scope = _parse_source(market)
        if market.ticker in seen_tickers:
            raise StrikeLatticeError('Duplicate contract tickers cannot form independent strike constraints.')
        seen_tickers.add(market.ticker)
        if same_scope is None:
            same_scope = scope
        elif scope != same_scope:
            raise StrikeLatticeError('Contracts differ in settlement terms, reference period, payout, retrieval or raw-response source.')
        quote = normalize_quote(market, now=current, max_age_seconds=max_quote_receipt_age_seconds)
        if not quote.usable:
            raise StrikeLatticeError('No admissible co-observed quote for ' + market.ticker + ': ' + quote.exclusion_reason)
        specs.append({'ticker': market.ticker, 'strike': strike,
                      'yes_quote_lower': quote.lower, 'yes_quote_upper': quote.upper,
                      'source_quote_receipt_utc': market.quote_timestamp,
                      'source_quote_exchange_timestamp_confirmed': False})
    specs.sort(key=lambda x: x['strike'])
    if any(specs[i]['strike'] >= specs[i + 1]['strike'] for i in range(len(specs) - 1)):
        raise StrikeLatticeError('Duplicate or unordered source strike values are not admissible.')
    n = len(specs)
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a: list[list[float]] = []
    for i, spec in enumerate(specs):
        row = [0.0] * (i + 1) + [1.0] * (n - i)
        a.append(row)
        a_ub.extend([row, [-v for v in row]])
        b_ub.extend([spec['yes_quote_upper'], -spec['yes_quote_lower']])
    kwargs = dict(A_ub=a_ub, b_ub=b_ub, A_eq=[[1.0] * (n + 1)], b_eq=[1.0],
                  bounds=[(0.0, 1.0)] * (n + 1), method='highs')
    rows = []
    names = ['settlement_value <= ' + format(specs[0]['strike'], '.12g') + '%']
    for j in range(1, n):
        names.append(format(specs[j - 1]['strike'], '.12g') + '% < settlement_value <= ' +
                     format(specs[j]['strike'], '.12g') + '%')
    names.append('settlement_value > ' + format(specs[-1]['strike'], '.12g') + '%')
    for j in range(n + 1):
        objective = np.eye(n + 1)[j].tolist()
        low = linprog(objective, **kwargs)
        high = linprog([-v for v in objective], **kwargs)
        if not (low.success and high.success):
            raise StrikeLatticeError('No common-Q settlement distribution satisfies all source quote intervals.')
        witness_low = [float(x) for x in low.x]
        witness_high = [float(x) for x in high.x]
        for witness in (witness_low, witness_high):
            if abs(math.fsum(witness) - 1.0) > 1e-8:
                raise StrikeLatticeError('A numerical LP witness failed simplex validation.')
            if any(x < -1e-8 for x in witness):
                raise StrikeLatticeError('A numerical LP witness has a negative atom.')
            for spec, incidence in zip(specs, a):
                probability = math.fsum(x * c for x, c in zip(witness, incidence))
                if not spec['yes_quote_lower'] - 1e-8 <= probability <= spec['yes_quote_upper'] + 1e-8:
                    raise StrikeLatticeError('A numerical LP witness contradicts a source quote constraint.')
        rows.append({'bin': names[j], 'pricing_measure_Q_interval': [max(0.0, float(low.fun)), min(1.0, -float(high.fun))],
                     'attaining_lower_witness_bin_masses': witness_low,
                     'attaining_upper_witness_bin_masses': witness_high})
    return {
        'status': 'CONDITIONAL_SAME_SOURCE_CPI_STRIKE_LP_WITNESSES',
        'scope': 'DISJOINT_EXACT_CONTRACT_SETTLEMENT_VALUE_BINS',
        'source': {'local_verified_response_sha256': same_scope['source_response_sha256'],
                   'source_request_url': same_scope['source_request_url'],
                   'retrieved_at_utc': same_scope['retrieved_at_utc'],
                   'venue_quote_update_time_verified': False,
                   'same_full_secondary_settlement_rules_sha256': same_scope['raw_secondary_sha256'],
                   'source_receipt_assurance': 'LOCAL_BYTES_NOT_EXCHANGE_SIGNATURE'},
        'event_ticker': same_scope['event_ticker'],
        'as_of_utc': current.isoformat(),
        'contract_quote_constraints': specs,
        'disjoint_event_bins': rows,
        'physical_probability_P': None,
        'equity_priced_in_q': None,
        'portfolio_event_return_coefficient': None,
        'assumptions': [
            'Every source contract resolves YES if the same single-decimal CPI settlement value strictly exceeds its own listed strike.',
            'All original primary rules have the same exact grammar apart from strike, and the full secondary terms and expiry fields are identical.',
            'All market records were observed within one locally hash-verified original public HTTP response body.',
            'Under a single coherent common unit-payout pricing measure Q (one numeraire/settlement and ignored transaction fees), overlapping marginal bid/ask intervals constrain a joint distribution.',
            'The LP bounds are mathematically sharp for those constraints; witness distributions are provided for both endpoints.',
            'Receipt age is checked but exchange-side quote-update time, depth at scale and contemporaneous execution are NOT independently verified.',
            'This is the market settlement statistic, not necessarily the initial BLS release as a separately defined economic question.',
            'No physical-event frequency, conditional stock return, causal impact or equity-market priced-in weight is inferred.',
        ],
    }
