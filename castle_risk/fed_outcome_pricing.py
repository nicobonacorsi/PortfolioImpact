"""Finite-outcome Kalshi Fed price bounds under explicit common-Q assumptions.

Distinct YES outcomes are independently verified as mutually exclusive by the
version-pinned full source secondary rules. Their quotes are co-observed in
ONE locally SHA-receipted HTTP response. A slack OTHER atom preserves all
unretrieved and unlisted outcomes. SciPy LP proves bound attainability under
the displayed marginal bid/ask constraints, with no assertion of liquidity,
exchange quote-update timing, physical P or contract-conditioned stock P&L.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog

from .adapters import Market, normalize_quote
from .agentic_bridge import _receipt, AgenticEvidenceError
from .source_native_outcomes import _fed_exclusive_pair


class FedOutcomePricingError(ValueError):
    """The source, quote or common pricing assumptions are not admissible."""


def evaluate_fed_outcome_quotes(
    markets: Sequence[Market], *, as_of: str | datetime | None = None,
    max_age_seconds: int = 300,
) -> dict[str, Any]:
    """Bound observed Fed outcome quote masses, including unlisted residual.

    This never assumes the returned catalog is exhaustive. One observed record
    per selected outcome, all locally witnessed within ONE provider source read,
    source-declared mutual exclusivity, and common $1 unit payout are required.
    """
    if not isinstance(markets, (list, tuple)) or not 2 <= len(markets) <= 12:
        raise FedOutcomePricingError('Between two and twelve distinct Fed outcome contracts required')
    if len({m.ticker for m in markets}) != len(markets):
        raise FedOutcomePricingError('Duplicate outcome identity')
    if any(not _fed_exclusive_pair(a, b) for a, b in combinations(markets, 2)):
        raise FedOutcomePricingError('Full original source rules do not certify pairwise exclusivity')
    identities = []
    for m in markets:
        try:
            receipt = _receipt(m)
        except (AgenticEvidenceError, ValueError, OSError) as exc:
            raise FedOutcomePricingError('Original source receipt integrity failure') from exc
        identities.append((receipt['source_response_sha256'], receipt['request_url'],
                           receipt['retrieved_at_utc'], m.event_ticker, m.close_time,
                           m.expiration_time))
    if any(x != identities[0] for x in identities[1:]):
        raise FedOutcomePricingError('Different source response, event, timing or retrieval: no common quote snapshot')
    source_time = datetime.fromisoformat(identities[0][2].replace('Z', '+00:00'))
    current_time = as_of or datetime.now(timezone.utc)
    if isinstance(current_time, str):
        current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
    if current_time.utcoffset() is None:
        raise FedOutcomePricingError('Observation time requires timezone')
    current_time = current_time.astimezone(timezone.utc)
    # A historical source-time replay and a current snapshot are different
    # claims. Neither substitutes for the exchange quote-update timestamp.
    source_view = []
    for m in markets:
        quote = normalize_quote(m, now=source_time, max_age_seconds=max_age_seconds)
        if not quote.usable or quote.lower is None or quote.upper is None:
            raise FedOutcomePricingError('Quote cannot be supported from co-observed source: '+m.ticker)
        source_view.append({'ticker': m.ticker, 'lower': quote.lower, 'upper': quote.upper})
    n = len(source_view)
    bounds = [(x['lower'], x['upper']) for x in source_view] + [(0.0, 1.0)]
    # The slack state represents UNLISTED outcomes, not a specific contract.
    optimizer = dict(A_eq=[[1.0] * (n + 1)], b_eq=[1.0], bounds=bounds, method='highs')
    labels = [x['ticker'] for x in source_view] + ['OTHER_UNLISTED_OUTCOMES']
    results = []
    for i, label in enumerate(labels):
        objective = np.eye(n+1)[i].tolist()
        lo = linprog(objective, **optimizer)
        hi = linprog([-c for c in objective], **optimizer)
        if not (lo.success and hi.success):
            raise FedOutcomePricingError('Infeasible single-Q outcome simplex for source bid/ask intervals')
        for witness in (lo.x, hi.x):
            if (abs(math.fsum(witness)-1.0) > 1e-8 or
                    any(p < -1e-8 for p in witness) or
                    any(not lower-1e-8 <= p <= upper+1e-8
                        for p, (lower,upper) in zip(witness, bounds))):
                raise FedOutcomePricingError('LP endpoint witness invalid')
        results.append({
            'source_outcome_id': label,
            'pricing_measure_Q_interval': [max(0.0, float(lo.fun)), min(1.0, float(-hi.fun))],
            'lower_attaining_joint_witness': [float(x) for x in lo.x],
            'upper_attaining_joint_witness': [float(x) for x in hi.x],
        })
    ages = [(current_time - datetime.fromisoformat(m.quote_timestamp.replace('Z','+00:00'))).total_seconds()
            for m in markets]
    expiry = datetime.fromisoformat(markets[0].close_time.replace('Z','+00:00'))
    now_usable = all(0 <= age <= max_age_seconds for age in ages) and current_time < expiry
    return {
        'status': ('FRESH_CONDITIONAL_SOURCE_NATIVE_FED_OUTCOME_Q_BOUNDS' if now_usable
                   else 'HISTORICAL_CONDITIONAL_SOURCE_NATIVE_FED_OUTCOME_Q_REPLAY'),
        'scope': 'SOURCE_DECLARED_MUTUALLY_EXCLUSIVE_CONTRACT_YES_OUTCOMES_WITH_UNLISTED_RESIDUAL',
        'as_of_source_receipt_utc': identities[0][2],
        'original_local_source_sha256': identities[0][0],
        'source_contract_quote_constraints': source_view,
        'source_outcome_bounds_with_lp_witnesses': results,
        'returned_catalog_is_exhaustive': False,
        'exchange_quote_update_time_verified': False,
        'required_model_assumptions': [
            'Every considered binary YES contract pays one common unit under one consistent settlement and numeraire.',
            'The bid/ask quotes are interpreted as compatible marginal pricing reference intervals under one common Q; fees, funding and liquidity execution are not modeled.',
            'Unlisted outcomes are retained as OTHER; the observed bounded catalog is not assumed exhaustive.',
            'The original local HTTP retrieval is a co-observation receipt, not an exchange-authenticated quote-update timestamp.',
        ],
        'physical_probability_P': None,
        'equity_priced_in_q': None,
        'contract_conditioned_portfolio_PnL': None,
    }
