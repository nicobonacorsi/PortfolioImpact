"""Explicit YES/NO original-position return assumptions, not inferred event effects.

This module is intentionally source- and network-agnostic. The caller may
associate the output with a source-reviewed contract YES question, but neither
that contract's bid/ask nor a generic event topic determines conditional equity
returns. Absent original-position assumptions are unknown, never zero.
"""
from __future__ import annotations

import math
import json
from hashlib import sha256
from typing import Any, Mapping

from .impact import analyze_scenario
from .portfolio import parse_portfolio, _symbol


class OperatorScenarioError(ValueError):
    """Invalid operator assumption or unsupported scenario scope."""


def _return_horizon(value: str | None) -> str | None:
    """Retain a literal operator declaration, never infer a settlement horizon."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 500:
        raise OperatorScenarioError('Common return horizon must be text of at most 500 characters')
    return value.strip() or None


def _returns_in_percent(values: Mapping[str, float], *, known_symbols: set[str]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise OperatorScenarioError('Expected a symbol-to-assumed-percent-return mapping')
    converted: dict[str, float] = {}
    for original_symbol, value in values.items():
        symbol = _symbol(original_symbol)
        if symbol in converted:
            raise OperatorScenarioError('Duplicate symbol after normalization')
        if symbol not in known_symbols:
            raise OperatorScenarioError('Scenarios may stress original portfolio positions only')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise OperatorScenarioError('Scenario return must be a finite numerical percentage')
        if value < -100:
            raise OperatorScenarioError('Original unlevered long-only arithmetic return cannot be below -100%')
        converted[symbol] = float(value) / 100.0
    return converted


def evaluate_explicit_yes_no_scenarios(
    *,
    portfolio_rows: list[dict[str, Any]],
    notional: float,
    source_contract_id: str,
    source_yes_question: str,
    assumed_yes_returns_pct: Mapping[str, float],
    assumed_no_returns_pct: Mapping[str, float],
    common_return_horizon: str | None = None,
) -> dict[str, Any]:
    """Calculate original-position scenario P&L under *operator-specified* inputs.

    The full YES-minus-NO difference is available only if every nonzero
    original position has an explicit return in both states. A shared set of
    known legs does give a descriptive partial contrast but is not the total
    portfolio effect. An absent return horizon leaves arithmetic available but
    does not define a comparable financial scenario. A supplied horizon is a
    literal operator declaration, not a source-verified calendar interval.
    This is not a forecast or event-conditional estimate.
    """
    if (not isinstance(source_contract_id, str) or not source_contract_id.strip()
            or len(source_contract_id) > 128):
        raise OperatorScenarioError('An exact source-defined contract ID is required')
    if (not isinstance(source_yes_question, str) or not source_yes_question.strip()
            or len(source_yes_question) > 1000):
        raise OperatorScenarioError('A source-defined YES question is required')
    horizon = _return_horizon(common_return_horizon)
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    known_symbols = {p.symbol for p in portfolio.positions}
    portfolio_signature = sha256(json.dumps(
        sorted((p.symbol, p.asset_type, p.weight) for p in portfolio.positions),
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    yes_returns = _returns_in_percent(assumed_yes_returns_pct, known_symbols=known_symbols)
    no_returns = _returns_in_percent(assumed_no_returns_pct, known_symbols=known_symbols)
    assumption_signature = sha256(json.dumps({
        'portfolio_position_signature_sha256': portfolio_signature,
        'notional_usd': portfolio.notional,
        'source_contract_id': source_contract_id.strip(),
        'source_yes_question': source_yes_question.strip(),
        'assumed_yes_returns': yes_returns,
        'assumed_no_returns': no_returns,
        'operator_declared_common_return_horizon': horizon,
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    assumption_note = ('Input is an operator-specified WHAT-IF assumption for exact contract YES/NO, '
                       'not an observed historical response, model forecast, or inferred market effect.')
    yes = analyze_scenario(portfolio, yes_returns,
        name='Operator-assumed YES scenario (not an event-effect forecast)',
        assumption_note=assumption_note).to_dict()
    no = analyze_scenario(portfolio, no_returns,
        name='Operator-assumed NO scenario (not an event-effect forecast)',
        assumption_note=assumption_note).to_dict()
    full = yes['total_pnl'] is not None and no['total_pnl'] is not None
    shared = [p for p in portfolio.positions
              if p.weight > 0 and p.symbol in yes_returns and p.symbol in no_returns]
    shared_weight = math.fsum(p.weight for p in shared)
    shared_delta = portfolio.notional * math.fsum(
        p.weight * (yes_returns[p.symbol] - no_returns[p.symbol]) for p in shared)
    if full and not math.isclose(shared_weight, 1.0, abs_tol=1e-10, rel_tol=0):
        raise OperatorScenarioError('Original-position scenario reconciliation failed')
    exact_delta = yes['total_pnl'] - no['total_pnl'] if full else None
    if full and not math.isclose(exact_delta, shared_delta, abs_tol=1e-7, rel_tol=0):
        raise OperatorScenarioError('Conditional scenario contrast arithmetic failed')
    return {
        'status': ('COMPLETE_OPERATOR_ASSUMED_YES_NO_RETURN_SCENARIOS'
                   if full else 'PARTIAL_OPERATOR_ASSUMED_SCENARIOS_UNKNOWN_REMAINDER'),
        'source_contract_id': source_contract_id.strip(),
        'source_yes_question': source_yes_question.strip(),
        'scenario_basis': 'EXPLICIT_OPERATOR_INPUT_ONLY_NOT_IDENTIFIED_EVENT_EFFECT',
        'operator_declared_common_return_horizon': horizon,
        'return_horizon_status': ('OPERATOR_DECLARED_NOT_SOURCE_INFERRED' if horizon
                                  else 'UNSPECIFIED_ARITHMETIC_ONLY'),
        'scenario_assumptions_sha256': assumption_signature,
        'original_portfolio_notional_usd': portfolio.notional,
        'original_portfolio_position_signature_sha256': portfolio_signature,
        'original_positions_only_no_etf_double_count': True,
        'yes': yes,
        'no': no,
        'operator_assumed_yes_minus_no_total_pnl_usd': exact_delta,
        'operator_assumed_yes_minus_no_shared_known_legs_pnl_usd': shared_delta,
        'common_explicitly_assumed_original_position_weight': shared_weight,
        'unresolved_total_portfolio_contrast_weight': 1.0 - shared_weight,
        'identified_real_world_conditional_pnl': None,
        'physical_probability_P': None,
        'equity_market_priced_in_q': None,
        'kalshi_Q_used_in_valuation': False,
        'not_a_forecast_or_certified_future_loss': True,
        'assumptions': [
            'The operator specified both state-conditional original-position returns directly.',
            ('Each state applies to the same fixed initial portfolio and operator-declared common return horizon.'
             if horizon else 'No common return horizon is declared; the amounts are arithmetic only and cannot be ranked across financial scenarios.'),
            'Unspecified returns are unknown rather than zero; partial known-leg contrasts are not portfolio totals.',
            'Neither the event probability, liquidity nor the equity price anticipatory effect is inferred.',
            'Taxes, fees, rebalance, leverage, intra-period path effects and additional macro shocks are excluded.',
        ],
    }


def compare_operator_scenario_downside(scenarios: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank only *complete* operator-defined worst-state downside assumptions.

    The rank is not expected loss, probability-weighted risk or a forecast. A
    scenario missing a YES or NO return for ANY original position is listed as
    unranked. All compared scenarios must share the same original notional and
    portfolio. Complete assumptions must also declare the same literal return
    horizon; unspecified or incompatible horizons remain unranked. Matching
    text is an operator assertion, not independent temporal verification.
    """
    if not isinstance(scenarios, list) or len(scenarios) > 32:
        raise OperatorScenarioError('Bounded scenario list required')
    seen: set[str] = set()
    notional: float | None = None
    portfolio_signature: str | None = None
    ranked: list[dict[str, Any]] = []
    unranked: list[dict[str, str]] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise OperatorScenarioError('Invalid comparison record')
        contract = scenario.get('source_contract_id')
        if not isinstance(contract, str) or not contract or contract in seen:
            raise OperatorScenarioError('Invalid or repeated source-contract identity')
        seen.add(contract)
        signature = scenario.get('original_portfolio_position_signature_sha256')
        if (not isinstance(signature,str) or len(signature) != 64
                or any(c not in '0123456789abcdef' for c in signature)):
            raise OperatorScenarioError('A reproducible original portfolio signature is required')
        if portfolio_signature is None:
            portfolio_signature = signature
        elif signature != portfolio_signature:
            raise OperatorScenarioError('Comparative scenarios must use the same portfolio positions and weights')
        value = scenario.get('original_portfolio_notional_usd')
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise OperatorScenarioError('Invalid original portfolio notional')
        if notional is None:
            notional = float(value)
        elif not math.isclose(notional, float(value), rel_tol=0, abs_tol=1e-8):
            raise OperatorScenarioError('Comparative scenarios must use the same portfolio notional')
        if (scenario.get('kalshi_Q_used_in_valuation') is not False
                or scenario.get('identified_real_world_conditional_pnl') is not None):
            raise OperatorScenarioError('Only explicit unpriced hypothetical scenarios can be ranked')
        horizon = _return_horizon(scenario.get('operator_declared_common_return_horizon'))
        expected_horizon_status = ('OPERATOR_DECLARED_NOT_SOURCE_INFERRED' if horizon
                                   else 'UNSPECIFIED_ARITHMETIC_ONLY')
        recorded_horizon_status = scenario.get('return_horizon_status')
        if recorded_horizon_status is not None and recorded_horizon_status != expected_horizon_status:
            raise OperatorScenarioError('Return horizon and its declared scope did not reconcile')
        states = [scenario.get('yes'), scenario.get('no')]
        if not all(isinstance(state, Mapping) for state in states):
            raise OperatorScenarioError('Both original-position scenario states are required')
        if any(state.get('total_pnl') is None for state in states):
            unranked.append({'source_contract_id': contract,
                'reason': 'INCOMPLETE_OPERATOR_RETURN_INPUTS_NOT_ASSUMED_ZERO'})
            continue
        totals=[]
        for state in states:
            pnl=state['total_pnl']
            if isinstance(pnl, bool) or not isinstance(pnl, (float,int)) or not math.isfinite(pnl):
                raise OperatorScenarioError('Invalid operator scenario P&L')
            contributions=state.get('contributions')
            if not isinstance(contributions,(list,tuple)) or not contributions or len(contributions)>256:
                raise OperatorScenarioError('Original-position contribution evidence required')
            reconstructed_positions = []
            for c in contributions:
                if not isinstance(c, Mapping):
                    raise OperatorScenarioError('Malformed position contribution')
                weight = c.get('weight')
                leg_pnl = c.get('pnl')
                if (isinstance(weight, bool) or not isinstance(weight, (float, int))
                        or not math.isfinite(weight) or weight < 0):
                    raise OperatorScenarioError('Invalid original-position weight')
                if (isinstance(leg_pnl, bool) or not isinstance(leg_pnl, (float, int))
                        or not math.isfinite(leg_pnl)):
                    raise OperatorScenarioError('Incomplete or invalid position P&L')
                assumed_return = c.get('assumed_return')
                if assumed_return is None and weight > 0:
                    raise OperatorScenarioError('Missing positive-weight return in complete state')
                if assumed_return is not None and (
                        isinstance(assumed_return, bool)
                        or not isinstance(assumed_return, (float, int))
                        or not math.isfinite(assumed_return) or assumed_return < -1):
                    raise OperatorScenarioError('Invalid original-position return')
                expected_leg = float(value) * weight * (assumed_return or 0.0)
                if not math.isclose(leg_pnl, expected_leg, abs_tol=1e-6, rel_tol=1e-12):
                    raise OperatorScenarioError('Position return, weight, notional and P&L did not reconcile')
                reconstructed_positions.append({'symbol': c.get('symbol'),
                    'asset_type': c.get('asset_type'), 'weight': weight})
            try:
                reconstructed = parse_portfolio(reconstructed_positions, notional=float(value))
            except (ValueError, TypeError, KeyError) as exc:
                raise OperatorScenarioError('Original-position contribution identities did not reconcile') from exc
            observed_signature = sha256(json.dumps(
                sorted((p.symbol, p.asset_type, p.weight) for p in reconstructed.positions),
                separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
            if observed_signature != signature:
                raise OperatorScenarioError('Contribution positions did not reconcile with the portfolio signature')
            observed = math.fsum(c['pnl'] for c in contributions)
            if not math.isclose(observed, pnl, abs_tol=1e-6, rel_tol=1e-12):
                raise OperatorScenarioError('State P&L did not reconcile with original-position contributions')
            totals.append(float(pnl))
        worst=min(totals)
        if horizon is None:
            unranked.append({'source_contract_id': contract,
                'reason': 'RETURN_HORIZON_UNSPECIFIED_ARITHMETIC_ONLY'})
            continue
        ranked.append({
            'source_contract_id':contract,
            'operator_declared_common_return_horizon': horizon,
            'worst_assumed_state': 'YES' if totals[0] <= totals[1] else 'NO',
            'operator_yes_pnl_usd':totals[0],
            'operator_no_pnl_usd':totals[1],
            'operator_worst_state_pnl_usd':worst,
            'operator_potential_downside_usd':max(0.,-worst),
            'physical_probability_P':None,
            'expected_loss_usd':None,
            'identified_event_conditional_pnl_usd':None,
        })
    declared_horizons = {entry['operator_declared_common_return_horizon'] for entry in ranked}
    if len(declared_horizons) > 1:
        unranked.extend({'source_contract_id': entry['source_contract_id'],
            'reason': 'INCOMPATIBLE_OPERATOR_DECLARED_RETURN_HORIZONS',
            'operator_declared_common_return_horizon': entry['operator_declared_common_return_horizon']}
            for entry in ranked)
        ranked = []
    ranked.sort(key=lambda x: (-x['operator_potential_downside_usd'], x['source_contract_id']))
    for i, entry in enumerate(ranked, 1):
        entry['operator_scenario_downside_rank_not_economic_event_rank'] = i
    return {
        'status':('PARTIAL_OPERATOR_SCENARIO_DOWNSIDE_COMPARISON_ONLY' if unranked
                  else 'COMPLETE_OPERATOR_SCENARIO_DOWNSIDE_COMPARISON_ONLY'),
        'ranking_basis':'WORST_OPERATOR_ASSUMED_YES_NO_PNL_NOT_PROBABILITY_WEIGHTED_NOT_MARKET_IMPACT',
        'same_original_portfolio_notional_usd':notional,
        'same_original_portfolio_position_signature_sha256':portfolio_signature,
        'same_operator_declared_common_return_horizon': (
            next(iter(declared_horizons)) if len(declared_horizons) == 1 else None),
        'horizon_comparison_basis': 'EXACT_OPERATOR_DECLARATION_NOT_VERIFIED_CALENDAR_EQUIVALENCE',
        'ranked_complete_assumptions':ranked,
        'unranked_assumptions':unranked,
        # Retained for existing consumers; reasons now also cover horizon scope.
        'unranked_incomplete_assumptions':unranked,
        'model_identified_expected_loss':None,
        'kalshi_Q_used':False,
        'physical_probability_P':None,
        'no_provider_or_model_requests':True,
    }
