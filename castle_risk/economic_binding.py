"""Separate a family sensitivity from a declared exact-event economic scenario.

This layer certifies only the presence and identity of explicit assumptions. It
does not turn a branch description, an association, or a market quote into a
causal estimate. The mathematical engines remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


BOUND = 'EXPLICIT_EXACT_EVENT_SCENARIO_ASSUMPTION'
DERIVED_ORIGINS = {
    'SOURCE_DEFINED_BINARY_REFINEMENT_NOT_CLAUDE',
    'SOURCE_DEFINED_CONTRACT_FALLBACK_NOT_CLAUDE',
}


def review_economic_binding(event, scenarios):
    """Check an event-specific binding; text fields are declared assumptions.

Legacy exact-target callers may still inspect their numeric family sensitivity
through ``impact``. They are explicitly unbound, excluded from v2 acceptance,
and presented separately by the UI. Newly derived questions never inherit it.
    """
    event_id = event['event_id']
    specific = scenarios.get(event_id)
    family = scenarios.get(event.get('family'), {})
    settings = specific if isinstance(specific, Mapping) else family
    if not isinstance(settings, Mapping):
        raise ValueError('Scenario settings must be mappings.')
    source_defined = event.get('candidate_origin') in DERIVED_ORIGINS
    exact = event.get('definition_origin') == 'CONTRACT_IDENTITY'
    binding = specific.get('binding') if isinstance(specific, Mapping) else None
    failures = []
    if isinstance(binding, Mapping):
        if binding.get('event_id') != event_id:
            failures.append('BINDING_EVENT_ID_MISMATCH')
        if not exact or binding.get('contract_id') != event.get('contract_id'):
            failures.append('BINDING_EXACT_SETTLEMENT_IDENTITY_REQUIRED')
        required = ('YES', 'NO', 'no_outcome_mixture', 'price_basis', 'event_window', 'assumption_basis')
        failures.extend('MISSING_' + key.upper() for key in required
                        if not isinstance(binding.get(key), str) or not binding[key].strip())
        if binding.get('return_unit') != 'fractional_return':
            failures.append('RETURN_UNIT_MUST_BE_FRACTIONAL_RETURN')
        if binding.get('includes_source_exceptions') is not True:
            failures.append('SOURCE_CANCELLATION_AND_OTHER_EXCEPTIONS_NOT_ACKNOWLEDGED')
        identities = [card for card in event.get('contracts', [])
                      if card.get('contract_id') == event.get('contract_id')
                      and card.get('event_relationship') == 'EQUIVALENT'
                      and card.get('event_relationship_status') == 'CERTIFIED'
                      and card.get('source_family_gate', {}).get('eligible')]
        if len(identities) != 1:
            failures.append('EXACT_SOURCE_IDENTITY_NOT_INDEPENDENTLY_CERTIFIED')
        status = BOUND if not failures else 'INVALID_EXPLICIT_ECONOMIC_BINDING'
    elif source_defined:
        status = 'SOURCE_QUESTION_REQUIRES_EVENT_SPECIFIC_SCENARIO'
        failures.append('NO_EVENT_SPECIFIC_YES_NO_ASSUMPTION')
    elif exact:
        status = 'LEGACY_EXACT_TARGET_FAMILY_SENSITIVITY_UNBOUND'
        failures.append('FAMILY_SETTINGS_DO_NOT_IDENTIFY_EXACT_EVENT_BRANCH_RETURNS')
    else:
        status = 'FAMILY_STRESS_ONLY_EVENT_BRANCHES_UNDEFINED'
        failures.append('EXACT_BINARY_ECONOMIC_BRANCHES_NOT_IDENTIFIED')
    return {
        'status': status, 'explicit_binding_valid': status == BOUND,
        'settings': dict(settings), 'binding': dict(binding) if isinstance(binding, Mapping) else None,
        'reasons': failures,
        'legacy_numeric_payload_retained': exact and not source_defined and binding is None,
        'causal_effect_identified': False, 'physical_probability_P': None,
        'interpretation': 'A valid binding records explicit scenario assumptions for one settlement question; it does not validate those assumptions or estimate the conditional NO mixture.',
    }


def complete_bound_case(event):
    """Strict v2 gate: quote, exact declared branches, finite complete economics."""
    if event.get('economic_binding', {}).get('status') != BOUND:
        return False
    q = event.get('probability', {}).get('pricing_measure_Q_interval')
    impact = event.get('impact', {})
    priced = impact.get('priced_in', {})
    if not isinstance(q, (list, tuple)) or len(q) != 2:
        return False
    values = [*q, impact.get('delta_pnl_usd'), priced.get('yes_pnl_usd'), priced.get('no_pnl_usd')]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        return False
    if not 0 <= q[0] <= q[1] <= 1:
        return False
    bounds = impact.get('delta_pnl_interval_usd')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in bounds):
        return False
    if not bounds[0] <= impact['delta_pnl_usd'] <= bounds[1]:
        return False
    if (impact.get('scope') != 'FULL_ORIGINAL_PORTFOLIO' or impact.get('missing_symbols') != []
            or impact.get('price_domain_status') != 'NO_ASSESSED_RANGE_VIOLATION'
            or priced.get('status') != 'CONDITIONAL_PRICING_WEIGHT_SCENARIO'
            or priced.get('q_interval') != list(q)):
        return False
    def number(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    def interval(value):
        return isinstance(value, (list, tuple)) and len(value) == 2 and all(number(v) for v in value) and value[0] <= value[1]
    def same(a, b):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
    q_reference = priced.get('q_reference')
    if not number(q_reference) or not q[0] <= q_reference <= q[1]:
        return False
    delta = impact['delta_pnl_usd']
    if not same(priced['yes_pnl_usd'], (1-q_reference)*delta) or not same(priced['no_pnl_usd'], -q_reference*delta):
        return False
    for key, multiplier in (('yes_pnl_interval_usd', [1-q[1], 1-q[0]]),
                            ('no_pnl_interval_usd', [-q[1], -q[0]])):
        value = priced.get(key)
        products = [a*b for a in multiplier for b in bounds]
        if not interval(value) or not same(value[0], min(products)) or not same(value[1], max(products)):
            return False
    positions = impact.get('positions')
    notional = impact.get('notional_usd')
    if not isinstance(positions, list) or not positions or not number(notional) or notional <= 0:
        return False
    symbols = set()
    for row in positions:
        if not isinstance(row, Mapping) or not isinstance(row.get('symbol'), str) or row['symbol'] in symbols:
            return False
        symbols.add(row['symbol'])
        if any(not number(row.get(key)) for key in ('weight', 'return_contrast', 'delta_pnl_usd')):
            return False
        if not 0 <= row['weight'] <= 1 or not interval(row.get('return_contrast_interval')) or not interval(row.get('delta_pnl_interval_usd')):
            return False
        if not row['return_contrast_interval'][0] <= row['return_contrast'] <= row['return_contrast_interval'][1]:
            return False
        domain = row.get('price_domain', {})
        if domain.get('status') != 'WITHIN_NONNEGATIVE_PRICE_DOMAIN' or domain.get('range_within_nonnegative_price_domain') is not True:
            return False
        products = [a*b for a in [1-q[1], 1-q[0], -q[1], -q[0]] for b in row['return_contrast_interval']]
        if min(products) < -1-1e-12:
            return False
        amount = notional*row['weight']
        if not same(row['delta_pnl_usd'], amount*row['return_contrast']) or any(
                not same(row['delta_pnl_interval_usd'][i], amount*row['return_contrast_interval'][i]) for i in (0, 1)):
            return False
    return (same(math.fsum(row['weight'] for row in positions), 1.)
            and same(math.fsum(row['delta_pnl_usd'] for row in positions), delta)
            and all(same(math.fsum(row['delta_pnl_interval_usd'][i] for row in positions), bounds[i]) for i in (0, 1)))
