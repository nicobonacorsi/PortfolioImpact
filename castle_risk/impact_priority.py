"""Castle Option 1: order displayable events by *conditional portfolio movement*.

Primary admitted model outcomes, independently verified autonomous paired
event scenarios, and complete explicitly assumed analyst scenarios can supply
navigation magnitudes. They remain distinct evidence tiers. No magnitude is
an event probability, an expected loss, or proof of a causal market reaction.
Unpriced hypotheses are not ranked by the capital a model happened to mention.
"""
from __future__ import annotations

from copy import deepcopy
import math


def _finite(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def _primary_magnitude(risk):
    admission = (risk.get('quality') or {}).get('model_admission') or {}
    if admission.get('primary_impact_allowed') is not True:
        return None
    outcomes = risk.get('outcomes')
    if not isinstance(outcomes, list) or not outcomes:
        return None
    values = [_finite(row.get('pnl')) if isinstance(row, dict) else None for row in outcomes]
    if any(value is None for value in values):
        return None
    return max(abs(value) for value in values)


def _analyst_magnitude(risk, positions, notional):
    proposal = risk.get('analyst_assumption') or {}
    if not isinstance(proposal, dict):
        return None
    result = proposal.get('result') or {}
    if not isinstance(result, dict):
        return None
    if (proposal.get('schema') != 'castle.analyst_assumption.v1'
            or proposal.get('status') != 'COMPLETE' or result.get('complete') is not True
            or proposal.get('unmodeled_symbols') or _finite(notional) is None or notional <= 0):
        return None
    shocks = proposal.get('position_return_gap_pct')
    if not isinstance(shocks, dict) or not isinstance(positions, list):
        return None
    seen, contributions, total_weight = set(), [], 0.0
    for position in positions:
        if not isinstance(position, dict):
            return None
        symbol = position.get('symbol')
        weight = _finite(position.get('weight'))
        gap = _finite(shocks.get(symbol))
        if (not isinstance(symbol, str) or not symbol or symbol in seen or
                weight is None or not 0 <= weight <= 1 or gap is None):
            return None
        seen.add(symbol)
        total_weight += weight
        contributions.append(notional * weight * gap / 100)
    if not seen or set(shocks) != seen or abs(total_weight - 1.) > 1e-8:
        return None
    if any(not math.isfinite(value) for value in contributions):
        return None
    try:
        gap_usd = math.fsum(contributions)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(gap_usd):
        return None
    yes, no = _finite(result.get('yes_pnl_usd')), _finite(result.get('no_pnl_usd'))
    reported_gap = _finite(result.get('known_event_gap_usd'))
    covered = _finite(result.get('covered_weight'))
    pricing = result.get('pricing') or {}
    if not isinstance(pricing, dict):
        return None
    q = _finite(pricing.get('equity_weight_q'))
    if any(value is None for value in (yes, no, reported_gap, covered, q)):
        return None
    tolerance = max(1e-6, 1e-8 * max(1., abs(gap_usd)))
    if (abs(covered - 1.) > 1e-8 or not 0 <= q <= 1 or
            abs(reported_gap - gap_usd) > tolerance or
            abs(yes - (1 - q) * gap_usd) > tolerance or
            abs(no + q * gap_usd) > tolerance):
        return None
    return max(abs(yes), abs(no))


def _autonomous_common_endpoint_magnitude(risk, positions, notional, observed_mark_context):
    """Only a reverified, exactly matched event may set the v2 navigation gap.

    The central paired YES-minus-NO gap (not YES-from-today) excludes the
    unrelated same-horizon NO drift. It is still an uncalibrated assumption.
    """
    row = risk.get('common_endpoint_impact')
    context = risk.get('event_probability_question')
    if not isinstance(row, dict) or not isinstance(context, dict) or not isinstance(observed_mark_context, dict):
        return None
    if (risk.get('id') != context.get('event_id') or
            risk.get('title') != context.get('target_condition') or
            risk.get('date') != context.get('deadline') or
            any(row.get(key) != context.get(key) for key in
                ('event_id', 'target_condition', 'deadline', 'as_of_utc')) or
            observed_mark_context.get('as_of_utc') != row.get('as_of_utc')):
        return None
    # Rechecking the entire saved model here prevents stale, forged, or
    # different-portfolio dollar fields from influencing the ranking.
    from .autonomous_common_endpoint_impact import _is_valid_saved_numeric
    try:
        valid = _is_valid_saved_numeric(
            {'positions': positions, 'notional': notional,
             'observed_mark_context': observed_mark_context}, row)
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    if not valid:
        return None
    gap = _finite(row.get('paired_event_gap_center_usd'))
    return abs(gap) if gap is not None else None


def conditional_movement_priority(risk, positions, notional, *, observed_mark_context=None):
    """A complete portfolio movement, or an explicit unranked state."""
    if not isinstance(risk, dict):
        return None
    base = {
        'max_abs_conditional_pnl_usd': None,
        'evidence_tier': 'C_UNQUANTIFIED',
        'basis': 'NO_ADMITTED_COMPLETE_CONDITIONAL_PORTFOLIO_PNL',
        'not_expected_loss': True,
        'not_event_probability_weighted': True,
        'cross_event_comparison': 'POINT_MAGNITUDES_DEPEND_ON_EACH_EVENTS_MODEL_AND_PREMISES',
    }
    magnitude = _primary_magnitude(risk)
    if magnitude is not None:
        return {**base, 'max_abs_conditional_pnl_usd': magnitude,
                'evidence_tier': 'A_ADMITTED_MODEL',
                'basis': 'MAX_ABS_SOURCE_ADMITTED_CONDITIONAL_OUTCOME_PNL_FROM_CURRENT_SNAPSHOT'}
    magnitude = _autonomous_common_endpoint_magnitude(risk, positions, notional, observed_mark_context)
    if magnitude is not None:
        return {**base, 'max_abs_conditional_pnl_usd': magnitude,
                'evidence_tier': 'B_AUTONOMOUS_COMMON_ENDPOINT_SCENARIO',
                'basis': 'ABS_MODEL_ASSUMED_CENTRAL_PAIRED_EVENT_YES_MINUS_NO_GAP_AT_ONE_DATE',
                'comparison_is_model_assumed': True,
                'not_empirically_causal_or_calibrated': True}
    magnitude = _analyst_magnitude(risk, positions, notional)
    if magnitude is not None:
        return {**base, 'max_abs_conditional_pnl_usd': magnitude,
                'evidence_tier': 'B_DECLARED_ANALYST_SCENARIO',
                'basis': 'MAX_ABS_ILLUSTRATIVE_YES_NO_PNL_AT_DECLARED_PRICING_WEIGHT'}
    return base


def sort_risks_by_conditional_movement(risks, positions, notional, *, observed_mark_context=None):
    """Rank A/B by modeled dollars, retain C as unranked and auxiliaries last.

    Sorting does not certify that the largest point estimate is truly riskier.
    In particular B stresses are model/analyst assumptions, and A vs B
    model classes have no common statistical confidence level.
    """
    if not isinstance(risks, list):
        raise ValueError('Risk list required')
    modeled, unknown, auxiliary = [], [], []
    for index, original in enumerate(risks):
        if not isinstance(original, dict):
            raise ValueError('Risk entries must be mappings')
        risk = deepcopy(original)
        if risk.get('source_defined_scenario') is True or risk.get('what_if_explorer'):
            auxiliary.append((index, risk))
            continue
        priority = conditional_movement_priority(risk, positions, notional,
                                                 observed_mark_context=observed_mark_context)
        risk['impact_priority'] = priority
        amount = priority['max_abs_conditional_pnl_usd']
        if amount is None:
            unknown.append((index, risk))
        else:
            tier_key = 0 if priority['evidence_tier'] == 'A_ADMITTED_MODEL' else 1
            modeled.append((-amount, tier_key, index, risk))
    modeled.sort(key=lambda row: row[:3])
    result = []
    for order, (_, _, _, risk) in enumerate(modeled, 1):
        risk['impact_priority']['modeled_display_order'] = order
        result.append(risk)
    for _, risk in unknown:
        risk['impact_priority']['modeled_display_order'] = None
        result.append(risk)
    result.extend(risk for _, risk in auxiliary)
    return result
