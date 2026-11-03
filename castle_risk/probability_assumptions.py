"""Castle Option 1 probability workbench: separate market Q and analyst P.

Unmatched events must not inherit odds from a different contract, an LLM,
or the portfolio scenario's *equity pricing* q. A user may instead provide
an explicit subjective credal interval, or observed reference-class counts
with independently stated exchangeability assumptions. Neither option is
certified market data or an empirically validated forecast by this module.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import math
import re

from .live_web_research import public_source_url


SCHEMA = "castle.event_probability_assumptions.v1"
MAX_HORIZON_DAYS = 366
MAX_FORECAST_TEXT = 650
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_Z95 = 1.959963984540054
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class ProbabilityInputError(ValueError):
    """A prior or source record failed validation; do not make up a result."""


def _number(value, label, *, minimum=0., maximum=1.):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ProbabilityInputError(f"{label} must be finite and numeric.")
    if not minimum <= value <= maximum:
        raise ProbabilityInputError(f"{label} must be between {minimum} and {maximum}.")
    return float(value)


def _text(value, label, *, min_length=15):
    if not isinstance(value, str):
        raise ProbabilityInputError(f"{label} is required.")
    value = " ".join(value.split())
    if not min_length <= len(value) <= MAX_FORECAST_TEXT or any(ord(c) < 32 for c in value):
        raise ProbabilityInputError(f"{label} must explain the assumption in plain language.")
    return value


def _source(value, *, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ProbabilityInputError("A public HTTPS context URL is required.")
    clean = public_source_url(value)
    if clean is None or clean != value:
        raise ProbabilityInputError("The source URL must be a public HTTPS URL without query tokens.")
    return clean


def probability_question_context(case, event):
    """Date- and predicate-bound context. Never promote outcome Q to event P."""
    if not isinstance(case, dict) or not isinstance(event, dict):
        return None
    if event.get('source_defined_scenario') is True or event.get('what_if_explorer'):
        return None
    identifier, condition = event.get('id'), event.get('title')
    if not isinstance(identifier, str) or not identifier or len(identifier) > 300:
        return None
    if not isinstance(condition, str) or not 4 <= len(condition) <= 650:
        return None
    instant = case.get('capture')
    if not isinstance(instant, str):
        return None
    try:
        as_of = datetime.fromisoformat(instant.replace('Z', '+00:00'))
        event_date = event.get('date')
        if not isinstance(event_date, str) or not _DATE.fullmatch(event_date):
            return None
        deadline = date.fromisoformat(event_date)
    except (ValueError, OverflowError):
        return None
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        return None
    as_of = as_of.astimezone(timezone.utc)
    horizon = (deadline - as_of.date()).days
    if not 1 <= horizon <= MAX_HORIZON_DAYS:
        return None
    # A scheduled *release* is not its adverse surprise. Market outcomes for
    # the release should be inspected directly instead of asking for the
    # probability that a known calendar date occurs.
    scheduled = (str(event.get('id') or '').endswith('-scheduled') or
                 '(scheduled)' in condition.lower())
    outcome_rows = event.get('outcomes') or event.get('market_outcomes') or []
    admitted_ids = {row.get('id') for row in event.get('admitted') or []
                    if isinstance(row, dict) and isinstance(row.get('id'), str)}
    quoted = []
    for row in outcome_rows:
        if not isinstance(row, dict) or not _valid_probability(row.get('q_reference')):
            continue
        references = row.get('source_contract_ids')
        references = {value for value in references if isinstance(value, str)} if isinstance(references, list) else set()
        if (isinstance(row.get('contract_id'), str) and row['contract_id'] in admitted_ids) or references & admitted_ids:
            quoted.append(row)
    return {
        'schema': SCHEMA, 'event_id': identifier, 'target_condition': condition,
        'as_of_utc': as_of.isoformat(), 'deadline': event_date,
        'horizon_days': horizon,
        'status': ('MARKET_OUTCOME_DISTRIBUTION' if quoted else
                   'SCHEDULED_RELEASE_NO_MATCHED_MARKET_OUTCOME') if scheduled else
                  'OPEN_FOR_DECLARED_EVENT_FORECAST',
        'source_market_Q_outcomes': len(quoted),
        'source_market_Q_is_physical_event_P': False,
        'physical_event_probability_P': None,
        'assumed_event_probability_P': None,
        'analyst_probability_is_not_equity_pricing_q': True,
        'analyst_probability_is_not_portfolio_return': True,
        'analyst_forecast_permission': not scheduled,
        'source_scope': 'SAVED_EVENT_DEFINITION_NOT_SOURCE_VALIDATION',
    }


def _valid_probability(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _predicate_template(value):
    # Only ONE settlement deadline may move between otherwise identical
    # questions. Replacing every date would silently change multi-date
    # observation windows, violating the reference-class identity gate.
    if not isinstance(value, str) or len(_ISO_DATE.findall(value)) != 1:
        return None
    return _ISO_DATE.sub('<DATE>', ' '.join(value.lower().split()), count=1)


def wilson_95_interval(successes: int, trials: int):
    """Finite-sample Wilson score interval; defined also at k=0 and k=n."""
    if (type(trials) is not int or type(successes) is not int or
            not 1 <= trials <= 100_000 or not 0 <= successes <= trials):
        raise ProbabilityInputError("Reference-class counts must be 0 <= successes <= trials.")
    p = successes / trials
    z2 = _Z95 * _Z95
    den = 1. + z2 / trials
    mid = (p + z2 / (2 * trials)) / den
    rad = _Z95 * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / den
    return [max(0., mid - rad), min(1., mid + rad)]


def assess_declared_event_probability(context, request):
    """A *separate*, explicit event-P scenario; not a quote or forecast fit.

    Reference-class frequencies are conditional on the user-asserted
    comparability/exchangeability of past event trials with the future target.
    There is no automatic promotion of user-supplied counts to sourced facts.
    """
    if (not isinstance(context, dict) or context.get('schema') != SCHEMA or
            context.get('analyst_forecast_permission') is not True or
            context.get('status') != 'OPEN_FOR_DECLARED_EVENT_FORECAST' or
            not isinstance(context.get('target_condition'), str) or
            type(context.get('horizon_days')) is not int or
            not 1 <= context['horizon_days'] <= MAX_HORIZON_DAYS or
            context.get('physical_event_probability_P') is not None or
            context.get('assumed_event_probability_P') is not None or
            context.get('source_market_Q_is_physical_event_P') is not False):
        raise ProbabilityInputError("The saved target is not eligible for an assumed event probability.")
    if not isinstance(request, dict):
        raise ProbabilityInputError("An explicit probability assumption is required.")
    mode = request.get('mode')
    common = {'mode', 'event_id', 'as_of_utc', 'rationale', 'counterevidence', 'context_url'}
    allowed = ({'low', 'center', 'high'} if mode == 'subjective_interval' else
               {'reference_question', 'horizon_days', 'trials', 'successes',
                'exchangeability_declared'} if mode == 'reference_frequency' else None)
    if allowed is None or set(request) - (common | allowed):
        raise ProbabilityInputError("Unsupported probability mode or fields.")
    if request.get('event_id') != context.get('event_id') or request.get('as_of_utc') != context.get('as_of_utc'):
        raise ProbabilityInputError("The declared prior does not match this exact event and snapshot.")
    rationale = _text(request.get('rationale'), 'Evidence and assumptions', min_length=25)
    alternative = _text(request.get('counterevidence'), 'Counterevidence', min_length=15)
    source_url = _source(request.get('context_url'), required=mode == 'reference_frequency')

    if mode == 'subjective_interval':
        low = _number(request.get('low'), 'Lower event probability')
        center = _number(request.get('center'), 'Central event probability')
        high = _number(request.get('high'), 'Upper event probability')
        if not low <= center <= high:
            raise ProbabilityInputError("The central assumption must lie inside the probability interval.")
        method, assurance = ('USER_DECLARED_CREDAL_EVENT_PROBABILITY_INTERVAL',
                             'SUBJECTIVE_SCENARIO_NOT_STATISTICAL_CONFIDENCE_INTERVAL')
        reference = None
    else:
        if (type(request.get('horizon_days')) is not int or
                request['horizon_days'] != context.get('horizon_days')):
            raise ProbabilityInputError("Reference trials must use the exact same observation horizon.")
        if request.get('exchangeability_declared') is not True:
            raise ProbabilityInputError("Comparable, independent reference trials must be explicitly assumed.")
        question = _text(request.get('reference_question'), 'Reference class event definition', min_length=25)
        if (_predicate_template(question) is None or
                _predicate_template(question) != _predicate_template(context['target_condition'])):
            raise ProbabilityInputError(
                "Reference observations must have the same event predicate; only the ISO deadline may differ.")
        n, k = request.get('trials'), request.get('successes')
        low, high = wilson_95_interval(k, n)
        center = k / n
        method, assurance = ('USER_REPORTED_REFERENCE_CLASS_FREQUENCY',
                             'WILSON_95_SAMPLING_INTERVAL_CONDITIONAL_ON_EXCHANGEABILITY')
        reference = {'event_definition': question, 'horizon_days': context['horizon_days'],
                     'trials': n, 'successes': k,
                     'independent_and_exchangeable': 'USER_ASSUMED_NOT_VERIFIED',
                     'raw_observations_inspected': False,
                     'source_url': source_url}

    return {
        'schema': SCHEMA, 'status': 'DECLARED_PROBABILITY_SCENARIO',
        'event_id': context['event_id'], 'target_condition': context['target_condition'],
        'as_of_utc': context['as_of_utc'], 'deadline': context['deadline'],
        'horizon_days': context['horizon_days'], 'mode': mode,
        'center': center, 'probability_interval': [low, high],
        'method': method, 'assurance': assurance,
        'rationale': rationale, 'counterevidence': alternative,
        'context_url': source_url, 'reference_class': reference,
        'physical_event_probability_identified_by_data': False,
        'prediction_market_Q_substituted': False,
        'market_quote_changed': False, 'market_match_changed': False,
        'equity_pricing_q_changed': False, 'portfolio_pnl_changed': False,
        'historical_calibration_verified': False,
        'source_inspection': 'USER_PROVIDED_CONTEXT_NOT_INDEPENDENTLY_VERIFIED',
    }


def attach_probability_questions(case):
    """Adds only *read-only* probability question metadata to target risks."""
    for risk in case.get('risks') or []:
        context = probability_question_context(case, risk)
        if context is not None:
            risk['event_probability_question'] = context
    return case
