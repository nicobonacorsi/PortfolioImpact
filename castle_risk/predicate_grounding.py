"""Typed, fail-closed *review* qualification for discovered settlement predicates.

This is not a theorem prover, quote verifier, or provider schema authority.
It rejects explicit conflicts before paying a language model to read terms.
A positive verdict only grants further review, never a logical implication.
No network request, dependency on current time, or portfolio mutation occurs.
"""
from __future__ import annotations

from hashlib import sha256
from datetime import datetime, date
import re
from typing import Any, Mapping


_NUMERIC = re.compile(
    r"(?P<op>at\s+least|no\s+less\s+than|more\s+than|greater\s+than|"
    r"strictly\s+above|above|exceeds?|less\s+than|below|under|"
    r"at\s+most|no\s+more\s+than|equal\s+to|exactly|>=|<=|>|<|=)\s*"
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>%|percent(?:age\s+points?)?|"
    r"basis\s+points?|bps)(?=$|[\s,.;:!?)])",
    flags=re.I,
)

_OP = {
    'at least': 'ge', 'no less than': 'ge', '>=': 'ge',
    'more than': 'gt', 'greater than': 'gt', 'strictly above': 'gt',
    'above': 'gt', 'exceed': 'gt', 'exceeds': 'gt', '>': 'gt',
    'less than': 'lt', 'below': 'lt', 'under': 'lt', '<': 'lt',
    'at most': 'le', 'no more than': 'le', '<=': 'le',
    'equal to': 'eq', 'exactly': 'eq', '=': 'eq',
}


def _measurement(text: str) -> dict[str, Any] | None:
    found = []
    for hit in _NUMERIC.finditer(text):
        word = ' '.join(hit.group('op').lower().split())
        unit = hit.group('unit').lower()
        unit = ('percent' if unit == '%' or unit.startswith('percent') else 'basis_points')
        found.append({'operator': _OP[word], 'threshold': float(hit.group('value')),
                      'unit': unit})
    if not found:
        return None
    # Multiple independent numerical predicates are a compound question.
    return found[0] if len(found) == 1 else {'ambiguous_multiple_conditions': True}


def _basis(text: str) -> str:
    lower = text.lower()
    if re.search(r'\bcore\s+cpi\b|\bcore\s+consumer\s+price\b|'
                 r'\b(?:less|exclud(?:ing|es))\s+food\s+(?:and|&)\s+energy\b', lower):
        return 'CORE'
    if re.search(r'\bheadline\b|\ball.items\b', lower):
        return 'HEADLINE'
    return 'UNSPECIFIED'


def _seasonal(text: str) -> str:
    lower = text.lower()
    if re.search(r'\b(?:not\s+seasonally\s+adjusted|non[- ]seasonally\s+adjusted|unadjusted|nsa)\b', lower):
        return 'NSA'
    if re.search(r'\b(?:seasonally\s+adjusted|sa)\b', lower):
        return 'SA'
    return 'UNSPECIFIED'


def _frequency(text: str) -> str:
    lower = text.lower()
    if re.search(r'\b(?:month[- ]over[- ]month|mom|monthly\s+(?:change|increase|inflation|rate))\b', lower):
        return 'MOM'
    if re.search(r'\b(?:year[- ]over[- ]year|yoy|12[- ]month\s+(?:change|increase|rate)|annual\s+inflation\s+rate)\b', lower):
        return 'YOY'
    return 'UNSPECIFIED'


def _fed_action(text: str) -> str:
    actions = set()
    if re.search(r'\b(?:cut|cuts|lower|lowers|decrease|decreases|reduc(?:e|es|tion))\b', text, re.I):
        actions.add('CUT')
    if re.search(r'\b(?:hike|hikes|rais(?:e|es)|increase|increases)\b', text, re.I):
        actions.add('HIKE')
    if re.search(r'\b(?:hold|holds|unchanged|no\s+change|maintain|maintains)\b', text, re.I):
        actions.add('HOLD')
    return next(iter(actions)) if len(actions) == 1 else 'UNSPECIFIED' if not actions else 'COMPOUND'


def explicit_event_profile(event: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Expose whether the question, *as proposed*, defines an exact binary target."""
    prose = f'{event.get("title", "")} {event.get("description", "")}'
    measurement = _measurement(prose)
    compound = bool(re.search(r'\b(?:or|and/or)\b', prose, re.I))
    if family == 'US_FOMC':
        action = _fed_action(prose)
        precise = action not in ('UNSPECIFIED', 'COMPOUND')
    elif family in ('US_CPI', 'US_TREASURY_YIELD'):
        action = None
        precise = bool(measurement and not measurement.get('ambiguous_multiple_conditions'))
    elif family in ('US_RECESSION', 'US_CHIP_EXPORT_POLICY'):
        action = None
        precise = bool(re.search(r'\b(?:nber|declared|enacted|prohibited|banned)\b', prose, re.I))
    else:
        action = None
        precise = False
    return {'family': family, 'binary_target_status': (
            'EXPLICIT_TESTABLE_OUTCOME_REQUIRES_SOURCE_CONFIRMATION' if precise and not compound
            else 'UNDERDEFINED_OR_COMPOUND_MACRO_QUESTION'),
            'explicit_threshold': measurement, 'explicit_action': action,
            'cpi_basis': _basis(prose) if family == 'US_CPI' else None,
            'cpi_seasonal_adjustment': _seasonal(prose) if family == 'US_CPI' else None,
            'cpi_change_frequency': _frequency(prose) if family == 'US_CPI' else None,
            'definition_origin': 'UNTRUSTED_EVENT_TEXT_NOT_SOURCE_EVIDENCE'}


def qualify_rule_predicate(event: Mapping[str, Any], market: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Reject explicit definition conflicts, without upgrading compatibility into equivalence.

    Source receipts are checked separately by the original catalog/bridge. Any
    numeric operators here are *lexical diagnostics*, not independent proofs
    about rounding, exceptional settlement, or logical relationships.
    """
    primary = market.get('rules_primary')
    secondary = market.get('rules_secondary')
    if not isinstance(primary, str) or not primary.strip():
        return {'review_compatible': False, 'reason': 'MISSING_PRIMARY_SETTLEMENT_PREDICATE'}
    if not isinstance(secondary, str):
        secondary = ''
    all_rules = primary + '\n' + secondary
    source_profile = {
        'rule_text_sha256': sha256(all_rules.encode('utf-8')).hexdigest(),
        'cpi_basis': _basis(all_rules) if family == 'US_CPI' else None,
        'cpi_seasonal_adjustment': _seasonal(all_rules) if family == 'US_CPI' else None,
        'cpi_change_frequency': _frequency(primary) if family == 'US_CPI' else None,
        'numeric_predicate': _measurement(primary),
        'close_time': str(market.get('close_time') or ''),
        'expiration_time': str(market.get('expiration_time') or ''),
        'fed_policy_action': _fed_action(primary) if family == 'US_FOMC' else None,
        'resolution_exception_flags': [label for label, pattern in (
            ('GOVERNMENT_SHUTDOWN', r'government\s+shutdown'),
            ('POSTPONEMENT_OR_DELAY', r'\b(?:delay|delayed|postpon(?:e|ed|ement))\b'),
            ('EXPIRATION_EXTENSION', r'\b(?:extend|extended|extension)\b'),
            ('CANCELLATION', r'\b(?:cancel|cancelled|cancellation)\b'),
            ('FALLBACK_DEFINITION', r'\b(?:fallback|otherwise|substitut(?:e|ion))\b'),
        ) if re.search(pattern, all_rules, re.I)],
        'full_rule_reading_required': True,
        'semantic_equivalence_certified': False,
    }
    event_profile = explicit_event_profile(event, family)
    # A dated outcome explicitly occurring ON the wrong reviewed decision day
    # is not the same target; expiry extensions are intentionally not compared
    # to release dates because a source's shutdown/fallback clause can extend
    # legal settlement beyond the scheduled economic announcement.
    reviewed = event.get('official_calendar_review')
    if not isinstance(reviewed, Mapping):
        reviewed = {}
    scheduled_day = reviewed.get('reviewed_date')
    when = re.search(r'\b(?:on|meeting\s+of|scheduled\s+for)\s+'
                     r'(January|February|March|April|May|June|July|August|September|October|November|December)'
                     r'\s+(\d{1,2}),\s+(20\d{2})\b', primary, re.I)
    if when:
        try:
            explicit_day = datetime.strptime(' '.join(when.groups()), '%B %d %Y').date().isoformat()
        except ValueError:
            explicit_day = 'INVALID_SOURCE_DATE'
        source_profile['explicit_primary_rule_date'] = explicit_day
    else:
        explicit_day = None
    def deny(reason: str) -> dict[str, Any]:
        return {'review_compatible': False, 'reason': reason,
                'economic_predicate': event_profile, 'settlement_predicate': source_profile}

    if explicit_day is not None and isinstance(scheduled_day, str) and explicit_day != scheduled_day:
        return deny('PRIMARY_SETTLEMENT_DATE_CONTRADICTS_REVIEWED_SCHEDULE')

    if family == 'US_CPI':
        for key, reason in (
            ('cpi_basis', 'CPI_HEADLINE_CORE_DEFINITION_CONFLICT_OR_UNPROVEN'),
            ('cpi_seasonal_adjustment', 'CPI_SEASONAL_ADJUSTMENT_CONFLICT_OR_UNPROVEN'),
            ('cpi_change_frequency', 'CPI_MOM_YOY_DEFINITION_CONFLICT_OR_UNPROVEN'),
        ):
            requested = event_profile[key]
            if requested != 'UNSPECIFIED' and requested != source_profile[key]:
                # Missing source detail also fails closed for an explicit target.
                return deny(reason)
    elif family == 'US_FOMC':
        action = event_profile['explicit_action']
        provider_action = source_profile['fed_policy_action']
        if action not in ('UNSPECIFIED', 'COMPOUND') and provider_action != action:
            return deny('FED_POLICY_ACTION_MISMATCH_OR_UNPROVEN')

    a, b = event_profile['explicit_threshold'], source_profile['numeric_predicate']
    if a and a.get('ambiguous_multiple_conditions'):
        return deny('AMBIGUOUS_MULTIPLE_EVENT_THRESHOLDS')
    if a and b:
        if b.get('ambiguous_multiple_conditions'):
            return deny('MULTIPLE_CONTRACT_RULE_THRESHOLDS_REQUIRE_MANUAL_REVIEW')
        if a != b:
            return deny('THRESHOLD_UNIT_OR_STRICTNESS_CONFLICT')
    # A numeric target cannot be attached to a contract omitting the relevant
    # threshold. This is NOT the same as the generic macro family being valid.
    if a and not b:
        return deny('EXPLICIT_THRESHOLD_MISSING_FROM_CONTRACT_RULES')

    return {'review_compatible': True, 'reason': 'NO_EXPLICIT_TYPED_PREDICATE_CONFLICT_DETECTED',
            'economic_predicate': event_profile, 'settlement_predicate': source_profile,
            'scope': 'CANDIDATE_FOR_FULL_RULE_REVIEW_NEVER_LOGICAL_CERTIFICATION'}
