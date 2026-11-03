"""Inspect exact Gamma rule clauses without creating a second semantic proof gate.

The parser recognizes declared clause forms, preserves every original span and
retains unfamiliar text. Recognition is not a complete payoff model. The text
entrypoint never authenticates caller-supplied source metadata; archive wrappers
independently reproduce the original receipt binding before qualifying origin.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .event_discovery import read_p1_capture

PARSER_VERSION = 'gamma-literal-clause-audit-v1'
MAX_RULE_CHARACTERS = 30_000
_IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,128}')
_QUOTE_YES = r'["\u201c]Yes["\u201d]'
_QUOTE_NO = r'["\u201c]No["\u201d]'
_ACTOR_QUALIFIER = re.compile(
    r'\b(?:if|unless|when|without|not|no|neither|either|nor|only|except|provided|assuming|'
    r'that|true|false|never|always|fails?|does|do|did|and|or)\b', re.I)


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('An exact bounded event and market ID is required')
    return value


def _instant(value: datetime | str) -> datetime:
    if not isinstance(value, (datetime, str)):
        raise ValueError('Timezone-aware evaluation time required')
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Timezone-aware evaluation time required')
    return dt.astimezone(timezone.utc)


def _clauses(text: str):
    # Keep offsets into the original Unicode text. Do not normalize whitespace
    # or split an abbreviation such as e.g. before a lowercase continuation.
    start = 0
    for boundary in re.finditer(r'(?<=[.!?])\s+(?=[A-Z"\u201c])|\n+', text):
        end = boundary.start()
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right, text[left:right]
        start = boundary.end()
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        yield left, right, text[left:right]


def audit_gamma_rule_text(
    *, event_id: str, market_id: str, question: str, full_rules: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Low-cost clause audit for an already matched market, or explicit text.

    The caller's source flags cannot establish original-source authenticity.
    The parent investigation may separately bind this exact rules hash to its
    verified source. Use the archive wrapper for a self-contained source check.
    """
    event_id, market_id = _identity(event_id), _identity(market_id)
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise ValueError('A bounded original market question is required')
    if (not isinstance(full_rules, str) or not full_rules.strip()
            or len(full_rules) > MAX_RULE_CHARACTERS):
        raise ValueError('Bounded, nonempty full market rules are required')
    if source_metadata is not None and not isinstance(source_metadata, Mapping):
        raise ValueError('Source metadata must be a mapping')
    metadata = source_metadata or {}
    digest = sha256(full_rules.encode('utf-8')).hexdigest()
    for field, expected in (('event_id', event_id), ('market_id', market_id), ('rules_sha256', digest)):
        if field in metadata and metadata[field] != expected:
            raise ValueError('Supplied source metadata does not match exact market rules identity')
    binding = {k: metadata[k] for k in ('archive_sha256', 'original_page_sha256', 'retrieved_at_utc')
               if k in metadata and isinstance(metadata[k], str)}
    facts, unknown, recognized = [], [], []
    definition_subjects: list[str] = []
    settlement_count = 0
    condition_ambiguities: list[str] = []

    for clause_start, clause_end, clause in _clauses(full_rules):
        count_before = len(facts)

        def add(facet, role, polarity, begin, end, rule_id):
            absolute_start, absolute_end = clause_start + begin, clause_start + end
            facts.append({'fact_id': f'f{len(facts) + 1:03d}', 'facet': facet,
                'role': role, 'polarity': polarity, 'quote': full_rules[absolute_start:absolute_end],
                'start': absolute_start, 'end': absolute_end,
                'clause_start': clause_start, 'clause_end': clause_end,
                'clause_quote': clause, 'rules_sha256': digest, 'rule_id': rule_id,
                'logical_scope': 'EXACT_CLAUSE_ONLY_BOOLEAN_CONNECTIVES_NOT_FORMALIZED',
                'interpretation': 'LITERAL_CLAUSE_ROLE_NOT_COMPLETE_EVENT_PREDICATE'})

        match = re.fullmatch(r'This market will resolve to ' + _QUOTE_YES + r' if (?P<condition>.+)\.', clause, re.I)
        if match:
            settlement_count += 1
            begin, end = match.span('condition')
            add('settlement_condition', 'requirement', 'positive', begin, end, 'yes-if-condition')
            condition = match.group('condition')
            offensive = re.fullmatch(
                r'(?P<actor>.+?) (?P<action>commences a military offensive) '
                r'(?P<intention>intended to establish control) over (?P<geography>.+?) by (?P<time_end>.+)',
                condition, re.I)
            encounter = re.fullmatch(
                r'there is a (?P<action>military encounter) between the military forces of '
                r'(?P<actor_a>.+?) and (?P<actor_b>.+?) between (?P<time_start>.+?), and (?P<time_end>.+)',
                condition, re.I)
            parsed_substructure = False
            for parsed, rule_id in ((offensive, 'offensive-onset-control-intention'),
                                    (encounter, 'military-encounter-actors-window')):
                if parsed:
                    actors = [value for name, value in parsed.groupdict().items() if name.startswith('actor')]
                    if any(_ACTOR_QUALIFIER.search(actor) or len(actor) > 240 for actor in actors):
                        condition_ambiguities.append('UNSUPPORTED_ACTOR_OR_PROPOSITION_QUALIFIER')
                        continue
                    parsed_substructure = True
                    for name in parsed.groupdict():
                        a, b = parsed.span(name)
                        add(name, 'requirement', 'positive', begin + a, begin + b, rule_id)
            if not parsed_substructure:
                condition_ambiguities.append('SETTLEMENT_CONDITION_SUBSTRUCTURE_UNPARSED')
        elif re.fullmatch(r'Otherwise, this market will resolve to ' + _QUOTE_NO + r'\.', clause, re.I):
            add('settlement_fallback', 'fallback', 'negative', 0, len(clause), 'otherwise-no')
        else:
            patterns = [
                ('territory-inclusion-exclusion',
                 r'Territory under the administration of (?P<administration>.+?) including '
                 r'(?P<geography_included>.+?) will qualify, however (?P<geography_excluded>.+?) will not qualify\.',
                 {'administration': ('definition', 'positive'), 'geography_included': ('inclusion', 'positive'),
                  'geography_excluded': ('exclusion', 'negative')}),
                ('explicit-term-definition',
                 r'(?:A|An) ["\u201c](?P<defined_term>[^"\u201d]+)["\u201d] is defined as (?P<action_definition>.+)\.',
                 {'defined_term': ('definition', 'not_applicable'), 'action_definition': ('definition', 'positive')}),
                ('nonqualifying-action-examples',
                 r'Non-violent actions, such as (?P<excluded_actions>.+?), will not qualify for a ' + _QUOTE_YES + r' resolution\.',
                 {'excluded_actions': ('exclusion_examples', 'negative')}),
                ('intentional-force-inclusion-exclusion',
                 r'(?P<included_force>Intentional .+?) will count toward a ' + _QUOTE_YES +
                 r' resolution, however (?P<excluded_force>.+?) will not\.',
                 {'included_force': ('inclusion', 'positive'), 'excluded_force': ('exclusion', 'negative')}),
                ('military-agency-inclusion-exclusion',
                 r'Note: the (?P<actor_included>.+?) is part of the military, however (?P<actor_excluded>.+?) is not\.',
                 {'actor_included': ('inclusion', 'positive'), 'actor_excluded': ('exclusion', 'negative')}),
                ('resolution-authority',
                 r'The resolution source for this market will be (?:will be )?(?P<resolution_authority>.+)\.',
                 {'resolution_authority': ('resolution_authority', 'not_applicable')}),
            ]
            for rule_id, pattern, roles in patterns:
                parsed = re.fullmatch(pattern, clause, re.I)
                if not parsed:
                    continue
                for name, (role, polarity) in roles.items():
                    a, b = parsed.span(name)
                    add(name, role, polarity, a, b, rule_id)
                if 'defined_term' in parsed.groupdict():
                    definition_subjects.append(parsed.group('defined_term').casefold())
                break
        if len(facts) == count_before:
            unknown.append({'quote': clause, 'start': clause_start, 'end': clause_end})
        else:
            recognized.append({'quote': clause, 'start': clause_start, 'end': clause_end})

    ambiguities = list(dict.fromkeys(condition_ambiguities))
    if settlement_count != 1:
        ambiguities.append('EXPECTED_ONE_EXPLICIT_YES_CONDITION')
    if len(definition_subjects) != len(set(definition_subjects)):
        ambiguities.append('MULTIPLE_DEFINITIONS_OF_THE_SAME_TERM')
    partial = bool(unknown or ambiguities)
    return {
        'schema': 'castle.gamma_rule_fact_audit.v1', 'parser_version': PARSER_VERSION,
        'event_id': event_id, 'market_id': market_id, 'question': question,
        'full_rules': full_rules, 'rules_sha256': digest,
        'source_origin': 'CALLER_SUPPLIED_TEXT_ORIGIN_NOT_REVERIFIED',
        'raw_bytes_reverified_here': False, 'supplied_parent_source_binding': binding,
        'facts_status': ('SOURCE_TEXT_SPANS_VERIFIED_PARTIAL' if partial else
                         'SOURCE_TEXT_SPANS_VERIFIED_RECOGNIZED_CLAUSES'),
        'facts': facts, 'recognized_clauses': recognized, 'unclassified_clauses': unknown,
        'parsing_ambiguities': ambiguities,
        'parser_scope': 'VERSIONED_LITERAL_CLAUSE_FORMS_NOT_COMPLETE_PAYOFF_SEMANTICS',
        'complete_semantic_model': False, 'relationship_status': 'NOT_CERTIFIED',
        'broader_geopolitical_event_equivalence': 'NOT_ESTABLISHED',
        'lp_constraint_admitted': False, 'physical_probability_P': None,
        'equity_priced_in_q': None, 'identified_event_conditional_return': None,
        'limitations': [
            'Span verification establishes the supplied text and clause role, not real-world occurrence or a complete payoff model.',
            'All clauses remain available; unfamiliar clauses and repeated definitions prevent full recognition claims.',
            'An unstated start, actor, intention or geographic condition remains unspecified rather than inferred.',
            'Intentionality in one inclusion clause is not generalized to every qualifying action.',
            'Matching source text or deadlines does not certify cross-contract implication, independence or equivalence.',
        ],
    }


def _archive_audits(archive_bytes, identities, evaluated_at):
    # Local import avoids a cycle when the parent investigation attaches the
    # inexpensive text audit after its own exact source matching.
    from .archived_nonmacro_case import P1_ARCHIVE_SHA256

    source = read_p1_capture(archive_bytes)
    now = _instant(evaluated_at)
    if now < max(_instant(r['completed_at_utc']) for r, _ in source['pages']):
        raise ValueError('Evaluation precedes completed source acquisition')
    output = []
    for event_id, market_id in identities:
        event_id, market_id = _identity(event_id), _identity(market_id)
        matches = []
        for receipt, body in source['pages']:
            if receipt['provider'] != 'polymarket':
                continue
            for event in body['events']:
                if (not isinstance(event, dict) or isinstance(event.get('id'), bool)
                        or not isinstance(event.get('id'), (str, int))
                        or str(event.get('id')) != event_id):
                    continue
                markets = event.get('markets', [])
                if not isinstance(markets, list):
                    continue
                for market in markets:
                    if (isinstance(market, dict) and not isinstance(market.get('id'), bool)
                            and isinstance(market.get('id'), (str, int))
                            and str(market.get('id')) == market_id):
                        matches.append((receipt, market))
        if len(matches) != 1:
            raise ValueError('Exact original event and market identity is missing or ambiguous')
        receipt, market = matches[0]
        try:
            outcomes = json.loads(market.get('outcomes', ''))
        except (ValueError, TypeError) as exc:
            raise ValueError('Exact source Yes/No outcomes required') from exc
        if outcomes != ['Yes', 'No']:
            raise ValueError('Exact source Yes/No outcomes required')
        metadata = {'archive_sha256': source['archive_sha256'],
                    'original_page_sha256': receipt['sha256'],
                    'retrieved_at_utc': receipt['completed_at_utc']}
        result = audit_gamma_rule_text(event_id=event_id, market_id=market_id,
            question=market.get('question'), full_rules=market.get('description'), source_metadata=metadata)
        pinned = source['archive_sha256'] == P1_ARCHIVE_SHA256
        result.update(source_origin=('PINNED_EXISTING_CANDIDATE_CAPTURE' if pinned else
                                     'SUPPLIED_RECEIPT_CONSISTENT_ARCHIVE_ORIGIN_UNVERIFIED'),
                      raw_bytes_reverified_here=True, evaluated_at_utc=now.isoformat())
        result['source'] = {**metadata, 'original_url': receipt['url'],
            'original_page_path': receipt['raw_file'], 'matches_recovered_capture_sha256': pinned,
            'provider_signed': False, 'historical_publication_time_verified': False}
        output.append(result)
    return output


def audit_archived_gamma_rule_facts(
    archive_bytes: bytes, *, event_id: str, market_id: str, evaluated_at: datetime | str,
) -> dict[str, Any]:
    """Verify original archive receipts and audit one exact market's full rules."""
    return _archive_audits(archive_bytes, [(event_id, market_id)], evaluated_at)[0]


def compare_gamma_rule_facts(
    archive_bytes: bytes, *, left_event_id: str, left_market_id: str,
    right_event_id: str, right_market_id: str, evaluated_at: datetime | str,
) -> dict[str, Any]:
    """Compare literal requirements; emit no certified relation or LP edge."""
    if (left_event_id, left_market_id) == (right_event_id, right_market_id):
        raise ValueError('Choose two distinct exact contracts for comparison')
    left, right = _archive_audits(archive_bytes,
        [(left_event_id, left_market_id), (right_event_id, right_market_id)], evaluated_at)
    facets = sorted({f['facet'] for report in (left, right) for f in report['facts']})
    differences = []
    for facet in facets:
        a = [f for f in left['facts'] if f['facet'] == facet]
        b = [f for f in right['facts'] if f['facet'] == facet]
        if not a or not b:
            status = 'UNSPECIFIED_IN_LEFT' if not a else 'UNSPECIFIED_IN_RIGHT'
        else:
            status = 'SAME_EXPLICIT_TEXT' if [f['quote'] for f in a] == [f['quote'] for f in b] else 'DIFFERENT_EXPLICIT_TEXT'
        differences.append({'facet': facet, 'comparison': status, 'left': a, 'right': b})
    interpretations = []
    offensive = next((r for r in (left, right) if any(f['rule_id'] == 'offensive-onset-control-intention'
                     for f in r['facts'])), None)
    encounter = next((r for r in (left, right) if any(f['rule_id'] == 'military-encounter-actors-window'
                     for f in r['facts'])), None)
    if offensive is not None and encounter is not None and offensive is not encounter:
        interpretations.append({
            'status': 'INTERPRETIVE_COUNTEREXAMPLE_TEMPLATE_NOT_CERTIFIED_MODEL',
            'illustration': 'Assuming all encounter clauses are met, a force incident without any territorial-control intention does not by itself establish the control-intention requirement.',
            'encounter_contract_id': encounter['market_id'], 'control_intention_contract_id': offensive['market_id'],
            'requires_review_of_all_other_clauses_and_other_occurrences': True,
            'evidence': [f for f in offensive['facts'] if f['facet'] == 'intention'] +
                        [f for f in encounter['facts'] if f['facet'] in ('action', 'actor_a', 'actor_b')],
        })
        exclusions = [f for f in encounter['facts'] if f['facet'] == 'actor_excluded']
        if exclusions:
            interpretations.append({
                'status': 'INTERPRETIVE_COUNTEREXAMPLE_TEMPLATE_NOT_CERTIFIED_MODEL',
                'illustration': 'A control-seeking offensive involving only an expressly excluded opposing agency need not establish engagement with qualifying military forces. Whether such an incident meets every other source clause is unverified.',
                'requires_review_of_all_other_clauses_and_other_occurrences': True,
                'evidence': exclusions + [f for f in offensive['facts'] if f['facet'] in ('intention', 'geography')],
            })
    return {'schema': 'castle.gamma_rule_fact_comparison.v1', 'left': left, 'right': right,
        'dimension_comparisons': differences, 'interpretive_counterexamples': interpretations,
        'relationship_status': 'NOT_CERTIFIED', 'certified_relationships': [],
        'lp_constraint_admitted': False, 'broader_geopolitical_event_equivalence': 'NOT_ESTABLISHED',
        'physical_probability_P': None, 'equity_priced_in_q': None,
        'identified_event_conditional_return': None,
        'scope': 'LITERAL_SOURCE_FACT_DIFFERENCES_AND_REVIEW_PROMPTS_NOT_FORMAL_IMPLICATION',
        'new_provider_requests': 0, 'new_model_calls': 0}
