"""Generic clause mutations plus optional genuine two-contract source replay."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from castle_risk.gamma_rule_fact_audit import (
    audit_gamma_rule_text, audit_archived_gamma_rule_facts, compare_gamma_rule_facts,
)
from castle_risk.archived_nonmacro_case import P1_ARCHIVE_RELATIVE_PATH, load_recovered_p1_archive

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
OFFENSIVE = (
    'This market will resolve to "Yes" if Orion commences a military offensive intended to establish control '
    'over any portion of Lyra by December 31, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No".\n\n'
    'Territory under the administration of Lyra including any inhabited islands will qualify, however '
    'uninhabited islands will not qualify.\n\n'
    'The resolution source for this market will be a consensus of credible reporting.'
)
ENCOUNTER = (
    'This market will resolve to "Yes" if there is a military encounter between the military forces of Orion '
    'and Lyra between November 11, 2025, and December 31, 2026, 11:59 PM ET. '
    'Otherwise, this market will resolve to "No".\n\n'
    'A "military encounter" is defined as any incident involving the use of force between Orion and Lyra forces. '
    'Non-violent actions, such as warning shots or launches into empty waters, will not qualify for a "Yes" resolution. '
    'Intentional ship ramming that results in significant damage will count toward a "Yes" resolution, '
    'however minor damage will not.\n\n'
    'Note: the Orion Coast Guard is part of the military, however Lyra Civil Coast Guard is not.\n\n'
    'The resolution source for this market will be a consensus of credible reporting.'
)


def text_audit(rules=OFFENSIVE, **changes):
    args = dict(event_id='e-fixture', market_id='m-fixture', question='Synthetic exact question?', full_rules=rules)
    args.update(changes)
    return audit_gamma_rule_text(**args)


def archive(*, first=OFFENSIVE, second=ENCOUNTER, tamper=False, missing_id=False):
    events = []
    for index, rules in enumerate((first, second), 1):
        e = {'id': f'e{index}', 'slug': f'fixture-{index}', 'markets': [
            {'id': f'm{index}', 'question': f'Fixture {index}?', 'description': rules,
             'outcomes': '["Yes", "No"]'}]}
        if missing_id and index == 1:
            del e['id']
        events.append(e)
    raw = json.dumps({'events': events}).encode()
    rec = {'provider': 'polymarket', 'status': 200, 'sha256': sha256(raw).hexdigest(),
           'raw_file': 'raw/001_polymarket_fixture.json', 'completed_at_utc': '2026-10-17T01:30:00+00:00',
           'url': 'https://gamma-api.polymarket.com/events/keyset', 'bytes': len(raw),
           'record_count': 2, 'provenance_kind': 'MOCK_TEST_NOT_LIVE'}
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        z.writestr('CAPTURE_SUMMARY.json', '{}')
        z.writestr('request_ledger.jsonl', json.dumps(rec) + '\n')
        z.writestr(rec['raw_file'], raw + b'!' if tamper else raw)
    return output.getvalue()


def compare(raw=None):
    return compare_gamma_rule_facts(archive() if raw is None else raw,
        left_event_id='e1', left_market_id='m1', right_event_id='e2', right_market_id='m2', evaluated_at=NOW)


@pytest.mark.parametrize('rules', [OFFENSIVE, ENCOUNTER, '🧭 Audit context.\n\n' + ENCOUNTER])
def test_every_quote_indexes_original_unicode_rules_and_keeps_clause_scope(rules):
    result = text_audit(rules)
    assert result['facts']
    for fact in result['facts']:
        assert rules[fact['start']:fact['end']] == fact['quote']
        assert rules[fact['clause_start']:fact['clause_end']] == fact['clause_quote']
        assert fact['rules_sha256'] == sha256(rules.encode()).hexdigest()
    assert result['complete_semantic_model'] is False
    assert result['lp_constraint_admitted'] is False


def test_generic_actor_names_and_geographic_negation_are_not_id_shortcuts():
    result = text_audit()
    facts = {f['facet']: f for f in result['facts']}
    assert facts['actor']['quote'] == 'Orion'
    assert facts['geography_included']['quote'] == 'any inhabited islands'
    assert facts['geography_excluded']['quote'] == 'uninhabited islands'
    assert facts['geography_excluded']['polarity'] == 'negative'
    assert facts['intention']['quote'] == 'intended to establish control'
    assert 'time_start' not in facts
    assert result['unclassified_clauses'] == []


def test_agency_roles_and_intention_stay_scoped_to_their_exact_clauses():
    facts = {f['facet']: f for f in text_audit(ENCOUNTER)['facts']}
    assert facts['actor_included']['quote'] == 'Orion Coast Guard'
    assert facts['actor_excluded']['quote'] == 'Lyra Civil Coast Guard'
    assert facts['actor_excluded']['role'] == 'exclusion'
    assert facts['included_force']['quote'].startswith('Intentional ship ramming')
    assert 'intention' not in facts  # No whole-event intent inferred from ramming.
    assert facts['time_start']['quote'] == 'November 11, 2025'


def test_negation_flip_cannot_retain_a_positive_agency_inclusion():
    changed = ENCOUNTER.replace('is part of the military', 'is not part of the military')
    result = text_audit(changed)
    assert not any(f['facet'] == 'actor_included' for f in result['facts'])
    assert any('is not part' in c['quote'] for c in result['unclassified_clauses'])
    assert result['facts_status'].endswith('PARTIAL')


def test_proposition_negation_cannot_be_absorbed_into_a_positive_actor_label():
    changed = OFFENSIVE.replace('if Orion commences', 'if it is not true that Orion commences')
    result = text_audit(changed)
    assert any(f['facet'] == 'settlement_condition' for f in result['facts'])
    assert not any(f['facet'] in ('actor', 'action', 'intention') for f in result['facts'])
    assert 'UNSUPPORTED_ACTOR_OR_PROPOSITION_QUALIFIER' in result['parsing_ambiguities']
    assert result['facts_status'].endswith('PARTIAL')


def test_unknown_substantive_exception_remains_visible_and_prevents_full_recognition():
    extra = 'Any otherwise qualifying action performed solely by autonomous vehicles is excluded.'
    result = text_audit(ENCOUNTER + '\n\n' + extra)
    assert extra in [c['quote'] for c in result['unclassified_clauses']]
    assert result['facts_status'].endswith('PARTIAL')
    assert result['rules_sha256'] != text_audit(ENCOUNTER)['rules_sha256']


def test_duplicate_term_definition_is_preserved_and_flagged_not_overwritten():
    result = text_audit(ENCOUNTER + '\nA "military encounter" is defined as any nonviolent diplomatic contact.')
    definitions = [f for f in result['facts'] if f['facet'] == 'action_definition']
    assert len(definitions) == 2
    assert 'MULTIPLE_DEFINITIONS_OF_THE_SAME_TERM' in result['parsing_ambiguities']
    assert result['facts_status'].endswith('PARTIAL')


def test_caller_flags_and_pinned_digest_claim_cannot_authenticate_text_origin():
    from castle_risk.archived_nonmacro_case import P1_ARCHIVE_SHA256
    result = text_audit(source_metadata={'matches_recovered_capture_sha256': True,
        'source_origin': 'PINNED_EXISTING_CANDIDATE_CAPTURE', 'archive_sha256': P1_ARCHIVE_SHA256})
    assert result['source_origin'] == 'CALLER_SUPPLIED_TEXT_ORIGIN_NOT_REVERIFIED'
    assert result['raw_bytes_reverified_here'] is False
    with pytest.raises(ValueError, match='does not match'):
        text_audit(source_metadata={'rules_sha256': '0' * 64})


def test_rehashed_synthetic_archive_keeps_unverified_origin_and_exact_source_binding():
    raw = archive()
    result = audit_archived_gamma_rule_facts(raw, event_id='e1', market_id='m1', evaluated_at=NOW)
    assert result['source_origin'] == 'SUPPLIED_RECEIPT_CONSISTENT_ARCHIVE_ORIGIN_UNVERIFIED'
    assert result['raw_bytes_reverified_here'] is True
    assert result['source']['archive_sha256'] == sha256(raw).hexdigest()
    assert result['source']['matches_recovered_capture_sha256'] is False


def test_archive_tamper_future_evaluation_and_missing_original_identity_fail_closed():
    with pytest.raises(ValueError):
        audit_archived_gamma_rule_facts(archive(tamper=True), event_id='e1', market_id='m1', evaluated_at=NOW)
    with pytest.raises(ValueError, match='precedes'):
        audit_archived_gamma_rule_facts(archive(), event_id='e1', market_id='m1', evaluated_at='2026-10-16T00:00:00Z')
    with pytest.raises(ValueError, match='missing or ambiguous'):
        audit_archived_gamma_rule_facts(archive(missing_id=True), event_id='None', market_id='m1', evaluated_at=NOW)


def test_comparison_exposes_shared_deadline_missing_start_and_no_formal_relation():
    result = compare()
    by_facet = {d['facet']: d for d in result['dimension_comparisons']}
    assert by_facet['time_end']['comparison'] == 'SAME_EXPLICIT_TEXT'
    assert by_facet['time_start']['comparison'] == 'UNSPECIFIED_IN_LEFT'
    assert by_facet['intention']['comparison'] == 'UNSPECIFIED_IN_RIGHT'
    assert result['interpretive_counterexamples']
    assert all(c['status'].endswith('NOT_CERTIFIED_MODEL') for c in result['interpretive_counterexamples'])
    assert result['relationship_status'] == 'NOT_CERTIFIED'
    assert result['certified_relationships'] == []
    assert result['lp_constraint_admitted'] is False


def test_changed_deadline_is_not_normalized_to_false_equivalence():
    result = compare(archive(second=ENCOUNTER.replace('December 31, 2026', 'December 30, 2026')))
    deadline = next(d for d in result['dimension_comparisons'] if d['facet'] == 'time_end')
    assert deadline['comparison'] == 'DIFFERENT_EXPLICIT_TEXT'


def test_real_pinned_source_binds_both_taiwan_contracts_and_preserves_exceptions():
    if not (ROOT / P1_ARCHIVE_RELATIVE_PATH).is_file():
        pytest.skip('Private original P1 capture deliberately not distributed')
    result = compare_gamma_rule_facts(load_recovered_p1_archive(ROOT),
        left_event_id='34044', left_market_id='567621',
        right_event_id='79230', right_market_id='677407', evaluated_at=NOW)
    for item in (result['left'], result['right']):
        assert item['source_origin'] == 'PINNED_EXISTING_CANDIDATE_CAPTURE'
        assert item['raw_bytes_reverified_here'] is True
        assert item['source']['matches_recovered_capture_sha256'] is True
        assert item['unclassified_clauses'] == []
        for fact in item['facts']:
            assert item['full_rules'][fact['start']:fact['end']] == fact['quote']
    facts = {f['facet']: f for f in result['right']['facts']}
    assert facts['actor_included']['quote'] == 'China Coast Guard (CCG)'
    assert facts['actor_excluded']['quote'] == "Taiwan's Coast Guard Administration (CGA)"
    assert 'minor damage' in facts['excluded_force']['quote']
    assert result['certified_relationships'] == []
