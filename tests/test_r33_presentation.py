"""R33 source attribution and monitoring presentation; all inputs are synthetic.

These tests make no provider requests, write no report archives, and do not test
investment-model accuracy. Numerical input values must survive presentation.
"""
from copy import deepcopy

import pytest

from castle_risk.portfolio_impact_export import _quote_metadata
from test_r30_export import decoded, example
from test_r32_presentation import failed_case, visible


SCOPE = 'INDICATIVE_GAMMA_FIRST_OUTCOME_YES_SUMMARY_NOT_CLOB_VERIFIED'
RECEIVED = '2026-10-28T03:13:13.108827+00:00'


def source_case():
    case = example()
    event = case['risks'][0]
    event.update(id='source-china', family='TRADE_TARIFFS',
                 title='General US tariff rate on Chinese goods on 2026-12-31',
                 original_question='General US tariff rate on Chinese goods on 2026-12-31?',
                 date='2026-12-31', source_defined_scenario=True,
                 related_to_event_id='mexico-target')
    event['quality'].update(model_kind='MEASURED_ANCHOR',
                            model_status='COMPLETE_MEASURED_ANCHOR_MODEL',
                            transfer_status='NOT_TESTED',
                            reason='Separately modeled source-contract rate scenario.',
                            model_admission={'state': 'CONDITIONAL_SCENARIO',
                                             'primary_impact_allowed': True,
                                             'economic_ranking_allowed': False})
    event['outcomes'][0].update(contract_id='4853272', source_contract_ids=['4853272'],
                                venue='Polymarket',
                                question='Will the general China tariff rate be below 10% on December 31, 2026?',
                                quote_scope=SCOPE, quote_received_at=RECEIVED)
    event['admitted'] = [{key: value for key, value in event['outcomes'][0].items()
                          if key in {'venue', 'question', 'quote_scope', 'quote_received_at'}} |
                         {'id': '4853272'}]
    return case


def test_source_venue_question_indicative_scope_and_receipt_are_not_fabricated():
    case = source_case()
    before = deepcopy(case)
    result = decoded(case)
    row = result['rows'][0]
    assert row['source_venue'] == 'Polymarket'
    assert row['question'] == case['risks'][0]['outcomes'][0]['question']
    assert row['quote_scope'] == SCOPE
    assert row['quote_received_at'] == RECEIVED
    assert row['quote_qualification'] == 'Indicative Gamma summary; CLOB/order-book execution has not been verified.'
    assert 'Polymarket' in visible(result['html'])
    assert 'Kalshi' not in result['html']
    assert row['quote_qualification'] in visible(result['html'])
    assert 'Quote received: ' + RECEIVED in visible(result['html'])
    assert 'last trade' not in result['html'].lower()
    assert float(row['modeled_change_usd']) == case['risks'][0]['outcomes'][0]['pnl']
    assert case == before == result['case']


def test_missing_receipt_never_uses_the_case_capture_time():
    case = source_case()
    event = case['risks'][0]
    for item in [*event['outcomes'], *event['admitted']]:
        item.pop('quote_received_at')
    result = decoded(case)
    assert result['rows'][0]['quote_received_at'] == ''
    assert 'Quote retrieval time was not retained.' in visible(result['html'])
    assert 'Quote received: ' + case['capture'] not in result['html']


def test_matching_source_metadata_fallback_does_not_use_an_unrelated_contract():
    case = source_case()
    event = case['risks'][0]
    outcome = event['outcomes'][0]
    for key in ('venue', 'question', 'quote_scope', 'quote_received_at'):
        outcome.pop(key)
    event['admitted'].append({'id': 'UNRELATED', 'venue': 'Wrong venue',
                              'question': 'Different question', 'quote_received_at': 'WRONG'})
    quote = _quote_metadata(event, outcome)
    assert quote['venue'] == 'Polymarket'
    assert quote['question'].startswith('Will the general China tariff rate')
    assert quote['quote_received_at'] == RECEIVED
    assert 'WRONG' not in str(quote)


def test_no_observed_venue_remains_unspecified_instead_of_defaulting_to_kalshi():
    event = example()['risks'][0]
    event['admitted'] = []
    quote = _quote_metadata(event, event['outcomes'][0])
    assert quote['venue'] == quote['quote_received_at'] == quote['qualification'] == ''


@pytest.mark.parametrize('per_outcome,expected', [(None, True), (False, False), (True, True)])
def test_no_response_fallback_keeps_explicit_boolean_and_original_numbers(per_outcome, expected):
    case = example()
    event = case['risks'][0]
    outcome = event['outcomes'][0]
    outcome['pnl_model_sensitivity_range_usd'] = [-2000., 3000.]
    if per_outcome is not None:
        outcome['no_response_alternative_included'] = per_outcome
    event['quality']['model_sensitivity'] = {
        'status': 'AVAILABLE', 'no_response_alternative_included': True}
    before = deepcopy(case)
    result = decoded(case)
    assert result['rows'][0]['no_response_alternative_included'] == str(expected)
    assert ('Includes a no-response alternative' in result['html']) == expected
    assert float(result['rows'][0]['model_sensitivity_low_usd']) == -2000.
    assert float(result['rows'][0]['model_sensitivity_high_usd']) == 3000.
    assert case == before == result['case']


@pytest.mark.parametrize('bounds', [None, [-1], [1, -1], [True, 2], [1, 2], [-2, -1]])
def test_no_response_fallback_requires_an_existing_valid_range_containing_zero(bounds):
    case = example()
    event = case['risks'][0]
    event['outcomes'][0]['pnl_model_sensitivity_range_usd'] = bounds
    event['quality']['model_sensitivity'] = {'no_response_alternative_included': True}
    result = decoded(case)
    assert result['rows'][0]['no_response_alternative_included'] == 'False'
    assert 'Includes a no-response alternative' not in result['html']


def test_forward_link_does_not_assign_source_weights_or_dollars_to_original_target():
    case = source_case()
    source = case['risks'][0]
    original = deepcopy(source)
    original.update(id='mexico-target', title='New tariffs on Mexican goods by 2027-01-26',
                    original_question='Does the US impose new broad tariffs on Mexico by 2027-01-26?',
                    date='2027-01-26', source_defined_scenario=False, outcomes=[], admitted=[],
                    related_source_scenario_ids=['source-china', 'missing-id'])
    original['quality'] = {'model_kind': 'UNAVAILABLE',
                           'model_admission': {'state': 'RELATED_SOURCE_ONLY',
                                               'primary_impact_allowed': False,
                                               'economic_ranking_allowed': False,
                                               'reason': 'Country and settlement question differ.'}}
    case['risks'] = [original, source]
    before = deepcopy(case)
    result = decoded(case)
    target_html = result['html'].split('<section class="event">')[1].split('</section>')[0]
    assert 'A related, different question is available.' in target_html
    assert "does not supply this target's probability or impact" in target_html
    assert 'href="#scenario-2"' in target_html
    assert source['title'] in target_html
    assert '<th>Market-implied weight</th>' not in target_html and '20.0–30.0%' not in target_html
    assert '−$1,531' not in target_html and '−$9' not in target_html
    assert result['rows'][0]['modeled_change_usd'] == ''
    assert result['rows'][1]['date'] == '2026-12-31'
    assert result['rows'][1]['unavailable_reason'] == ''
    assert result['rows'][1]['model_scope_note'] == source['quality']['reason']
    assert case == before == result['case']


def test_forward_link_requires_an_explicit_separate_source_scenario():
    case = example()
    case['risks'][0]['related_source_scenario_ids'] = ['test']
    assert 'A related, different question is available.' not in decoded(case)['html']


def test_watchlist_is_closed_research_only_and_does_not_create_outcomes_or_numbers():
    case = example()
    case['risk_watchlist'] = [{
        'risk_id': 'labor-risk', 'title': 'Railway labor disruption',
        'monitoring_signal': 'Strike authorization or a bargaining deadline.',
        'exposure_rationale': 'Possible interruption to UNP operations.',
        'exposed_positions': [{'symbol': 'UNP', 'weight': .25}],
        'status': 'UNVERIFIED_MONITORING_HYPOTHESIS',
        'probability': .98765, 'pnl': -987654321, 'portfolio_impact_usd': -987654321}]
    before = deepcopy(case)
    result = decoded(case)
    fragment = result['html'].split('<details class="business-watchlist">')[1].split('</details>')[0]
    assert '1 unverified research leads' in fragment
    assert 'Research hypotheses, not confirmed events or priced scenarios.' in fragment
    assert 'No market probability or portfolio-impact estimate is assigned.' in fragment
    assert 'Railway labor disruption' in fragment and 'Strike authorization' in fragment
    assert 'Relevant holdings: UNP' in fragment
    assert 'Railway labor disruption' not in visible(result['html'])
    assert '98765' not in fragment and '$' not in fragment
    assert len(result['rows']) == len(case['risks']) == 1
    assert result['names'] == {'report.html', 'scenario_results.csv', 'portfolio_data.json', 'README.txt', 'past_episode_replay.csv'}
    assert case == before == result['case']


def test_source_and_watchlist_text_are_html_escaped():
    case = source_case()
    attack = '<script>alert("test")</script>'
    case['risks'][0]['outcomes'][0]['question'] = attack
    case['risk_watchlist'] = [{'title': attack, 'monitoring_signal': attack,
                              'exposure_rationale': attack,
                              'exposed_positions': [{'symbol': attack}]}]
    result = decoded(case)
    assert attack not in result['html']
    assert '&lt;script&gt;alert(&quot;test&quot;)&lt;/script&gt;' in result['html']
    assert result['rows'][0]['question'] == attack


def test_failed_payroll_diagnostic_stays_closed_with_source_metadata():
    case = failed_case()
    outcome = case['risks'][0]['outcomes'][0]
    outcome.update(venue='Kalshi', question='Exact synthetic payroll question',
                   quote_received_at=RECEIVED)
    result = decoded(case)
    assert result['rows'][0]['model_admission_state'] == 'DIAGNOSTIC_ONLY'
    assert 'Impact model did not pass' in visible(result['html'])
    for value in ('−$4,321', '−$6,543', '+$1,234', '−$8,000', '+$8,000'):
        assert value in result['html'] and value not in visible(result['html'])


@pytest.mark.parametrize('version', ['R32', 'R33'])
def test_saved_review_notice_accepts_old_and_current_review_versions(version):
    case = example()
    case['presentation_review'] = {'status': 'PUBLIC_EXPORT_REVIEW_ONLY_' + version}
    result = decoded(case)
    assert 'Saved export reviewed locally; original numbers and quote date retained. No new fit or source requests.' in result['html']
    assert case == result['case']
