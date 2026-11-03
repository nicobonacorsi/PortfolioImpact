"""Rechecking an old public export cannot invent missing inputs or new prices."""
from copy import deepcopy

from castle_risk.portfolio_impact_adapter import review_saved_case
from test_r32_saved_export import original
from test_r33_event_definition_review import YIELD, MEXICO, GDP


def test_saved_false_definition_pass_is_rechecked_without_new_model():
    case = original()
    template = deepcopy(case['risks'][-1])
    template.update(outcomes=[], admitted=[], market_outcomes=[], rejected=[], family='OTHER')
    template['quality']['definition_review'] = {'status': 'DEFINED_FOR_REVIEW', 'reason': ''}
    case['risks'] = [{**deepcopy(template), **definition, 'id': str(index)}
                     for index, definition in enumerate((YIELD, MEXICO, GDP))]
    checked = review_saved_case(case)
    assert {event['quality']['definition_review']['status'] for event in checked['risks']} == {
        'INCOMPLETE_PREDICATE', 'COMPOSITE_PREDICATE', 'UNCONFIRMED_RELEASE_SOURCE'}
    assert all(not event['quality']['model_admission']['primary_impact_allowed']
               for event in checked['risks'])
    assert checked['capture'] == case['capture']
    assert checked['presentation_version'] == 'R33'
    assert checked['presentation_review']['new_source_requests'] == 0
    assert checked['presentation_review']['numerical_refit'] is False
    assert review_saved_case(checked) == checked


def test_saved_no_response_metadata_requires_own_recorded_support():
    case = original()
    event = case['risks'][0]
    event['quality']['model_sensitivity'] = {'no_response_alternative_included': True}
    outcomes = event['outcomes'][:2]
    event['outcomes'] = outcomes
    outcomes[0]['pnl_model_sensitivity_range_usd'] = [-12., 34.]
    outcomes[1]['pnl_model_sensitivity_range_usd'] = [12., 34.]
    for row in outcomes:
        row.pop('no_response_alternative_included', None)
    before = deepcopy(case)
    checked = review_saved_case(case)
    changed = next(risk for risk in checked['risks'] if risk['id'] == event['id'])
    assert changed['outcomes'][0]['no_response_alternative_included'] is True
    assert 'no_response_alternative_included' not in changed['outcomes'][1]
    assert [row['pnl'] for row in changed['outcomes']] == [row['pnl'] for row in outcomes]
    assert case == before
    assert review_saved_case(checked) == checked
