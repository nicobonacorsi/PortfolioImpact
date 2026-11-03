"""Portable, shareable reports from the public Portfolio Impact case only.

No provider archives, private acquisition objects or credentials enter these
exports. Every quantitative value is copied from the assessed public case.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
from io import BytesIO, StringIO
import json
import math
from zipfile import ZIP_DEFLATED, ZipFile

from .model_admission import event_model_admission
from .portfolio_impact_adapter import _source_only_inspection_view
from .business_interruption_assessment import component_view
from .mexico_tariff_transmission import mexico_component_view
from .priced_in_sensitivity import admitted_binary_contract_sensitivity


def _pct(value, places=1):
    return f"{100 * value:.{places}f}%" if _finite(value) else "—"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _model_range(outcome):
    values = (outcome or {}).get('pnl_interval')
    if (isinstance(values, (list, tuple)) and len(values) == 2
            and all(_finite(x) for x in values) and values[0] <= values[1]):
        return list(values)
    return None


def _sensitivity_range(outcome):
    values = (outcome or {}).get('pnl_model_sensitivity_range_usd')
    if (isinstance(values, (list, tuple)) and len(values) == 2
            and all(_finite(x) for x in values) and values[0] <= values[1]):
        return list(values)
    return None


def _admission(event):
    """Hard scope blocks outrank diagnostics; never restore a stale primary."""
    quality = event.get('quality') or {}
    raw = quality.get('model_admission')
    reported = dict(raw) if isinstance(raw, dict) else {}
    states = {'EXACT_IDENTITY', 'CONDITIONAL_SCENARIO', 'DIAGNOSTIC_ONLY',
              'RELATED_SOURCE_ONLY', 'UNAVAILABLE'}
    unavailable = {'state': 'UNAVAILABLE', 'primary_impact_allowed': False,
                   'economic_ranking_allowed': False,
                   'reason': 'The saved model scope is incomplete or incompatible; its impact is not admitted.'}
    if raw is not None and (not isinstance(raw, dict) or reported.get('state') not in states):
        return unavailable
    if reported.get('state') in {'UNAVAILABLE', 'RELATED_SOURCE_ONLY'}:
        return {**reported, 'primary_impact_allowed': False, 'economic_ranking_allowed': False}
    derived = event_model_admission(event)
    if (derived['state'] in {'UNAVAILABLE', 'RELATED_SOURCE_ONLY'} and
            derived['evidence_scope'] != 'NO_ADMITTED_COMPLETE_RESPONSE'):
        return derived
    definition = quality.get('definition_review') or {}
    if definition.get('status') not in (None, '', 'DEFINED_FOR_REVIEW'):
        return {**unavailable, 'reason': definition.get('reason') or unavailable['reason']}
    expected_status = {'CHANNEL_AWARE': 'COMPLETE_DECLARED_MODEL',
                       'MEASURED_PAYROLL': 'COMPLETE_MEASURED_SURPRISE_RESPONSE',
                       'LEGACY_BETA': 'DECLARED_DEFAULT_COMPONENT_MODEL',
                       'TARIFF_ANCHOR': 'COMPLETE_MEASURED_ANCHOR_MODEL',
                       'MEASURED_ANCHOR': 'COMPLETE_MEASURED_ANCHOR_MODEL',
                       'NOMINAL_CASH_IDENTITY': 'IDENTIFIED_NOMINAL_USD_CASH_ZERO',
                       'DECLARED_CONTRACT_SCENARIO': 'DECLARED_CONTRACT_SCENARIO'}
    kind, status = quality.get('model_kind'), quality.get('model_status')
    if ((kind is not None and not isinstance(kind, str)) or
            (status is not None and not isinstance(status, str))):
        return unavailable
    if (kind and kind not in expected_status or
            status and status != expected_status.get(kind)):
        return unavailable
    public_contract = kind == 'DECLARED_CONTRACT_SCENARIO'
    if public_contract:
        # The narrow archived-contract path only applies to public, source-
        # identified outcome rows. A valid row can still fail its transfer
        # check: preserve that diagnostic decision rather than relabel it.
        raw_inputs = any(isinstance(event.get(key), dict) and event[key]
                         for key in ('outcome_table', 'economics'))
        if raw_inputs or derived['state'] not in {'CONDITIONAL_SCENARIO', 'DIAGNOSTIC_ONLY'}:
            return derived if derived['state'] in {'UNAVAILABLE', 'RELATED_SOURCE_ONLY'} else unavailable
        if (derived['state'] == 'CONDITIONAL_SCENARIO' and
                derived['evidence_scope'] != 'SOURCE_IDENTIFIED_CONTRACT_STRESS_SCENARIO'):
            return unavailable
    if reported.get('state') == 'EXACT_IDENTITY' and derived['state'] != 'EXACT_IDENTITY':
        return unavailable
    if reported.get('state') == 'DIAGNOSTIC_ONLY':
        return {**reported, 'primary_impact_allowed': False, 'economic_ranking_allowed': False}
    numeric = any(_finite(row.get('pnl')) for row in event.get('outcomes') or [] if isinstance(row, dict))
    if public_contract and derived['state'] == 'DIAGNOSTIC_ONLY':
        return derived
    if numeric and (quality.get('transfer_status') == 'FAIL' or quality.get('portfolio_transfer_status') == 'FAIL'):
        return {**reported, **derived, 'state': 'DIAGNOSTIC_ONLY', 'primary_impact_allowed': False,
                'economic_ranking_allowed': False,
                'reason': (derived['reason'] if derived['state'] == 'DIAGNOSTIC_ONLY' else
                           'The impact model did not pass its reported historical check. '
                           'Its numbers are retained only as diagnostic evidence.')}
    if public_contract:
        return {**reported, **derived,
                'primary_impact_allowed': reported.get('primary_impact_allowed') is True if reported else True,
                'economic_ranking_allowed': False}
    if reported:
        if (reported.get('primary_impact_allowed') is True and
                (not numeric or kind not in expected_status)):
            return unavailable
        return {**reported, 'primary_impact_allowed': reported.get('primary_impact_allowed') is True,
                'economic_ranking_allowed': reported.get('economic_ranking_allowed') is True}
    if derived['evidence_scope'] == 'NO_ADMITTED_COMPLETE_RESPONSE' and quality.get('reason'):
        return {**derived, 'reason': quality['reason']}
    return derived


def _unavailable_explanation(event, admission, fallback=''):
    """Show the hard gate and recorded cause without altering either decision.

    A generic admission category must not erase a specific acquisition or
    settlement failure. Recorded details are subordinate to the hard gate;
    they do not grant admission or repair an event definition.
    """
    quality = event.get('quality') or {}
    definition = quality.get('definition_review') or {}
    reasons = []
    definition_reason = definition.get('reason') if definition.get('status') not in (None, '', 'DEFINED_FOR_REVIEW') else None
    for value in (admission.get('reason'), definition_reason, quality.get('reason')):
        if isinstance(value, str) and value.strip() and value.strip() not in reasons:
            reasons.append(value.strip())
    if not reasons:
        return fallback
    return reasons[0] + (' Recorded detail: ' + ' '.join(reasons[1:]) if len(reasons) > 1 else '')


def _event_outcomes(event):
    """Preserve recorded numeric rows and independently available market rows."""
    rows = list(event.get('outcomes') or [])
    seen = {str(row.get('id')) for row in rows}
    for row in event.get('market_outcomes') or []:
        if str(row.get('id')) not in seen:
            rows.append(row)
            seen.add(str(row.get('id')))
    return rows


def _unmatched_source_only(event, admission):
    """A stale explicit hard gate cannot attest a tariff source question/date."""
    return event.get('source_defined_scenario') is not True and (
        (event.get('quality') or {}).get('model_kind') in {'MEASURED_ANCHOR', 'TARIFF_ANCHOR'}
        or (event.get('outcome_table') or {}).get('status') == 'COMPLETE_MEASURED_ANCHOR_MODEL'
        or admission.get('evidence_scope') == 'UNATTESTED_SEPARATE_TARIFF_SOURCE_SCOPE')


def _no_response_alternative(event, outcome):
    value = outcome.get('no_response_alternative_included')
    if isinstance(value, bool):
        return value
    sensitivity = (event.get('quality') or {}).get('model_sensitivity') or {}
    bounds = _sensitivity_range(outcome)
    return (bounds is not None and bounds[0] <= 0 <= bounds[1]
            and sensitivity.get('no_response_alternative_included') is True)


def _quote_metadata(event, outcome):
    identifiers = set(outcome.get('source_contract_ids') or [])
    if outcome.get('contract_id'):
        identifiers.add(outcome['contract_id'])
    sources = [row for row in _contracts(event.get('admitted'))
               if not identifiers or row['id'] in identifiers]
    records = [outcome, *sources]
    def values(key):
        return list(dict.fromkeys(row[key].strip() for row in records
                                  if isinstance(row.get(key), str) and row[key].strip()))
    quality = event.get('quality') or {}
    scopes = values('quote_scope')
    if not scopes and quality.get('quote_scope'):
        scopes = [str(quality['quote_scope'])]
    indicative = any('INDICATIVE' in value.upper() or 'NOT_CLOB_VERIFIED' in value.upper()
                     for value in scopes)
    venues = values('venue')
    if not venues and quality.get('source_venue'):
        venues = [str(quality['source_venue'])]
    questions = values('question')
    question = (str(outcome.get('question') or '').strip() or
                (questions[0] if len(questions) == 1 else '') or
                (str(event.get('original_question') or '') if event.get('source_defined_scenario') else ''))
    return {'venue': '; '.join(venues), 'quote_scope': '; '.join(scopes), 'question': question,
            'quote_received_at': '; '.join(values('quote_received_at')) or quality.get('quote_received_at') or '',
            'qualification': ('Indicative Gamma summary; CLOB/order-book execution has not been verified.'
                              if indicative else '')}


def _watchlist_html(case):
    rows = [row for row in case.get('risk_watchlist') or [] if isinstance(row, dict)
            and isinstance(row.get('title'), str) and row['title'].strip()]
    if not rows:
        return ''
    e = lambda value: escape(str(value if value is not None else ''), quote=True)
    entries = []
    for row in rows:
        symbols = [str(item['symbol']) for item in row.get('exposed_positions') or []
                   if isinstance(item, dict) and item.get('symbol')]
        entries.append('<li><strong>' + e(row['title']) + '</strong>' +
                       ('<p>Watch for: ' + e(row['monitoring_signal']) + '</p>' if row.get('monitoring_signal') else '') +
                       ('<p>' + e(row['exposure_rationale']) + '</p>' if row.get('exposure_rationale') else '') +
                       ('<small>Relevant holdings: ' + e(' · '.join(symbols)) + '</small>' if symbols else '') + '</li>')
    return ('<details class="business-watchlist"><summary>Business risks to monitor · ' + str(len(rows)) +
            ' unverified research leads</summary><p class="scope">Research hypotheses, not confirmed events '
            'or priced scenarios. No market probability or portfolio-impact estimate is assigned.</p><ul>' +
            ''.join(entries) + '</ul></details>')


def _web_research_html(case):
    from .live_web_research import public_web_research
    data = public_web_research(case.get('web_research'))
    if not data['sources']:
        return ''
    e = lambda value: escape(str(value if value is not None else ''), quote=True)
    items = ''.join('<li><a href="' + e(item['url']) + '">' + e(item['title'] or item['domain']) +
                    '</a> <small>(' + e(item['domain']) + ')</small></li>'
                    for item in data['sources'])
    return ('<details class="business-watchlist"><summary>Online risk research · ' +
            str(len(data['sources'])) + ' result links</summary>' +
            '<p class="scope">Public web search leads used for discovery, not independently validated '
            'facts, source-defined event probabilities or event-conditioned returns.</p><ul>' +
            items + '</ul></details>')


def _direction(outcome):
    bounds = _model_range(outcome)
    if bounds is None:
        return 'NOT_ESTABLISHED_WITHOUT_MODEL_RANGE'
    if bounds[0] < 0 < bounds[1]:
        return 'DIRECTION_UNCERTAIN'
    if bounds[0] == bounds[1] == 0:
        return 'ZERO_UNDER_STATED_MODEL'
    return 'NONNEGATIVE_UNDER_STATED_MODEL' if bounds[0] >= 0 else 'NONPOSITIVE_UNDER_STATED_MODEL'


def _range_meaning(outcome):
    if (outcome or {}).get('pnl_interval_interpretation'):
        return str(outcome['pnl_interval_interpretation'])
    if (outcome or {}).get('pnl_interval_kind') == 'MODEL_CONDITIONAL_5_95_ENVELOPE':
        return ('Model-conditional 5–95% envelope over parameter and market-weight uncertainty; '
                'not a future-return prediction interval.')
    return ('Range under the recorded model assumptions; not guaranteed bounds on '
            "the portfolio's future return.")


def _transfer_status(quality):
    return str((quality or {}).get('transfer_status') or 'NOT_REPORTED')


def _check_label(status):
    return {'PASS': 'Passed association check', 'FAIL': 'Did not pass',
            'NOT_TESTED': 'Not independently tested', 'NOT_REPORTED': 'Not reported'}.get(
                status, str(status or 'Not reported').replace('_', ' '))


def _model_label(kind):
    return {'CHANNEL_AWARE': 'Equity + interest-rate channels',
            'MEASURED_PAYROLL': 'Measured jobs-surprise response',
            'LEGACY_BETA': 'Legacy market-beta scenario',
            'NOMINAL_CASH_IDENTITY': 'Fixed USD cash identity'}.get(kind, 'Recorded scenario model')


def _reported_failure_share(quality):
    status = _transfer_status(quality)
    if status not in {'PASS', 'FAIL'}:
        return None
    if status == 'FAIL' and (quality.get('failure_scope') == 'PORTFOLIO_ONLY' or not quality.get('failed_symbols')):
        return None
    value = quality.get('failed_portfolio_weight')
    return value if _finite(value) else None


def _failure_description(quality):
    symbols = quality.get('failed_symbols') or []
    portfolio = (quality.get('portfolio_transfer_status') == 'FAIL' or
                 quality.get('failure_scope') in {'PORTFOLIO_ONLY', 'PORTFOLIO_AND_HOLDINGS'})
    prefix = 'Portfolio-level chronological check failed. ' if portfolio else ''
    if not symbols:
        return prefix or 'Historical transfer check failed; holding-level attribution was not reported. '
    weight = _reported_failure_share(quality)
    scope = ' (' + _pct(weight, 0) + ' of portfolio)' if weight is not None and weight > 0 else ''
    return prefix + 'Historical transfer check failed for ' + ', '.join(map(str, symbols)) + scope + '. '


def _contracts(rows):
    """Search notes are not contracts; preserve only distinct identified records."""
    result, seen = [], set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get('id') or '').strip()
        key = (str(row.get('venue') or ''), identifier)
        if not identifier or identifier.lower() in {'unmatched question', 'candidate'} or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _review_notes(event):
    legacy_notes = [r for r in event.get('rejected') or [] if isinstance(r, dict) and not _contracts([r])]
    notes, seen = [], set()
    for row in [*(event.get('review_notes') or []), *legacy_notes]:
        if not isinstance(row, dict):
            continue
        reason = str(row.get('reason') or row.get('decision') or '')
        if reason and reason not in seen:
            notes.append(reason)
            seen.add(reason)
    return notes


def _source_only_inspection_html(row):
    source = _source_only_inspection_view(row)
    if source is None:
        return ''
    e = lambda value: escape(str(value or ''), quote=True)
    interval = source['quote_interval']
    quote = ('Source-only YES quote at recorded evaluation: ' + _pct(interval[0]) + ' to ' + _pct(interval[1])
             if interval is not None else 'Source-only YES quote unavailable at the recorded evaluation time.')
    scope = ('Historical quote at source receipt; not current.'
             if source['quote_scope'].startswith('HISTORICAL_') else
             'Quote checked at the recorded evaluation time; not a live refresh.')
    return ('<div class="source-only-inspection"><strong>Recorded exact-contract inspection · Kalshi</strong>'
            '<p>' + e(source['question']) + '</p><p>' + e(quote) + '</p>'
            '<p class="scope">This exact contract quote is not the probability of your target '
            'and does not identify a portfolio impact.</p><small>' + e(scope) + ' Received: ' +
            e(source['quote_received_at']) + ' · Evaluated: ' + e(source['quote_evaluated_at']) + '</small>' +
            ('<p>' + e(source['quote_unavailable_reason']) + '</p>' if interval is None else '') +
            '<details><summary>Recorded source provenance</summary><small>Response SHA-256: ' +
            e(source['response_sha256']) + '<br>Full-rules SHA-256: ' + e(source['rules_sha256']) +
            '</small><p><a href="' + e(source['source_url']) + '" rel="noopener noreferrer">Public contract source</a>'
            '</p><small>Recorded local receipt-byte check, not a provider signature or new independent '
            'verification of this exported file.</small></details></div>')


def _comparison_columns(quality):
    models = {m.get('id'): m for m in quality.get('model_comparisons') or [] if isinstance(m, dict)}
    result = {}
    for prefix, identifier in (('daily', 'economics'), ('announcement', 'announcement_day_economics')):
        model = models.get(identifier) or {}
        status = _transfer_status(model)
        result.update({
            prefix + '_transfer_status': status,
            prefix + '_failed_symbols': '; '.join(model.get('failed_symbols') or []),
            prefix + '_failed_portfolio_share': _reported_failure_share(model),
        })
    return result


def _money(value):
    if not _finite(value):
        return "—"
    return f"{'−' if value < 0 else '+' if value > 0 else ''}${abs(value):,.0f}"


def _chance(outcome):
    interval = outcome.get('q_interval')
    if isinstance(interval, list) and len(interval) == 2 and all(_finite(x) for x in interval):
        start, end = _pct(interval[0]), _pct(interval[1])
        return start if start == end else f'{start}–{end}'
    return _pct(outcome.get('q_reference'))


def _safe_cell(value):
    """Keep external question text inert if a CSV is opened in Excel."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    item = str(value if value is not None else '')
    if item.lstrip().startswith(('=', '+', '-', '@')) or item[:1] in ('\t', '\r', '\n'):
        return "'" + item
    return item


def _identity_notice(case):
    review = case.get('instrument_identity') or {}
    blocked = review.get('blocked_symbols') or []
    if blocked:
        return ('Investment type needs review: ' + ', '.join(map(str, blocked)) +
                '. Affected portfolio estimates remain unavailable until source identity is resolved.')
    names = {'stock': 'company stock', 'etf': 'investment fund'}
    checked = [row for row in review.get('positions') or [] if isinstance(row, dict)
               and row.get('local_response_verified') is True
               and row.get('resolved_asset_type') in names
               and row.get('declared_asset_type') != row.get('resolved_asset_type')]
    if not checked:
        return ''
    return 'Source-checked types: ' + ' · '.join(
        str(row.get('symbol')) + ' — ' + names[row['resolved_asset_type']] +
        (' (corrected)' if row.get('declared_asset_type') != 'auto' else '')
        for row in checked) + '.'


def _rows(case):
    for event in case.get('risks', []):
        quality = event.get('quality') or {}
        coverage = quality.get('distribution_display_coverage') or {}
        definition = quality.get('definition_review') or {}
        status = _transfer_status(quality)
        admission = _admission(event)
        sensitivity = quality.get('model_sensitivity') or {}
        details = {
            **_comparison_columns(quality),
            'instrument_identity_status': (case.get('instrument_identity') or {}).get('status') or 'NOT_REPORTED',
            'instrument_type_note': _identity_notice(case),
            'identity_blocked_symbols': '; '.join((quality.get('instrument_identity_gate') or {}).get('blocked_symbols') or []),
            'event': event.get('title', ''),
            'date': event.get('date', ''),
            'family': event.get('family', ''),
            'verification': status,
            'verification_scope': 'EVENT_LEVEL_REPEATED_ACROSS_OUTCOMES',
            'portfolio_transfer_status': quality.get('portfolio_transfer_status') or 'NOT_REPORTED',
            'failure_scope': quality.get('failure_scope') or 'NONE_OR_NOT_REPORTED',
            'failed_symbols': '; '.join(quality.get('failed_symbols') or []),
            'unvalidated_portfolio_share': _reported_failure_share(quality),
            'omitted_market_weight': coverage.get('omitted_reference_mass'),
            'definition_status': definition.get('status', ''),
            'contracts_used': len(_contracts(event.get('admitted'))),
            'contracts_excluded': len(_contracts(event.get('rejected'))),
            'model_kind': quality.get('model_kind') or 'NOT_REPORTED',
            'model_admission_state': admission.get('state') or 'UNAVAILABLE',
            'primary_impact_allowed': admission.get('primary_impact_allowed') is True,
            'economic_ranking_allowed': admission.get('economic_ranking_allowed') is True,
            'model_admission_reason': admission.get('reason') or '',
            'model_evidence_scope': admission.get('evidence_scope') or '',
            'model_scope_note': quality.get('reason') or '',
            'source_defined_scenario': event.get('source_defined_scenario') is True,
            'related_to_event_id': event.get('related_to_event_id') or '',
            'related_to_event_ids': '; '.join(dict.fromkeys([
                *([event['related_to_event_id']] if event.get('related_to_event_id') else []),
                *(event.get('related_to_event_ids') or [])])),
            'model_sensitivity_kind': sensitivity.get('kind') or '',
            'model_comparison_checks': '; '.join(
                str(m.get('label') or m.get('id') or 'Comparison model') + ': ' +
                str(m.get('transfer_status') or 'NOT_REPORTED') +
                ((' (flagged: ' + ', '.join(m['failed_symbols']) + ')') if m.get('failed_symbols') else '')
                for m in quality.get('model_comparisons') or [] if isinstance(m, dict)),
            'review_notes': '; '.join(_review_notes(event)),
            'excluded_source_inspections': json.dumps([
                source for row in _contracts(event.get('rejected'))
                if (source := _source_only_inspection_view(row)) is not None
            ], ensure_ascii=False, allow_nan=False, separators=(',', ':')),
            'business_interruption_component': json.dumps(component_view(
                event.get('business_interruption_component'), event_id=event.get('id'),
                portfolio_notional=case.get('notional'), portfolio_positions=case.get('positions')),
                ensure_ascii=False, allow_nan=False, separators=(',', ':')),
            'mexico_tariff_component': json.dumps(mexico_component_view(
                event.get('mexico_tariff_component'), event_id=event.get('id'),
                portfolio_notional=case.get('notional'), portfolio_positions=case.get('positions')),
                ensure_ascii=False, allow_nan=False, separators=(',', ':')),
            'unavailable_reason': (_unavailable_explanation(event, admission)
                                   if admission.get('primary_impact_allowed') is not True else ''),
        }
        for outcome in _event_outcomes(event) or [None]:
            outcome = outcome or {}
            bounds = _model_range(outcome)
            stress = _sensitivity_range(outcome)
            quote = _quote_metadata(event, outcome)
            yield {**details, 'outcome': outcome.get('label', ''),
                   'chance': _chance(outcome),
                   'probability_kind': ('Q_UNMATCHED_SOURCE_NOT_ORIGINAL_TARGET_PROBABILITY'
                                        if _unmatched_source_only(event, admission)
                                        else 'Q_PRICE_IMPLIED_NOT_PHYSICAL_EVENT_PROBABILITY'),
                   'modeled_change_usd': outcome.get('pnl'),
                   'model_range_low_usd': bounds[0] if bounds else None,
                   'model_range_high_usd': bounds[1] if bounds else None,
                   'model_range_kind': outcome.get('pnl_interval_kind') or ('RECORDED_MODEL_RANGE' if bounds else ''),
                   'model_range_interpretation': _range_meaning(outcome) if bounds else '',
                   'model_sensitivity_low_usd': stress[0] if stress else None,
                   'model_sensitivity_high_usd': stress[1] if stress else None,
                   'model_sensitivity_interpretation': (
                       'Separate stress range across stated model assumptions; not a 5-95% parameter range, '
                       'prediction band, or empirical validation.' if stress else ''),
                   'no_response_alternative_included': _no_response_alternative(event, outcome),
                   'direction': _direction(outcome) if outcome else '',
                   'modeled_change_pct': (
                       round(outcome['pnl'] / case['notional'] * 100, 5)
                       if _finite(outcome.get('pnl')) and case.get('notional') else None),
                   'contract_id': outcome.get('contract_id') or '',
                   'source_contract_ids': '; '.join(outcome.get('source_contract_ids') or []),
                   'question': quote['question'], 'source_venue': quote['venue'],
                   'quote_scope': quote['quote_scope'], 'quote_received_at': quote['quote_received_at'],
                   'quote_qualification': quote['qualification']}


def _csv_export(rows):
    out = StringIO(newline='')
    fieldnames = ('event', 'date', 'family', 'outcome', 'chance', 'probability_kind', 'modeled_change_usd',
                  'model_range_low_usd', 'model_range_high_usd', 'model_range_kind',
                  'model_range_interpretation', 'direction', 'model_kind',
                  'modeled_change_pct', 'verification', 'verification_scope',
                  'portfolio_transfer_status', 'failure_scope',
                  'failed_symbols', 'unvalidated_portfolio_share',
                  'omitted_market_weight', 'definition_status', 'contracts_used',
                  'contracts_excluded', 'contract_id', 'source_contract_ids', 'question',
                  'model_comparison_checks', 'daily_transfer_status', 'daily_failed_symbols',
                  'daily_failed_portfolio_share', 'announcement_transfer_status',
                  'announcement_failed_symbols', 'announcement_failed_portfolio_share',
                  'review_notes', 'unavailable_reason', 'instrument_identity_status',
                  'instrument_type_note', 'identity_blocked_symbols',
                  'model_admission_state', 'primary_impact_allowed', 'economic_ranking_allowed',
                  'model_admission_reason', 'model_evidence_scope', 'model_scope_note', 'source_defined_scenario',
                  'related_to_event_id', 'related_to_event_ids', 'model_sensitivity_low_usd', 'model_sensitivity_high_usd',
                  'model_sensitivity_kind', 'model_sensitivity_interpretation',
                  'no_response_alternative_included', 'source_venue', 'quote_scope',
                  'quote_received_at', 'quote_qualification', 'excluded_source_inspections',
                  'business_interruption_component', 'mexico_tariff_component')
    writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator='\r\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_cell(row.get(key)) for key in fieldnames})
    return out.getvalue()


def _pricing_weight_sensitivity_html(event):
    """Separate unmeasured equity-state-price weight from contract quote Q."""
    if _admission(event).get('primary_impact_allowed') is not True:
        return ''
    study = admitted_binary_contract_sensitivity(event)
    if not study:
        return ''
    body = ''.join(
        '<tr><td>' + _pct(row['assumed_pi'], 0) + '</td>' +
        '<td>' + _money(row['yes_pnl_usd']) + '</td>' +
        '<td>' + _money(row['no_pnl_usd']) + '</td></tr>'
        for row in study['point_grid'])
    return (
        '<details class="pricing-weight-sensitivity"><summary>What if equity prices already reflect '
        'more (or less) of this event?</summary>'
        '<p class="scope">Assumed effective equity pricing weight π, NOT the physical event '
        'probability and NOT the prediction-market quote Q. The exact source contract has '
        'Q = ' + _pct(study['contract_quote_q_reference'], 2) +
        ', but this does not identify π for the portfolio. In an explicit two-state '
        'discounted-value model, YES = (1−π)Δ and NO = −πΔ, where the model-assumed '
        'YES-minus-NO portfolio contrast Δ = ' + _money(study['contrast_usd']) + '.</p>'
        '<table><thead><tr><th>Assumed π</th><th>If YES</th><th>If NO</th></tr></thead>'
        '<tbody>' + body + '</tbody></table>'
        '<p class="scope">With no justified restriction beyond 0 ≤ π ≤ 1, the sharp '
        'model-conditional YES range is ' + _money(study['yes_pnl_usd_range'][0]) +
        ' to ' + _money(study['yes_pnl_usd_range'][1]) +
        '; NO is ' + _money(study['no_pnl_usd_range'][0]) +
        ' to ' + _money(study['no_pnl_usd_range'][1]) +
        '. These are assumption sensitivity bounds, NOT calibrated empirical '
        'confidence intervals or forecasts. This contract does not price its '
        'broader parent geopolitical event.</p></details>')


def _sensitivity_html(event):
    e = lambda value: escape(str(value if value is not None else ''), quote=True)
    sensitivity = (event.get('quality') or {}).get('model_sensitivity') or {}
    rows = []
    for outcome in _event_outcomes(event):
        bounds = _sensitivity_range(outcome)
        if bounds is None:
            continue
        rows.append('<tr><td>' + e(outcome.get('label')) + '</td><td>' +
                    e(_money(bounds[0]) + ' to ' + _money(bounds[1])) + '</td><td>' +
                    ('Includes a no-response alternative' if _no_response_alternative(event, outcome)
                     else 'Recorded assumptions only') + '</td></tr>')
    if not rows:
        return ('<h3>Model-assumption sensitivity</h3><p class="scope">' + e(sensitivity['reason']) +
                '</p>') if sensitivity.get('reason') else ''
    notes = [*(sensitivity.get('assumptions') or []), *(sensitivity.get('limitations') or [])]
    content = ('<h3>Model-assumption sensitivity</h3><p class="scope">Separate stress range across the '
               'stated model assumptions. This is not the 5–95% parameter range, a prediction band, '
               'or evidence of predictive accuracy. A no-response alternative is not selected as '
               'the best forecast.</p><table><thead><tr><th>Outcome</th><th>Assumption stress range</th>'
               '<th>Scope</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>' +
               ('<ul>' + ''.join('<li>' + e(note) + '</li>' for note in notes) + '</ul>' if notes else ''))
    return content if _admission(event).get('primary_impact_allowed') is True else (
        '<details class="diagnostic-evidence"><summary>Diagnostic model-assumption sensitivity</summary>' +
        content + '</details>')


def _business_component_html(event, case):
    v = component_view(event.get('business_interruption_component'), event_id=event.get('id'),
                       portfolio_notional=case.get('notional'), portfolio_positions=case.get('positions'))
    if v is None:
        return ''
    money = lambda value: '$' + format(abs(value), ',.2f')
    rows = ''.join('<tr><td>' + str(r['duration_days']) + ' calendar days</td><td>' +
        money(r['known_position_component_change_range_usd'][1]) + '–' +
        money(r['known_position_component_change_range_usd'][0]) + '</td></tr>' for r in v['scenarios'])
    sign = lambda value: ('−' if value < 0 else '+' if value > 0 else '') + money(value)
    priced = ''.join('<tr><td>' + _pct(p['assumed_equity_embedded_weight'], 0) + '</td><td>' +
        ' to '.join(sign(x) for x in p['yes_component_range_usd']) + '</td><td>' +
        ' to '.join(sign(x) for x in p['no_component_range_usd']) + '</td></tr>'
        for p in v['scenarios'][1]['pricing_sensitivity'])
    return ('<div class="cashflow-component"><h3>' + escape(v['condition']) + '</h3>'
        '<p>Source-anchored freight cash-flow component · known UNP exposure ' +
        _pct(v['exposure']['known_issuer_weight']) + ' of this portfolio.</p>'
        '<table><thead><tr><th>Declared shutdown</th><th>Modeled loss in known UNP sleeve</th></tr></thead><tbody>' +
        rows + '</tbody></table><p><strong>Not a bound on the stock or portfolio\'s total loss.</strong> '
        'A strike at another railroad does not establish a UNP shutdown. Additional costs, future labor agreements, '
        'lasting customer losses, financing and market repricing are excluded.</p><p>Unresolved fund NAV for this '
        'issuer-exposure calculation: ' + _pct(v['exposure']['unresolved_fund_weight']) + '; not treated as zero.</p>'
        '<details><summary>Assumptions, sources and already-priced-in scenarios</summary>'
        '<p>Reported Q3 freight revenue $6.813 billion / 92 calendar days is spread uniformly. Full network shutdown; '
        'contribution margin, eventual catch-up and immediately usable tax shield each span 0–100%, with no midpoint. '
        'Discounting is zero; catch-up delay is 30 days. Additional costs are excluded, not estimated as zero.</p>'
        '<p>The table above is interruption versus no interruption, not a share-price change from today. '
        'For a 7-day shutdown, hypothetical embedded equity weights give:</p><table><thead><tr>'
        '<th>Assumed already priced in</th><th>Interruption occurs</th><th>It does not occur</th></tr></thead><tbody>' +
        priced + '</tbody></table><p>These weights are assumptions, not prediction-market Q or estimated physical P. '
        'Both branches share the same underlying cash-flow loss.</p><p><a href="' + escape(v['filing_source']['url'], quote=True) +
        '">Union Pacific SEC filing</a>. Source revenue: historical report, not a forward projection. Normalization: '
        'captured reference capitalization, not an executable price. Component evaluated ' + escape(v['evaluated_at']) +
        '; original assessment ' + escape(v['original_assessment_at']) + '. Later evidence does not backdate quotes.</p></details></div>')


def _mexico_tariff_component_html(event, case):
    """Present only the source-backed direct freight channel, never target P&L."""
    v = mexico_component_view(event.get('mexico_tariff_component'), event_id=event.get('id'),
                              portfolio_notional=case.get('notional'), portfolio_positions=case.get('positions'))
    if v is None:
        return ''
    money = lambda value: '$' + format(abs(value), ',.2f')
    rows = ''.join('<tr><td>' + str(r['horizon_days']) + ' days</td><td>' +
                   money(r['known_unp_position_component_change_usd'][1]) + '–' +
                   money(r['known_unp_position_component_change_usd'][0]) + '</td></tr>'
                   for r in v['scenarios'])
    domain_rows = ''.join('<li>' + escape(k.replace('_', ' ')) + ': ' +
                          _pct(interval['lower'], 0) + '–' + _pct(interval['upper'], 0) + '</li>'
                          for k, interval in v['declared_domains'].items())
    return ('<div class="cashflow-component"><h3>' + escape(v['condition']) + '</h3>'
            '<p>SEC Q3 2026: $820 million freight revenue for shipments <strong>to and from Mexico</strong>. '
            'US-import-affected share, traffic response, contribution margin, recovery and tax shield '
            'are all unknown. Each horizon begins at hypothetical onset of lost traffic, not the '
            'assessment date. The following is a declared, fixed-run-rate direct cash-flow scenario, '
            'not a forecast.</p>'
            '<table><thead><tr><th>Declared horizon</th><th>Modeled loss in known UNP sleeve</th></tr>'
            '</thead><tbody>' + rows + '</tbody></table>'
            '<p><strong>Not a bound on stock return or whole-portfolio loss.</strong> '
            'The original Mexico-import tariff event has no matching admitted probability and '
            'no identified already-priced-in estimate. No causal traffic elasticity has been calibrated.</p>'
            '<p>Known UNP exposure ' + _pct(v['exposure']['known_issuer_weight']) +
            '; unresolved fund weight ' + _pct(v['exposure']['unresolved_fund_weight']) +
            ' (not assumed zero).</p>'
            '<details><summary>Source, scope and assumptions</summary><ul>' + domain_rows + '</ul>'
            '<p>2026-Q3 both-direction revenue / 92 days is extrapolated over stated future windows; '
            'future revenue, impact onset, import-only traffic, tariff-caused volume loss and market repricing '
            'are not identified. Full offset, no effect and zero loss remain admissible.</p>'
            '<p><a href="' + escape(v['filing_source']['url'], quote=True) + '">'
            'Source: Union Pacific 2026-Q3 SEC filing</a>. Later inspection ' +
            escape(v['evaluated_at']) + '; original assessment ' +
            escape(v['original_assessment_at']) +
            '. Historical market quotes were not updated or backdated.</p></details></div>')


def _report_html(case):
    e = lambda value: escape(str(value if value is not None else ''), quote=True)
    replay_status = (case.get('offline_replay') or {}).get('status')
    replay_note = ''
    if replay_status == 'OFFLINE_SAVED_INPUTS_RECOMPUTED_R30':
        replay_note = 'Saved analysis recalculated locally'
    elif replay_status == 'ORIGINAL_RESULTS_RETAINED':
        replay_note = 'Original saved results retained'
    if replay_note:
        replay_note = ('<p class="softwarning">' + e(replay_note) +
                       ' · quotes remain dated ' + e(case.get('capture') or 'Unavailable') +
                       '. No new source requests.</p>')
    if (case.get('presentation_review') or {}).get('status') in (
            'PUBLIC_EXPORT_REVIEW_ONLY_R32', 'PUBLIC_EXPORT_REVIEW_ONLY_R33'):
        replay_note = ('<p class="softwarning">Saved export reviewed locally; original numbers and '
                       'quote date retained. No new fit or source requests.</p>')
    positions = ''.join(
        '<tr><td>' + e(p.get('symbol')) + '</td><td>' + e(p.get('type')) + '</td><td>' +
        e(_pct(p.get('weight'))) + '</td></tr>' for p in case.get('positions', []))
    identity_note = _identity_notice(case)
    identity_html = '<p class="softwarning">' + e(identity_note) + '</p>' if identity_note else ''
    sections = []
    for index, event in enumerate(case.get('risks', []), 1):
        quality = event.get('quality') or {}
        admission = _admission(event)
        primary = admission.get('primary_impact_allowed') is True
        unmatched_source = _unmatched_source_only(event, admission)
        coverage = quality.get('distribution_display_coverage') or {}
        definition = quality.get('definition_review') or {}
        lines = []
        meanings = []
        market_lines = []
        quote_notes = []
        for outcome in _event_outcomes(event):
            quote = _quote_metadata(event, outcome)
            pnl = outcome.get('pnl')
            bounds = _model_range(outcome)
            uncertain = _direction(outcome) == 'DIRECTION_UNCERTAIN'
            range_class = ('neutral' if uncertain or bounds is None or bounds == [0, 0]
                           else 'loss' if bounds[1] <= 0 else 'gain')
            range_value = (_money(bounds[0]) + ' to ' + _money(bounds[1])) if bounds else 'Range unavailable'
            direction = ('Direction uncertain' if uncertain else
                         'Fixed nominal USD cash value' if quality.get('model_kind') == 'NOMINAL_CASH_IDENTITY' else
                         'Under stated model assumptions') if bounds else ''
            identifiers = list(dict.fromkeys([*(outcome.get('source_contract_ids') or []),
                                             *([outcome['contract_id']] if outcome.get('contract_id') else [])]))
            if bounds and _range_meaning(outcome) not in meanings:
                meanings.append(_range_meaning(outcome))
            source_question = ('<small>Source question: ' + e(quote['question']) + '</small>') if quote['question'] else ''
            if quote['qualification']:
                note = ((quote['venue'] + ' · ') if quote['venue'] else '') + quote['qualification'] + ' ' + (
                    'Quote received: ' + quote['quote_received_at'] if quote['quote_received_at']
                    else 'Quote retrieval time was not retained.')
                if note not in quote_notes:
                    quote_notes.append(note)
            market_lines.append('<tr><td>' + e(outcome.get('label')) + source_question + '</td><td>' +
                                e(_chance(outcome)) + '</td><td class="identifiers">' +
                                e('; '.join(identifiers) or 'Not reported') + '</td></tr>')
            lines.append('<tr><td>' + e(outcome.get('label')) + source_question + '</td><td>' + e(_chance(outcome)) +
                         '</td><td class="' + range_class + '"><strong>' + e(range_value) + '</strong><small>' +
                         e(direction) + '</small></td><td class="point">' + e(_money(pnl)) +
                         '</td><td class="identifiers">' + e('; '.join(identifiers) or 'Not reported') + '</td></tr>')
        outcomes = ('<table><thead><tr><th>Outcome</th><th>Market-implied weight</th><th>Model range</th>'
                    '<th>Point estimate</th><th>Source contract IDs</th>'
                    '</tr></thead><tbody>' + ''.join(lines) + '</tbody></table>' if lines else
                    '<p class="empty">' + e(_unavailable_explanation(event, admission,
                                           'No supported priced portfolio outcome for this event.')) + '</p>')
        if unmatched_source and market_lines:
            outcomes = ('<p class="empty"><strong>No matching probability or impact for this target.</strong> ' +
                        e(_unavailable_explanation(event, admission)) + '</p><details class="diagnostic-model">' +
                        '<summary>View unmatched-source diagnostic</summary><p class="scope">Saved source '
                        'weights and model calculations belong to another proposition. They are not the '
                        'probability or impact of the target above.</p>' + outcomes + '</details>')
        elif not primary and market_lines:
            market_only = ('<table class="market-only"><thead><tr><th>Outcome</th>'
                           '<th>Market-implied weight</th><th>Source contract IDs</th></tr></thead><tbody>' +
                           ''.join(market_lines) + '</tbody></table>')
            if any(_finite(row.get('pnl')) for row in _event_outcomes(event)):
                summary = ('View failed-model diagnostic' if admission.get('state') == 'DIAGNOSTIC_ONLY'
                           else 'View unadmitted model diagnostic')
                outcomes = ('<p class="scope">' + e(_unavailable_explanation(event, admission,
                            'No admitted portfolio impact estimate.')) + '</p>' + market_only +
                            '<details class="diagnostic-model"><summary>' + summary +
                            '</summary><p class="scope">Preserved model calculation. Not a validated forecast '
                            'and not used to rank portfolio impact.</p>' + outcomes + '</details>')
            else:
                outcomes = market_only + '<p class="empty">Impact estimate unavailable. ' + e(
                    _unavailable_explanation(event, admission)) + '</p>'
        sources = []
        for admitted, title in ((_contracts(event.get('admitted')), 'Used'),
                                (_contracts(event.get('rejected')), 'Excluded')):
            for item in admitted:
                sources.append('<li><strong>' + title + ' · ' + e(item.get('id')) + '</strong>' +
                               '<span>' + e(item.get('venue') or '') + '</span><span>' +
                               e(item.get('question') or item.get('reason') or '') + '</span>' +
                               (_source_only_inspection_html(item) if title == 'Excluded' else '') + '</li>')
        notes = _review_notes(event)
        note_html = ('<h3>Search notes</h3><ul>' + ''.join('<li>' + e(note) + '</li>' for note in notes) +
                     '</ul>') if notes else ''
        status = _transfer_status(quality)
        check = _check_label(status) + ' [' + status + ']'
        if quality.get('failed_symbols'):
            check += ': ' + ', '.join(map(str, quality['failed_symbols']))
        warnings = []
        warnings.extend('<p class="softwarning">' + e(note) + '</p>' for note in quote_notes)
        if admission.get('state') == 'DIAGNOSTIC_ONLY':
            warnings.append('<p class="warning"><strong>Impact model did not pass:</strong> ' +
                            e(_failure_description(quality)) +
                            'Market weights remain available. Dollar amounts are retained only as '
                            'a failed-model diagnostic, not validated portfolio forecasts.</p>')
        elif quality.get('model_kind') == 'NOMINAL_CASH_IDENTITY':
            warnings.append('<p class="scope">Fixed nominal USD cash value is an accounting identity, '
                            'not an empirical forecast. It excludes interest income, inflation and '
                            'exchange-rate effects.</p>')
        elif primary and quality.get('model_kind') == 'DECLARED_CONTRACT_SCENARIO':
            warnings.append('<p class="scope">This result prices the selected source-contract question, '
                            'not the broader research hypothesis.</p>')
        elif primary and status != 'PASS' and event.get('outcomes'):
            warnings.append('<p class="softwarning">The selected model is ' +
                            ('not independently tested' if status == 'NOT_TESTED' else 'not independently verified') +
                            '. Historical checks of comparison models do not validate this combined model.</p>')
        missing = coverage.get('omitted_reference_mass')
        if _finite(missing) and missing > 1e-6:
            warnings.append('<p class="softwarning">Incomplete displayed distribution: ' +
                            e(_pct(missing, 2)) + ' of source market-implied weight is omitted '
                            '(small, unquoted or ambiguous buckets). Remaining weights are '
                            'not renormalized.</p>')
        if definition.get('status') not in (None, '', 'DEFINED_FOR_REVIEW'):
            source_link = (' <a href="https://www.bea.gov/news/schedule/next-year">BEA release schedule</a>'
                           if definition.get('source_url') == 'https://www.bea.gov/news/schedule/next-year' else '')
            warnings.append('<p class="warning"><strong>Research hypothesis not yet valid as specified:</strong> ' +
                            e(definition.get('reason')) + source_link + '</p>')
        comparisons = []
        for model in quality.get('model_comparisons') or []:
            if not isinstance(model, dict):
                continue
            mstatus = _transfer_status(model)
            failed = model.get('failed_symbols') or []
            mweight = _reported_failure_share(model)
            comparisons.append('<li><strong>' + e(model.get('label') or model.get('id')) + '</strong><span>' +
                               e(_check_label(mstatus) + ' [' + mstatus + ']') +
                               (e(' · ' + ', '.join(failed)) if failed else '') +
                               (e(' · ' + _pct(mweight, 0) + ' of portfolio flagged')
                                if mstatus == 'FAIL' and _finite(mweight) else '') + '</span></li>')
        comparison_warning = (
            '<p class="warning"><strong>Model reliability warning:</strong> At least one comparison model '
            'did not pass its historical transfer check. Those failures remain visible below; they '
            'are not erased by selecting a different model.</p>'
            if any(_transfer_status(m) == 'FAIL' for m in quality.get('model_comparisons') or []
                   if isinstance(m, dict)) else '')
        diagnostics = ('<h3>Comparison model checks</h3>' + comparison_warning +
                       '<p class="scope">These are separate diagnostics, '
                       'not validation of the selected model or a forecast of future performance.</p><ul>' +
                       ''.join(comparisons) + '</ul>') if comparisons else ''
        range_notes = ('<h3>What the range means</h3>' +
                       ''.join('<p class="scope">' + e(text) + '</p>' for text in meanings)) if meanings else ''
        basis = event.get('exposure_basis')
        exposure_heading = {'MODEL_COVERED_HOLDINGS': 'Holdings covered by this model',
                            'NAMED_SOURCE_CHANNELS': 'Named exposure channels'}.get(basis, 'Portfolio holdings')
        exposure = event.get('exposure') or case.get('positions') or []
        exposure_html = ('<p class="scope"><strong>' + exposure_heading + ':</strong> ' +
                         e(' · '.join(str(p.get('symbol') or '') + ' ' + _pct(p.get('weight'))
                                      for p in exposure)) + '</p>')
        related_note = ''
        if event.get('source_defined_scenario'):
            target_ids = list(dict.fromkeys([
                *([event['related_to_event_id']] if event.get('related_to_event_id') else []),
                *(event.get('related_to_event_ids') or [])]))
            targets = [row for row in case.get('risks') or [] if row.get('id') in target_ids]
            related_note = ('<p class="scope"><strong>Related source-defined scenario.</strong> '
                            'The rate is measured on this scenario\'s own date. These weights do not price ' +
                            e('; '.join(str(row.get('title') or row.get('id')) for row in targets) or
                              'the original action/window event') + '.</p>')
        related_ids = set(event.get('related_source_scenario_ids') or [])
        forward = [(idx, row) for idx, row in enumerate(case.get('risks') or [], 1)
                   if row.get('source_defined_scenario') is True and row.get('id') in related_ids]
        forward_html = ('<div class="related-source-links"><p class="scope">A related, different question is available. '
                        'It does not supply this target\'s probability or impact.</p>' +
                        ''.join('<p><a href="#scenario-' + str(idx) + '">View ' + e(row.get('title')) +
                                ' →</a></p>' for idx, row in forward) + '</div>') if forward else ''
        sections.append('<section class="event"><div class="eventtop" id="scenario-' + str(index) + '"><small>EVENT ' + str(index).zfill(2) +
                        ' · ' + e(event.get('date')) + '</small><h2>' + e(event.get('title')) +
                        '</h2><span class="check">' + e(_model_label(quality.get('model_kind'))) +
                        ' · selected-model check: ' + e(check) +
                        '</span></div>' + related_note + ''.join(warnings) + outcomes + _pricing_weight_sensitivity_html(event) + _analogue_html(event, case.get('notional') or 0) + _business_component_html(event, case) + _mexico_tariff_component_html(event, case) + forward_html + exposure_html +
                        '<details open><summary>Prediction-market evidence (' + str(len(sources)) +
                        ' contracts)</summary><ul>' + ''.join(sources) + '</ul>' + note_html +
                        range_notes + diagnostics + _sensitivity_html(event) + '</details></section>')
    printed = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    style = '''
    :root{font-family:Arial,Helvetica,sans-serif;color:#eaf1e8;background:#071017}
    *{box-sizing:border-box}body{margin:0;padding:46px 5%;line-height:1.5}
    header{border-bottom:1px solid #354851;padding-bottom:30px;margin-bottom:25px}
    header small,.eyebrow{color:#b6ef8b;letter-spacing:1.5px;font-weight:700}
    h1{font-size:42px;line-height:1.1;margin:14px 0}h2{font-size:25px;margin:8px 0 16px}h3{font-size:16px}
    .meta{color:#b6c4c3}.top{display:flex;gap:20px;flex-wrap:wrap;justify-content:space-between}
    .value{font-size:32px;font-weight:750;color:#d0ff8b}.event{background:#122029;border:1px solid #354851;
    border-radius:15px;padding:26px;margin:20px 0;break-inside:avoid}.eventtop{margin-bottom:20px}
    .eventtop small{color:#b6ef8b}.check{color:#c5d4d2;font-size:14px}
    .warning{padding:13px;border:1px solid #ba7684;border-radius:10px;background:#3a272d;color:#ffe6e8}
    .softwarning{padding:11px;border:1px solid #787146;border-radius:10px;background:#302f24;color:#f4ebc5}
    .warning a{color:#d9ffa0}
    table{border-collapse:collapse;width:100%;margin:12px 0 18px}td,th{border-bottom:1px solid #405059;
    padding:11px;text-align:left;font-size:14px;vertical-align:top}th{color:#b6ef8b;font-size:12px}
    .loss{color:#ff7d97;font-weight:bold}.gain{color:#79f4c0;font-weight:bold}.neutral{color:#d9e2e0}
    td small{display:block;font-size:11px;font-weight:400;margin-top:4px}.point{color:#b6c4c3}
    .identifiers{font-size:11px;overflow-wrap:anywhere;max-width:200px}
    summary{cursor:pointer;font-weight:700}details{margin-top:16px}li{padding:6px 0}
    li span{display:block;color:#a8bdc0}.empty,.scope{color:#b8c8c7}.scope{max-width:950px}
    .related-source-links a{color:#d0ff8b}.business-watchlist{border-top:1px solid #354851;padding:15px 0;margin:20px 0}
    .business-watchlist li{margin:10px 0}.business-watchlist li p{margin:5px 0;font-size:14px;color:#b8c8c7}
    @media print{body{color:#152126;background:#fff;padding:16mm 12mm}
    .value,.eyebrow,header small,.eventtop small,th{color:#245b35}
    .event{color:#152126;background:#fff;border:1px solid #a9b9b9;break-inside:avoid}
    .check,.meta,.empty,.scope,li span,.point,.neutral{color:#334446}.loss{color:#a21d40}.gain{color:#096e4f}
    .warning,.softwarning{background:#fff;border-color:#b1a89f;color:#382b2f}
    details{display:block}details>ul{display:block}}
    '''
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Portfolio Impact · Report</title>'
            '<style>' + style + '</style></head><body>'
            '<header><small>PORTFOLIO IMPACT · CASTLE TECHNOLOGIES OPTION 1</small>'
            '<h1>Your portfolio analysis</h1><div class="top"><div><div class="value">' +
            e(_money(case.get('notional')).lstrip('+')) + '</div><div class="meta">Portfolio value</div></div>'
            '<div class="meta">Analysis captured: ' + e(case.get('capture') or 'Unavailable') +
            '<div>Report created: ' + e(printed) + '</div></div></div>' + replay_note +
            '<p class="scope">' + e(case.get('verification_status', '')) +
            '. Values shown are modeled scenario components using the available source evidence. '
            'Market-implied weights are not measured physical probabilities. Model ranges are not '
            'guaranteed bounds on future returns; their interpretation is stated with each event.</p></header>'
            '<section><h2>Investments</h2><table><thead><tr><th>Code</th><th>Type</th><th>Portfolio share</th>'
            '</tr></thead><tbody>' + positions + '</tbody></table>' + identity_html + '</section>' +
            _watchlist_html(case) + _web_research_html(case) + ''.join(sections) +
            '</body></html>')


def _analogue_html(event, notional):
    """Measured past-episode replay and the priced part of a partially blocked table."""
    out = []
    a = event.get('analogue_stress') or {}
    rows = [x for x in a.get('episodes') or [] if x.get('status') == 'MEASURED']
    if a.get('episode_set') == 'HORMUZ_TRANSIT_SHOCK' or a.get('direct_disruption_2026'):
        observed = [x for x in rows if x.get('direct_disruption_observation_not_causal_estimate') is True]
        if not observed and a.get('direct_disruption_2026', {}).get('schema') == 'castle.r87.realized_hormuz_reference.v1':
            observed = [a['direct_disruption_2026']]
        if observed:
            out.append('<p class="scope"><strong>Directly observed 2026 Hormuz/Gulf disruption:</strong> ' +
                       escape(_money(observed[0]['pnl_usd'])) +
                       ' on the current portfolio during March 2–6, 2026. '
                       'The 2019–25 comparators were milder and not closures. '
                       'This one crisis is confounded by war and policy news, and does not identify '
                       'the next closure\'s causal or already-priced-in equity impact.</p>')
        else:
            out.append('<p class="scope">The directly observed 2026 Gulf shock could not be '
                       'measured for all owned positions; prior mild episodes are not closure forecasts.</p>')
    if a.get('status') == 'MEASURED_RANGE':
        ordinary = a.get('ordinary_window_5_95')
        ordinary_text = ('. Ordinary ' + str(a.get('window_sessions')) + '-session 5th–95th percentile: ' +
                         escape(_money(notional * ordinary[0])) + ' to ' +
                         escape(_money(notional * ordinary[1])) +
                         (' (excluding windows touching any reviewed episode)' if a.get('ordinary_baseline_excludes_full_episode_windows')
                          else ' (earlier captured comparator may include neighbouring event days)')) if ordinary else ''
        # Also works for the R85 recorded public examples, which have the
        # original per-episode numbers but not R86's newly named diagnostics.
        returns = [x['portfolio_return'] for x in rows]
        influence_text, overlap_text = '', ''
        if len(returns) >= 3:
            range_lo, range_hi = min(returns), max(returns)
            influence = max((
                (max(min(returns[:k] + returns[k + 1:]) - range_lo,
                     range_hi - max(returns[:k] + returns[k + 1:])), row['episode'])
                for k, row in enumerate(rows)), key=lambda x: x[0])
            influence_text = ('. Most influential range endpoint: ' + escape(influence[1]) +
                              ' (' + escape(_money(notional * influence[0])) + ' endpoint influence)')
            clean = [x['portfolio_return'] for x in rows if not x.get('overlapping_episodes')]
            if len(clean) >= 3:
                overlap_text = ('. Excluding catalogued overlaps, ' + str(len(clean)) + ' episodes ranged ' +
                                escape(_money(notional * min(clean))) + ' to ' +
                                escape(_money(notional * max(clean))) + ' (sensitivity, not causal identification)')
        out.append('<h3>Past-episode replay: ' + escape(a.get('episode_set_title') or '') + ' (' + str(len(rows)) +
                   ' episodes, first ' + str(a.get('window_sessions')) + ' sessions)</h3><p>Your current holdings moved ' +
                   escape(_money(a['pnl_range_usd'][0])) + ' to ' + escape(_money(a['pnl_range_usd'][1])) + '; median ' +
                   escape(_money(a.get('median_pnl_usd'))) + ordinary_text + overlap_text + influence_text +
                   '. Measured history, not an out-of-sample prediction interval.</p>')
    elif a:
        out.append('<h3>Past-episode replay</h3><p class="scope">' + escape(a.get('reason') or 'Not measured') + '</p>')
    if rows:
        out.append('<table><thead><tr><th>Episode</th><th>Window</th><th>Portfolio change</th></tr></thead><tbody>' +
                   ''.join('<tr><td>' + escape(x['episode']) +
                           ('<small>Window also contains: ' + escape('; '.join(x['overlapping_episodes'])) + '</small>'
                            if x.get('overlapping_episodes') else '') + '</td><td>' + escape(x['base_session']) + ' → ' +
                           escape(x['end_session']) + '</td><td class="' + ('loss' if x['pnl_usd'] < 0 else 'gain') + '">' +
                           escape(_money(x['pnl_usd'])) + '</td></tr>' for x in rows) + '</tbody></table>')
    if a:
        out.append('<p class="scope">Measured history, not a forecast. ' + escape(a.get('caveat') or '') + '</p>')
    p = event.get('partial_priced_component') or {}
    if p.get('outcomes'):
        out.append('<h3>Priced part of the portfolio (' + _pct(p.get('priced_weight'), 0) + ' of value)</h3><table><thead>'
                   '<tr><th>Outcome</th><th>Priced positions</th></tr></thead><tbody>' +
                   ''.join('<tr><td>' + escape(o['label']) + '</td><td>' + escape(_money(o['priced_pnl_usd'])) + '</td></tr>'
                           for o in p['outcomes']) + '</tbody></table><p class="scope">Not priced: ' +
                   escape(', '.join(u['symbol'] for u in p.get('unpriced') or [])) + '. ' + escape(p.get('scope') or '') + '</p>')
    return ''.join(out)


def _analogue_csv(case):
    out = StringIO()
    writer = csv.writer(out, lineterminator='\r\n')
    writer.writerow(['event_id', 'event_title', 'episode_set', 'episode', 'onset_session', 'base_session',
                     'end_session', 'portfolio_return', 'portfolio_change_usd', 'overlapping_episodes'])
    for event in case.get('risks') or []:
        a = event.get('analogue_stress') or {}
        for x in a.get('episodes') or []:
            measured = x.get('status') == 'MEASURED'
            writer.writerow([_safe_cell(v) for v in (event.get('id'), event.get('title'), a.get('episode_set'),
                             x.get('episode'), x.get('onset_session'), x.get('base_session') if measured else '',
                             x.get('end_session') if measured else '', x.get('portfolio_return') if measured else '',
                             x.get('pnl_usd') if measured else '', '; '.join(x.get('overlapping_episodes') or []))])
        severe = a.get('direct_disruption_2026') or {}
        if severe.get('schema') == 'castle.r87.realized_hormuz_reference.v1':
            writer.writerow([_safe_cell(v) for v in (
                event.get('id'), event.get('title'), 'DIRECT_2026_HORMUZ_OBSERVATION',
                severe.get('episode'), severe.get('onset_session'), severe.get('base_session'),
                severe.get('end_session'), severe.get('portfolio_return'),
                severe.get('pnl_usd'), 'Severe 2026 event confounded with contemporaneous war/policy news')])
    return out.getvalue()


def make_report_zip(case):
    """Return ZIP bytes containing no more information than public browser data."""
    if not isinstance(case, dict) or case.get('user_supplied') is not True:
        raise ValueError('Only a completed user portfolio can be exported')
    if not isinstance(case.get('risks'), list) or not isinstance(case.get('positions'), list):
        raise ValueError('Incomplete case')
    rows = list(_rows(case))
    note = ('PORTFOLIO IMPACT — exported analysis\n\n'
            'Open report.html for a readable, printable full-event report.\n'
            'Open scenario_results.csv for event/outcome estimates in a spreadsheet.\n'
            'Open past_episode_replay.csv for the measured past-episode moves of your current holdings.\n'
            'Open portfolio_data.json for the public structured evidence.\n\n'
            'All numbers come from the completed portfolio analysis.\n'
            'A missing estimate means the model could not establish that consequence.\n'
            'Model ranges retain their recorded interval kind and interpretation. They are '
            'NOT guaranteed future-return bounds or automatic prediction intervals.\n'
            'A range crossing zero means the direction is uncertain under that model.\n'
            'Market-implied Q weights are NOT measured physical event probabilities.\n'
            'Small/unquoted/ambiguous outcome buckets may be omitted in the display; '
            'remaining Q weights are NOT renormalized.\n'
            'A FAIL model transfer check applies to an EVENT/HOLDING combination, '
            'not to each separate CSV row. Its status is repeated across outcomes.\n'
            'NOT_REPORTED and NOT_TESTED are not PASS. Missing flagged portfolio '
            'shares stay blank, not zero. Daily and announcement comparison diagnostics '
            'are kept separately from the selected model check.\n'
            'Illustrative P&L for failed or untested transfer checks is NOT a '
            'validated investment forecast. Research hypotheses with unresolved '
            'settlement definitions or deadlines are NOT priced outcomes.\n'
            'Model-admission columns distinguish conditional scenarios from diagnostic-only results. '
            'Failed-model dollar calculations remain in CSV/JSON and behind closed diagnostic details '
            'in the HTML; they are not used as the primary impact estimate.\n'
            'Model-assumption sensitivity is exported separately from the original model interval. '
            'It is a stress range, NOT a new prediction band or evidence that a failed model passed.\n'
            'For admitted exact binary source-contract stress cards, the report separately varies '
            'the unmeasured effective equity pricing weight pi over 0 to 100 percent. '
            'A source contract quote Q is not proof that equity pi equals Q; resulting YES and NO '
            'dollar ranges are conditional on the declared economic contrast and two-state identity, '
            'NOT measured event-caused returns or statistical confidence intervals.\n'
            'A separately named source-defined tariff scenario retains its own date and contract '
            'meaning; it does NOT assign probabilities to the linked original target event.\n'
            'Source venue, quote qualification and retained quote timestamps are exported explicitly. '
            'Indicative Gamma summaries are not verified CLOB execution prices.\n'
            'The separate excluded_source_inspections CSV field retains recorded exact-contract '
            'questions, quote timing, and source provenance. These are NOT probabilities of the '
            'target, admitted outcomes, or portfolio-impact estimates; historical quotes are not current.\n'
            'Business-risk watchlist entries are unverified research hypotheses in HTML/JSON, '
            'not priced events or additional portfolio-impact estimates.\n'
            'The separate business_interruption_component field is a source-anchored, '
            'assumption-dependent freight cash-flow channel, NOT a bound on total stock/portfolio losses. '
            'Its declared embedded-equity weights are neither market Q nor physical P. '
            'Later source inspection does not alter the original assessment timestamp.\n'
            'The separate mexico_tariff_component field holds a declared direct freight-cash-flow '
            'channel, supported by a Q3-2026 UNP filing that includes shipments TO and FROM Mexico. '
            'Import-affected share and causal tariff response are both unknown. The channel has '
            'no identified target-market Q, physical P, stock-return or full portfolio P&L. '
            'Its zero-to-upper scenario is not a real-world tariff-loss forecast.\n'
            'The 7/30/90-day horizons begin at an unknown hypothetical onset of traffic loss, '
            'not the analysis timestamp or the tariff-contract settlement date.\n'
            'Provider-only files, paid data archives and API credentials are excluded.\n')
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED, compresslevel=7) as archive:
        archive.writestr('report.html', _report_html(case))
        archive.writestr('scenario_results.csv', '\ufeff' + _csv_export(rows))
        archive.writestr('past_episode_replay.csv', '\ufeff' + _analogue_csv(case))
        archive.writestr('portfolio_data.json', json.dumps(case, indent=2, ensure_ascii=False, allow_nan=False))
        archive.writestr('README.txt', note)
    return buffer.getvalue()
