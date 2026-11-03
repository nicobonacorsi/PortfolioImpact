"""Small, licensed-data-safe display adapter for a real Option 1 assessment.

The analysis engine remains in live_assessment.py. This module only presents its
already-computed, derived results to the Portfolio Impact R26 interface. Missing
probabilities, unmatched contracts and unvalidated returns are never filled in.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import re
from urllib.parse import urlsplit

from .event_definition_review import review_event_definition
from .model_admission import eligible_contract_scenarios, event_model_admission


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _probability(value):
    number = _number(value)
    return number if number is not None and 0 <= number <= 1 else None


def _interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = (_probability(part) for part in value)
    if low is None or high is None or low > high + 1e-9:
        return None
    return [low, max(low, high)]


def _date(value):
    if isinstance(value, str) and len(value) >= 10:
        try:
            return datetime.strptime(value[:10], '%Y-%m-%d').date().isoformat()
        except ValueError:
            pass
    return None


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def _observed_quote_time(record):
    """Keep an actual source-receipt time, never the analysis/evaluation clock."""
    if not isinstance(record, dict):
        return None
    source = record.get('source') or {}
    candidates = [record]
    if isinstance(source, dict):
        candidates.append(source)
    for item in candidates:
        for key in ('quote_received_at', 'retrieved_at_utc', 'retrieved_at', 'completed_at_utc'):
            value = item.get(key)
            if not isinstance(value, str):
                continue
            try:
                parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                return value
    return None


def _source_quote_metadata(distribution, outcome_id=None):
    """Public, source-bound labels; absence is not a reason to invent a venue."""
    retained = distribution.get('source_quote_evidence') or {}
    record = retained.get(str(outcome_id), {}) if isinstance(retained, dict) else {}
    record = record if isinstance(record, dict) else {}
    source = distribution.get('source') or {}
    source = source if isinstance(source, dict) else {}
    venue = _text(record.get('venue')) or _text(distribution.get('venue'))
    if not venue:
        request_url = source.get('request_url') or source.get('source_url') or ''
        try:
            host = (urlsplit(request_url).hostname or '').lower()
        except (TypeError, ValueError):
            host = ''
        if host == 'kalshi.com' or host.endswith('.kalshi.com'):
            venue = 'Kalshi'
        elif host == 'polymarket.com' or host.endswith('.polymarket.com'):
            venue = 'Polymarket'
    question = (_text(record.get('question')) or
                _text(distribution.get('probability_question')))
    received_at = _observed_quote_time(record) or _observed_quote_time(distribution)
    if received_at is None and outcome_id is None and isinstance(retained, dict):
        times = {_observed_quote_time(value) for value in retained.values() if isinstance(value, dict)}
        # A single event timestamp is meaningful only if every retained
        # contract has that same observed receipt. Per-contract times survive.
        if len(times) == 1 and None not in times:
            received_at = next(iter(times))
    return {'venue': venue, 'question': question,
            'question_basis': ('SOURCE_CONTRACT_TEXT' if _text(record.get('question')) else
                               'SOURCE_DISTRIBUTION_QUESTION' if question else 'NOT_RETAINED'),
            'quote_scope': (_text(record.get('quote_scope')) or
                            _text(distribution.get('quote_scope')) or
                            'Source-derived pricing distribution'),
            'quote_received_at': received_at}


def _retained_tariff_quote_evidence(distribution, candidates):
    """Join existing source metadata only when contract, venue and quote agree.

    Rejected target matches can still identify their own source contract. This
    join does not change that rejection, quotes, model inputs or admission.
    """
    evidence = {}
    venue = _text(distribution.get('venue'))
    link = _text(distribution.get('event_link'))
    for atom in distribution.get('atoms') or []:
        if not isinstance(atom, dict):
            continue
        identifier = _text(atom.get('outcome_id'))
        quoted = _interval(atom.get('quote_interval'))
        if not identifier or quoted is None:
            continue
        matches = []
        for item in candidates:
            if not isinstance(item, dict) or str(item.get('contract_id') or '') != identifier:
                continue
            if _text(item.get('venue')) != venue:
                continue
            source_link = _text(item.get('source_url'))
            if link and source_link and source_link != link:
                continue
            reference = item.get('contract_quote_reference') or {}
            reference = reference if isinstance(reference, dict) else {}
            interval = _interval(item.get('quote_interval')) or _interval(reference.get('interval'))
            if interval is None or any(abs(a - b) > 1e-12 for a, b in zip(interval, quoted)):
                continue
            matches.append({'question': _text(item.get('market_question')),
                            'quote_scope': _text(item.get('quote_scope')) or _text(reference.get('scope')),
                            'quote_received_at': _observed_quote_time(item)})
        if not matches:
            continue
        questions = {item['question'] for item in matches if item['question']}
        scopes = {item['quote_scope'] for item in matches if item['quote_scope']}
        times = [item['quote_received_at'] for item in matches if item['quote_received_at']]
        # Repeated acquisition of an identical quote does not justify claiming
        # it was first observed at the newest time. Retain the oldest receipt.
        received_at = min(times, key=lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))) if times else None
        evidence[identifier] = {
            'venue': venue, 'question': next(iter(questions)) if len(questions) == 1 else '',
            'quote_scope': next(iter(scopes)) if len(scopes) == 1 else '',
            'quote_received_at': received_at,
            'quote_time_basis': 'EARLIEST_MATCHING_RECORDED_SOURCE_RECEIPT' if received_at else 'NOT_RETAINED',
        }
    return evidence


def _risk_watchlist(report, positions):
    """Retain the scout's hypotheses separately from priced/resolved events."""
    rows, seen = [], set()
    raw = report.get('risk_watchlist')
    if not isinstance(raw, list):
        return rows
    for risk in raw:
        if not isinstance(risk, dict):
            continue
        identifier, title = _text(risk.get('risk_id')), _text(risk.get('risk_title'))
        if not identifier or not title or identifier in seen:
            continue
        rationale = _text(risk.get('exposure_rationale'))
        named = [{'symbol': position['symbol'], 'weight': position['weight']}
                 for position in positions if re.search(
                     r'(?<![A-Z0-9])' + re.escape(position['symbol']) + r'(?![A-Z0-9])',
                     rationale.upper())]
        linked_weight = math.fsum(p['weight'] for p in named)
        rows.append({'risk_id': identifier, 'title': title,
                     'economic_channel': _text(risk.get('economic_channel')),
                     'monitoring_signal': _text(risk.get('monitoring_signal')),
                     'exposure_rationale': rationale, 'exposed_positions': named,
                     'linked_direct_portfolio_weight': linked_weight,
                     'inspection_priority_basis': 'UNRANKED_UNTIL_CONDITIONAL_IMPACT_IS_IDENTIFIED',
                     'inspection_priority': None,
                     'exposure_basis': 'NAMED_HOLDINGS_IN_UNVERIFIED_SCOUT_RATIONALE',
                     'status': 'UNVERIFIED_MONITORING_HYPOTHESIS',
                     'evidence_scope': 'SCOUT_HYPOTHESIS_NOT_PRICED_SCENARIO',
                     'origin': 'CLAUDE_RISK_HYPOTHESIS_NOT_INDEPENDENTLY_VERIFIED',
                     'market_status': 'NOT_ESTABLISHED_RISK_SCOUT_NOT_A_CONTRACT_SEARCH',
                     'probability': None, 'pnl': None, 'portfolio_impact_usd': None})
        seen.add(identifier)
    # Preserve the scout's candidate list without converting named NAV share
    # into an unjustified priority. A small holding can have a much larger
    # conditional scenario move than a large holding.
    return rows


_SOURCE_ONLY_SCHEMA = 'castle.source_only_inspection.v1'
_SOURCE_ONLY_SCOPE = 'EXACT_CONTRACT_QUOTE_NOT_TARGET_PROBABILITY'
_QUOTE_SCOPES = {'SOURCE_RECEIPT_FRESHNESS_CHECKED_AT_EXPLICIT_EVALUATION',
                 'HISTORICAL_QUOTE_AT_RECEIPT_TIME_NOT_CURRENT_AT_EVALUATION'}
_QUOTE_UNAVAILABLE = 'No admissible source quote at the recorded evaluation time.'
_QUOTE_STALE = 'The source receipt is stale or outside the recorded evaluation time.'
_QUOTE_INVALID = 'The retained source quote failed display consistency checks.'


def _source_instant(value):
    if (not isinstance(value, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)', value)):
        return None
    try:
        when = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return when if when.tzinfo is not None and when.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def _strict_source_interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
            return None
        low, high = map(float, value)
        return [low, high] if math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1 else None
    except (ValueError, OverflowError):
        return None


def _safe_contract_source_url(value, identifier):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}', identifier):
        return None
    # Exact adapter URL only: no credentials, query parameters, fragments,
    # alternate host, encoded path or arbitrary source-payload link forwarding.
    expected = 'https://external-api.kalshi.com/trade-api/v2/markets/' + identifier
    return value if value == expected else None


def _source_only_inspection_view(row):
    """Validate/allowlist public display metadata; this is not source authentication.

    Original-byte inspection occurs in live_event_brief. A portable public JSON
    document cannot establish those bytes independently; hashes describe the
    recorded engine check, not a new provider signature or target relation.
    """
    value = row.get('source_only_evidence') if isinstance(row, dict) else None
    if not isinstance(value, dict):
        return None
    identifier = _text(row.get('id'))
    received, evaluated = _source_instant(value.get('quote_received_at')), _source_instant(value.get('quote_evaluated_at'))
    question, url = _text(value.get('question')), _safe_contract_source_url(value.get('source_url'), identifier)
    if (value.get('schema') != _SOURCE_ONLY_SCHEMA or value.get('scope') != _SOURCE_ONLY_SCOPE
            or value.get('contract_id') != identifier or value.get('venue') != 'Kalshi'
            or not question or not url or received is None or evaluated is None
            or _text(value.get('quote_scope')) not in _QUOTE_SCOPES
            or any(not isinstance(value.get(key), str) or not re.fullmatch(r'[a-f0-9]{64}', value[key])
                   for key in ('response_sha256', 'rules_sha256'))
            or _text(value.get('quote_status')) not in {'AVAILABLE_AT_RECORDED_EVALUATION', 'UNAVAILABLE_AT_RECORDED_EVALUATION'}):
        return None
    interval = _strict_source_interval(value.get('quote_interval'))
    reason = _text(value.get('quote_unavailable_reason'))
    reason = reason if reason in {_QUOTE_UNAVAILABLE, _QUOTE_STALE, _QUOTE_INVALID} else _QUOTE_UNAVAILABLE
    if value['quote_status'] != 'AVAILABLE_AT_RECORDED_EVALUATION':
        interval = None  # An old numeric field cannot revive an excluded quote.
    elif interval is None:
        reason = _QUOTE_INVALID
    elif not 0 <= (evaluated - received).total_seconds() <= 300:
        interval, reason = None, _QUOTE_STALE
    elif ('HISTORICAL_QUOTE_AT_RECEIPT_TIME' in value['quote_scope'] and evaluated != received):
        interval, reason = None, _QUOTE_INVALID
    return {'schema': _SOURCE_ONLY_SCHEMA, 'scope': _SOURCE_ONLY_SCOPE,
            'contract_id': identifier, 'venue': 'Kalshi', 'question': question,
            'quote_interval': interval,
            'quote_status': ('AVAILABLE_AT_RECORDED_EVALUATION' if interval is not None else
                             'UNAVAILABLE_AT_RECORDED_EVALUATION'),
            'quote_unavailable_reason': '' if interval is not None else reason,
            'quote_received_at': value['quote_received_at'], 'quote_evaluated_at': value['quote_evaluated_at'],
            'quote_scope': value['quote_scope'], 'source_url': url,
            'response_sha256': value['response_sha256'], 'rules_sha256': value['rules_sha256']}


def _retained_source_only_inspection(entry, identifier):
    """Project an existing original-byte inspection, never a copied admission flag."""
    if (entry.get('source_only_inspection') is not True
            or entry.get('evidence_scope') != 'VERIFIED_CONTRACT_INSPECTION_NOT_TARGET_PROBABILITY'
            or _text(entry.get('exact_contract_status')) not in {'CONTRACT_IDENTITY_Q_BOUND', 'CONTRACT_IDENTITY_NO_ADMISSIBLE_QUOTE'}
            or entry.get('event_relationship') != 'NONE'
            or _text(entry.get('event_relationship_status')) not in {
                'UNVERIFIED', 'EVENT_DATE_NOT_ADMISSIBLE_FOR_LIVE_RESEARCH'}
            or entry.get('broad_macro_event_matched') is not False
            or entry.get('event_pricing_measure_Q_interval') is not None
            or entry.get('pricing_measure_Q_interval') is not None):
        return None
    question, source, quote = (entry.get(key) for key in ('source_question', 'source', 'quote'))
    if not all(isinstance(item, dict) for item in (question, source, quote)):
        return None
    primary, secondary = entry.get('market_rules_primary'), entry.get('market_rules_secondary')
    if (not isinstance(primary, str) or not primary.strip() or not isinstance(secondary, str)
            or question.get('contract_id') != identifier
            or not _text(question.get('yes_question'))
            or _text(entry.get('market_question')) != _text(question.get('yes_question'))
            or question.get('binary_definition') != 'THIS_EXACT_CONTRACT_SETTLES_YES_AS_DEFINED_BY_FULL_RULES'
            or question.get('macro_hypothesis_equivalence') != 'NOT_ESTABLISHED'
            or question.get('equity_impact_conditional_on_yes') != 'NOT_IDENTIFIED'
            or source.get('assurance') != 'LOCAL_HTTP_RECEIPT_BYTES_VERIFIED_NOT_PROVIDER_SIGNATURE'
            or source.get('exchange_quote_update_time_verified') is not False
            or quote.get('timestamp_kind') != 'retrieval_receipt'):
        return None
    rules_hash = hashlib.sha256((primary + '\n' + secondary).encode('utf-8')).hexdigest()
    if question.get('verified_rules_sha256') != rules_hash:
        return None
    received = _source_instant(entry.get('quote_observed_at_utc'))
    if (received is None or received != _source_instant(source.get('retrieved_at_utc'))
            or received != _source_instant(quote.get('quote_timestamp'))):
        return None
    interval = _strict_source_interval(entry.get('contract_pricing_measure_Q_interval'))
    normalized = _strict_source_interval([quote.get('lower'), quote.get('upper')])
    available = (entry.get('event_relationship_status') == 'UNVERIFIED'
                 and entry['exact_contract_status'] == 'CONTRACT_IDENTITY_Q_BOUND' and quote.get('usable') is True
                 and interval is not None and normalized == interval)
    raw = {'schema': _SOURCE_ONLY_SCHEMA, 'scope': _SOURCE_ONLY_SCOPE,
           'contract_id': identifier, 'venue': 'Kalshi', 'question': question.get('yes_question'),
           'quote_interval': interval if available else None,
           'quote_status': 'AVAILABLE_AT_RECORDED_EVALUATION' if available else 'UNAVAILABLE_AT_RECORDED_EVALUATION',
           'quote_unavailable_reason': (_QUOTE_STALE if quote.get('exclusion_reason') == 'Stale quotation receipt'
                                        else _QUOTE_INVALID if quote.get('usable') is True and not available
                                        else _QUOTE_UNAVAILABLE),
           'quote_received_at': entry.get('quote_observed_at_utc'),
           'quote_evaluated_at': entry.get('quote_evaluated_at_utc'), 'quote_scope': entry.get('quote_scope'),
           'source_url': entry.get('source_contract_url'),
           'response_sha256': source.get('source_response_sha256'), 'rules_sha256': rules_hash}
    return _source_only_inspection_view({'id': identifier, 'source_only_evidence': raw})


def _comparison_review(event, supporting_events=()):
    """Preserve rejected evidence without granting a semantic relation.

    Default scenario rows omit pre-review quarantines. Join their research
    records only by the same event ID; a source-defined child must not inherit
    its parent's rejection. Model explanations remain explicitly unverified
    notes alongside, never replacements for, deterministic rejection reasons.
    """
    event_id = _text(event.get('event_id') or event.get('id'))
    sources = [event, *(row for row in supporting_events if isinstance(row, dict)
                       and event_id and _text(row.get('event_id') or row.get('id')) == event_id)]
    result, notes, seen, note_keys = [], [], {}, set()
    source_inspections, conflicting_inspections = {}, set()

    def add_note(row):
        key = (row.get('contract_id') or '', row['reason'])
        if key not in note_keys:
            notes.append(row)
            note_keys.add(key)

    for source in sources:
        comparisons = [*(source.get('original_matching_comparisons') or []),
                       *(source.get('matching_comparisons') or [])]
        for entry in comparisons:
            if not isinstance(entry, dict):
                continue
            decision = str(entry.get('decision') or '')
            if decision.startswith(('ACCEPT', 'ADMIT', 'MATCHED')):
                continue
            identifier = _text(entry.get('contract_id'))
            row = {'decision': decision,
                   'reason': str(entry.get('difference') or decision or 'Settlement terms differ')[:600]}
            if not identifier or identifier.lower() in ('unmatched question', 'candidate'):
                add_note(row)
                continue
            if identifier not in seen:
                rejected = {'id': identifier, **row}
                result.append(rejected)
                seen[identifier] = rejected
            inspection = _retained_source_only_inspection(entry, identifier)
            if inspection is not None:
                previous = source_inspections.get(identifier)
                if previous is not None and previous != inspection:
                    conflicting_inspections.add(identifier)
                    seen[identifier].pop('source_only_evidence', None)
                    seen[identifier]['source_inspection_status'] = 'CONFLICTING_RETAINED_SOURCE_INSPECTIONS'
                source_inspections[identifier] = inspection
                if identifier not in conflicting_inspections:
                    seen[identifier]['source_only_evidence'] = inspection
            semantic = entry.get('independent_semantic_check')
            explanation = _text(semantic.get('proposed_explanation')) if isinstance(semantic, dict) else ''
            if explanation:
                add_note({'contract_id': identifier,
                          'decision': 'UNVERIFIED_MODEL_RULE_REVIEW', 'status': 'UNVERIFIED',
                          'reason': 'Unverified LLM rule-review explanation for ' + identifier + ': ' + explanation[:1600],
                          'evidence_scope': 'SUPPORTING_MODEL_TEXT_NOT_INDEPENDENT_SEMANTIC_CERTIFICATION'})
        for entry in source.get('quarantined_candidates') or []:
            if not isinstance(entry, dict):
                continue
            identifier = _text(entry.get('ticker') or entry.get('contract_id') or entry.get('market_id'))
            reason = _text(entry.get('reason'))[:600]
            if not identifier or identifier.lower() in ('unmatched question', 'candidate') or not reason:
                continue
            if identifier not in seen:
                rejected = {'id': identifier, 'decision': 'SOURCE_GATE_QUARANTINED',
                            'reason': reason,
                            'evidence_scope': 'BOUNDED_SEARCH_SOURCE_GATE_REJECTION_NOT_GLOBAL_MARKET_ABSENCE',
                            'source_gate_reasons': [reason]}
                url = _safe_contract_source_url(entry.get('source_url'), identifier)
                if url:
                    rejected['source_url'] = url
                if _text(entry.get('source_receipt_status')) in {'LOCAL_BYTES_VERIFIED', 'UNVERIFIED', 'UNKNOWN'}:
                    rejected['source_receipt_status'] = entry['source_receipt_status']
                result.append(rejected)
                seen[identifier] = rejected
            else:
                rejected = seen[identifier]
                reasons = rejected.setdefault('source_gate_reasons', [])
                if reason not in reasons:
                    reasons.append(reason)
            # Retain the deterministic exclusion separately from the bounded
            # source-only inspection exported in its own CSV field.
            add_note({'contract_id': identifier, 'decision': 'SOURCE_GATE_QUARANTINED',
                      'reason': 'Source-gate exclusion in bounded search for ' + identifier + ': ' + reason,
                      'evidence_scope': 'BOUNDED_SEARCH_SOURCE_GATE_REJECTION_NOT_GLOBAL_MARKET_ABSENCE'})
    return result[:24], notes[:12]


def _comparison_rejections(event):
    return _comparison_review(event)[0]


def _selected_model(event):
    """Fixed schema precedence, never selection by the most attractive result.

    The complete asset-class-aware table is already the model used by the
    engine's ranking. Selecting a passing alternative after seeing diagnostics
    is deliberately NOT this rule. Partial tables cannot become full-portfolio
    estimates and none of these model scopes certifies a future return.
    """
    table = event.get('outcome_table') or {}
    economics = event.get('economics') or {}
    status = table.get('status')
    identity_gate = event.get('instrument_identity_gate') or table.get('instrument_identity_gate') or {}
    if (identity_gate.get('blocked_symbols') or
            event.get('legacy_fallback_allowed') is False or
            table.get('legacy_fallback_allowed') is False or
            status == 'BLOCKED_INSTRUMENT_IDENTITY'):
        # A rejected identity/classification cannot escape the channel guard
        # by falling back to the old, otherwise numerically complete beta fit.
        return {'model': {**table, 'status': 'BLOCKED_INSTRUMENT_IDENTITY'},
                'source': [], 'value_key': 'pnl', 'bounds_key': 'pnl_interval',
                'scope': 'Instrument identity or asset-class evidence needs review',
                'kind': 'UNAVAILABLE', 'interval_kind': 'UNAVAILABLE',
                'interval_interpretation': ''}
    if status == 'IDENTIFIED_NOMINAL_USD_CASH_ZERO' and table.get('outcomes'):
        return {'model': table, 'source': table['outcomes'],
                'value_key': 'pnl_if_outcome_usd', 'bounds_key': 'pnl_model_range_5_95_usd',
                'scope': 'Fixed nominal USD cash price component', 'kind': 'NOMINAL_CASH_IDENTITY',
                'interval_kind': 'EXACT_NOMINAL_CASH_IDENTITY',
                'interval_interpretation': 'Exact zero nominal price change for fixed USD cash; '
                    'excludes interest income, inflation, purchasing power and foreign exchange.'}
    if status in ('COMPLETE_DECLARED_MODEL', 'COMPLETE_MEASURED_SURPRISE_RESPONSE') and table.get('outcomes'):
        payroll = status == 'COMPLETE_MEASURED_SURPRISE_RESPONSE'
        return {'model': table, 'source': table['outcomes'],
                'value_key': 'pnl_if_outcome_usd', 'bounds_key': 'pnl_model_range_5_95_usd',
                'scope': 'Estimated jobs-announcement response' if payroll else 'Asset-class-aware announcement scenario',
                'kind': 'MEASURED_PAYROLL' if payroll else 'CHANNEL_AWARE',
                'interval_kind': 'MODEL_CONDITIONAL_5_95_ENVELOPE',
                'interval_interpretation': 'A model-conditional parameter/state/market-weight range, '
                    'not a confidence guarantee or a prediction interval for future portfolio returns.'}
    if (event.get('source_defined_scenario') is True and
            status == 'COMPLETE_MEASURED_ANCHOR_MODEL' and table.get('outcomes')):
        return {'model': table, 'source': table['outcomes'],
                'value_key': 'pnl_if_outcome_usd', 'bounds_key': 'pnl_model_range_5_95_usd',
                'scope': 'Separately defined dated general-China tariff-rate scenario',
                'kind': 'MEASURED_ANCHOR',
                'interval_kind': 'HISTORICAL_ANCHOR_CONDITIONAL_ENVELOPE',
                'interval_interpretation': 'Sensitivity to the recorded tariff anchors, bracket states '
                    'and source market weights. Not the impact of the original unmatched target, '
                    'not an independently validated causal effect or future-return prediction interval.'}
    if economics.get('outcome_cases'):
        return {'model': economics, 'source': economics['outcome_cases'],
                'value_key': 'component_pnl_usd', 'bounds_key': 'component_pnl_interval_usd',
                'scope': 'Modelled announcement-day component', 'kind': 'LEGACY_BETA',
                'interval_kind': 'SCENARIO_PARAMETER_ENVELOPE',
                'interval_interpretation': 'A conditional scenario and parameter envelope. '
                    'It is not a future-return prediction interval or guaranteed loss bound.'}
    return {'model': {}, 'source': [], 'value_key': 'pnl', 'bounds_key': 'pnl_interval',
            'scope': 'Not estimated', 'kind': 'UNAVAILABLE',
            'interval_kind': 'UNAVAILABLE', 'interval_interpretation': ''}


def _pnl_interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = (_number(part) for part in value)
    return [low, high] if low is not None and high is not None and low <= high else None


def _contributions(value, model):
    by_symbol = {str(row.get('symbol')): row for row in model.get('positions') or []
                 if isinstance(row, dict) and row.get('symbol')}
    rows = value.get('position_contributions') or value.get('positions') or []
    result = []
    for row in rows:
        if not isinstance(row, dict) or not row.get('symbol'):
            continue
        pnl = _number(row.get('pnl_usd', row.get('component_pnl_usd')))
        if pnl is None:
            continue
        holding = by_symbol.get(str(row['symbol']), {})
        result.append({'symbol': str(row['symbol']),
                       'weight': _number(row.get('weight', holding.get('weight'))),
                       'channel': str(row.get('channel') or holding.get('channel') or 'MODEL_COMPONENT'),
                       'pnl': pnl})
    return result


def _failure_weight(symbols, positions, status):
    if status in ('NOT_TESTED', 'NOT_REPORTED', '', None) or (status == 'FAIL' and not symbols):
        return None
    names = set(symbols)
    return round(sum(p['weight'] for p in positions if p['symbol'] in names), 9)


def _model_comparisons(event, positions, selected_kind):
    result = []
    for key, label in (('economics', 'Daily-beta comparison'),
                       ('announcement_day_economics', 'Announcement-day-beta comparison')):
        model = event.get(key) or {}
        if not model.get('outcome_cases') or (selected_kind == 'LEGACY_BETA' and key == 'economics'):
            continue
        status = str(model.get('scenario_transfer_status') or 'NOT_REPORTED')
        failed = list(model.get('scenario_transfer_failed_symbols') or [])
        result.append({'id': key, 'label': label,
                       'transfer_status': status, 'failed_symbols': failed,
                       'failed_portfolio_weight': _failure_weight(failed, positions, status),
                       'interpretation': 'Historical association diagnostic of this comparison specification; '
                           'not validation of the selected asset-class-aware model or a causal forecast.',
                       'outcomes': [{'id': str(row.get('outcome_id') or ''),
                                     'pnl': _number(row.get('component_pnl_usd')),
                                     'pnl_interval': _pnl_interval(row.get('component_pnl_interval_usd'))}
                                    for row in model['outcome_cases'] if isinstance(row, dict)]})
    return result


def _two_factor_positions(model):
    # Funds priced through their measured SPY and 10Y-yield response, with the holdout evidence.
    rows = []
    for row in (model or {}).get('positions') or []:
        fit = row.get('two_factor_fit') if isinstance(row, dict) and row.get('channel') == 'TWO_FACTOR' else None
        if fit:
            rows.append({'symbol': str(row['symbol']), 'spy_loading': fit.get('spy_loading'),
                         'rate_loading_per_10bp': (fit.get('rate_loading_per_bp') or 0.0) * 10,
                         'holdout_oos_r2': fit.get('holdout_oos_r2'), 'holdout_event_days': fit.get('holdout_event_days'),
                         'holdout_event_rmse': fit.get('holdout_event_rmse'),
                         'holdout_event_zero_rmse': fit.get('holdout_event_zero_rmse'),
                         'issuer_duration_check': deepcopy(fit.get('issuer_duration_check'))})
    return rows


def _exposure_for_event(event, positions, selected):
    named = (event.get('named_exposure_sensitivity') or {}).get('per_symbol') or []
    rows = [{'symbol': str(row['symbol']), 'weight': _number(row.get('portfolio_weight'))}
            for row in named if isinstance(row, dict) and row.get('symbol')
            and _number(row.get('portfolio_weight')) is not None]
    # Drop numerical look-through residues (e.g. 1e-8 of a fund holding itself);
    # they would otherwise be listed as a 0.0% channel.
    rows = [row for row in rows if abs(row['weight']) >= 1e-6]
    if rows:
        return rows, 'NAMED_SOURCE_CHANNELS'
    model_rows = selected['model'].get('positions') or []
    weights = {p['symbol']: p['weight'] for p in positions}
    rows = [{'symbol': str(row['symbol']), 'weight': weights[str(row['symbol'])]}
            for row in model_rows if isinstance(row, dict) and str(row.get('symbol')) in weights]
    if selected['source']:
        return rows or [{'symbol': p['symbol'], 'weight': p['weight']} for p in positions], 'MODEL_COVERED_HOLDINGS'
    return [], 'PORTFOLIO_CONTEXT_ONLY'


def _reliability(selected, positions):
    model = selected['model']
    explicit = model.get('model_reliability')
    if isinstance(explicit, dict):
        return explicit
    if selected['kind'] == 'CHANNEL_AWARE':
        # Old saved reports predate the new metadata. No combined-model test
        # can be inferred from component-level diagnostic results.
        diagnostics = []
        for row in model.get('positions') or []:
            if not isinstance(row, dict):
                continue
            diagnostics.append({'symbol': str(row.get('symbol') or ''),
                                'channel': str(row.get('channel') or ''),
                                'basis': str(row.get('basis') or ''),
                                'daily_holdout_status': str(row.get('daily_holdout_status') or 'NOT_REPORTED'),
                                'event_day_holdout': str(row.get('event_day_holdout') or 'NOT_REPORTED')})
        return {'status': 'NOT_INDEPENDENTLY_VALIDATED',
                'combined_model_holdout_available': False, 'forward_prediction_certified': False,
                'positions': diagnostics,
                'interpretation': 'Asset-class-aware scenario model. Component association checks '
                    'do not constitute an independent validation of the combined model.'}
    return {'status': 'NOT_REPORTED' if not model.get('scenario_transfer_status') else model['scenario_transfer_status'],
            'forward_prediction_certified': False}


def _modeled_outcomes(event):
    distribution = event.get('distribution') or {}
    if not _displayable_distribution(event):
        return []
    selected = _selected_model(event)
    source = selected['source']
    value_key, bounds_key, scope = selected['value_key'], selected['bounds_key'], selected['scope']
    included = set(_distribution_source_ids(distribution))
    sensitivity_metadata = event.get('model_sensitivity') or {}
    sensitivities = {row.get('outcome_id'): row for row in
                     sensitivity_metadata.get('outcomes') or [] if isinstance(row, dict)}
    rows = []
    for value in source:
        if not isinstance(value, dict):
            continue
        label = str(value.get('label') or '')
        q = _probability(value.get('q_reference'))
        pnl = _number(value.get(value_key))
        # An unquoted tail/bucket is never promoted to a fully priced outcome.
        if (q is None or q < .01 or pnl is None or
                'unquoted' in label.lower() or 'ambiguous' in label.lower()):
            continue
        outcome_id = str(value.get('outcome_id') or '')
        source_ids = [str(i) for i in value.get('source_contract_ids') or [] if i]
        if not source_ids and outcome_id in included:
            source_ids = [outcome_id]
        q_range = _interval(value.get('q_interval'))
        pnl_range = _pnl_interval(value.get(bounds_key))
        sensitivity = sensitivities.get(outcome_id) or {}
        stress_range = _pnl_interval(sensitivity.get('pnl_model_sensitivity_range_usd'))
        rows.append({'id': outcome_id, 'label': label,
                     'q_reference': q, 'q_interval': q_range,
                     'pnl': pnl, 'pnl_interval': pnl_range,
                     'pnl_interval_kind': selected['interval_kind'],
                     'pnl_interval_interpretation': selected['interval_interpretation'],
                     'pnl_model_sensitivity_range_usd': stress_range,
                     'model_sensitivity_kind': sensitivity_metadata.get('kind') if stress_range else None,
                     'no_response_alternative_included': (stress_range is not None and
                         sensitivity_metadata.get('no_response_alternative_included') is True),
                     'direction_robust_to_declared_set': sensitivity.get('direction_robust_to_declared_set'),
                     'model_sensitivity_interpretation': (
                         'Range over an explicitly declared assumption set including no response; '
                         'not a confidence or prediction interval.' if sensitivity else ''),
                     'position_contributions': _contributions(value, selected['model']),
                     'direction_uncertain': (pnl_range[0] < 0 < pnl_range[1]) if pnl_range else None,
                     'scope': scope,
                     'contract_id': source_ids[0] if source_ids else None,
                     'source_contract_ids': source_ids,
                     **_source_quote_metadata(distribution, outcome_id),
                     'partial': False, 'no_channel': False})
    return rows


def _display_probability_coverage(event, displayed):
    """Disclose Q-mass omitted by the UI; never renormalize a quoted bucket."""
    distribution = event.get('distribution') or {}
    if not _displayable_distribution(event):
        return None
    source = _selected_model(event)['source'] or distribution.get('atoms') or []
    source_q = [_probability(row.get('q_reference')) for row in source if isinstance(row, dict)]
    if not source_q or any(q is None for q in source_q):
        return None
    total = sum(source_q)
    shown = sum(row['q_reference'] for row in displayed)
    if total > 1 + 1e-5 or shown > total + 1e-7:
        return None
    hidden = max(0.0, total - shown)
    return {'source_reference_mass': round(total, 9),
            'shown_reference_mass': round(shown, 9),
            'omitted_reference_mass': round(hidden, 9),
            'source_outcome_count': len(source),
            'shown_outcome_count': len(displayed),
            'coverage_complete': hidden <= 1e-6 and abs(total - 1) <= 1e-5,
            'method': 'Q_PRICE_WEIGHTS_NOT_PHYSICAL_PROBABILITIES_NO_RENORMALIZATION'}


def _distribution_source_ids(distribution):
    return list(dict.fromkeys(str(value) for value in
        (distribution.get('source_contract_ids') or distribution.get('contract_ids') or []) if value))


def _displayable_distribution(event):
    quality = event.get('quality') or {}
    table = event.get('outcome_table') or {}
    tariff = event.get('tariff_review') or {}
    scope_sources = [event, quality, table, tariff,
                     event.get('target_scope_review') or {},
                     quality.get('target_scope_review') or {},
                     table.get('target_scope_review') or {},
                     tariff.get('target_scope_review') or {}]
    if (event.get('source_defined_scenario') is not True and
            any(isinstance(source, dict) and
                source.get('source_scenario_is_exact_target_match') is False for source in scope_sources)):
        return False
    distribution = event.get('distribution') or {}
    if distribution.get('status') == 'CONDITIONAL_SOURCE_DISTRIBUTION':
        return True
    return (event.get('source_defined_scenario') is True and
            distribution.get('status') == 'COHERENT_BRACKET_DISTRIBUTION')


def _market_outcomes(event):
    """Market weights do not depend on whether a response model succeeded."""
    if not _displayable_distribution(event):
        return []
    distribution = event['distribution']
    included = set(_distribution_source_ids(distribution))
    rows = []
    for atom in distribution.get('atoms') or []:
        if not isinstance(atom, dict):
            continue
        q = _probability(atom.get('q_reference'))
        label = str(atom.get('label') or '')
        if q is None or q < .01 or re.search(r'unquoted|ambiguous', label, re.I):
            continue
        identifier = str(atom.get('outcome_id') or '')
        source_ids = [str(i) for i in atom.get('source_contract_ids') or [] if i]
        if not source_ids and identifier in included:
            source_ids = [identifier]
        rows.append({'id': identifier, 'label': label, 'q_reference': q,
                     'q_interval': _interval(atom.get('q_interval')),
                     'pnl': None, 'pnl_interval': None,
                     'source_contract_ids': source_ids,
                     'contract_id': source_ids[0] if source_ids else None,
                     **_source_quote_metadata(distribution, identifier),
                     'scope': 'Market pricing evidence; impact independently assessed'})
    return rows


def _events_with_related_source_scenarios(events, source_candidates=()):
    """Expose actual retained tariff tables without changing the target event.

    The input report is not mutated. Equivalent source tables are displayed
    once and linked to every related hypothesis. Missing tables are not rebuilt
    from an export or populated from a different portfolio.
    """
    originals = [deepcopy(event) for event in events if isinstance(event, dict)]
    related = {}
    for original in originals:
        review = original.get('tariff_review') or {}
        if not isinstance(review, dict):
            continue
        source = review.get('related_general_rate_scenario') or review
        distribution = source.get('distribution') or {}
        if distribution.get('status') != 'COHERENT_BRACKET_DISTRIBUTION':
            continue
        table = source.get('outcome_table') or {}
        date = _date(distribution.get('measurement_date'))
        if not date or not _distribution_source_ids(distribution):
            continue
        scope = review.get('target_scope_review') or {}
        original['source_scenario_is_exact_target_match'] = False
        original['reason'] = str(scope.get('reason') or
            'The retrieved contracts measure a general China tariff rate on one date, '
            'not the original tariff action/window. See the separately defined source scenario.')
        digest = hashlib.sha256(json.dumps({'distribution': distribution, 'table': table},
                                 sort_keys=True, allow_nan=False).encode()).hexdigest()[:14]
        identifier = 'source-china-rate-' + date + '-' + digest
        target_id = str(original.get('event_id') or '')
        original.setdefault('related_source_scenario_ids', []).append(identifier)
        if identifier in related:
            existing = related[identifier]
            if target_id not in existing['related_to_event_ids']:
                existing['related_to_event_ids'].append(target_id)
            # An identical numeric table cannot launder another parent's hard
            # identity/class restriction by being encountered first.
            incoming_gate = original.get('instrument_identity_gate') or {}
            retained_gate = existing['instrument_identity_gate']
            for key in ('blocked_symbols', 'identity_blocked_symbols', 'asset_class_blocked_symbols'):
                restricted = sorted(set(retained_gate.get(key) or []) | set(incoming_gate.get(key) or []))
                if restricted:
                    retained_gate[key] = restricted
                    retained_gate['status'] = 'BLOCKED_INSTRUMENT_IDENTITY'
            if original.get('legacy_fallback_allowed') is False:
                existing['legacy_fallback_allowed'] = False
            continue
        question = ('General US tariff rate on Chinese goods on ' + date +
                    ', under the source contract rules; item-specific duties excluded.')
        source_distribution = {**deepcopy(distribution), 'probability_question': question}
        source_distribution['source_quote_evidence'] = _retained_tariff_quote_evidence(
            distribution, [*(review.get('candidate_reviews') or []), *source_candidates])
        related[identifier] = {
            'event_id': identifier, 'title': 'General China tariff rate on ' + date,
            'family': 'TRADE_TARIFFS', 'event_date': date,
            'original_question': question, 'source_defined_scenario': True,
            'related_to_event_id': target_id, 'related_to_event_ids': [target_id],
            'relation_reason': 'Separate settlement question and date. This does not price '
                               'the original tariff-action, product or country hypothesis.',
            'source_scenario_is_exact_original_target_match': False,
            'distribution': source_distribution,
            'outcome_table': deepcopy(table),
            'status': str(table.get('status') or 'SOURCE_MARKET_WITHOUT_RESPONSE_MODEL'),
            'reason': ('This is the separate general-China tariff-rate scenario on ' + date +
                       '. Its source-contract outcomes do not price the related tariff-action, '
                       'country or product hypotheses.'),
            'instrument_identity_gate': deepcopy(original.get('instrument_identity_gate') or
                                                 table.get('instrument_identity_gate') or {}),
            'legacy_fallback_allowed': original.get('legacy_fallback_allowed',
                                                   table.get('legacy_fallback_allowed', True)),
        }
    return originals + list(related.values())


def _definition_review(event):
    """Keep the historical adapter API; the definition gate is independently tested."""
    return review_event_definition(event)


def _lookthrough(report):
    analysis = report.get('default_analysis') or {}
    rows = []
    for entry in analysis.get('aggregated_issuer_exposure') or []:
        if not isinstance(entry, dict) or not entry.get('symbol'):
            continue
        weight = _number(entry.get('portfolio_weight'))
        if weight is None:
            continue
        rows.append({'symbol': str(entry['symbol']), 'weight': weight,
                     'direct': _number(entry.get('direct_weight')) or 0,
                     'via': {str(k): _number(v) for k, v in (entry.get('via_etf') or {}).items()
                             if _number(v) is not None},
                     'unresolved': entry.get('basis') == 'UNRESOLVED_FUND_MASS_KEPT_AS_PARENT_UNIT',
                     'basis': str(entry.get('basis') or '')})
    return rows


def _fund_details(report):
    details = []
    analysis = report.get('default_analysis') or {}
    for fund, value in (analysis.get('fund_signed_compositions') or {}).items():
        if not isinstance(value, dict):
            continue
        segments = []
        for segment, row in (value.get('segments') or {}).items():
            weight = _number(row.get('net_weight')) if isinstance(row, dict) else None
            if weight is not None and abs(weight) >= .001:
                segments.append({'name': str(segment).replace('_', ' ').capitalize(), 'weight': weight})
        segments.sort(key=lambda item: -abs(item['weight']))
        details.append({'symbol': str(fund), 'date': _date(value.get('effective_date')),
                        'segments': segments[:7],
                        'residual': _number(value.get('unreported_or_residual_nav_weight'))})
    return details


def _identity_review(report):
    """Small public identity audit; never forward raw provider/private fields.

    Older saved reports already contain source-type conflict diagnostics. They
    must not regain complete stock-model numbers just because offline replay
    could not read the original local provider files.
    """
    source = report.get('instrument_identity') or {}
    if not isinstance(source, dict):
        source = {}
    keys = ('symbol', 'declared_asset_type', 'resolved_asset_type',
            'provider_instrument_type', 'status', 'reason',
            'local_response_verified', 'source_url', 'source_response_sha256')
    rows = [{key: row[key] for key in keys if key in row}
            for row in (source.get('positions') or source.get('audit') or [])
            if isinstance(row, dict) and row.get('symbol')]
    originals = {str(row.get('symbol')): row for row in source.get('positions') or []
                 if isinstance(row, dict) and row.get('symbol')}
    for row in rows:
        original = originals.get(str(row['symbol']), {})
        row['local_response_verified'] = original.get('source_verified') is True or row.get('local_response_verified') is True
        provenance = original.get('source') or {}
        if isinstance(provenance, dict):
            for key in ('source_url', 'source_response_sha256', 'metadata_as_of', 'retrieved_at'):
                if isinstance(provenance.get(key), str):
                    row[key] = provenance[key]
    blocked = {str(symbol) for symbol in source.get('blocked_symbols') or []}
    for position in (report.get('exposures') or {}).get('positions') or []:
        if not isinstance(position, dict):
            continue
        declared = position.get('asset_type')
        provider = position.get('provider_instrument_type')
        status = position.get('classification_status')
        conflict = (status == 'DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED' and
                    declared in {'stock', 'etf'} and
                    provider != ('CS' if declared == 'stock' else 'ETF') and
                    not (declared == 'etf' and provider == 'ETV'))
        if not conflict and status not in {'UNSUPPORTED_PROVIDER_INSTRUMENT',
                                           'LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED'}:
            continue
        symbol = str(position.get('symbol') or '')
        if not symbol:
            continue
        blocked.add(symbol)
        if not any(row.get('symbol') == symbol for row in rows):
            rows.append({'symbol': symbol, 'declared_asset_type': declared,
                         'resolved_asset_type': None, 'provider_instrument_type': provider,
                         'status': 'SOURCE_IDENTITY_CONFLICT_REQUIRES_REANALYSIS',
                         'reason': 'Saved source metadata conflicts with the modeled instrument type or supported valuation scope.'})
    if not rows and not blocked and not source:
        return None
    return {'status': 'BLOCKED_INSTRUMENT_IDENTITY' if blocked else str(source.get('status') or 'RECORDED_IDENTITY_REVIEW'),
            'positions': rows, 'blocked_symbols': sorted(blocked),
            'scope': 'Instrument identity is separate from economic-model validation.'}


def _venue_retrieval_window(report, venue):
    """Earliest and latest recorded response times of one venue's geopolitical search."""
    times = sorted(_text(row.get('retrieved_at')) for row in
                   ((report.get('geopolitical_search') or {}).get('source_records') or [])
                   if isinstance(row, dict) and row.get('venue') == venue and _source_instant(row.get('retrieved_at')))
    return [times[0], times[-1]] if times else None


def _contract_first_public_children(event, portfolio_positions, notional):
    """Make separate source-contract cards without upgrading the parent event.

    These are existing declared stress scenarios, not fitted or newly sourced
    forecasts. Preserve the contract's own settlement and Q interval; require
    two complementary source-named outcomes and exact Q-centering. Incomplete
    position models, mismatched monetary units and malformed evidence abstain.
    """
    result = []
    parent_id = str(event.get('event_id') or '')
    if not parent_id or str(event.get('family')) not in {'TAIWAN_SUPPLY', 'EXPORT_CONTROLS'}:
        return result
    for scenario in eligible_contract_scenarios(event):
        contract = scenario.get('contract') or {}
        identifier, question = _text(contract.get('contract_id')), _text(contract.get('market_question'))
        venue, until = _text(contract.get('venue')), _text(contract.get('contract_end_time'))
        rules_hash = _text(contract.get('rules_sha256'))
        if (not identifier or not question or venue not in {'Kalshi', 'Polymarket'}
                or not re.fullmatch(r'[a-fA-F0-9]{64}', rules_hash)
                or not _source_instant(until) or not _interval(contract.get('quote_interval'))):
            continue
        rows = scenario.get('outcomes') or []
        if not isinstance(rows, list) or len(rows) != 2:
            continue
        outcomes, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                break
            row_id = _text(row.get('outcome_id'))
            polarity = row_id.removeprefix(identifier + ':')
            q = _probability(row.get('q_reference'))
            q_band = _interval(row.get('q_interval'))
            pnl = _number(row.get('pnl_if_outcome_usd'))
            pnl_band = _pnl_interval(row.get('pnl_model_range_5_95_usd'))
            if (polarity not in {'YES', 'NO'} or polarity in seen or q is None
                    or q_band is None or not q_band[0] - 1e-10 <= q <= q_band[1] + 1e-10
                    or pnl is None or pnl_band is None or not pnl_band[0] <= pnl <= pnl_band[1]):
                break
            seen.add(polarity)
            outcomes.append({'id': row_id, 'label': _text(row.get('label')) or polarity,
                             'q_reference': q, 'q_interval': q_band, 'pnl': pnl,
                             'pnl_interval': pnl_band,
                             'pnl_interval_kind': 'DECLARED_STRESS_MONTE_CARLO_5_95',
                             'pnl_interval_interpretation': 'Quantiles from declared stress ranges and parameter assumptions; not a calibrated future-return confidence interval.',
                             'position_contributions': [],
                             'scope': 'Contract-specific hypothetical complete-portfolio response, conditional on assumed stress sizes.',
                             'contract_id': identifier, 'source_contract_ids': [identifier],
                             'venue': venue, 'question': question,
                             'question_basis': 'SOURCE_CONTRACT_TEXT',
                             'quote_scope': _text(contract.get('quote_scope')) or 'Dated indicative source quote',
                             'quote_received_at': None,
                             'partial': False, 'no_channel': False})
        if len(outcomes) != 2 or seen != {'YES', 'NO'}:
            continue
        by_polarity = {row['id'].removeprefix(identifier + ':'): row for row in outcomes}
        yes, no = by_polarity['YES'], by_polarity['NO']
        q = yes['q_reference']
        quote = _interval(contract['quote_interval'])
        contrast = _number(scenario.get('delta_pnl_usd'))
        positions = scenario.get('positions') or []
        if (abs(q + no['q_reference'] - 1.) > 1e-8
                or any(abs(a - b) > 1e-8 for a, b in zip(quote, yes['q_interval']))
                or any(abs(a - b) > 1e-8 for a, b in zip([1-quote[1], 1-quote[0]], no['q_interval']))
                or contrast is None or abs(yes['pnl'] - no['pnl'] - contrast) > 1e-5 * max(1., abs(contrast))
                or abs(q * yes['pnl'] + (1-q) * no['pnl']) > 1e-5 * max(1., abs(contrast))
                or not isinstance(positions, list) or not positions
                or any(not isinstance(row, dict) or _number(row.get('pnl_usd')) is None for row in positions)
                or abs(sum(row['pnl_usd'] for row in positions) - contrast) > 1e-5 * max(1., abs(contrast))):
            continue
        child = {'id': 'source-contract-' + venue.lower() + '-' + identifier,
                 'title': venue + ': ' + question, 'family': str(event['family']),
                 'date': _date(until), 'original_question': question,
                 'source_defined_scenario': True, 'related_to_event_id': parent_id,
                 'relation_reason': 'Different source-defined settlement and horizon; does not price the original broader target.',
                 'outcomes': outcomes, 'market_outcomes': [],
                 'exposure': [{'symbol': p['symbol'], 'weight': p['weight']} for p in portfolio_positions],
                 'exposure_basis': 'ORIGINAL_POSITIONS_IN_DECLARED_CONTRACT_SCENARIO',
                 'admitted': [{'id': identifier, 'venue': venue, 'question': question,
                               'quote_scope': outcomes[0]['quote_scope'], 'quote_received_at': None}],
                 'rejected': [], 'review_notes': [],
                 'quality': {'model_status': 'DECLARED_CONTRACT_SCENARIO',
                             'model_kind': 'DECLARED_CONTRACT_SCENARIO',
                             'transfer_status': 'NOT_EMPIRICALLY_VALIDATED',
                             'reason': 'Source-defined YES/NO stress under stated economic assumptions, not a verified prediction or the parent risk.',
                             'scope': 'MODEL_CONDITIONAL_FULL_PORTFOLIO_STRESS_NOT_FORECAST',
                             'source_venue': venue, 'quote_scope': outcomes[0]['quote_scope'],
                             'quote_received_at': None, 'contract_end_time_utc': until,
                             'rules_sha256': rules_hash, 'physical_probability_P': None,
                             'future_total_return_prediction_available': False,
                             'assumptions': [str(x) for x in scenario.get('assumptions') or [] if isinstance(x, str)][:12],
                             'declared_scenario_window': str((scenario.get('scenario') or {}).get('window') or ''),
                             'uncertainty': str((scenario.get('uncertainty') or {}).get('interpretation') or '')}}
        admission = event_model_admission(child)
        if admission['state'] != 'CONDITIONAL_SCENARIO' or not admission['primary_impact_allowed']:
            continue
        child['quality']['model_admission'] = admission
        result.append(child)
    return result


def _partial_priced_component(event, report):
    """Known-position P&L of a macro table blocked only by unresolved fund classes.

    The engine already prices every position it can (own measured equity
    association or source-bound duration). When a fund's asset class is
    unresolved, the complete total stays inadmissible; this component shows the
    priced part and names the unpriced funds instead of discarding both.
    """
    table = (event or {}).get('outcome_table')
    gate = (event or {}).get('instrument_identity_gate') or {}
    if (not isinstance(table, dict) or table.get('status') != 'PARTIAL_KNOWN_POSITIONS_ONLY'
            or gate.get('identity_blocked_symbols') or not table.get('outcomes')):
        return None
    notional = _number(table.get('notional_usd'))
    weights = {p.get('symbol'): p.get('weight') for p in (report.get('portfolio') or {}).get('positions') or []}
    missing = [str(s) for s in table.get('missing_symbols') or []]
    outcomes = []
    for row in table['outcomes']:
        value = _number(row.get('known_positions_only_pnl_usd'))
        if value is None:
            return None
        outcomes.append({'label': str(row.get('label') or ''),
                         'source_contract_ids': [str(x) for x in row.get('source_contract_ids') or []],
                         'priced_pnl_usd': value,
                         'contributions': [{'symbol': str(c.get('symbol')), 'channel': str(c.get('channel')),
                                            'pnl_usd': _number(c.get('pnl_usd'))}
                                           for c in row.get('position_contributions') or []]})
    return {'schema': 'castle.option1.partial_priced_component.v1',
            'notional_usd': notional,
            'priced_weight': 1.0 - (_number(table.get('missing_weight')) or 0.0),
            'unpriced': [{'symbol': s, 'weight': _number(weights.get(s))} for s in missing],
            'outcomes': outcomes,
            'scope': ('Priced positions only, at the point model parameters. Unpriced funds have no identified '
                      'event response and are not treated as zero; this is not a whole-portfolio total.')}


def _documented_relevance_weight(risk):
    """Named source-linked NAV weight, not an event loss or probability."""
    if not isinstance(risk, dict) or risk.get('source_defined_scenario') is True:
        return None
    if risk.get('what_if_explorer') or risk.get('exposure_basis') != 'NAMED_SOURCE_CHANNELS':
        return None
    if ((risk.get('quality') or {}).get('definition_review') or {}).get('status') != 'DEFINED_FOR_REVIEW':
        return None
    links = risk.get('exposure')
    if not isinstance(links, list) or not links:
        return None
    total, seen = 0.0, set()
    for row in links:
        if not isinstance(row, dict):
            return None
        symbol = str(row.get('symbol') or '').strip().upper()
        weight = _number(row.get('weight'))
        if not symbol or symbol in seen or weight is None or not 0 <= weight <= 1:
            return None
        seen.add(symbol)
        total += weight
    return min(1.0, total) if total <= 1 + 1e-9 else None


def _order_live_risks_by_documented_relevance(risks):
    """Material source links first; one distinct event family before repeats.

    Multiple releases from one macro family must not occupy the entire first
    screen merely because they have dates. Ranking is a display rule, never an
    inferred probability, expected loss, or newly validated discovery model.
    """
    material, dated_macro, small_links, unverified, auxiliary = [], [], [], [], []
    scheduled = {'FED_POLICY', 'FOMC', 'US_INFLATION', 'CPI',
                 'US_PAYROLLS', 'PAYROLLS', 'US_LABOR', 'LABOR_MARKET'}
    for index, risk in enumerate(risks):
        if risk.get('source_defined_scenario') is True or risk.get('what_if_explorer'):
            auxiliary.append(risk)
            continue
        weight = _documented_relevance_weight(risk)
        if weight is not None:
            risk = {**risk, 'display_relevance': {
                'basis': 'NAMED_SOURCE_CHANNELS', 'linked_weight': weight,
                'event_effect_identified': False,
                'meaning': 'Source-linked portfolio weight, not return, loss or event probability'}}
        important = weight is not None and weight >= .25
        family = str(risk.get('family') or '').strip().upper()
        bucket = (material if important else dated_macro if family in scheduled
                  else small_links if weight is not None else unverified)
        bucket.append((-(weight or 0), index, risk))

    def diverse(rows):
        seen, first, repeats = set(), [], []
        for _, index, risk in rows:
            family = str(risk.get('family') or '').strip().upper()
            # OTHER / generic geography are not homogeneous economic risks.
            key = family if family not in ('', 'OTHER', 'GEOPOLITICAL') else f'UNIQUE:{index}'
            (repeats if key in seen else first).append(risk)
            seen.add(key)
        return first + repeats

    # A 10% incidental fund sleeve is not allowed to push speculative China
    # headlines ahead of major macro risks in a 90% Treasury portfolio.
    return (diverse(sorted(material)) + diverse(dated_macro) +
            diverse(small_links) + diverse(unverified) + auxiliary)


_ACCOUNTING_SCOPE = ('Original-position arithmetic only. A balanced model decomposition '
                     'does not validate event causality, probability, future P&L, or market pricing.')


def _position_accounting_audit(positions, outcome):
    """Check an event's modeled dollar rows against original, once-valued holdings.

    Never impute missing holdings to zero. This certificate is *not* economic
    validation: it verifies identity, position mass and a sum of model outputs.
    """
    base = {'schema': 'castle.option1.position_accounting.v1',
            'scope': _ACCOUNTING_SCOPE}
    pnl = _number(outcome.get('pnl')) if isinstance(outcome, dict) else None
    rows = outcome.get('position_contributions') if isinstance(outcome, dict) else None
    expected = {}
    for row in positions:
        if not isinstance(row, dict):
            return {**base, 'status': 'INVALID', 'reason': 'Invalid original holding.'}
        name = _text(row.get('symbol')).upper()
        weight = _number(row.get('weight'))
        if not name or name in expected or weight is None or weight < 0:
            return {**base, 'status': 'INVALID', 'reason': 'Invalid or duplicated original holding.'}
        if weight > 0:
            expected[name] = weight
    if not expected or not math.isclose(math.fsum(expected.values()), 1.0, abs_tol=1e-6):
        return {**base, 'status': 'INVALID', 'reason': 'Original portfolio weights do not reconcile.'}
    if pnl is None:
        return {**base, 'status': 'NOT_AUDITABLE', 'reason': 'No numeric total was retained.'}
    if not isinstance(rows, list) or not rows:
        return {**base, 'status': 'NOT_AUDITABLE', 'reason': 'No original-position model contributions were retained.'}
    covered, dollars = set(), []
    for row in rows:
        if not isinstance(row, dict):
            return {**base, 'status': 'INVALID', 'reason': 'Invalid model contribution.'}
        name = _text(row.get('symbol')).upper()
        weight, value = _number(row.get('weight')), _number(row.get('pnl'))
        if (name not in expected or name in covered or weight is None or value is None
                or not math.isclose(weight, expected[name], abs_tol=1e-6)):
            return {**base, 'status': 'INVALID',
                    'reason': 'Duplicate, unknown, nonfinite, or weight-mismatched model contribution.'}
        covered.add(name)
        dollars.append(value)
    reconciled = math.fsum(dollars)
    missing = sorted(set(expected) - covered)
    mass = math.fsum(expected[name] for name in covered)
    if not math.isclose(reconciled, pnl, rel_tol=1e-8, abs_tol=1e-6):
        return {**base, 'status': 'INVALID',
                'reason': 'Position contributions do not sum to the reported model total.',
                'covered_weight': mass, 'missing_symbols': missing}
    return {**base, 'status': 'FULL_RECONCILED' if not missing else 'PARTIAL_COVERAGE',
            'covered_weight': mass, 'missing_symbols': missing,
            'checked_original_positions': len(covered)}


def live_report_to_case(report):
    """Convert a real *public derived* engine report to the recorded UI schema.

    This function deliberately does not call any provider or model. The caller
    must run public_assessment_view before handing it to the browser.
    """
    if report.get('schema') != 'castle.option1.live_assessment.v2':
        raise ValueError('Unsupported Castle live-assessment schema')
    if not report.get('export_scope', '').startswith('DERIVED_'):
        raise ValueError('Browser conversion requires public_assessment_view first')
    portfolio = report.get('portfolio') or {}
    total = _number(portfolio.get('notional'))
    positions = portfolio.get('positions') or []
    if total is None or total <= 0 or not positions:
        raise ValueError('Portfolio or notional is invalid')
    simplified = []
    for entry in positions:
        weight = _number(entry.get('weight'))
        symbol = str(entry.get('symbol') or '').upper()
        kind = str(entry.get('asset_type') or '').lower()
        if weight is None or weight < 0 or kind not in ('stock', 'etf', 'cash') or not symbol:
            raise ValueError('Invalid portfolio holding')
        simplified.append({'symbol': symbol, 'weight': weight, 'type': kind})
    default = report.get('default_analysis') or {}
    identity = _identity_review(report)
    research_events = {}
    for research_event in report.get('events') or []:
        if isinstance(research_event, dict) and _text(research_event.get('event_id')):
            research_events.setdefault(_text(research_event['event_id']), []).append(research_event)
    quote_candidates = [candidate for target in (report.get('geopolitical_search') or {}).get('targets') or []
                        if isinstance(target, dict)
                        for candidate in target.get('candidates') or [] if isinstance(candidate, dict)]
    risks = []
    for index, event in enumerate(_events_with_related_source_scenarios(
            default.get('events') or [], quote_candidates)):
        if not isinstance(event, dict):
            continue
        if identity and identity['blocked_symbols']:
            event = {**event, 'legacy_fallback_allowed': False,
                     'instrument_identity_gate': {
                         'status': 'BLOCKED_INSTRUMENT_IDENTITY',
                         'blocked_symbols': identity['blocked_symbols'],
                         'reason': 'Instrument identity needs review for ' + ', '.join(identity['blocked_symbols']) +
                                   '. Re-analyze with source-verified investment types; saved stock-model values are not promoted.'}}
        distribution = event.get('distribution') or {}
        selected = _selected_model(event)
        model = selected['model']
        admission = event_model_admission(event)
        sensitivity = {'status': 'UNAVAILABLE',
                       'reason': 'A complete supported macro model and its source state constraints are required.'}
        if (selected['kind'] == 'CHANNEL_AWARE' and admission['primary_impact_allowed']):
            from .scenario_sensitivity import build_sensitivity_envelope
            sensitivity = build_sensitivity_envelope(model, distribution)
        event['model_sensitivity'] = sensitivity
        modeled = _modeled_outcomes(event)
        for outcome in modeled:
            outcome['position_accounting_audit'] = _position_accounting_audit(simplified, outcome)
        market = _market_outcomes(event)
        definition = _definition_review(event)
        if definition['status'] != 'DEFINED_FOR_REVIEW':
            modeled = []
            market = []
            admission = {**admission, 'state': 'UNAVAILABLE', 'primary_impact_allowed': False,
                         'economic_ranking_allowed': False, 'reason': definition['reason']}
        coverage = _display_probability_coverage(event, market or modeled)
        transfer = str(model.get('scenario_transfer_status') or
                       ('NOT_TESTED' if selected['kind'] in ('CHANNEL_AWARE', 'MEASURED_ANCHOR') else 'NOT_REPORTED'))
        failed_symbols = list(model.get('scenario_transfer_failed_symbols') or [])
        failed_weight = _failure_weight(failed_symbols, simplified, transfer)
        reliability = _reliability(selected, simplified)
        portfolio_transfer = str((reliability.get('portfolio_transfer_check') or {}).get('status') or 'NOT_REPORTED')
        failure_scope = ('PORTFOLIO_AND_HOLDINGS' if portfolio_transfer == 'FAIL' and failed_symbols else
                         'PORTFOLIO_ONLY' if portfolio_transfer == 'FAIL' else
                         'HOLDINGS' if failed_symbols else 'NONE_OR_NOT_REPORTED')
        available_contracts = (_distribution_source_ids(distribution)
                               if _displayable_distribution(event) and not definition['reason'] else [])
        quote_metadata = _source_quote_metadata(distribution)
        source_label = str(distribution.get('probability_question') or 'Market price distribution')
        exposure, exposure_basis = _exposure_for_event(event, simplified, selected)
        rejected, review_notes = _comparison_review(
            event, research_events.get(_text(event.get('event_id')), ()))
        identity_gate = event.get('instrument_identity_gate') or (event.get('outcome_table') or {}).get('instrument_identity_gate') or {}
        reason = str((identity_gate.get('reason') if identity_gate.get('blocked_symbols') else '') or
                     event.get('reason') or (event.get('market_search') or {}).get('reason') or
                     distribution.get('reason') or '')
        if not modeled and not reason:
            reason = admission['reason']
        source_status = str(model.get('status') or event.get('status') or '')
        if admission['state'] == 'RELATED_SOURCE_ONLY':
            source_status = 'ORIGINAL_TARGET_NOT_PRICED_RELATED_SCENARIO_SEPARATE'
            reason = reason or admission['reason']
        risk = {'id': str(event.get('event_id') or f'default-{index}'),
                'title': str(event.get('title') or 'Market event'),
                'family': str(event.get('family') or 'OTHER'),
                'date': _date(event.get('event_date')),
                'original_question': str(event.get('original_question') or ''),
                'outcomes': modeled,
                # Numeric outcomes already retain their market side even when
                # their dollars are diagnostic-only. Avoid duplicating the
                # same ladder in the browser payload; market-only events need
                # an independent list because they have no numerical rows.
                'market_outcomes': [] if modeled else market,
                'exposure': exposure, 'exposure_basis': exposure_basis,
                'two_factor_positions': _two_factor_positions(model),
                'admitted': [{'id': identifier, **_source_quote_metadata(distribution, identifier),
                              'question': _source_quote_metadata(distribution, identifier)['question'] or source_label}
                             for identifier in available_contracts],
                'rejected': rejected, 'review_notes': review_notes,
                'quality': {
                    'model_status': source_status,
                    'model_kind': selected['kind'],
                    'model_admission': admission,
                    'model_sensitivity': {key: deepcopy(value) for key, value in sensitivity.items()
                                          if key not in ('outcomes', 'parameter_sets')},
                    'selection_policy': 'COMPLETE_ASSET_CLASS_AWARE_TABLE_BEFORE_LEGACY_COMPONENT_MODEL; '
                                        'FIXED_BY_SCHEMA_NOT_BY_DIAGNOSTIC_SUCCESS',
                    'model_reliability': reliability,
                    'model_comparisons': _model_comparisons(event, simplified, selected['kind']),
                    'transfer_status': transfer,
                    'portfolio_transfer_status': portfolio_transfer,
                    'failure_scope': failure_scope,
                    'failed_symbols': failed_symbols,
                    'failed_portfolio_weight': failed_weight,
                    'reason': reason,
                    'instrument_identity_gate': identity_gate,
                    'distribution_display_coverage': coverage,
                    'definition_review': definition,
                    'scope': modeled[0]['scope'] if modeled else 'Not estimated',
                    'future_total_return_prediction_available': False,
                    'quote_received_at': quote_metadata['quote_received_at'],
                    'quote_scope': quote_metadata['quote_scope'],
                    'source_venue': quote_metadata['venue'],
                    'probability_kind': 'Q_PRICE_IMPLIED_NOT_PHYSICAL_EVENT_PROBABILITY'
                }}
        for key in ('source_defined_scenario', 'related_to_event_id', 'related_to_event_ids',
                    'relation_reason', 'source_scenario_is_exact_original_target_match',
                    'related_source_scenario_ids'):
            if key in event:
                risk[key] = deepcopy(event[key])
        from .business_interruption_assessment import component_view
        component = component_view(event.get('business_interruption_component'),
                                   event_id=risk['id'], portfolio_notional=total,
                                   portfolio_positions=simplified)
        if component is not None:
            risk['business_interruption_component'] = component
        from .mexico_tariff_transmission import mexico_component_view
        mexico = mexico_component_view(event.get('mexico_tariff_component'),
                                      event_id=risk['id'], portfolio_notional=total,
                                      portfolio_positions=simplified)
        if mexico is not None:
            risk['mexico_tariff_component'] = mexico
        risks.append(risk)
    # Existing contract-first work is displayed on the contract's own card;
    # the original broader question remains visibly unresolved.
    source_cards, source_ids = [], set()
    original_by_id = {str(e.get('event_id')): e for e in default.get('events') or [] if isinstance(e, dict)}
    for parent in risks:
        original = original_by_id.get(parent['id'])
        if original is None:
            continue
        for child in _contract_first_public_children(original, simplified, total):
            if child['id'] in source_ids:
                continue
            window = _venue_retrieval_window(report, child['quality'].get('source_venue'))
            if window:
                # The exact Gamma response is not linked to the contract id in the
                # public record; its search window bounds when the quote was read.
                child['quality']['quote_retrieval_window_utc'] = window
                for row in child['outcomes']:
                    row['quote_retrieval_window_utc'] = window
            source_ids.add(child['id'])
            source_cards.append(child)
            parent.setdefault('related_source_scenario_ids', []).append(child['id'])
    risks.extend(source_cards)
    # The engine separately investigates geopolitical proposals. Retain its
    # reviewed discoveries even when no exact market or P&L can be identified.
    known = {str(r['id']): r for r in risks}
    geo = (report.get('geopolitical_search') or {}).get('targets') or []
    for target in geo:
        if not isinstance(target, dict):
            continue
        identifier = str(target.get('event_id') or '')
        if not identifier:
            continue
        reviews, notes, seen = [], [], set()
        for candidate in target.get('candidates') or []:
            if isinstance(candidate, dict):
                contract_id = str(candidate.get('contract_id') or '').strip()
                row = {'decision': str(candidate.get('decision') or 'NO_EXACT_FIT'),
                       'reason': str(candidate.get('difference') or candidate.get('decision') or 'Exact settlement match not established')[:600]}
                if not contract_id or contract_id.lower() in ('candidate', 'unmatched question'):
                    notes.append(row)
                elif contract_id not in seen:
                    reviews.append({'id': contract_id, **row})
                    seen.add(contract_id)
        if identifier in known:
            # The source-backed macro/default event is already present. Attach
            # its independent Kalshi/Polymarket semantic review in place.
            existing_ids = {r['id'] for r in known[identifier]['rejected']}
            known[identifier]['rejected'].extend(r for r in reviews[:24] if r['id'] not in existing_ids)
            known[identifier].setdefault('review_notes', []).extend(notes[:12])
            continue
        definition = _definition_review({'original_question': target.get('question_we_care_about'),
                                         'title': target.get('question_we_care_about'),
                                         'event_date': target.get('event_date')})
        risk = {'id': identifier,
                      'title': str(target.get('question_we_care_about') or 'Geopolitical risk')[:180],
                      'family': str(target.get('family') or 'GEOPOLITICAL'),
                      'date': _date(target.get('event_date')),
                      'original_question': str(target.get('question_we_care_about') or ''),
                      'outcomes': [], 'exposure': [], 'exposure_basis': 'PORTFOLIO_CONTEXT_ONLY',
                      'admitted': [], 'rejected': reviews[:24], 'review_notes': notes[:12],
                      'quality': {'model_status': str(target.get('decision') or 'NO_EXACT_MATCH'),
                                  'model_kind': 'UNAVAILABLE', 'transfer_status': 'NOT_REPORTED',
                                  'failed_symbols': [], 'failed_portfolio_weight': None,
                                  'reason': str(target.get('reason') or ''),
                                  'scope': 'Not estimated', 'future_total_return_prediction_available': False,
                                  'definition_review': definition}}
        risks.append(risk)
        known[identifier] = risk
    for risk in risks:
        risk.setdefault('market_outcomes', [])
        risk['quality'].setdefault('model_admission', event_model_admission(risk))
    analogues = report.get('analogue_stress') or {}
    defaults = {e.get('event_id'): e for e in ((report.get('default_analysis') or {}).get('events') or [])
                if isinstance(e, dict)}
    for risk in risks:
        partial = _partial_priced_component(defaults.get(risk.get('id')), report)
        if partial:
            risk['partial_priced_component'] = partial
    for risk in risks:
        result = analogues.get(risk.get('id'))
        if isinstance(result, dict) and result.get('schema') == 'castle.option1.historical_analogue_stress.v1':
            risk['analogue_stress'] = deepcopy(result)
    bands = report.get('announcement_bands') or {}
    if bands.get('status') == 'COMPUTED':
        from .announcement_band import FAMILY_KEYS, load_evidence
        try:
            evidence = load_evidence().get('families', {})
        except (OSError, ValueError):
            evidence = {}
        for risk in risks:
            key = FAMILY_KEYS.get(str(risk.get('family') or '').upper())
            if key and not str(risk.get('title') or '').startswith('Polymarket:'):
                band = (bands.get('families') or {}).get(key)
                if isinstance(band, dict):
                    risk['announcement_band'] = deepcopy(band)
                if key in evidence:
                    risk['priced_in_evidence'] = deepcopy(evidence[key])
    if not risks:
        risks = [{'id': 'no-priced-events', 'title': 'Events under review', 'family': 'OTHER',
                  'date': None, 'outcomes': [], 'exposure': [], 'admitted': [], 'rejected': [],
                  'quality': {'model_status': 'NO_REVIEWED_EVENTS'}}]
    from .scenario_response_surface import attach_scenario_responses
    attach_scenario_responses(risks, report, lookthrough=_lookthrough(report))
    from .scenario_lab import public_factor_model
    from .live_web_research import public_web_research
    scenario_lab_model = public_factor_model(report, simplified, notional=total)
    evaluated = str(report.get('evaluated_at_utc') or '')
    digest = hashlib.sha256(json.dumps(simplified, sort_keys=True).encode('utf-8') + evaluated.encode()).hexdigest()[:12]
    stages = report.get('stages') or {}
    case = {'id': 'live-' + digest, 'label': 'Your portfolio', 'user_supplied': True,
            'capture': evaluated, 'notional': total, 'positions': simplified,
            'presentation_version': 'R33',
            'instrument_identity': identity,
            'offline_replay': report.get('offline_replay'),
            'scenario_lab_model': scenario_lab_model,
            'lookthrough': _lookthrough(report), 'fund_details': _fund_details(report),
            'risk_watchlist': _risk_watchlist(report, simplified),
            'risk_watchlist_status': _text(report.get('risk_watchlist_status')) or 'NOT_RECORDED',
            'web_research': public_web_research(report.get('web_research')),
            'risks': risks, 'verification_status': str(report.get('verification_status') or 'UNKNOWN'),
            'provider_status': {name: str((stages.get(name) or {}).get('status') or 'NOT_RECORDED')
                                for name in ('Massive', 'Claude', 'Kalshi', 'GeopoliticalSearch', 'AutonomousForecast', 'AutonomousImpact', 'CommonEndpointImpact')},
            'method_scope': 'Scenario change around event announcements; not total future portfolio return'}
    # Only original-holding price changes are public derived values; no
    # provider raw bars, causal event attribution or inferred equity q.
    from .observed_portfolio_mark import review_observed_mark
    mark = report.get("observed_mark_context")
    if (isinstance(mark, dict) and
            mark.get("status") == "OBSERVED_MARK_ONLY_NO_CAUSAL_ATTRIBUTION" and
            not (identity.get("blocked_symbols") or [])):
        case["observed_mark_context"] = deepcopy(mark)
        review_observed_mark(case)
    # Source-admitted outcomes, analyst-declared stresses, and verified
    # autonomous same-endpoint scenario gaps are distinct evidence classes.
    # Rank only after all corresponding event records have been attached.
    from .analyst_scenarios import attach_analyst_scenarios
    from .impact_priority import sort_risks_by_conditional_movement
    from .probability_assumptions import attach_probability_questions
    attach_analyst_scenarios(case)
    attach_probability_questions(case)
    from .structural_event_forecast import attach_autonomous_forecasts
    attach_autonomous_forecasts(case, report.get('autonomous_event_forecasts'))
    # R125: identity-matched prediction-market reviews and targeted evidence
    # travel next to the forecast; quotes stay labelled as market Q.
    from .event_market_search import attach_event_market_reviews
    attach_event_market_reviews(case, report.get('event_market_reviews'))
    from .event_evidence_research import attach_event_evidence
    attach_event_evidence(case, report.get('event_evidence_research'))
    from .autonomous_event_impact import attach_autonomous_impacts
    attach_autonomous_impacts(case, report.get('autonomous_portfolio_impacts'))
    from .autonomous_common_endpoint_impact import attach_common_endpoint_impacts
    attach_common_endpoint_impacts(case, report.get('common_endpoint_impacts'))
    case['risks'] = sort_risks_by_conditional_movement(
        case['risks'], simplified, total,
        observed_mark_context=case.get('observed_mark_context'))
    return case


def review_saved_case(case):
    """Apply current evidence-use rules to a public export, with no model refit.

    This is deliberately weaker than replaying original acquired inputs. Public
    display exports do not retain full response coefficients/state constraints
    or omitted tariff tables. They cannot support a reconstructed sensitivity
    envelope, a new economic fit, or new quoted outcomes.
    """
    if (not isinstance(case, dict) or case.get('user_supplied') is not True or
            not isinstance(case.get('risks'), list) or not isinstance(case.get('positions'), list)):
        raise ValueError('A completed public portfolio export is required')
    if _number(case.get('notional')) is None or case['notional'] <= 0:
        raise ValueError('A positive finite portfolio notional is required')
    result = deepcopy(case)
    original_version = str(case.get('presentation_version') or 'NOT_REPORTED')
    corrections = []
    identity = case.get('instrument_identity') or {}
    blocked = set(identity.get('blocked_symbols') or [])
    types = {str(row.get('symbol')): row.get('type') for row in case['positions'] if isinstance(row, dict)}
    for row in identity.get('positions') or []:
        if not isinstance(row, dict) or row.get('symbol') not in types:
            continue
        symbol = row['symbol']
        resolved = row.get('resolved_asset_type')
        provider = row.get('provider_instrument_type')
        verified = row.get('local_response_verified') is True
        incompatible_resolved = resolved in ('stock', 'etf') and resolved != types[symbol]
        incompatible_provider = (verified and types[symbol] in ('stock', 'etf') and provider and
            provider not in ({'CS', 'ADRC'} if types[symbol] == 'stock' else {'ETF', 'ETV'}))
        if incompatible_resolved or incompatible_provider:
            blocked.add(symbol)
    blocked = sorted(blocked)
    if set(blocked) != set(identity.get('blocked_symbols') or []):
        result['instrument_identity'] = {**deepcopy(identity),
            'status': 'BLOCKED_SAVED_IDENTITY_CONTRADICTION',
            'blocked_symbols': blocked,
            'review_note': 'Public identity records contradict the saved modeling type. '
                          'No instrument correction or economic refit is inferred from this export.'}
    for risk in result['risks']:
        if not isinstance(risk, dict) or not isinstance(risk.get('quality'), dict):
            raise ValueError('Every public event must have its recorded quality metadata')
        from .business_interruption_assessment import component_view
        if 'business_interruption_component' in risk:
            component = component_view(risk['business_interruption_component'],
                event_id=risk.get('id'), portfolio_notional=result['notional'],
                portfolio_positions=result['positions'])
            if component is None or 'UNP' in blocked:
                risk.pop('business_interruption_component', None)
                corrections.append({'event_id': risk.get('id'),
                                    'correction': 'INVALID_RECORDED_CASHFLOW_COMPONENT_REMOVED'})
            else:
                risk['business_interruption_component'] = component
        from .mexico_tariff_transmission import mexico_component_view
        if 'mexico_tariff_component' in risk:
            mexico = mexico_component_view(risk['mexico_tariff_component'],
                event_id=risk.get('id'), portfolio_notional=result['notional'],
                portfolio_positions=result['positions'])
            if mexico is None or 'UNP' in blocked:
                risk.pop('mexico_tariff_component', None)
                corrections.append({'event_id': risk.get('id'),
                                    'correction': 'INVALID_RECORDED_MEXICO_TARIFF_COMPONENT_REMOVED'})
            else:
                risk['mexico_tariff_component'] = mexico
        quality = risk['quality']
        current_definition = _definition_review(risk)
        recorded_definition = quality.get('definition_review') or {}
        if (current_definition['status'] != 'DEFINED_FOR_REVIEW' and
                recorded_definition.get('status') in (None, '', 'DEFINED_FOR_REVIEW')):
            quality['definition_review'] = current_definition
            corrections.append({'event_id': risk.get('id'),
                                'correction': 'EXPLICIT_DEFINITION_DEFECT_RECHECKED'})
        elif not recorded_definition:
            quality['definition_review'] = current_definition
        sensitivity = quality.get('model_sensitivity') or {}
        if sensitivity.get('no_response_alternative_included') is True:
            for outcome in risk.get('outcomes') or []:
                bound = outcome.get('pnl_model_sensitivity_range_usd')
                if ('no_response_alternative_included' not in outcome and
                        isinstance(bound, list) and len(bound) == 2 and
                        all(_number(value) is not None for value in bound) and
                        bound[0] <= 0 <= bound[1]):
                    outcome['no_response_alternative_included'] = True
        if (risk.get('source_defined_scenario') is True and
                quality.get('model_kind') in ('MEASURED_ANCHOR', 'TARIFF_ANCHOR') and
                'country other than China' in str(quality.get('reason') or '')):
            quality['original_parent_scope_reason'] = quality['reason']
            quality['reason'] = ('Separate source-defined China tariff-rate scenario under its own '
                                 'contract question and date. Portfolio response uses the recorded '
                                 'historical-anchor assumptions; it does not price the original target.')
            corrections.append({'event_id': risk.get('id'),
                                'correction': 'SOURCE_SCENARIO_NOT_PARENT_REJECTION'})
        if blocked:
            quality['instrument_identity_gate'] = {
                **(quality.get('instrument_identity_gate') or {}),
                'status': 'BLOCKED_INSTRUMENT_IDENTITY', 'blocked_symbols': list(blocked)}
        if risk.get('family') in ('TRADE_TARIFFS', 'US_TARIFFS'):
            from .tariff_scenarios import target_scope_review
            scope = target_scope_review({'question_we_care_about': risk.get('original_question') or risk.get('title')})
            china_reviewed = any(re.search(r'PRC|china|chinese',
                str(row.get('id') or '') + ' ' + str(row.get('reason') or ''), re.I)
                for row in risk.get('rejected') or [] if isinstance(row, dict))
            if (china_reviewed and scope['status'] == 'DIFFERENT_OR_MIXED_COUNTRY_TARIFF_SCOPE'
                    and 'product' in str(quality.get('reason') or '').lower()):
                quality['original_unavailable_reason'] = quality.get('reason')
                quality['reason'] = scope['reason']
                quality['target_scope_review'] = scope
                corrections.append({'event_id': risk.get('id'),
                                    'correction': 'COUNTRY_MISMATCH_NOT_NEGATED_PRODUCT_RESTRICTION'})
            if (quality.get('model_kind') == 'UNAVAILABLE' and not risk.get('outcomes') and
                    quality.get('model_status') == 'SOURCE_BRACKET_DISTRIBUTION_WITH_MEASURED_ANCHOR_MODEL'):
                quality['original_model_status'] = quality['model_status']
                quality['model_status'] = 'ORIGINAL_TARGET_NOT_PRICED_RELATED_TABLE_NOT_IN_PUBLIC_EXPORT'
                quality['reason'] = (scope['reason'] + ' The related source-rate table was not retained '
                                     'in this public export, so no replacement values are reconstructed here.')
                quality['source_scenario_is_exact_target_match'] = False
                corrections.append({'event_id': risk.get('id'),
                                    'correction': 'STALE_TARIFF_STATUS_WITH_MISSING_RELATED_TABLE'})
        quality['model_admission'] = event_model_admission(risk)
        specific_scope_reason = (
            quality.get('original_model_status') == 'SOURCE_BRACKET_DISTRIBUTION_WITH_MEASURED_ANCHOR_MODEL'
            or (quality.get('original_unavailable_reason') and
                (quality.get('target_scope_review') or {}).get('status') ==
                'DIFFERENT_OR_MIXED_COUNTRY_TARIFF_SCOPE'))
        if (specific_scope_reason and not risk.get('outcomes') and quality.get('reason')
                and quality['model_admission'].get('evidence_scope') in {
                    'RELATED_SOURCE_SCENARIO_NOT_ORIGINAL_TARGET', 'NO_ADMITTED_COMPLETE_RESPONSE'}):
            # Export/UI prefer admission.reason. Preserve the verified specific
            # source-scope correction, including absence of a retained table,
            # without overriding any identity, definition or incomplete-model gate.
            quality['model_admission']['reason'] = quality['reason']
        definition = quality.get('definition_review') or {}
        if definition.get('status') not in (None, '', 'DEFINED_FOR_REVIEW'):
            quality['model_admission'].update(state='UNAVAILABLE', primary_impact_allowed=False,
                                             economic_ranking_allowed=False,
                                             reason=str(definition.get('reason') or 'Unresolved event definition'))
        definition_invalid = definition.get('status') not in (None, '', 'DEFINED_FOR_REVIEW')
        if quality['model_admission']['state'] == 'RELATED_SOURCE_ONLY' or definition_invalid:
            # Retain stale public evidence explicitly outside this target's
            # probability/impact lists. It is not a quoted equivalent event.
            if risk.get('outcomes') or risk.get('admitted'):
                quality['unmatched_source_diagnostic'] = {
                    'outcomes': deepcopy(risk.get('outcomes') or []),
                    'admitted_in_original_export': deepcopy(risk.get('admitted') or [])}
                risk['outcomes'] = []
                risk['admitted'] = []
            risk['market_outcomes'] = []
        if 'market_outcomes' not in risk:
            risk['market_outcomes'] = [
                {**{key: deepcopy(row[key]) for key in
                    ('id', 'label', 'q_reference', 'q_interval', 'source_contract_ids',
                     'contract_id', 'venue', 'question') if key in row},
                 'pnl': None, 'pnl_interval': None,
                 'scope': 'Original saved market-pricing evidence; no new source request'}
                for row in risk.get('outcomes') or [] if isinstance(row, dict)
                and _probability(row.get('q_reference')) is not None]
        quality.setdefault('model_sensitivity', {
            'status': 'UNAVAILABLE',
            'reason': 'Public export lacks the full response-parameter and state/Q constraint inputs. '
                      'Original numerical model ranges are retained; no new envelope is invented.'})
    original_order = [event.get('id') for event in result['risks']]
    # Retain the relative order within each evidence-use group. No new loss
    # estimate, P probability or selective model score is used in this order.
    result['risks'].sort(key=lambda event: (
        0 if event['quality']['model_admission']['economic_ranking_allowed'] else
        1 if event.get('market_outcomes') else 2))
    previous_review = case.get('presentation_review') or {}
    result['presentation_version'] = 'R33'
    result['presentation_review'] = {
        'status': 'PUBLIC_EXPORT_REVIEW_ONLY_R33',
        'original_presentation_version': previous_review.get('original_presentation_version', original_version),
        'capture_preserved': True, 'new_source_requests': 0, 'numerical_refit': False,
        'identity_blocked_symbols': blocked,
        'original_event_order': previous_review.get('original_event_order', original_order),
        'event_order_policy': 'PRIMARY_CONDITIONAL_SCENARIOS_THEN_MARKET_ONLY_THEN_UNAVAILABLE',
        'corrections': previous_review.get('corrections', []) + corrections,
        'scope': 'Model-use and presentation review only. No new forecast performance '
                 'or unavailable underlying model can be established from this export.'}
    # A modeled YES-from-today scenario is admitted only when the already
    # reviewed synchronized provider mark context matches the original NAV.
    from .observed_portfolio_mark import review_observed_mark
    review_observed_mark(result)
    from .autonomous_event_impact import recheck_saved_autonomous_impacts
    recheck_saved_autonomous_impacts(result)
    from .autonomous_common_endpoint_impact import recheck_saved_common_endpoint_impacts
    recheck_saved_common_endpoint_impacts(result)
    # An original R123 navigation amount must not survive after a tampered or
    # mismatched saved v2 scenario has been discarded during reinspection.
    if any(isinstance(risk, dict) and (
            (risk.get('impact_priority') or {}).get('evidence_tier') ==
            'B_AUTONOMOUS_COMMON_ENDPOINT_SCENARIO' or
            (risk.get('common_endpoint_impact') or {}).get('status') ==
            'MODEL_COMMON_ENDPOINT_STRESS_UNCALIBRATED') for risk in result['risks']):
        from .impact_priority import sort_risks_by_conditional_movement
        result['risks'] = sort_risks_by_conditional_movement(
            result['risks'], result['positions'], result['notional'],
            observed_mark_context=result.get('observed_mark_context'))
    return result
