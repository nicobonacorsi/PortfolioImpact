"""One offline portfolio brief joining discovery, exact contracts and impact.

The same source verifiers used by the existing application remain authoritative.
This module creates no provider request and never turns a documentary candidate
into a certified contract match or a missing economic model into a zero loss.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .adapters import normalize_quote
from .agentic_bridge import inspect_exact_contract
from .contract_questions import source_defined_yes_question
from .macro_economic_bridge import (
    DEFAULT_CONTRACT, FomcEconomicAssumptions, build_fomc_economic_case,
)
from .portfolio import parse_portfolio
from .portfolio_event_discovery import discover_archived_portfolio_events


def portfolio_input_fingerprint(portfolio_rows: list[dict], notional: float,
                                holdings: Mapping | None = None) -> str:
    """Bind a rendered result to the complete original portfolio and holdings."""
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    if holdings is not None and not isinstance(holdings, Mapping):
        raise ValueError('Holdings must be a snapshot mapping or None')
    body = {'portfolio': portfolio.to_dict(), 'holdings': dict(holdings or {})}
    def supported_object(value):
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        raise TypeError('Portfolio evidence must contain JSON values or supported snapshot dataclasses/dates')
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False,
                         default=supported_object)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def build_option1_brief(
    *, portfolio_rows: list[dict], notional: float = 100_000.0,
    holdings: Mapping | None = None, root: Path | None = None,
    evaluated_at: str | datetime | None = None, horizon_days: int = 90,
    selected_fomc_contract: str = DEFAULT_CONTRACT,
    economic_assumptions: FomcEconomicAssumptions | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible report for the current portfolio, using archives.

    Current time controls candidate eligibility and holdings admission. Every
    quote carries its original receipt time, while the macro model explicitly
    freezes those archived expectations for a hypothetical announcement case.
    This is neither a historical portfolio backtest nor a current price feed.
    """
    now = datetime.now(timezone.utc) if evaluated_at is None else evaluated_at
    if isinstance(now, str):
        now = datetime.fromisoformat(now.replace('Z', '+00:00'))
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError('A timezone-aware evaluation time is required')
    now = now.astimezone(timezone.utc)
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    discovery = discover_archived_portfolio_events(
        portfolio_rows=portfolio_rows, notional=notional, holdings=holdings,
        evaluated_at=now, horizon_days=horizon_days)
    originals = discovery['originals']
    events = []
    for candidate in discovery['events']:
        contracts = []
        for contract_id in candidate['matched_contract_ids']:
            market = originals[contract_id]
            inspected = inspect_exact_contract(market, portfolio_rows=portfolio_rows,
                notional=notional, observed_at=now)
            question = source_defined_yes_question(market)
            historical = normalize_quote(market, now=market.retrieved_at)
            closes = datetime.fromisoformat(market.close_time.replace('Z', '+00:00'))
            receipt = datetime.fromisoformat(market.retrieved_at.replace('Z', '+00:00'))
            historical_interval = ([historical.lower, historical.upper]
                if historical.usable and receipt < closes else None)
            contracts.append({
                'contract_id': contract_id,
                'yes_question': question['yes_question'],
                'full_rules_sha256': question['verified_rules_sha256'],
                'numeric_rule_status': question['numeric_rule_status'],
                'typed_predicate': question['typed_predicate'],
                'additional_model_assumptions': question['additional_model_assumptions'],
                'exception_flags': question['exception_flags'],
                'macro_hypothesis_equivalence': question['macro_hypothesis_equivalence'],
                'admission': 'SOURCE_BACKED_EXACT_CONTRACT_IDENTITY',
                'source_url': market.source_url,
                'source_receipt_utc': inspected['source']['retrieved_at_utc'],
                'source_response_sha256': inspected['source']['source_response_sha256'],
                'historical_market_implied_Q_interval': historical_interval,
                # This deliverable is explicitly an archive route even when a
                # caller chooses an evaluation time coinciding with its receipt.
                'current_live_probability': None,
                'physical_probability_P': None,
                'scope': 'EXACT_CONTRACT_YES_ONLY_NOT_A_BROAD_ECONOMIC_EVENT',
            })
        event = {**candidate, 'contracts': contracts,
                 'rejected_contract_examples': discovery['macro_discovery'].get(
                     'quarantined_provider_contracts', {}).get(candidate['event_id'], [])}
        if candidate['family'] == 'FED_POLICY':
            event['economic_impact'] = build_fomc_economic_case(
                portfolio, root=root, selected_contract_id=selected_fomc_contract,
                analysis_as_of=now, assumptions=economic_assumptions,
                markets=[originals[x] for x in candidate['matched_contract_ids']])
        else:
            event['economic_impact'] = {
                'status': 'UNAVAILABLE',
                'reason': ('No reviewed announcement-to-asset model is supplied for this event. '
                           'Its probability, where available, does not identify a portfolio return.'),
                'branches': [], 'total_portfolio_pnl_usd': None,
                'missing_impact_is_zero': False,
            }
        events.append(event)
    settings = {
        'evaluated_at_utc': now.isoformat(), 'horizon_days': horizon_days,
        'selected_fomc_contract': selected_fomc_contract,
        'economic_assumptions': asdict(economic_assumptions or FomcEconomicAssumptions()),
    }
    input_fingerprint = portfolio_input_fingerprint(portfolio_rows, notional, holdings)
    investigation_fingerprint = hashlib.sha256(json.dumps(
        {'input_fingerprint': input_fingerprint, 'settings': settings},
        sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
    return {
        'schema': 'castle.option1.portfolio_brief.v1',
        'status': discovery['status'],
        'evaluated_at_utc': now.isoformat(),
        'execution_mode': 'OFFLINE_RECOMPUTATION_FROM_INCLUDED_ARCHIVED_EVIDENCE',
        'input_fingerprint': input_fingerprint,
        'investigation_fingerprint': investigation_fingerprint,
        'analysis_settings': settings,
        'portfolio': portfolio.to_dict(), 'coverage': discovery['coverage'],
        'exposure_ledger': discovery['exposure_ledger'], 'events': events,
        'excluded_events': discovery['excluded_events'],
        'ranking_rule': discovery['ranking_rule'],
        'new_provider_requests': 0, 'new_model_calls': 0,
        'historical_live_probe_reproduced_by_this_report': False,
        'limitations': discovery['limitations'] + [
            'Only the FOMC candidate has a sourced, assumption-dependent economic component model.',
            'No weighted physical expected loss is computed from prediction-market pricing weights.',
            'Economic source estimates, proxy transfers and deliberately chosen stress ranges retain separate labels.',
        ],
    }


def render_option1_brief_markdown(report: Mapping[str, Any]) -> str:
    """Readable evidence export of the same result shown in the application."""
    lines = ['# Portfolio event brief', '',
        f"Evaluation: {report['evaluated_at_utc']}",
        f"Mode: {report['execution_mode']}",
        f"Status: {report['status']}",
        f"Original portfolio NAV: ${report['portfolio']['notional']:,.2f}",
        f"Input fingerprint: `{report['input_fingerprint']}`",
        f"Investigation fingerprint: `{report['investigation_fingerprint']}`", '',
        '## Analysis settings', '', '```json',
        json.dumps(report['analysis_settings'], sort_keys=True, indent=2, allow_nan=False),
        '```', '',
        'Market references below are historical exact-contract pricing intervals. '
        'They are not current quotes or calibrated physical probabilities.', '',
        '## Coverage', '',
        f"Supported instrument channels: {report['coverage']['supported_instrument_weight']:.1%}",
        f"Unsupported instrument channels: {report['coverage']['unsupported_instrument_weight']:.1%}",
        f"Explicit cash: {report['coverage']['explicit_cash_weight']:.1%}",
        f"Unresolved ETF constituent identity: {report['coverage']['unresolved_etf_identity_weight']:.1%}", '',
        report['ranking_rule'], '']
    if not report['events']:
        lines += ['Nothing supported in this bounded event catalog for the current portfolio and window. '
                  'This does not mean zero risk.', '']
    for event in report['events']:
        lines += [f"## {event['inspection_priority']}. {event['title']}", '',
                  f"Date or deadline: {event['event_date']}",
                  f"Linked NAV: ${event['relevance_notional_usd']:,.2f}; exposure relevance, not predicted loss.",
                  f"Evidence class: {event['source_class']}", '']
        for component in event['relevance_components']:
            lines.append(f"- {component['original_position']} / {component['instrument']}: "
                         f"${component['notional_usd']:,.2f}. {component['reason']}")
        lines += ['', '### Exact contract probability references', '']
        if not event['contracts']:
            lines += ['Nothing certified from the included provider bytes. Probability unavailable.', '']
        for contract in event['contracts']:
            interval = contract['historical_market_implied_Q_interval']
            display = f"[{interval[0]:.1%}, {interval[1]:.1%}]" if interval else 'Unavailable'
            lines += [f"- **{contract['contract_id']}** — {contract['yes_question']}",
                      f"  Historical Q: {display}; receipt {contract['source_receipt_utc']}.",
                      f"  Source: {contract['source_url']}",
                      f"  Source response SHA-256: `{contract['source_response_sha256']}`.",
                      f"  Rules SHA-256: `{contract['full_rules_sha256']}`."]
            if contract['additional_model_assumptions']:
                lines.append('  Numeric interpretation is conditional on: ' + '; '.join(contract['additional_model_assumptions']))
            if contract['exception_flags']:
                lines.append('  Settlement exceptions: ' + ', '.join(contract['exception_flags']))
        for rejected in event['rejected_contract_examples']:
            lines.append(f"Rejected match: {rejected['ticker']} — {rejected['reason']}")
        impact = event['economic_impact']
        lines += ['', '### Economic impact', '', impact['reason'], '']
        if impact['branches']:
            ref = impact['reference_expectation']
            lines += [impact['horizon'], '',
                f"Selected exact YES: {impact['selected_contract_id']}",
                f"Frozen illustrative reference policy change: {ref['mean_policy_change_bps']:.6f} bp.", '',
                '| State | Surprise (bp) | Modeled component (USD) | Combined sensitivity enclosure (USD) |',
                '|---|---:|---:|---|']
            for branch in impact['branches']:
                amount = branch['total_pnl_component_usd']
                display = f"{amount:,.2f}" if amount is not None else 'Unknown total; subtotal ' + f"{branch['known_pnl_component_usd']:,.2f}"
                low, high = branch['known_combined_model_sensitivity_pnl_range_usd']
                lines.append(f"| {branch['state']} | {branch['policy_surprise_bps']:.6f} | {display} | [{low:,.2f}, {high:,.2f}] |")
            lines += ['', 'Sensitivity enclosures are chosen model stresses, not confidence or prediction intervals.', '',
                      '| State | Original position | Modeled component (USD) | Basis |',
                      '|---|---|---:|---|']
            for branch in impact['branches']:
                for item in branch['contributions']:
                    amount = item['pnl_component_usd']
                    lines.append(f"| {branch['state']} | {item['symbol']} | {amount:,.2f} | {item['model_basis']} |" if amount is not None
                                 else f"| {branch['state']} | {item['symbol']} | Unknown | {item['model_basis']} |")
            lines += ['', '#### NO mixture', '',
                      'NO includes every other source outcome and unlisted OTHER under the same '
                      'illustrative reference Q. These conditional weights are model inputs, '
                      'not estimated physical probabilities. Opaque outcome magnitudes retain '
                      'the explicit assumptions listed below.', '',
                      '| Source outcome | Conditional reference Q given NO | Policy change (bp) |',
                      '|---|---:|---:|']
            for state in ref['NO_mixture']:
                lines.append(f"| {state['source_outcome_id']} | "
                             f"{state['conditional_reference_Q_weight']:.6%} | "
                             f"{state['policy_change_bps']:.6f} |")
            lines += ['', '#### Assumptions', ''] + ['- ' + x for x in impact['assumptions']]
            lines += ['', '#### Source references', '']
            for source in impact['sources']:
                lines.append('- ' + str(source.get('title', source.get('symbol', 'Issuer duration'))) + ': '
                             + str(source.get('url', source.get('source_url', 'See JSON source ledger'))))
    if report['excluded_events']:
        lines += ['', '## Excluded candidates', '']
        lines += [f"- {event['title']}: {event['reason']}" for event in report['excluded_events']]
    lines += ['', '## Remaining limitations', ''] + ['- ' + x for x in report['limitations']]
    return '\n'.join(lines) + '\n'
