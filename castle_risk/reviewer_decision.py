"""Compact, source-bounded assessment ledger for the original Castle Option 1.

The presentation deliberately does NOT combine prices from different capture times
into a synthetic contemporaneous portfolio quote or transfer a related contract's
price to a broad geopolitical hypothesis. No acquisition, fitting or prediction.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


class ReviewerEvidenceError(ValueError):
    """A required recorded source is changed, incompatible or unsupported."""


def _portfolio_key(portfolio: Mapping[str, Any]) -> tuple[Any, ...]:
    """Compare the actual portfolio, including original types and exact weights."""
    try:
        notional = float(portfolio['notional'])
        positions = portfolio['positions']
        result = tuple(sorted((p['symbol'], p['asset_type'], float(p['weight'])) for p in positions))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewerEvidenceError('Recorded portfolio is not valid.') from exc
    if notional <= 0 or not math.isfinite(notional) or any(not math.isfinite(p[2]) for p in result):
        raise ReviewerEvidenceError('Recorded portfolio has nonfinite or invalid weights.')
    return (notional, result)


def read_separately_dated_matching_record(root: Path, label: str, portfolio: Mapping[str, Any]) -> dict:
    """Verify the original cross-venue captured report before showing its refusals."""
    folder = Path(root) / 'data' / 'verified_assessments'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'castle.round8.recorded_reanalysis_manifest.v1':
        raise ReviewerEvidenceError('Unknown original-source report manifest.')
    entries = [entry for entry in manifest['reports'] if entry['label'] == label]
    if len(entries) != 1:
        raise ReviewerEvidenceError('Original-source portfolio identifier is missing or ambiguous.')
    entry = entries[0]
    filename = entry['file']
    if Path(filename).name != filename or not filename.endswith('.json'):
        raise ReviewerEvidenceError('Invalid original-source report path.')
    raw = (folder / filename).read_bytes()
    if sha256(raw).hexdigest() != entry['sha256']:
        raise ReviewerEvidenceError('Original-source portfolio report hash mismatch.')
    result = json.loads(raw)
    if _portfolio_key(result['portfolio']) != _portfolio_key(portfolio):
        raise ReviewerEvidenceError('Separate evidence belongs to a different portfolio.')
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _dollars(value: Any) -> str:
    return f'${value:+,.2f}' if _finite(value) else 'UNKNOWN'


def _macro_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    result = []
    for event in report['default_analysis']['events']:
        if event['family'] not in {'FED_POLICY', 'US_INFLATION'}:
            continue
        distribution = event.get('distribution') or {}
        economics = event.get('announcement_day_economics') or event.get('economics') or {}
        cases = economics.get('outcome_cases') or []
        usable = [case for case in cases if _finite(case.get('q_reference'))
                  and _finite(case.get('component_pnl_usd')) and isinstance(case.get('q_interval'), list)
                  and len(case['q_interval']) == 2 and all(_finite(x) for x in case['q_interval'])]
        if usable and distribution.get('status') == 'CONDITIONAL_SOURCE_DISTRIBUTION':
            # The dominant *recorded pricing-weight* outcome is a transparent,
            # predeclared illustration, never the outcome selected to maximize P&L.
            selected = sorted(usable, key=lambda c: (-c['q_reference'], c['label']))[0]
            low, high = selected['q_interval']
            if not 0 <= low <= selected['q_reference'] + 1e-9 <= high + 1e-9 <= 1 + 1e-9:
                raise ReviewerEvidenceError('Contradictory source weight for representative outcome.')
            priced = selected.get('priced_in') or {}
            result.append({
                'Event': event['title'],
                'Matching / probability evidence': (f"Source-derived Q, {selected['label']}: "
                                                    f'{low:.1%}–{high:.1%}'),
                'Conditional portfolio impact': (_dollars(selected['component_pnl_usd']) +
                                                 ' · modeled announcement component'),
                'Already priced in': (_dollars(priced.get('YES')) + ' YES residual · assumed π=Q'
                                      if _finite(priced.get('YES')) else 'UNKNOWN'),
                'Evidence scope': 'Recorded macro source; conditional model, not total return',
                '_kind': 'MODELLED_MACRO_COMPONENT',
            })
        else:
            result.append({
                'Event': event['title'],
                'Matching / probability evidence': 'ABSTAIN · ' + distribution.get('status', 'no admissible distribution'),
                'Conditional portfolio impact': 'UNKNOWN · no complete source-consistent model',
                'Already priced in': 'UNKNOWN',
                'Evidence scope': 'Recorded macro source; no repaired or renormalized quotes',
                '_kind': 'ABSTAIN_MACRO',
            })
    return result


def _geopolitical_rows(original: Mapping[str, Any]) -> list[dict[str, str]]:
    result = []
    for event in original['default_analysis']['events']:
        if event['family'] not in {'TAIWAN_SUPPLY', 'EXPORT_CONTROLS'}:
            continue
        search = event.get('market_search') or {}
        # A recorded candidate may have its own YES quote. That quote must never
        # become a probability for the user's wider question without a proof.
        candidates = search.get('candidates') or []
        if any(c.get('broad_macro_event_matched') for c in candidates):
            # Do not silently flatten a newly admitted market into this older
            # rejection-only reviewer component.
            raise ReviewerEvidenceError('Original geopolitical evidence unexpectedly admits a broad match.')
        result.append({
            'Event': event['title'],
            'Matching / probability evidence': 'ABSTAIN · related markets are not equivalent',
            'Conditional portfolio impact': 'UNKNOWN total · issuer channel sensitivity only',
            'Already priced in': 'UNKNOWN',
            'Evidence scope': 'Separate original two-venue capture, not a current search',
            '_kind': 'ABSTAIN_GEO',
        })
    return result


def _taiwan_row(report: Mapping[str, Any], stress: Mapping[str, Any]) -> dict[str, str]:
    required = (100000.0, (('NVDA', 'stock', .35), ('QQQ', 'etf', .45), ('TLT', 'etf', .20)))
    if _portfolio_key(report['portfolio']) != required:
        raise ReviewerEvidenceError('This fixed Taiwan stress does not apply to the selected portfolio.')
    if (stress.get('status') != 'DECLARED_HISTORICALLY_ANCHORED_CONDITIONAL_SCENARIO_NOT_CAUSAL_ESTIMATE'
            or stress.get('public_contract_source', {}).get('contract_id') != '567621'
            or stress.get('portfolio_weights') != {'NVDA': .35, 'QQQ': .45, 'TLT': .2}
            or float(stress.get('notional_usd', 0)) != 100000.0):
        raise ReviewerEvidenceError('Taiwan historical source, terms or portfolio do not match.')
    q = stress.get('market_price', {}).get('reference')
    delta = stress.get('assumed_yes_minus_no_incremental_pnl_usd', {}).get('reference_median')
    yes = stress.get('priced_in_under_assumed_transfer', {}).get('YES_residual_usd')
    if not (_finite(q) and 0 < q < 1 and _finite(delta) and _finite(yes)):
        raise ReviewerEvidenceError('No complete reference scenario or quote.')
    if not math.isclose(yes, (1-q)*delta, abs_tol=1e-4):
        raise ReviewerEvidenceError('Recorded already-priced-in accounting does not reconcile.')
    return {
        'Event': 'Exact Polymarket invasion settlement (NOT general Taiwan disruption)',
        'Matching / probability evidence': f'Exact contract YES · archived price proxy {q:.2%}',
        'Conditional portfolio impact': _dollars(delta) + ' · assumed risk-off one-day stress',
        'Already priced in': _dollars(yes) + ' YES residual · assumed π=Q',
        'Evidence scope': 'Separately timed quote + hypothetical stress; not invasion-effect estimate',
        '_kind': 'ASSUMED_EXACT_CONTRACT_STRESS',
    }


def assemble_reviewer_answer_sheet(report: Mapping[str, Any], *,
                                   original: Mapping[str, Any] | None = None,
                                   taiwan: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """One inspectable answer per event with UNKNOWN where identification fails.

    Combining this summary DOES NOT produce a joint time-synchronized market
    quote, estimate the physical measure or rank event probabilities by risk.
    """
    rows = _macro_rows(report)
    if original is not None:
        if _portfolio_key(original['portfolio']) != _portfolio_key(report['portfolio']):
            raise ReviewerEvidenceError('Cross-capture portfolio mismatch.')
        rows.extend(_geopolitical_rows(original))
    if taiwan is not None:
        rows.append(_taiwan_row(report, taiwan))
    if not rows:
        raise ReviewerEvidenceError('No verified event rows can be presented.')
    return rows
