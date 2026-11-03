"""Portfolio-to-contract investigation from receipt-consistent archived bytes.

No live acquisition, model call, probability calibration or return forecast is
claimed. The original catalog page remains the hashed source; child-market
extraction is a derived view and never forged into a new HTTP GET receipt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .event_discovery import read_p1_capture, discover
from .gamma_indicative_quotes import _price
from .portfolio import parse_portfolio, recover_exposures
from .operator_event_scenarios import evaluate_explicit_yes_no_scenarios
from .gamma_rule_fact_audit import audit_gamma_rule_text

P1_ARCHIVE_RELATIVE_PATH = 'runtime/recovered_p1/P1_CAPTURE_PRIVATE_EVIDENCE.zip'
P1_ARCHIVE_SHA256 = '6e30090f72cae85f6e7266537b6431db661e8356570b53d415ffe6863cbf026b'


def load_recovered_p1_archive(root: Path | str) -> bytes:
    raw = (Path(root) / P1_ARCHIVE_RELATIVE_PATH).read_bytes()
    if sha256(raw).hexdigest() != P1_ARCHIVE_SHA256:
        raise ValueError('Recovered P1 archive does not match the verified original capture')
    return raw


def _instant(value: str | datetime) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise ValueError('An ISO timestamp or datetime is required')
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Timezone-aware evaluation and source times required')
    return dt.astimezone(timezone.utc)


def investigate_archived_nonmacro(
    archive_bytes: bytes, *, portfolio_rows: list[dict], notional: float,
    holdings: Mapping[str, Any] | None = None, event_id: str | None = None,
    market_id: str | None = None, evaluated_at: datetime | str | None = None,
    assumed_yes_returns_pct: Mapping[str, float] | None = None,
    assumed_no_returns_pct: Mapping[str, float] | None = None,
) -> dict:
    """Discover from verified catalogs, then inspect an exact Gamma YES contract.

    Automatic selection is the top Gamma nonmacro candidate by the existing
    declared exposure heuristic. It is NOT selection by estimated expected loss.
    Historical quotes stay historical even when evaluated at capture time.
    """
    now = _instant(datetime.now(timezone.utc) if evaluated_at is None else evaluated_at)
    source = read_p1_capture(archive_bytes)
    acquisition_completed_at = max(_instant(r['completed_at_utc']) for r, _ in source['pages'])
    ranking_at = max((_instant(r['completed_at_utc']) for r, _ in source['pages']
                      if r['provider'] != 'official'), default=None)
    if ranking_at is None:
        raise ValueError('No date-stamped market response for archived ranking')
    if now < acquisition_completed_at:
        raise ValueError('Evaluation precedes the completed source acquisition')
    pinned_capture = source['archive_sha256'] == P1_ARCHIVE_SHA256
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    capture_exposure = recover_exposures(portfolio, holdings or {}, now=ranking_at).to_dict()
    evaluated_exposure = recover_exposures(portfolio, holdings or {}, now=now).to_dict()
    discovered = discover(source['pages'], as_of=now.date().isoformat(),
        horizon_days=90, portfolio_rows=portfolio_rows, holdings=holdings or {},
        archive_sha256=source['archive_sha256'], max_candidates=12)
    excluded = {'FED_RATES', 'TREASURY_YIELD', 'US_CPI', 'CPI', 'FOMC', 'US_INFLATION'}
    candidates = [c for c in discovered['all_relevant_candidates']
                  if c['venue'] == 'polymarket' and c['family'] not in excluded]
    if (event_id is None) != (market_id is None):
        raise ValueError('Provide both exact event ID and market ID, or neither')
    if event_id is None:
        selected = candidates[0] if candidates else None
    else:
        found = [c for c in candidates if c['event_id'] == event_id and c['market_id'] == market_id]
        selected = found[0] if found else None
    if selected is None:
        raise ValueError('No eligible exact nonmacro contract for this portfolio in the captured sample')
    matches = []
    for receipt, body in source['pages']:
        if receipt['provider'] != 'polymarket' or receipt['sha256'] != selected['raw_page_sha256']:
            continue
        for event in body['events']:
            if not isinstance(event, dict):
                continue
            if str(event.get('id')) != selected['event_id']:
                continue
            for market in event.get('markets', []):
                if not isinstance(market, dict):
                    continue
                if str(market.get('id')) == selected['market_id']:
                    matches.append((receipt, event, market))
    if len(matches) != 1:
        raise ValueError('Exact selected source identity is missing or ambiguous')
    receipt, event, market = matches[0]
    question = market.get('question')
    slug = event.get('slug')
    if (any(isinstance(value, bool) or not isinstance(value, (str, int))
            or not str(value).strip() or len(str(value)) > 128
            for value in (event.get('id'), market.get('id')))
            or not isinstance(question, str) or not question.strip() or len(question) > 1000
            or not isinstance(slug, str) or not slug.strip()
            or not selected['market_url']):
        raise ValueError('Exact source market question and event URL identity are required')
    rules = market.get('description')
    if (not isinstance(rules, str) or not rules.strip() or len(rules) > 30_000
            or sha256(rules.encode()).hexdigest() != selected['full_rules_digest_sha256']):
        raise ValueError('Complete market-level rules cannot be reconciled with discovery')
    try:
        outcomes = json.loads(market.get('outcomes', ''))
    except (TypeError, ValueError) as exc:
        raise ValueError('Source binary outcome identity missing') from exc
    if outcomes != ['Yes', 'No']:
        raise ValueError('Exact Yes/No outcome order required')
    historical = {'status': 'HISTORICAL_PRICE_UNAVAILABLE', 'bid': None, 'ask': None}
    try:
        bid, ask = _price(market.get('bestBid')), _price(market.get('bestAsk'))
        if bid > ask or market.get('active') is not True or market.get('closed') is not False:
            raise ValueError('Invalid historical market state or prices')
        historical = {'status': 'HISTORICAL_INDICATIVE_GAMMA_YES_PRICE_REFERENCE',
                      'bid': bid, 'ask': ask}
    except ValueError:
        pass
    historical.update(observed_at_utc=receipt['completed_at_utc'],
        current_quote_admitted=False, physical_probability_P=None,
        source_origin_matches_recovered_archive=pinned_capture,
        interpretation='Gamma first-outcome YES summary; no independent CLOB token-book verification',
        fees_depth_and_slippage_included=False, exchange_quote_update_time_verified=False)
    scenario = None
    if assumed_yes_returns_pct is not None or assumed_no_returns_pct is not None:
        scenario = evaluate_explicit_yes_no_scenarios(
            portfolio_rows=portfolio_rows, notional=notional,
            source_contract_id='polymarket-gamma:'+selected['market_id'],
            source_yes_question=question,
            assumed_yes_returns_pct=assumed_yes_returns_pct or {},
            assumed_no_returns_pct=assumed_no_returns_pct or {})
    limitations = [
        ('This recovered October 17 capture was used during development; no blind holdout claim.'
         if pinned_capture else
         'Supplied archive passes local receipt consistency only; real acquisition and source origin are unverified.'),
        'Source integrity is local SHA consistency, not a provider signature or full market coverage.',
        'Ranking uses holdings admissible at the catalog capture time; the separately evaluated ledger can differ.',
        'The supplied portfolio allocation is applied at both clocks; historical portfolio ownership is not verified.',
        'Historical indicative market prices are not current quotes or calibrated physical probabilities.',
        'Operator state returns, when supplied, are assumptions relative to current portfolio prices.',
        'No extra loss is added for an event merely because its risk is known; equity pricing-in remains unidentified.',
    ]
    if selected['family'] == 'TAIWAN_CONFLICT':
        limitations.append('A Taiwan-related contract is not automatically equivalent to a main-island invasion; full market rules control settlement.')
    return {
        'schema': 'castle.option1.archived_nonmacro_investigation.v1',
        'status': ('REAL_CAPTURE_REPLAY_EXACT_SOURCE_YES_NOT_FRESH_OR_AUTONOMOUS_LLM'
                   if pinned_capture else
                   'LOCAL_RECEIPT_CONSISTENT_ARCHIVE_REPLAY_SOURCE_ORIGIN_UNVERIFIED'),
        'evaluated_at_utc': now.isoformat(),
        'portfolio': portfolio.to_dict(),
        'exposure': evaluated_exposure,
        'source_capture_exposure': capture_exposure,
        'exposure_timing': {
            'ranking_as_of_utc': ranking_at.isoformat(),
            'evaluated_at_utc': now.isoformat(),
            'catalog_acquisition_completed_at_utc': acquisition_completed_at.isoformat(),
            'holdings_admission_differs': capture_exposure['entries'] != evaluated_exposure['entries'],
            'historical_portfolio_ownership_verified': False,
        },
        'source': {'archive_sha256': source['archive_sha256'],
            'matches_recovered_capture_sha256': pinned_capture,
            'origin_status': ('PINNED_EXISTING_CANDIDATE_CAPTURE'
                              if pinned_capture else 'SUPPLIED_ARCHIVE_ORIGIN_UNVERIFIED'),
            'verified_response_count': source['source_count'],
            'original_page_sha256': receipt['sha256'], 'original_page_bytes': receipt['bytes'],
            'original_page_path': receipt['raw_file'], 'original_url': receipt['url'],
            'retrieved_at_utc': receipt['completed_at_utc'],
            'receipt_provenance_kind': receipt.get('provenance_kind', 'UNSPECIFIED'),
            'provider_signed': False, 'child_extraction_is_derived_not_new_http_receipt': True},
        'selection': {'family': selected['family'],
            'ranking_as_of_utc': ranking_at.isoformat(),
            'candidate_deadline_window_as_of': now.date().isoformat(),
            'portfolio_relevance_score': selected['portfolio_relevance_score'],
            'method': 'EXISTING_DECLARED_PORTFOLIO_CHANNEL_HEURISTIC_NOT_FINANCIAL_LOSS',
            'relevance_evidence': selected['relevance_evidence'],
            'alternatives': [{k:c.get(k) for k in ('family','event_id','market_id','market_title',
                                                  'portfolio_relevance_score')}
                             for c in candidates[:12]]},
        'contract': {'event_id': selected['event_id'], 'market_id': selected['market_id'],
            'event_slug': slug, 'market_url': selected['market_url'],
            'yes_question': question, 'full_market_rules': rules,
            'rules_sha256': sha256(rules.encode()).hexdigest(),
            'source_defined_yes_identity': True,
            'broader_geopolitical_event_equivalence': 'NOT_ESTABLISHED'},
        'historical_price_reference': historical,
        'source_rule_facts': audit_gamma_rule_text(event_id=selected['event_id'],
            market_id=selected['market_id'], question=question, full_rules=rules,
            source_metadata={'archive_sha256': source['archive_sha256'],
                'original_page_sha256': receipt['sha256'],
                'retrieved_at_utc': receipt['completed_at_utc']}),
        'operator_assumed_original_position_scenario': scenario,
        'physical_probability_P': None, 'equity_priced_in_q': None,
        'identified_event_conditional_return': None,
        'new_provider_requests': 0, 'new_model_calls': 0,
        'limitations': limitations,
    }
