"""Read-only, bounded Polymarket Gamma event-source capture and replay.

This supplements the existing P1 discovery engine, without treating Gamma's
market price as a physical probability or source-page terms as an equity return.
The capture performs exactly one public HTTPS GET per explicitly selected slug,
only on a caller action. No network request at import time, no API keys, trades,
redirects, quotes from cached models, or automatic source promotion.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable, Sequence, Any
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

_HOST = 'gamma-api.polymarket.com'
_SLUG = re.compile(r'\A[a-z0-9](?:[a-z0-9-]{0,110}[a-z0-9])?\Z')
_CAP = 2_000_000


class GammaEvidenceError(ValueError):
    """Fail-closed validation for a public nonmacro source capture."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_get(url: str, timeout: float) -> tuple[int, bytes]:
    request = Request(url, method='GET', headers={
        'User-Agent': 'CastlePortfolioResearch/0.4 public-read-only',
        'Accept': 'application/json',
    })
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        return response.status, response.read(_CAP + 1)


def _source_url(slug: str) -> str:
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        raise GammaEvidenceError('Unsafe or invalid Gamma event slug')
    return f'https://{_HOST}/events/slug/{quote(slug, safe="")}'


def _clock(value: str) -> datetime:
    if not isinstance(value, str):
        raise GammaEvidenceError('UTC timestamp required')
    try:
        instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise GammaEvidenceError('Invalid receipt observation time') from exc
    if instant.tzinfo is None:
        raise GammaEvidenceError('Receipt time must be timezone aware')
    return instant.astimezone(timezone.utc)


def _parse_source(body: bytes, slug: str) -> dict:
    if not isinstance(body, bytes) or not 0 < len(body) <= _CAP:
        raise GammaEvidenceError('Invalid or oversized Gamma response')
    try:
        doc = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise GammaEvidenceError('Gamma response is not valid JSON') from exc
    if not isinstance(doc, dict) or doc.get('slug') != slug or not isinstance(doc.get('id'), str):
        raise GammaEvidenceError('Event identity does not match requested slug')
    markets = doc.get('markets')
    if not isinstance(markets, list) or not 1 <= len(markets) <= 50:
        raise GammaEvidenceError('Unbounded or missing Gamma markets')
    records: list[dict[str, Any]] = []
    seen_ids = set()
    for market in markets:
        if not isinstance(market, dict) or not isinstance(market.get('id'), str):
            raise GammaEvidenceError('Unrecognized market identity')
        if market['id'] in seen_ids:
            raise GammaEvidenceError('Duplicate Gamma market ID')
        seen_ids.add(market['id'])
        question = market.get('question')
        if not isinstance(question, str) or not 1 <= len(question) <= 700:
            raise GammaEvidenceError('Missing or overlong source question')
        try:
            outcomes = json.loads(market.get('outcomes', ''))
        except (TypeError, ValueError):
            raise GammaEvidenceError('Missing source-native binary outcomes') from None
        if outcomes != ['Yes', 'No']:
            raise GammaEvidenceError('Expected explicit binary YES/NO source order')
        rules = market.get('description')
        if not isinstance(rules, str) or not rules.strip() or len(rules) > 30_000:
            # Event-level description may describe a different child market.
            raise GammaEvidenceError('Market-level settlement rules are required')
        price = {'status': 'NOT_CERTIFIED_AS_LIVE_BID_ASK', 'bid': None, 'ask': None}
        records.append({
            'market_id': market['id'], 'question': question,
            'settlement_rules': rules, 'rules_sha256': sha256(rules.encode()).hexdigest(),
            'market_slug': market.get('slug'), 'status': market.get('active'),
            'closed': market.get('closed'), 'best_bid_source_field': market.get('bestBid'),
            'best_ask_source_field': market.get('bestAsk'), 'quote': price,
            'binary_yes_no_source_identity': True,
            'equivalence_to_a_proposed_economic_event': 'NOT_ESTABLISHED',
            'physical_probability_P': None, 'equity_priced_in_q': None,
            'conditional_portfolio_pnl_usd': None,
        })
    return {'event_id': doc['id'], 'event_slug': slug,
            'event_title': doc.get('title'), 'markets': records}


def capture_gamma_events(*, slugs: Sequence[str], output_dir: Path | str,
                         transport: Callable[[str, float], tuple[int, bytes]] | None = None,
                         timeout: float = 8., observed_at: datetime | None = None) -> list[dict]:
    """Write original bytes plus transparent receipt metadata, never provider-signed.

    `transport` is solely for offline tests and always marks its files FIXTURE.
    An HTTP read failure is propagated, not rewritten as a fictitious contract.
    """
    if not isinstance(slugs, (list, tuple)) or not 1 <= len(slugs) <= 4:
        raise GammaEvidenceError('Between one and four bounded event slugs required')
    if len(set(slugs)) != len(slugs):
        raise GammaEvidenceError('Duplicate public event slugs')
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not 0 < timeout <= 20:
        raise GammaEvidenceError('Timeout must be bounded')
    urls = [_source_url(slug) for slug in slugs]  # Validate ALL inputs before any GET.
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    reader = transport or _read_get
    captured = []
    for slug, url in zip(slugs, urls):
        status, body = reader(url, timeout)
        if status != 200:
            raise GammaEvidenceError('Non-200 public Gamma response')
        source = _parse_source(body, slug)
        now = observed_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise GammaEvidenceError('UTC observation time required')
        timestamp = now.astimezone(timezone.utc).isoformat()
        file = base / f'gamma-{slug}.json'
        receipt_file = base / f'gamma-{slug}.receipt.json'
        # Persist original provider response bytes unchanged, rather than a
        # re-serialized model-visible JSON dict.
        file.write_bytes(body)
        receipt = {
            'source_url': url, 'method': 'GET', 'http_status': status,
            'sha256': sha256(body).hexdigest(), 'bytes': len(body),
            'retrieved_at_utc': timestamp, 'source_kind': ('INJECTED_FIXTURE_NOT_LIVE'
                              if transport is not None else 'PUBLIC_HTTPS_GET_LOCALLY_CAPTURED'),
            'clock_source': ('CALLER_OVERRIDE_NOT_LIVE' if observed_at is not None
                             else 'SYSTEM_UTC_AFTER_RESPONSE'),
            'provider_signed': False, 'exchange_quote_update_time_verified': False,
        }
        receipt_file.write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        captured.append({'slug': slug, 'receipt_sha256': receipt['sha256'],
                         'source_path': str(file), 'receipt_path': str(receipt_file),
                         'captured_contract_count': len(source['markets']),
                         'source_kind': receipt['source_kind']})
    return captured


def replay_gamma_event(*, raw_path: Path | str, receipt_path: Path | str,
                       expected_slug: str) -> dict:
    """Verify original bytes, request origin and market-level rules without a GET.

    A fixture receipt remains fixture-class even if its text matches a real
    market. Capture metadata and SHA are local checks, not an exchange signature.
    """
    expected_url = _source_url(expected_slug)
    raw = Path(raw_path).read_bytes()
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    u = urlsplit(str(receipt.get('source_url', '')))
    if (u.scheme != 'https' or u.hostname != _HOST or u.username or u.password
            or u.query or u.fragment or receipt.get('source_url') != expected_url
            or receipt.get('method') != 'GET' or receipt.get('http_status') != 200
            or receipt.get('sha256') != sha256(raw).hexdigest()
            or receipt.get('bytes') != len(raw)):
        raise GammaEvidenceError('Captured Gamma response and receipt mismatch')
    source = _parse_source(raw, expected_slug)
    _clock(receipt['retrieved_at_utc'])
    if receipt.get('source_kind') not in ('PUBLIC_HTTPS_GET_LOCALLY_CAPTURED',
                                          'INJECTED_FIXTURE_NOT_LIVE'):
        raise GammaEvidenceError('Unrecognized receipt provenance')
    source.update({
        'source_assurance': ('ORIGINAL_LOCAL_PUBLIC_GET_BYTES_HASH_VERIFIED_NOT_PROVIDER_SIGNED'
                             if receipt['source_kind'] == 'PUBLIC_HTTPS_GET_LOCALLY_CAPTURED'
                             else 'FIXTURE_BYTES_VERIFIED_NOT_LIVE_EVIDENCE'),
        'source_url': expected_url, 'raw_body_sha256': receipt['sha256'],
        'retrieved_at_utc': receipt['retrieved_at_utc'],
        'source_is_live_captured': receipt['source_kind'] == 'PUBLIC_HTTPS_GET_LOCALLY_CAPTURED',
        'market_quote_intervals_admitted': 0,
        'full_exchange_settlement_documents_independently_verified': False,
        'original_macro_event_equivalence': 'NOT_ESTABLISHED',
        'physical_probability_P': None, 'conditional_equity_pnl_usd': None,
    })
    return source
