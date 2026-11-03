"""Receipt-bound Gamma price references; never executable or physical probabilities.

RECONSTRUCTED_FROM_USER_REPORTED_BEHAVIOR: the later 564-file source was absent.
The five-minute limit applies to local retrieval, NOT to exchange quote updates.
Gamma's first-outcome bid/ask convention is explicit and restricted to Yes/No
ordering. A CLOB token-specific book is not independently verified here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path

from .polymarket_gamma_receipts import GammaEvidenceError, _clock, replay_gamma_event

MAX_RECEIPT_AGE_SECONDS = 300


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _price(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError('Missing or nonnumeric price')
    if isinstance(value, str) and (not value or value.strip() != value):
        raise ValueError('Malformed decimal price')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Malformed decimal price') from exc
    if not number.is_finite() or not 0 <= number <= 1:
        raise ValueError('Price outside unit interval')
    return float(number)


def review_gamma_indicative_quotes(*, raw_path: Path | str,
                                   receipt_path: Path | str,
                                   expected_slug: str) -> dict:
    """Re-read source identity and hashes; no caller clock or TTL override.

Local receipts are unsigned: an actor replacing both bytes and receipt can
forge them. Integrity is relative to the capture, not provider authentication.
Fixtures and captures using a caller-supplied time never qualify.
"""
    review = replay_gamma_event(raw_path=raw_path, receipt_path=receipt_path,
                                expected_slug=expected_slug)
    raw = Path(raw_path).read_bytes()
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    # Reject changes between source replay and quote evaluation.
    if (sha256(raw).hexdigest() != review['raw_body_sha256']
            or receipt.get('sha256') != review['raw_body_sha256']
            or receipt.get('retrieved_at_utc') != review['retrieved_at_utc']
            or receipt.get('source_url') != review['source_url']
            or receipt.get('source_kind') != ('PUBLIC_HTTPS_GET_LOCALLY_CAPTURED'
                if review['source_is_live_captured'] else 'INJECTED_FIXTURE_NOT_LIVE')):
        raise GammaEvidenceError('Source changed during quote review')
    document = json.loads(raw)
    now = _utc_now()
    age = (now - _clock(review['retrieved_at_utc'])).total_seconds()
    common_reasons = []
    if not review['source_is_live_captured']:
        common_reasons.append('FIXTURE_NOT_LIVE')
    if receipt.get('clock_source') != 'SYSTEM_UTC_AFTER_RESPONSE':
        common_reasons.append('UNVERIFIED_OR_OVERRIDDEN_CAPTURE_CLOCK')
    if not 0 <= age <= MAX_RECEIPT_AGE_SECONDS:
        common_reasons.append('FUTURE_OR_STALE_RECEIPT')
    if document.get('active') is not True or document.get('closed') is not False:
        common_reasons.append('EVENT_NOT_CONFIRMED_ACTIVE')
    markets = []
    for original, identity in zip(document['markets'], review['markets'], strict=True):
        reasons = list(common_reasons)
        if (original.get('active') is not True or original.get('closed') is not False
                or original.get('archived') is not False
                or original.get('acceptingOrders') is not True
                or original.get('enableOrderBook') is not True):
            reasons.append('MARKET_NOT_CONFIRMED_OPEN_FOR_ORDERS')
        if original.get('umaResolutionStatus') in ('resolved', 'disputed'):
            reasons.append('RESOLUTION_STATUS_UNSUITABLE')
        try:
            if _clock(original.get('endDate')) <= now:
                reasons.append('MARKET_END_TIME_PASSED')
        except GammaEvidenceError:
            reasons.append('MARKET_END_TIME_UNVERIFIED')
        try:
            bid, ask = _price(original.get('bestBid')), _price(original.get('bestAsk'))
            if bid > ask:
                raise ValueError('Crossed price reference')
        except ValueError:
            bid = ask = None
            reasons.append('INVALID_BID_ASK')
        markets.append({
            'market_id': identity['market_id'], 'question': identity['question'],
            'rules_sha256': identity['rules_sha256'],
            'status': 'NOT_ADMITTED' if reasons else 'INDICATIVE_GAMMA_YES_PRICE_REFERENCE',
            'bid': None if reasons else bid, 'ask': None if reasons else ask,
            'reasons': reasons,
            'outcome_convention': 'FIRST_OUTCOME_YES_GAMMA_SUMMARY_NOT_CLOB_VERIFIED',
            'physical_probability_P': None, 'equity_priced_in_q': None,
            'conditional_portfolio_pnl_usd': None,
        })
    return {
        'implementation_origin': 'RECONSTRUCTED_FROM_USER_REPORTED_BEHAVIOR',
        'event_id': review['event_id'], 'event_slug': expected_slug,
        'source_sha256': review['raw_body_sha256'], 'source_url': review['source_url'],
        'retrieved_at_utc': review['retrieved_at_utc'],
        'evaluated_at_utc': now.isoformat(), 'receipt_age_seconds': age,
        'max_receipt_age_seconds': MAX_RECEIPT_AGE_SECONDS,
        'indicative_intervals_admitted': sum(not x['reasons'] for x in markets),
        'markets': markets,
        'exchange_quote_update_time_verified': False,
        'executable_order_book_verified': False, 'fees_and_slippage_included': False,
        'physical_probability_P': None, 'equity_priced_in_q': None,
        'limitations': [
            'Recent receipt does not establish recent exchange quote updates.',
            'Gamma summary prices are indicative; token-specific CLOB depth is not verified.',
            'Fees, slippage, financing, fill size and settlement risks are not included.',
            'Local SHA receipts are not provider signatures or tamper-proof external timestamps.',
            'These references do not identify physical probabilities or conditional equity returns.',
        ],
    }
