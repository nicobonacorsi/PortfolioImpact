"""Narrow, reproducible financial anchors for a reviewed operating-risk channel.

This is not a generic XBRL parser, a forecast, or a source of event probability.
Only the reviewed filing's exact bytes, facts, periods, units and dimensions
are supported. A new filing requires a new source review, not a silent fallback.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse


SOURCE_URL = 'https://www.sec.gov/Archives/edgar/data/100885/000010088526000321/unp-20260930.htm'
SOURCE_SHA256 = 'ec38c804ad5597fc531971e2dc28d03c99055c78b371bc9b6286b38d07fca10b'
SOURCE_RECEIVED = '2026-10-28T16:35:54.481956+00:00'
SOURCE_PATH = Path(__file__).resolve().parents[1] / 'data' / 'company_facts' / 'unp_2026q3.html'
_HASH = re.compile(r'[0-9a-f]{64}')
_ISO_INSTANT = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?'
    r'(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)'
)


class BusinessSourceError(ValueError):
    """A financial anchor cannot be independently reconstructed as specified."""


def _instant(value):
    try:
        if isinstance(value, str):
            if not _ISO_INSTANT.fullmatch(value):
                raise ValueError
            moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            moment = value
        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise ValueError
        return moment.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise BusinessSourceError('A timezone-aware evaluation/source time is required.') from exc


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessSourceError(f'Invalid positive numeric {name}.')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise BusinessSourceError(f'Invalid positive numeric {name}.') from exc
    if not math.isfinite(number) or number <= 0:
        raise BusinessSourceError(f'Invalid positive numeric {name}.')
    return number


def _record_bytes(value, *, name, maximum, suffix=None):
    """Read one bounded original record, with controlled malformed-path failures."""
    if not isinstance(value, (str, Path)):
        raise BusinessSourceError(f'Invalid {name} path.')
    try:
        path = Path(value)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > maximum
                or suffix is not None and path.suffix != suffix):
            raise BusinessSourceError(f'{name} unavailable or invalid.')
        body = path.read_bytes()
    except (OSError, ValueError, TypeError) as exc:
        raise BusinessSourceError(f'{name} unavailable or invalid.') from exc
    if len(body) > maximum:
        raise BusinessSourceError(f'{name} exceeds its allowed byte size.')
    return path, body


class _ReviewedFilingParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.contexts, self.facts, self.units = {}, [], {}
        self.context = self.fact = self.unit = None
        self.context_field = self.member = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'xbrli:context':
            self.context = {'id': attrs.get('id'), 'members': []}
        elif self.context is not None and tag in ('xbrli:identifier', 'xbrli:startdate', 'xbrli:enddate', 'xbrli:instant'):
            self.context_field = tag.split(':')[1]
            self.context[self.context_field] = ''
        elif self.context is not None and tag == 'xbrldi:explicitmember':
            self.member = [attrs.get('dimension'), '']
        elif tag == 'xbrli:unit':
            self.unit = {'id': attrs.get('id'), 'measure': ''}
        elif tag == 'ix:nonfraction' and attrs.get('id') in {'f-24', 'f-30'}:
            if self.fact is not None:
                raise BusinessSourceError('Nested numeric filing facts are unsupported.')
            self.fact = {'attrs': attrs, 'text': ''}

    def handle_data(self, data):
        if self.fact is not None:
            self.fact['text'] += data
        if self.context is not None and self.context_field:
            self.context[self.context_field] += data
        if self.member is not None:
            self.member[1] += data
        if self.unit is not None:
            self.unit['measure'] += data

    def handle_endtag(self, tag):
        if tag == 'ix:nonfraction' and self.fact is not None:
            self.facts.append(self.fact)
            self.fact = None
        elif tag == 'xbrli:context' and self.context is not None:
            key = self.context['id']
            if key in self.contexts:
                raise BusinessSourceError('Duplicate filing context.')
            self.contexts[key] = self.context
            self.context = None
        elif tag in ('xbrli:identifier', 'xbrli:startdate', 'xbrli:enddate', 'xbrli:instant'):
            self.context_field = None
        elif tag == 'xbrldi:explicitmember' and self.context is not None and self.member is not None:
            self.context['members'].append(tuple(self.member))
            self.member = None
        elif tag == 'xbrli:unit' and self.unit is not None:
            if self.unit['id'] in self.units:
                raise BusinessSourceError('Duplicate filing unit.')
            self.units[self.unit['id']] = self.unit['measure'].strip()
            self.unit = None


def _reviewed_fact(parser, *, identifier, concept, context_id, unit, scale):
    rows = [row for row in parser.facts if row['attrs'].get('id') == identifier]
    if len(rows) != 1:
        raise BusinessSourceError('The reviewed filing fact is missing or duplicated.')
    row = rows[0]
    attrs = row['attrs']
    if (attrs.get('name') != concept or attrs.get('contextref') != context_id
            or attrs.get('unitref') != unit or attrs.get('scale') != str(scale)
            or attrs.get('sign') not in (None, '+')
            or attrs.get('nil') in ('true', '1') or attrs.get('xsi:nil') in ('true', '1')):
        raise BusinessSourceError('Reviewed fact concept, unit, period or sign mismatch.')
    text = row['text'].strip().replace(',', '')
    if not re.fullmatch(r'\d+(?:\.\d+)?', text):
        raise BusinessSourceError('Unsupported numeric representation in filing fact.')
    try:
        number = float(Decimal(text) * (Decimal(10) ** scale))
    except (InvalidOperation, OverflowError) as exc:
        raise BusinessSourceError('Invalid filing number.') from exc
    return _positive(number, 'reviewed filing fact')


def load_unp_source_anchor(*, evaluated_at, source_path=None):
    """Reconstruct facts only once the actual retained source was received.

    The 120-day period-end age cap is a refresh policy, not a statement that
    seasonality or future business conditions equal the recorded quarter.
    """
    now = _instant(evaluated_at)
    received = _instant(SOURCE_RECEIVED)
    if now < received:
        raise BusinessSourceError('This source inspection postdates the requested evaluation.')
    if (now.date() - date(2026, 9, 30)).days > 120:
        raise BusinessSourceError('The reviewed financial anchor needs refresh.')
    _, body = _record_bytes(SOURCE_PATH if source_path is None else source_path,
                            name='Reviewed filing bytes', maximum=2_000_000)
    if sha256(body).hexdigest() != SOURCE_SHA256:
        raise BusinessSourceError('The retained filing does not reproduce its reviewed source hash.')
    parser = _ReviewedFilingParser()
    try:
        parser.feed(body.decode('utf-8'))
    except UnicodeError as exc:
        raise BusinessSourceError('Reviewed filing is not valid UTF-8.') from exc
    if parser.units.get('usd') != 'iso4217:USD' or parser.units.get('shares') != 'xbrli:shares':
        raise BusinessSourceError('Filing units differ from the reviewed USD/share units.')
    context = parser.contexts.get('c-3', {})
    if (context.get('identifier') != '0000100885' or context.get('startdate') != '2026-07-01'
            or context.get('enddate') != '2026-09-30'
            or context.get('members') != [('srt:ProductOrServiceAxis', 'us-gaap:CargoAndFreightMember')]):
        raise BusinessSourceError('Issuer, freight segment or fiscal-quarter context mismatch.')
    share_context = parser.contexts.get('c-2', {})
    if (share_context.get('identifier') != '0000100885' or share_context.get('instant') != '2026-10-16'
            or share_context.get('members') != []):
        raise BusinessSourceError('Outstanding common-share date/scope mismatch.')
    revenue = _reviewed_fact(parser, identifier='f-30',
        concept='us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
        context_id='c-3', unit='usd', scale=6)
    shares = _reviewed_fact(parser, identifier='f-24',
        concept='dei:EntityCommonStockSharesOutstanding', context_id='c-2', unit='shares', scale=0)
    if revenue != 6_813_000_000 or shares != 594_250_416:
        raise BusinessSourceError('Fact values differ from the completed documentary review.')
    anchor = {'issuer': 'UNP', 'revenue_usd': revenue,
              'period_start': '2026-07-01', 'period_end': '2026-09-30',
              'revenue_metric': 'Reported freight revenue (including fuel surcharges)',
              'source_record_id': 'SEC_0000100885_26_000321_F30'}
    source = {'status': 'EXACT_REVIEWED_FILING_BYTES_AND_FACTS_REPRODUCED',
              'source_url': SOURCE_URL, 'response_sha256': SOURCE_SHA256,
              'received_at_utc': SOURCE_RECEIVED, 'publication_date': '2026-10-22',
              'period_start': anchor['period_start'], 'period_end': anchor['period_end'],
              'reported_revenue_usd': revenue, 'reported_revenue_precision_usd': 1_000_000,
              'revenue_context_id': 'c-3', 'revenue_fact_id': 'f-30',
              'common_shares_outstanding': shares, 'common_shares_as_of': '2026-10-16',
              'share_fact_id': 'f-24', 'source_record_id': anchor['source_record_id'],
              'verification_scope': 'Local byte consistency and reviewed filing facts, not a provider signature or a forecast.'}
    return anchor, source


class _FilingPlainText(HTMLParser):
    """Normalize narrative whitespace without rewriting the original filing."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def load_unp_mexico_exposure_anchor(*, evaluated_at, source_path=None):
    """Recover a *direction-agnostic* Mexico-linked freight revenue baseline.

    The paragraph describes both inbound and outbound freight. It does not
    identify the import-only share, tariff elasticity or a future-quarter run
    rate. Exact pinned bytes, publisher context, dates and receipt precede use.
    """
    freight, source = load_unp_source_anchor(evaluated_at=evaluated_at, source_path=source_path)
    _, body = _record_bytes(SOURCE_PATH if source_path is None else source_path,
                            name='Reviewed Mexico revenue filing', maximum=2_000_000)
    if sha256(body).hexdigest() != SOURCE_SHA256:
        raise BusinessSourceError('The Mexico freight statement is not in the reviewed filing bytes.')
    narrative = _FilingPlainText()
    try:
        narrative.feed(body.decode('utf-8'))
    except UnicodeError as exc:
        raise BusinessSourceError('The reviewed Mexico narrative has invalid encoding.') from exc
    normalized = ' '.join(' '.join(narrative.parts).split())
    heading = 'Mexico business'
    if normalized.count(heading) != 1:
        raise BusinessSourceError('The Mexico business paragraph is missing or ambiguous.')
    paragraph = normalized[normalized.index(heading):][:900]
    reviewed = re.compile(
        r'Mexico business\s*[–-]\s*Freight revenue\s*s from each of our commodity groups '
        r'includes revenu\s*es from shipments to and from Mexico, which increased 9%, '
        r'to \$820 million, in the third quarter of 2026\b'
    )
    if not reviewed.search(paragraph):
        raise BusinessSourceError('The original Mexico two-direction revenue claim changed.')
    mexico = 820_000_000.
    if mexico <= 0 or mexico > freight['revenue_usd']:
        raise BusinessSourceError('Mexico exposure is inconsistent with total reviewed freight revenues.')
    anchor = {'issuer': 'UNP', 'mexico_linked_freight_revenue_usd': mexico,
              'period_start': '2026-07-01', 'period_end': '2026-09-30',
              'calendar_days': 92,
              'revenue_metric': 'Reported freight revenues from shipments to AND from Mexico',
              'source_record_id': 'SEC_0000100885_26_000321_MEXICO_BUSINESS_PARAGRAPH',
              'mexico_import_only_fraction_identified': False,
              'future_period_revenue_identified': False,
              'tariff_effect_identified': False}
    source = {**source, 'reported_mexico_linked_freight_revenue_usd': mexico,
              'narrative_locator': '2026 Q3 MD&A; Mexico business paragraph',
              'direction_scope': 'TO_AND_FROM_MEXICO_INCLUDES_NON_IMPORTS',
              'reported_percentage_change': 0.09,
              'verification_scope': 'Exact source bytes and reviewed narrative; no import share or tariff elasticity inferred.'}
    return anchor, source


def checked_unp_capitalization(metadata, *, evaluated_at):
    """Use an already captured Massive common-stock reference response only.

    Normalized summaries, issuer descriptions and copied approval flags cannot
    supply the number. No network request is made. The recorded market cap is
    a reference denominator, not a real-time executable share quote.
    """
    now = _instant(evaluated_at)
    if not isinstance(metadata, dict) or metadata.get('ticker') != 'UNP' or metadata.get('type') != 'CS':
        raise BusinessSourceError('Original UNP common-stock metadata is required.')
    received = _instant(metadata.get('retrieved_at'))
    if not timedelta(0) <= now - received <= timedelta(days=3):
        raise BusinessSourceError('The capitalization capture is future-dated or needs refresh.')
    url = metadata.get('source_url')
    if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise BusinessSourceError('Missing capitalization source URL.')
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=4)
    except (ValueError, TypeError) as exc:
        raise BusinessSourceError('Invalid capitalization source URL.') from exc
    if (parsed.scheme != 'https' or parsed.netloc != 'api.massive.com'
            or parsed.path != '/v3/reference/tickers/UNP' or parsed.params or parsed.fragment
            or query != {'date': [received.date().isoformat()]}):
        raise BusinessSourceError('Capitalization source identity/date mismatch.')
    raw = metadata.get('raw_path')
    digest = metadata.get('source_response_sha256')
    if not isinstance(raw, str) or not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise BusinessSourceError('Original capitalization bytes and hash are required.')
    path, body = _record_bytes(raw, name='Original capitalization record', maximum=1_000_000,
                               suffix='.json')
    if sha256(body).hexdigest() != digest:
        raise BusinessSourceError('Capitalization source hash mismatch.')
    meta_path = path.with_suffix('.meta.json')
    _, receipt_bytes = _record_bytes(meta_path, name='Capitalization receipt', maximum=1_000_000)
    try:
        payload = json.loads(body)
        receipt = json.loads(receipt_bytes)
    except (ValueError, UnicodeError) as exc:
        raise BusinessSourceError('Invalid capitalization JSON.') from exc
    if (not isinstance(receipt, dict) or receipt.get('sha256') != digest or receipt.get('url') != url
            or type(receipt.get('status')) is not int or receipt.get('status') != 200
            or receipt.get('fixture') is not False
            or receipt.get('retrieved_at') != metadata.get('retrieved_at')
            or not isinstance(payload, dict) or payload.get('status') != 'OK'):
        raise BusinessSourceError('Capitalization receipt or payload identity mismatch.')
    result = payload.get('results')
    if (not isinstance(result, dict) or result.get('ticker') != 'UNP' or result.get('type') != 'CS'
            or not isinstance(result.get('currency_name'), str)
            or result['currency_name'].lower() != 'usd' or result.get('active') is not True):
        raise BusinessSourceError('Capitalization issuer, class or currency is unsupported.')
    value = _positive(result.get('market_cap'), 'market capitalization')
    shares = _positive(result.get('share_class_shares_outstanding'), 'share class outstanding')
    # This reviewed component supports UNP ordinary common stock, not an ADR,
    # post-merger share count, split-adjusted synthetic series or another class.
    if abs(shares / 594_250_416 - 1) > .01:
        raise BusinessSourceError('Share count changed materially; reviewed valuation bridge needs refresh.')
    return {'market_capitalization_usd': value,
            'source_url': url, 'response_sha256': digest,
            'received_at_utc': metadata['retrieved_at'], 'currency': 'USD',
            'source_record_id': 'MASSIVE_UNP_REFERENCE_' + digest[:16],
            'basis': 'RECORDED_REFERENCE_MARKET_CAP_NOT_INTRADAY_EXECUTABLE_PRICE',
            'share_class_shares_outstanding': shares}
