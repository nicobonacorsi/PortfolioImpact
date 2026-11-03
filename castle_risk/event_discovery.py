"""Evidence-gated portfolio event discovery from dated, bounded public catalogs.

This is discovery of *candidates*, not contract-payoff equivalence, real-world
probabilities, or event-to-equity return estimates. Historical P1 input is
explicitly development evidence; a later evaluation set must be separately frozen.
No network call occurs on import or during historical replay.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit
from zipfile import ZipFile, BadZipFile

from .portfolio import parse_portfolio, recover_exposures

MAX_ARCHIVE_BYTES = 12_000_000
MAX_RAW_BYTES = 16_000_000
MAX_EXPANDED_BYTES = 180_000_000
MAX_CAPTURE_REQUESTS = 60
MAX_EVENT_RECORDS = 100_000
FAMILY_VERSION = 'DISCOVERY_FAMILIES_V1_C1C3_2026_10_17'

# Each expression is a fixed, auditable *candidate detector*, not a settlement
# verifier. No catalog entry is assigned a relationship solely by matching text.
# The specificity rules prevent generic mentions of an industry from equating to
# verified holdings in a company with a related product.
FAMILY_RULES = (
    ('HORMUZ_TRAFFIC', r'\bhormuz\b', 'energy_supply', 'ENERGY_SUPPLY'),
    ('US_IRAN_CONFLICT', r'\b(iran|iranian)\b.{0,90}\b(invad(?:e|es|ing|ed)|invasion|strike|war|attack|military|regime|nuclear)\b|\b(invad(?:e|es|ing|ed)|invasion|strike|war|attack|military)\b.{0,80}\b(iran|iranian)\b', 'geopolitics', 'ENERGY_SUPPLY'),
    ('TAIWAN_CONFLICT', r'\b(taiwan|roc)\b.{0,90}\b(invad(?:e|es|ing|ed)|invasion|attack|military|blockade|clash|offensive)\b|\b(invad(?:e|es|ing|ed)|invasion|attack|blockade|offensive|military)\b.{0,90}\b(taiwan|roc)\b', 'geopolitics', 'TECH_SUPPLY'),
    ('US_SOVEREIGN_CREDIT', r'\b(us|u\.s\.|united states|american|sovereign|treasury)\b.{0,80}\b(defaults?|downgrades?|credit rating)\b|\b(defaults?|downgrades?)\b.{0,80}\b(us|u\.s\.|treasury|sovereign)\b', 'macro_credit', 'TREASURY_CREDIT'),
    ('US_DEBT_LIMIT', r'\b(debt limit|debt ceiling|statutory limit on the public debt)\b', 'macro_credit', 'TREASURY_CREDIT'),
    ('TREASURY_YIELD', r'\b(treasury|10[- ]?year|10y|30[- ]?year|30y)\b.{0,65}\b(yield|rate)\b|\b(yield|rate)\b.{0,60}\b(treasury|10[- ]?year|10y|30[- ]?year|30y)\b', 'macro_rates', 'DURATION'),
    ('ENERGY_POLICY', r'\b(opec|oil output|oil production|oil price|crude oil|petroleum|lng|natural gas)\b', 'energy', 'ENERGY_SUPPLY'),
    ('EXPORT_CONTROL', r'\b(export control|chip ban|semiconductor export|advanced chips|ai chips|nvidia export|nvda export)\b', 'technology_policy', 'TECH_SUPPLY'),
    ('TRADE_TARIFF', r'\b(tariff|import dut(?:y|ies)|section 301|trade restriction)\b', 'trade_policy', 'TRADE'),
    ('US_CORPORATE_TAX', r'\b(corporate taxes|corporate tax|corporation tax(?:es)?|corporate income tax|corporate tax rate)\b', 'fiscal_policy', 'EARNINGS'),
    ('FED_RATES', r'\b(fomc|federal reserve|fed interest rate|fed rate decision)\b', 'macro_rates', 'DURATION'),
    ('US_INFLATION', r'\b(cpi|consumer price index|inflation release)\b', 'macro_inflation', 'DURATION'),
    ('VENEZUELA_ENERGY', r'\b(venezuela|venezuelan)\b.{0,90}\b(opec|oil|military|invasion|intervene|attack)\b|\b(opec|oil|military|invasion)\b.{0,80}\bvenezuela\b', 'geopolitics', 'ENERGY_SUPPLY'),
    ('NATO_RUSSIA', r'\bnato\b.{0,70}\brussia\b|\brussia\b.{0,70}\bnato\b', 'geopolitics', 'GLOBAL_RISK'),
)
# These are prospective, labeled economic-transmission assumptions, NOT
# calibrated sensitivities. A missing ETF holdings snapshot never magically
# becomes a verified component position.
PARENT_CHANNELS = {
    'NVDA': {'TECH_SUPPLY': (1., 'issuer/product supply-chain hypothesis, not verified by the exchange'),
             'EARNINGS': (.5, 'issuer earnings/tax sensitivity is hypothetical'),
             'GLOBAL_RISK': (.2, 'unvalidated global-risk sensitivity'),
             'TRADE': (.35, 'trade-policy pass-through is hypothetical')},
    'QQQ': {'TECH_SUPPLY': (.45, 'technology-heavy ETF parent-level proxy; not a constituent audit'),
             'DURATION': (.35, 'equity duration hypothesis, not an estimated beta'),
             'EARNINGS': (.45, 'broad earnings/tax hypothesis'),
             'GLOBAL_RISK': (.2, 'global-risk hypothesis'),
             'TREASURY_CREDIT': (.15, 'market-wide funding hypothesis'),
             'TRADE': (.25, 'broad trade-policy hypothesis')},
    'XLE': {'ENERGY_SUPPLY': (1., 'energy-sector ETF parent-level channel; constituent weights not inferred'),
             'GLOBAL_RISK': (.25, 'risk-premium channel is sign-indeterminate'),
             'TRADE': (.15, 'energy tariff channel requires product exclusions review')},
    'TLT': {'DURATION': (1., 'long Treasury duration; issuer-grounded duration exists separately'),
             'TREASURY_CREDIT': (.8, 'US Treasury issuer/funding channel; direction not assumed'),
             'GLOBAL_RISK': (.25, 'flight-to-safety/cross-asset return direction unknown')},
    'IEF': {'DURATION': (.8, 'intermediate Treasury duration parent proxy'),
             'TREASURY_CREDIT': (.65, 'US Treasury issuer/funding channel')},
    'SPY': {'DURATION': (.2, 'broad equity rate sensitivity hypothesis'),
             'GLOBAL_RISK': (.2, 'market-wide risk-premium hypothesis'),
             'EARNINGS': (.5, 'broad earnings hypothesis'),
             'TRADE': (.25, 'broad US trade-policy exposure hypothesis')},
}
# Primary issuer disclosures support *exposure pathways only*. They provide no
# event-to-price coefficient. Retrieval via research browser does not create a
# raw HTTP receipt in this application; dates/versions are stated explicitly.
ISSUER_CHANNEL_SOURCES = {
  ('NVDA','TECH_SUPPLY'): {
    'url':'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm',
    'description':'NVIDIA FY2026 Form 10-K: Taiwan Semiconductor Manufacturing Company named among semiconductor foundries.',
    'source_class':'ISSUER_SEC_FILING_REFERENCED_NOT_RAW_ARCHIVED_IN_THIS_BUILD',
    'limitations':'Supply-chain dependence is supported, event conditional equity returns are not identified.'},
  ('XLE','ENERGY_SUPPLY'): {
    'url':'https://www.ssga.com/us/en/institutional/etfs/state-street-energy-select-sector-spdr-etf-xle',
    'description':'Issuer states XLE targets oil, gas, energy equipment and services sector exposure.',
    'source_class':'ISSUER_OFFICIAL_PRODUCT_DESCRIPTION_NOT_HOLDINGS_CAPTURE',
    'limitations':'No member-specific verified constituent weights or oil-return coefficient inferred.'},
  ('TLT','DURATION'): {
    'url':'https://www.ishares.com/us/products/239454/ishares-20%20-year-treasury-bond-etf',
    'description':'Official iShares TLT issuer duration characteristics are separately archived and verified by event_impact.',
    'source_class':'EXISTING_ARCHIVED_ISSUER_CHARACTERISTIC',
    'limitations':'The rate change conditional on a new event is not identified.'},
}

MONTHS = {s: i for i, name in enumerate(('January','February','March','April','May','June','July','August','September','October','November','December'),1)
          for s in (name.lower(), name[:3].lower())}
DATE_TOKEN = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
DATE_RE = re.compile(r'\b(?P<dayprefix>before|by|through|until|on|in)\s+(?:the\s+)?(?P<month>' + DATE_TOKEN + r')\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>20\d\d)\b', re.I)
MONTH_RE = re.compile(r'\b(?:in|during|through)\s+(?P<month>' + DATE_TOKEN + r')\s+(?P<year>20\d\d)\b', re.I)
YEAR_RE = re.compile(r'\b(?:before|by)\s+(?P<year>20\d\d)\b', re.I)

# Use the *second* explicit date of a 'between DATE and DATE' window.
# The first date is an origin, not the future outcome deadline. Neither date
# proves a contract payoff or that the event has not already occurred.
BETWEEN_RE = re.compile(
    r'\bbetween\s+(?:the\s+)?' + DATE_TOKEN +
    r'\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*20\d\d)?\s*,?\s+and\s+'
    r'(?:the\s+)?(?P<month>' + DATE_TOKEN +
    r')\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>20\d\d)\b', re.I)
# A positive clause may contain 'or otherwise ceases'. Stop only at the
# *start of another sentence*, not on any occurrence of the word 'otherwise'.
FALLBACK_SENTENCE_RE = re.compile(
    r'(?:[.!?]\s+|\n+)\s*(?:otherwise\s*,|if\s+not\b)', re.I)



class DiscoveryError(ValueError):
    """A source cannot be admitted without a complete validated provenance chain."""


def _safe_time(x: Any) -> datetime | None:
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        d = datetime.fromisoformat(x.replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except ValueError:
        return None


def _str(v: Any, limit: int = 12000) -> str:
    return v[:limit] if isinstance(v, str) else ''


def _temporal_definition(text: str) -> tuple[str | None, str]:
    """Extract a *candidate* economic deadline from explicit terms; never certify payoff."""
    text = _str(text, 6500)
    # Only the first decisive positive-settlement clause should set the target.
    # A fallback paragraph may contain several unrelated dates.
    positive = FALLBACK_SENTENCE_RE.split(text, maxsplit=1)[0]
    between = BETWEEN_RE.search(positive)
    if between:
        try:
            endpoint = date(int(between['year']), MONTHS[between['month'].lower()[:3]], int(between['day']))
            return endpoint.isoformat(), 'RULES_EXPLICIT_BETWEEN_END_CANDIDATE_NOT_PAYOFF_CERTIFIED'
        except ValueError:
            return None, 'INVALID_DATE_IN_RULES'
    m = DATE_RE.search(positive)
    if m:
        try:
            value = date(int(m['year']),MONTHS[m['month'].lower().rstrip('.')[:3]],int(m['day']))
            return value.isoformat(), 'RULES_EXPLICIT_DATE_CANDIDATE_NOT_PAYOFF_CERTIFIED'
        except ValueError:
            return None,'INVALID_DATE_IN_RULES'
    m = MONTH_RE.search(positive)
    if m:
        try:
            yr,mo = int(m['year']),MONTHS[m['month'].lower()[:3]]
            nxt=date(yr+1,1,1) if mo==12 else date(yr,mo+1,1)
            return (nxt-timedelta(days=1)).isoformat(),'RULES_EXPLICIT_MONTH_END_CANDIDATE'
        except ValueError:
            return None,'INVALID_MONTH_IN_RULES'
    m=YEAR_RE.search(positive)
    if m:
        return f"{int(m['year'])-1}-12-31",'RULES_EXPLICIT_YEAR_END_CANDIDATE'
    return None,'NO_VERIFIED_EVENT_DEADLINE_IN_RULES'


def _classify(e: dict, market: dict, venue: str) -> list[tuple[str,str,str]]:
    # Event title and market YES question are candidate identity; generic rules
    # are not searched for topic keywords (they can mention other governments).
    title = ' '.join((_str(e.get('title')), _str(market.get('title')), _str(market.get('question')),
                      _str(market.get('yes_sub_title'))))[:1000]
    found = []
    for name, pattern, category, channel in FAMILY_RULES:
        if not re.search(pattern, title, re.I):
            continue
        # Voting on Iran-related *issues* is not a war or military event.
        if name == 'US_IRAN_CONFLICT' and re.search(
                r'\b(?:voters?|polls?|polling|surveys?|respondents?|'
                r'rank(?:ed|ing|s)?|election issue|most important issue)\b', title, re.I):
            continue
        # A Fed chairperson departure/nomination is not an interest-rate decision.
        if name == 'FED_RATES' and not re.search(
                r'\b(?:rates?|interest|monetary policy|basis points?|bps)\b', title, re.I):
            continue
        found.append((name, category, channel))
    return found


def _market_open(m: dict, venue: str) -> bool:
    if venue=='kalshi':return m.get('status') in ('open','active')
    return m.get('closed') is False and m.get('active') is True


def _market_id(m: dict, venue: str) -> str:
    return str(m.get('ticker') or m.get('id') or '')


def _market_url(m: dict, e: dict, venue: str) -> str:
    if venue=='kalshi':
        t=_market_id(m,venue)
        return 'https://kalshi.com/markets/' + re.sub('[^A-Za-z0-9-]','',t.lower()) if t else ''
    slug=re.sub('[^a-zA-Z0-9-]','',_str(e.get('slug'),160))
    return 'https://polymarket.com/event/' + slug if slug else ''


def _normalize_market(e: dict, m: dict, venue: str, rec: dict, as_of: date, horizon: date) -> dict | None:
    if not isinstance(m,dict) or not _market_open(m,venue): return None
    ids=_market_id(m,venue)
    if not ids: return None
    rules=_str(m.get('rules_primary') if venue=='kalshi' else (m.get('description') or e.get('description')))
    deadline,basis=_temporal_definition(rules)
    if deadline is None: return None   # Never claim resolution date from exchange close metadata
    d=date.fromisoformat(deadline)
    if not as_of<=d<=horizon:return None
    if not rules.strip():return None
    families=_classify(e,m,venue)
    if not families:return None
    return {'venue':venue,'event_id':str(e.get('event_ticker') if venue=='kalshi' else e.get('id')),
            'market_id':ids,'event_title':_str(e.get('title'),300),
            'market_title':_str(m.get('title') or m.get('question'),360),
            'candidate_deadline':deadline,'deadline_basis':basis,
            'contract_close_time':_str(m.get('close_time') if venue=='kalshi' else m.get('endDate'),100),
            'expected_expiration_time':_str(m.get('expected_expiration_time') if venue=='kalshi' else m.get('endDate'),100),
            'market_url':_market_url(m,e,venue),'raw_page_sha256':rec['sha256'],
            'raw_file':rec['raw_file'],'retrieved_at_utc':rec['completed_at_utc'],
            'source_url':rec['url'],'provenance_kind':rec.get('provenance_kind','P1_HASH_VALIDATED_HTTP_RECEIPT'),
            'rules_excerpt':rules[:800],'full_rules_digest_sha256':sha256(rules.encode()).hexdigest(),
            'candidate_families':families,
            'yes_bid_observed':m.get('yes_bid_dollars') if venue=='kalshi' else m.get('bestBid'),
            'yes_ask_observed':m.get('yes_ask_dollars') if venue=='kalshi' else m.get('bestAsk'),
            'quote_label':'UNNORMALIZED_PROVIDER_QUOTE_NOT_PHYSICAL_PROBABILITY_NOT_PAYOFF_CERTIFIED',
            'source_currency': 'USD' if venue=='kalshi' else 'unspecified',
            'market_liquidity_hint':m.get('open_interest_fp') if venue=='kalshi' else m.get('liquidityNum'),
            'outcome_status':'FUTURE_WINDOW_FROM_SETTLEMENT_TEXT_NOT_FUTURE_OUTCOME_PROOF',
            'prospective_caveat':'CONTRACT_MAY_COVER_EVENTS_BEFORE_CAPTURE_DATE; NO_RETROACTIVE_PHYSICAL_PROBABILITY'}


def _validate_records(records: list[dict]) -> None:
    if len(records)>MAX_CAPTURE_REQUESTS:raise DiscoveryError('Source ledger exceeds request budget')
    for record in records:
        provider=record.get('provider')
        if provider not in {'kalshi','polymarket','official'}:raise DiscoveryError('Unknown source provider in ledger')
        u=urlsplit(str(record.get('url','')))
        allowed={'kalshi':{'external-api.kalshi.com'},'polymarket':{'gamma-api.polymarket.com'},
                 'official':{'www.eia.gov','www.bis.gov','ustr.gov','www.ustr.gov','www.federalregister.gov'}}[provider]
        if u.scheme!='https' or u.hostname not in allowed or u.username or u.password:
            raise DiscoveryError('Source ledger includes untrusted URL or credentials')
        if record.get('status')!=200:raise DiscoveryError('P1 archive includes unsuccessful source in raw ledger')
        if not _safe_time(record.get('completed_at_utc')):raise DiscoveryError('Source receipt time is missing/invalid')
        raw_path=record.get('raw_file')
        if not isinstance(raw_path,str) or not re.fullmatch(r'raw/[0-9]{3}_[a-z0-9_\-]+\.(?:json|bin)',raw_path):
            raise DiscoveryError('Source raw-file path invalid')
        if not isinstance(record.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',record['sha256']):
            raise DiscoveryError('Source SHA-256 invalid')


def read_p1_capture(content: bytes, *, max_requests: int=MAX_CAPTURE_REQUESTS) -> dict:
    """Verify user-supplied P1 raw zip byte-for-byte. No network, no provider-signature claim."""
    if not isinstance(content,bytes) or len(content)>MAX_ARCHIVE_BYTES or len(content)==0:
        raise DiscoveryError('P1 archive missing or larger than the 12 MB local import limit')
    try:
        with ZipFile(BytesIO(content)) as z:
            infos=z.infolist()
            if len(infos)>max_requests+12: raise DiscoveryError('Too many ZIP entries')
            names=[i.filename for i in infos]
            if len(names)!=len(set(names)):raise DiscoveryError('Duplicate ZIP entry')
            if any((name.startswith('/') or '\\' in name or
                    any(part in ('.','..','') for part in name.split('/')) or
                    not (name in {'CAPTURE_SUMMARY.json','request_ledger.jsonl',
                                  'PROVISIONAL_REVIEW_QUEUE_NOT_P1_RESULTS.csv'} or
                         re.fullmatch(r'raw/[0-9]{3}_[a-z0-9_\-]+\.(?:json|bin)',name))) for name in names):
                raise DiscoveryError('ZIP entry path is not an allowed captured evidence path')
            if any(i.is_dir() or i.flag_bits & 0x1 or (i.external_attr >> 16)&0o170000==0o120000 for i in infos):
                raise DiscoveryError('Unsupported ZIP directory, link or encryption')
            if any(i.file_size>MAX_RAW_BYTES or i.file_size<0 for i in infos):
                raise DiscoveryError('Oversize raw page')
            if sum(i.file_size for i in infos)>MAX_EXPANDED_BYTES:raise DiscoveryError('ZIP expanded-size limit exceeded')
            if not {'request_ledger.jsonl','CAPTURE_SUMMARY.json'}.issubset(names):
                raise DiscoveryError('Required P1 source ledger missing')
            ledger_bytes=z.read('request_ledger.jsonl')
            if len(ledger_bytes)>150_000:raise DiscoveryError('P1 source ledger too large')
            try:
                records=[json.loads(line) for line in ledger_bytes.decode('utf-8').splitlines() if line.strip()]
            except (UnicodeError,ValueError,TypeError):raise DiscoveryError('Invalid P1 source ledger') from None
            _validate_records(records)
            if len(records)>max_requests:raise DiscoveryError('P1 capture request cap exceeded')
            out=[]
            for rec in records:
                b=z.read(rec['raw_file'])
                if len(b)!=rec.get('bytes') or sha256(b).hexdigest()!=rec['sha256']:
                    raise DiscoveryError('P1 raw receipt SHA-256/byte count mismatch')
                if rec['provider'] in {'kalshi','polymarket'}:
                    try:data=json.loads(b)
                    except (UnicodeError,ValueError):raise DiscoveryError('Invalid JSON in P1 captured page') from None
                    if not isinstance(data,dict) or not isinstance(data.get('events'),list):
                        raise DiscoveryError('P1 response is not an events catalog')
                    if len(data['events'])!=rec.get('record_count'):
                        raise DiscoveryError('P1 page count does not match source ledger')
                else:data=b
                out.append((rec,data))
            # The metadata is a receipt/hash consistency check, not provider authentication.
            return {'pages':out,'archive_sha256':sha256(content).hexdigest(),
                    'source_count':len(records), 'source_kind':'CAPTURED_HTTP_RECEIPTS_NOT_PROVIDER_SIGNED',
                    'retrieved_first_utc':min(r['completed_at_utc'] for r in records),
                    'retrieved_last_utc':max(r['completed_at_utc'] for r in records)}
    except DiscoveryError:raise
    except (BadZipFile,KeyError,OSError,RuntimeError,ValueError,EOFError):
        raise DiscoveryError('Cannot parse or trust P1 evidence ZIP') from None


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.parts=[]
    def handle_data(self,d):self.parts.append(d)


def official_eia_candidates(source_pages: list[tuple[dict,Any]], start:date,end:date) -> list[dict]:
    """Extract the actual EIA STEO schedule from archived official bytes, without forecasting its content."""
    out=[];seen=set()
    for rec,payload in source_pages:
        if rec['provider']!='official' or urlsplit(rec['url']).hostname!='www.eia.gov':continue
        text=payload.decode('utf-8','replace') if isinstance(payload,bytes) else ''
        parser=_Text()
        try:parser.feed(text)
        except Exception:continue
        normalized=' '.join(' '.join(parser.parts).split())
        if 'Short-Term Energy Outlook' not in normalized:continue
        # Scheduled releases are deterministic calendars; the report surprise
        # itself is not defined or probabilistically priced by these dates.
        for mo,da,yr in re.findall(r'\b(\d{2})/(\d{2})/(20\d\d)\b',normalized):
            try:d=date(int(yr),int(mo),int(da))
            except ValueError:continue
            if not start<=d<=end or d in seen:continue
            seen.add(d)
            out.append({'event_id':f'EIA-STEO-{d.isoformat()}','event_title':'EIA Short-Term Energy Outlook scheduled publication',
                        'candidate_deadline':d.isoformat(),'venue':'official',
                        'source_url':rec['url'],'retrieved_at_utc':rec['completed_at_utc'],
                        'raw_page_sha256':rec['sha256'],'raw_file':rec['raw_file'],
                        'family':'EIA_STEO_RELEASE','risk_channel':'ENERGY_SUPPLY',
                        'event_kind':'SCHEDULED_PUBLICATION_NOT_BINARY_UNCERTAIN_OUTCOME',
                        'matching_status':'NO_MATCH_ESTABLISHED_IN_BOUNDED_CAPTURE_NOT_PROVEN_ABSENT',
                        'confidence_label':'OFFICIAL_PAGE_SCHEDULE_NOT_EVENT_IMPACT',
                        'interpretation':'Date of EIA report release, not an outcome of its forecast or a verified shock to XLE.'})
    return sorted(out,key=lambda x:x['candidate_deadline'])



def official_trade_candidates(source_pages: list[tuple[dict,Any]], start:date,end:date) -> list[dict]:
    """USTR's explicitly dated exclusions are a policy deadline, not a predicted new tariff."""
    output=[]
    for rec,payload in source_pages:
        if rec['provider']!='official' or urlsplit(rec['url']).hostname not in {'ustr.gov','www.ustr.gov'}:continue
        parser=_Text()
        try:parser.feed(payload.decode('utf-8','replace'))
        except Exception:continue
        text=' '.join(' '.join(parser.parts).split())
        # Match both the named subject and the stated expiry in the same section.
        match=re.search(r'\b178 exclusions\b.{0,200}\bextended until\s+November\s+10,\s+2026\b',text,re.I)
        target=date(2026,11,10)
        if not match or not start<=target<=end:continue
        output.append({'event_id':'USTR-SECTION301-EXCLUSIONS-2026-11-10',
            'event_title':'USTR Section 301 China tariff-exclusion extension expires',
            'candidate_deadline':target.isoformat(),'venue':'official',
            'source_url':rec['url'],'retrieved_at_utc':rec['completed_at_utc'],
            'raw_page_sha256':rec['sha256'],'raw_file':rec['raw_file'],
            'family':'USTR_SECTION301_EXCLUSIONS','risk_channel':'TRADE',
            'event_kind':'POLICY_EXPIRY_DATE_NOT_CERTIFIED_NEW_TARIFF_EVENT',
            'matching_status':'NO_MATCH_ESTABLISHED_IN_BOUNDED_CAPTURE_NOT_PROVEN_ABSENT',
            'confidence_label':'PRIMARY_OFFICIAL_POLICY_TEXT_WITH_CONDITIONAL_FUTURE_ACTION',
            'source_excerpt':match.group(0),
            'interpretation':'Expiration of exemptions is documented; whether government extends or replaces them is unresolved, and exposure of individual fund holdings is not certified.'})
    return output

def _instrument_mapping_coverage(portfolio_rows: list[dict], holdings: Mapping[str, Any] | None,
                                 captured_at: datetime) -> dict:
    """Account for *every* unit of portfolio NAV not supported by the declared
    instrument/channel table. A mapped ETF-parent proxy can be used without
    identifying its constituents; the independent unresolved-NAV measure remains.
    No missing instrument is assigned zero economic risk.
    """
    ledger = recover_exposures(parse_portfolio(portfolio_rows), holdings or {}, now=captured_at)
    mapped = []
    unmapped = []
    breakdown: dict[str, list[float]] = defaultdict(list)
    for item in ledger.entries:
        parent_mapped = item.original_symbol in PARENT_CHANNELS
        component_mapped = item.status != 'unresolved_etf' and item.symbol in PARENT_CHANNELS
        # Parent mappings cover parent-level risk proxies, including unresolved
        # holdings. Unknown ETFs do not acquire coverage merely by being ETFs.
        supported = parent_mapped or component_mapped
        (mapped if supported else unmapped).append(item.portfolio_weight)
        if not supported:
            breakdown[item.original_symbol].append(item.portfolio_weight)
    mapped_weight = round(math.fsum(mapped), 12)
    unmapped_weight = round(math.fsum(unmapped), 12)
    if not math.isclose(mapped_weight + unmapped_weight, 1.0, abs_tol=1e-9):
        raise DiscoveryError('Instrument mapping coverage does not reconcile to portfolio NAV')
    status = ('UNSUPPORTED_INSTRUMENTS_NO_DISCOVERY_EVALUATION' if mapped_weight <= 1e-12
              else 'PARTIAL_INSTRUMENT_MAPPING_RANKINGS' if unmapped_weight > 1e-12
              else 'DECLARED_INSTRUMENT_MAPPINGS_AVAILABLE_NOT_FULL_ECONOMIC_COVERAGE')
    return {'status': status,
            'mapped_portfolio_weight': mapped_weight,
            'unmapped_portfolio_weight': unmapped_weight,
            'unmapped_instruments': [{'symbol': symbol, 'weight': round(math.fsum(weights), 12)}
                                      for symbol, weights in sorted(breakdown.items())],
            'unresolved_etf_identity_weight': round(ledger.unresolved_weight, 12),
            'method': 'EXPLICIT_PARENT_OR_VERIFIED_COMPONENT_MAPPING_ONLY_NO_FABRICATED_BETAS',
            'limitation': 'Unmapped positions are not zero-risk; mapped proxies do not cover every possible event.'}


def _rank_exposure(family:str,channel:str,portfolio_rows:list[dict],snapshots:Mapping[str,Any] | None,
                   captured_at:datetime) -> tuple[float,list[dict],dict]:
    p=parse_portfolio(portfolio_rows)
    ledger=recover_exposures(p,snapshots or {},now=captured_at)
    by_parent={p.symbol:p for p in p.positions}
    reasons=[];contrib=[]
    # Use every economic position at most once. Where authentic look-through
    # exists, use reported components, not *also* the parent ETF allocation.
    for exposure in ledger.entries:
        if exposure.status=='unresolved_etf':
            parent=by_parent.get(exposure.original_symbol)
            if parent and exposure.original_symbol in PARENT_CHANNELS and channel in PARENT_CHANNELS[exposure.original_symbol]:
                # A fund with *no* verified holdings can still be assessed as
                # an ETF-parent risk proxy, but NEVER as an identified constituent.
                strength,reason=PARENT_CHANNELS[exposure.original_symbol][channel]
                coeff=strength*exposure.portfolio_weight
                reasons.append({'instrument':exposure.original_symbol,'status':'ETF_PARENT_PROXY_UNRESOLVED_COMPONENTS',
                                'underlying_weight':exposure.portfolio_weight, 'channel_weight':strength,
                                'contribution':coeff,'reason':reason,
                                'issuer_channel_source':ISSUER_CHANNEL_SOURCES.get((exposure.original_symbol,channel))})
                contrib.append(coeff)
            continue
        symbol=exposure.symbol
        parent=(by_parent.get(exposure.original_symbol))
        values=PARENT_CHANNELS.get(symbol,{})
        # Certain economic channels are already supplied by direct ownership.
        if channel in values:
            strength,reason=values[channel]
            kind='DIRECT_POSITION' if exposure.status=='direct' else 'REPORTED_ETF_HOLDING'
            c=exposure.portfolio_weight*strength
            reasons.append({'instrument':symbol,'via':exposure.original_symbol,'status':kind,
                            'underlying_weight':exposure.portfolio_weight,'channel_weight':strength,
                            'contribution':c,'reason':reason,
                            'issuer_channel_source':ISSUER_CHANNEL_SOURCES.get((symbol,channel))})
            contrib.append(c)
        elif exposure.status=='look_through' and parent and channel in PARENT_CHANNELS.get(parent.symbol,{}):
            # Parent risks not solely explained by identified component ticker
            # stay in the parent residual channel; mark as model assumption.
            strength,reason=PARENT_CHANNELS[parent.symbol][channel]
            c=exposure.portfolio_weight*strength
            reasons.append({'instrument':symbol,'via':parent.symbol,'status':'ETF_COMPONENT_IN_PARENT_SECTOR_PROXY',
                            'underlying_weight':exposure.portfolio_weight,'channel_weight':strength,
                            'contribution':c,'reason':reason+'; constituent event beta unknown',
                            'issuer_channel_source':ISSUER_CHANNEL_SOURCES.get((parent.symbol,channel))})
            contrib.append(c)
    raw=sum(contrib)
    return round(raw,8),reasons,{'unresolved_weight':ledger.unresolved_weight,
                                'lookthrough': bool(snapshots),
                                'economic_basis':'EXPLICIT_HEURISTIC_CHANNEL_NOT_ESTIMATED_EVENT_RETURN',
                                'mass_decomposition':'EACH_DIRECT_OR_ETF_COMPONENT_ONCE_NO_ADDITIVE_PARENT_PNL'}


def discover(pages: list[tuple[dict,Any]], *, as_of:str,horizon_days:int,portfolio_rows:list[dict],
             holdings:Mapping[str,Any]|None=None, archive_sha256:str='',
             max_candidates:int=5) -> dict:
    """Find and rank sourced candidates; never admit logical contract relations."""
    if isinstance(horizon_days,bool) or type(horizon_days)!=int or not 1<=horizon_days<=366:
        raise DiscoveryError('Horizon must be an integer in [1,366]')
    start=date.fromisoformat(as_of);end=start+timedelta(days=horizon_days)
    if type(max_candidates)!=int or not 1<=max_candidates<=20:raise DiscoveryError('Invalid ranking cap')
    market_rows=[];counts=Counter();pagecount=Counter();page_end={}
    for rec,data in pages:
        provider=rec['provider'];pagecount[provider]+=1
        if provider not in {'kalshi','polymarket'}:continue
        if not isinstance(data,dict) or not isinstance(data.get('events'),list):continue
        for e in data['events']:
            if not isinstance(e,dict):continue
            counts[f'{provider}_event_records']+=1
            ms=e.get('markets',[])
            if not isinstance(ms,list):continue
            for m in ms:
                counts[f'{provider}_market_records']+=1
                row=_normalize_market(e,m,provider,rec,start,end)
                if row:market_rows.append(row)
        page_end[provider]=str(rec.get('next_page') or data.get('next_cursor') or data.get('cursor') or '')
    if sum(pagecount.values())==0:raise DiscoveryError('No source pages')
    cnt=Counter((m['venue'],m['event_id'],fam) for m in market_rows for fam,_,_ in m['candidate_families'])
    grouped={}
    captured_at=max((_safe_time(r['completed_at_utc']) for r,d in pages if r['provider']!='official'),default=None)
    if captured_at is None:raise DiscoveryError('No date-stamped market response')
    mapping_coverage = _instrument_mapping_coverage(portfolio_rows, holdings, captured_at)
    for m in market_rows:
        for fam,category,channel in m['candidate_families']:
            key=(m['venue'],m['event_id'],fam)
            score,reasons,ledger=_rank_exposure(fam,channel,portfolio_rows,holdings,captured_at)
            if score<=0:continue
            weighted=round(score,8)
            proposed={k:v for k,v in m.items() if k!='candidate_families'}
            proposed.update(family=fam,category=category,channel=channel,portfolio_relevance_score=weighted,
                            relevance_evidence=reasons,exposure_coverage=ledger,
                            ranking_coverage_status=mapping_coverage['status'],
                            unmapped_portfolio_weight=mapping_coverage['unmapped_portfolio_weight'],
                            event_market_count=cnt[key],
                            semantic_matching='NOT_ADMITTED_DISCOVERY_ONLY',
                            probability_status='MARKET_QUOTE_NOT_PHYSICAL_PROBABILITY',
                            conditional_pnl_status='NOT_ESTIMATED_OR_INFERRED',
                            event_source_status=('ARCHIVED_PUBLIC_CATALOG_EVENT_CANDIDATE_NOT_INDEPENDENTLY_RESOLVED'
                                                 if archive_sha256 else 'LIVE_PUBLIC_CATALOG_EVENT_CANDIDATE_NOT_INDEPENDENTLY_RESOLVED'))
            current=grouped.get(key)
            if current is None or (proposed['candidate_deadline'],proposed['market_id'])<(current['candidate_deadline'],current['market_id']):
                grouped[key]=proposed
    all_candidates=sorted(grouped.values(),key=lambda x:(-x['portfolio_relevance_score'],x['candidate_deadline'],x['family'],x['event_id']))
    # The shortlist is family-diverse; avoid 3 strikes of one event or venue
    # drowning unrelated sectors. Keep max one economic family, not just one ID.
    distinct=[];seen_families=set();seen_events=set()
    for c in all_candidates:
        if c['family'] in seen_families or (c['venue'],c['event_id']) in seen_events:continue
        seen_families.add(c['family']);seen_events.add((c['venue'],c['event_id']));distinct.append(c)
        if len(distinct)>=max_candidates:break
    official=official_eia_candidates(pages,start,end)+official_trade_candidates(pages,start,end)
    # Source complement relevance remains explicit, never an inferred P&L.
    for row in official:
        sc,trace,_=_rank_exposure('',row['risk_channel'],portfolio_rows,holdings,captured_at)
        row['portfolio_relevance_score']=sc;row['relevance_evidence']=trace
    timeline={'as_of':as_of,'horizon_days':horizon_days,'last_event_date_inclusive':end.isoformat(),
              'evaluation_type':'DEVELOPMENT_REPLAY_P1_NOT_BLIND_HOLDOUT' if archive_sha256 else 'DATE_TARGETED_ACQUISITION_NOT_HISTORICAL_HOLDOUT'}
    return {'status': ('DISCOVERY_UNEVALUABLE_UNSUPPORTED_PORTFOLIO' if mapping_coverage['mapped_portfolio_weight'] <= 1e-12
                       else 'CANDIDATES_DISCOVERED_NO_AUTOMATIC_SEMANTIC_CERTIFICATION'),
            'timeline':timeline,'portfolio_rows':portfolio_rows,
            'portfolio_instrument_coverage':mapping_coverage,
            'source_kind':'P1_CAPTURED_PUBLIC_RESPONSES_HASH_CHECKED' if archive_sha256 else 'LIVE_BOUNDED_PUBLIC_READ_ONLY',
            'archive_sha256':archive_sha256,
            'coverage':{'pages_by_provider':dict(pagecount),'last_cursor_by_provider':page_end,
                        'catalog_completeness':'NOT_ESTABLISHED_UNBOUNDED_PROVIDER_UNIVERSE',
                        'unmatched_claim_scope':'ONLY_FETCHED_PAGES','raw_counts':dict(counts),
                        'candidate_market_records':len(market_rows),'relevant_group_count':len(grouped)},
            'top_candidates':distinct,'official_source_candidates':official,
            'all_relevant_candidates':all_candidates,
            'labels':{'probability':'UNAVAILABLE_AS_PHYSICAL_PROBABILITY',
                      'impact':'NOT_ESTIMATED',
                      'logic':'CANDIDATE_PAIR_REQUIRES_SEPARATE_DETERMINISTIC_VERIFIER',
                      'ranking':'DECLARED_ECONOMIC_CHANNEL_HEURISTIC_NOT_EXPECTED_LOSS',
                      'replay':'P1 WAS INSPECTED_DURING_DEVELOPMENT_NOT_OUT_OF_SAMPLE'}}


def analyze_p1_zip(content:bytes, *, as_of:str='2026-10-16',horizon_days:int=90,
                   portfolio_rows:list[dict],holdings:Mapping[str,Any]|None=None) -> dict:
    input_data=read_p1_capture(content)
    report=discover(input_data['pages'],as_of=as_of,horizon_days=horizon_days,
                    portfolio_rows=portfolio_rows,holdings=holdings,
                    archive_sha256=input_data['archive_sha256'])
    report['receipts']={'archive_sha256':input_data['archive_sha256'],
                        'source_count':input_data['source_count'],
                        'retrieved_first_utc':input_data['retrieved_first_utc'],
                        'retrieved_last_utc':input_data['retrieved_last_utc'],
                        'provenance_limit':'HTTP_RECEIPT_AND_LOCAL_SHA256_NOT_PROVIDER_SIGNATURE'}
    return report


def public_report(report:Mapping[str,Any]) -> dict:
    """Trim large archival data to a portable review packet; no raw vendor pages."""
    return {k:v for k,v in report.items() if k not in {'all_relevant_candidates'}}
