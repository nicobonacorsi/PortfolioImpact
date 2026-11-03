"""Bounded, independently sourced public Kalshi search for the live Claude tool loop.

This is a *discovery adapter*, not a semantic verifier. It replaces the former
unfiltered /series download for every user query. Only real Kalshi GET responses
can produce market records; inferred tickers are lookup candidates, never
reported as actual contracts unless they occur in provider responses.

No Claude tokens, API keys, trading endpoints or automatically triggered calls.
"""
from __future__ import annotations

from datetime import date
import re
from typing import Any
from urllib.parse import urlencode

from .adapters import DataAccessError, KALSHI_BASE, KalshiClient, Market, normalize_market

_MONTHS = {
    'january': 'JAN', 'february': 'FEB', 'march': 'MAR', 'april': 'APR',
    'may': 'MAY', 'june': 'JUN', 'july': 'JUL', 'august': 'AUG',
    'september': 'SEP', 'october': 'OCT', 'november': 'NOV', 'december': 'DEC',
}
_STOP = {'the', 'and', 'in', 'at', 'of', 'will', 'by', 'for', 'does', 'more', 'than',
         'us', 'united', 'states', 'a', 'on', '2026', '2027', '2028', '2029'}
_TOKEN_ALIASES = {'tariffs': 'tariff', 'duties': 'duty', 'yields': 'yield',
                  'chips': 'chip', 'semiconductors': 'semiconductor',
                  'hurricanes': 'hurricane', 'restrictions': 'restriction',
                  'cells': 'cell', 'modules': 'module', 'batteries': 'battery',
                  'mexican': 'mexico', 'canadian': 'canada', 'chinese': 'china',
                  'aluminium': 'aluminum'}
_COUNTRIES = {
    'mexico': r'\b(?:mexico|mexican)\b', 'canada': r'\b(?:canada|canadian)\b',
    'china': r'\b(?:china|chinese|prc)\b', 'india': r'\b(?:india|indian)\b',
    'japan': r'\b(?:japan|japanese)\b', 'korea': r'\b(?:korea|korean)\b',
    'eu': r'\b(?:european union|eu)\b', 'brazil': r'\b(?:brazil|brazilian)\b',
}
_PRODUCTS = {
    'steel': r'\bsteel\b', 'aluminum': r'\balumin(?:um|ium)\b',
    'solar': r'\b(?:solar|photovoltaic)\b', 'battery': r'\bbatter(?:y|ies)\b',
    'chip': r'\b(?:chips?|semiconductors?)\b',
    'machinery': r'\b(?:machinery|heavy equipment)\b',
    'vehicle': r'\b(?:autos?|automobiles?|vehicles?)\b',
    'drug': r'\b(?:drugs?|pharma(?:ceutical)?s?)\b', 'copper': r'\bcopper\b',
}
_TARIFF_WORDS = (r'\btariffs?\b|\b(?:import|countervailing)\s+dut(?:y|ies)\b|'
                 r'\banti[- ]dumping\b|\b(?:ad valorem|section\s+(?:232|301))\s+'
                 r'(?:[a-z]+\s+){0,3}dut(?:y|ies)\b')
_HURRICANE_WORDS = r'\bhurricanes?\b|\btropical\s+cyclones?\b'
# Retrieval geography, not settlement geography.  In particular a city/region
# alias helps choose a provider collection to inspect; it never certifies that
# a contract covers the whole state, its islands, or the target time window.
_US_STATE_NAMES = (
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
    'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
    'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
    'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio',
    'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina',
    'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
    'washington', 'west virginia', 'wisconsin', 'wyoming', 'puerto rico',
)
_HURRICANE_STATE_ALIASES = {
    'florida': r'\b(?:FL|Fla\.?)\b', 'texas': r'\bTX\b',
    'north carolina': r'\bNC\b', 'south carolina': r'\bSC\b',
    'new york': r'\bNY\b', 'new jersey': r'\bNJ\b',
}
_HURRICANE_CITY_ALIASES = {
    'florida': r'\b(?:Miami|Tampa|Orlando|Jacksonville|Florida Keys)\b',
    'texas': r'\b(?:Houston|Galveston)\b',
    'louisiana': r'\bNew Orleans\b',
    'georgia': r'\bSavannah\b',
    'north carolina': r'\b(?:Hatteras(?: Island)?|Wilmington)\b',
    'south carolina': r'\b(?:North Myrtle Beach|Myrtle Beach|Charleston)\b',
    'virginia': r'\bNorfolk\b',
    'new york': r'\b(?:NYC|New York City)\b',
}
_HURRICANE_REGIONS = {
    r'\b(?:northeast|north[- ]east)\b': {
        'connecticut', 'maine', 'massachusetts', 'new hampshire', 'rhode island',
        'vermont', 'new jersey', 'new york', 'pennsylvania',
    },
    r'\bgulf coast\b': {'florida', 'alabama', 'mississippi', 'louisiana', 'texas'},
}
_LANDFALL_RETRIEVAL_WORDS = r'\blandfalls?\b|\bhits?\b'
_HURRICANE_COUNT_WORDS = r'\bnumber\b|\bcounts?\b|\bhow many\b|\btotal\b'
_HURRICANE_WARNING_WORDS = r'\bwarnings?\b|\bwatches?\b'


def _hurricane_named_states(text: str) -> set[str]:
    result = {name for name in _US_STATE_NAMES
              if re.search(r'\b' + re.escape(name) + r'\b', text, re.I)}
    # Remove a shorter state's incidental name from its longer counterpart.
    # "West Virginia" alone is not a Virginia collection.
    if 'west virginia' in result and not re.search(r'(?<!west )\bvirginia\b', text, re.I):
        result.discard('virginia')
    for name, pattern in _HURRICANE_STATE_ALIASES.items():
        if re.search(pattern, text, re.I):
            result.add(name)
    return result


def _hurricane_places(text: str) -> set[str]:
    """Conservative vocabulary for candidate ranking, never a source proof."""
    result = _hurricane_named_states(text)
    for name, pattern in _HURRICANE_CITY_ALIASES.items():
        if re.search(pattern, text, re.I):
            result.add(name)
    for pattern, members in _HURRICANE_REGIONS.items():
        if re.search(pattern, text, re.I):
            result.update(members)
    return result


def _hurricane_intensity_hint(text: str) -> int | None:
    """Only an intensity retrieval hint; '+' and equality are not conflated as predicates."""
    explicit = set(re.findall(r'\bcat(?:egory)?\.?\s*([1-5])\b', text, re.I))
    if len(explicit) == 1:
        return int(next(iter(explicit)))
    if len(explicit) > 1:
        return None
    return 3 if re.search(r'\bmajor\s+(?:hurricanes?|tropical cyclones?)\b', text, re.I) else None


def _hurricane_retrieval_priority(query: str, title: str) -> tuple[int, int, int, int]:
    """Separate a geographic landfall search from global formation counts.

    Source metadata does not establish a matching active contract.  Unknown
    scope remains eligible, and a different intensity remains inspectable but
    ranks after non-contradictory collections.  No ticker, price, spread or
    model result participates in these priorities.
    """
    if not re.search(_HURRICANE_WORDS, query, re.I):
        return (0, 0, 0, 0)
    query_places, source_places = _hurricane_places(query), _hurricane_places(title)
    query_intensity = _hurricane_intensity_hint(query)
    source_intensity = _hurricane_intensity_hint(title)
    intensity_compatible = int(query_intensity is None or source_intensity is None
                               or query_intensity == source_intensity)
    wants_count = bool(re.search(_HURRICANE_COUNT_WORDS, query, re.I))
    wants_landfall = bool(re.search(_LANDFALL_RETRIEVAL_WORDS, query, re.I))
    wants_warning = bool(re.search(_HURRICANE_WARNING_WORDS, query, re.I))
    source_landfall = bool(re.search(_LANDFALL_RETRIEVAL_WORDS, title, re.I))
    source_count = bool(re.search(_HURRICANE_COUNT_WORDS, title, re.I)) and not source_landfall
    source_warning = bool(re.search(_HURRICANE_WARNING_WORDS, title, re.I))
    # A place-qualified hazard with no explicit count asks for a regional
    # occurrence. A generic count collection is still retained as a lower-
    # priority, unknown-scope candidate, not declared globally irrelevant.
    if wants_warning and not wants_landfall:
        measurement_priority = 2 if source_warning else (0 if source_landfall or source_count else 1)
    elif wants_landfall or (query_places and not wants_count):
        measurement_priority = 2 if source_landfall else (0 if source_count else 1)
    elif wants_count:
        measurement_priority = 2 if source_count else (0 if source_landfall else 1)
    else:
        measurement_priority = 0
    # A state-qualified question must not spend the second lookup on just one
    # city when a generic landfall collection could expose the requested state.
    # City/state containment remains retrieval vocabulary, not certified event
    # containment. The independent full-rule gate is intentionally unchanged.
    location_priority = 1
    if query_places:
        direct_overlap = _hurricane_named_states(query) & _hurricane_named_states(title)
        same_city = any({value.casefold() for value in re.findall(pattern, query, re.I)}
                        & {value.casefold() for value in re.findall(pattern, title, re.I)}
                        for pattern in _HURRICANE_CITY_ALIASES.values())
        same_region = any(re.search(pattern, query, re.I) and re.search(pattern, title, re.I)
                          for pattern in _HURRICANE_REGIONS)
        if direct_overlap or same_city or same_region:
            location_priority = 3
        elif not source_places:
            location_priority = 2
    intensity_specificity = int(query_intensity is not None and query_intensity == source_intensity)
    return intensity_compatible, measurement_priority, location_priority, intensity_specificity
# Named agreement anchors are retrieval vocabulary, not semantic equivalence.
# In particular NAFTA is not an alias for USMCA, and an extension is not a
# withdrawal. Full source rules and the existing independent gates still apply.
_AGREEMENT_SEPARATOR = r'[-\s\u2010-\u2015\u2212]+'
_TRADE_AGREEMENTS = {
    'usmca': (rf'\b(?:USMCA|CUSMA|T{_AGREEMENT_SEPARATOR}MEC)\b|'
              rf'\bUnited\s+States{_AGREEMENT_SEPARATOR}Mexico{_AGREEMENT_SEPARATOR}Canada\s+Agreement\b'),
    'nafta': rf'\bNAFTA\b|\bNorth\s+American\s+Free{_AGREEMENT_SEPARATOR}Trade\s+Agreement\b',
    'cptpp': (r'\bCPTPP\b|\bComprehensive\s+and\s+Progressive\s+Agreement\s+for\s+'
              rf'Trans{_AGREEMENT_SEPARATOR}Pacific\s+Partnership\b'),
    'rcep': r'\bRCEP\b|\bRegional\s+Comprehensive\s+Economic\s+Partnership\b',
}
_TRADE_AGREEMENT_WORDS = (
    rf'\b(?:trade\s+(?:agreements?|treaty|treaties|pacts?)|free{_AGREEMENT_SEPARATOR}trade(?:\s+agreements?)?)\b|'
    + '|'.join(_TRADE_AGREEMENTS.values())
)


def _named_entities(text: str, patterns: dict[str, str]) -> set[str]:
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def _query_topic_compatible(query: str, corpus: str, *, metadata: bool) -> bool:
    """Require the requested action/subject, not a shared incidental word.

    Metadata can name a broad sector/country collection whose market rows need
    inspection. Individual returned contracts do not get that relaxation.
    This selects source candidates only and never establishes equivalence.
    """
    q, source = query.lower(), corpus.lower()
    if not metadata:
        # Broad venue metadata often returns unrelated contracts because a
        # single shared word matched. The real 2026-11-01 evaluation retrieved
        # wage data for Microsoft's earnings, a Fed rate decision for Basel
        # capital rules and government Gold Card receipts for gold ETF flows.
        # Filter distinct subjects BEFORE spending review-model tokens.
        if re.search(r'\b(?:microsoft|msft)\b', q) and not re.search(r'\b(?:microsoft|msft)\b', source):
            return False
        if re.search(r'\b(?:cybersecurity|data breach|cyber incident)\b', q) and not re.search(
                r'\b(?:cybersecurity|cyberattack|cyber.attack|data breach|security incident|'
                r'material cyber|cyber incident|item 1\.05)\b', source):
            return False
        if re.search(r'\bazure\b', q) and not re.search(r'\bazure\b', source):
            return False
        if re.search(r'\b(?:jpmorgan|jpm|j\.p\.\s*morgan)\b', q) and not re.search(
                r'\b(?:jpmorgan|jpm|j\.p\.\s*morgan)\b', source):
            return False
        if re.search(r'\b(?:basel|bank capital|capital requirements?|risk.based capital)\b', q) and not re.search(
                r'\b(?:basel|bank capital|capital requirements?|risk.based capital|banking capital rules?)\b', source):
            return False
        if re.search(r'\bquarterly refunding\b|\bauction sizes?\b', q) and not re.search(
                r'\b(?:refunding|auction sizes?|coupon auction)\b', source):
            return False
        if re.search(r'\b(?:gold etf|gold holdings|gold outflows|spdr gold)\b', q):
            if re.search(r'\bgold card\b', source) or not re.search(
                    r'\b(?:gold etf|gold holdings?|gold bullion|spdr|precious metal|xau|gld)\b', source):
                return False
        if re.search(r'\b(?:u\.?s\.?|united states)\b.*\b(?:credit rating|rating downgrade)\b', q):
            if not re.search(r'\b(?:u\.?s\.?|united states|american sovereign)\b', source):
                return False
    if re.search(_HURRICANE_WORDS, query, re.I) and not re.search(_HURRICANE_WORDS, corpus, re.I):
        return False  # Florida gasoline prices are not hurricane evidence.
    if re.search(_HURRICANE_WORDS, query, re.I):
        query_places, source_places = _hurricane_places(query), _hurricane_places(corpus)
        if query_places and source_places and not query_places.intersection(source_places):
            return False  # Explicitly disjoint weather locations are not the requested subject.
    if re.search(_TARIFF_WORDS, query, re.I):
        if not re.search(_TARIFF_WORDS, corpus, re.I):
            return False  # A solar tax credit is not an import-duty action.
        countries = _named_entities(query, _COUNTRIES)
        if countries and not countries.intersection(_named_entities(corpus, _COUNTRIES)):
            if not (metadata and re.search(r'\bcountries\b', corpus, re.I)):
                return False
        products = _named_entities(query, _PRODUCTS)
        if products and not products.intersection(_named_entities(corpus, _PRODUCTS)):
            if not (metadata and re.search(r'\bsectors?\b', corpus, re.I)):
                return False
    elif re.search(_TRADE_AGREEMENT_WORDS, query, re.I):
        # Explicit tariff lookup retains its action/country/product checks
        # above even if the query also mentions a treaty as context. Requiring
        # that contextual name in every tariff title would discard genuine
        # source candidates before their complete rules can be reviewed.
        if not re.search(_TRADE_AGREEMENT_WORDS, corpus, re.I):
            return False  # Tax/TikTok extensions and generic decisions are not trade agreements.
        agreements = _named_entities(query, _TRADE_AGREEMENTS)
        if agreements and not agreements.issubset(_named_entities(corpus, _TRADE_AGREEMENTS)):
            return False  # A different or unnamed agreement cannot supply a named treaty's subject.
    return True


class CatalogSearchError(RuntimeError):
    """Only sanitized errors, never HTTP bodies or environment secrets."""


def _query_month_year(query: str, today: date) -> str | None:
    lower = query.lower()
    months = [abbr for word, abbr in _MONTHS.items() if re.search(r'\b' + word + r'\b', lower)]
    if len(months) != 1:
        return None
    years = re.findall(r'\b20\d\d\b', lower)
    if len(set(years)) > 1:
        return None
    year = int(years[0]) if years else today.year
    if year < today.year - 1 or year > today.year + 2:
        return None
    return f'{year % 100:02d}{months[0]}'


def _family(query: str) -> str | None:
    """Route an explicitly US-Fed or CPI question to its known venue family.

    A3 omitted common word orders such as "Fed December 2026 rate decision",
    falling back to a generic global economics series and retrieving Chile.
    Generic "rate cut" WITHOUT Fed context is not assumed to be US FOMC.
    """
    q = query.lower()
    # The institution alone is NOT an interest-rate predicate. A Fed capital
    # regulation proposal must not be sent to KXFEDDECISION by lexical overlap.
    is_bank_regulation = bool(re.search(
        r'\b(?:basel|capital requirements?|stress capital|bank supervision|capital rules?|'
        r'banking regulation|gsib surcharge)\b', q))
    rate_or_fomc = bool(re.search(
        r'\bfomc\b|\bfederal funds\b|\b(?:interest|policy|target)\s+rates?\b|'
        r'\brate\s+(?:decision|cut|hike|meeting|hold)\b|'
        r'\b(?:cut|hike|raise|lower|hold)\s+(?:interest\s+)?rates?\b|'
        r'\bmonetary policy meeting\b', q))
    fed_institution = bool(re.search(r'\bfomc\b|\bfederal reserve\b|\bfederal funds\b|\bfed\b', q))
    if fed_institution and rate_or_fomc and not is_bank_regulation:
        return 'KXFEDDECISION'
    if re.search(r'\bcpi\b', q) or 'consumer price index' in q:
        return 'KXCPI'
    return None


UNION_POLICY_CATEGORIES = ('Politics', 'Economics')


def _category(query: str) -> str:
    q = query.lower()
    # This exact category label is present in the retained provider catalog.
    # Routing changes no request cap and does not claim any hurricane exists.
    if re.search(_HURRICANE_WORDS, q):
        return 'Climate and Weather'
    if re.search(_TARIFF_WORDS, q):
        return 'Politics'
    if re.search(_TRADE_AGREEMENT_WORDS, q, re.I):
        return 'Politics'
    if any(term in q for term in ('tariff', 'export control', 'china', 'iran', 'war', 'nvidia', 'taiwan')):
        return 'Politics'
    return 'Economics'


def _safe_tokens(query: str) -> set[str]:
    tokens = {_TOKEN_ALIASES.get(token, token)
              for token in re.findall(r'[a-z0-9]+', query.lower()) if token not in _STOP}
    # Preserve a common retrieval anchor for source-written names such as
    # USMCA/CUSMA/T-MEC; this does not infer any source settlement action.
    return tokens | _named_entities(query, _TRADE_AGREEMENTS)


def _rank_series(rows: list, query: str) -> list[str]:
    """Rank only source-returned series using explicit subject/action overlap."""
    tokens = _safe_tokens(query)
    q = query.lower()
    chip_intent = bool(re.search(r'\bnvidia\b|\bnvda\b|\bchips?\b|\bsemiconductors?\b|export control', q))
    yield_intent = bool(re.search(r'\btreasury\b|10[- ]year|\byield\b', q))
    recession_intent = 'recession' in q
    rankings = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get('ticker', ''))
        if not re.fullmatch(r'KX[A-Z0-9]+', ticker):
            continue
        title = str(row.get('title', ''))
        raw_tags = row.get('tags')
        tags = ' '.join(item for item in raw_tags if isinstance(item, str)) if isinstance(raw_tags, list) else ''
        corpus = (title + ' ' + tags + ' ' + ticker).lower()
        if not _query_topic_compatible(query, corpus, metadata=True):
            continue
        if chip_intent and not re.search(r'\b(nvidia|nvda|chip|chips|semiconductor|semiconductors)\b|export control|chipchina|chipban', corpus):
            continue
        if yield_intent and not re.search(r'\b(treasury|yield|yields)\b|10[- ]year', corpus):
            continue
        if recession_intent and not re.search(r'\brecession\b|economic downturn', corpus):
            continue
        # Singular/plural normalization fixes "steel tariff" losing to an
        # aluminum contract merely because the exact title says "Steel tariffs".
        title_tokens = _safe_tokens(title)
        score = 2 * len(tokens & title_tokens) + len(tokens & _safe_tokens(tags))
        if score:
            # Equal overlap should prefer "Steel tariffs" to "50% tariffs on
            # Canadian autos and steel", not depend on ticker alphabet order.
            weather_priority = _hurricane_retrieval_priority(query, title)
            rankings.append((weather_priority, score, len(title_tokens - tokens), ticker))
    return [ticker for _, _, _, ticker in sorted(
        rankings, key=lambda item: (tuple(-part for part in item[0]), -item[1], item[2], item[3]))]


def _diversified_candidate_prefix(scored: list[tuple[int, Market]], limit: int) -> list[Market]:
    """Round-robin *retrieval* across settlement events, never semantic admission.

    A broad issue search can return hundreds of threshold bins in one event
    collection. The old global top-N can spend every Claude review slot on
    those near-duplicates even if another relevant settlement event was
    actually retrieved. Keep the relevance order *within* each source event,
    then expose one from each event before its second threshold. If all
    candidates are from one event this changes nothing. Explicit source
    series/date lookups retain their original exhaustive ranking behavior.
    """
    grouped: dict[str, list[Market]] = {}
    for _, market in sorted(scored, key=lambda item: (-item[0], item[1].ticker)):
        # The venue-supplied event ticker defines settlement families; a
        # malformed/missing event identity is never fabricated from titles.
        key = market.event_ticker or f"MISSING:{market.ticker}"
        grouped.setdefault(key, []).append(market)
    selected: list[Market] = []
    round_index = 0
    while len(selected) < limit:
        another_round = False
        for bucket in grouped.values():
            if round_index < len(bucket):
                another_round = True
                selected.append(bucket[round_index])
                if len(selected) == limit:
                    break
        if not another_round:
            break
        round_index += 1
    return selected


class BoundedKalshiCatalog:
    """Safe public read-only queries, with opt-in expanded R59 pagination.

    The preflight is a separate explicit action. Search snapshots and per-query
    failure reasons are retained in a user-inspectable, secret-free ledger.
    """

    def __init__(self, client: KalshiClient, *, as_of: date | None = None,
                 expanded: bool = False,
                 diversify_generic_candidate_events: bool = True,
                 max_series_per_query: int | None = None):
        self.client = client
        self.as_of = as_of or date.today()
        self.expanded = expanded
        # R125: the per-event market search caps how many ranked series one
        # generic query may page through, so eight events cannot exhaust the
        # public read budget. None keeps the original R59 behaviour.
        if max_series_per_query is not None and (type(max_series_per_query) is not int
                                                 or not 1 <= max_series_per_query <= 8):
            raise ValueError('max_series_per_query must be between 1 and 8')
        self.max_series_per_query = max_series_per_query
        self.diversify_generic_candidate_events = diversify_generic_candidate_events
        self.search_ledger: list[dict[str, Any]] = []
        # Lifetime is one assessment. Reuse does not change receipt timestamps,
        # replay old-run data, cache market quotes, or enlarge HTTP attempt caps.
        self._series_cache: dict[str, tuple[list, dict]] = {}
        self._series_diagnostics: dict[str, Any] = {}
        # R59 reuses immutable source snapshots only *within the same run*.
        # Every Market retains its original receipt timestamp and raw path;
        # no response is re-stamped or represented as a fresh quote.
        self._market_cache: dict[tuple[str, str, int], tuple[list[Market], bool]] = {}

    def preflight(self) -> dict[str, Any]:
        """One free GET verifies that the *market* endpoint serves actual rules.

        The ticker is a 2026-10-16 source-observed Kalshi event. A present market
        still does not establish future availability, a probability, or the
        quality of arbitrary natural-language searches. No credentials are sent.
        """
        try:
            markets, unfinished = self._markets(event='KXCPI-26OCT', max_rows=2)
            if not markets or not any(m.rules_primary and m.ticker for m in markets):
                raise CatalogSearchError('Kalshi market endpoint returned no usable source rules for the observed reference event')
            # Provenance remains a local HTTP receipt, not an exchange signature.
            first = markets[0]
            return {'status': 'PUBLIC_CATALOG_READY', 'provider': 'Kalshi',
                    'reference_event_ticker': 'KXCPI-26OCT',
                    'retrieved_contracts': len(markets),
                    'retrieved_at': first.retrieved_at,
                    'coverage': 'ONE_PUBLIC_MARKET_GET_WITH_REAL_RULES_NOT_COMPLETE_MARKET_AVAILABILITY'}
        except DataAccessError as exc:
            raise CatalogSearchError(str(exc)) from None

    def _markets(self, *, event: str | None = None, series: str | None = None,
                 max_rows: int = 80) -> tuple[list[Market], bool]:
        # All tickers here are generated from a tightly restricted local grammar
        # or are exact provider-returned series names, never raw user URLs.
        ticker = event or series or ''
        if not re.fullmatch(r'KX[A-Z0-9]+(?:-[0-9]{2}[A-Z]{3})?', ticker):
            raise CatalogSearchError('Unsafe or unrecognized Kalshi ticker for bounded lookup')
        cache_key = ('event' if event else 'series', ticker, max_rows)
        if self.expanded and cache_key in self._market_cache:
            return self._market_cache[cache_key]
        page_size = min(250, max_rows) if self.expanded else max_rows
        max_pages = 4 if self.expanded else 1
        args: dict[str, str | int] = {'limit': page_size, 'status': 'open', 'mve_filter': 'exclude'}
        args['event_ticker' if event else 'series_ticker'] = ticker
        result: list[Market] = []
        cursors_seen: set[str] = set()
        for _ in range(max_pages):
            data, meta = self.client._get(KALSHI_BASE + '/markets?' + urlencode(args))
            rows = data.get('markets')
            if not isinstance(rows, list) or any(not isinstance(x, dict) for x in rows):
                raise CatalogSearchError('Public markets response has no verified market array')
            result.extend(normalize_market(row, meta['retrieved_at'], meta['raw_path']) for row in rows)
            cursor = data.get('cursor')
            if not cursor:
                answer = (result, False)
                if self.expanded:
                    self._market_cache[cache_key] = answer
                return answer
            if not isinstance(cursor, str) or len(cursor) > 2000 or cursor in cursors_seen:
                raise CatalogSearchError('Invalid or repeated Kalshi page cursor')
            cursors_seen.add(cursor)
            args['cursor'] = cursor
        answer = (result, True)  # unvisited cursor must never be reported as complete
        if self.expanded:
            self._market_cache[cache_key] = answer
        return answer

    def _category_series(self, query: str) -> list[str]:
        # Documented, case-sensitive Kalshi category filter. Unlike the previous
        # adapter this NEVER fetches the potentially oversized global /series.
        category = _category(query)
        # R94: keyword routing to a single category missed exact contracts filed under
        # the other policy category (FY2027 NDAA is a Kalshi "Politics" series but the
        # query routed to "Economics"). Expanded live runs rank the union of both policy
        # categories; each extra category is one cached GET per assessment. Admission
        # still goes through every downstream rules and review gate.
        categories = [category]
        if self.expanded:
            categories += [c for c in UNION_POLICY_CATEGORIES if c != category]
        reused = all(c in self._series_cache for c in categories)
        rows, meta = [], {}
        seen_tickers: set = set()
        for cat in categories:
            if cat in self._series_cache:
                cat_rows, cat_meta = self._series_cache[cat]
            else:
                data, cat_meta = self.client._get(KALSHI_BASE + '/series?' + urlencode({'category': cat}))
                cat_rows = data.get('series')
                if not isinstance(cat_rows, list):
                    raise CatalogSearchError('Filtered Kalshi series response has no series array')
                self._series_cache[cat] = (cat_rows, cat_meta)
            meta = meta or cat_meta
            for row in cat_rows:
                key = row.get('ticker') if isinstance(row, dict) else id(row)
                if key in seen_tickers:
                    continue
                seen_tickers.add(key)
                rows.append(row)
        ranked = _rank_series(rows, query)
        self._series_diagnostics = {
            'category': category, 'categories_ranked': categories, 'series_catalog_rows': len(rows),
            'series_metadata_same_run_reuse': reused,
            'series_metadata_retrieved_at': meta.get('retrieved_at'),
            'series_metadata_response_sha256': meta.get('sha256'),
            'relevant_series_count': len(ranked),
            'series_shortlist': ranked[:8] if self.expanded else ranked[:2],
            'series_scope': 'SOURCE_METADATA_CANDIDATES_NOT_CONFIRMED_EVENT_MATCHES',
        }
        return ranked[:8] if self.expanded else ranked[:2]

    def search(self, query: str, *, limit: int = 8) -> list[Market]:
        if not isinstance(query, str) or not (2 <= len(query) <= 120):
            raise CatalogSearchError('Search must be between 2 and 120 characters')
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 12):
            raise CatalogSearchError('Invalid bounded result limit')
        before = self.client.request_count
        search_strategy = 'UNKNOWN'
        cursor_remaining = False
        self._series_diagnostics = {}
        examined_market_rows = None
        query_intent_excluded = 0
        attempted_series = []
        market_page_size = 200 if self.expanded else 80
        try:
            ticker_query = query.strip().upper()
            if re.fullmatch(r'KX[A-Z0-9]+(?:-[0-9]{2}[A-Z]{3})?', ticker_query):
                search_strategy = 'EXPLICIT_EXCHANGE_TICKER'
                if '-' in ticker_query:
                    markets, cursor_remaining = self._markets(event=ticker_query, max_rows=market_page_size)
                else:
                    markets, cursor_remaining = self._markets(series=ticker_query, max_rows=market_page_size)
            else:
                family = _family(query)
                suffix = _query_month_year(query, self.as_of)
                if family and suffix:
                    search_strategy = 'SOURCE_SERIES_AND_DATE_SCOPED_EVENT_LOOKUP'
                    # Hypothesized event-ticker *lookup*, not provider evidence.
                    markets, cursor_remaining = self._markets(event=f'{family}-{suffix}', max_rows=market_page_size)
                    if not markets and (self.expanded or self.client.request_count - before < 2):
                        search_strategy += '_WITH_SERIES_FALLBACK'
                        markets, cursor_remaining = self._markets(series=family, max_rows=market_page_size)
                elif family:
                    search_strategy = 'KNOWN_SOURCE_SERIES_SCOPED_LOOKUP'
                    markets, cursor_remaining = self._markets(series=family, max_rows=market_page_size)
                else:
                    search_strategy = 'FILTERED_SERIES_METADATA_LOOKUP'
                    selected = self._category_series(query)
                    markets = []
                    # Legacy mode preserves at most two logical GET operations:
                    # fresh catalog + one market page; or two market pages when
                    # this run's already-received category metadata is reused.
                    slots = (8 if self.expanded else
                             (2 if self._series_diagnostics['series_metadata_same_run_reuse'] else 1))
                    if self.max_series_per_query is not None:
                        slots = min(slots, self.max_series_per_query)
                    examined_market_rows = 0
                    for selected_series in selected[:slots]:
                        attempted_series.append(selected_series)
                        fetched, remaining = self._markets(series=selected_series, max_rows=market_page_size)
                        cursor_remaining = cursor_remaining or remaining
                        examined_market_rows += len(fetched)
                        eligible = [market for market in fetched if _query_topic_compatible(query,
                            f'{market.title} {market.subtitle} {market.rules_primary}', metadata=False)]
                        query_intent_excluded += len(fetched) - len(eligible)
                        if self.expanded:
                            markets.extend(eligible)
                        else:
                            markets = eligible
                        if markets and not self.expanded:
                            break
                    if len(attempted_series) > 1:
                        search_strategy += '_WITH_SAME_BUDGET_EMPTY_SERIES_FALLBACK'
            # No inferred event is returned. The provider response must contain
            # each normalized ticker and original resolution rules.
            qtokens = _safe_tokens(query)
            family = _family(query)
            suffix = _query_month_year(query, self.as_of)
            if family and suffix:
                # Do not present another month's contract as a result for this
                # event when the exact-month query yielded no provider records.
                markets = [m for m in markets if m.event_ticker == f'{family}-{suffix}']
            scored = []
            for market in markets:
                if not market.ticker or not market.metadata.get('raw'):
                    continue
                corpus = f'{market.title} {market.subtitle} {market.rules_primary} {market.event_ticker}'.lower()
                score = len(qtokens & _safe_tokens(corpus))
                if family and market.series_ticker == family:
                    score += 4
                if suffix and suffix in market.event_ticker:
                    score += 5
                scored.append((score, market))
            # Diversify generic issue discovery without dropping any real
            # contract or changing the known-series payoff-ladder path.
            # This is retrieval recall only: every market still goes through
            # source, settlement-predicate and Claude review gates downstream.
            diversified = (self.expanded and self.diversify_generic_candidate_events
                           and search_strategy.startswith('FILTERED_SERIES_METADATA'))
            result = (_diversified_candidate_prefix(scored, limit) if diversified else
                      [m for _,m in sorted(scored, key=lambda p:(-p[0],p[1].ticker))[:limit]])
            self.search_ledger.append({'query': query, 'strategy': search_strategy,
                        'source_http_gets': self.client.request_count - before,
                        'provider_records': len(markets) if examined_market_rows is None else examined_market_rows,
                        'query_intent_excluded_records': query_intent_excluded,
                        'attempted_series': attempted_series, **self._series_diagnostics,
                        'returned_contracts': len(result),
                        'candidate_event_diversification': diversified,
                        'candidate_event_tickers_examined': len({m.event_ticker for _, m in scored}),
                        'candidate_event_tickers_returned': len({m.event_ticker for m in result}),
                        'cursor_remaining': cursor_remaining,
                        'coverage': ('EXPANDED_PAGINATED_CATEGORY_SEARCH_STILL_NOT_FULL_PROVIDER_UNIVERSE'
                                     if self.expanded else 'BOUNDED_SAMPLING_NOT_FULL_PROVIDER_UNIVERSE'),
                        'status': 'OK'})
            return result
        except (DataAccessError, CatalogSearchError, TypeError, ValueError, KeyError) as exc:
            safe_reason = (str(exc) if isinstance(exc, (DataAccessError, CatalogSearchError))
                           else 'Invalid data schema in retrieved Kalshi response')
            self.search_ledger.append({'query': query, 'strategy': search_strategy,
                     'source_http_gets': self.client.request_count - before,
                     'attempted_series': attempted_series, **self._series_diagnostics,
                     'returned_contracts': 0, 'status': 'FAILED',
                     'failure_phase': ('PROVIDER_REQUEST' if isinstance(exc, (DataAccessError, CatalogSearchError))
                                       else 'RESPONSE_SCHEMA_VALIDATION'),
                     'reason': safe_reason})
            raise CatalogSearchError(safe_reason) from None
