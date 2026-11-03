"""Independent source-identity and release-calendar gates for agent discovery.

This is a **conservative domain-specific retrieval filter**, not a semantic
implication verifier and not an all-events ontology. It can exclude a contract
from the Claude review queue but can NEVER certify an outcome relation.

The calendar entries below were manually reviewed against official Fed/BLS
calendar pages on 2026-10-18. They are *curated transcriptions*, not cached
provider pages, and must not be presented as independently downloaded bytes.
Unknown events abstain from calendar confirmation.
"""
from __future__ import annotations

from datetime import date
import re
from typing import Any, Mapping

from .predicate_grounding import qualify_rule_predicate
from .live_catalog import _COUNTRIES, _PRODUCTS, _HURRICANE_WORDS, _TARIFF_WORDS, _named_entities
from .event_subject_intent import (explicit_non_tariff_agreement_title,
                                  without_tariff_meta_exclusions)

_CALENDAR = {
    ('US_FOMC', '2026-10'): (
        '2026-10-28', 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
        'Federal Reserve FOMC meeting October 27–28, 2026 (second day)'),
    ('US_FOMC', '2026-12'): (
        '2026-12-09', 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
        'Federal Reserve FOMC meeting December 8–9, 2026 (second day)'),
    ('US_CPI', '2026-10'): (
        '2026-11-10', 'https://www.bls.gov/schedule/2026/11_sched_list.htm',
        'BLS release of October 2026 CPI, November 10 at 08:30 ET'),
    ('US_CPI', '2026-11'): (
        '2026-12-10', 'https://www.bls.gov/schedule/2026/12_sched_list.htm',
        'BLS release of November 2026 CPI, December 10 at 08:30 ET'),
}
_MONTHS = {m: i for i, m in enumerate(('january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'), 1)}
_COASTAL_LOCATIONS = {
    name: r'\b' + name.replace(' ', r'\s+') + r'\b'
    for name in ('florida', 'texas', 'louisiana', 'mississippi', 'alabama',
                 'georgia', 'south carolina', 'north carolina', 'virginia',
                 'maryland', 'delaware', 'new jersey', 'new york',
                 'massachusetts', 'connecticut', 'rhode island', 'maine',
                 'hawaii', 'puerto rico', 'virgin islands')
}

# This is a *candidate research* gate, never a probability/admission gate.
# Unknown real-world policies must not be dropped solely because we have not
# implemented their economic models. Equally, topical words such as "credit
# card" cannot transform a delinquency statistic into a bill-enactment quote.
_GENERIC_EVENT_WORDS = frozenset({
    'a', 'an', 'as', 'at', 'by', 'in', 'is', 'it', 'no', 'not', 'of', 'on', 'or',
    'per', 'to', 'us', 'yes', 'if', 'are', 'has', 'have', 'had', 'be', 'been',
    'the', 'and', 'for', 'from', 'with', 'without', 'into', 'over', 'after', 'before',
    'event', 'market', 'markets', 'result', 'outcome', 'price', 'prices', 'change',
    'changes', 'increase', 'increases', 'increased', 'decrease', 'decreases',
    'will', 'would', 'could', 'should', 'that', 'this', 'those', 'what', 'when',
    'during', 'there', 'about', 'below', 'above', 'under', 'more', 'than', 'less',
    'year', 'years', 'month', 'months', 'date', 'united', 'states', 'american',
    'government', 'federal', 'policy', 'policies', 'official', 'announcement',
    'report', 'reports', 'decides', 'decision', 'does', 'happen', 'likely',
})

# The title's *outcome action*, not just its subject, must occur in the
# verified settlement rules before an otherwise unreviewed family can consume
# an LLM matching slot. These are conservative research-only prerequisites,
# not interchangeable definitions. A source may still fail jurisdiction,
# negation, threshold, venue or deadline review after this check.
_OPEN_ACTIONS = (
    ('LEGAL_ENACTMENT',
     r'\b(?:enact(?:ed|ment)?|becomes?\s+law|signed\s+into\s+law|'
     r'\b(?:legislation|bill|act)\s+(?:pass(?:es|ed)?|signed))\b',
     r'\b(?:enact(?:ed|ment)?|becomes?\s+law|signed\s+into\s+law|'
     r'\b(?:legislation|bill|act)\s+(?:pass(?:es|ed)?|signed))\b'),
    ('TREATY_OR_CHANNEL_CLOSURE',
     r'\b(?:clos(?:e|ed|es|ure)|block(?:ed|ade)?|shut(?:down)?|sealed|halted)\b',
     r'\b(?:clos(?:e|ed|es|ure)|block(?:ed|ade)?|shut(?:down)?|sealed|halted)\b'),
    ('OIL_OUTPUT_OR_PRODUCTION',
     r'\b(?:opec|production|output|supply\s+quota|oil\s+quota)\b',
     r'\b(?:opec|production|output|supply\s+quota|oil\s+quota)\b'),
    ('LABOR_STOPPAGE',
     r'\b(?:strike|strikes|striking|walkout|work\s+stoppage|labor\s+stoppage)\b',
     r'\b(?:strike|strikes|striking|walkout|work\s+stoppage|labor\s+stoppage)\b'),
    ('COURT_OR_ANTITRUST_DECISION',
     r'\b(?:antitrust|court\s+(?:ruling|decision|order)|judg(?:e|ment)|'
     r'verdict|lawsuit|doj\s+(?:ruling|action|decision))\b',
     r'\b(?:antitrust|court\s+(?:ruling|decision|order)|judg(?:e|ment)|'
     r'verdict|lawsuit|doj\s+(?:ruling|action|decision))\b'),
    ('INCIDENT_OR_OUTAGE',
     r'\b(?:cyber(?:security)?\s+(?:incident|attack)|data\s+breach|'
     r'(?:network|service)\s+outage|8-k\s+incident)\b',
     r'\b(?:cyber(?:security)?\s+(?:incident|attack)|data\s+breach|'
     r'(?:network|service)\s+outage|8-k\s+incident)\b'),
    ('APPROVAL_OR_REJECTION',
     r'\b(?:approval|approved|approves|rejected|rejects|denied|denial)\b',
     r'\b(?:approval|approved|approves|rejected|rejects|denied|denial)\b'),
    ('SANCTION_ACTION',
     r'\b(?:sanction|sanctions|embargo|export\s+ban|export\s+control)\b',
     r'\b(?:sanction|sanctions|embargo|export\s+ban|export\s+control)\b'),
)


def _open_predicate_action_gate(event: Mapping[str, Any], rules: str) -> dict[str, Any]:
    """Admit only matching *outcome action classes* to candidate review.

    The class is intentionally broad and does not verify identical settlement.
    Otherwise a common topic (e.g. credit cards, Hormuz, crude oil) causes
    irrelevant contracts to be reviewed and may obscure a real candidate.
    """
    event_text = ' '.join(str(event.get(key) or '') for key in
                          ('title', 'description', 'question_we_care_about')).lower()
    source_text = rules.lower()
    triggered = [(name, source_pattern) for name, proposal_pattern, source_pattern in _OPEN_ACTIONS
                 if re.search(proposal_pattern, event_text, re.I)]
    if not triggered:
        return {'eligible': False, 'reason': 'OPEN_FAMILY_OUTCOME_ACTION_NOT_REVIEWED'}
    confirmed = [name for name, source_pattern in triggered
                 if re.search(source_pattern, source_text, re.I)]
    if not confirmed:
        return {'eligible': False, 'reason': 'OPEN_FAMILY_SOURCE_OUTCOME_ACTION_MISMATCH',
                'proposal_action_classes': [name for name, _ in triggered]}
    return {'eligible': True, 'matched_outcome_action_classes': confirmed}


def _open_family_review_gate(event: Mapping[str, Any], primary_rules: str) -> dict[str, Any]:
    """Allow narrowly anchored unknown subjects into *non-certifying* review.

    Only verified exchange rules are compared. Shared distinctive sequences
    are sufficient to *inspect* a candidate, not to equate two propositions.
    In particular, the event's LLM-supplied family/predicate is never trusted
    as an independently defined source outcome. A semantic reviewer still has
    to check jurisdictions, negation, thresholds, dates and settlement.
    """
    title = str(event.get('title') or '')[:800].lower()
    question = str(event.get('question_we_care_about') or event.get('description') or '')[:1600].lower()
    rules = primary_rules[:12000].lower()
    words = lambda value: re.findall(r'[a-z][a-z0-9-]*', value)
    title_words = words(title)
    rule_words = words(rules)
    rule_norm = ' '.join(rule_words)
    if not title_words or not rule_words:
        return {'eligible': False, 'reason': 'OPEN_FAMILY_NO_DEFINED_SOURCE_SUBJECT'}
    anchors: list[str] = []
    # A shared three-word sequence from the actual event title establishes a
    # candidate subject with more specificity than single topical keywords.
    for i in range(len(title_words) - 2):
        segment = title_words[i:i + 3]
        if sum(token not in _GENERIC_EVENT_WORDS for token in segment) < 2:
            continue
        phrase = ' '.join(segment)
        if phrase in rule_norm:
            anchors.append(phrase)
    if not anchors:
        # Some precise source predicates are abbreviated in the event title;
        # accept a rare two-word subject only when both texts also share an
        # independent third distinctive word. "credit card" alone fails.
        source_tokens = set(rule_words) - _GENERIC_EVENT_WORDS
        question_tokens = (set(words(title + ' ' + question)) - _GENERIC_EVENT_WORDS)
        shared = sorted(source_tokens & question_tokens)
        for i in range(len(title_words) - 1):
            pair = title_words[i:i + 2]
            if (all(token not in _GENERIC_EVENT_WORDS for token in pair)
                    and ' '.join(pair) in rule_norm
                    and len(set(shared) - set(pair)) >= 2):
                anchors.append(' '.join(pair))
        if not anchors:
            return {'eligible': False, 'reason': 'OPEN_FAMILY_NO_DISTINCTIVE_SHARED_SOURCE_RULES_SUBJECT',
                    'shared_distinctive_tokens_count': len(shared)}
    action = _open_predicate_action_gate(event, primary_rules)
    if not action['eligible']:
        return action
    return {'eligible': True, 'reason': 'OPEN_FAMILY_SOURCE_SUBJECT_REVIEWABLE_NOT_IDENTICAL',
            'review_anchors': sorted(set(anchors))[:8],
            'reviewed_outcome_action_classes': action['matched_outcome_action_classes'],
            'typed_predicate_diagnostics': {
                'status': 'UNREVIEWED_OPEN_FAMILY_CANDIDATE_ONLY',
                'review_compatible': False,
                'settlement_equivalence_verified': False,
                'missing_proofs': ['independent event predicate', 'jurisdiction and action',
                                   'deadline/measurement period', 'threshold and strictness',
                                   'negation and settlement authority'],
            }}


def _subjects_in_text(text: str) -> set[str]:
    """Recognize event subjects, never explanatory exposure or correlation claims.

    Multiple subjects remain ambiguous: a compound macro proposition is not a
    binary FOMC/CPI predicate merely because its first keyword is recognizable.
    """
    result: set[str] = set()
    if re.search(r'\bfomc\b|\bfederal open market\b|\bfederal reserve\b|\bfederal funds\b|\bfed\b', text):
        result.add('US_FOMC')
    if re.search(r'\bcpi\b|consumer price index', text):
        result.add('US_CPI')
    if ('export control' in text or 'export restriction' in text or
            (('chip' in text or 'semiconductor' in text) and ('china' in text or 'nvidia' in text))):
        result.add('US_CHIP_EXPORT_POLICY')
    # These are explicit research families, not official calendar entries or
    # certified binary propositions. Preserve the existing chip-export route.
    if re.search(_TARIFF_WORDS, text) and 'US_CHIP_EXPORT_POLICY' not in result:
        result.add('US_IMPORT_TARIFF_POLICY')
    if re.search(_HURRICANE_WORDS, text) and re.search(r'\blandfalls?\b', text):
        result.add('HURRICANE_LANDFALL')
    if 'treasury yield' in text or '10-year' in text:
        result.add('US_TREASURY_YIELD')
    if re.search(r'\brecession\b', text):
        result.add('US_RECESSION')
    return result


def _family(event: Mapping[str, Any]) -> str:
    """Classify the question, not incidental discussion of its consequences.

    Source first: the event TITLE defines what is being investigated. Descriptive
    context can mention correlated CPI/FOMC reactions without changing the
    subject. If the title is compound, abstain rather than silently choosing
    a convenient first keyword. Descriptions are fallback only for title-opaque
    candidates. Identifiers and LLM ``family`` fields are not authoritative.
    """
    title = without_tariff_meta_exclusions(str(event.get('title', '')).lower())
    description = without_tariff_meta_exclusions(str(event.get('description', '')).lower())
    if explicit_non_tariff_agreement_title(title):
        # A treaty extension may discuss tariff consequences without defining
        # a tariff action. No reviewed agreement family or eligibility is added.
        return 'UNREVIEWED_EVENT_TYPE'
    # A clause explicitly headed by a release may precede incidental context:
    # 'November CPI release, one day after the December FOMC meeting'.
    subject_clause = re.split(r'[,;:(—]|\s+(?:before|after|following)\s+', title, maxsplit=1)[0]
    head = _subjects_in_text(subject_clause)
    if len(head) == 1:
        return next(iter(head))
    subjects = _subjects_in_text(title)
    if subjects == {'US_RECESSION', 'US_TREASURY_YIELD'}:
        return 'UNDEFINED_YIELD_OR_RECESSION_COMPOSITE'
    if len(subjects) == 1:
        return next(iter(subjects))
    if len(subjects) > 1:
        return 'UNREVIEWED_COMPOSITE_EVENT_FAIL_CLOSED'
    subjects = _subjects_in_text(description)
    if subjects == {'US_RECESSION', 'US_TREASURY_YIELD'}:
        return 'UNDEFINED_YIELD_OR_RECESSION_COMPOSITE'
    return next(iter(subjects)) if len(subjects) == 1 else 'UNREVIEWED_EVENT_TYPE'


def _named_month(text: str) -> int | None:
    months = [number for name, number in _MONTHS.items() if re.search(r'\b' + name + r'\b', text.lower())]
    return months[0] if len(months) == 1 else None


def calendar_context(event: Mapping[str, Any]) -> dict[str, Any]:
    """Expose independently reviewed dates and contradictions, never rewrite the LLM.

    CPI reference month is specified by the release's *data month*, not merely
    its December press-release date. Ambiguous generic CPI references are not
    assumed to mean the month of publication.
    """
    family = _family(event)
    proposed = str(event.get('event_date', ''))
    title = str(event.get('title', ''))
    description = str(event.get('description', ''))
    date_value: date | None
    try:
        date_value = date.fromisoformat(proposed)
    except ValueError:
        date_value = None
    month_ref: str | None = None
    if family == 'US_FOMC':
        if date_value:
            month_ref = f'{date_value.year:04d}-{date_value.month:02d}'
        month_in_title = _named_month(title)
        if month_in_title is not None and date_value and month_in_title != date_value.month:
            month_ref = None  # competing proposal facts; do not resolve silently
    elif family == 'US_CPI':
        # Prefer an explicitly labelled reference/data month, then a month
        # immediately naming the CPI statistic. Do NOT infer reference month
        # from the release date or an incidental FOMC/calendar mention.
        hit = re.search(r'\b(' + '|'.join(_MONTHS) + r')\s+(20\d\d)\s+data\b', title.lower())
        if not hit:
            hit = re.search(r'(?:\bfor\b|\bin\b)\s+(' + '|'.join(_MONTHS) + r')\s+(20\d\d)\b', title.lower())
        if not hit:
            hit = re.search(r'\b(' + '|'.join(_MONTHS) + r')\s+(20\d\d)\s+(?:headline\s+|core\s+)?(?:cpi|consumer price index)\b', title.lower())
        if hit:
            month_ref = f'{int(hit.group(2)):04d}-{_MONTHS[hit.group(1)]:02d}'
        else:
            # A title can mention both 'November CPI' and 'December FOMC'.
            # Only the headline subject clause is eligible for a month hint.
            clause = re.split(r'[,;:(—]|\s+(?:before|after)\s+', title, maxsplit=1)[0]
            month = _named_month(clause)
            if month is not None and date_value:
                # A shorthand is only a reviewed-calendar discovery hint.
                month_ref = f'{date_value.year:04d}-{month:02d}'
    anchor = _CALENDAR.get((family, month_ref)) if month_ref else None
    if not anchor:
        return {'family': family, 'status': 'NO_REVIEWED_CALENDAR_ANCHOR',
                'proposed_date': proposed, 'reference_month': month_ref,
                'source_authority': 'NONE'}
    correct, url, description = anchor
    status = ('PROPOSAL_DATE_MATCHES_REVIEWED_OFFICIAL_SCHEDULE' if proposed == correct
              else 'PROPOSAL_DATE_CONTRADICTS_REVIEWED_OFFICIAL_SCHEDULE')
    return {'family': family, 'status': status, 'proposed_date': proposed,
            'reviewed_date': correct, 'reference_month': month_ref,
            'official_url': url, 'official_calendar_excerpt_summary': description,
            'source_authority': 'MANUALLY_REVIEWED_OFFICIAL_PAGE_TRANSCRIPTION_NOT_RAW_HTML',
            'checked_on': '2026-10-18',
            'model_date_was_overwritten': False}


def evaluate_market(event: Mapping[str, Any], market: Mapping[str, Any],
                    context: Mapping[str, Any]) -> dict[str, Any]:
    """Independently quarantine obviously unrelated contracts before Claude.

    Positive means *same instrument/event family for research*, not identical
    propositions, a market-implied macro probability, or admissible LP axiom.
    Errors and missing details fail closed. Uses exact source rules plus series
    identity; neither market title nor topic similarity can override a mismatch.
    """
    family = context.get('family', 'UNREVIEWED_EVENT_TYPE')
    ticker = market.get('ticker') or market.get('market_id')
    event_ticker = market.get('event_ticker')
    rules = market.get('rules_primary')
    if not all(isinstance(s, str) and s.strip() for s in (ticker, event_ticker, rules)):
        return {'eligible': False, 'reason': 'MISSING_MARKET_IDENTITY_OR_RULES'}
    if not isinstance(market.get('source_url'), str) or not market['source_url'].startswith(
            'https://external-api.kalshi.com/trade-api/v2/markets/'):
        return {'eligible': False, 'reason': 'NO_SUPPORTED_EXACT_CONTRACT_SOURCE'}
    if market.get('source_receipt_status') != 'LOCAL_BYTES_VERIFIED':
        return {'eligible': False, 'reason': 'UNVERIFIED_OR_MISSING_ORIGINAL_PROVIDER_BYTES'}
    if market.get('status') not in {'active', 'open'}:
        return {'eligible': False, 'reason': 'NOT_AN_OPEN_CONTRACT_IN_SOURCE_SNAPSHOT'}
    primary = rules.lower()
    candidate = f'{event.get("title", "")} {event.get("description", "")}'.lower()
    if family == 'US_FOMC':
        if not re.fullmatch(r'KXFEDDECISION-\d{2}[A-Z]{3}', event_ticker):
            return {'eligible': False, 'reason': 'WRONG_INSTITUTION_OR_FOMC_CONTRACT_FAMILY'}
        if not re.search(r'\bfederal reserve\b|\bfomc\b', primary):
            return {'eligible': False, 'reason': 'FOMC_INSTITUTION_NOT_IN_RESOLUTION_RULES'}
        month_ref = context.get('reference_month')
        if month_ref:
            year, month = map(int, month_ref.split('-'))
            month_abbr = list(_MONTHS)[month - 1][:3].upper()
            if event_ticker != f'KXFEDDECISION-{year%100:02d}{month_abbr}':
                return {'eligible': False, 'reason': 'WRONG_FOMC_MEETING_MONTH'}
    elif family == 'US_CPI':
        if not re.fullmatch(r'KXCPI-\d{2}[A-Z]{3}', event_ticker):
            return {'eligible': False, 'reason': 'WRONG_CPI_SERIES_OR_REFERENCE_MONTH'}
        if not (re.search(r'consumer price index|\bcpi\b', primary) and re.search(r'\b(?:increases?|rises?|rose|changes?)\b', primary)):
            return {'eligible': False, 'reason': 'CPI_STATISTIC_NOT_PRESENT_IN_RULES'}
        month_ref = context.get('reference_month')
        if month_ref:
            year, month = map(int, month_ref.split('-'))
            month_word = list(_MONTHS)[month - 1]
            ticker_expected = f'KXCPI-{year%100:02d}{month_word[:3].upper()}'
            if ticker_expected != event_ticker or f'{month_word} {year}' not in primary:
                return {'eligible': False, 'reason': 'WRONG_CPI_REFERENCE_MONTH_OR_YEAR'}
    elif family == 'US_CHIP_EXPORT_POLICY':
        # A bank's sanction *for* importing/exporting chips is still a bank
        # sanctions contract, not automatically a US export-control decision.
        if re.search(r'\b(bank|banks|banking|sanc(?:tion|tions|tioned))\b', primary):
            return {'eligible': False, 'reason': 'BANK_SANCTIONS_NOT_AN_EXPORT_CONTROL_DECISION'}
        if not re.search(r'\b(chip|chips|semiconductor|nvidia|nvda|advanced computing)\b', primary):
            return {'eligible': False, 'reason': 'BANK_SANCTIONS_OR_UNRELATED_TRADE_POLICY_NOT_CHIP_EXPORT'}
        if not re.search(r'\b(export|restrictions?|tariffs?|licens(?:e|ing)|ban)\b', primary):
            return {'eligible': False, 'reason': 'NO_PRECISE_CHIP_EXPORT_OR_TARIFF_ACTION_IN_RULES'}
        if not re.search(r'china|chinese|united states|\bu\.s\.\b|\bus\b', primary):
            return {'eligible': False, 'reason': 'CHIP_POLICY_JURISDICTION_NOT_VERIFIED'}
    elif family == 'US_IMPORT_TARIFF_POLICY':
        # Shared product words alone do not turn a tax credit into an import
        # tariff. All checks below read the actual primary resolution rule.
        if not re.search(_TARIFF_WORDS, primary):
            return {'eligible': False, 'reason': 'IMPORT_TARIFF_ACTION_NOT_IN_PRIMARY_RULES'}
        if not re.search(r'\bunited states\b|\bus\b|(?<!\w)u\.s\.(?!\w)|\bsection\s+(?:232|301)\b', primary):
            return {'eligible': False, 'reason': 'US_IMPORT_TARIFF_JURISDICTION_NOT_VERIFIED'}
        title = str(event.get('title', '')).lower()
        requested = title if re.search(_TARIFF_WORDS, title) else str(event.get('description', '')).lower()
        countries = _named_entities(requested, _COUNTRIES)
        if countries and not countries.intersection(_named_entities(primary, _COUNTRIES)):
            return {'eligible': False, 'reason': 'IMPORT_TARIFF_COUNTRY_MISMATCH_OR_UNPROVEN'}
        products = _named_entities(requested, _PRODUCTS)
        if products and not products.intersection(_named_entities(primary, _PRODUCTS)):
            return {'eligible': False, 'reason': 'IMPORT_TARIFF_PRODUCT_MISMATCH_OR_UNPROVEN'}
    elif family == 'HURRICANE_LANDFALL':
        if not (re.search(_HURRICANE_WORDS, primary) and re.search(r'\blandfalls?\b', primary)):
            return {'eligible': False, 'reason': 'HURRICANE_LANDFALL_NOT_IN_PRIMARY_RULES'}
        if not re.search(r'\bnhc\b|\bnoaa\b|national hurricane cent(?:er|re)', primary):
            return {'eligible': False, 'reason': 'HURRICANE_SETTLEMENT_AUTHORITY_NOT_VERIFIED'}
        title = str(event.get('title', '')).lower()
        requested = title if re.search(_HURRICANE_WORDS, title) else str(event.get('description', '')).lower()
        locations = _named_entities(requested, {**_COUNTRIES, **_COASTAL_LOCATIONS})
        if locations and not locations.intersection(_named_entities(primary, {**_COUNTRIES, **_COASTAL_LOCATIONS})):
            return {'eligible': False, 'reason': 'HURRICANE_LANDFALL_LOCATION_MISMATCH_OR_UNPROVEN'}
        # Do not translate "major" into a wind speed or category threshold.
        # A narrower intensity request must be explicit in source text too.
        if re.search(r'\bmajor\s+hurricane', requested) and not re.search(r'\bmajor\s+hurricane', primary):
            return {'eligible': False, 'reason': 'HURRICANE_INTENSITY_MISMATCH_OR_UNPROVEN'}
        requested_categories = set(re.findall(r'\bcategory\s+([1-5])\b', requested))
        if requested_categories and not requested_categories.intersection(re.findall(r'\bcategory\s+([1-5])\b', primary)):
            return {'eligible': False, 'reason': 'HURRICANE_INTENSITY_MISMATCH_OR_UNPROVEN'}
    elif family == 'US_TREASURY_YIELD':
        # A precise US 10-year-yield settlement can be reviewed as the same
        # family; a generic bond / FOMC / inflation quote cannot stand in.
        if not (re.search(r'10[- ]year', primary) and
                re.search(r'\btreasury\b', primary) and
                re.search(r'\byield\b', primary)):
            return {'eligible': False, 'reason': 'NOT_THE_US_10_YEAR_TREASURY_YIELD_STATISTIC'}
    elif family == 'US_RECESSION':
        # Recession is definition-dependent; a GDP or unemployment value may
        # be a signal but cannot certify any unspecified recession predicate.
        if not re.search(r'\brecession\b', primary):
            return {'eligible': False, 'reason': 'RECESSION_DEFINITION_NOT_PRESENT_IN_MARKET_RULES'}
    elif family == 'UNDEFINED_YIELD_OR_RECESSION_COMPOSITE':
        return {'eligible': False, 'reason': 'NO_PRECISE_BINARY_MACRO_TARGET_FOR_THIS_CANDIDATE'}
    else:
        # The existing USMCA/CUSMA joint-review route has a separate
        # multi-sovereign confirmation predicate that is *not* equivalent to
        # a generic settlement mentioning the same treaty. Retain the
        # historical fail-closed certificate until that predicate has its
        # own independent verifier; broad candidate lexical review is not
        # a substitute for the missing multi-country evidence.
        if explicit_non_tariff_agreement_title(str(event.get('title') or '')):
            return {'eligible': False, 'reason': 'UNREVIEWED_EVENT_FAMILY_FAIL_CLOSED'}
        candidate_review = _open_family_review_gate(event, primary)
        if not candidate_review['eligible']:
            return candidate_review
        return {**candidate_review,
                'macro_relationship_verified': False,
                'contract_identity_needs_SEPARATE_EXACT_RULE_AND_QUOTE_GATE': True,
                'admission_scope': 'CLAUDE_REVIEW_CANDIDATE_NOT_LP_CONSTRAINT',
                'source_quote_can_price_original_target': False}
    # A second, typed pre-review gate prevents source-family overlap from
    # laundering an explicit CPI basis, frequency, strictness or Fed action
    # contradiction into a paid semantic review. This is NOT the LP verifier.
    typed = qualify_rule_predicate({**event, "official_calendar_review": context}, market, family)
    if not typed["review_compatible"]:
        return {"eligible": False, "reason": typed["reason"],
                "typed_predicate_diagnostics": typed}
    # The family gate never asserts the model's broad macro question is the
    # exchange settlement, even when the instrument/category is unambiguous.
    return {'eligible': True, 'reason': 'SOURCE_CONTRACT_FAMILY_AND_PERIOD_COMPATIBLE_ONLY',
            'macro_relationship_verified': False,
            'contract_identity_needs_SEPARATE_EXACT_RULE_AND_QUOTE_GATE': True,
            'admission_scope': 'CLAUDE_REVIEW_CANDIDATE_NOT_LP_CONSTRAINT',
            'typed_predicate_diagnostics': typed}


def public_calendar_context(as_of: str, horizon_days: int) -> dict[str, Any]:
    """Supply concise official schedule anchors to Claude, not an inference engine.

    Current scope is the four reviewed 2026 events. A future release requires a
    new validated calendar; these anchors intentionally do not extrapolate.
    """
    start = date.fromisoformat(as_of[:10])
    stop = start.toordinal() + horizon_days
    anchors = []
    for (family, ref_month), (release, url, desc) in sorted(_CALENDAR.items()):
        d = date.fromisoformat(release)
        if start.toordinal() <= d.toordinal() <= stop:
            anchors.append({'event_family': family, 'reference_or_meeting_month': ref_month,
                            'event_date': release, 'official_url': url,
                            'scope': 'CURATED_OFFICIAL_SCHEDULE_NOT_DOWNLOADED_HTML',
                            'description': desc})
    return {'anchors': anchors,
            'instruction': 'Use these reviewed official dates for matching events; never invent a replacement date. '
                           'Calendar schedule does not establish a numeric event outcome. '
                           'CPI reference month differs from the month of its release; '
                           'market settlement/fallback may differ from BLS news releases.'}
