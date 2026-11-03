"""Pure, conservative checks on an event's written definition.

This is a presentation/admission guard, not a semantic-equivalence prover.
``DEFINED_FOR_REVIEW`` means that these explicit defects were not found; it
does not certify a source, a calendar date, a market match or an economic model.
The original title, question, date and evidence are never rewritten. In
particular, an unresolved target cannot borrow its threshold, date or meaning
from a merely related source-defined scenario.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import re


# Preserve the existing dated, documentary R29 check. This is not a fresh
# calendar request, a general GDP-calendar inference or an immutable schedule.
_Q4_2026_GDP_ADVANCE_DATE = "2027-01-28"
_BEA_SCHEDULE = "https://www.bea.gov/news/schedule/next-year"

_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = r"(?:%|percent(?:age\s+points?)?|basis\s+points?|bps?)(?![a-z])"
_COMPARATOR = (
    r"(?:at\s+least|at\s+most|no\s+(?:less|more)\s+than|not\s+(?:less|more)\s+than|"
    r"(?:greater|less|more|higher|lower)\s+than|(?:strictly\s+)?(?:above|below)|"
    r"over|under|exceeds?|reaches?|hits?|equals?|equal\s+to|exactly|>=|<=|>|<|=)"
)
_NUMERIC_YIELD = re.compile(
    rf"(?:{_COMPARATOR}\s*{_NUMBER}\s*{_UNIT}|"
    rf"\b(?:threshold|cutoff|strike)\s*(?:(?:of|at|is)\s+|:\s*)?{_NUMBER}\s*{_UNIT}|"
    rf"\b(?:ris(?:e|es)|rais(?:e|es)|increas(?:e|es)|falls?|drops?|declin(?:e|es)|"
    rf"decreas(?:e|es))\s+(?:by\s+)?(?:at\s+least\s+)?{_NUMBER}\s*{_UNIT}|"
    rf"\bbetween\s+{_NUMBER}\s*(?:{_UNIT})?\s+and\s+{_NUMBER}\s*{_UNIT}|"
    rf"(?<!\w){_NUMBER}\s*(?:{_UNIT})?\s*(?:to|through|-)\s*{_NUMBER}\s*{_UNIT})",
    re.I,
)
_PENDING_THRESHOLD = re.compile(
    r"\b(?:threshold|cutoff|strike|numerical\s+(?:trigger|level))\b[^.!?]{0,150}"
    r"(?:\b(?:has|have)\s+not\s+(?:yet\s+)?been\s+(?:chosen|set|specified|defined|selected)|"
    r"\b(?:is|remains?)\s+(?:still\s+)?(?:unspecified|undefined|unknown|undetermined|unselected)|"
    r"\b(?:is\s+)?not\s+(?:yet\s+)?(?:chosen|set|specified|defined|selected)|"
    r"\b(?:to\s+be|must\s+be|needs?\s+to\s+be|will\s+be)\s+"
    r"(?:chosen|set|specified|defined|selected|retrieved|obtained)|"
    r"\b(?:must|will|needs?\s+to)\s+come\s+from|\bTBD\b)", re.I,
)

# Distinct settlement actions, not every occurrence of the word "or". In
# particular, product lists, "new or higher", source documents and numerical
# ranges are not different event predicates merely because they contain OR.
_TRIGGER_PATTERNS = {
    "import_tariff": r"\btariffs?\b|\bimport\s+dut(?:y|ies)\b",
    "border_restriction": r"\bborder\s+(?:restrictions?|closures?|closings?|shutdowns?)\b|"
                          r"\bclos(?:e|es|ing)\s+(?:the\s+)?border\b",
    "trade_agreement_withdrawal": r"\b(?:withdraw(?:al|s|ing)?|terminat(?:e|es|ion))\b"
                                 r"[^.!?]{0,100}\b(?:USMCA|NAFTA|trade\s+agreement)\b|"
                                 r"\b(?:USMCA|NAFTA|trade\s+agreement)\b[^.!?]{0,100}"
                                 r"\b(?:withdraw(?:al|s|ing)?|terminat(?:e|es|ion))\b|"
                                 r"\bwithdrawal\s+notice\b",
    "recession_declaration": r"\bNBER\b[^.!?]{0,70}\b(?:recession|declares?|announces?)\b|"
                             r"\b(?:recession|declares?|announces?)\b[^.!?]{0,70}\bNBER\b",
    "negative_gdp_release": r"\b(?:GDP|gross\s+domestic\s+product)\b[^.!?]{0,80}"
                            r"\b(?:negative|below\s+zero|contract(?:s|ion)?)\b|"
                            r"\bnegative\b[^.!?]{0,50}\b(?:GDP|gross\s+domestic\s+product)\b",
}
_TRIGGER_PATTERNS = {key: re.compile(pattern, re.I) for key, pattern in _TRIGGER_PATTERNS.items()}

# These bounded corporate checks require an explicit result comparison or a
# guidance revision, not merely names of financial metrics. They are matched
# on separate OR alternatives below so document/date choices do not become
# unions just because two metrics occur elsewhere in the same sentence.
_RESULT_COMPARISON = r"(?:miss(?:es|ed)?|beats?|exceed(?:s|ed)?|above|below|falls?\s+short\s+of)"
_GUIDANCE_REVISION = r"(?:cuts?|lower(?:s|ed)?|reduc(?:e|es|ed)|rais(?:e|es|ed)|withdraw(?:s|n)?|suspend(?:s|ed)?)"
_CORPORATE_TRIGGER_PATTERNS = {
    "corporate_revenue_result":
        rf"\b(?:revenues?|sales)\b(?!\s+guidance\b)[^.!?;,]{{0,60}}\b{_RESULT_COMPARISON}\b|"
        rf"\b{_RESULT_COMPARISON}\b[^.!?;,]{{0,60}}\b(?:revenues?|sales)\b(?!\s+guidance\b)",
    "corporate_earnings_result":
        rf"\b(?:EPS|earnings(?:\s+per\s+share)?)\b(?!\s+(?:guidance|release|call)\b)"
        rf"[^.!?;,]{{0,60}}\b{_RESULT_COMPARISON}\b|"
        rf"\b{_RESULT_COMPARISON}\b[^.!?;,]{{0,60}}\b(?:EPS|earnings(?:\s+per\s+share)?)\b"
        rf"(?!\s+(?:guidance|release|call)\b)",
    "corporate_guidance_revision":
        rf"\b{_GUIDANCE_REVISION}\b[^.!?;,]{{0,45}}\bguidance\b|"
        rf"\bguidance\b[^.!?;,]{{0,45}}\b{_GUIDANCE_REVISION}\b",
}
_CORPORATE_TRIGGER_PATTERNS = {
    key: re.compile(pattern, re.I) for key, pattern in _CORPORATE_TRIGGER_PATTERNS.items()
}

_OFFICIAL_RELEASE = re.compile(
    r"\b(?:BEA|BLS|GDP|gross\s+domestic\s+product|CPI|payrolls?|FOMC|"
    r"Federal\s+Reserve|Bureau\s+of\s+(?:Economic\s+Analysis|Labor\s+Statistics))\b", re.I)
_CORPORATE_RELEASE = re.compile(
    r"\b(?:earnings|EPS|financial\s+results|(?:quarterly|annual)\s+results)\b|"
    r"\b(?:revenue|sales|guidance)\b[^.!?]{0,70}\b(?:release|report|announcement|results?)\b|"
    r"\b(?:release|report|announcement|results?)\b[^.!?]{0,70}\b(?:revenue|sales|guidance)\b", re.I)
# Unlike an unknown market price or earnings outcome, an explicitly unknown
# corporate release date/source is a definition defect. This does not fetch or
# infer a company's calendar, and a supplied monitoring deadline cannot fix it.
_CORPORATE_RELEASE_SUBJECT = re.compile(
    r"\b(?:earnings|results|revenue|EPS|guidance)(?:\s+(?:release|report|announcement|call))?\s+(?:date|time)\b|"
    r"\b(?:release|publication|announcement|report|call)\s+(?:date|time)\b|"
    r"\b(?:date|time)\s+of\s+(?:the\s+)?(?:earnings|results|release|publication|announcement|report|call)\b|"
    r"\b(?:official|documentary|exact)\s+(?:source|schedule|calendar)\b|"
    r"\b(?:the|its|their|official|exact)\s+(?:date|time)\b|"
    r"\b(?:calendar|schedule)\b", re.I)
_CORPORATE_PENDING_CALENDAR = re.compile(
    r"\bTBD\b|\b(?:has|have)\s+not\s+(?:yet\s+)?been\s+(?:announced|published)\b|"
    r"\bnot\s+(?:yet\s+)?(?:announced|published)\b|"
    r"\b(?:remains?|is)\s+(?:unscheduled|undetermined)\b", re.I)
_RELEASE_SUBJECT = re.compile(
    r"\b(?:release|publication|announcement|meeting|decision)\s+(?:date|time)|"
    r"\bdate\s+of\s+(?:the\s+)?(?:release|publication|announcement|meeting|decision)|"
    r"\b(?:official|documentary|exact)\s+(?:source|schedule|calendar)|"
    r"\b(?:the|its|their|official|exact)\s+(?:date|time|source)|"
    r"\b(?:calendar|schedule)\b", re.I)
_UNCONFIRMED = re.compile(
    r"\b(?:needs?\s+to\s+be|must\s+be|(?:is|are)\s+to\s+be)\s+"
    r"(?:confirmed|verified|checked|corroborated|established|retrieved)|"
    r"\b(?:has|have)\s+not\s+(?:yet\s+)?been\s+"
    r"(?:confirmed|verified|checked|corroborated|established)|"
    r"\b(?:not|never)\s+(?:yet\s+)?(?:a\s+)?(?:date[- ]confirmed|confirmed|verified)|"
    r"\b(?:unconfirmed|unverified|unresolved|unknown)\b|"
    r"\b(?:still\s+)?to\s+be\s+(?:confirmed|verified|checked)\b", re.I)


def _text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.translate(str.maketrans({"\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"})).split())


def _question(event):
    for key in ("original_question", "question_we_care_about", "description"):
        if value := _text(event.get(key)):
            return value
    return ""


def _day(event):
    value = event.get("event_date", event.get("date"))
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            pass
    return None


def _proposition(text):
    # Later rationale or instructions do not add triggers to the predicate:
    # "tariffs ... . Do not combine this with a withdrawal" is not a union.
    text = re.split(r"[.!?]\s+", text, maxsplit=1)[0]
    return re.split(r"\b(?:because|counting\s+requires|market\s+rules|do\s+not\s+combine|"
                    r"distinct\s+from)\b", text, maxsplit=1, flags=re.I)[0]


def _compound_triggers(text):
    proposition = _proposition(text)
    if not re.search(r"\bor\b|\beither\b", proposition, re.I):
        return set()
    triggers = {name for name, pattern in _TRIGGER_PATTERNS.items() if pattern.search(proposition)}
    alternatives = [
        {name for name, pattern in _CORPORATE_TRIGGER_PATTERNS.items() if pattern.search(part)}
        for part in re.split(r"\bor\b", proposition, flags=re.I)
    ]
    nonempty = [names for names in alternatives if names]
    if len(nonempty) >= 2 and len(set.union(*nonempty)) > 1:
        triggers.update(set.union(*nonempty))
    return triggers


def review_event_definition(event) -> dict:
    """Return a non-mutating written-definition review with the existing shape.

    Accepted inputs include raw engine events and public display cards. No
    stored positive admission, related source table or event identifier can
    override an explicit defect in this event's own title/question. A separate
    source child's own question is checked independently of its parent.
    """
    if not isinstance(event, Mapping):
        return {"status": "INCOMPLETE_PREDICATE", "reason": "No inspectable event definition was provided."}
    title, question = _text(event.get("title")), _question(event)
    text = title + " " + question
    if not text.strip():
        return {"status": "INCOMPLETE_PREDICATE", "reason": "No inspectable event definition was provided."}

    is_treasury_yield = (bool(re.search(r"\byields?\b", text, re.I)) and
                        bool(re.search(r"\bTreasury\b|\bconstant[- ]maturity\b|"
                                       r"\b(?:\d+|ten|two|five|thirty)[- ](?:year|yr)\b|\b\d+y\b", text, re.I)))
    if is_treasury_yield and (_PENDING_THRESHOLD.search(question or title) or not _NUMERIC_YIELD.search(text)):
        return {"status": "INCOMPLETE_PREDICATE",
                "reason": "No resolved numerical Treasury-yield trigger is specified in this event's own definition. "
                          "State the yield series, numerical level or change, observation convention and window "
                          "before pricing it; a related contract does not supply a missing threshold."}

    if (re.search(r"\bBEA\b", text, re.I) and re.search(r"\badvance\s+estimate\b", text, re.I)
            and re.search(r"\b(?:Q4|fourth\s+quarter)\s*2026\b", text, re.I)
            and (day := _day(event)) and day < _Q4_2026_GDP_ADVANCE_DATE):
        return {"status": "INCOMPATIBLE_RELEASE_DEADLINE",
                "reason": "The BEA calendar checked Oct 27, 2026 schedules Q4 2026 advance "
                          "GDP for Jan 28, 2027, after this proposed event deadline. "
                          "The event needs a corrected horizon and revalidation.",
                "source_url": _BEA_SCHEDULE, "source_checked_on": "2026-10-27"}

    title_triggers, question_triggers = _compound_triggers(title), _compound_triggers(question)
    if len(title_triggers) > 1 or len(question_triggers) > 1:
        mismatch = len(title_triggers) > 1 and bool(question) and not title_triggers.issubset(
            {name for name, pattern in {**_TRIGGER_PATTERNS, **_CORPORATE_TRIGGER_PATTERNS}.items()
             if pattern.search(_proposition(question))})
        return {"status": "COMPOSITE_PREDICATE",
                "reason": ("The headline combines distinct alternative settlement triggers, while the detailed "
                           "question specifies a different or narrower event. Align them as one explicit predicate "
                           "before assigning a market probability."
                           if mismatch else
                           "The question combines distinct settlement triggers with OR. Split and verify them "
                           "separately; one trigger's market probability does not price their union.")}

    is_macro_release = bool(_OFFICIAL_RELEASE.search(text))
    is_corporate_release = bool(_CORPORATE_RELEASE.search(text))
    if is_macro_release or is_corporate_release:
        for clause in re.split(r"[.!?;,]\s+|\b(?:but|whereas|while)\b", text, flags=re.I):
            macro_pending = (is_macro_release and _RELEASE_SUBJECT.search(clause)
                             and _UNCONFIRMED.search(clause))
            # Keep a confirmed date separate from an unknown earnings result
            # joined by AND. "Date and exact source remain unknown" still has
            # a calendar/source subject in the second part.
            corporate_pending = is_corporate_release and any(
                _CORPORATE_RELEASE_SUBJECT.search(part)
                and (_UNCONFIRMED.search(part) or _CORPORATE_PENDING_CALENDAR.search(part))
                for part in re.split(r"\band\b", clause, flags=re.I)
            )
            if macro_pending or corporate_pending:
                return {"status": "UNCONFIRMED_RELEASE_SOURCE",
                        "reason": "The event text explicitly leaves its official release date or source unconfirmed. "
                                  "Keep it as a research hypothesis until documentary evidence fixes the release "
                                  "and deadline; a monitoring-window end is not a confirmed announcement date."}
    return {"status": "DEFINED_FOR_REVIEW", "reason": ""}


__all__ = ["review_event_definition"]
