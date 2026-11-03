"""Narrow lexical exclusions for research routing, never admission evidence.

A treaty extension is not an import-tariff action merely because its context
discusses tariffs. Conversely, a negative policy predicate (no new tariffs,
tariff removal or exemptions) is still about tariffs. These helpers neither
parse settlement rules nor establish a reviewed trade-agreement taxonomy.
"""
from __future__ import annotations

import re


_TARIFF_META_EXCLUSION = re.compile(
    r"\bnot(?:\s*,?\s+(?:by\s+)?itself\s*,?)?\s+(?:an?\s+)?"
    r"(?:withdrawal\s+or\s+(?:an?\s+)?)?"
    r"(?:tariff|import[- ]duty)\s+(?:event|question|hypothesis|scenario|risk)s?\b",
    re.I,
)
_AGREEMENT = re.compile(
    r"\b(?:USMCA|CUSMA|T[-\s\u2010-\u2015\u2212]+MEC|NAFTA|CPTPP|RCEP)\b|"
    r"\btrade[- ](?:agreements?|treaty|treaties|pacts?)\b|"
    r"\bUnited\s+States[-\s\u2010-\u2015\u2212]+Mexico[-\s\u2010-\u2015\u2212]+Canada\s+Agreement\b",
    re.I,
)
_AGREEMENT_ACTION = re.compile(
    r"\b(?:extend(?:s|ed|ing)?|extensions?|renew(?:s|ed|ing|als?)?|"
    r"ratif(?:y|ies|ied|ication)|sign(?:s|ed|ing)?|"
    r"withdraw(?:s|n|ing|als?)?|terminat(?:e|es|ed|ion)|"
    r"confirm(?:s|ed|ation)?|joint[- ]review|review(?:s|ed)?)\b",
    re.I,
)
# This is an exclusion guard, not a new tariff classifier. Keep every explicit
# tariff/duty subject available to the existing routers and their stricter gates.
_EXPLICIT_TARIFF_SUBJECT = re.compile(
    r"\btariffs?\b|\b(?:import|countervailing)\s+dut(?:y|ies)\b|"
    r"\banti[- ]dumping\b|\bsection\s+(?:232|301)\b",
    re.I,
)
_TRAILING_CONTEXT = re.compile(r"\b(?:before|after|following)\b", re.I)


def without_tariff_meta_exclusions(text: str) -> str:
    """Ignore only explicit *classification* exclusions in a routing copy.

    In particular ``no new tariffs``, ``not impose tariffs`` and ``not only a
    tariff event`` are unchanged. Original descriptions and source rules must
    remain intact; this normalized string is not retained as event evidence.
    """
    return _TARIFF_META_EXCLUSION.sub(" ", text)


def explicit_non_tariff_agreement_title(title: str) -> bool:
    """Recognize a treaty-action title with no explicit tariff subject.

    Title-head scope is deliberate: descriptive consequences cannot turn an
    extension into a tariff action, and a treaty mentioned only after a temporal
    qualifier cannot replace a CPI/Fed subject. Do not split on commas: country
    lists such as ``US, Mexico and Canada`` belong to the same subject. Compound
    heads that also mention a tariff subject remain with the existing gates.
    True means unmodeled agreement research, not a verified binary proposition.
    """
    subject = _TRAILING_CONTEXT.split(without_tariff_meta_exclusions(title), maxsplit=1)[0]
    return bool(_AGREEMENT.search(subject) and _AGREEMENT_ACTION.search(subject)
                and not _EXPLICIT_TARIFF_SUBJECT.search(subject))
