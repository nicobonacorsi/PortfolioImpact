"""Contract-first geopolitical scenarios: the contract fixes the event, a declared scenario fixes the impact.

The chain for an unscheduled risk such as Taiwan or US chip-export policy:

1. The portfolio decides which risks are searched (discovery, unchanged).
2. Each searched contract is classified from its own resolution RULES, never its
   title: a blockade, an offensive, a military encounter, a statute that subjects
   AI-chip exports to review. A travel advisory, a fine or a price index is a
   different event and is rejected with its reason. A contract is admitted only
   when it is open, quoted, and its deadline falls inside the risk's horizon.
3. An admitted contract becomes a scenario about exactly its event. The impact is
   a declared scenario for that event class (window, broad-market move, reviewed
   issuer residuals, gold and long-yield moves), with sources and assumptions. The contract's
   price never sets the shocks; it only weights YES against NO:

       Delta   = V_YES - V_NO                         (declared scenario contrast)
       YES row = (1 - q) * Delta,  NO row = -q * Delta  (move not already in prices)

   Reviewed issuers count once: directly and inside funds through the dated holdings ledger.
   Inside a fund the non-NVDA rest moves with beta_rest = (beta_fund - s*beta_NVDA)/(1-s),
   so the fund's total market beta is unchanged.
4. The broad risk keeps its own card. A lower bound Q(E) >= Q(C) is reported only
   when an implication C => E is established from both definitions and both
   deadlines: the family's reviewed risk definition is read into a typed predicate
   (triggers, required effects, qualifiers), and the contract's rules must establish
   one named trigger and every required effect. Otherwise its probability stays unknown.
   Bounds from different venues are not combined (no common pricing measure).

No network, clock read or model call happens here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, time, timezone
import math
import re

import numpy as np

from .adapters import parse_time
from .outcome_table import (combine_loadings, _bond_duration, _partial_bond_component,
                           instrument_model_gate)
from .fund_composition import uncovered_bond_symbols
from .portfolio import Portfolio
from .geopolitical_channels import channel_specs, issuer_exposure_weights

SCHEMA = "castle.option1.contract_scenarios.v1"
DEFAULT_DRAWS = 20000
DEFAULT_SEED = 20261023

LPL_2026 = {
    "title": "LPL Research (George Smith), Iran Escalation: How Markets Have Reacted to Geopolitical Events",
    "url": "https://www.lpl.com/research/blog/iran-escalation-how-markets-have-reacted-to-geopolitical-events.html",
    "date": "2026-03-04",
    "locator": "Table of geopolitical events: S&P 500 one-day reaction, total drawdown, trading days to bottom",
    "values": {"average_one_day": -.010, "average_total_drawdown": -.044, "average_days_to_bottom": 17.9,
               "september_11_2001_drawdown": -.116, "pearl_harbor_1941_drawdown": -.198,
               "kuwait_invasion_1990_drawdown": -.169, "russia_ukraine_2022_drawdown": -.068},
}
CAPTURED_GEOPOLITICAL_DAYS = (
    "Captured Massive daily closes (2023-10-23 to 2026-10-22): TLT -1.55% on 2024-04-15 and -0.96% on 2025-06-13 "
    "(Middle East escalations), but +0.46% and +1.09% on the 2025-04-03/04 equity sell-off. The long-yield response "
    "to a geopolitical shock has no identified sign.")
# Long-yield change implied by TLT over the same 20-session windows as the gold
# measurement (scripts/r10_measure_geopolitical_episodes.py), one row per LPL episode.
TREASURY_EPISODES = {"episodes": 12, "up": 7, "min_bp": -43.7, "max_bp": 38.4,
                     "report": "reports/r10_validation/geopolitical_episodes.json"}
MEASURED_LONG_YIELD = {
    "point": 0.0, "range": [TREASURY_EPISODES["min_bp"], TREASURY_EPISODES["max_bp"]],
    "unit": "long Treasury yield change, bp", "basis": "MEASURED_EPISODES_SIGN_NOT_IDENTIFIED",
    "anchors": ("Over 20 sessions after each of the 12 LPL episodes inside TLT's history, the long yield implied by TLT "
                "rose in 7 and fell in 5, from -43.7 to +38.4 bp: no identified sign, so the point is zero and the range "
                "spans the episodes. " + CAPTURED_GEOPOLITICAL_DAYS)}
H20_ANCHOR = {
    "event": "US license requirement for NVIDIA H20 exports to China, disclosed in an SEC filing on 2025-04-15 after the US close",
    "press_source": "https://www.malaymail.com/news/money/2025/04/16/us-chip-curbs-on-china-to-cost-nvidia-us55b-shares-tumble/173227",
    "first_session": "2025-04-16",
    "captured_close_to_close": {"NVDA": -.0687, "SPY": -.0222, "QQQ": -.0302},
    "nvda_daily_beta_on_captured_sample": 2.05,
    "nvda_net_of_beta_times_spy": -.0232,
    "qqq_net_of_beta_times_spy": -.0020,
}

PROFILES = {
    "TAIWAN_MAJOR_MILITARY_ACTION": {
        "label": "Major PRC military action against Taiwan",
        "window": "20 trading days from onset (LPL average: 17.9 trading days from event to the S&P 500 low)",
        "market_return": {"point": -.116, "range": [-.198, -.044], "unit": "SPY total move over the window",
                          "basis": "ANCHORED_ON_CITED_EPISODES",
                          "anchors": "Range from the LPL average geopolitical drawdown (-4.4%) to the largest in its table "
                                     "(Pearl Harbor, -19.8%); point at the 9/11 drawdown (-11.6%).",
                          "source": LPL_2026},
        "nvda_addon": {"point": -.10, "range": [-.20, 0.0], "unit": "NVDA return beyond its market beta",
                       "basis": "DECLARED_STRESS_NO_PRECEDENT",
                       "anchors": "No PRC blockade or invasion of Taiwan has occurred, so no episode measures the "
                                  "supply-interruption add-on. The range runs from no extra effect to -20%."},
        "long_yield_bp": dict(MEASURED_LONG_YIELD),
    },
    "TAIWAN_LIMITED_USE_OF_FORCE": {
        "label": "Limited PRC-Taiwan use of force",
        "window": "20 trading days from the incident",
        "market_return": {"point": -.044, "range": [-.116, -.010], "unit": "SPY total move over the window",
                          "basis": "ANCHORED_ON_CITED_EPISODES",
                          "anchors": "Range from the LPL average one-day reaction (-1.0%) to the 9/11 drawdown (-11.6%); "
                                     "point at the LPL average drawdown (-4.4%).",
                          "source": LPL_2026},
        "nvda_addon": {"point": 0.0, "range": [-.05, 0.0], "unit": "NVDA return beyond its market beta",
                       "basis": "DECLARED_STRESS_NO_PRECEDENT",
                       "anchors": "An incident need not interrupt shipping; the add-on runs from none to -5%."},
        "long_yield_bp": dict(MEASURED_LONG_YIELD),
    },
    "US_AI_CHIP_EXPORT_CURB": {
        "label": "US curb on AI-chip exports reaching NVIDIA's China sales",
        "window": "Resolution session, close to close",
        "market_return": {"point": 0.0, "range": [0.0, 0.0], "unit": "SPY move",
                          "basis": "NO_BROAD_MARKET_CHANNEL_DECLARED",
                          "anchors": "On the anchor day QQQ moved -0.20% net of beta x SPY: no broad channel is added."},
        "nvda_addon": {"point": H20_ANCHOR["nvda_net_of_beta_times_spy"], "range": [H20_ANCHOR["captured_close_to_close"]["NVDA"], 0.0],
                       "unit": "NVDA return beyond its market beta", "basis": "ANCHORED_ON_ONE_EPISODE",
                       "anchors": "H20 license requirement (first session 2025-04-16): NVDA -6.87% close to close, -2.32% net "
                                  "of beta x SPY. A statute that only enables review is at most as severe as that de facto ban; "
                                  "the range runs from no effect to the raw -6.87%.",
                       "source": deepcopy(H20_ANCHOR)},
        "long_yield_bp": {"point": 0.0, "range": [0.0, 0.0], "unit": "long Treasury yield change, bp",
                          "basis": "NO_RATE_CHANNEL_DECLARED", "anchors": "Not a macro event for Treasuries."},
    },
}

_ACTOR = r"\b(?:china|people'?s republic|prc)\b"
CLASSES = {
    "TAIWAN": [
        {"event_class": "PRC_BLOCKADE_OF_TAIWAN", "profile": "TAIWAN_MAJOR_MILITARY_ACTION",
         "require": [r"\bblockade\b", r"commercial (?:traffic|shipping|vessels)", _ACTOR, r"\btaiwan\b"]},
        {"event_class": "PRC_MILITARY_OFFENSIVE_AGAINST_TAIWAN", "profile": "TAIWAN_MAJOR_MILITARY_ACTION",
         "require": [r"military offensive", r"(?:establish|take|seize) control", r"\btaiwan\b"]},
        {"event_class": "PRC_TAIWAN_MILITARY_ENCOUNTER", "profile": "TAIWAN_LIMITED_USE_OF_FORCE",
         "require": [r"military encounter|exchange of gunfire|direct military engagement", r"\btaiwan\b", _ACTOR]},
    ],
    "US_CHIP_EXPORT_POLICY": [
        {"event_class": "US_STATUTE_REVIEW_OR_BLOCK_OF_AI_CHIP_EXPORTS", "profile": "US_AI_CHIP_EXPORT_CURB",
         "require": [r"signed into law|become law", r"\b(?:ai[- ]chips?|advanced (?:ai )?chips?|gpus?)\b", r"\bexports?\b",
                     r"congressional review|blocking period|prohibit|\bban\b|\bblock"]},
        {"event_class": "US_EXECUTIVE_AI_CHIP_EXPORT_RESTRICTION", "profile": "US_AI_CHIP_EXPORT_CURB",
         "require": [r"bureau of industry and security|department of commerce|commerce department|\bbis\b|executive order",
                     r"\bexports?\b", r"\b(?:chips?|semiconductors?|gpus?)\b", r"restrict|license requirement|prohibit|\bban\b|entity list"]},
        {"event_class": "US_STATUTE_EXPORT_COMPLIANCE_CONDITION", "profile": None,
         "require": [r"signed into law|become law", r"\b(?:chips?|semiconductors?)\b", r"\bexports?\b",
                     r"location[- ]verification|security mechanism"]},
    ],
}
REJECTIONS = (
    ("GOVERNMENT_TRAVEL_ADVISORY", r"travel advisory",
     "A travel advisory is a policy warning; it is neither necessary nor sufficient for military action or a supply disruption."),
    ("MONETARY_POLICY_DECISION", r"central bank|monetary policy|discount rate", "A central-bank rate decision is a different event."),
    ("CORPORATE_ENFORCEMENT_PENALTY", r"\bfine\b|monetary penalty",
     "A fine on a company is enforcement, not a restriction on exports."),
    ("GOVERNMENT_EQUITY_STAKE", r"takes? a stake|equity ownership",
     "A government equity stake changes who owns a company; it is not a military action, a blockade or an export restriction."),
    # "index" alone also occurs in exclusions such as "index or mutual funds will not qualify".
    ("PRICE_INDEX_LEVEL", r"\bindex (?:price|value|level|displays|chart)\b|\bprice index\b",
     "A price-index level is a market outcome, not a policy or military event."),
    ("POLITICAL_OFFICE", r"\bpresident of\b|election|resign", "A change of office is a different event."),
)
# Reviewed implications C => E. E is a reviewed definition per risk family, not
# the discovery's free text: live runs on 2026-10-23 worded the Taiwan risk four
# different ways, and a bound that depends on wording is not reproducible.
# The reviewed text is read into a typed predicate (triggers named, effects
# required, qualifiers); anything the reader does not recognize blocks a proof.
REVIEWED_RISKS = {
    # The portfolio channel is Taiwan's chip supply, so the risk requires disruption.
    "TAIWAN": ("The trigger would be a blockade, quarantine or military action by China against Taiwan that disrupts "
               "commercial shipping or chip production or exports. The window runs from the evaluation date to the "
               "horizon end."),
    "US_CHIP_EXPORT_POLICY": ("A formal US Commerce/BIS rule, license-policy change or executive action that newly "
                              "restricts exports of advanced AI chips to China. The window runs from the evaluation "
                              "date to the horizon end."),
}
RISK_TRIGGERS = {
    "BLOCKADE": r"\bblockade\b",
    "QUARANTINE": r"\bquarantine\b",
    "MILITARY_ACTION": r"\bmilitary (?:action|offensive|attack|strike)s?\b|\buse of force\b|\binvasion\b|\binvade\b",
}
_RISK_EFFECT = (r"\b(?:that|which)\s+(?:is\s+|are\s+)?(?:reported\s+to\s+)?(?:materially\s+)?disrupts?\s+"
                r"(?:commercial\s+)?(?:shipping|chips?|semiconductors?)"
                r"(?:\s+(?:or|and)\s+(?:commercial\s+)?(?:shipping|chips?|semiconductors?|production|exports?))*"
                r"(?:\s+(?:production|exports?))?(?:\s+(?:or|and)\s+(?:production|exports?))*")
_RISK_STRIP = [  # recognized scaffolding: window, actor, object, deferral to the contract's own rules
    r",?\s*(?:as defined (?:in|by) (?:the |a )?(?:actual |matched )?contract(?:'s)? rules|per the exact wording of any matched contract)",
    r"\bbetween \d{4}-\d{2}-\d{2} and \d{4}-\d{2}-\d{2}(?: inclusive)?",
    r"\b(?:china|the prc|prc)\s+(?:carries out|conducts|imposes|launches|establishes)\b",
    r"\bby (?:china|the prc|prc)\b",
    r"\b(?:against|involving|affecting|on|around) taiwan\b",
    r"\bcustoms\b",
]
_RISK_ALLOWED = {"a", "an", "the", "or", "and", "blockade", "quarantine", "military", "action", "actions", "offensive",
                 "attack", "strike", "use", "of", "force", "invasion", "invade"}
_RISK_EXCLUSIONS = {"rhetoric", "routine", "exercises", "exercise", "military", "drills", "statements", "alone", "or",
                    "and", "would", "not", "count", "unless", "the", "contract", "says", "so"}
_RISK_NARROWING = r"\bonly\b|\bat least\b|\bmore than\b|\blasting\b|\bfull[- ]scale\b|\bsustained\b"
# What each admitted contract class establishes, from patterns in its own rules.
CONTRACT_ESTABLISHES = {
    "PRC_BLOCKADE_OF_TAIWAN": {
        "triggers": {"BLOCKADE": r"\bblockade\b"},
        "effect": (r"(?:prevent|block|halt)\w*[^.]{0,80}commercial (?:traffic|shipping|vessels)",
                   "Its blockade must prevent foreign commercial traffic to or from Taiwan's main ports or airports for at "
                   "least 24 hours, which disrupts commercial shipping and the air cargo that carries chip exports."),
    },
    "PRC_MILITARY_OFFENSIVE_AGAINST_TAIWAN": {
        "triggers": {"MILITARY_ACTION": r"military offensive"},
        "effect": None,
    },
}
NOT_PROVEN = {
    "PRC_TAIWAN_MILITARY_ENCOUNTER": "An encounter between the two militaries can be started by either side and need not be a blockade, offensive or other action the risk names.",
    "US_STATUTE_REVIEW_OR_BLOCK_OF_AI_CHIP_EXPORTS": "The target lists Commerce/BIS rules, license-policy changes or executive actions; a statute creating review is not shown to be one, and review need not newly restrict exports.",
    "US_STATUTE_EXPORT_COMPLIANCE_CONDITION": "Location verification conditions exports; it is not shown to newly restrict or tighten limits on exports to China.",
}


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.;])\s+", text) if part.strip()]


def risk_predicate(definition):
    """Typed reading of the broad risk's own definition; unrecognized qualifiers are reported, never dropped."""
    text = " ".join(str(definition or "").split())
    sentences = _sentences(text)
    lower = [sentence.lower() for sentence in sentences]
    def names_trigger(sentence):
        return any(re.search(pattern, sentence) for pattern in RISK_TRIGGERS.values())
    intro = r"\b(?:trigger would be|counts if)\b"
    index = next((i for i, s in enumerate(lower) if re.search(intro, s) and names_trigger(s)),
                 next((i for i, s in enumerate(lower) if names_trigger(s)), None))
    if index is None:
        return {"status": "NO_REVIEWED_TRIGGER", "triggers": [], "required_effects": [], "qualifiers": [],
                "unrecognized": [], "trigger_clause": None}
    clause = lower[index]
    match = re.search(intro, clause)
    clause = clause[match.end():] if match else clause
    effects = []
    if re.search(_RISK_EFFECT, clause):
        effects.append("SHIPPING_OR_CHIP_DISRUPTION")
        clause = re.sub(_RISK_EFFECT, " ", clause)
    qualifiers = []
    declared = r"\b(?:declared|announced|formal)\s+(?=(?:customs\s+)?(?:blockade|quarantine|military)\b)"
    if re.search(declared, clause):
        # Read as applying to every listed trigger: the conservative reading of an ambiguous modifier.
        qualifiers.append("DECLARED")
        clause = re.sub(declared, " ", clause)
    for pattern in _RISK_STRIP:
        clause = re.sub(pattern, " ", clause)
    unrecognized = sorted(set(re.findall(r"[a-z0-9]+", clause)) - _RISK_ALLOWED)
    for i, sentence in enumerate(lower):
        if i == index:
            continue
        if re.search(r"would not count|does not count|do not count|not qualify|excluded", sentence):
            extra = sorted(set(re.findall(r"[a-z]+", sentence)) - _RISK_EXCLUSIONS)
            unrecognized += ["exclusion: " + word for word in extra]
        elif re.search(_RISK_NARROWING, sentence):
            unrecognized.append("narrowing sentence: " + sentences[i][:120])
    triggers = [name for name, pattern in RISK_TRIGGERS.items() if re.search(pattern, lower[index])]
    return {"status": "PARSED" if not unrecognized else "UNRECOGNIZED_QUALIFIER", "triggers": triggers,
            "required_effects": effects, "qualifiers": qualifiers, "unrecognized": unrecognized,
            "trigger_clause": sentences[index]}


def _excerpt(text, pattern):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
    end = min([i for i in (text.find(".", match.end()), text.find("\n", match.end())) if i >= 0] or [len(text)])
    return text[start:end + 1].strip()[:400]


_YES_PREFIX = re.compile(r'^\s*(?:this|the) market (?:will )?resolves? to ["\u201c\u201d\']?yes["\u201c\u201d\']? if\s+', re.I)
_YES_SUFFIX = re.compile(r',?\s+then (?:this|the) market (?:will )?resolves? to ["\u201c\u201d\']?yes["\u201c\u201d\']?\.?\s*$', re.I)
_PRC_SUBJECT = r"(?:china|the prc|prc|the people's republic of china)(?:\s*\([^)]*\))?\s+"
_AFFIRMATIVE_ACTION = {
    "PRC_BLOCKADE_OF_TAIWAN": _PRC_SUBJECT + r"(?:imposes?|establishes?|enforces?|launches?|announces it has established or otherwise de facto establishes)\s+(?:an?\s+)?(?:aerial or naval\s+|naval\s+|aerial\s+)?blockade\b",
    "PRC_MILITARY_OFFENSIVE_AGAINST_TAIWAN": _PRC_SUBJECT + r"(?:commences?|launches?|begins?)\s+(?:a\s+)?military offensive\b",
    "PRC_TAIWAN_MILITARY_ENCOUNTER": r"there is (?:a|an) military encounter between (?:the )?military forces of\b",
    "US_STATUTE_REVIEW_OR_BLOCK_OF_AI_CHIP_EXPORTS": r"legislation that (?:subjects covered ai[- ]chip exports to mandatory congressional review with a statutory blocking period|(?:would )?(?:prohibits?|bans?|blocks?)\s+(?:the\s+)?(?:ai[- ]chip|advanced (?:ai )?chip|gpu) exports)\b",
    "US_EXECUTIVE_AI_CHIP_EXPORT_RESTRICTION": r"(?:the\s+)?(?:u\.?s\.?\s+)?(?:bureau of industry and security|department of commerce|commerce department|bis)\s+(?:imposes?|introduces?|issues?|adopts?|announces?|implements?)\s+(?:new\s+)?(?:export restrictions?|(?:an?\s+)?(?:export\s+)?(?:ban|license requirement|entity list)|(?:a\s+)?rule (?:restricting|prohibiting))\b",
    "US_STATUTE_EXPORT_COMPLIANCE_CONDITION": r"legislation that (?:would )?require[s]? location[- ]verification security mechanisms\b",
}
_NEGATED_OR_CONDITIONAL = re.compile(
    r"\b(?:not|never|neither|no|unless|except|until|whether|without|fails? to|failure to|avoids?|"
    r"instead of|rather than|merely|hypothetical|proposed|proposal|plans? to|threatens? to)\b|n['\u2019]t\b", re.I)
_REVIEWED_OR_PHRASES = (
    "announces it has established or otherwise de facto establishes", "aerial or naval",
    "ingress or egress", "to or from", "threat or use of force", "part or whole",
)


def _yes_condition(rules):
    """Read a bounded resolver grammar; never infer YES from a keyword elsewhere.

    Only the first resolution paragraph defines the event. Negative exclusions,
    source descriptions and examples cannot establish a positive trigger. Unknown
    syntax is deliberately left for review, not treated as semantic certainty.
    """
    first = str(rules).strip().split("\n", 1)[0].strip()
    first = re.split(r"(?<=[.!?])\s+Otherwise,?\s+", first, maxsplit=1, flags=re.I)[0].strip()
    match = _YES_PREFIX.match(first)
    if match:
        condition = first[match.end():].rstrip(". ")
    elif re.match(r"^if\s+", first, re.I) and (match := _YES_SUFFIX.search(first)):
        condition = re.sub(r"^if\s+", "", first[:match.start()], flags=re.I).rstrip(". ")
    else:
        return None, "No supported affirmative YES-if resolution clause was found."
    # More than one resolver, or a second sentence hidden in the first line,
    # requires explicit review. U.S. is a recognized abbreviation, not a stop.
    unabridged = re.sub(r"\bu\.s\.", "us", condition, flags=re.I)
    if re.search(r"[.!?]\s+[A-Z]|\b(?:resolves?|settles?|pays?\s+out|alternatively)\b", unabridged, re.I):
        return None, "Multiple or ambiguous resolution clauses require semantic review."
    if _NEGATED_OR_CONDITIONAL.search(condition):
        return None, "The YES condition contains negation, an exception, or an unreviewed conditional action."
    alternatives = condition.lower()
    for phrase in _REVIEWED_OR_PHRASES:
        alternatives = alternatives.replace(phrase, "reviewed_phrase")
    if re.search(r"\bor\b", alternatives):
        return None, "An unreviewed alternative YES event could settle without the modeled action."
    # A later independent positive settlement rule would expand the YES event.
    tail = str(rules).strip().split("\n", 1)[1] if "\n" in str(rules).strip() else ""
    other_yes = (r"(?:market|contract)\s+(?:will\s+)?(?:also\s+)?(?:resolves?|settles?|pays?\s+out)"
                 r"\s+(?:(?:to|as)\s+)?[\"\u201c\u201d']?yes\b|"
                 r"also\s+(?:resolves?|settles?|pays?\s+out)\s+(?:to\s+)?[\"\u201c\u201d']?yes\b|"
                 r"yes\s+(?:payout|resolution)\s+(?:also\s+)?(?:occurs?|applies?|happens?)\s+(?:if|when)\b|"
                 r"\balternatively\b")
    if re.search(other_yes, tail, re.I):
        return None, "A later alternative YES resolution clause requires semantic review."
    return condition, None


def _blockade_effect(rules, condition):
    """Mandatory disruption only, excluding optional examples and negative definitions."""
    pattern = CONTRACT_ESTABLISHES["PRC_BLOCKADE_OF_TAIWAN"]["effect"][0]
    if (re.search(pattern, condition, re.I)
            and not re.search(r"\b(?:may|might|could|can|optionally|for example|potentially)\b", condition, re.I)):
        return condition
    # The captured positive definition precedes a separate 'is not' list. Read
    # only the first mandatory bullet; do not pool predicates across exclusions.
    definition = re.search(r"\ba qualifying blockade is:\s*(?:-\s*)?([^\n]+)", rules, re.I)
    if definition:
        clause = definition.group(1).strip()
        if (re.match(r"(?:prevents?|blocks?|halts?)\b", clause, re.I)
                and re.search(pattern, clause, re.I)
                and not _NEGATED_OR_CONDITIONAL.search(clause)
                and not re.search(r"\b(?:may|might|could|optionally|for example)\b", clause, re.I)):
            return clause
    return None


def classify_contract(candidate, search_family):
    """Classify supported affirmative resolution clauses, never title/keyword polarity."""
    rules = str(candidate.get("rules_text") or "")
    lower = rules.lower()
    condition, semantic_reason = _yes_condition(rules)
    for spec in CLASSES.get(search_family, []):
        if condition is None or not re.match(_AFFIRMATIVE_ACTION[spec["event_class"]], condition, re.I):
            continue
        effect = _blockade_effect(rules, condition) if spec["event_class"] == "PRC_BLOCKADE_OF_TAIWAN" else None
        if spec["event_class"] == "PRC_BLOCKADE_OF_TAIWAN" and effect is None:
            continue
        evidence_text = condition + ("\n" + effect if effect else "")
        if all(re.search(pattern, evidence_text, re.I) for pattern in spec["require"]):
            return {"event_class": spec["event_class"], "profile": spec["profile"], "admitted_class": True,
                    "rules_evidence": [_excerpt(evidence_text, pattern) for pattern in spec["require"]],
                    "yes_condition": condition, "mandatory_disruption_clause": effect,
                    "semantic_status": "REVIEWED_AFFIRMATIVE_RESOLVER_GRAMMAR",
                    "basis": "AFFIRMATIVE_YES_RULE_PREDICATES_NOT_TITLE"}
    for name, pattern, reason in REJECTIONS:
        if re.search(pattern, lower):
            return {"event_class": name, "profile": None, "admitted_class": False, "reason": reason,
                    "rules_evidence": [_excerpt(rules, pattern)], "basis": "RULES_TEXT_PREDICATES_NOT_TITLE"}
    return {"event_class": "UNREVIEWED_EVENT_CLASS", "profile": None, "admitted_class": False,
            "reason": semantic_reason or "The affirmative YES condition matches no reviewed action grammar for this risk.",
            "semantic_status": "UNREVIEWED_OR_NONAFFIRMATIVE_RESOLVER", "rules_evidence": [],
            "basis": "RULES_TEXT_PREDICATES_NOT_TITLE"}


def horizon_end(target):
    day = datetime.fromisoformat(str(target["event_date"])).date()
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def review_candidate(candidate, target):
    """Admission of one searched contract as a scenario for exactly its own event."""
    classification = classify_contract(candidate, target["family"])
    quote = (candidate.get("contract_quote_reference") or {}).get("interval")
    try:
        end = parse_time(candidate.get("contract_end_time"))
    except (ValueError, TypeError, AttributeError):
        end = None
    limit = horizon_end(target)
    try:
        evaluated = parse_time((candidate.get("contract_quote_reference") or {}).get("evaluated_at"))
    except (ValueError, TypeError, AttributeError):
        evaluated = None
    valid_quote = (isinstance(quote, (list, tuple)) and len(quote) == 2
                   and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in quote)
                   and 0 <= quote[0] <= quote[1] <= 1)
    stated = re.search(r"\b(?:by|before) ([A-Za-z]+ \d{1,2}, \d{4})\b", classification.get("yes_condition", ""))
    stated_day = None
    if stated:
        try:
            stated_day = datetime.strptime(stated.group(1), "%B %d, %Y").date()
        except ValueError:
            pass
    # The reviewed public contracts use US dates; UTC settlement can fall on
    # that date or the following one. A different year/month is not equivalent.
    deadline_consistent = not (stated_day and end and (end.date() - stated_day).days not in (0, 1))
    row = {"venue": candidate.get("venue"), "contract_id": candidate.get("contract_id"),
           "market_question": candidate.get("market_question"), "rules_sha256": candidate.get("rules_sha256"),
           "contract_end_time": candidate.get("contract_end_time"), "horizon_end": limit.isoformat(),
           "quote_interval": deepcopy(quote),
           "quote_scope": (candidate.get("contract_quote_reference") or {}).get("scope"),
           **classification}
    if not classification["admitted_class"]:
        row.update(decision="REJECTED_DIFFERENT_EVENT")
    elif end is None or end > limit:
        row.update(decision="REJECTED_DEADLINE_AFTER_HORIZON",
                   reason="The contract can settle YES after the risk's horizon, so its price includes later outcomes.")
    elif not deadline_consistent:
        row.update(decision="REJECTED_RULE_DEADLINE_MISMATCH",
                   reason="The explicit YES-rule deadline disagrees with the contract's settlement metadata.")
    elif evaluated is not None and end <= evaluated:
        row.update(decision="REJECTED_EXPIRED_WINDOW",
                   reason="The contract deadline has already passed at quote evaluation, even if the provider still reports it open.")
    elif not candidate.get("market_open_at_evaluation") or not valid_quote:
        row.update(decision="REJECTED_NOT_OPEN_OR_NOT_QUOTED",
                   reason="No valid current probability quote at the evaluation time: " + ", ".join((candidate.get("contract_quote_reference") or {}).get("reasons", [])))
    elif classification["profile"] is None:
        row.update(decision="ADMITTED_IMPACT_NOT_QUANTIFIED",
                   reason="The contract's event is admitted, but no reviewed scenario profile quantifies its portfolio effect.")
    else:
        row.update(decision="ADMITTED_CONTRACT_SCENARIO")
    return row


def implication_bound(review, target, definition=None):
    """Q(E) >= Q(C) only when C => E follows from both definitions and both deadlines.

    E is the family's reviewed definition unless one is passed explicitly.
    """
    def not_proven(reason, predicate=None):
        return {"status": "IMPLICATION_NOT_PROVEN", "contract_id": review["contract_id"], "reason": reason,
                "risk_predicate": predicate}
    rules = str(review.get("_rules_text") or "")
    checked = classify_contract({"rules_text": rules}, target.get("family"))
    if (not str(review.get("decision", "")).startswith("ADMITTED") or not checked["admitted_class"]
            or checked["event_class"] != review.get("event_class")):
        return not_proven("The actual YES resolution clause has no consistent admitted affirmative event.")
    rule = CONTRACT_ESTABLISHES.get(review["event_class"]) if target.get("family") == "TAIWAN" else None
    if rule is None:
        return not_proven(NOT_PROVEN.get(review["event_class"], "No reviewed implication from this event class to the target."))
    definition = definition or REVIEWED_RISKS["TAIWAN"]
    predicate = risk_predicate(definition)
    if predicate["status"] == "NO_REVIEWED_TRIGGER":
        return not_proven("The risk's definition names no reviewed trigger (blockade, quarantine or military action).", predicate)
    if predicate["unrecognized"]:
        return not_proven("The risk's definition adds qualifiers the reviewed reading does not cover: "
                          + ", ".join(predicate["unrecognized"]) + ".", predicate)
    # Only the affirmative condition and its mandatory definition establish facts.
    # Rejected examples elsewhere in the rules must never prove an implication.
    evidence = checked["yes_condition"] + "\n" + (checked.get("mandatory_disruption_clause") or "")
    lower = evidence.lower()
    shared = [name for name in predicate["triggers"]
              if name in rule["triggers"] and re.search(rule["triggers"][name], lower)]
    if not shared:
        return not_proven("The contract's event is none of the triggers the risk names ("
                          + ", ".join(predicate["triggers"]) + ").", predicate)
    if "DECLARED" in predicate["qualifiers"] and (not re.search(r"\bannounc", lower) or re.search(r"de facto|otherwise", lower)):
        return not_proven("The risk counts only a declared action; this contract also settles YES on one China never "
                          "announces (for example a de facto blockade).", predicate)
    steps = [f"The risk's reviewed definition: \"{predicate['trigger_clause']}\" It names {', '.join(predicate['triggers'])}"
             + (f" and requires {', '.join(predicate['required_effects'])}." if predicate["required_effects"] else
                " and requires no further effect."),
             f"The contract establishes {shared[0]}: \"{_excerpt(evidence, rule['triggers'][shared[0]])}\""]
    if predicate["required_effects"]:
        effect = rule["effect"]
        if effect is None or not checked.get("mandatory_disruption_clause"):
            return not_proven("The risk requires disruption of shipping or chips; this contract can settle YES without it "
                              "(for example an offensive against an outlying island).", predicate)
        steps.append("Its mandatory blockade definition requires disruption of commercial traffic. Rules: "
                     + f"\"{checked['mandatory_disruption_clause']}\"")
    if re.search(r"would not count|does not count|do not count|not qualify|excluded", definition.lower()):
        steps.append("The risk excludes rhetoric and exercises; the contract's event is an actual action, so the exclusion does not apply.")
    start_assumption = ("No qualifying contract event occurred before evaluation. An open, unresolved market alone "
                        "does not establish this: reporting or settlement can lag the event.")
    return {"status": "CONDITIONAL_LOWER_BOUND_FROM_IMPLYING_CONTRACT", "contract_id": review["contract_id"],
            "venue": review["venue"], "lower_bound": None, "conditional_lower_bound": review["quote_interval"][0],
            "condition": start_assumption, "unconditional_bound_available": False, "risk_predicate": predicate,
            "inequality": "Only under the stated start-window assumption: Q(E) >= Q(C) >= contract bid, for any pricing measure consistent with this quote",
            "proof": steps + [
                f"Deadline: the contract settles by {review['contract_end_time']}, before the horizon end {review['horizon_end']}.",
                "Unverified start-window assumption: " + start_assumption],
            "not_claimed": "No upper bound, and no combination with bounds from another venue (no common pricing measure)."}


def _nvda_share(exposure_report, portfolio):
    share = {p.symbol: 0.0 for p in portfolio.positions}
    for entry in exposure_report.entries:
        if entry.symbol == "NVDA" and entry.asset_type == "stock" and entry.status in {"direct", "look_through"}:
            share[entry.original_symbol] = share.get(entry.original_symbol, 0.0) + entry.portfolio_weight
    return share


def _uniform(spec, rng, n):
    lo, hi = spec["range"]
    return rng.uniform(lo, hi, n) if hi > lo else np.full(n, float(spec["point"]))


def build_contract_scenario(portfolio, exposure_report, daily_beta_report, review, *, rate_sensitivity=None,
                            holdings_quality=None, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED,
                            issuer_shock_overrides=None, instrument_exposures=None,
                            identity_blocked_symbols=None):
    """YES/NO table for exactly one admitted contract under its declared scenario profile."""
    if not isinstance(portfolio, Portfolio):
        raise ValueError("A validated original-position portfolio is required")
    if review.get("decision") != "ADMITTED_CONTRACT_SCENARIO":
        raise ValueError("Only an admitted contract defines a scenario")
    scope_gate = instrument_model_gate(portfolio, instrument_exposures=instrument_exposures,
        identity_blocked_symbols=identity_blocked_symbols, rate_sensitivity=rate_sensitivity)
    blocked_scope = set(scope_gate["blocked_symbols"])
    instrument_sources = {row.get("symbol"): row
                          for row in (instrument_exposures or {}).get("positions", [])}
    profile = deepcopy(PROFILES[review["profile"]])
    specs, excluded_sources = channel_specs(review["profile"], as_of=exposure_report.evaluated_at,
        nvda_spec=profile["nvda_addon"], overrides=issuer_shock_overrides)
    if review["profile"] == "US_AI_CHIP_EXPORT_CURB":
        rules = str(review.get("_rules_text") or " ".join(review.get("rules_evidence", []))).lower()
        names_nvda = bool(re.search(r"\bnvidia\b|\bnvda\b|\b(?:h20|h100|h200|b200)\b", rules))
        names_amd = bool(re.search(r"\bamd\b|advanced micro devices|\bmi\d{3}\b", rules))
        if names_nvda and not names_amd:
            specs.pop("AMD", None)
            excluded_sources.append({"symbol": "AMD", "reason": "NVIDIA_SPECIFIC_RULE_DOES_NOT_ESTABLISH_AMD_PRODUCT_SCOPE"})
        elif names_amd and not names_nvda:
            specs.pop("NVDA", None)
            excluded_sources.append({"symbol": "NVDA", "reason": "AMD_SPECIFIC_RULE_DOES_NOT_ESTABLISH_NVIDIA_PRODUCT_SCOPE"})
    profile["nvda_addon"] = specs.get("NVDA", {"point": 0., "range": [0., 0.], "unit": "NVDA residual return",
        "basis": "NO_REVIEWED_NVIDIA_PRODUCT_CHANNEL", "anchors": "This issuer-specific contract does not establish NVIDIA coverage."})
    issuer_weights, issuer_ledger = issuer_exposure_weights(portfolio, exposure_report, specs)
    q_lo, q_hi = (float(v) for v in review["quote_interval"])
    if not 0 <= q_lo <= q_hi <= 1:
        raise ValueError("Invalid contract quote interval")
    q_ref = (q_lo + q_hi) / 2
    rng = np.random.default_rng(seed)
    notional = portfolio.notional
    market = _uniform(profile["market_return"], rng, n_draws)
    addon = _uniform(profile["nvda_addon"], rng, n_draws)
    a_lo, a_hi = profile["nvda_addon"]["range"]
    severity = (addon - a_lo) / (a_hi - a_lo) if a_hi > a_lo else rng.uniform(0., 1., n_draws)
    # Each issuer shares one draw across every direct/fund occurrence. Supply
    # residuals are comonotone; no artificial diversification from resampling.
    # A channel whose sign history does not identify is drawn on its own stream,
    # so it neither hedges nor amplifies the supply shock and no other draw changes.
    independent = np.random.default_rng(seed + 1).uniform(0., 1., n_draws)
    channel_draws = {s: spec["range"][0] + (spec["range"][1] - spec["range"][0]) *
                    (independent if spec.get("dependence") == "INDEPENDENT_OF_SUPPLY_SEVERITY" else
                     1. - severity if spec["mode"] == "REPLACES_MARKET_BETA" else severity)
                    for s, spec in specs.items()}
    dy = _uniform(profile["long_yield_bp"], rng, n_draws)
    m0, a0, y0 = profile["market_return"]["point"], profile["nvda_addon"]["point"], profile["long_yield_bp"]["point"]
    daily = daily_beta_report.get("instruments", {})
    nvda_fit = combine_loadings(daily.get("NVDA"), None)
    nvda_beta = None if nvda_fit is None else (nvda_fit["loading"], rng.normal(nvda_fit["loading"], nvda_fit["se"], n_draws))
    blocked_bonds = uncovered_bond_symbols(rate_sensitivity)
    shares = {s: weights.get("NVDA", 0.) for s, weights in issuer_weights.items()}
    parts, missing, beta_only, partial_bonds = [], [], [], []
    delta_point, delta_draw = 0.0, np.zeros(n_draws)
    beta_dollars = bond_dollars = bond_weight = bond_usd_per_bp = 0.0
    for position in portfolio.positions:
        weight = position.weight
        amount = notional * weight
        if position.asset_type == "cash" or weight == 0:
            parts.append({"symbol": position.symbol, "component": "Unchanged cash", "channel": "NONE", "pnl_usd": 0.0})
            continue
        if position.symbol in blocked_scope:
            missing.append(position.symbol)
            missing_basis = ("DOLLAR_DV01_PORTFOLIO_BINDING_MISMATCH_OR_MISSING"
                             if position.symbol in scope_gate["dollar_binding_blocked_symbols"]
                             and position.symbol not in scope_gate["identity_blocked_symbols"]
                             else "KNOWN_BOND_WITHOUT_DURATION_NO_EQUITY_PROXY"
                             if position.symbol in blocked_bonds
                             and position.symbol not in scope_gate["identity_blocked_symbols"]
                             else "INSTRUMENT_IDENTITY_OR_FUND_CLASS_UNRESOLVED")
            part = {"symbol": position.symbol, "component": "Whole-fund economic scope unavailable; equity proxy prohibited",
                    "channel": "UNKNOWN", "pnl_usd": None,
                    "missing_basis": missing_basis}
            if position.symbol not in scope_gate["identity_blocked_symbols"]:
                partial = _partial_bond_component(position, rate_sensitivity,
                                                   instrument_sources.get(position.symbol), notional=notional)
                if partial:
                    part["covered_bond_component"] = partial
                    partial_bonds.append({"symbol": position.symbol,
                        "delta_pnl_usd": partial["covered_usd_per_bp_parallel_rise"] * y0 + 0.,
                        "covered_fund_nav_weight": partial["covered_fund_nav_weight"],
                        "remaining_fund_nav_weight_not_priced": partial["remaining_fund_nav_weight_not_priced"],
                        "remaining_nav_weight_is_signed_accounting_not_risk_coverage": True,
                        "scope": partial["scope"], "yield_change_is_declared_scenario_assumption": True})
            parts.append(part)
            continue
        replacement = specs.get(position.symbol)
        if replacement is not None and replacement["mode"] == "REPLACES_MARKET_BETA" and position.asset_type == "etf":
            point = amount * replacement["point"]
            draw = amount * channel_draws[position.symbol]
            parts.append({"symbol": position.symbol, "component": replacement["label"] + " (replaces market beta)",
                          "channel": "COMMODITY", "pnl_usd": point, "issuer": position.symbol,
                          "shock_basis": replacement["basis"], "source": deepcopy(replacement.get("source")),
                          "portfolio_weight": weight})
            delta_point += point
            delta_draw += draw
            continue
        bond = _bond_duration(position, notional, rate_sensitivity,
                              instrument_sources.get(position.symbol))
        if bond is not None:
            duration = rng.uniform(*bond["duration_range"], n_draws) if bond["duration_range"][1] > bond["duration_range"][0] else np.full(n_draws, bond["duration"])
            point = -amount * bond["duration"] * y0 / 1e4 + 0.0  # no negative zero in the display
            draw = -amount * duration * dy / 1e4
            parts.append({"symbol": position.symbol, "component": f"{position.symbol} via duration {bond['duration']:.2f}",
                          "channel": "RATES", "pnl_usd": point,
                          "data_quality": deepcopy((holdings_quality or {}).get(position.symbol))})
            bond_dollars += point
            bond_weight += weight
            bond_usd_per_bp += -amount * bond["duration"] / 1e4
            delta_point += point
            delta_draw += draw
            continue
        if position.symbol in blocked_bonds:
            missing.append(position.symbol)
            parts.append({"symbol": position.symbol, "component": "Bond duration unavailable; equity proxy prohibited",
                          "channel": "UNKNOWN", "pnl_usd": None,
                          "missing_basis": "KNOWN_BOND_WITHOUT_DURATION_NO_EQUITY_PROXY"})
            continue
        fit = combine_loadings(daily.get(position.symbol), None)
        if fit is None:
            missing.append(position.symbol)
            parts.append({"symbol": position.symbol, "component": "No sensitivity", "channel": "UNKNOWN", "pnl_usd": None})
            continue
        beta_draw = rng.normal(fit["loading"], fit["se"], n_draws)
        beta_dollars += amount * fit["loading"]
        nvda_weight = min(shares.get(position.symbol, 0.0), weight)
        s = nvda_weight / weight
        own_channels = issuer_weights[position.symbol]
        if not own_channels:
            beta_only.append(position.symbol)
        if position.symbol == "NVDA" or s <= 0 or s >= 1 - 1e-12 or nvda_beta is None:
            point = amount * (fit["loading"] * m0 + s * a0)
            draw = amount * (beta_draw * market + s * addon)
            label = "NVDA direct" if position.symbol == "NVDA" else position.symbol + " market beta" + (" + NVDA share add-on" if s > 0 else "")
            parts.append({"symbol": position.symbol, "component": label, "channel": "EQUITY", "pnl_usd": point})
        else:
            rest_point = (fit["loading"] - s * nvda_beta[0]) / (1 - s)
            rest_draw = (beta_draw - s * nvda_beta[1]) / (1 - s)
            inside_point = notional * nvda_weight * (nvda_beta[0] * m0 + a0)
            inside_draw = notional * nvda_weight * (nvda_beta[1] * market + addon)
            rest_amount = amount - notional * nvda_weight
            parts.append({"symbol": position.symbol, "component": f"NVDA inside {position.symbol} ({s:.1%} of the fund)",
                          "channel": "EQUITY", "pnl_usd": inside_point,
                          "data_quality": deepcopy((holdings_quality or {}).get(position.symbol))})
            parts.append({"symbol": position.symbol, "component": f"Rest of {position.symbol} (beta {rest_point:.2f})",
                          "channel": "EQUITY", "pnl_usd": rest_amount * rest_point * m0})
            point = inside_point + rest_amount * rest_point * m0
            draw = inside_draw + rest_amount * rest_draw * market
        # Fund baseline is already present exactly once above. Add only the
        # reviewed issuer residual, never that constituent's full beta return.
        for issuer, issuer_weight in own_channels.items():
            if issuer == "NVDA":
                continue
            spec = specs[issuer]
            increment = notional * issuer_weight * spec["point"]
            point += increment
            draw += notional * issuer_weight * channel_draws[issuer]
            parts.append({"symbol": position.symbol, "issuer": issuer,
                          "component": issuer + " residual " + ("direct" if position.symbol == issuer else "inside " + position.symbol),
                          "channel": "EQUITY", "pnl_usd": increment,
                          "portfolio_weight": issuer_weight, "shock_basis": spec["basis"],
                          "source": deepcopy(spec.get("source")),
                          "data_quality": deepcopy((holdings_quality or {}).get(position.symbol))})
        delta_point += point
        delta_draw += draw
    complete = not missing
    q_edges = (q_lo, q_hi)
    yes_edges = [(1 - q) * delta_draw for q in q_edges]
    no_edges = [-q * delta_draw for q in q_edges]

    def _range(edges):
        return [float(np.percentile(np.minimum(*edges), 5)), float(np.percentile(np.maximum(*edges), 95))]

    def _sign(point, draw):
        return None if abs(point) < 1e-9 else float(np.mean(np.sign(draw) == np.sign(point)))

    yes_point, no_point = (1 - q_ref) * delta_point, -q_ref * delta_point
    yes_range, no_range = _range(yes_edges), _range(no_edges)
    rows = [
        {"outcome_id": review["contract_id"] + ":YES", "label": "YES: " + review["market_question"],
         "q_interval": [q_lo, q_hi], "q_reference": q_ref, "pnl_if_outcome_usd": yes_point,
         "pnl_model_range_5_95_usd": yes_range, "sign_probability_under_model": _sign(yes_point, (1 - q_ref) * delta_draw),
         "sign_determined_for_every_feasible_q": bool(yes_range[0] > 0 or yes_range[1] < 0),
         "q_weighted_contribution_usd": q_ref * yes_point},
        {"outcome_id": review["contract_id"] + ":NO", "label": "NO: the contract settles NO", "q_interval": [1 - q_hi, 1 - q_lo],
         "q_reference": 1 - q_ref, "pnl_if_outcome_usd": no_point, "pnl_model_range_5_95_usd": no_range,
         "sign_probability_under_model": _sign(no_point, -q_ref * delta_draw),
         "sign_determined_for_every_feasible_q": bool(no_range[0] > 0 or no_range[1] < 0),
         "q_weighted_contribution_usd": (1 - q_ref) * no_point},
    ]
    if partial_bonds:
        partial_point = math.fsum(row["delta_pnl_usd"] for row in partial_bonds)
        for row, multiplier in zip(rows, (1. - q_ref, -q_ref)):
            row["partial_covered_bond_contributions"] = [
                {**part, "pnl_usd": multiplier * part["delta_pnl_usd"] + 0.}
                for part in partial_bonds]
            row["known_positions_plus_covered_bond_components_pnl_usd"] = (
                row["pnl_if_outcome_usd"] + multiplier * partial_point)
    if missing:
        for row in rows:
            row["known_positions_only_pnl_usd"] = row["pnl_if_outcome_usd"]
            row["pnl_if_outcome_usd"] = None
            row["q_weighted_contribution_usd"] = None
            row["pnl_model_range_5_95_usd"] = None
            row["sign_probability_under_model"] = None
            row["sign_determined_for_every_feasible_q"] = None
    equity_dollars = math.fsum(p["pnl_usd"] for p in parts if p["channel"] == "EQUITY")
    nvda_total = math.fsum(min(shares.get(p.symbol, 0.0), p.weight) for p in portfolio.positions
                           if p.symbol not in missing)
    channels = []
    for issuer, spec in specs.items():
        total = math.fsum(w.get(issuer, 0.) for symbol, w in issuer_weights.items() if symbol not in missing)
        if spec["mode"] == "REPLACES_MARKET_BETA":
            total = math.fsum(p.weight for p in portfolio.positions
                             if p.symbol == issuer and p.asset_type == "etf" and p.symbol not in missing)
        if total > 0:
            channels.append({"issuer": issuer, "portfolio_weight": total,
                             "usd_per_one_percent_return": notional * total * .01,
                             "declared_channel_pnl_usd": notional * total * spec["point"], **deepcopy(spec)})
    return {
        "schema": SCHEMA, "status": "DECLARED_CONTRACT_SCENARIO" if complete else "PARTIAL_KNOWN_POSITIONS_ONLY",
        "instrument_identity_gate": scope_gate,
        "legacy_fallback_allowed": not bool(scope_gate["blocked_symbols"]),
        "contract": {k: deepcopy(review[k]) for k in ("venue", "contract_id", "market_question", "rules_sha256",
                                                       "contract_end_time", "quote_interval", "quote_scope",
                                                       "event_class", "rules_evidence")},
        "scenario": {"profile": review["profile"], **deepcopy(profile)},
        "delta_pnl_usd": delta_point if complete else None,
        "known_positions_only_delta_pnl_usd": delta_point if not complete else None,
        "partial_covered_bond_contributions": partial_bonds,
        "known_positions_plus_covered_bond_components_delta_pnl_usd": (
            delta_point + math.fsum(row["delta_pnl_usd"] for row in partial_bonds) if partial_bonds else None),
        "delta_pnl_model_range_5_95_usd": ([float(v) for v in np.percentile(delta_draw, [5, 95])]
                                            if complete else None),
        "positions": parts, "missing_symbols": missing,
        "issuer_channels": channels, "issuer_exposure_ledger": issuer_ledger,
        "excluded_channel_sources": excluded_sources,
        "outcomes": rows,
        "summary": {"q_standard_deviation_usd": math.sqrt(q_ref * (1 - q_ref)) * abs(delta_point) if complete else None,
                    "known_positions_only_q_standard_deviation_usd": math.sqrt(q_ref * (1 - q_ref)) * abs(delta_point) if not complete else None,
                    "q_weighted_pnl_usd": q_ref * yes_point + (1 - q_ref) * no_point if complete else None,
                    "usd_per_one_percent_market_move_through_betas": beta_dollars * .01,
                    "usd_per_one_bp_long_yield_rise": bond_usd_per_bp if bond_weight else None,
                    "usd_per_one_percent_nvda_addon": notional * nvda_total * .01,
                    "bond_return_that_offsets_equity_contrast": None if bond_weight <= 0 else -equity_dollars / (notional * bond_weight),
                    "nvda_weight_direct_and_through_funds": nvda_total,
                    "issuer_channel_weights": {c["issuer"]: c["portfolio_weight"] for c in channels},
                    "usd_per_one_percent_issuer_residual": {c["issuer"]: c["usd_per_one_percent_return"] for c in channels},
                    "market_beta_only_symbols": beta_only,
                    # Every position is known and none moves in any draw: the scenario's
                    # channels (for example NVIDIA's China sales) miss this portfolio.
                    "no_portfolio_channel": bool(complete and not np.any(delta_draw))},
        "uncertainty": {"method": "SEEDED_MONTE_CARLO_UNIFORM_DECLARED_RANGES", "draws": n_draws, "seed": seed,
                        "includes": ["declared market, reviewed issuer residual, gold replacement and long-yield ranges", "daily betas (normal, HC3 standard error)",
                                     "bond duration coverage", "every Q between the contract bid and ask"],
                        "interpretation": "Scenario-conditional 5-95% range; the declared ranges, not data, set its width."},
        "physical_probability_P": None,
        "assumptions": [
            "The scenario describes exactly this contract's YES event under its full rules, not the broader risk.",
            "The contract price weights YES against NO; it does not set the shock sizes, which come from the declared profile.",
            "Asset prices are assumed to embed the contract's Q, so the Q-weighted sum of the two rows is zero.",
            "Each whole-position market beta is counted once. Reviewed NVDA/AAPL/AMD/TSM/MU residuals use only direct or dated identified fund exposure; no sector weight or unresolved NAV is invented.",
            "Issuer filings establish the economic link, not the assumed stock loss. Channel points and ranges are scenario choices, not causal estimates or calibrated loss bounds.",
            "The same issuer shock is shared across direct and fund holdings. Supply residuals share one severity rank; gold, whose sign the measured episodes do not identify, is drawn independently of it. This dependence is declared, not estimated.",
            "GLD uses a sign-uncertain whole-gold scenario that replaces its ordinary-market beta. A gain is not guaranteed. Unreviewed positions retain market beta only.",
            "Daily betas estimated in ordinary markets are assumed to hold in the stressed window.",
            "Overlapping contract scenarios (for example a blockade and an invasion) must not be added.",
        ],
    }


def contract_first_review(portfolio, exposure_report, daily_beta_report, target, *, rate_sensitivity=None,
                          holdings_quality=None, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED,
                          instrument_exposures=None, identity_blocked_symbols=None):
    """Classify every searched contract, quantify admitted ones, keep the broad risk's own card."""
    reviews, scenarios, bounds = [], [], []
    for candidate in target.get("candidates", []):
        review = review_candidate(candidate, target)
        review["_rules_text"] = candidate.get("rules_text", "")
        if review["decision"].startswith("ADMITTED"):
            bound = implication_bound(review, target)
            review["implication_to_broad_risk"] = bound
            if bound["status"] == "CONDITIONAL_LOWER_BOUND_FROM_IMPLYING_CONTRACT":
                bounds.append(bound)
        if review["decision"] == "ADMITTED_CONTRACT_SCENARIO":
            try:
                scenarios.append(build_contract_scenario(portfolio, exposure_report, daily_beta_report, review,
                                                         rate_sensitivity=rate_sensitivity, holdings_quality=holdings_quality,
                                                         n_draws=n_draws, seed=seed,
                                                         instrument_exposures=instrument_exposures,
                                                         identity_blocked_symbols=identity_blocked_symbols))
            except (ValueError, KeyError, TypeError) as exc:
                review.update(decision="ADMITTED_SCENARIO_UNAVAILABLE", reason=str(exc))
        review.pop("_rules_text", None)
        reviews.append(review)
    # A partial scenario has no full-portfolio ranking. Keep it inspectable
    # after the ranked complete scenarios, without turning None into a risk 0.
    scenarios.sort(key=lambda s: (s["summary"]["q_standard_deviation_usd"] is None,
                                  -(s["summary"]["q_standard_deviation_usd"] or 0.)))
    by_venue = {}
    for bound in bounds:
        if bound["venue"] not in by_venue or bound["conditional_lower_bound"] > by_venue[bound["venue"]]["conditional_lower_bound"]:
            by_venue[bound["venue"]] = bound
    return {
        "schema": SCHEMA,
        "status": "QUANTIFIED_CONTRACT_SCENARIOS" if scenarios else "NO_ADMITTED_QUANTIFIED_CONTRACT",
        "candidate_reviews": reviews, "contract_scenarios": scenarios,
        "broad_risk": {
            "definition": target.get("question_we_care_about"), "horizon_end": horizon_end(target).isoformat(),
            "probability_definition": REVIEWED_RISKS.get(target.get("family")),
            "probability_definition_origin": "REVIEWED_FAMILY_DEFINITION_NOT_MODEL_TEXT",
            "probability": {"status": "CONDITIONAL_LOWER_BOUND_ONLY" if by_venue else "UNKNOWN",
                            "lower_bounds_by_venue": [],
                            "conditional_lower_bounds_by_venue": list(by_venue.values()),
                            "upper_bound": None,
                            "scope": "Each bound holds under that venue's pricing measure; venues are not combined."},
            "kept_visible": ("The broad risk stays listed with its own channel sensitivity. Contract scenarios quantify only "
                             "their own events; outcomes outside every admitted contract remain unpriced and unquantified."),
        },
        "venue_bounds_combined": False,
        "new_provider_or_model_calls": 0,
    }
