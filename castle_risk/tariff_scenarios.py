"""US tariffs on China: contracts classified from their rules, a coherent bracket distribution, measured anchors.

The portfolio decides whether tariffs are searched: a held issuer's annual report documents the
channel (Apple: manufacturing partners primarily in China mainland; new US tariffs in 2025), or the
model proposes a tariff risk. Every searched contract is classified from its own rules and question:
a bracket on the general US tariff rate on Chinese imports is admitted; tariffs on another country or
an unrelated product, tariff dividends, court rulings and trade deals are different events and are
rejected with the reason.

When the admitted brackets partition the rate and their quotes admit a coherent distribution, each
bracket is an outcome with a pricing weight Q, like a CPI band, and its P&L is the move not already in
prices:

    PnL_b = sum_i amount_i * s_i * (r_b - E_Q[r])

r_b is the bracket's rate (its midpoint; the open top bracket uses a declared representative) and s_i
is position i's own close-to-close return per percentage point of the general China rate on two dated
announcements (ANCHORS):

  * 12 May 2025, Geneva joint statement: -115 points (145% to 30%), China only. This is the point.
  * 3 April 2025, first session after the reciprocal tariffs: +34 points on China, announced together
    with tariffs on every other partner, so its per-point move is an upper envelope. The range covers
    both anchors for every position together, with no distribution assumed between them.

Asset prices are assumed to embed Q, so the Q-weighted P&L is zero. No network, clock read or model
call happens here.

An "at least X%" ladder on the same general rate (Kalshi measures it on 1 January) is not merged with
the brackets: its date differs. It checks them. Where the two venues' quotes cannot both hold, the card
says so and bounds how far the not-yet-priced column moves if the ladder's Q is the one in prices.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, time, timezone
import json
import math
import re

import numpy as np

from .historical_chain_sources import project_box_simplex
from .outcome_table import combine_loadings
from .payroll_evidence import same_values
from .event_subject_intent import without_tariff_meta_exclusions

SCHEMA = "castle.option1.tariff_scenarios.v1"
SEARCH_FAMILY = "US_TARIFFS"
CARD_FAMILY = "TRADE_TARIFFS"
DEFAULT_DRAWS = 20000
DEFAULT_SEED = 20261024

_TARIFF = re.compile(r"\btariffs?\b|\bimport dut(?:y|ies)\b|\bsection (?:232|301)\b", re.I)
# A tariff proposal that names AI chips, export controls or export licensing stays with the export-control family.
_AI_CHIP_EXPORT = re.compile(r"\bai[- ]?chips?\b|\bgpus?\b|\bh20\b|\bh200\b|advanced (?:ai )?chips?|export[- _]?(?:licen|controls?\b)", re.I)


def is_tariff_risk(text):
    """Routing rule shared by the search and the default view."""
    text = without_tariff_meta_exclusions(str(text or ""))
    return bool(_TARIFF.search(text)) and not _AI_CHIP_EXPORT.search(text)


REVIEWED_RISK = ("A new or higher US tariff on imports from China takes effect: the general US tariff rate on Chinese "
                 "goods rises, or a product tariff covering consumer electronics or semiconductors from China takes "
                 "effect. The window runs from the evaluation date to the horizon end.")
REVIEWED_TITLE = "US tariffs on Chinese imports raised (general rate or electronics and semiconductors)"

# Dated announcements with a documented change in the general US tariff rate on China.
# Chosen for the announcement, before measuring any return; one clean, one confounded.
ANCHORS = (
    {"anchor_id": "GENEVA_2025_05_12", "session": "2025-05-12", "previous_session": "2025-05-09",
     "change_pp": -115.0, "role": "POINT_CHINA_ONLY",
     "event": "US-China joint statement in Geneva, released before the US open on 12 May 2025: the US suspended 24 "
              "points of the 34% reciprocal rate and removed the April increases, cutting the general rate on "
              "Chinese goods from 145% to 30% for 90 days.",
     "sources": [{"title": "White House, Joint Statement on U.S.-China Economic and Trade Meeting in Geneva (12 May 2025)",
                  "url": "https://www.whitehouse.gov/briefings-statements/2025/05/joint-statement-on-u-s-china-economic-and-trade-meeting-in-geneva/",
                  "locator": "\"by suspending 24 percentage points of that rate for an initial period of 90 days\""},
                 {"title": "Malay Mail (Reuters), 12 May 2025: US, China agree to slash tariffs by 115pc",
                  "url": "https://www.malaymail.com/news/money/2025/05/12/us-china-agree-to-slash-tariffs-by-115pc-pause-trade-war-for-90-days/176499",
                  "locator": "\"by 115 percentage points, taking US tariffs to 30 per cent\""}]},
    {"anchor_id": "RECIPROCAL_2025_04_03", "session": "2025-04-03", "previous_session": "2025-04-02",
     "change_pp": 34.0, "role": "ENVELOPE_GLOBAL_ANNOUNCEMENT",
     "event": "Reciprocal tariffs announced on 2 April 2025 after the close: 34% on China on top of the 20% already "
              "in force, together with tariffs on every other trading partner. The China share of the move is not "
              "identified, so its per-point response bounds the China-only effect from above.",
     "sources": [{"title": "Brownstein client alert, 3 April 2025: Liberation Day, Trump announces reciprocal tariffs",
                  "url": "https://www.bhfs.com/insights/alerts-articles/2025/-liberation-day-trump-announces-reciprocal-tariffs-ends-de-minimis-treatment-for-china",
                  "locator": "China 34%; \"the minimum duty rate applied to Chinese imports on April 9 will be 54%\""}]},
)
# Every dated change in the general rate on Chinese goods from 2025 to the capture, with why it is or
# is not an anchor. Compiled after the anchors were chosen, from the announcements only; no return on
# these sessions was measured to compile it. Threats that changed no rate in force are not listed.
CONSIDERED_SESSIONS = (
    {"session": "2025-02-03", "change": "+10 points (0% to 10%)", "china_only": False, "use": "NOT_USED",
     "reason": "The orders signed on 1 February also put 25% on Canada and Mexico, so the China share of the move is not identified.",
     "source": {"title": "Cleary Gottlieb, President Trump Imposes Tariffs on Canada and Mexico, Additional Tariffs on China",
                "url": "https://www.clearygottlieb.com/news-and-insights/publication-listing/president-trump-imposes-tariffs-on-canada-and-mexico-additional-tariffs-on-china"}},
    {"session": "2025-03-04", "change": "+10 points (10% to 20%)", "china_only": False, "use": "NOT_USED",
     "reason": "Ordered on 3 March after the close; the 25% Canada and Mexico tariffs took effect the same day.",
     "source": {"title": "Executive Order 14228 of March 3, 2025 (Federal Register)",
                "url": "https://www.federalregister.gov/documents/full_text/html/2025/03/07/2025-03775.html"}},
    {"session": "2025-04-03", "change": "+34 points", "china_only": False, "use": "ENVELOPE",
     "reason": "Announced with tariffs on every other partner; its per-point move bounds the China-only effect from above.",
     "source": {"title": ANCHORS[1]["sources"][0]["title"], "url": ANCHORS[1]["sources"][0]["url"]}},
    {"session": "2025-04-08", "change": "+50 points (reciprocal 34% to 84%)", "china_only": True, "use": "NOT_USED",
     "reason": "Threatened on 7 April with an 8 April deadline and ordered on 8 April: the change is spread over two sessions.",
     "source": {"title": "Axios, 7 April 2025: Trump threatens extra 50% tariff on China",
                "url": "https://www.axios.com/2025/04/07/trump-tariffs-china"}},
    {"session": "2025-04-09", "change": "+41 points (reciprocal 84% to 125%)", "china_only": False, "use": "NOT_USED",
     "reason": "The same order paused the reciprocal tariffs of every other partner at 10%, a large change the other way.",
     "source": {"title": "Wiley, Trump Reduces Global Reciprocal Tariffs but Increases Them for China",
                "url": "https://www.wiley.law/alert-Trump-Reduces-Global-Reciprocal-Tariffs-but-Increases-Them-for-China"}},
    {"session": "2025-05-12", "change": "-115 points (145% to 30%)", "china_only": True, "use": "POINT",
     "reason": "China only, released before the open, the largest change: it sets the per-point response.",
     "source": {"title": ANCHORS[0]["sources"][0]["title"], "url": ANCHORS[0]["sources"][0]["url"]}},
    {"session": "2025-10-30", "change": "-10 points (fentanyl duty 20% to 10%)", "china_only": True, "use": "NOT_USED",
     "reason": "China only, announced after the Busan meeting during the US night, ordered on 4 November. At the Geneva "
               "per-point response it moves each position by under a tenth of its 12 May 2025 return, within an ordinary day's move.",
     "source": {"title": "White House, Modifying Duties Addressing the Synthetic Opioid Supply Chain in the PRC (4 November 2025)",
                "url": "https://www.whitehouse.gov/presidential-actions/2025/11/modifying-duties-addressing-the-synthetic-opioid-supply-chain-in-the-peoples-republic-of-china/"}},
    {"session": "2026-02-20", "change": "IEEPA duties void; a global Section 122 tariff replaces them from 24 February", "china_only": False,
     "use": "NOT_USED", "reason": "The Supreme Court ruling and its replacement apply to every partner.",
     "source": {"title": "Thomson Reuters, Supreme Court rejects IEEPA as a tariff statute (20 February 2026)",
                "url": "https://www.thomsonreuters.com/en-us/help/onesource-global-trade/regulatory-insights/2026/february-20th/supreme-court-rejects-ieepa-as-a-tariff-statute"}},
    {"session": "2026-07-24", "change": "Section 122 expires; Section 301 duties of 10% or 12.5% on 60 economies", "china_only": False,
     "use": "NOT_USED", "reason": "Applies to 60 economies at once, and changes the China rate by a few points.",
     "source": {"title": "Penn Wharton Budget Model, Effective Tariff Rates and Revenues (updated 14 October 2026)",
                "url": "https://budgetmodel.wharton.upenn.edu/p/2026-10-14-effective-tariff-rates-and-revenues-updated-october-14-2026/"}},
)

# The open top bracket has no upper bound; 145% is the highest general rate in force in 2025.
OPEN_TOP_REPRESENTATIVE = {"point_offset_pp": 10.0, "upper_pp": 145.0,
                           "basis": "DECLARED_REPRESENTATIVE_NEXT_10_POINT_STEP_RANGE_TO_2025_PEAK"}

_BRACKET_RULES = re.compile(r"resolve according to the general tariff rate on imports into the united states from "
                            r"the people(?:'|\u2019)?s republic of china on ([a-z]+ \d{1,2}, \d{4})", re.I)
_BRACKET_QUESTION = (
    (re.compile(r"\bbe less than (\d+(?:\.\d+)?)% on\b", re.I), lambda m: (0.0, float(m.group(1)))),
    (re.compile(r"\bbe between (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)% on\b", re.I),
     lambda m: (float(m.group(1)), float(m.group(2)))),
    (re.compile(r"\bbe (\d+(?:\.\d+)?)% or (?:greater|more|higher) on\b", re.I), lambda m: (float(m.group(1)), None)),
)
_REQUIRED_BRACKET_CLAUSES = (
    ("ONLY_TARIFFS_IN_EFFECT", r"only tariffs which are in effect will qualify"),
    ("ITEM_SPECIFIC_EXCLUDED", r"item specific exceptions or increases will not be considered"),
    ("TIES_TO_HIGHER_BRACKET", r"falls exactly between two brackets.{0,80}higher range bracket"),
)
_THRESHOLD_RULES = re.compile(r"general import tariff rate on imports from (?:the )?people(?:'|\u2019)?s republic of "
                              r"china into the united states on ([a-z]+ \d{1,2}, \d{4}) is at least (\d+(?:\.\d+)?)%", re.I)
_REQUIRED_THRESHOLD_CLAUSES = (
    ("ONLY_TARIFFS_IN_EFFECT", r"only tariffs that are currently active and being collected count"),
    ("PRODUCT_SPECIFIC_EXCLUDED", r"not including product-specific duties"),
)
_COUNTRIES = r"\b(?:canada|canadian|mexico|mexican|european union|\beu\b|india|japan|brazil|switzerland|vietnam|korea|united kingdom|uk)\b"
REJECTIONS = (
    ("TARIFF_REVENUE_TRANSFER", r"\bdividend\b|\brebate\b|\brefund",
     "A tariff dividend, rebate or refund is a fiscal transfer, not a tariff rate on Chinese goods."),
    ("COURT_RULING", r"supreme court|court of international trade|\bscotus\b|\bruling\b",
     "A court ruling changes legal authority; it is not itself a tariff rate in effect on Chinese goods."),
    ("TRADE_AGREEMENT", r"trade deal|trade agreement|tariff agreement|\baccepts? a\b",
     "A trade agreement is a negotiation outcome; it does not fix the tariff rate on Chinese goods."),
    ("DIFFERENT_PRODUCT_TARIFF", r"\bcopper\b|\bsteel\b|\balumin(?:i)?um\b|\blumber\b|\btimber\b|pharmaceutical|\btrucks?\b|furniture|\bautos?\b",
     "A duty on this product does not cover consumer electronics or semiconductors, the documented channel."),
)


def _quote(candidate):
    quote = (candidate.get("contract_quote_reference") or {}).get("interval")
    if (isinstance(quote, (list, tuple)) and len(quote) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in quote)
            and 0 <= quote[0] <= quote[1] <= 1):
        return [float(quote[0]), float(quote[1])]
    return None


def classify_candidate(candidate, horizon_end):
    """Rules first: the bracket grammar of the general China rate, then reviewed rejections."""
    rules = str(candidate.get("rules_text") or "")
    question = str(candidate.get("market_question") or "")
    row = {"venue": candidate.get("venue"), "contract_id": candidate.get("contract_id"),
           "market_question": question, "rules_sha256": candidate.get("rules_sha256"),
           "contract_end_time": candidate.get("contract_end_time"), "source_url": candidate.get("source_url"),
           "horizon_end": horizon_end.isoformat(), "quote_interval": _quote(candidate),
           "quote_scope": (candidate.get("contract_quote_reference") or {}).get("scope"),
           "basis": "RULES_AND_QUESTION_PREDICATES"}
    match = _BRACKET_RULES.search(rules)
    if match:
        missing = [name for name, pattern in _REQUIRED_BRACKET_CLAUSES if not re.search(pattern, rules, re.I | re.S)]
        bounds = None
        for pattern, reader in _BRACKET_QUESTION:
            found = pattern.search(question)
            if found:
                bounds = reader(found)
                break
        try:
            measured = datetime.strptime(match.group(1).title(), "%B %d, %Y").date()
        except ValueError:
            measured = None
        stated = re.search(r"\bon ([A-Z][a-z]+ \d{1,2})\??$", question.strip())
        if bounds is None or measured is None or missing:
            row.update(event_class="UNREVIEWED_BRACKET_WORDING", admitted_class=False,
                       decision="REJECTED_DIFFERENT_EVENT",
                       reason="The rules name the general China rate, but the bracket or a required settlement clause "
                              "could not be read: " + ", ".join(missing or ["bracket bounds or measurement date"]) + ".")
            return row
        row.update(event_class="US_GENERAL_TARIFF_RATE_ON_CHINA_BRACKET", admitted_class=True,
                   bracket_pp=[bounds[0], bounds[1]], measurement_date=measured.isoformat(),
                   rules_evidence=[match.group(0)] + [name for name, _ in _REQUIRED_BRACKET_CLAUSES],
                   question_date_text=stated.group(1) if stated else None)
        if datetime.combine(measured, time(12), tzinfo=timezone.utc) > horizon_end:
            row.update(decision="REJECTED_DEADLINE_AFTER_HORIZON",
                       reason="The rate is measured after the risk's horizon, so the bracket prices a later state.")
        elif not candidate.get("market_open_at_evaluation") or row["quote_interval"] is None:
            row.update(decision="REJECTED_NOT_OPEN_OR_NOT_QUOTED",
                       reason="No valid current quote at the evaluation time: " + ", ".join(
                           (candidate.get("contract_quote_reference") or {}).get("reasons", [])))
        else:
            row.update(decision="ADMITTED_OUTCOME_FAMILY_MEMBER",
                       reason="An exact bracket of the general US tariff rate on Chinese goods on the stated date.")
        return row
    match = _THRESHOLD_RULES.search(rules)
    if match:
        missing = [name for name, pattern in _REQUIRED_THRESHOLD_CLAUSES if not re.search(pattern, rules, re.I | re.S)]
        measured = None
        for layout in ("%B %d, %Y", "%b %d, %Y"):  # Kalshi writes "Jan 1, 2027"
            try:
                measured = datetime.strptime(match.group(1).title(), layout).date()
                break
            except ValueError:
                continue
        if measured is None or missing:
            row.update(event_class="UNREVIEWED_THRESHOLD_WORDING", admitted_class=False,
                       decision="REJECTED_DIFFERENT_EVENT",
                       reason="The rules name the general China rate, but a required settlement clause or the date "
                              "could not be read: " + ", ".join(missing or ["measurement date"]) + ".")
            return row
        row.update(event_class="US_GENERAL_TARIFF_RATE_ON_CHINA_THRESHOLD", admitted_class=False,
                   threshold_pp=float(match.group(2)), measurement_date=measured.isoformat(),
                   rules_evidence=[match.group(0)] + [name for name, _ in _REQUIRED_THRESHOLD_CLAUSES])
        if datetime.combine(measured, time(12), tzinfo=timezone.utc) > horizon_end:
            row.update(decision="REJECTED_DEADLINE_AFTER_HORIZON",
                       reason="The rate is measured after the risk's horizon, so the threshold prices a later state.")
        elif not candidate.get("market_open_at_evaluation") or row["quote_interval"] is None:
            row.update(decision="REJECTED_NOT_OPEN_OR_NOT_QUOTED",
                       reason="No valid current quote at the evaluation time: " + ", ".join(
                           (candidate.get("contract_quote_reference") or {}).get("reasons", [])))
        else:
            row.update(decision="CROSS_CHECK_SAME_RATE_THRESHOLD",
                       reason=f"The same general China rate, as 'at least {float(match.group(2)):g}%' on {measured.isoformat()}: "
                              "a different date and form from the brackets, so it checks their Q and is not merged with them.")
        return row
    text = (question + "\n" + rules).lower()
    if re.search(_COUNTRIES, text) and not re.search(r"\bchina\b|\bchinese\b|people(?:'|\u2019)?s republic", text):
        row.update(event_class="DIFFERENT_COUNTRY_TARIFF", admitted_class=False, decision="REJECTED_DIFFERENT_EVENT",
                   reason="Tariffs on another country are a different event; the documented channel is manufacturing in China.")
        return row
    for name, pattern, reason in REJECTIONS:
        if re.search(pattern, text):
            row.update(event_class=name, admitted_class=False, decision="REJECTED_DIFFERENT_EVENT", reason=reason)
            return row
    row.update(event_class="UNREVIEWED_EVENT_CLASS", admitted_class=False, decision="REJECTED_DIFFERENT_EVENT",
               reason="The rules match no reviewed tariff grammar (a bracket of the general US rate on Chinese goods).")
    return row


def _bracket_label(low, high, day):
    when = datetime.fromisoformat(day).strftime("%d %b %Y").lstrip("0")
    if low == 0:
        return f"General US tariff on China below {high:g}% on {when}"
    if high is None:
        return f"General US tariff on China {low:g}% or higher on {when}"
    return f"General US tariff on China {low:g}% to {high:g}% on {when}"


def bracket_distribution(reviews):
    """One coherent distribution per complete bracket family; incomplete or incoherent families stay unknown."""
    groups = {}
    for review in reviews:
        if review.get("decision") == "ADMITTED_OUTCOME_FAMILY_MEMBER":
            groups.setdefault((review["venue"], review["measurement_date"], review.get("source_url")), []).append(review)
    results = []
    for (venue, day, link), members in sorted(groups.items(), key=lambda item: (item[0][1], str(item[0]))):
        members = sorted(members, key=lambda r: r["bracket_pp"][0])
        lows = [m["bracket_pp"][0] for m in members]
        highs = [m["bracket_pp"][1] for m in members]
        reason = None
        if lows[0] != 0 or highs[-1] is not None or any(h is None for h in highs[:-1]):
            reason = "The admitted brackets do not cover the rate from 0% to an open top."
        elif any(h != lo for h, lo in zip(highs, lows[1:])):
            reason = "The admitted brackets leave a gap or overlap."
        lower = [m["quote_interval"][0] for m in members]
        upper = [m["quote_interval"][1] for m in members]
        family = {"venue": venue, "measurement_date": day, "event_link": link,
                  "contract_ids": [m["contract_id"] for m in members], "status": None}
        if reason is None and (math.fsum(lower) > 1 + 1e-12 or math.fsum(upper) < 1 - 1e-12):
            reason = "The quotes admit no distribution: the bids sum above 1 or the asks below 1."
        if reason is not None:
            family.update(status="NO_COHERENT_BRACKET_DISTRIBUTION", reason=reason)
            results.append(family)
            continue
        q = project_box_simplex([(lo + hi) / 2 for lo, hi in zip(lower, upper)], lower, upper)
        rest_lo, rest_hi = math.fsum(lower), math.fsum(upper)
        atoms = []
        for member, lo, hi, qk in zip(members, lower, upper, q):
            low, high = member["bracket_pp"]
            if high is None:
                point = low + OPEN_TOP_REPRESENTATIVE["point_offset_pp"]
                rate_range = [low, max(OPEN_TOP_REPRESENTATIVE["upper_pp"], point)]
                basis = OPEN_TOP_REPRESENTATIVE["basis"]
            else:
                point, rate_range, basis = (low + high) / 2, [low, high], "BRACKET_MIDPOINT_UNIFORM_WITHIN_BRACKET"
            atoms.append({"outcome_id": member["contract_id"], "label": _bracket_label(low, high, day),
                          "bracket_pp": [low, high], "rate_representative_pp": point, "rate_range_pp": rate_range,
                          "representative_basis": basis, "q_reference": qk,
                          "q_interval": [max(lo, 1 - (rest_hi - hi)), min(hi, 1 - (rest_lo - lo))],
                          "quote_interval": [lo, hi]})
        family.update(status="COHERENT_BRACKET_DISTRIBUTION", atoms=atoms,
                      q_method="EUCLIDEAN_PROJECTION_OF_MIDPOINTS_ONTO_THE_QUOTE_BOX_AND_SIMPLEX",
                      quote_scope=members[0].get("quote_scope"))
        results.append(family)
    return results


def _mean_range(atoms):
    """Smallest and largest Q-expected rate over the quote box intersected with the simplex."""
    def extreme(descending):
        q = [a["quote_interval"][0] for a in atoms]
        left = 1 - math.fsum(q)
        for k in sorted(range(len(atoms)), key=lambda k: atoms[k]["rate_representative_pp"], reverse=descending):
            step = min(atoms[k]["quote_interval"][1] - q[k], left)
            q[k] += step
            left -= step
        return math.fsum(qk * a["rate_representative_pp"] for qk, a in zip(q, atoms))
    return [extreme(False), extreme(True)]


def threshold_ladders(reviews):
    """'At least X%' ladders on the general China rate, one per venue and date, with their survival intervals."""
    groups = {}
    for review in reviews:
        if review.get("decision") == "CROSS_CHECK_SAME_RATE_THRESHOLD":
            groups.setdefault((review["venue"], review["measurement_date"]), []).append(review)
    ladders = []
    for (venue, day), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r["threshold_pp"])
        thresholds = [m["threshold_pp"] for m in members]
        ladder = {"venue": venue, "measurement_date": day, "contract_ids": [m["contract_id"] for m in members]}
        if len(set(thresholds)) != len(thresholds):
            ladders.append({**ladder, "status": "NO_COHERENT_THRESHOLD_LADDER", "reason": "Two contracts share a threshold."})
            continue
        bids = [m["quote_interval"][0] for m in members]
        asks = [m["quote_interval"][1] for m in members]
        # P(rate >= t) cannot rise with t: a higher threshold's bid above a lower one's ask admits no distribution.
        lower = [max(bids[k:]) for k in range(len(bids))]
        upper = [min(asks[:k + 1]) for k in range(len(asks))]
        if any(lo > hi + 1e-12 for lo, hi in zip(lower, upper)):
            ladders.append({**ladder, "status": "NO_COHERENT_THRESHOLD_LADDER",
                            "reason": "A higher threshold is bid above a lower threshold's ask."})
            continue
        ladders.append({**ladder, "status": "COHERENT_THRESHOLD_LADDER",
                        "thresholds": [{"threshold_pp": t, "contract_id": m["contract_id"], "quote_interval": list(m["quote_interval"]),
                                        "survival_interval": [lo, hi]}
                                       for t, m, lo, hi in zip(thresholds, members, lower, upper)]})
    return ladders


def _bracket_survival(atoms, threshold):
    """Every P(rate >= threshold) the bracket quotes allow (box intersected with the simplex; ties go up)."""
    above = [a for a in atoms if a["bracket_pp"][0] >= threshold]
    below = [a for a in atoms if a["bracket_pp"][1] is not None and a["bracket_pp"][1] <= threshold]
    straddle = [a for a in atoms if a not in above and a not in below]
    total = lambda rows, k: math.fsum(a["quote_interval"][k] for a in rows)
    return [max(total(above, 0), 1 - total(below + straddle, 1)), min(total(above + straddle, 1), 1 - total(below, 0))]


def _ladder_mean_range(thresholds, top):
    """Smallest and largest E[rate] = integral of P(rate >= x) dx, for a non-increasing survival inside the intervals."""
    edges = [0.0] + [t["threshold_pp"] for t in thresholds]
    low = math.fsum((b - a) * t["survival_interval"][0] for a, b, t in zip(edges, edges[1:], thresholds))
    upper_left = [1.0] + [t["survival_interval"][1] for t in thresholds]
    high = math.fsum((b - a) * s for a, b, s in zip(edges, edges[1:], upper_left))
    return [low, high + (top - edges[-1]) * thresholds[-1]["survival_interval"][1]]


def cross_venue_check(distribution, ladders, table):
    """Compare a bracket family with each threshold ladder on the same rate; bound the shift in the P&L column."""
    checks = []
    for ladder in ladders:
        if ladder["status"] != "COHERENT_THRESHOLD_LADDER":
            checks.append({"venue": ladder["venue"], "measurement_date": ladder["measurement_date"],
                           "status": "LADDER_NOT_USABLE", "reason": ladder["reason"]})
            continue
        rows = []
        for t in ladder["thresholds"]:
            allowed = _bracket_survival(distribution["atoms"], t["threshold_pp"])
            rows.append({"threshold_pp": t["threshold_pp"], "ladder_interval": t["survival_interval"],
                         "bracket_family_interval": allowed,
                         "compatible": bool(allowed[0] <= t["survival_interval"][1] + 1e-12
                                            and t["survival_interval"][0] <= allowed[1] + 1e-12)})
        gap = (datetime.fromisoformat(ladder["measurement_date"]) - datetime.fromisoformat(distribution["measurement_date"])).days
        same_date = gap == 0
        quote_intervals_overlap = all(r["compatible"] for r in rows)
        if same_date:
            status = "COMPATIBLE" if quote_intervals_overlap else "INCOMPATIBLE_QUOTES"
        else:
            status = ("DATED_CROSS_CHECK_CONSISTENT_WITH_UNCHANGED_RATE" if quote_intervals_overlap
                      else "DATED_CROSS_CHECK_REQUIRES_RATE_CHANGE_OR_DIFFERENT_Q")
        check = {"venue": ladder["venue"], "measurement_date": ladder["measurement_date"],
                 "bracket_venue": distribution["venue"], "bracket_measurement_date": distribution["measurement_date"],
                 "days_between_measurements": gap, "rows": rows,
                 "status": status, "same_settlement_date": same_date,
                 "assumed_rate_unchanged_between_measurements": None if same_date else True,
                 "q_transfer_between_venues_proven": False,
                 "conditional_overlap_under_no_rate_change": quote_intervals_overlap,
                 "comparison_scope": ("SAME_SETTLEMENT_DATE_QUOTE_INTERVAL_OVERLAP_ONLY" if same_date else
                                      "COUNTERFACTUAL_SAME_RATE_AND_TRANSFERABLE_Q_NOT_A_SAME_EVENT_TEST"),
                 "ladder_mean_range_pp": _ladder_mean_range(ladder["thresholds"], OPEN_TOP_REPRESENTATIVE["upper_pp"]),
                 "mean_basis": "Integral of the survival bounds; rate capped at the declared 145% top.",
                 "physical_probability_P": None}
        if table is not None:
            reference = table["reference_expectation"]
            summary = table["summary"]
            # Row P&L is slope * (rate - E_Q); replacing E_Q shifts every row by slope * (E_brackets - E_ladder).
            for key, slope in (("pnl_shift_usd_range", summary["usd_per_point_rate_surprise"]),
                               ("pnl_shift_usd_range_envelope", summary["usd_per_point_rate_surprise_envelope"])):
                ends = [slope * (reference - mean) for mean in check["ladder_mean_range_pp"]]
                check[key] = [min(ends), max(ends)]
        checks.append(check)
    return checks


def _session_returns(rows, session, previous):
    closes = {}
    for row in rows or []:
        try:
            closes[str(row["date"])[:10]] = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
    if closes.get(previous, 0) > 0 and closes.get(session, 0) > 0:
        earlier = [d for d in closes if previous < d < session]
        if not earlier:
            return closes[session] / closes[previous] - 1
    return None


def measure_anchors(prices_by_symbol, symbols):
    """Close-to-close return of each symbol on each anchor session, from the run's own acquired prices."""
    result = {}
    for symbol in symbols:
        rows = (prices_by_symbol or {}).get(symbol)
        result[symbol] = {a["anchor_id"]: _session_returns(rows, a["session"], a["previous_session"]) for a in ANCHORS}
    return result


def build_tariff_table(portfolio, family, prices_by_symbol, daily_beta_report, *, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """Not-yet-priced P&L for every bracket; measured per-point responses at both anchors."""
    atoms = family["atoms"]
    point_anchor, envelope_anchor = ANCHORS
    symbols = sorted({p.symbol for p in portfolio.positions if p.asset_type != "cash"} | {"SPY"})
    measured = measure_anchors(prices_by_symbol, symbols)
    spy = measured.get("SPY", {})
    daily = (daily_beta_report or {}).get("instruments", {})
    positions, missing = [], []
    for position in portfolio.positions:
        row = {"symbol": position.symbol, "weight": position.weight}
        if position.asset_type == "cash" or position.weight == 0:
            positions.append({**row, "channel": "NONE", "loading": 0.0, "loading_envelope": 0.0,
                              "basis": "UNCHANGED_USD_CASH_OR_ZERO_WEIGHT"})
            continue
        own = measured.get(position.symbol, {})
        a, b = own.get(point_anchor["anchor_id"]), own.get(envelope_anchor["anchor_id"])
        if a is not None and b is not None:
            positions.append({**row, "channel": "MEASURED_ANCHOR", "loading": a / point_anchor["change_pp"],
                              "loading_envelope": b / envelope_anchor["change_pp"],
                              "anchor_returns": {point_anchor["anchor_id"]: a, envelope_anchor["anchor_id"]: b},
                              "basis": "OWN_CLOSE_TO_CLOSE_RETURN_ON_BOTH_ANCHOR_SESSIONS"})
            continue
        fit = combine_loadings(daily.get(position.symbol), None)
        sa, sb = spy.get(point_anchor["anchor_id"]), spy.get(envelope_anchor["anchor_id"])
        if fit is not None and sa is not None and sb is not None:
            positions.append({**row, "channel": "BETA_TRANSFER", "loading": fit["loading"] * sa / point_anchor["change_pp"],
                              "loading_envelope": fit["loading"] * sb / envelope_anchor["change_pp"], "se": fit["se"],
                              "basis": "DAILY_BETA_TIMES_SPY_ANCHOR_RESPONSE_NO_OWN_PRICE_ON_AN_ANCHOR_SESSION"})
            continue
        missing.append(position.symbol)
        positions.append({**row, "channel": "UNKNOWN", "loading": None, "loading_envelope": None,
                          "basis": "NO_ANCHOR_PRICE_AND_NO_DAILY_BETA"})
    known = [p for p in positions if p["loading"] is not None]
    notional = portfolio.notional
    per_pp = math.fsum(notional * p["weight"] * p["loading"] for p in known)
    per_pp_envelope = math.fsum(notional * p["weight"] * p["loading_envelope"] for p in known)
    q = np.array([a["q_reference"] for a in atoms])
    reps = np.array([a["rate_representative_pp"] for a in atoms])
    mean_point = float(q @ reps)
    mean_low, mean_high = _mean_range(atoms)
    rng = np.random.default_rng(seed)
    # No distribution is put on the response between the anchors: every draw is
    # evaluated at both, and at both ends of the Q-expected rate the quotes allow.
    slopes = (per_pp, per_pp_envelope)
    rows = []
    pnl_point = per_pp * (reps - mean_point)
    for k, atom in enumerate(atoms):
        lo_r, hi_r = atom["rate_range_pp"]
        rate = rng.uniform(lo_r, hi_r, n_draws)
        corners = np.stack([slope * (rate - mean) for slope in slopes for mean in (mean_low, mean_high)])
        low, high = float(np.percentile(corners.min(axis=0), 5)), float(np.percentile(corners.max(axis=0), 95))
        point = float(pnl_point[k])
        at_ref = np.stack([slope * (rate - mean_point) for slope in slopes])
        sign = None if abs(point) < 1e-9 else float(np.mean(np.all(np.sign(at_ref) == np.sign(point), axis=0)))
        rows.append({"outcome_id": atom["outcome_id"], "label": atom["label"], "q_interval": list(atom["q_interval"]),
                     "q_reference": atom["q_reference"], "state": atom["rate_representative_pp"],
                     "state_interval": list(atom["rate_range_pp"]), "state_basis": atom["representative_basis"],
                     "surprise": atom["rate_representative_pp"] - mean_point,
                     "pnl_if_outcome_usd": point, "pnl_model_range_5_95_usd": [low, high],
                     "model_5_95_range_excludes_zero": bool(low > 0 or high < 0),
                     "sign_probability_under_model": sign, "q_weighted_contribution_usd": atom["q_reference"] * point,
                     "position_contributions": [{"symbol": p["symbol"], "channel": p["channel"],
                                                 "pnl_usd": None if p["loading"] is None else
                                                 notional * p["weight"] * p["loading"] * (atom["rate_representative_pp"] - mean_point)}
                                                for p in positions]})
    weighted = float(pnl_point @ q)
    if abs(weighted) > 1e-6 * max(1.0, notional):
        raise ValueError("Q-centering identity failed")
    material = [r for r in rows if r["q_reference"] >= .01]
    worst = min(material or rows, key=lambda r: r["pnl_if_outcome_usd"])
    best = max(material or rows, key=lambda r: r["pnl_if_outcome_usd"])
    likely = max(rows, key=lambda r: r["q_reference"])
    missing_weight = math.fsum(p.weight for p in portfolio.positions if p.symbol in missing)
    return {
        "schema": SCHEMA, "status": "COMPLETE_MEASURED_ANCHOR_MODEL" if not missing else "PARTIAL_KNOWN_POSITIONS_ONLY",
        "family": CARD_FAMILY, "event_date": family["measurement_date"], "data_quality_gaps": [],
        "notional_usd": notional, "missing_symbols": missing, "missing_weight": missing_weight,
        "reference_expectation": mean_point, "reference_expectation_range_over_q_polytope": [mean_low, mean_high],
        "state_unit": "general US tariff rate on Chinese goods, percent",
        "outcomes": rows,
        "summary": {
            "most_likely_outcome": {"label": likely["label"], "q_reference": likely["q_reference"],
                                    "pnl_if_outcome_usd": likely["pnl_if_outcome_usd"]},
            "largest_loss_outcome_reference_q_at_least_1pct": {"label": worst["label"], "q_reference": worst["q_reference"],
                                                               "pnl_if_outcome_usd": worst["pnl_if_outcome_usd"],
                                                               "pnl_model_range_5_95_usd": worst["pnl_model_range_5_95_usd"]},
            "largest_gain_outcome_reference_q_at_least_1pct": {"label": best["label"], "q_reference": best["q_reference"],
                                                               "pnl_if_outcome_usd": best["pnl_if_outcome_usd"],
                                                               "pnl_model_range_5_95_usd": best["pnl_model_range_5_95_usd"]},
            "q_standard_deviation_usd": float(math.sqrt(max(0.0, float((pnl_point ** 2) @ q)))),
            "q_weighted_pnl_usd": weighted,
            "usd_per_point_rate_surprise": per_pp, "usd_per_point_rate_surprise_envelope": per_pp_envelope,
            "ranking_basis": "Q-standard deviation of the not-yet-priced P&L across the rate brackets.",
        },
        "positions": positions,
        "factors": {"anchors": deepcopy(list(ANCHORS)), "open_top_representative": deepcopy(OPEN_TOP_REPRESENTATIVE),
                    "measured_anchor_returns": measured},
        "uncertainty": {"method": "HULL_OVER_BOTH_ANCHORS_AND_Q_POLYTOPE_PERCENTILES_OVER_RATE_IN_BRACKET",
                        "draws": n_draws, "seed": seed,
                        "includes": ["the per-point response at either anchor, applied to every position together (no distribution between them)",
                                     "the rate inside each bracket (uniform; the open top from its floor to 145%), 5th and 95th percentiles",
                                     "every Q-expected rate the quotes allow (the range's edges)"],
                        "excludes": ["other news on the anchor sessions beyond the stated confounding",
                                     "nonlinearity of the response in the rate level", "timing of the announcement"],
                        "interpretation": "Model-conditional envelope; the two dated anchors, not a fitted model, set its width.",
                        "sign_probability_scope": "Share of in-bracket rate draws, under the reference Q, with the row's sign at both anchors."},
        "physical_probability_P": None,
        "assumptions": [
            "The bracket quotes are the outcome weights embedded in asset prices, so the Q-weighted P&L is zero and each row is the move not yet in prices if that bracket settles.",
            "Each position moves by its own return per point of the general China rate on the two anchor sessions; the clean 12 May 2025 session sets the point and the 3 April 2025 session, confounded by tariffs on every partner, the far end.",
            "The response is linear in the rate surprise and the same at today's rate level as in 2025; the 145% to 30% cut and the 0% to 34% rise bracket that assumption from both sides.",
            "Item-specific tariffs, Chinese retaliation and the timing of announcements are outside the contract and outside this table.",
        ],
    }


def compare_anchor_calibrations(table):
    """Keep both *measured* tariff anchors visible under the same frozen Q.

    This is a two-specification stress comparison, NOT an independent causal
    estimate or a statistical prediction interval.  Both anchors can contain
    unrelated announcements and the April session is explicitly confounded.
    """
    if not isinstance(table, dict) or not isinstance(table.get("outcomes"), list):
        raise ValueError("A priced tariff outcome table is required")
    summary = table.get("summary") or {}
    geneva = summary.get("usd_per_point_rate_surprise")
    april = summary.get("usd_per_point_rate_surprise_envelope")
    mean = table.get("reference_expectation")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (geneva, april, mean)):
        raise ValueError("Both measured tariff anchors and the same Q expectation are required")
    outcomes, weighted = [], [0.0, 0.0]
    for row in table["outcomes"]:
        state, q = row.get("state"), row.get("q_reference")
        if (type(state) not in (int, float) or type(q) not in (int, float)
                or not math.isfinite(state) or not math.isfinite(q) or not 0 <= q <= 1):
            raise ValueError("Invalid tariff state or pricing weight")
        delta = state - mean
        pnl_a, pnl_b = geneva * delta, april * delta
        if not math.isclose(row.get("pnl_if_outcome_usd", math.nan), pnl_a, rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError("Recorded tariff point P&L disagrees with its Geneva anchor")
        weighted[0] += q * pnl_a
        weighted[1] += q * pnl_b
        scale = max(abs(pnl_a), abs(pnl_b))
        if scale < 1e-8:
            sign = "BOTH_NEAR_ZERO"
        elif pnl_a * pnl_b < -1e-12:
            sign = "DIRECTION_REVERSES_ACROSS_ANCHORS"
        elif min(abs(pnl_a), abs(pnl_b)) < 1e-8:
            sign = "DIRECTION_UNRESOLVED_ONE_ANCHOR_ZERO"
        else:
            sign = "DIRECTION_AGREES_CONDITIONALLY"
        outcomes.append({"outcome_id": row["outcome_id"], "label": row["label"],
                         "q_reference": float(q), "state_pp": float(state),
                         "pnl_geneva_usd": pnl_a, "pnl_april_usd": pnl_b,
                         "signed_difference_usd": pnl_b - pnl_a,
                         "anchor_hull_usd": [min(pnl_a, pnl_b), max(pnl_a, pnl_b)],
                         "relative_disagreement_to_larger_magnitude":
                             abs(pnl_a - pnl_b) / scale if scale >= 1e-8 else 0.0,
                         "direction_status": sign})
    if abs(math.fsum(row["q_reference"] for row in outcomes) - 1) > 1e-5:
        raise ValueError("Tariff Q is not a normalized outcome partition")
    if any(abs(v) > 1e-5 * max(1.0, table.get("notional_usd", 1)) for v in weighted):
        raise ValueError("One tariff calibration is not Q-centered")
    material = [r for r in outcomes if r["q_reference"] >= .01] or outcomes
    geneva_worst = min(material, key=lambda r: r["pnl_geneva_usd"])["outcome_id"]
    april_worst = min(material, key=lambda r: r["pnl_april_usd"])["outcome_id"]
    reversals = sum(r["direction_status"] == "DIRECTION_REVERSES_ACROSS_ANCHORS" for r in outcomes)
    return {
        "schema": "castle.option1.tariff_anchor_sensitivity.v1",
        "status": "TWO_HISTORICAL_SPECIFICATIONS_NOT_CAUSAL_IDENTIFICATION",
        "coverage_status": table.get("status"), "missing_symbols": list(table.get("missing_symbols") or []),
        "anchor_ids": [a["anchor_id"] for a in ANCHORS],
        "usd_per_point_rate_surprise": {ANCHORS[0]["anchor_id"]: geneva,
                                        ANCHORS[1]["anchor_id"]: april},
        "same_q_reference_expectation_pp": mean,
        "q_centering_residual_usd": weighted,
        "outcomes": outcomes, "direction_reversal_count": reversals,
        "worst_material_outcome_geneva": geneva_worst,
        "worst_material_outcome_april": april_worst,
        "worst_case_identity_changes": geneva_worst != april_worst,
        "largest_absolute_pnl_disagreement_usd": max(abs(r["signed_difference_usd"]) for r in outcomes),
        "limitations": [
            "A measured market move on an announcement session is not a causal China-tariff coefficient.",
            "April 2025 combines China tariffs with tariffs on many countries; the division by 34 is a stress specification, not an upper-confidence bound.",
            "The two calibrations use the same Q reference and no model is selected by subsequent observed returns.",
            "No physical event probability or total portfolio return is inferred.",
        ],
    }


def _horizon_end(target):
    day = datetime.fromisoformat(str(target["event_date"])).date()
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def _escalation_q(atoms, threshold=15.0):
    """Q of the exact contract event 'general rate at or above the threshold', over the quote box and simplex."""
    inside = [a for a in atoms if a["bracket_pp"][0] >= threshold]
    outside = [a for a in atoms if a["bracket_pp"][0] < threshold]
    if not inside or not outside:
        return None
    lo_in = math.fsum(a["quote_interval"][0] for a in inside)
    hi_in = math.fsum(a["quote_interval"][1] for a in inside)
    lo_out = math.fsum(a["quote_interval"][0] for a in outside)
    hi_out = math.fsum(a["quote_interval"][1] for a in outside)
    return {"threshold_pp": threshold, "q_reference": math.fsum(a["q_reference"] for a in inside),
            "q_interval": [max(lo_in, 1 - hi_out), min(hi_in, 1 - lo_out)]}


_SPECIFIC_SCOPE_TERM = r"(?:product|item|company|issuer|firm|sector)[- ]specific"
_PRODUCT_SCOPE = re.compile(
    r"\b(?:pharma(?:ceutical)?s?|drugs?|medicines?|medications?|active (?:pharmaceutical )?ingredients?"
    r"|copper|steel|alumin(?:i)?um|lumber|timber|trucks?|furniture|autos?|automobiles?"
    r"|electronics|semiconductors?|iphones?|solar|photovoltaic|batter(?:y|ies)"
    r"|electric vehicles?|" + _SPECIFIC_SCOPE_TERM + r")\b")
# Only an explicitly negative, standalone definition clause can remove a
# generic "sector-specific" / "product-specific" mention from the scope test.
# A negative word elsewhere ("not yet in effect", "not merely threatened")
# must never erase a positive product restriction. Concrete product names and
# unreviewed/ambiguous phrasings remain conservative exclusions.
_EXCLUDED_SPECIFIC_CLAUSE = re.compile(
    r"(?:^|[.;!?]\s*)(?:the\s+)?(?P<scope>" + _SPECIFIC_SCOPE_TERM + r")\s+"
    r"(?:tariffs?|dut(?:y|ies)|measures?|exceptions?|increases?)"
    r"(?:\s+(?:that|which)\s+[^.;!?]{1,140}?)?\s+"
    r"(?:(?:do not|don't|will not|won't)\s+(?:count|qualify)|are\s+(?:excluded|not included))"
    r"\s*(?=[.;!?]|$)")

# A common edited-risk wording puts the exclusion after a coordinated clause:
# "Sector-specific tariffs only, or threats without formal action, do not count."
# The isolated word "only" is not positive scope evidence here. Keep this
# grammar narrow: a positive or ambiguous product-specific condition still
# fails closed under the ordinary scope parser.
_EXCLUDED_SPECIFIC_ONLY = re.compile(
    r"(?:^|[.;!?]\s*)(?:the\s+)?(?P<scope>" + _SPECIFIC_SCOPE_TERM + r")\s+"
    r"(?:tariffs?|dut(?:y|ies)|measures?|increases?)\s+only"
    r"(?:\s*,\s*or\s+[^.;!?]{1,100})?\s*,?\s*"
    r"(?:do not|don't|will not|won't)\s+(?:count|qualify)\s*(?=[.;!?]|$)")


def _target_product_scope(text):
    """Return positive scope evidence, keeping explicit exclusion evidence.

    This narrow grammar is not a general natural-language entailment system.
    Ignoring an exclusion can only enable a *related* general-rate scenario;
    it never establishes equivalence of the target and a dated contract.
    """
    exclusions = list(_EXCLUDED_SPECIFIC_CLAUSE.finditer(text)) + list(
        _EXCLUDED_SPECIFIC_ONLY.finditer(text))
    excluded_spans = [match.span("scope") for match in exclusions]
    positive = [match.group(0) for match in _PRODUCT_SCOPE.finditer(text)
                if not any(start <= match.start() and match.end() <= end
                           for start, end in excluded_spans)]
    return positive, [match.group(0).lstrip(".;!? ") for match in exclusions]


def target_scope_review(target):
    """Keep a general-China rate scenario separate from a different tariff risk.

    This is a conservative presentation gate, not proof of event equivalence.
    Even an in-scope broad China risk keeps its own probability unknown: a
    rate on one date is not the event of a tariff increase over a window.
    Scope comes from the actual question, never from an LLM family label.
    """
    question = str(target.get("question_we_care_about") or "").strip()
    text = " ".join(question.lower().replace("\u2019", "'").split())
    canonical = " ".join(REVIEWED_RISK.lower().split())
    explicit_general_branch = re.fullmatch(
        r"(?:between \d{4}-\d{2}-\d{2} and \d{4}-\d{2}-\d{2}: )?" + re.escape(canonical),
        text) is not None
    product, exclusions = _target_product_scope(text)
    other_country = re.search(_COUNTRIES, text, re.I)
    china = re.search(r"\b(?:china|chinese|prc)\b", text)
    united_states = re.search(r"\b(?:us|u\.s\.?|united states|american)\b", text)
    if other_country:
        # Country mismatch is decisive even if the definition also mentions
        # product-specific duties, positively or as an explicit exclusion.
        status = "DIFFERENT_OR_MIXED_COUNTRY_TARIFF_SCOPE"
        reason = ("This question includes a country other than China. A distribution of the general "
                  "US rate on Chinese goods does not price that country-specific or combined event.")
        admitted = False
    elif product and not explicit_general_branch:
        status = "PRODUCT_SPECIFIC_TARIFF_NOT_GENERAL_CHINA_RATE"
        reason = ("This question concerns a product-, sector-, or issuer-specific tariff. The retrieved brackets exclude "
                  "item-specific duties and measure only the general US rate on Chinese goods; "
                  "their probabilities and P&L do not price this event.")
        admitted = False
    elif not (china and united_states and is_tariff_risk(question)):
        status = "GENERAL_US_CHINA_RATE_SCOPE_NOT_ESTABLISHED"
        reason = ("The question does not establish the general US tariff scope on Chinese goods. "
                  "The retrieved China-rate family remains a separate source-defined scenario.")
        admitted = False
    else:
        status = "RELATED_GENERAL_US_CHINA_RATE_SCENARIO"
        reason = ("The question includes the general US tariff channel on Chinese goods. The dated "
                  "rate brackets can be shown as a related source-defined outcome family; they "
                  "do not assign a probability to the original window event.")
        admitted = True
    return {"status": status, "question": question, "reason": reason,
            "allow_primary_source_outcome_table": admitted,
            "original_event_probability_reassigned": False,
            "source_scenario_is_exact_target_match": False,
            "scope_evidence": {"positive_product_terms": product,
                               "explicit_product_exclusion_clauses": exclusions,
                               "other_country_term": other_country.group(0) if other_country else None}}


def tariff_review(portfolio, daily_beta_report, target, *, prices_by_symbol, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """Classify source contracts and isolate rate scenarios that do not match the target's scope."""
    limit = _horizon_end(target)
    reviews = [classify_candidate(c, limit) for c in target.get("candidates", [])]
    families = bracket_distribution(reviews)
    priced = [f for f in families if f["status"] == "COHERENT_BRACKET_DISTRIBUTION"]
    table, distribution, reason = None, None, None
    if priced:
        distribution = priced[0]
        try:
            table = build_tariff_table(portfolio, distribution, prices_by_symbol, daily_beta_report, n_draws=n_draws, seed=seed)
        except (ValueError, KeyError, TypeError) as exc:
            reason = "The bracket family is coherent, but its P&L could not be computed: " + str(exc)
        else:
            if table["summary"]["usd_per_point_rate_surprise"] == 0 and not table["missing_symbols"]:
                reason = "No anchor price moved any position."
    elif families:
        reason = families[0].get("reason")
    else:
        reason = "No searched contract defines a bracket of the general US tariff rate on Chinese goods inside the horizon."
    scope = target_scope_review(target)
    escalation = _escalation_q(distribution["atoms"]) if distribution else None
    ladders = threshold_ladders(reviews)
    result = {
        "schema": SCHEMA, "status": "PRICED_BRACKET_FAMILY" if table else "NO_PRICED_TARIFF_CONTRACT",
        "target_scope_review": scope,
        "candidate_reviews": reviews, "bracket_families": families, "threshold_ladders": ladders,
        "cross_venue_checks": cross_venue_check(distribution, ladders, table) if distribution else [],
        "distribution": distribution, "outcome_table": table, "reason": reason,
        "anchor_sensitivity": compare_anchor_calibrations(table) if table else None,
        "broad_risk": {
            "definition": target.get("question_we_care_about"), "horizon_end": limit.isoformat(),
            "probability_definition": target.get("question_we_care_about"),
            "probability": {"status": "UNKNOWN", "upper_bound": None, "lower_bounds_by_venue": [],
                            "scope": "The brackets price the general China rate on one date. Product-specific duties "
                                     "are excluded, and today's rate is not certified here, so no probability bound "
                                     "on the original question follows."},
            "closest_exact_event": None if escalation is None else {
                **escalation, "venue": distribution["venue"], "measurement_date": distribution["measurement_date"],
                "definition": f"The general US tariff rate on Chinese goods is {escalation['threshold_pp']:g}% or higher on "
                              f"{distribution['measurement_date']}, under the contract's rules (tariffs in effect; item-specific excluded).",
                "differences": ["Measured on one date, not over the whole window.",
                                "General rate only: product-specific tariffs do not count.",
                                "A rise that is reversed before the date does not count; a paused or announced tariff does not count."]},
        },
        "physical_probability_P": None, "new_provider_or_model_calls": 0,
    }
    if not scope["allow_primary_source_outcome_table"]:
        result["broad_risk"]["probability"]["scope"] = scope["reason"]
    if distribution and not scope["allow_primary_source_outcome_table"]:
        # Preserve the useful calculation, but never let it become the target's
        # headline probability, conditional P&L, or risk-ranking input.
        result["related_general_rate_scenario"] = {
            "status": "SEPARATE_SOURCE_SCENARIO_NOT_TARGET_EVENT",
            "title": "General US tariff rate on Chinese goods on " + distribution["measurement_date"],
            "reason": scope["reason"], "original_event_probability_reassigned": False,
            "distribution": distribution, "outcome_table": table,
            "anchor_sensitivity": result["anchor_sensitivity"],
            "cross_venue_checks": result["cross_venue_checks"],
            "closest_exact_event": result["broad_risk"]["closest_exact_event"],
        }
        result.update(status="TARGET_TARIFF_SCOPE_NOT_MATCHED", distribution=None, outcome_table=None,
                      anchor_sensitivity=None, cross_venue_checks=[], reason=scope["reason"])
        result["broad_risk"]["closest_exact_event"] = None
        result["broad_risk"]["probability"]["scope"] = scope["reason"]
    return result


def normalize_recorded_tariff_scopes(report):
    """Return a display copy with current tariff scope admission, plus corrections.

    Older reports can attach a general-China rate table to a sector-specific
    hypothesis. Keep that exact table as a separately named source scenario;
    never let its numbers supply the hypothesis's probability, P&L or ranking.
    This function neither recomputes economic values nor edits source bytes.
    """
    displayed = deepcopy(report)
    events = (displayed.get("default_analysis") or {}).get("events") or []
    corrections = []
    for event in events:
        if event.get("family") != "TRADE_TARIFFS":
            continue
        review = event.get("tariff_review") or {}
        question = event.get("original_question") or event.get("question_we_care_about") or ""
        scope = target_scope_review({"question_we_care_about": question})
        if scope["allow_primary_source_outcome_table"]:
            continue
        inherited = any(event.get(key) is not None for key in ("distribution", "outcome_table", "economics"))
        inherited = inherited or any(review.get(key) is not None for key in ("distribution", "outcome_table"))
        inherited = inherited or (event.get("risk_ranking") or {}).get("q_standard_deviation_usd") is not None
        if not inherited:
            continue
        distribution = review.get("distribution") or event.get("distribution")
        table = review.get("outcome_table") or event.get("outcome_table")
        if distribution and not review.get("related_general_rate_scenario"):
            review["related_general_rate_scenario"] = {
                "status": "SEPARATE_SOURCE_SCENARIO_NOT_TARGET_EVENT",
                "title": "General US tariff rate on Chinese goods on " + distribution["measurement_date"],
                "reason": scope["reason"], "original_event_probability_reassigned": False,
                "distribution": distribution, "outcome_table": table,
                "anchor_sensitivity": review.get("anchor_sensitivity"),
                "cross_venue_checks": review.get("cross_venue_checks") or [],
                "closest_exact_event": (review.get("broad_risk") or {}).get("closest_exact_event"),
            }
        broad = review.get("broad_risk") or {}
        broad.update(definition=question, probability_definition=question, closest_exact_event=None,
                     probability={"status": "UNKNOWN", "upper_bound": None,
                                  "lower_bounds_by_venue": [], "scope": scope["reason"]})
        review.update(status="TARGET_TARIFF_SCOPE_NOT_MATCHED", target_scope_review=scope,
                      distribution=None, outcome_table=None, anchor_sensitivity=None,
                      cross_venue_checks=[], reason=scope["reason"], broad_risk=broad,
                      physical_probability_P=None)
        event.update(status="TARGET_TARIFF_SCOPE_NOT_MATCHED", tariff_review=review,
                     distribution=None, outcome_table=None, economics=None,
                     physical_probability_P=None, original_question_probability_reassigned=False,
                     reason=scope["reason"],
                     risk_ranking={"q_standard_deviation_usd": None,
                                   "basis": "Original tariff scope has no admitted outcome table."})
        if "announcement_day_economics" in event:
            event["announcement_day_economics"] = None
        ranking = event.setdefault("ranking", {})
        ranking.update(abs_delta_pnl_usd=None, abs_delta_pnl_interval_usd=None,
                       selected_outcome_id=None, basis="TARGET_TARIFF_SCOPE_NOT_MATCHED")
        corrections.append({"event_id": event.get("event_id"), "title": event.get("title"),
                            "reason": scope["reason"], "scope_status": scope["status"]})
    if corrections:
        # Reapply the existing display order to the admitted values. A withdrawn
        # tariff calculation cannot retain its old numerical ranking position.
        events.sort(key=lambda event: (
            (event.get("risk_ranking") or {}).get("q_standard_deviation_usd") is None,
            -((event.get("risk_ranking") or {}).get("q_standard_deviation_usd") or 0.0),
            (event.get("ranking") or {}).get("abs_delta_pnl_usd") is None,
            -((event.get("ranking") or {}).get("abs_delta_pnl_usd") or 0.0),
            event.get("event_date") or "9999-12-31", event.get("event_id", "")))
        for index, event in enumerate(events, 1):
            event.setdefault("ranking", {})["display_order"] = index
    return displayed, corrections


def review_recorded_card(event):
    """The current contract decisions for a recorded tariff card: same search and quotes, no new data.

    The bracket family must come out exactly as recorded, so its quotes, Q and P&L table are kept unchanged;
    only the decisions on the other contracts and the cross-venue checks are read again.
    """
    review = deepcopy(event["tariff_review"])
    limit = datetime.fromisoformat(review["broad_risk"]["horizon_end"])
    reviews = [classify_candidate(c, limit) for c in (event.get("market_search") or {}).get("candidates", [])]
    families = bracket_distribution(reviews)
    priced = [f for f in families if f["status"] == "COHERENT_BRACKET_DISTRIBUTION"]
    source_scenario = review.get("related_general_rate_scenario") or review
    if not same_values(json.loads(json.dumps(priced[0] if priced else None)), source_scenario["distribution"]):
        raise ValueError("The current rules do not reproduce the recorded bracket family.")
    ladders = threshold_ladders(reviews)
    source_scenario.update(
        cross_venue_checks=cross_venue_check(source_scenario["distribution"], ladders, source_scenario["outcome_table"])
        if source_scenario["distribution"] else [],
        anchor_sensitivity=compare_anchor_calibrations(source_scenario["outcome_table"])
        if source_scenario.get("outcome_table") else None)
    review.update(candidate_reviews=reviews, bracket_families=families, threshold_ladders=ladders,
                  reviewed_again_from_recorded_search={"provider_or_model_calls": 0,
                                                       "kept_as_recorded": ["quotes", "bracket distribution", "outcome table"]})
    return review
