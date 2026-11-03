"""R125 IDF-weighted candidate ranking for large contract pools (from the first live probe)."""
from __future__ import annotations

from castle_risk.event_market_search import rank_candidates
from castle_risk.probability_assumptions import probability_question_context


CASE = {"capture": "2026-11-02T10:00:00+00:00"}
EVENT = {"id": "TAIWAN_Q", "date": "2027-01-30",
         "title": ("Between 2026-11-02 00:00 UTC and 2027-01-30 23:59 UTC, the government of China publicly declares and "
                   "actually enforces a maritime quarantine that bars foreign-flagged commercial cargo vessels from "
                   "entering Taiwan's main seaports.")}
TEMPLATE = ("This market will resolve to Yes if {who} enters the territory of {where} on an official government visit "
            "by December 31, 2026, including arrival by maritime or air route. Otherwise it resolves to No.")


def _card(cid, question, rules):
    return {"venue": "Polymarket", "contract_id": cid, "question": question, "rules_text": rules,
            "market_open_at_evaluation": True, "quote_interval": [.01, .02], "close_time": "2027-01-01T00:00:00Z"}


def test_template_words_do_not_outrank_a_specific_subject_match():
    context = probability_question_context(CASE, EVENT)
    pool = [_card(str(i), f"Will Donald Trump visit {place} in 2026?", TEMPLATE.format(who="Donald Trump", where=place))
            for i, place in enumerate(["Pakistan", "Lebanon", "Venezuela", "Portugal", "Israel", "Canada", "Mexico",
                                       "Saudi Arabia", "India", "Japan", "Korea", "Brazil", "Chile", "Peru", "Norway",
                                       "Sweden", "Finland", "Egypt", "Kenya", "Ghana", "Nigeria", "Spain", "Italy"])]
    pool.append(_card("TW", "Will Donald Trump visit Taiwan in 2026?", TEMPLATE.format(who="Donald Trump", where="Taiwan")))
    pool.append(_card("BLOCK", "Will China blockade Taiwan in 2026?",
                      "Resolves Yes if China declares or de facto establishes a naval blockade preventing foreign "
                      "commercial traffic to Taiwan's main ports, enforced by threat or use of force."))
    pool.append(_card("CPI", "US CPI above 0.3% in October?", "Resolves Yes if the BLS CPI release shows more than 0.3%."))
    review, excluded = rank_candidates(context, pool, definition_detail="PRC maritime quarantine of Taiwan seaports.")
    assert review[0]["contract_id"] == "BLOCK"
    ids = {row["contract_id"] for row in review}
    assert len(ids & {str(i) for i in range(23)}) == 0, "template-word-only contracts are not reviewed"
    # A Taiwan-specific but weaker contract is either reviewed or pruned for
    # insufficient overlap relative to the best candidate; never silently lost.
    tw = [row for row in excluded if row["contract_id"] == "TW"]
    assert "TW" in ids or (tw and tw[0]["rejection_basis"] == "INSUFFICIENT_SUBJECT_OVERLAP_PREFILTER")
    bases = {row["rejection_basis"] for row in excluded}
    assert "INSUFFICIENT_SUBJECT_OVERLAP_PREFILTER" in bases and "NO_SHARED_SUBJECT_TERM_PREFILTER" in bases
    assert all(row["decision"] == "REJECTED_INCOMPATIBLE" for row in excluded)


def test_small_pools_keep_the_simple_overlap_rule():
    context = probability_question_context(CASE, EVENT)
    pool = [_card("A", "Will China blockade Taiwan in 2026?", "naval blockade of Taiwan ports"),
            _card("B", "Will the Lakers win?", "basketball")]
    review, excluded = rank_candidates(context, pool)
    assert [row["contract_id"] for row in review] == ["A"]
    assert excluded[0]["contract_id"] == "B"


def test_topical_pool_keeps_subject_words_and_reviews_open_contracts_first():
    """Hormuz live probe, 2026-11-02 07:16 UTC: a pool of 261 strait contracts made 'hormuz'
    and 'strait' frequent, and a rarity-anchored rule wrongly excluded every open related contract."""
    hormuz = {"id": "HORMUZ", "date": "2027-01-30",
              "title": ("From 2026-11-02 00:00 UTC through 2027-01-30 23:59 UTC, commercial oil-tanker transits through "
                        "the Strait of Hormuz fall to zero for at least 72 consecutive hours.")}
    context = probability_question_context(CASE, hormuz)
    pool = [_card(f"b{i}", f"Will there be between {5*i} and {5*i+5} average daily transits of the Strait of Hormuz on October 31?",
                  "Resolves on the IMF PortWatch 7-day average of transit calls through the Strait of Hormuz for all commercial ships.")
            for i in range(14)]
    for row in pool:
        row["market_open_at_evaluation"] = False
        row["quote_interval"] = None
    pool += [_card(f"w{i}", f"Will there be more than {15+i} transit calls through the Strait of Hormuz this week?",
                   "Resolves Yes if IMF PortWatch transit calls through the Strait of Hormuz exceed the threshold.")
             for i in range(8)]
    pool.append(_card("NORMAL", "Strait of Hormuz traffic returns to normal by December 31?",
                      "Resolves Yes if tanker traffic through the Strait of Hormuz returns to normal transit levels."))
    pool.append(_card("RENAME", "Trump renames Strait of Hormuz by December 31?", "Resolves Yes if an executive order renames the strait."))
    review, excluded = rank_candidates(context, pool)
    ids = [row["contract_id"] for row in review]
    assert "NORMAL" in ids and any(i.startswith("w") for i in ids), "open subject-matching contracts are reviewed"
    assert ids.index("NORMAL") < min(ids.index(i) for i in ids if i.startswith("b")) if any(i.startswith("b") for i in ids) else True
    assert all(row["market_open_at_evaluation"] for row in review[:9]), "open contracts come before closed brackets"
    assert len(review) == 12
