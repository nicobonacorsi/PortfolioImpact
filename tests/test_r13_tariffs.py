"""US tariffs on China: rules-first contract review, bracket distribution, measured anchors, routing.

The bracket rules and questions are the Polymarket texts of "U.S. tariff rate on China on December 31,
2026?" as read on 2026-10-24, with that day's displayed bid/ask. Prices are synthetic: round returns on
the two anchor sessions, not provider data.
"""
from datetime import datetime, timezone
from hashlib import sha256
import math
import socket

import pytest

from castle_risk import adapters
from castle_risk.default_assessment import _family as card_family, build_default_assessment
from castle_risk.geopolitical_market_search import (REVIEWED_SEARCH_FAMILIES, _family as search_family, _targets,
                                                    documented_issuer_links, reviewed_family_events)
from castle_risk.portfolio import parse_portfolio
from castle_risk.tariff_scenarios import (ANCHORS, bracket_distribution, build_tariff_table, classify_candidate,
                                          is_tariff_risk, measure_anchors, tariff_review)

RULES = (
    "This market will resolve according to the general tariff rate on imports into the United States from the "
    "People's Republic of China on December 31, 2026, 12:00 PM ET.\n\n"
    "The general tariff rate refers to the base tariff rate paid on imports, including any general tariff the "
    "U.S. imposes on all imports (e.g. a 10% tariff on all U.S. imports and a 10% tariff on top of that on Chinese "
    "imports would equal a 20% tariff).\n\n"
    "If the reported value falls exactly between two brackets, then this market will resolve to the higher range bracket.\n\n"
    "Item specific exceptions or increases will not be considered (i.e. this market does not refer to the "
    "effective tariff rate).\n\n"
    "Only tariffs which are in effect will qualify. Tariffs which are paused, or which have been announced but have not "
    "yet gone into effect will not be considered.\n\n"
    "This market's primary resolution source will be official information from the Trump administration, however a "
    "consensus of credible information will also be used.")
BRACKETS = (
    ("4853272", "Will the U.S. tariff rate on China be less than 5% on December 31?", [.012, .061]),
    ("4853273", "Will the U.S. tariff rate on China be between 5% and 15% on December 31?", [.86, .91]),
    ("4853274", "Will the U.S. tariff rate on China be between 15% and 25% on December 31?", [.07, .10]),
    ("4853275", "Will the U.S. tariff rate on China be between 25% and 35% on December 31?", [.013, .063]),
    ("4853276", "Will the U.S. tariff rate on China be 35% or greater on December 31?", [.006, .029]),
)
HORIZON = datetime(2027, 1, 22, 23, 59, 59, tzinfo=timezone.utc)
LINK = "https://polymarket.com/event/us-tariff-rate-on-china-on-december-31-2026"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("The tariff review must not acquire data.")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(adapters._ReadOnlyClient, "_get", deny)


def candidate(identifier, question, quote, *, rules=RULES, open_=True, link=LINK, end="2026-12-31T00:00:00Z"):
    return {"venue": "Polymarket", "contract_id": identifier, "market_question": question, "rules_text": rules,
            "rules_sha256": sha256(rules.encode()).hexdigest(), "source_url": link, "contract_end_time": end,
            "market_open_at_evaluation": open_,
            "contract_quote_reference": {"interval": quote, "scope": "INDICATIVE_GAMMA_FIRST_OUTCOME_YES_SUMMARY_NOT_CLOB_VERIFIED",
                                         "reasons": [] if quote else ["INVALID_OR_UNSUPPORTED_GAMMA_QUOTE"]}}


def family_candidates():
    return [candidate(*row) for row in BRACKETS]


def other_candidates():
    return [
        candidate("2730834", "Tariff increase on Canada in effect by December 31, 2026?", [.013, .057],
                  rules="This market will resolve to Yes if a Canada-specific increase in the general tariff rate on imports "
                        "from Canada goes into effect by December 31, 2026. Global tariffs will not count."),
        candidate("3029903", "Copper cable hit with Sec. 232 tariffs by December 31, 2026?", [.12, .46],
                  rules="Resolves Yes if a Section 232 instrument imposes a duty above 0% on HTSUS 8544.49.3040 by the deadline."),
        candidate("2734360", "Will Trump create a tariff dividend by December 31, 2026?", [.02, .05],
                  rules="Resolves Yes if a tariff-funded dividend or rebate for individual taxpayers is created by the deadline."),
    ]


def prices(returns, symbols=("SPY", "AAPL", "JPM", "GLD")):
    """Synthetic closes: 100 before each anchor, 100 * (1 + r) on the anchor session."""
    rows = {}
    for symbol in symbols:
        geneva, april = returns[symbol]
        rows[symbol] = [{"date": "2025-04-01", "close": 100.0}, {"date": "2025-04-02", "close": 100.0},
                        {"date": "2025-04-03", "close": 100.0 * (1 + april)}, {"date": "2025-05-08", "close": 100.0},
                        {"date": "2025-05-09", "close": 100.0}, {"date": "2025-05-12", "close": 100.0 * (1 + geneva)}]
    return rows


RETURNS = {"SPY": (.03, -.05), "AAPL": (.06, -.09), "JPM": (.03, -.07), "GLD": (-.03, -.006)}
BOOK = parse_portfolio([{"symbol": "AAPL", "weight": .35, "asset_type": "stock"},
                        {"symbol": "JPM", "weight": .35, "asset_type": "stock"},
                        {"symbol": "GLD", "weight": .30, "asset_type": "etf"}], notional=100000.)


def test_every_real_bracket_is_admitted_from_its_rules_and_question():
    rows = [classify_candidate(c, HORIZON) for c in family_candidates()]
    assert [row["decision"] for row in rows] == ["ADMITTED_OUTCOME_FAMILY_MEMBER"] * 5
    assert [row["bracket_pp"] for row in rows] == [[0., 5.], [5., 15.], [15., 25.], [25., 35.], [35., None]]
    assert {row["measurement_date"] for row in rows} == {"2026-12-31"}
    assert all(row["event_class"] == "US_GENERAL_TARIFF_RATE_ON_CHINA_BRACKET" for row in rows)
    # The same rules with a typographic apostrophe are the same contract grammar.
    curly = classify_candidate(candidate(*BRACKETS[0], rules=RULES.replace("People's", "People’s")), HORIZON)
    assert curly["decision"] == "ADMITTED_OUTCOME_FAMILY_MEMBER"


def test_other_tariff_contracts_are_different_events_with_reasons():
    rows = {row["contract_id"]: row for row in (classify_candidate(c, HORIZON) for c in other_candidates())}
    assert rows["2730834"]["event_class"] == "DIFFERENT_COUNTRY_TARIFF"
    assert rows["3029903"]["event_class"] == "DIFFERENT_PRODUCT_TARIFF"
    assert rows["2734360"]["event_class"] == "TARIFF_REVENUE_TRANSFER"
    assert all(row["decision"] == "REJECTED_DIFFERENT_EVENT" and row["reason"] for row in rows.values())


def test_missing_settlement_clause_late_measurement_or_no_quote_is_not_admitted():
    partial = RULES.replace("Only tariffs which are in effect will qualify.", "")
    row = classify_candidate(candidate(*BRACKETS[1], rules=partial), HORIZON)
    assert row["event_class"] == "UNREVIEWED_BRACKET_WORDING" and "ONLY_TARIFFS_IN_EFFECT" in row["reason"]
    early = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert classify_candidate(candidate(*BRACKETS[1]), early)["decision"] == "REJECTED_DEADLINE_AFTER_HORIZON"
    assert classify_candidate(candidate(BRACKETS[1][0], BRACKETS[1][1], None), HORIZON)["decision"] == "REJECTED_NOT_OPEN_OR_NOT_QUOTED"
    closed = classify_candidate(candidate(*BRACKETS[1], open_=False), HORIZON)
    assert closed["decision"] == "REJECTED_NOT_OPEN_OR_NOT_QUOTED"


def test_complete_brackets_form_one_coherent_distribution():
    reviews = [classify_candidate(c, HORIZON) for c in family_candidates() + other_candidates()]
    (family,) = bracket_distribution(reviews)
    assert family["status"] == "COHERENT_BRACKET_DISTRIBUTION"
    atoms = family["atoms"]
    assert math.isclose(math.fsum(a["q_reference"] for a in atoms), 1.0, abs_tol=1e-9)
    lower, upper = [b[2][0] for b in BRACKETS], [b[2][1] for b in BRACKETS]
    for atom, lo, hi in zip(atoms, lower, upper):
        assert lo - 1e-12 <= atom["q_reference"] <= hi + 1e-12
        expected = [max(lo, 1 - (sum(upper) - hi)), min(hi, 1 - (sum(lower) - lo))]
        assert atom["q_interval"] == pytest.approx(expected)
    assert atoms[0]["label"] == "General US tariff on China below 5% on 31 Dec 2026"
    assert atoms[-1]["rate_representative_pp"] == 45. and atoms[-1]["rate_range_pp"] == [35., 145.]


def test_gaps_and_incoherent_quotes_stay_unknown():
    reviews = [classify_candidate(c, HORIZON) for c in family_candidates()]
    (gap,) = bracket_distribution([r for r in reviews if r["contract_id"] != "4853274"])
    assert gap["status"] == "NO_COHERENT_BRACKET_DISTRIBUTION" and "gap" in gap["reason"]
    rich = [candidate(i, q, [lo + .05, hi + .05]) for i, q, (lo, hi) in BRACKETS]
    (incoherent,) = bracket_distribution([classify_candidate(c, HORIZON) for c in rich])
    assert incoherent["status"] == "NO_COHERENT_BRACKET_DISTRIBUTION" and "sum above 1" in incoherent["reason"]


def test_anchor_returns_need_adjacent_sessions():
    measured = measure_anchors(prices(RETURNS), ["AAPL"])
    assert measured["AAPL"]["GENEVA_2025_05_12"] == pytest.approx(.06)
    assert measured["AAPL"]["RECIPROCAL_2025_04_03"] == pytest.approx(-.09)
    gappy = prices(RETURNS)
    gappy["AAPL"] = [row for row in gappy["AAPL"] if row["date"] != "2025-05-09"]
    assert measure_anchors(gappy, ["AAPL"])["AAPL"]["GENEVA_2025_05_12"] is None


def table():
    reviews = [classify_candidate(c, HORIZON) for c in family_candidates()]
    (family,) = bracket_distribution(reviews)
    return build_tariff_table(BOOK, family, prices(RETURNS), {"instruments": {}}, n_draws=4000)


def test_bracket_pnl_is_the_unpriced_move_and_centres_under_q():
    result = table()
    assert result["status"] == "COMPLETE_MEASURED_ANCHOR_MODEL"
    point, envelope = ANCHORS
    loads = {p["symbol"]: p for p in result["positions"]}
    assert loads["AAPL"]["loading"] == pytest.approx(.06 / point["change_pp"])
    assert loads["AAPL"]["loading_envelope"] == pytest.approx(-.09 / envelope["change_pp"])
    rows = result["outcomes"]
    assert math.fsum(r["q_reference"] * r["pnl_if_outcome_usd"] for r in rows) == pytest.approx(0., abs=1e-6)
    per_pp = result["summary"]["usd_per_point_rate_surprise"]
    mean = result["reference_expectation"]
    for row in rows:
        assert row["pnl_if_outcome_usd"] == pytest.approx(per_pp * (row["state"] - mean))
        low, high = row["pnl_model_range_5_95_usd"]
        assert low <= row["pnl_if_outcome_usd"] <= high
    # Long equities lose when the rate settles above what prices imply. The open top bracket has
    # reference Q below 1% (its bid is 0.6%), so the 1% rule picks 25% to 35%.
    top = rows[-1]
    assert top["q_reference"] < .01
    assert top["pnl_if_outcome_usd"] < 0 and top["model_5_95_range_excludes_zero"]
    worst = result["summary"]["largest_loss_outcome_reference_q_at_least_1pct"]
    assert worst["label"] == "General US tariff on China 25% to 35% on 31 Dec 2026" and worst["pnl_if_outcome_usd"] < 0
    low_mean, high_mean = result["reference_expectation_range_over_q_polytope"]
    assert low_mean <= mean <= high_mean


def test_gold_with_opposite_anchor_signs_keeps_both_signs_in_its_range():
    result = table()
    gold = next(p for p in result["positions"] if p["symbol"] == "GLD")
    assert gold["loading"] > 0 > gold["loading_envelope"]


def test_cash_is_unchanged_missing_prices_use_beta_or_stay_unknown():
    book = parse_portfolio([{"symbol": "AAPL", "weight": .5, "asset_type": "stock"},
                            {"symbol": "MSFT", "weight": .3, "asset_type": "stock"},
                            {"symbol": "XYZ", "weight": .1, "asset_type": "stock"},
                            {"symbol": "USD", "weight": .1, "asset_type": "cash"}])
    reviews = [classify_candidate(c, HORIZON) for c in family_candidates()]
    (family,) = bracket_distribution(reviews)
    daily = {"instruments": {"MSFT": {"beta": 1.2, "beta_interval": [1.1, 1.3]}}}
    result = build_tariff_table(book, family, prices(RETURNS, ("SPY", "AAPL")), daily, n_draws=2000)
    by_symbol = {p["symbol"]: p for p in result["positions"]}
    assert by_symbol["USD"]["channel"] == "NONE" and by_symbol["USD"]["loading"] == 0
    assert by_symbol["MSFT"]["channel"] == "BETA_TRANSFER"
    assert by_symbol["MSFT"]["loading"] == pytest.approx(1.2 * .03 / -115)
    assert by_symbol["XYZ"]["channel"] == "UNKNOWN"
    assert result["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY" and result["missing_symbols"] == ["XYZ"]
    assert result["missing_weight"] == pytest.approx(.1)


def test_review_prices_the_family_but_never_the_broad_risk():
    target = {"event_date": "2027-01-21", "question_we_care_about": "Will new US tariffs on Chinese goods take effect?",
              "candidates": family_candidates() + other_candidates()}
    review = tariff_review(BOOK, {"instruments": {}}, target, prices_by_symbol=prices(RETURNS), n_draws=2000)
    assert review["status"] == "PRICED_BRACKET_FAMILY"
    assert review["broad_risk"]["probability"]["status"] == "UNKNOWN"
    closest = review["broad_risk"]["closest_exact_event"]
    atoms = review["distribution"]["atoms"]
    assert closest["threshold_pp"] == 15. and closest["q_reference"] == pytest.approx(math.fsum(a["q_reference"] for a in atoms[2:]))
    assert closest["q_interval"][0] <= closest["q_reference"] <= closest["q_interval"][1]
    assert review["physical_probability_P"] is None and review["new_provider_or_model_calls"] == 0
    none = tariff_review(BOOK, {"instruments": {}}, dict(target, candidates=other_candidates()), prices_by_symbol=prices(RETURNS))
    assert none["status"] == "NO_PRICED_TARIFF_CONTRACT" and none["outcome_table"] is None and none["reason"]


def test_routing_keeps_ai_chip_and_export_control_risks_with_export_controls():
    assert is_tariff_risk("New US tariffs on Chinese consumer electronics")
    assert is_tariff_risk("Section 232 semiconductor tariffs take effect")
    assert not is_tariff_risk("US export controls and tariffs on advanced AI chips to China")
    assert not is_tariff_risk("New BIS export licensing for H20 GPUs")
    assert search_family({"title": "US tariffs on Chinese imports raised"}) == "US_TARIFFS"
    assert search_family({"title": "Section 232 semiconductor tariffs"}) == "US_TARIFFS"
    assert search_family({"title": "US semiconductor export controls"}) == "US_CHIP_EXPORT_POLICY"
    assert search_family({"title": "China blockade of Taiwan raises tariffs"}) == "TAIWAN"
    assert card_family({"title": "US tariffs on Chinese electronics", "family": "OTHER"}) == "TRADE_TARIFFS"
    # A recorded target keeps the family it was searched as.
    assert search_family({"title": "Section 232 semiconductor tariffs", "family": "US_CHIP_EXPORT_POLICY"}) == "US_CHIP_EXPORT_POLICY"


def _holding(symbol, parent=None, weight=.35):
    return {"original_symbol": parent or symbol, "instrument_symbol": symbol, "portfolio_weight": weight,
            "identity_status": "look_through" if parent else "direct", "asset_type": "stock", "instrument_class": "stock"}


def test_apple_filing_lists_the_tariff_risk_in_the_live_family_set_only():
    links = documented_issuer_links([_holding("AAPL")])
    assert "Beginning in the second quarter of 2025, new tariffs were announced" in links["US_TARIFFS"][0]["statement"]
    assert [e["family"] for e in reviewed_family_events([], links, start="2026-10-23", horizon_days=90)] == ["TAIWAN"]
    added = reviewed_family_events([], links, start="2026-10-23", horizon_days=90, families=REVIEWED_SEARCH_FAMILIES)
    assert [e["family"] for e in added] == ["TAIWAN", "US_TARIFFS"]
    tariff = added[1]
    assert tariff["question_we_care_about"].startswith("Between 2026-10-23 and 2027-01-21: A new or higher US tariff")
    assert _targets(added)[1]["family"] == "US_TARIFFS"
    # A model proposal on tariffs covers the reviewed one.
    proposed = [{"event_id": "apple-tariff", "title": "US tariffs on Chinese-made iPhones", "family": "OTHER"}]
    again = reviewed_family_events(proposed, links, start="2026-10-23", horizon_days=90, families=REVIEWED_SEARCH_FAMILIES)
    assert [e["family"] for e in again] == ["TAIWAN"]
    assert documented_issuer_links([_holding("NVDA")]).get("US_TARIFFS") is None


NOW = datetime.fromisoformat("2026-10-24T16:00:00+00:00")


def _report(event):
    return {"portfolio": {"positions": [{"symbol": "AAPL", "weight": .35, "asset_type": "stock"},
                                        {"symbol": "JPM", "weight": .35, "asset_type": "stock"},
                                        {"symbol": "GLD", "weight": .30, "asset_type": "etf"}], "notional": 100000.},
            "empirical": {"daily_betas": {"instruments": {}}, "release_day_studies": {}},
            "events": [event]}


def _search(event, family, candidates):
    return {"evaluated_at": NOW.isoformat(), "capture_evaluated_at": NOW.isoformat(),
            "targets": [{"event_id": event["event_id"], "event_date": event["event_date"], "family": family,
                         "question_we_care_about": event["question_we_care_about"], "candidates": candidates,
                         "closest_by_venue": {"Polymarket": {"status": "SEARCH_COMPLETED_WITHIN_BOUNDS"}},
                         "decision": "NO_EXACT_FIT_WITHIN_SEARCHED_SCOPE"}]}


def test_default_view_retains_tariff_table_without_ranking_a_different_target():
    event = {"event_id": "apple-tariff", "title": "US tariffs on Chinese imports raised", "family": "OTHER",
             "event_date": "2027-01-21", "question_we_care_about": "Will a new US tariff on Chinese goods take effect by 2027-01-21?"}
    result = build_default_assessment(_report(event), market_objects={}, inputs={"prices_by_symbol": prices(RETURNS)},
                                      evaluated_at=NOW,
                                      geopolitical_search=_search(event, "US_TARIFFS", family_candidates() + other_candidates()))
    card = next(e for e in result["events"] if e["event_id"] == "apple-tariff")
    assert card["family"] == "TRADE_TARIFFS"
    assert card["status"] == "SOURCE_BRACKET_DISTRIBUTION_WITH_MEASURED_ANCHOR_MODEL"
    # R32: a dated tariff-level table does not price/rank an action-by-date
    # hypothesis. Keep the original computed spread as an audit diagnostic.
    spread = card["outcome_table"]["summary"]["q_standard_deviation_usd"]
    assert spread > 0
    assert card["risk_ranking"]["q_standard_deviation_usd"] is None
    assert card["economic_ranking_diagnostics"]["risk_ranking"]["q_standard_deviation_usd"] == spread
    assert card["model_admission"]["state"] == "RELATED_SOURCE_ONLY"
    assert card["tariff_review"]["broad_risk"]["probability"]["status"] == "UNKNOWN"
    assert card["physical_probability_P"] is None


def test_recorded_export_search_of_a_tariff_text_keeps_its_old_card():
    event = {"event_id": "apple-tariff", "title": "US tariffs on Chinese semiconductors", "family": "OTHER",
             "event_date": "2027-01-21", "question_we_care_about": "Will new US tariffs hit Chinese semiconductors by 2027-01-21?"}
    result = build_default_assessment(_report(event), market_objects={}, inputs={}, evaluated_at=NOW,
                                      geopolitical_search=_search(event, "US_CHIP_EXPORT_POLICY", []))
    card = next(e for e in result["events"] if e["event_id"] == "apple-tariff")
    assert card["family"] == "EXPORT_CONTROLS" and "tariff_review" not in card


def test_tariff_card_renders_without_errors():
    from castle_risk.default_assessment_ui import _largest_loss, _probability_summary, _state_text, _tariff
    event = {"event_id": "apple-tariff", "title": "US tariffs on Chinese imports raised", "family": "OTHER",
             "event_date": "2027-01-21", "question_we_care_about": "Will a new US tariff on Chinese goods take effect by 2027-01-21?"}
    result = build_default_assessment(_report(event), market_objects={}, inputs={"prices_by_symbol": prices(RETURNS)},
                                      evaluated_at=NOW,
                                      geopolitical_search=_search(event, "US_TARIFFS", family_candidates() + other_candidates()))
    card = next(e for e in result["events"] if e["event_id"] == "apple-tariff")
    assert _state_text(card) == "Every rate bracket has Q and P&L; broad risk probability unknown"
    assert _probability_summary(card).startswith("5 brackets of the general US rate on China on 2026-12-31")
    assert _largest_loss(card)[1].startswith("General US tariff on China 25% to 35%")

    calls = []

    class Recorder:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                calls.append(name)
                if name == "columns":
                    return [Recorder() for _ in range(args[0] if isinstance(args[0], int) else len(args[0]))]
                if name == "expander":
                    return self
                return None
            return call

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        column_config = type("CC", (), {"TextColumn": staticmethod(lambda **k: None),
                                        "NumberColumn": staticmethod(lambda **k: None)})

    _tariff(Recorder(), card)
    assert "dataframe" in calls and "metric" in calls
