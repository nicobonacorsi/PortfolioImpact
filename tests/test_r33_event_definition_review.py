"""Definition regressions from the actual public (4) export and paraphrases.

These are lexical consistency checks, not tests that source facts are true.
No event ID, current clock, provider response, API key or network is needed.
"""
from copy import deepcopy
import socket

import pytest

from castle_risk.event_definition_review import review_event_definition


YIELD = {
    "title": "US 10-year Treasury yield closes at a materially higher level than at 2026-10-28 on any day by 2027-01-26",
    "original_question": "Hypothesis: a rates-level trigger, such as a daily 10-year constant-maturity yield close above a threshold. "
                         "The threshold must come from an actual retrieved contract rule and is not invented here. "
                         "Distinct from the scheduled FOMC and CPI releases, because it could arise from fiscal supply, term premium or auction stress.",
    "date": "2027-01-26",
}
MEXICO = {
    "title": "US imposes new tariffs or border restrictions on Mexican goods, or USMCA review ends in withdrawal notice, by 2027-01-26",
    "original_question": "Hypothesis: single trigger is the US formally imposing new or higher tariffs on goods imported from Mexico, effective by 2027-01-26. "
                         "Counting requires an official order or notice. Market rules must be retrieved to confirm the definition. "
                         "Do not combine this with other outcomes.",
    "date": "2027-01-26",
}
GDP = {
    "title": "BEA reports negative real US GDP growth for Q3 2026 in its advance estimate by 2027-01-26",
    "original_question": "Hypothesis: a single predicate is the BEA advance estimate of Q3 2026 real GDP growth being below zero. "
                         "The date of the release and the exact source need to be confirmed from a documentary schedule and retrieved rules. "
                         "It is not a date-confirmed calendar entry here.",
    "date": "2027-01-26",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Event-definition review is pure and offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize("event,status", [(YIELD, "INCOMPLETE_PREDICATE"),
                                          (MEXICO, "COMPOSITE_PREDICATE"),
                                          (GDP, "UNCONFIRMED_RELEASE_SOURCE")])
def test_actual_export_questions_fail_for_their_own_definition_not_id(event, status):
    event = {**deepcopy(event), "event_id": "arbitrary-changed-id"}
    before = deepcopy(event)
    result = review_event_definition(event)
    assert result["status"] == status
    assert result["reason"]
    assert event == before


@pytest.mark.parametrize("wording", [
    "US 10-year Treasury yield closes materially higher by 2027-01-26",
    "US Treasury yield closes above a threshold by 2027-01-26",
    "US 2-year Treasury yield exceeds a level to be determined",
    "Ten-year Treasury yields increase significantly in the next month",
    "US 10y Treasury yield rises to an unspecified level",
    "10-year constant-maturity yield above an undetermined trigger",
    "US 10\u2011year Treasury yield above an unspecified threshold",
])
def test_missing_yield_number_is_not_supplied_by_dates_maturity_or_related_source(wording):
    event = {"title": wording, "admitted": [{"question": "Treasury yield above 4.5%"}],
             "outcome_table": {"threshold": 4.5},
             "related_source_scenario_ids": ["source-yield-4.5"]}
    assert review_event_definition(event)["status"] == "INCOMPLETE_PREDICATE"


@pytest.mark.parametrize("predicate", [
    "above 4.5%", "below 3.75 percent", "at least 4.25%", ">= 4.50%",
    "between 4% and 5%", "in the range 4 to 5%", "rises by 50 basis points",
    "increases at least 25 bps", "at a threshold of 4.5%", "exactly 4.5%",
    "below 3% or above 5%",
])
def test_explicit_numeric_yield_predicates_remain_available_for_source_review(predicate):
    event = {"title": "US 10-year Treasury yield " + predicate + " by 2027-01-26"}
    assert review_event_definition(event) == {"status": "DEFINED_FOR_REVIEW", "reason": ""}


def test_concrete_question_can_complete_abbreviated_headline():
    event = {"title": "US 10-year Treasury yield event",
             "original_question": "10-year constant-maturity Treasury yield closes above 4.5% by 2027-01-26"}
    assert review_event_definition(event)["status"] == "DEFINED_FOR_REVIEW"


@pytest.mark.parametrize("pending", [
    "The threshold has not been chosen.", "The threshold is not specified.",
    "The threshold will be selected from a retrieved rule.",
    "The numerical trigger needs to come from an actual contract.",
    "The cutoff remains undetermined.",
])
def test_explicit_unresolved_threshold_wins_over_incidental_numeric_reference(pending):
    event = {"title": "US 10-year Treasury yield above 4.5%?",
             "original_question": "The baseline is 4.5%. " + pending}
    assert review_event_definition(event)["status"] == "INCOMPLETE_PREDICATE"


def test_old_yield_and_composite_checks_remain_supported():
    assert review_event_definition({"title": "US 10-year Treasury yield event",
        "original_question": "Yield closes above a threshold by Jan 25. The threshold has not been chosen.",
        "event_date": "2027-01-25"})["status"] == "INCOMPLETE_PREDICATE"
    assert review_event_definition({"title": "US Mexico tariff or USMCA",
        "original_question": "New general tariff on Mexico, or the US government gives formal notice of withdrawal from USMCA.",
        "event_date": "2027-01-25"})["status"] == "COMPOSITE_PREDICATE"


def test_dated_r29_gdp_calendar_check_is_preserved_without_new_calendar_claim():
    event = {"title": "BEA GDP Q4 2026", "original_question": "BEA's advance estimate of Q4 2026 real GDP is negative by Jan 25.",
             "event_date": "2027-01-25"}
    result = review_event_definition(event)
    assert result["status"] == "INCOMPATIBLE_RELEASE_DEADLINE"
    assert "Jan 28, 2027" in result["reason"]
    assert result["source_url"] == "https://www.bea.gov/news/schedule/next-year"
    assert result["source_checked_on"] == "2026-10-27"
    event["event_date"] = "2027-01-29"
    event["original_question"] = "BEA advance estimate of Q4 2026 GDP is negative"
    assert review_event_definition(event)["status"] == "DEFINED_FOR_REVIEW"


@pytest.mark.parametrize("title,question", [
    ("New import tariffs or a formal trade agreement withdrawal", "New US tariffs are imposed on imported goods"),
    ("US closes the border or raises tariffs", "US raises tariffs on goods from a named country"),
    ("NBER declares a recession or BEA reports negative GDP", "BEA reports negative real GDP growth"),
])
def test_headline_union_cannot_masquerade_as_narrower_question(title, question):
    result = review_event_definition({"title": title, "original_question": question})
    assert result["status"] == "COMPOSITE_PREDICATE"
    assert "headline" in result["reason"]


@pytest.mark.parametrize("event", [
    {"title": "US imposes new or higher tariffs on steel, aluminum or heavy machinery",
     "original_question": "New tariffs on steel or aluminum, documented in an official proclamation or Federal Register notice."},
    {"title": "New US tariffs on Mexico", "original_question": "New US tariffs are imposed on Mexican imports. Do not combine this with USMCA withdrawal."},
    {"title": "General China tariff rate on 2026-12-31", "source_defined_scenario": True,
     "original_question": "General US tariff rate on Chinese goods on 2026-12-31, under source contract rules; item-specific duties excluded.",
     "relation_reason": "The parent mentions Mexican tariffs or USMCA withdrawal."},
    {"title": "General US China tariff rate at least 5% and below 15% on Dec 31, 2026", "source_defined_scenario": True},
    {"title": "FOMC holds rates or the scheduled meeting is canceled", "source_defined_scenario": True},
])
def test_product_lists_document_choices_and_source_brackets_are_not_false_unions(event):
    assert review_event_definition(event)["status"] == "DEFINED_FOR_REVIEW"


@pytest.mark.parametrize("uncertainty", [
    "The official release date is unknown.",
    "The release date has not yet been confirmed.",
    "The documentary source must be verified.",
    "The release schedule is still to be confirmed.",
    "This is not a date-confirmed calendar entry.",
])
def test_explicit_unconfirmed_official_calendar_is_a_hypothesis(uncertainty):
    event = {"title": "BEA advance estimate reports negative real GDP", "original_question": uncertainty,
             "date": "2027-01-26", "calendar_source": {"url": "https://www.bea.gov/"}}
    assert review_event_definition(event)["status"] == "UNCONFIRMED_RELEASE_SOURCE"


@pytest.mark.parametrize("event", [
    {"title": "BEA advance estimate reports negative Q3 GDP", "original_question": "The official release date was confirmed from the documented calendar."},
    {"title": "BEA reports negative real GDP", "original_question": "Market rules must be retrieved to confirm a prediction-market match."},
    {"title": "BEA reports negative real GDP", "original_question": "The GDP release outcome is unknown."},
    {"title": "BEA reports negative real GDP", "original_question": "The release date was confirmed, but the GDP result remains unknown."},
    {"title": "FOMC policy decision", "original_question": "The calendar date was confirmed while the decision remains unknown."},
    {"title": "US may impose tariffs by 2027-01-26", "original_question": "This is an unscheduled policy window, not a claim of a confirmed announcement date."},
    {"title": "General US tariff rate on Chinese goods on 2026-12-31", "source_defined_scenario": True},
])
def test_no_new_universal_calendar_or_market_verification_requirement(event):
    assert review_event_definition(event)["status"] == "DEFINED_FOR_REVIEW"


@pytest.mark.parametrize("canonical_key", ["original_question", "question_we_care_about", "description"])
def test_raw_and_public_definition_field_names_share_the_same_review(canonical_key):
    event = {"title": MEXICO["title"], canonical_key: MEXICO["original_question"]}
    assert review_event_definition(event)["status"] == "COMPOSITE_PREDICATE"


@pytest.mark.parametrize("event", [None, 1, [], {}, {"title": {}}, {"description": []}])
def test_malformed_or_missing_definition_is_not_certified(event):
    assert review_event_definition(event)["status"] == "INCOMPLETE_PREDICATE"


# R36 follow-up: synthetic latent corporate-definition regressions, not claims
# about a company's real calendar or the contents of the latest user export.
@pytest.mark.parametrize("title,uncertainty", [
    ("CAT Q3 2026 earnings release", "The official release date remains unconfirmed."),
    ("CAT Q3 2026 earnings release", "The release date has not yet been confirmed."),
    ("UNP quarterly EPS announcement", "The earnings announcement date is unknown."),
    ("NEE annual financial results", "The documentary source must be verified."),
    ("Company quarterly results", "The results date is TBD."),
    ("Company revenue report", "The release calendar has not yet been announced."),
    ("Company sales announcement", "The report time is not yet published."),
    ("Company guidance announcement", "The announcement date is undetermined."),
    ("Company earnings call", "The date of the earnings call must be confirmed."),
    ("Company earnings report", "The release date and exact source remain unknown."),
])
def test_explicit_pending_corporate_release_is_a_hypothesis(title, uncertainty):
    event = {
        "title": title, "original_question": uncertainty,
        "event_id": "synthetic-id-not-a-calendar-proof", "date": "2026-10-29",
        "calendar_source": {"url": "https://example.invalid/investor-relations"},
        "quality": {"model_admission": {"state": "CONDITIONAL_SCENARIO", "primary_impact_allowed": True}},
        "related_source_scenario_ids": ["a-different-source-does-not-resolve-the-date"],
    }
    before = deepcopy(event)
    result = review_event_definition(event)
    assert result["status"] == "UNCONFIRMED_RELEASE_SOURCE"
    assert "monitoring-window end" in result["reason"]
    assert "source_url" not in result  # No inferred date or newly verified source.
    assert event == before


@pytest.mark.parametrize("question", [
    "The official release date was confirmed from the issuer calendar.",
    "The official release date is confirmed and the earnings result is unknown.",
    "The release date is confirmed, but the EPS outcome remains unknown.",
    "The earnings result is unknown while the calendar date is confirmed.",
    "The source market probability is unknown.",
    "Market rules must be retrieved to confirm a prediction-market match.",
    "The analyst revenue forecast has not yet been published.",
    "EPS is below 2 dollars on release; the realized EPS is unknown beforehand.",
    "The company reports quarterly revenue on 2026-10-29.",
    "Company earnings release within the research horizon; no confirmed calendar claim is made.",
])
def test_corporate_guard_does_not_infer_a_calendar_or_block_unknown_outcomes(question):
    event = {"title": "Company quarterly earnings release", "original_question": question,
             "date": "2026-10-29"}
    assert review_event_definition(event) == {"status": "DEFINED_FOR_REVIEW", "reason": ""}


@pytest.mark.parametrize("title,question", [
    ("CAT misses revenue or cuts full-year guidance", "CAT reports revenue below analyst consensus."),
    ("Company revenue misses consensus or EPS exceeds consensus", "Company revenue misses consensus."),
    ("Company EPS beats expectations or guidance is withdrawn", "Company EPS beats expectations."),
])
def test_corporate_headline_union_cannot_borrow_one_component_market(title, question):
    event = {"title": title, "original_question": question}
    before = deepcopy(event)
    result = review_event_definition(event)
    assert result["status"] == "COMPOSITE_PREDICATE"
    assert "headline" in result["reason"]
    assert event == before


@pytest.mark.parametrize("question", [
    "CAT reports revenue below consensus or lowers full-year guidance.",
    "UNP EPS misses consensus or revenue exceeds consensus.",
    "Either NEE earnings beat consensus or the company suspends guidance.",
    "Company sales fall short of expectations or earnings exceed consensus.",
])
def test_explicit_separate_corporate_result_alternatives_need_separate_definitions(question):
    result = review_event_definition({"title": "Corporate results scenario", "description": question})
    assert result["status"] == "COMPOSITE_PREDICATE"
    assert "with OR" in result["reason"]


@pytest.mark.parametrize("event", [
    {"title": "CAT revenue below 10 billion or above 12 billion"},
    {"title": "CAT cuts revenue or EPS guidance"},
    {"title": "CAT raises revenue guidance or lowers EPS guidance"},
    {"title": "CAT earnings or revenue release on October 29"},
    {"title": "CAT EPS and revenue miss consensus on Wednesday or Thursday"},
    {"title": "CAT revenue and EPS miss consensus, documented in an SEC filing or earnings release"},
    {"title": "CAT revenue misses consensus", "original_question":
     "Revenue misses consensus. Do not combine this with an EPS miss or guidance cut."},
    {"title": "Company revenue below 10 billion", "source_defined_scenario": True,
     "original_question": "Company quarterly revenue is below 10 billion under the source contract rules.",
     "relation_reason": "The parent combines a revenue miss or guidance cut; its calendar is unconfirmed."},
])
def test_corporate_metric_lists_date_choices_and_single_source_predicates_remain_allowed(event):
    assert review_event_definition(event) == {"status": "DEFINED_FOR_REVIEW", "reason": ""}


@pytest.mark.parametrize("question_key", ["original_question", "question_we_care_about", "description"])
def test_corporate_date_guard_is_shared_by_raw_and_public_event_shapes(question_key):
    event = {"title": "Company earnings release", question_key: "The release date is unknown."}
    assert review_event_definition(event)["status"] == "UNCONFIRMED_RELEASE_SOURCE"
