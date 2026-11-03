"""R32 scope regressions: an excluded tariff measure is not a positive target.

These deterministic cases check semantic admission and preserved abstention,
not investment performance. Existing R13 prices are explicitly synthetic.
"""
from copy import deepcopy

import pytest

from castle_risk.tariff_scenarios import target_scope_review, tariff_review
from test_r13_tariffs import BOOK, RETURNS, family_candidates, prices


MEXICO_QUESTION = (
    "Hypothesis, not a confirmed calendar. Counts if, between 2026-10-28 and "
    "2027-01-26 inclusive, the US government puts into effect (not merely "
    "threatens) a new or increased tariff applying broadly to goods imported "
    "from Mexico, as shown by an official executive order or Federal Register "
    "notice. Sector-specific measures that don't cover Mexico-origin goods "
    "generally do not count. Exact contract wording must be matched against "
    "actual resolution rules."
)
GENERAL_CHINA = "Will a new US tariff applying broadly to Chinese goods take effect?"


def scope(question):
    return target_scope_review({"question_we_care_about": question})


def test_actual_mexico_definition_reports_country_mismatch_not_product_scope():
    result = scope(MEXICO_QUESTION)
    assert result["status"] == "DIFFERENT_OR_MIXED_COUNTRY_TARIFF_SCOPE"
    assert "country other than China" in result["reason"]
    assert "question concerns a product" not in result["reason"]
    assert result["allow_primary_source_outcome_table"] is False
    assert result["scope_evidence"]["other_country_term"] == "mexico"
    assert result["scope_evidence"]["positive_product_terms"] == []
    assert len(result["scope_evidence"]["explicit_product_exclusion_clauses"]) == 1


@pytest.mark.parametrize("exclusion", [
    "Sector-specific measures do not count.",
    "Product-specific duties will not qualify.",
    "Item-specific increases are excluded.",
    "Issuer-specific tariffs are not included.",
    "Sector-specific measures that don't cover Chinese-origin goods generally do not count.",
    "Sector-specific measures that don’t cover Chinese-origin goods generally do not count.",
    "Sector-specific tariffs only, or threats without formal action, do not count.",
])
def test_explicit_standalone_exclusion_does_not_become_positive_product_constraint(exclusion):
    result = scope(GENERAL_CHINA + " " + exclusion)
    assert result["status"] == "RELATED_GENERAL_US_CHINA_RATE_SCENARIO"
    assert result["scope_evidence"]["positive_product_terms"] == []
    assert result["scope_evidence"]["explicit_product_exclusion_clauses"]
    # An allowed related rate scenario is not an exact target or its probability.
    assert result["source_scenario_is_exact_target_match"] is False
    assert result["original_event_probability_reassigned"] is False


@pytest.mark.parametrize("question", [
    "Will a sector-specific US tariff on Chinese goods take effect?",
    "Will a product-specific US tariff on Chinese goods take effect, not merely be threatened?",
    "Will US tariffs on Chinese semiconductors take effect? Announcements alone do not count.",
    "Will US tariffs on Chinese solar cells take effect?",
    "Will US tariffs on Chinese lithium-ion batteries take effect?",
    "Will US tariffs on Chinese electric vehicles take effect?",
    GENERAL_CHINA + " Product-specific measures are not excluded.",
    GENERAL_CHINA + " Sector-specific measures do not count unless imposed under Section 232.",
    GENERAL_CHINA + " Sector-specific measures that do not count elsewhere qualify.",
    GENERAL_CHINA + " Sector-specific measures count, but threatened general tariffs do not count.",
    "Will US tariffs on Chinese pharmaceuticals take effect? Sector-specific measures do not count.",
])
def test_positive_or_ambiguous_product_restriction_still_fails_closed(question):
    result = scope(question)
    assert result["status"] == "PRODUCT_SPECIFIC_TARIFF_NOT_GENERAL_CHINA_RATE"
    assert result["allow_primary_source_outcome_table"] is False
    assert result["scope_evidence"]["positive_product_terms"]


@pytest.mark.parametrize("question", [
    "Will US tariffs on Mexican semiconductors take effect?",
    "Will US tariffs on Chinese and Mexican goods increase?",
    "Will a new US tariff on Canadian goods take effect? Product-specific duties do not count.",
    "Will a new US tariff on Indian goods take effect?",
])
def test_country_mismatch_has_priority_without_admitting_products_or_mixed_country_targets(question):
    result = scope(question)
    # 'Indian' is not part of the reviewed country grammar; absence of China
    # still blocks it, without inventing an exact classification.
    assert result["status"] in {"DIFFERENT_OR_MIXED_COUNTRY_TARIFF_SCOPE",
                                 "GENERAL_US_CHINA_RATE_SCOPE_NOT_ESTABLISHED"}
    assert result["allow_primary_source_outcome_table"] is False
    assert result["original_event_probability_reassigned"] is False


def test_negated_specificity_does_not_supply_missing_country_evidence():
    result = scope("Will US tariffs increase? Sector-specific measures do not count.")
    assert result["status"] == "GENERAL_US_CHINA_RATE_SCOPE_NOT_ESTABLISHED"
    assert result["allow_primary_source_outcome_table"] is False


def test_mexico_review_keeps_source_china_rate_model_separate_and_original_unknown():
    target = {"event_date": "2027-01-26", "question_we_care_about": MEXICO_QUESTION,
              "candidates": family_candidates()}
    before = deepcopy(target)
    result = tariff_review(BOOK, {"instruments": {}}, target,
                           prices_by_symbol=prices(RETURNS), n_draws=200)
    assert target == before
    assert result["status"] == "TARGET_TARIFF_SCOPE_NOT_MATCHED"
    assert result["distribution"] is None and result["outcome_table"] is None
    assert result["broad_risk"]["probability"]["status"] == "UNKNOWN"
    assert result["broad_risk"]["closest_exact_event"] is None
    related = result["related_general_rate_scenario"]
    assert related["title"] == "General US tariff rate on Chinese goods on 2026-12-31"
    assert related["outcome_table"]["status"] == "COMPLETE_MEASURED_ANCHOR_MODEL"
    assert related["original_event_probability_reassigned"] is False


def test_general_china_exclusion_does_not_turn_dated_rate_into_window_event():
    target = {"event_date": "2027-01-26",
              "question_we_care_about": GENERAL_CHINA + " Sector-specific measures do not count.",
              "candidates": family_candidates()}
    result = tariff_review(BOOK, {"instruments": {}}, target,
                           prices_by_symbol=prices(RETURNS), n_draws=200)
    assert result["broad_risk"]["probability"]["status"] == "UNKNOWN"
    assert result["distribution"]["measurement_date"] == "2026-12-31"
    assert result["target_scope_review"]["source_scenario_is_exact_target_match"] is False
    assert result["target_scope_review"]["original_event_probability_reassigned"] is False


def test_exclusion_clause_cannot_override_a_contract_after_target_horizon():
    target = {"event_date": "2026-12-01",
              "question_we_care_about": GENERAL_CHINA + " Sector-specific measures do not count.",
              "candidates": family_candidates()}
    result = tariff_review(BOOK, {"instruments": {}}, target,
                           prices_by_symbol=prices(RETURNS), n_draws=200)
    assert result["distribution"] is None and result["outcome_table"] is None
    assert {row["decision"] for row in result["candidate_reviews"]} == {"REJECTED_DEADLINE_AFTER_HORIZON"}
    assert result["broad_risk"]["probability"]["status"] == "UNKNOWN"
