"""Appendable source-defined binary questions, without rewriting model hypotheses.

This local adapter answers a narrower question when a research candidate is too
broad to match. The original candidate and its failed match remain separate.
Only already-associated, independently inspected, same-date contracts with a
reviewed numeric predicate qualify. Selection never reads price or impact.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from hashlib import sha256
from typing import Any

from .agentic_bridge import catalog_record
from .instrument_exposures import evaluation_instant
from .live_event_brief import prepare_live_event_brief
from .semantics import make_contract_target


_ORIGIN = "SOURCE_DEFINED_BINARY_REFINEMENT_NOT_CLAUDE"
_SELECTION = "ONE_PER_PARENT_BY_LEXICOGRAPHIC_CONTRACT_ID_AFTER_SOURCE_TYPED_FAMILY_DATE_GATES"


def _empty_discovery() -> dict[str, Any]:
    return {
        "status": _ORIGIN, "execution_mode": "LOCAL_SOURCE_QUESTION_DERIVATION_NO_MODEL_CALL",
        "events": [], "markets_by_event": {}, "matches_by_event": {}, "match_statuses": {},
        "quarantined_provider_contracts": {}, "searches": [], "request_count": 0,
        "coverage_status": "ORIGINAL_RETAINED_CANDIDATES_ONLY_NOT_FULL_MARKET_UNIVERSE",
        "all_market_availability_verified": False,
    }


def derive_source_defined_questions(
    discovery: Mapping[str, Any], market_objects: Mapping[str, Any], *,
    portfolio_rows: list[dict[str, Any]], notional: float = 100_000.0,
    evaluated_at: Any, execution_mode: str = "LIVE", max_refinements: int = 8,
) -> dict[str, Any]:
    """Return original and derived discoveries separately; never silently merge.

    The caller may explicitly append ``refined_discovery`` to the original and
    rerun the assessment. It must display ``original_macro_question`` and the
    different origin. A family-level economic scenario is NOT copied: applying
    one to the new YES/NO question requires an explicit declared assumption.

    At most one question per unmatched broad parent, and at most eight total,
    is selected in parent-ID then contract-ID order. Freshness affects whether
    the selected question has usable Q, never which outcome is selected. The
    preserved broad question is not assigned the selected contract's Q.
    """
    if isinstance(max_refinements, bool) or not isinstance(max_refinements, int) or not 1 <= max_refinements <= 8:
        raise ValueError("max_refinements must be an integer from 1 through 8.")
    now = evaluation_instant(evaluated_at)
    original_brief = prepare_live_event_brief(
        discovery, market_objects, portfolio_rows=portfolio_rows, notional=notional,
        evaluated_at=now, execution_mode=execution_mode,
    )
    original_by_id = {event["event_id"]: event for event in discovery.get("events", [])}
    derived = _empty_discovery()
    exclusions, refinements = [], []
    existing_ids = set(original_by_id)

    for parent in sorted(original_brief["events"], key=lambda event: event["event_id"]):
        parent_id = parent["event_id"]
        if parent.get("definition_origin") == "CONTRACT_IDENTITY":
            continue
        if parent["probability"]["pricing_measure_Q_interval"] is not None:
            continue
        if parent["family"] not in {"FED_POLICY", "US_INFLATION"}:
            exclusions.append({"parent_event_id": parent_id, "reason": "UNSUPPORTED_REFINEMENT_FAMILY"})
            continue
        try:
            parent_date = date.fromisoformat(parent["event_date"])
        except (ValueError, TypeError, KeyError):
            exclusions.append({"parent_event_id": parent_id, "reason": "PARENT_DATE_UNRESOLVED"})
            continue
        if execution_mode == "LIVE" and parent_date < now.date():
            exclusions.append({"parent_event_id": parent_id, "reason": "PARENT_EVENT_ALREADY_PAST"})
            continue
        associated_ids = {
            row.get("ticker", row.get("market_id"))
            for row in discovery.get("markets_by_event", {}).get(parent_id, [])
            if isinstance(row, Mapping)
        }
        candidates = []
        for card in parent["contracts"]:
            identifier = card.get("contract_id")
            reason = None
            if identifier is None:
                reason = "NO_ASSOCIATED_SOURCE_CONTRACT_RETAINED"
            elif identifier not in associated_ids or identifier not in market_objects or getattr(market_objects[identifier], "ticker", None) != identifier:
                reason = "ORIGINAL_MARKET_IDENTITY_NOT_ASSOCIATED_WITH_PARENT"
            elif not card.get("source_family_gate", {}).get("eligible"):
                reason = "PARENT_SOURCE_FAMILY_DATE_OR_PREDICATE_GATE_NOT_PASSED"
            elif card.get("calendar_review", {}).get("status") == "PROPOSAL_DATE_CONTRADICTS_REVIEWED_OFFICIAL_SCHEDULE":
                reason = "PARENT_DATE_CONTRADICTS_REVIEWED_OFFICIAL_CALENDAR"
            elif not isinstance(card.get("source_question", {}).get("typed_predicate"), Mapping):
                reason = "NUMERIC_SOURCE_QUESTION_UNVERIFIED_TITLE_CANNOT_REPAIR_RULES"
            else:
                market = market_objects[identifier]
                if evaluation_instant(market.close_time).date() != parent_date:
                    reason = "CONTRACT_CLOSE_DATE_DIFFERS_FROM_PARENT_EVENT_DATE"
            if reason:
                exclusions.append({"parent_event_id": parent_id, "contract_id": identifier, "reason": reason})
            else:
                candidates.append(card)
        if not candidates:
            continue
        selected = min(candidates, key=lambda card: card["contract_id"])
        identifier = selected["contract_id"]
        if len(derived["events"]) >= max_refinements:
            exclusions.append({"parent_event_id": parent_id, "contract_id": identifier,
                               "reason": "EXPLICIT_REFINEMENT_COUNT_LIMIT_REACHED"})
            continue
        market = market_objects[identifier]
        record = catalog_record(market)
        target = make_contract_target(record)
        child_id = "source-refinement-" + sha256(parent_id.encode("utf-8")).hexdigest()[:12] + "-" + identifier
        if child_id in existing_ids:
            exclusions.append({"parent_event_id": parent_id, "contract_id": identifier,
                               "reason": "REFINEMENT_EVENT_ID_ALREADY_PRESENT"})
            continue
        source_question = selected["source_question"]
        question = f"Does Kalshi contract {identifier} settle YES under its complete source rules: {source_question['yes_question']}"
        target.update(
            event_id=child_id, family=parent["family"], event_date=parent_date.isoformat(),
            question_we_care_about=question, candidate_origin=_ORIGIN,
            candidate_status="DISTINCT_SOURCE_SETTLEMENT_QUESTION_ORIGINAL_MACRO_MATCH_UNCHANGED",
            parent_event_id=parent_id, original_macro_question=parent["question_we_care_about"],
            original_macro_probability_status=parent["probability"]["status"],
            source_question=deepcopy(source_question),
            refinement_difference="The parent describes a broad economic research event. This distinct binary question asks whether one identified source contract settles YES; no equivalence to the parent has been established.",
            model_proposed_this_binary_question=False, broad_macro_hypothesis_equivalence="NOT_ESTABLISHED",
            original_parent_was_rewritten=False, selection_basis=_SELECTION,
            selection_eligible_contract_ids=sorted(card["contract_id"] for card in candidates),
            scenario_scope={
                "status": "NO_ECONOMIC_SCENARIO_TRANSFERRED_BY_REFINEMENT",
                "YES": "This exact source contract settles YES under all its resolution provisions.",
                "NO": "The complement includes every other source settlement outcome; it is not one opposite policy action or inflation value.",
                "NO_mixture": "UNSPECIFIED_CONDITIONAL_MIXTURE_NOT_IDENTIFIED_BY_BINARY_QUOTE",
                "cancellation_policy": source_question["typed_predicate"].get("cancellation_policy", "See complete source rules."),
                "exception_flags": deepcopy(source_question["exception_flags"]),
                "additional_model_assumptions": deepcopy(source_question["additional_model_assumptions"]),
                "economic_branch_mapping": "REQUIRES_EXPLICIT_DECLARED_ASSUMPTION_NOT_ESTIMATED_BY_CONTRACT_OR_FAMILY_MEMBERSHIP",
                "physical_probability_P": None,
            },
        )
        one = _empty_discovery()
        one["events"] = [target]
        one["markets_by_event"][child_id] = [record]
        one["match_statuses"][child_id] = "EXACT_SOURCE_QUESTION_REQUIRES_INDEPENDENT_IDENTITY_GATE"
        checked = prepare_live_event_brief(
            one, market_objects, portfolio_rows=portfolio_rows, notional=notional,
            evaluated_at=now, execution_mode=execution_mode,
        )["events"][0]
        card = checked["contracts"][0]
        if (card.get("event_relationship") != "EQUIVALENT"
                or card.get("event_relationship_status") != "CERTIFIED"
                or not card.get("source_family_gate", {}).get("eligible")
                or not card.get("decision", "").startswith("CERTIFIED_EXACT_CONTRACT_IDENTITY")):
            exclusions.append({"parent_event_id": parent_id, "contract_id": identifier,
                               "reason": "DERIVED_QUESTION_FAILED_INDEPENDENT_EXACT_IDENTITY_GATE"})
            continue
        derived["events"].append(target)
        derived["markets_by_event"][child_id] = [record]
        derived["matches_by_event"][child_id] = []
        derived["match_statuses"][child_id] = one["match_statuses"][child_id]
        existing_ids.add(child_id)
        refinements.append({
            "parent_event_id": parent_id, "event_id": child_id, "contract_id": identifier,
            "original_macro_question": parent["question_we_care_about"], "source_defined_question": question,
            "selection_basis": _SELECTION, "eligible_contract_ids": target["selection_eligible_contract_ids"],
            "source_response_sha256": card["source"]["source_response_sha256"],
            "verified_rules_sha256": source_question["verified_rules_sha256"],
            "typed_predicate": deepcopy(source_question["typed_predicate"]),
            "source_identity_status": card["decision"],
            "pricing_measure_Q_interval": checked["probability"]["pricing_measure_Q_interval"],
            "broad_macro_event_matched": False, "model_proposed_this_binary_question": False,
        })
    return {
        "status": "SEPARATE_SOURCE_QUESTIONS_DERIVED_OR_ORIGINAL_UNKNOWNS_RETAINED",
        "original_discovery": deepcopy(discovery), "original_brief": original_brief,
        "refined_discovery": derived,
        "refinement_brief": prepare_live_event_brief(
            derived, market_objects, portfolio_rows=portfolio_rows, notional=notional,
            evaluated_at=now, execution_mode=execution_mode,
        ),
        "refinements": refinements, "exclusions": exclusions,
        "selection_basis": _SELECTION, "max_refinements": max_refinements,
        "original_questions_preserved": True, "no_provider_or_model_calls": True,
        "no_economic_scenarios_transferred": True,
        "limitations": [
            "The narrower source questions were derived locally, not proposed by Claude; broad event probabilities remain unchanged.",
            "Only original associated contracts are eligible; quarantined or unselected provider search results are not rescued.",
            "A usable Q is a quote-implied pricing weight for exact settlement, not physical P or causal identification of economic impact.",
            "Family-level scenarios must be explicitly assumed for the new YES/NO branches, including an unspecified NO mixture and source exceptions.",
        ],
    }
