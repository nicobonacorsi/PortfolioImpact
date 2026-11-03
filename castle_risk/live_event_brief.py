"""Explicit-date event presentation over the existing source and semantic gates.

No provider or model is called here. Live Claude proposals must originate in
``source_first_discovery``; this adapter independently checks every available
original contract again. Historical fallback is always labelled as a replay.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, timedelta
import math
from typing import Any

from .a8_offline_demo import archived_a8_demo
from .adapters import Market
from .agent_budgeted_discovery import (MAX_MARKETS_PER_MATCH, MAX_PUBLIC_SEARCH_QUERIES,
                                       R59_SEARCH_QUERIES)
from .agentic_bridge import AgenticEvidenceError, catalog_record, inspect_exact_contract
from .contract_questions import source_defined_yes_question
from .instrument_exposures import evaluation_instant
from .nonmacro_primary_desk_case import taiwan_primary_page_desk_case
from .portfolio import parse_portfolio
from .semantics import make_contract_target, verify_relationship
from .source_grounding import calendar_context, evaluate_market


_ARCHIVE_FAMILIES = {
    "archive-fomc-oct26": "FED_POLICY", "archive-cpi-oct26": "US_INFLATION",
}
_TAIWAN_ID = "documentary-taiwan-commercial-blockade-2026"
_MODES = {"LIVE", "HISTORICAL_REPLAY"}


def _source_inspection_limit(discovery):
    """Bound legacy reinspection; fully audit *already retrieved* R59 records.

    R59 expanded discovery permits 24 source searches and up to 12 research
    candidates each. The old six-search/three-candidate local inspection
    ceiling inadvertently truncated unrelated, already downloaded source
    records. Local review causes no extra provider/model HTTP requests, so an
    R59 discovery may inspect **all distinct associated quarantine IDs** once
    per parent event. Legacy discovery budgets retain their frozen limits for
    deterministic replay compatibility. No remote request budget is changed.
    """
    budget = discovery.get("provider_search_budget", MAX_PUBLIC_SEARCH_QUERIES)
    if type(budget) is int and budget == R59_SEARCH_QUERIES:
        groups = discovery.get("quarantined_provider_contracts")
        if not isinstance(groups, Mapping):
            return 0
        total = 0
        for rows in groups.values():
            if not isinstance(rows, list):
                continue
            identifiers = {
                (row.get("ticker") or row.get("market_id") or row.get("contract_id"))
                for row in rows if isinstance(row, Mapping)
                and isinstance((row.get("ticker") or row.get("market_id") or row.get("contract_id")), str)
                and (row.get("ticker") or row.get("market_id") or row.get("contract_id")).strip()
            }
            total += len(identifiers)
        return total
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 12:
        budget = MAX_PUBLIC_SEARCH_QUERIES
    return budget * MAX_MARKETS_PER_MATCH


def _unknown_impact():
    return {
        "status": "UNKNOWN_NO_CALIBRATED_OR_DECLARED_EVENT_IMPACT",
        "scope": "NO_EVENT_CONDITIONAL_PORTFOLIO_RETURN_IDENTIFIED",
        "delta_pnl_usd": None, "delta_pnl_interval_usd": None,
    }


def _question(event):
    return str(event.get("question_we_care_about") or event.get("description") or event.get("title") or "Unspecified event question")


def _proposal(checked):
    """Recover only model proposal fields; reported CERTIFIED flags confer no authority."""
    if not isinstance(checked, Mapping):
        return None
    if "proposed_relationship" in checked:
        return {
            "market_id": checked.get("market_id"),
            "relationship": checked.get("proposed_relationship"),
            "explanation": checked.get("proposed_explanation", ""),
            "evidence": checked.get("proposed_evidence", []),
        }
    if "status" not in checked and "verification_status" not in checked and "relationship" in checked:
        return {key: checked.get(key) for key in ("market_id", "relationship", "explanation", "evidence")}
    return None


def _contract_comparison(event, market, *, checked, portfolio_rows, notional, now, mode):
    card = {
        "contract_id": market.ticker if isinstance(market, Market) else None,
        "question_we_care_about": _question(event), "market_question": None,
        "difference": "Original source evidence has not passed the independent contract gate.",
        "decision": "REJECTED_LOCAL_SOURCE_OR_PROVIDER_ADAPTER",
        "event_relationship": "NONE", "event_relationship_status": "UNVERIFIED",
        "contract_pricing_measure_Q_interval": None, "event_pricing_measure_Q_interval": None,
        "pricing_measure_Q_interval": None, "broad_macro_event_matched": False,
        "physical_probability_P": None, "execution_mode": mode,
    }
    if not isinstance(market, Market):
        card["difference"] = "No supported original provider Market object is available for independent inspection."
        return card
    try:
        receipt_time = evaluation_instant(market.retrieved_at)
        if receipt_time > now:
            card["difference"] = "The source receipt is after the explicit evaluation time."
            card["decision"] = "REJECTED_FUTURE_SOURCE_RECEIPT"
            return card
        # Replay Q is checked at its own receipt clock. This never asserts that
        # an old quote was executable at the later chosen evaluation time.
        quote_clock = receipt_time if mode == "HISTORICAL_REPLAY" else now
        if evaluation_instant(market.close_time) <= quote_clock:
            card["difference"] = "The contract close time is at or before the explicit quote-evaluation time."
            card["decision"] = "REJECTED_CONTRACT_CLOSED_AT_QUOTE_EVALUATION"
            return card
        exact = inspect_exact_contract(market, portfolio_rows=portfolio_rows, notional=notional,
                                       observed_at=quote_clock, max_age_seconds=300)
        record = catalog_record(market)
        source_question = source_defined_yes_question(market)
        review = calendar_context(event)
        source_gate = evaluate_market(event, record, review)
        semantic = verify_relationship(event, record, _proposal(checked))
    except (AgenticEvidenceError, ValueError, TypeError, KeyError):
        # Never copy a provider body, filesystem path or arbitrary exception
        # text into a report that may be shared.
        card["difference"] = "The original response, receipt, typed source rules or event input failed independent inspection."
        return card
    card.update(
        market_question=source_question["yes_question"],
        market_rules_primary=market.rules_primary,
        market_rules_secondary=market.rules_secondary,
        source_question=source_question, source=exact["source"],
        exact_contract_status=exact["status"], source_family_gate=source_gate,
        calendar_review=review, event_relationship=semantic["relationship"],
        event_relationship_status=semantic["status"], independent_semantic_check=semantic,
        contract_pricing_measure_Q_interval=exact["pricing_measure_Q_interval"],
        pricing_measure_Q_interval=exact["pricing_measure_Q_interval"],
        quote=exact["quote"], quote_observed_at_utc=market.retrieved_at,
        quote_evaluated_at_utc=quote_clock.isoformat(),
        quote_scope=("HISTORICAL_QUOTE_AT_RECEIPT_TIME_NOT_CURRENT_AT_EVALUATION"
                     if mode == "HISTORICAL_REPLAY" else "SOURCE_RECEIPT_FRESHNESS_CHECKED_AT_EXPLICIT_EVALUATION"),
    )
    certified = semantic["status"] in {"CERTIFIED", "CONDITIONAL_VERIFIED"}
    if not source_gate["eligible"]:
        card.update(decision="REJECTED_SOURCE_FAMILY_OR_TYPED_PREDICATE_FOR_EVENT",
                    difference=source_gate["reason"])
    elif review["status"] == "PROPOSAL_DATE_CONTRADICTS_REVIEWED_OFFICIAL_SCHEDULE":
        card.update(decision="REJECTED_EVENT_DATE_CONTRADICTS_OFFICIAL_CALENDAR",
                    difference="The proposed event date conflicts with the reviewed official calendar; contract identity does not repair the proposal.")
    elif certified and semantic["relationship"] == "EQUIVALENT":
        card.update(
            decision=("CONDITIONAL_EVENT_EQUIVALENCE_WITH_DECLARED_ASSUMPTIONS" if semantic["status"] == "CONDITIONAL_VERIFIED"
                      else "CERTIFIED_EXACT_CONTRACT_IDENTITY" if event.get("definition_origin") == "CONTRACT_IDENTITY"
                      else "CERTIFIED_EVENT_EQUIVALENCE"),
            difference=semantic["reason"],
            event_pricing_measure_Q_interval=exact["pricing_measure_Q_interval"],
            broad_macro_event_matched=event.get("definition_origin") != "CONTRACT_IDENTITY",
        )
        if exact["pricing_measure_Q_interval"] is None:
            card["decision"] += "_NO_ADMISSIBLE_QUOTE"
    elif certified and semantic["relationship"] != "NONE":
        card.update(decision="VERIFIED_PARTIAL_RELATION_REQUIRES_EXISTING_BOUND_ENGINE",
                    difference=semantic["reason"])
    else:
        card.update(decision="EXACT_CONTRACT_ONLY_EVENT_RELATION_NOT_CERTIFIED",
                    difference=semantic["reason"])
    return card


def _quarantined_source_comparison(event, identifier, market, rejection_reasons, *,
                                  portfolio_rows, notional, now, mode):
    """Inspect one previously rejected reference without reconsidering its target.

    Exact contract self-identity is separate from admission for the parent's
    question. Neither a caller-supplied match nor even a contract-identity parent
    can turn this diagnostic path into a target relation or target quote.
    """
    same_identity = isinstance(market, Market) and market.ticker == identifier
    card = _contract_comparison(
        event, market if same_identity else None, checked=None,
        portfolio_rows=portfolio_rows, notional=notional, now=now, mode=mode,
    )
    card["contract_id"] = identifier
    verified = (same_identity
                and card.get("exact_contract_status") in {
                    "CONTRACT_IDENTITY_Q_BOUND", "CONTRACT_IDENTITY_NO_ADMISSIBLE_QUOTE"}
                and card.get("source", {}).get("assurance") ==
                    "LOCAL_HTTP_RECEIPT_BYTES_VERIFIED_NOT_PROVIDER_SIGNATURE")
    rejection = rejection_reasons[0]
    original_difference = card["difference"]
    original_decision = card["decision"]
    card.update(
        source_only_inspection=verified,
        evidence_scope=("VERIFIED_CONTRACT_INSPECTION_NOT_TARGET_PROBABILITY" if verified
                        else "QUARANTINED_CONTRACT_SOURCE_UNAVAILABLE"),
        original_source_rejection_reason=rejection,
        original_source_rejection_reasons=list(rejection_reasons),
        event_relationship="NONE", event_relationship_status="UNVERIFIED",
        independent_semantic_check={
            "market_id": identifier, "relationship": "NONE",
            "status": "PROPOSED_UNVERIFIED", "verification_status": "PROPOSED_UNVERIFIED",
            "evidence": [],
            "reason": "Previously quarantined source reference: the original target rejection remains in force.",
        },
        event_pricing_measure_Q_interval=None, pricing_measure_Q_interval=None,
        physical_probability_P=None, broad_macro_event_matched=False,
        portfolio_yes_no_pnl_usd=None,
    )
    if verified:
        card.update(
            decision="SOURCE_ONLY_CONTRACT_INSPECTION_TARGET_REJECTION_RETAINED",
            difference="Not used for this target: " + "; ".join(rejection_reasons),
            source_contract_url=market.source_url,
            source_contract_close_time=market.close_time,
            source_contract_expiration_time=market.expiration_time,
        )
    else:
        card.update(decision=original_decision,
                    difference="Original target rejection: " + rejection + ". " + original_difference,
                    contract_pricing_measure_Q_interval=None)
    return card


def prepare_live_event_brief(
    discovery: Mapping[str, Any], market_objects: Mapping[str, Market], *,
    portfolio_rows: list[dict[str, Any]], notional: float = 100_000.0,
    evaluated_at: datetime | str, execution_mode: str = "LIVE",
) -> dict[str, Any]:
    """Preserve candidates and unknowns while presenting exact-match decisions.

    A source-first result is research evidence, not certification. Every real
    ``Market`` is passed through ``inspect_exact_contract`` and the existing
    independent relationship verifier. Caller-supplied CERTIFIED flags are not
    used. Related-contract Q is never silently assigned to the proposed event.
    """
    now = evaluation_instant(evaluated_at)
    parse_portfolio(portfolio_rows, notional=notional)
    if execution_mode not in _MODES:
        raise ValueError("Execution mode must be LIVE or HISTORICAL_REPLAY.")
    if not isinstance(discovery, Mapping) or not isinstance(market_objects, Mapping):
        raise ValueError("A discovery result and original market-object mapping are required.")
    candidates = discovery.get("events", [])
    if not isinstance(candidates, list) or any(not isinstance(event, Mapping) for event in candidates):
        raise ValueError("Discovery events must be a list of event mappings.")
    identifiers = [event.get("event_id") for event in candidates]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Discovery event identities must be nonempty and unique.")
    if execution_mode == "LIVE" and any(event.get("definition_origin") == "SCRIPTED_OFFLINE_ARCHIVE" for event in candidates):
        raise ValueError("Scripted archived candidates require HISTORICAL_REPLAY mode.")
    inspection_limit = _source_inspection_limit(discovery)
    source_inventory_inspection = type(discovery.get("provider_search_budget")) is int and (
        discovery.get("provider_search_budget") == R59_SEARCH_QUERIES)
    inspection_counts = {
        "quarantined_reference_occurrences": 0,
        "invalid_reference_occurrences": 0,
        "accepted_reference_duplicates": 0,
        "duplicate_quarantined_references": 0,
        "distinct_source_only_references": 0,
        "inspected_references": 0,
        "verified_contract_inspections": 0,
        "unavailable_contract_inspections": 0,
        "truncated_references": 0,
    }
    quarantine_groups = discovery.get("quarantined_provider_contracts", {})
    if not isinstance(quarantine_groups, Mapping):
        quarantine_groups = {}
    output = []
    for event in candidates:
        event_id = event["event_id"]
        records = discovery.get("markets_by_event", {}).get(event_id, [])
        checked = {row.get("market_id"): row for row in discovery.get("matches_by_event", {}).get(event_id, [])
                   if isinstance(row, Mapping)}
        cards, seen = [], set()
        for row in records:
            if not isinstance(row, Mapping):
                continue
            identifier = row.get("ticker", row.get("market_id"))
            if not isinstance(identifier, str) or identifier in seen:
                continue
            seen.add(identifier)
            card = _contract_comparison(
                event, market_objects.get(identifier), checked=checked.get(identifier),
                portfolio_rows=portfolio_rows, notional=notional, now=now, mode=execution_mode,
            )
            if card["contract_id"] is None:
                card["contract_id"] = identifier
            cards.append(card)
        # Source-family rejection is not disappearance of the retrieved source.
        # Inspect only this event's already-associated quarantined IDs, never
        # every original in the session or any copied model-authored raw body.
        local_counts = {key: 0 for key in inspection_counts}
        quarantined = quarantine_groups.get(event_id, [])
        references: dict[str, list[str]] = {}
        if isinstance(quarantined, list):
            local_counts["quarantined_reference_occurrences"] = len(quarantined)
            for row in quarantined:
                identifier = ((row.get("ticker") or row.get("market_id") or row.get("contract_id"))
                              if isinstance(row, Mapping) else None)
                if not isinstance(identifier, str) or not identifier.strip():
                    local_counts["invalid_reference_occurrences"] += 1
                    continue
                if identifier in seen:
                    local_counts["accepted_reference_duplicates"] += 1
                    continue
                reason = row.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    reason = "ORIGINAL_QUARANTINE_REASON_UNSPECIFIED"
                if identifier in references:
                    local_counts["duplicate_quarantined_references"] += 1
                    if reason not in references[identifier]:
                        references[identifier].append(reason)
                else:
                    references[identifier] = [reason]
        local_counts["distinct_source_only_references"] = len(references)
        for identifier, reasons in references.items():
            if inspection_counts["inspected_references"] + local_counts["inspected_references"] >= inspection_limit:
                local_counts["truncated_references"] += 1
                continue
            card = _quarantined_source_comparison(
                event, identifier, market_objects.get(identifier), reasons,
                portfolio_rows=portfolio_rows, notional=notional, now=now, mode=execution_mode,
            )
            cards.append(card)
            local_counts["inspected_references"] += 1
            local_counts["verified_contract_inspections" if card["source_only_inspection"]
                         else "unavailable_contract_inspections"] += 1
        for key, value in local_counts.items():
            inspection_counts[key] += value
        if not cards:
            documentary = event.get("documentary_matching")
            # Documentary text may explain a mismatch but can never supply a
            # quote, identity check or relationship-admission flag. In
            # particular, a caller cannot insert Q via an empty-source branch.
            cards = [{
                "contract_id": None, "question_we_care_about": _question(event), "market_question": None,
                "difference": "No independently inspectable source-compatible contract was retained in this bounded investigation.",
                "decision": "NOTHING_FITS_WITHIN_RETRIEVED_EVIDENCE_NOT_GLOBAL_MARKET_ABSENCE",
                "event_relationship": "NONE", "event_relationship_status": "UNVERIFIED",
                "contract_pricing_measure_Q_interval": None, "event_pricing_measure_Q_interval": None,
                "pricing_measure_Q_interval": None, "physical_probability_P": None,
                "broad_macro_event_matched": False,
            }]
            if isinstance(documentary, Mapping):
                cards[0].update(
                    market_question=str(documentary.get("market_question", "Documentary market-question claim without an admitted original provider record.")),
                    difference=str(documentary.get("difference", "Documentary text is not independently verified provider evidence.")),
                    decision="DOCUMENTARY_TEXT_ONLY_NO_SOURCE_NATIVE_PROBABILITY_ADMISSION",
                    event_relationship_status="DOCUMENTARY_TEXT_NOT_INDEPENDENTLY_CERTIFIED",
                )
        try:
            event_date = date.fromisoformat(event.get("event_date", ""))
            time_status = "PAST_RELATIVE_TO_EVALUATION" if event_date < now.date() else "WITHIN_OR_AFTER_EVALUATION_DATE"
        except (ValueError, TypeError):
            time_status = "EVENT_DATE_UNRESOLVED"
        if execution_mode == "LIVE" and time_status != "WITHIN_OR_AFTER_EVALUATION_DATE":
            for card in cards:
                card.update(
                    contract_pricing_measure_Q_interval=None, event_pricing_measure_Q_interval=None,
                    pricing_measure_Q_interval=None, broad_macro_event_matched=False,
                    decision="REJECTED_EVENT_DATE_PAST_OR_UNRESOLVED",
                    difference="A past, missing or malformed event date cannot admit a live event quote; no replacement date is invented.",
                    event_relationship_status="EVENT_DATE_NOT_ADMISSIBLE_FOR_LIVE_RESEARCH",
                )
        admissible = [card for card in cards if card.get("event_pricing_measure_Q_interval") is not None]
        if len(admissible) == 1:
            probability_status = ("EXACT_CONTRACT_QUOTE_ONLY_NO_BROAD_MACRO_EVENT_ADMISSION" if event.get("definition_origin") == "CONTRACT_IDENTITY"
                                  else "EVENT_EQUIVALENCE_QUOTE_CONDITIONAL_ON_DISPLAYED_SOURCE_AND_MEASURE_ASSUMPTIONS")
            event_q = admissible[0]["event_pricing_measure_Q_interval"]
        else:
            probability_status = ("MULTIPLE_EVENT_QUOTES_REQUIRE_EXISTING_COMMON_MEASURE_BOUND_ENGINE" if admissible
                                  else "UNKNOWN_EVENT_PROBABILITY_RELATED_CONTRACT_QUOTES_DO_NOT_IDENTIFY_IT")
            event_q = None
        # Preserve independently recognized sector/geopolitical subjects as
        # risk *hypotheses*, even when no market or economic model is available.
        # The older projection erased everything except Fed, CPI and Taiwan as
        # OTHER. Basel proposals were thereby confused with FOMC rate cases.
        from .default_assessment import _family as reviewed_family
        family = reviewed_family(event)
        recognized = {"FED_POLICY", "US_INFLATION", "US_PAYROLLS", "TAIWAN_SUPPLY",
                      "EXPORT_CONTROLS", "TRADE_TARIFFS", "HORMUZ_ENERGY",
                      "OPEC_SUPPLY", "RUSSIA_ENERGY_SANCTIONS", "US_DRUG_PRICING",
                      "FDA_DRUG_APPROVAL", "BANK_CREDIT_STRESS", "BANK_CAPITAL_RULES"}
        if family not in recognized:
            family = {"US_FOMC": "FED_POLICY", "US_CPI": "US_INFLATION"}.get(
                calendar_context(event).get("family"), "OTHER")
        output.append({
            **deepcopy(event), "question_we_care_about": _question(event),
            "family": family,
            "event_definition_scope": ("EXACT_CONTRACT_YES_SETTLEMENT" if event.get("definition_origin") == "CONTRACT_IDENTITY"
                                       else "ECONOMIC_HYPOTHESIS_REQUIRES_INDEPENDENT_MATCHING"),
            "broad_macro_event_matched": any(card.get("broad_macro_event_matched") for card in cards),
            "candidate_status": event.get("candidate_status", "SOURCE_FIRST_RESEARCH_CANDIDATE_NOT_CERTIFIED_ECONOMIC_EVENT"),
            "event_time_status": time_status,
            "matching_status": discovery.get("match_statuses", {}).get(event_id, "NO_MATCHING_RESULT_RECORDED"),
            "matching_comparisons": cards,
            "contracts": cards,
            "quarantined_candidates": deepcopy(quarantined),
            "source_only_inspection_summary": {
                **local_counts, "shared_inspection_limit": inspection_limit,
                "target_admission_changed": False, "provider_or_model_calls": 0,
            },
            "probability": {"status": probability_status, "pricing_measure_Q_interval": event_q,
                            "physical_probability_P": None,
                            "scope": "PRICING_MEASURE_Q_IS_NOT_PHYSICAL_PROBABILITY_P",
                            "equivalent_quote_contract_ids": [c["contract_id"] for c in admissible]},
            "impact": deepcopy(event.get("impact", _unknown_impact())),
            "priced_in": deepcopy(event.get("priced_in", {
                "status": "UNKNOWN_UNTIL_CONDITIONING_EVENT_IMPACT_AND_PRICE_WEIGHT_ARE_DECLARED",
                "yes_pnl_usd": None, "no_pnl_usd": None,
            })),
        })
    return {
        "status": "SOURCE_GATES_RERUN_WITH_UNKNOWN_OUTCOMES_RETAINED",
        "execution_mode": execution_mode,
        "evaluated_at_utc": now.isoformat(),
        "events": output,
        "discovery_status": discovery.get("status", "UNKNOWN"),
        "discovery_execution_mode": discovery.get("execution_mode", "UNRECORDED"),
        "coverage_status": discovery.get("coverage_status", "BOUNDED_INPUT_EVIDENCE_ONLY"),
        "all_market_availability_verified": False,
        "no_provider_or_model_calls_in_this_adapter": True,
        "source_only_inspection_summary": {
            **inspection_counts, "inspection_limit": inspection_limit,
            "limit_basis": ("ALL_DISTINCT_ALREADY_RETRIEVED_ASSOCIATED_REFERENCES_NO_NETWORK"
                            if source_inventory_inspection else
                            "VALIDATED_LEGACY_SEARCH_BUDGET_TIMES_MAX_MARKETS_PER_MATCH"),
            "status": ("BOUNDED_SOURCE_INSPECTION_TRUNCATED" if inspection_counts["truncated_references"]
                       else "ASSOCIATED_QUARANTINE_REFERENCES_INSPECTED_WITHIN_BOUND"),
            "target_admission_changed": False, "provider_or_model_calls": 0,
        },
        "limitations": [
            "Model proposals, related source-family contracts and equivalent binary events are separate objects.",
            "Nothing fits describes retained bounded evidence, not proof that no market exists anywhere.",
            "No event impact, equity pricing weight or physical probability is inferred from source semantics.",
            "Historical replay checks old quotes at their own receipt clocks and never claims current executability.",
            "Quarantined source inspections retain their original target rejection; exact contract evidence cannot supply target probability or portfolio impact.",
        ],
    }


def source_defined_contract_fallback(
    market_objects: Mapping[str, Market], *, exposure_report: Mapping[str, Any],
    evaluated_at: datetime | str, horizon_days: int = 90,
) -> dict[str, Any]:
    """Select actual exact-contract questions when model discovery is unavailable.

    This bounded public-source fallback does not claim an LLM run or macro-event
    equivalence. Settlement identity remains the event definition; the caller
    must display that restriction before fitting an economic scenario to it.
    """
    now = evaluation_instant(evaluated_at)
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or not 1 <= horizon_days <= 366:
        raise ValueError("horizon_days must be an integer from 1 to 366.")
    if not isinstance(market_objects, Mapping) or not isinstance(exposure_report, Mapping):
        raise ValueError("Original market objects and an exposure report are required.")
    if evaluation_instant(exposure_report.get("evaluated_at_utc")) != now:
        raise ValueError("Exposure and event evaluation times must agree.")
    portfolio = exposure_report["portfolio"]
    end = now.date() + timedelta(days=horizon_days)
    eligible = {"FED_POLICY": [], "US_INFLATION": []}
    exclusions = []
    for identifier, market in market_objects.items():
        if not isinstance(market, Market) or market.ticker != identifier:
            exclusions.append({"contract_id": identifier, "reason": "MISSING_OR_INCONSISTENT_ORIGINAL_MARKET"})
            continue
        series = market.series_ticker or market.event_ticker.split("-")[0]
        family = {"KXFEDDECISION": "FED_POLICY", "KXCPI": "US_INFLATION"}.get(series)
        if family is None or not exposure_report["channels_by_family"].get(family):
            continue
        try:
            closes = evaluation_instant(market.close_time)
            receipt = evaluation_instant(market.retrieved_at)
            if not now.date() <= closes.date() <= end or receipt > now:
                raise ValueError("Outside explicit date scope")
            exact = inspect_exact_contract(market, portfolio_rows=portfolio["positions"], notional=portfolio["notional"],
                                           observed_at=now, max_age_seconds=300)
            question = source_defined_yes_question(market)
            record = catalog_record(market)
            target = make_contract_target(record)
            target.update(
                event_id="source-contract-" + market.ticker,
                title=("Federal Reserve policy contract: " if family == "FED_POLICY" else "CPI contract: ") + market.title,
                description="This exact retrieved contract settles YES under its complete source resolution rules.",
                question_we_care_about="Does this exact contract settle YES: " + market.title,
                event_date=closes.date().isoformat(), family=family,
                candidate_origin="SOURCE_DEFINED_CONTRACT_FALLBACK_NOT_CLAUDE",
                candidate_status="EXACT_SOURCE_SETTLEMENT_QUESTION_NOT_GENERAL_MACRO_DISCOVERY",
                contract_close_time=market.close_time,
                event_date_basis="SOURCE_CONTRACT_CLOSE_DATE_NOT_INFERRED_ECONOMIC_RELEASE_TIME",
                exposure_channels=deepcopy(exposure_report["channels_by_family"][family]),
            )
            source_gate = evaluate_market(target, record, calendar_context(target))
            if not source_gate["eligible"]:
                exclusions.append({"contract_id": identifier, "reason": source_gate["reason"]})
                continue
            interval = exact["pricing_measure_Q_interval"]
            spread = interval[1] - interval[0] if interval is not None else math.inf
            # Date first, then source-readable numeric question and admissible
            # quote width. This is contract selection, never event-loss ranking.
            key = (closes.date(), question["typed_predicate"] is None, interval is None, spread, market.ticker)
            eligible[family].append((key, target, record))
        except (AgenticEvidenceError, ValueError, TypeError, KeyError):
            exclusions.append({"contract_id": identifier, "reason": "SOURCE_RECEIPT_IDENTITY_OR_DATE_CHECK_FAILED"})
    selected = [min(rows, key=lambda item: item[0]) for rows in eligible.values() if rows]
    events = [row[1] for row in selected]
    return {
        "status": "SOURCE_DEFINED_PUBLIC_CONTRACT_FALLBACK_MODEL_DISCOVERY_NOT_EXECUTED",
        "execution_mode": "PUBLIC_EXACT_CONTRACT_FALLBACK_NO_CLAUDE",
        "events": events,
        "markets_by_event": {target["event_id"]: [record] for _, target, record in selected},
        "matches_by_event": {event["event_id"]: [] for event in events},
        "match_statuses": {event["event_id"]: "EXACT_CONTRACT_IDENTITY_REQUIRES_INDEPENDENT_INSPECTION" for event in events},
        "quarantined_provider_contracts": {}, "searches": [],
        "excluded_source_contracts": exclusions,
        "missing_families": [family for family, rows in eligible.items() if not rows],
        "request_count": 0, "claude_called": False,
        "coverage_status": "SUPPLIED_PUBLIC_FOMC_CPI_CONTRACTS_ONLY_NO_COMPLETE_VENUE_OR_EVENT_COVERAGE",
        "all_market_availability_verified": False,
        "selection_basis": "NEAREST_CONTRACT_CLOSE_THEN_TYPED_RULE_AND_QUOTE_WIDTH_NOT_EXPECTED_LOSS",
    }


def _magnitude_interval(interval):
    low, high = interval
    return [0.0 if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high))]


def _ranking_input(event):
    impact = event.get("impact")
    if not isinstance(impact, Mapping):
        return None, "NO_MODELED_TOTAL_IMPACT"
    point, interval = impact.get("delta_pnl_usd"), impact.get("delta_pnl_interval_usd")
    if point is None or interval is None or impact.get("missing_symbols"):
        return None, "TOTAL_IMPACT_OR_INTERVAL_UNKNOWN_PARTIAL_COMPONENTS_NOT_RANKED_AS_TOTAL"
    if (isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point)
            or not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval)
            or interval[0] > point or point > interval[1]):
        return None, "INVALID_MODELED_IMPACT_OR_INTERVAL_RETAINED_AS_UNKNOWN"
    return {"point": abs(float(point)), "interval": _magnitude_interval(interval),
            "signed_interval": list(interval), "impact_scope": impact.get("scope", "UNSPECIFIED")}, None


def rank_events_by_modeled_contrast(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order by abs(YES-NO), annotate interval overlap, and retain unknowns last.

    Ranking is conditional on each displayed model and its supplied interval.
    It is neither expected loss nor a probability that one event is worse than
    another. Partial known components never masquerade as total portfolio P&L.
    """
    if not isinstance(events, list) or any(not isinstance(event, Mapping) for event in events):
        raise ValueError("Events must be a list of mappings.")
    identifiers = [event.get("event_id") for event in events]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Ranking requires unique nonempty event identities.")
    prepared = []
    for index, event in enumerate(events):
        estimate, reason = _ranking_input(event)
        prepared.append((index, deepcopy(event), estimate, reason))
    prepared.sort(key=lambda item: (item[2] is None, -(item[2]["point"] if item[2] else 0), item[0]))
    answer = []
    modeled_rank = 0
    for display_order, (_, event, estimate, reason) in enumerate(prepared, 1):
        ranking = {
            "display_order": display_order, "modeled_movement_rank": None,
            "status": reason, "abs_delta_pnl_usd": None, "abs_delta_pnl_interval_usd": None,
            "overlapping_interval_event_ids": [], "known_components_used_as_total": False,
            "strictly_larger_movement_than_event_ids": [],
            "strictly_smaller_movement_than_event_ids": [],
            "interval_comparison_scope": "MODELED_MAGNITUDE_RANGES_NOT_STATISTICAL_CONFIDENCE_BOUNDS",
            "basis": "ABS_MODELED_YES_MINUS_NO_NOT_EXPECTED_LOSS_OR_PHYSICAL_PROBABILITY",
        }
        if estimate is not None:
            modeled_rank += 1
            low, high = estimate["interval"]
            overlap = [other["event_id"] for _, other, candidate, _ in prepared
                       if candidate is not None and other["event_id"] != event["event_id"]
                       and max(low, candidate["interval"][0]) <= min(high, candidate["interval"][1])]
            # A strict priority is defensible under the declared ranges only
            # when *every* admissible magnitude for A exceeds every magnitude
            # for B: lower(A) > upper(B). Non-overlap in the other direction
            # proves the opposite; overlapping intervals are incomparable.
            larger = [other["event_id"] for _, other, candidate, _ in prepared
                      if candidate is not None and other["event_id"] != event["event_id"]
                      and low > candidate["interval"][1]]
            smaller = [other["event_id"] for _, other, candidate, _ in prepared
                       if candidate is not None and other["event_id"] != event["event_id"]
                       and candidate["interval"][0] > high]
            ranking.update(
                modeled_movement_rank=modeled_rank,
                status="POINT_ESTIMATE_ORDER_INTERVALS_OVERLAP" if overlap else "POINT_ESTIMATE_ORDER_CONDITIONAL_ON_MODEL_INTERVALS",
                abs_delta_pnl_usd=estimate["point"], abs_delta_pnl_interval_usd=estimate["interval"],
                overlapping_interval_event_ids=overlap,
                strictly_larger_movement_than_event_ids=larger,
                strictly_smaller_movement_than_event_ids=smaller,
                direction_status="SIGN_UNCERTAIN_WITHIN_MODEL_INTERVAL" if estimate["signed_interval"][0] < 0 < estimate["signed_interval"][1] else "SIGN_CONDITIONAL_ON_MODEL_INTERVAL",
                impact_scope=estimate["impact_scope"],
            )
        event["ranking"] = ranking
        answer.append(event)
    return answer


def discover_class_based_archived_events(
    *, exposure_report: Mapping[str, Any], evaluated_at: datetime | str,
    horizon_days: int = 90,
) -> dict[str, Any]:
    """Explicit historical fallback with generic class channels and original bytes.

    This bounded scripted catalog is not live Claude discovery. An arbitrary
    declared stock or unresolved ETF can generate an investigation, but neither
    an unverified class nor an unknown sensitivity becomes measured exposure.
    """
    now = evaluation_instant(evaluated_at)
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or not 1 <= horizon_days <= 366:
        raise ValueError("horizon_days must be an integer from 1 to 366.")
    if not isinstance(exposure_report, Mapping) or "channels_by_family" not in exposure_report:
        raise ValueError("A class-based instrument exposure report is required.")
    if evaluation_instant(exposure_report.get("evaluated_at_utc")) != now:
        raise ValueError("Exposure and event evaluation times must agree.")
    discovery, originals = archived_a8_demo()
    end = now.date() + timedelta(days=horizon_days)
    selected, exclusions = [], []
    channels = exposure_report["channels_by_family"]
    for candidate in discovery["events"]:
        event_id = candidate["event_id"]
        family = _ARCHIVE_FAMILIES[event_id]
        records = discovery["markets_by_event"][event_id]
        if any(record.get("source_receipt_status") != "LOCAL_BYTES_VERIFIED" for record in records):
            raise ValueError("Archived original source receipt did not pass byte verification.")
        available = max(evaluation_instant(record["retrieved_at"]) for record in records)
        event_day = date.fromisoformat(candidate["event_date"])
        components = channels.get(family, [])
        reason = ("SOURCE_NOT_AVAILABLE_AT_EVALUATION" if available > now else
                  "EVENT_BEFORE_EXPLICIT_EVALUATION_DATE" if event_day < now.date() else
                  "EVENT_OUTSIDE_EXPLICIT_HORIZON" if event_day > end else
                  "NO_NONCASH_CLASS_OR_UNRESOLVED_INSTRUMENT_TO_INVESTIGATE" if not components else None)
        if reason:
            exclusions.append({"event_id": event_id, "reason": reason})
            continue
        selected.append({
            **deepcopy(candidate), "family": family,
            "candidate_status": "SCRIPTED_HISTORICAL_SOURCE_FAMILY_CLASS_CHANNEL_INVESTIGATION",
            "exposure_channels": deepcopy(components),
            "candidate_channel_weight": math.fsum(c["portfolio_weight"] for c in components),
            "unclassified_channel_weight": math.fsum(c["portfolio_weight"] for c in components
                if c["channel_status"] == "UNCLASSIFIED_INSTRUMENT_INVESTIGATION_REQUIRED"),
            "source_available_at_utc": available.isoformat(),
            "source_response_sha256": sorted({r["source_response_sha256"] for r in records}),
            "current_quote_claimed": False, "impact": _unknown_impact(),
        })
    macro_ids = {event["event_id"] for event in selected}
    macro_discovery = deepcopy(discovery)
    macro_discovery["events"] = deepcopy(selected)
    for key in ("markets_by_event", "matches_by_event", "match_statuses", "calendar_reviews_by_event", "quarantined_provider_contracts"):
        macro_discovery[key] = {k: v for k, v in discovery.get(key, {}).items() if k in macro_ids}
    taiwan_channels = channels.get("TAIWAN_SUPPLY", [])
    if taiwan_channels:
        portfolio = exposure_report["portfolio"]
        desk = taiwan_primary_page_desk_case(portfolio_rows=portfolio["positions"], notional=portfolio["notional"])
        event_day = date(2026, 12, 31)
        if date.fromisoformat(desk["research_review_as_of"]) > now.date():
            exclusions.append({"event_id": _TAIWAN_ID, "reason": "DOCUMENTARY_REVIEW_NOT_AVAILABLE_AT_EVALUATION"})
        elif not now.date() <= event_day <= end:
            exclusions.append({"event_id": _TAIWAN_ID, "reason": "EVENT_OUTSIDE_EXPLICIT_DATE_AND_HORIZON"})
        else:
            selected.append({
                "event_id": _TAIWAN_ID, "title": "Taiwan main-island commercial blockade by December 31, 2026",
                "description": "A qualifying enforced main-island commercial-access restriction lasting at least 24 hours.",
                "event_date": event_day.isoformat(), "family": "TAIWAN_SUPPLY",
                "definition_origin": "SCRIPTED_OFFLINE_ARCHIVE",
                "candidate_status": "DOCUMENTARY_SOURCE_HYPOTHESIS_NO_SOURCE_NATIVE_ADMITTED_MARKET",
                "exposure_channels": deepcopy(taiwan_channels),
                "candidate_channel_weight": math.fsum(c["portfolio_weight"] for c in taiwan_channels),
                "unclassified_channel_weight": 0.0, "current_quote_claimed": False,
                "documentary_matching": {
                    "contract_id": None,
                    "question_we_care_about": "Does a qualifying main-island commercial blockade occur by the deadline?",
                    "market_question": "Does a territorial-control military offensive occur, including qualifying inhabited offshore islands?",
                    "difference": "Blockade and territorial offensive do not imply each other under the previously reviewed page excerpts; main-island access and territorial control have different conditions.",
                    "decision": "DOCUMENTARY_NON_EQUIVALENCE_NO_SOURCE_NATIVE_PROBABILITY_ADMISSION",
                    "event_relationship": "NONE", "event_relationship_status": "MANUAL_PAGE_REVIEW_NOT_PROVIDER_RULE_CERTIFICATION",
                    "contract_pricing_measure_Q_interval": None, "event_pricing_measure_Q_interval": None,
                    "physical_probability_P": None, "sources": desk["sources"],
                },
                "impact": _unknown_impact(),
            })
    selected.sort(key=lambda event: (event["event_date"], event["event_id"]))
    complete_discovery = deepcopy(macro_discovery)
    complete_discovery["events"] = deepcopy(selected)
    if any(event["event_id"] == _TAIWAN_ID for event in selected):
        complete_discovery["markets_by_event"][_TAIWAN_ID] = []
        complete_discovery["matches_by_event"][_TAIWAN_ID] = []
        complete_discovery["match_statuses"][_TAIWAN_ID] = "DOCUMENTARY_NON_EQUIVALENCE_NO_ADMITTED_PROVIDER_CONTRACT"
    return {
        "status": "SCRIPTED_CLASS_BASED_HISTORICAL_REPLAY_NOT_LIVE_DISCOVERY",
        "execution_mode": "HISTORICAL_REPLAY", "evaluated_at_utc": now.isoformat(),
        "horizon_days": horizon_days, "events": selected,
        "discovery": complete_discovery, "macro_discovery": macro_discovery,
        "originals": originals, "excluded_candidates": exclusions,
        "ranking_basis": "UNRANKED_UNTIL_MODELED_YES_NO_IMPACT_AND_INTERVAL_ARE_AVAILABLE",
        "live_provider_calls": 0, "live_model_calls": 0,
        "limitations": [
            "The archive contains two fixed macro families and one prior documentary supply-chain case; it is not general live discovery.",
            "Class-based investigation accepts arbitrary symbols without validating their security identity from a name.",
            "Unknown classes and sensitivities remain unknown; candidate NAV is not modeled loss magnitude.",
            "Quotes remain historical and the explicit replay date never defaults to the computer clock.",
        ],
    }
