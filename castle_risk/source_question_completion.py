"""Bounded exact questions from already admitted portfolio-relevant source atoms.

This optional live completion step does not rewrite a research question or borrow
its probability. It adds separately identified contract questions only when the
existing exact-identity and economic-binding gates both pass. Historical replay
does not invoke it implicitly.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256

from .agentic_bridge import catalog_record
from .exact_event_binding import derive_exact_bindings
from .instrument_exposures import evaluation_instant
from .live_event_brief import prepare_live_event_brief
from .semantics import make_contract_target


ORIGIN = "SOURCE_DISTRIBUTION_CONTRACT_SUPPLEMENT_NOT_CLAUDE"
SELECTION = "EVENT_DATE_THEN_CONTRACT_ID_ONE_PER_FAMILY_DATE_NO_PRICE_OR_PNL_RANKING"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _identities(report):
    return {e["event_id"]: {k: deepcopy(e.get(k)) for k in (
        "contract_id", "title", "parent_event_id", "probability", "candidate_origin")}
        for e in report.get("events", [])}


def _distributions(report):
    return [{k: deepcopy(card.get(k)) for k in (
        "event_id", "source_event_ticker", "status", "distribution")}
        for card in report.get("default_analysis", {}).get("events", [])]


def supplement_source_questions(report, *, build_kwargs, horizon_days=90, max_questions=4):
    """Return a new report and build kwargs with bounded source-only questions.

    Original events, independent quotes, and admitted source distributions must
    survive reconstruction unchanged. Added questions use current input objects,
    never a packaged example. Selection is independent of economic size/sign.
    The caller must retain ``source_question_completion`` after any later rebuild.
    """
    from .live_assessment import build_assessment_from_evidence

    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or not 1 <= horizon_days <= 366:
        raise ValueError("horizon_days must be an integer from 1 through 366")
    if isinstance(max_questions, bool) or not isinstance(max_questions, int) or not 1 <= max_questions <= 8:
        raise ValueError("max_questions must be an integer from 1 through 8")
    now = evaluation_instant(build_kwargs["evaluated_at"])
    cutoff = now.date() + timedelta(days=horizon_days)
    market_objects = build_kwargs["market_objects"]
    discovery = deepcopy(build_kwargs["discovery"])
    original_ids = _identities(report)
    existing_contracts = {e.get("contract_id") for e in report.get("events", [])
                          if e.get("definition_origin") == "CONTRACT_IDENTITY"}
    quarantined = set(_strings(discovery.get("quarantined_provider_contracts", {})))
    candidates, exclusions, additions = [], [], []
    for card in report.get("default_analysis", {}).get("events", []):
        family, parent_id = card.get("family"), card.get("event_id")
        if family not in {"FED_POLICY", "US_INFLATION"} or parent_id not in original_ids:
            continue
        try:
            event_day = date.fromisoformat(card["event_date"])
        except (KeyError, TypeError, ValueError):
            continue
        if not now.date() <= event_day <= cutoff:
            continue
        if not card.get("source_event_ticker") or not isinstance(card.get("distribution"), dict):
            continue
        model = card.get("announcement_day_economics") or {}
        for case in model.get("outcome_cases", []):
            ids = case.get("source_contract_ids")
            if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str):
                continue
            candidates.append((event_day.isoformat(), ids[0], parent_id, family, card))

    selected_family_dates = set()
    for day, contract, parent_id, family, card in sorted(candidates, key=lambda row: row[:3]):
        if contract in existing_contracts or (family, day) in selected_family_dates:
            continue
        if len(additions) >= max_questions:
            exclusions.append({"contract_id": contract, "reason": "EXPLICIT_QUESTION_LIMIT_REACHED"})
            continue
        market = market_objects.get(contract)
        if contract in quarantined or market is None or getattr(market, "ticker", None) != contract:
            exclusions.append({"contract_id": contract, "reason": "MISSING_OR_QUARANTINED_SOURCE_IDENTITY"})
            continue
        try:
            close = evaluation_instant(market.close_time)
            if close <= now or close.date().isoformat() != day or close.date() > cutoff:
                raise ValueError("Contract time does not match the current source card")
            record = catalog_record(market)
            target = make_contract_target(record)
            child_id = "source-completion-" + sha256(parent_id.encode()).hexdigest()[:12] + "-" + contract
            if child_id in original_ids:
                continue
            target.update(
                event_id=child_id, family=family, event_date=day,
                parent_event_id=parent_id, candidate_origin=ORIGIN,
                question_we_care_about="Does this exact contract settle YES under its complete source rules: " + market.title,
                candidate_status="DISTINCT_SOURCE_SETTLEMENT_QUESTION_ORIGINAL_MACRO_MATCH_UNCHANGED",
                original_macro_question=card.get("original_question", card.get("title")),
                broad_macro_hypothesis_equivalence="NOT_ESTABLISHED",
                original_parent_was_rewritten=False, model_proposed_this_binary_question=False,
                selection_basis=SELECTION,
                source_completion_basis="ALREADY_ADMITTED_SAME_DATE_SOURCE_ATOM_NOT_NEW_ECONOMIC_EVIDENCE",
            )
            one = {"events": [target], "markets_by_event": {child_id: [record]},
                   "matches_by_event": {child_id: []}, "match_statuses": {},
                   "quarantined_provider_contracts": {}, "status": ORIGIN}
            checked = prepare_live_event_brief(
                one, market_objects, portfolio_rows=build_kwargs["portfolio_rows"],
                notional=build_kwargs["notional"], evaluated_at=now, execution_mode="LIVE",
            )["events"][0]
            contracts = checked.get("contracts", [])
            if len(contracts) != 1:
                raise ValueError("Independent exact identity is not unique")
            evidence = contracts[0]
            if (evidence.get("event_relationship") != "EQUIVALENT"
                    or evidence.get("event_relationship_status") != "CERTIFIED"
                    or not evidence.get("source_family_gate", {}).get("eligible")
                    or evidence.get("decision") != "CERTIFIED_EXACT_CONTRACT_IDENTITY"
                    or checked.get("probability", {}).get("pricing_measure_Q_interval") is None):
                raise ValueError("Independent exact identity, family/date, or fresh quote gate failed")
            binding = derive_exact_bindings({"events": [checked], "default_analysis": report.get("default_analysis", {})})
            if child_id not in binding["overrides"]:
                exclusions.append({"contract_id": contract, "reason": "EXISTING_ECONOMIC_BINDING_GATE_REJECTED",
                                   "binding_ledger": deepcopy(binding["ledger"])})
                continue
        except (ValueError, TypeError, KeyError, AttributeError):
            exclusions.append({"contract_id": contract, "reason": "EXISTING_SOURCE_IDENTITY_GATE_REJECTED"})
            continue
        discovery.setdefault("events", []).append(target)
        discovery.setdefault("markets_by_event", {})[child_id] = [record]
        discovery.setdefault("matches_by_event", {})[child_id] = []
        discovery.setdefault("match_statuses", {})[child_id] = "EXACT_SOURCE_QUESTION_REQUIRES_INDEPENDENT_IDENTITY_GATE"
        existing_contracts.add(contract)
        selected_family_dates.add((family, day))
        additions.append({"event_id": child_id, "contract_id": contract, "parent_event_id": parent_id,
                          "family": family, "event_date": day, "candidate_origin": ORIGIN,
                          "source_event_ticker": card["source_event_ticker"], "selection_basis": SELECTION,
                          "broad_macro_event_matched": False})
    params = dict(build_kwargs)
    params["discovery"] = discovery
    result = build_assessment_from_evidence(**params) if additions else deepcopy(report)
    rebuilt_ids = _identities(result)
    if any(rebuilt_ids.get(key) != value for key, value in original_ids.items()):
        raise ValueError("Source question completion changed an existing event identity or quote")
    if _distributions(result) != _distributions(report):
        raise ValueError("Source question completion changed the admitted source distributions")
    result["source_question_completion"] = {
        "schema": "castle.option1.source_question_completion.v1",
        "status": "BOUNDED_SOURCE_QUESTIONS_APPENDED" if additions else "NO_ADDITIONAL_ADMISSIBLE_SOURCE_QUESTION",
        "added_questions": additions, "exclusions": exclusions,
        "horizon_days": horizon_days, "max_questions": max_questions,
        "selection_basis": SELECTION, "original_questions_and_quotes_preserved": True,
        "source_distributions_preserved": True, "new_provider_or_model_calls": 0,
        "economic_assumptions_or_acceptance_gate_changed": False,
    }
    return result, params
