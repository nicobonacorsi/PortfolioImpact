"""Fixed economic-display and ranking policy, independent of market evidence.

Admission is not model selection or a new statistical test. In particular a
failed measured response remains in the report as a diagnostic: it is neither
replaced by zero nor replaced by whichever comparison happens to pass. Source
quotes remain usable independently of whether an economic impact is admitted.
"""
from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping


SCHEMA = "castle.model_admission.v1"


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _answer(state, reason, evidence_scope, *, primary=False, ranking=False,
            diagnostics=False):
    return {"schema": SCHEMA, "state": state,
            "primary_impact_allowed": primary, "economic_ranking_allowed": ranking,
            "diagnostic_impact_available": diagnostics, "reason": reason,
            "evidence_scope": evidence_scope, "forward_prediction_certified": False,
            "market_probability_independent": True,
            "policy": "FIXED_SCOPE_AND_REPORTED_EVIDENCE_GATE_NOT_HOLDOUT_MODEL_SELECTION"}


def _numeric_outcomes(rows):
    if not isinstance(rows, (list, tuple)):
        return False
    return any(isinstance(row, Mapping) and any(_finite(row.get(key)) for key in
               ("pnl_if_outcome_usd", "component_pnl_usd", "pnl")) for row in rows or [])


def _has_missing_positions(model):
    missing = model.get("missing_symbols") or model.get("unknown_symbols")
    weight = model.get("missing_weight")
    return bool(missing) or (_finite(weight) and weight > 1e-12)


def _intact_cash_outcome(row):
    if not isinstance(row, Mapping):
        return False
    points = [row[key] for key in ("pnl_if_outcome_usd", "pnl") if key in row]
    if not points or not all(_finite(value) and value == 0 for value in points):
        return False
    for key in ("pnl_model_range_5_95_usd", "pnl_interval"):
        bounds = row.get(key)
        if bounds is not None and not (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                                      and all(_finite(value) and value == 0 for value in bounds)):
            return False
    return True


def _intact_public_contract_outcome(row):
    """A public outcome is attached to its own named source contract, not a family."""
    if not isinstance(row, Mapping):
        return False
    identifier, source_ids, question = (row.get(key) for key in
                                       ("contract_id", "source_contract_ids", "question"))
    return (isinstance(identifier, str) and bool(identifier.strip())
            and isinstance(source_ids, list) and bool(source_ids)
            and all(isinstance(value, str) and bool(value.strip()) for value in source_ids)
            and identifier in source_ids
            and isinstance(question, str) and bool(question.strip())
            and row.get("partial") is False and row.get("no_channel") is False
            and _finite(row.get("pnl")) and _finite(row.get("q_reference"))
            and 0 <= row["q_reference"] <= 1)


def eligible_contract_scenarios(event):
    """Return the one shared eligibility set for source-contract ranking.

    The existence of one sound contract cannot admit a different partial or
    malformed contract with a larger numerical statistic. Returned objects are
    original read-only references; this function never repairs their evidence.
    """
    scenarios = _mapping(_mapping(event).get("contract_first")).get("contract_scenarios") or []
    if not isinstance(scenarios, (list, tuple)):
        return []
    eligible = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or scenario.get("status") != "DECLARED_CONTRACT_SCENARIO":
            continue
        identifier = _mapping(scenario.get("contract")).get("contract_id")
        spread = _mapping(scenario.get("summary")).get("q_standard_deviation_usd")
        gate = _mapping(scenario.get("instrument_identity_gate"))
        if (not isinstance(identifier, str) or not identifier.strip() or
                not _finite(spread) or spread < 0 or _has_missing_positions(scenario) or
                gate.get("blocked_symbols") or scenario.get("legacy_fallback_allowed") is False):
            continue
        eligible.append(scenario)
    return eligible


def event_model_admission(event):
    """Return a read-only admission decision for raw or public display events.

    Return states are EXACT_IDENTITY, CONDITIONAL_SCENARIO, DIAGNOSTIC_ONLY,
    RELATED_SOURCE_ONLY, and UNAVAILABLE. A CONDITIONAL_SCENARIO is usable under
    stated assumptions, never a certification of forecasting or causal skill.
    ``economic_ranking_allowed`` only admits an explicitly conditional model
    ranking. It does not turn market-price weights into physical probabilities.
    """
    event = _mapping(event)
    table = _mapping(event.get("outcome_table"))
    quality = _mapping(event.get("quality"))
    economics = _mapping(event.get("economics"))
    kind = str(quality.get("model_kind") or "")
    model = table or economics or quality
    status = str(table.get("status") or quality.get("model_status") or economics.get("status") or "")
    rows = table.get("outcomes") if table else economics.get("outcome_cases") or event.get("outcomes") or []
    diagnostics = _numeric_outcomes(rows)

    # Contradictory identity evidence and inherited hard blocks always win,
    # even if an older saved model carries a complete/PASS label.
    gates = [event, table, quality,
             _mapping(event.get("instrument_identity_gate")),
             _mapping(table.get("instrument_identity_gate")),
             _mapping(quality.get("instrument_identity_gate")),
             _mapping(event.get("instrument_identity"))]
    if any(g.get("blocked_symbols") or g.get("identity_blocked_symbols") or
           g.get("asset_class_blocked_symbols") or
           g.get("legacy_fallback_allowed") is False or
           g.get("status") == "BLOCKED_INSTRUMENT_IDENTITY" for g in gates):
        return _answer("UNAVAILABLE", "Instrument identity or required asset-class evidence is unresolved; "
                       "complete-portfolio impact and ranking are not admitted.",
                       "INSTRUMENT_SCOPE_GATE", diagnostics=diagnostics)

    # A useful calculation about another proposition is not an answer to the
    # original question, even when the country/channel is economically related.
    tariff = _mapping(event.get("tariff_review"))
    scopes = [event, table, quality, tariff,
              _mapping(event.get("target_scope_review")),
              _mapping(table.get("target_scope_review")),
              _mapping(tariff.get("target_scope_review")),
              _mapping(quality.get("target_scope_review"))]
    separate_source_scenario = event.get("source_defined_scenario") is True
    if not separate_source_scenario and any(s.get("source_scenario_is_exact_target_match") is False for s in scopes):
        return _answer("RELATED_SOURCE_ONLY", "The priced source scenario has a different settlement "
                       "question or date. Its impact cannot price or rank this target event.",
                       "RELATED_SOURCE_SCENARIO_NOT_ORIGINAL_TARGET", diagnostics=diagnostics)

    if (status.startswith(("PARTIAL", "BLOCKED", "UNSUPPORTED")) or
            _has_missing_positions(model)):
        return _answer("UNAVAILABLE", "A complete supported-portfolio response is not available; "
                       "partial calculations are diagnostics only.",
                       "INCOMPLETE_PORTFOLIO_MODEL", diagnostics=diagnostics)

    display_statuses = {"CHANNEL_AWARE": "COMPLETE_DECLARED_MODEL",
                        "MEASURED_PAYROLL": "COMPLETE_MEASURED_SURPRISE_RESPONSE",
                        "LEGACY_BETA": "DECLARED_DEFAULT_COMPONENT_MODEL",
                        "TARIFF_ANCHOR": "COMPLETE_MEASURED_ANCHOR_MODEL",
                        "MEASURED_ANCHOR": "COMPLETE_MEASURED_ANCHOR_MODEL",
                        "NOMINAL_CASH_IDENTITY": "IDENTIFIED_NOMINAL_USD_CASH_ZERO",
                        "DECLARED_CONTRACT_SCENARIO": "DECLARED_CONTRACT_SCENARIO"}
    display_form = not table and not economics
    display_complete = (display_form and kind in display_statuses
                        and status in {"", display_statuses[kind]})
    public_contract = display_form and kind == "DECLARED_CONTRACT_SCENARIO"
    if public_contract and not (status == "DECLARED_CONTRACT_SCENARIO" and isinstance(rows, list)
                                and bool(rows) and all(_intact_public_contract_outcome(row) for row in rows)):
        return _answer("UNAVAILABLE", "The saved contract scenarios do not retain complete matching source "
                       "identifiers, questions, market weights and supported impact rows. No broader-event estimate is inferred.",
                       "INCOMPLETE_PUBLIC_SOURCE_CONTRACT_SCOPE", diagnostics=diagnostics)
    if display_form and diagnostics and not display_complete:
        # Hard model-scope inconsistency takes precedence over a reported FAIL:
        # the failed check cannot establish what this incompatible model was.
        return _answer("UNAVAILABLE", "The saved model kind and status do not identify a compatible "
                       "complete response. Recorded values and checks remain diagnostic evidence only.",
                       "INCOMPATIBLE_PUBLIC_MODEL_METADATA", diagnostics=True)

    anchor_model = (kind in {"MEASURED_ANCHOR", "TARIFF_ANCHOR"}
                    or status == "COMPLETE_MEASURED_ANCHOR_MODEL")
    if diagnostics and anchor_model and not separate_source_scenario:
        return _answer("RELATED_SOURCE_ONLY", "The tariff-anchor amounts are not attested as a separately "
                       "defined source-rate scenario. They cannot price or economically rank the broader "
                       "tariff-action or action-window question. Source dates and questions are not inferred from labels.",
                       "UNATTESTED_SEPARATE_TARIFF_SOURCE_SCOPE", diagnostics=True)

    if status == "IDENTIFIED_NOMINAL_USD_CASH_ZERO" or kind == "NOMINAL_CASH_IDENTITY":
        exact_rows = (status in {"", "IDENTIFIED_NOMINAL_USD_CASH_ZERO"}
                      and isinstance(rows, (list, tuple)) and bool(rows)
                      and all(_intact_cash_outcome(row) for row in rows))
        if not exact_rows:
            return _answer("UNAVAILABLE", "The saved nominal-cash identity does not contain intact zero-price outcomes.",
                           "MALFORMED_CASH_IDENTITY", diagnostics=diagnostics)
        return _answer("EXACT_IDENTITY", "Fixed nominal USD cash has zero price change in this model; "
                       "interest, inflation and foreign exchange are outside this identity.",
                       "NOMINAL_USD_CASH_ACCOUNTING_IDENTITY", primary=True, ranking=True, diagnostics=True)

    # Explicit failure evidence is monotone: a conflicting PASS label in a
    # different layer cannot erase a reported FAIL. Component association
    # failures of a macro what-if model are deliberately not portfolio tests.
    # ``economics`` is a separate legacy comparison when a primary table
    # exists. Its result must not be relabelled as the selected table's test.
    selected_evidence = (event, model, quality)
    reliabilities = [_mapping(item.get("model_reliability")) for item in selected_evidence]
    portfolio_statuses = [str(_mapping(item.get("portfolio_transfer_check")).get("status") or "")
                          for item in reliabilities]
    portfolio_statuses += [str(item.get("portfolio_transfer_status") or "")
                           for item in selected_evidence]
    transfer_statuses = [str(item.get(key) or "") for item in selected_evidence
                         for key in ("scenario_transfer_status", "transfer_status")]
    transfer_statuses += [str(item.get(key) or "") for item in reliabilities
                          for key in ("status", "scenario_transfer_status")]
    measured_payroll = status == "COMPLETE_MEASURED_SURPRISE_RESPONSE" or kind == "MEASURED_PAYROLL"
    if diagnostics and ("FAIL" in portfolio_statuses or measured_payroll and "FAIL" in transfer_statuses):
        return _answer("DIAGNOSTIC_ONLY", "The measured response did not pass its reported historical check. "
                       "Its numerical results are retained for inspection but cannot serve as the primary impact estimate or economic ranking.",
                       "REPORTED_FAILED_RESPONSE_CHECK", diagnostics=True)

    if public_contract:
        return _answer("CONDITIONAL_SCENARIO", "This result prices the selected source-contract question "
                       "under stated stress assumptions, not the broader research hypothesis. "
                       "No probability or economic ranking is assigned to the broader target.",
                       "SOURCE_IDENTIFIED_CONTRACT_STRESS_SCENARIO", primary=True, diagnostics=True)

    complete = bool(table or economics) and status in {
        "COMPLETE_DECLARED_MODEL", "COMPLETE_MEASURED_SURPRISE_RESPONSE",
        "COMPLETE_MEASURED_ANCHOR_MODEL", "DECLARED_DEFAULT_COMPONENT_MODEL"}
    legacy_complete = not table and bool(economics.get("outcome_cases")) and not status
    if diagnostics and (complete or display_complete or legacy_complete):
        if separate_source_scenario:
            return _answer("CONDITIONAL_SCENARIO", "A separately defined source-contract scenario under stated assumptions; "
                           "its impact does not price or rank the related original target event.",
                           "SEPARATE_SOURCE_DEFINED_RESPONSE_NOT_ORIGINAL_TARGET", primary=True, diagnostics=True)
        return _answer("CONDITIONAL_SCENARIO", "A conditional response under the stated model assumptions; "
                       "not an independently certified forecast of portfolio returns.",
                       "MODEL_CONDITIONAL_RESPONSE_NOT_FUTURE_TOTAL_RETURN", primary=True, ranking=True, diagnostics=True)

    # Existing explicitly named geopolitical contract scenarios may be ranked
    # under their own contract ID. This never supplies a numeric estimate for
    # the broader unmatched research question.
    if eligible_contract_scenarios(event):
        return _answer("CONDITIONAL_SCENARIO", "Only separately identified source-contract stress scenarios "
                       "are economically ranked; they do not establish the broader target's probability or total impact.",
                       "SOURCE_IDENTIFIED_CONTRACT_STRESS_SCENARIO", ranking=True, diagnostics=True)

    return _answer("UNAVAILABLE", "No complete supported economic response is available for this event. "
                   "Market evidence, where available, remains independently inspectable.",
                   "NO_ADMITTED_COMPLETE_RESPONSE", diagnostics=diagnostics)


def apply_model_admission_to_event(event):
    """Annotate an engine event in place and gate only its primary rankings.

    Numerical model tables, source distributions and original validation checks
    are unchanged. Unadmitted ranking statistics are preserved as diagnostics.
    Reapplying this operation is idempotent for those preserved diagnostics.
    """
    admission = event_model_admission(event)
    event["model_admission"] = admission
    if admission["economic_ranking_allowed"]:
        for key in ("risk_ranking", "ranking"):
            if isinstance(event.get(key), dict):
                event[key]["economic_scope"] = admission["evidence_scope"]
                event[key]["forward_prediction_certified"] = False
        return admission
    if "economic_ranking_diagnostics" not in event:
        event["economic_ranking_diagnostics"] = {
            "risk_ranking": deepcopy(event.get("risk_ranking") or {}),
            "component_ranking": deepcopy(event.get("ranking") or {}),
            "reason_not_admitted": admission["reason"],
            "diagnostic_only": True,
        }
    event["risk_ranking"] = {"q_standard_deviation_usd": None,
                             "basis": admission["reason"],
                             "economic_scope": admission["evidence_scope"],
                             "forward_prediction_certified": False}
    event["ranking"] = {"abs_delta_pnl_usd": None, "abs_delta_pnl_interval_usd": None,
                        "selected_outcome_id": None, "basis": admission["reason"],
                        "economic_scope": admission["evidence_scope"],
                        "forward_prediction_certified": False}
    return admission
