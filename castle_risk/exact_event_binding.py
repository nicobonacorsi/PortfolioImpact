"""Exact YES/NO economic bindings derived from the default source-outcome model.

The original v2 gate (``economic_binding.complete_bound_case``) counts an event
only when an exact source contract carries explicit, event-specific YES and NO
assumptions. Family stresses cannot satisfy it. This module derives those
assumptions for an exact contract from the same joint source distribution and
published response model that produce the default announcement components:

    YES state     x_j                    (the contract's own settlement outcome)
    NO  state     E_Q[x | not j]         (Q-reference mixture of every other atom)
    SPY contrast  c * (x_j - E_Q[x | not j]),  interval c_bounds x [x_j_lo - NO_hi, x_j_hi - NO_lo]

where c is the published broad-equity slope. A binding is produced only when
the contract is exactly one source atom (a band built from two threshold
contracts is not that contract's YES event), the atom's own state is finite,
and the complement mean has a finite LP range. Anything else stays unbound with
its reason. Nothing here changes the gate, the engines, or the source question.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from .event_distributions import MAX_QUOTE_RELAXATION, QUOTE_RELAXATION_LP_SLACK

SCHEMA = "castle.option1.exact_event_bindings.v1"
DERIVED_ORIGINS = {"SOURCE_DEFINED_BINARY_REFINEMENT_NOT_CLAUDE", "SOURCE_DEFINED_CONTRACT_FALLBACK_NOT_CLAUDE"}


def _finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def _product(a, b):
    values = [x * y for x in a for y in b]
    return [min(values), max(values)]


def _admissible_interval(values, *, domain=None):
    """Return ordered finite bounds or None; never repair malformed source ranges."""
    if not isinstance(values, (list, tuple)) or len(values) != 2 or not _finite(*values):
        return None
    lower, upper = values
    if lower > upper:
        return None
    if domain is not None and (lower < domain[0] or upper > domain[1]):
        return None
    return lower, upper


def _inside(value, bounds):
    return _finite(value) and bounds[0] <= value <= bounds[1]


def _source_quote_relaxation(card):
    """The epsilon* the source polytope was built with; 0 unless validly disclosed."""
    relaxation = (card.get("distribution") or {}).get("quote_relaxation") or {}
    epsilon = relaxation.get("epsilon")
    if (relaxation.get("status") != "MINIMAL_UNIFORM_QUOTE_RELAXATION_APPLIED" or not _finite(epsilon)
            or not 0 < epsilon <= MAX_QUOTE_RELAXATION):
        return 0.0
    return epsilon


def _source_cases(default_analysis, contract_id, specification):
    """Find *all* candidates; first-match is unsafe if the source is ambiguous."""
    candidates = []
    for card in (default_analysis or {}).get("events", []):
        if not isinstance(card, dict):
            continue
        model = card.get(specification)
        if not isinstance(model, dict):
            continue
        for case in model.get("outcome_cases", []):
            if isinstance(case, dict) and contract_id in case.get("source_contract_ids", []):
                candidates.append((card, model, case))
    return candidates


def derive_exact_bindings(report, *, specification="announcement_day_economics"):
    """Scenario overrides for exact source contracts, plus an explicit ledger of the rest."""
    if specification not in {"announcement_day_economics", "economics"}:
        raise ValueError("Unknown economic specification")
    default_analysis = report.get("default_analysis") or {}
    overrides, ledger = {}, []
    for event in report.get("events", []):
        contract = event.get("contract_id")
        exact = event.get("definition_origin") == "CONTRACT_IDENTITY" and isinstance(contract, str)
        if not exact:
            continue
        row = {"event_id": event["event_id"], "contract_id": contract, "title": event.get("title"),
               "candidate_origin": event.get("candidate_origin")}
        candidates = _source_cases(default_analysis, contract, specification)
        if not candidates:
            ledger.append({**row, "status": "UNBOUND_NO_ADMITTED_SOURCE_DISTRIBUTION_CONTAINS_CONTRACT",
                           "reason": "The contract is not an atom of any admitted default source distribution at this evaluation time."})
            continue
        if len(candidates) != 1:
            ledger.append({**row, "status": "UNBOUND_AMBIGUOUS_SOURCE_DISTRIBUTIONS",
                           "reason": "More than one source distribution contains this settlement identifier; never choose the first one.",
                           "candidate_count": len(candidates)})
            continue
        card, model, case = candidates[0]
        if case.get("source_contract_ids") != [contract]:
            ledger.append({**row, "status": "UNBOUND_ATOM_IS_NOT_THIS_CONTRACT_ALONE",
                           "reason": "The source atom combines several contracts; its outcome is not this contract's YES event."})
            continue
        state = case.get("state_model") or {}
        x_interval = state.get("value_interval")
        priced = case.get("priced_in") or {}
        no_mean, no_interval = priced.get("NO_mean_state"), priced.get("NO_mean_state_interval")
        response = model.get("response_model") or {}
        c, c_bounds = response.get("coefficient"), response.get("coefficient_interval")
        value_bounds = _admissible_interval(x_interval)
        no_bounds = _admissible_interval(no_interval)
        coefficient_bounds = _admissible_interval(c_bounds)
        if (not _finite(state.get("value"), no_mean, c) or value_bounds is None
                or no_bounds is None or coefficient_bounds is None):
            ledger.append({**row, "status": "UNBOUND_NONFINITE_STATE_OR_COMPLEMENT",
                           "reason": "The atom's own state, the complement mean or the response range is not a finite ordered interval."})
            continue
        if (not _inside(state["value"], value_bounds) or not _inside(no_mean, no_bounds)
                or not _inside(c, coefficient_bounds)):
            ledger.append({**row, "status": "UNBOUND_INCOHERENT_ECONOMIC_INTERVALS",
                           "reason": "A central economic estimate lies outside its declared admissible model interval."})
            continue
        q_interval = case.get("q_interval")
        source_q_bounds = _admissible_interval(q_interval, domain=(0., 1.))
        event_probability = event.get("probability") or {}
        event_quote_supplied = "pricing_measure_Q_interval" in event_probability
        event_q_interval = event_probability.get("pricing_measure_Q_interval")
        # An old or explicitly non-probabilistic research target may have no
        # independent event quote. This permits only a source-based ECONOMIC
        # branch definition; it does not fill the missing event probability,
        # and therefore cannot by itself pass complete_bound_case. If a quote
        # IS supplied, it must agree with the admitted source distribution.
        event_q_bounds = (_admissible_interval(event_q_interval, domain=(0., 1.))
                          if event_quote_supplied else None)
        if source_q_bounds is None or (event_quote_supplied and event_q_bounds is None):
            ledger.append({**row, "status": "UNBOUND_INVALID_SOURCE_OR_EVENT_Q_INTERVAL",
                           "reason": "Each supplied source/event pricing-Q interval must be finite, ordered and inside [0,1]."})
            continue
        epsilon = _source_quote_relaxation(card)
        if event_quote_supplied and epsilon:
            # The source polytope admits every quote widened by epsilon*; the
            # event quote is compared on that same relaxed basis, not raw.
            slack = epsilon + QUOTE_RELAXATION_LP_SLACK
            event_q_bounds = (max(0., event_q_bounds[0] - slack), min(1., event_q_bounds[1] + slack))
            row["event_quote_relaxation_epsilon"] = epsilon
        overlap = (max(source_q_bounds[0], event_q_bounds[0]),
                   min(source_q_bounds[1], event_q_bounds[1])) if event_quote_supplied else source_q_bounds
        if overlap[0] > overlap[1]:
            ledger.append({**row, "status": "UNBOUND_EVENT_QUOTE_CONTRADICTS_SOURCE_POLYTOPE",
                           "reason": "The matched event's quote and the settlement distribution admit disjoint pricing-Q intervals.",
                           "source_q_interval": q_interval, "event_q_interval": event_q_interval})
            continue
        if (overlap[1] <= 0 or overlap[0] >= 1
                or not _inside(case.get("q_reference"), overlap)):
            ledger.append({**row, "status": "UNBOUND_DEGENERATE_OR_INCOMPATIBLE_Q_REFERENCE",
                           "reason": "The complement mixture has no non-degenerate common pricing reference inside both admitted Q intervals."})
            continue
        surprise = state["value"] - no_mean
        surprise_interval = [x_interval[0] - no_interval[1], x_interval[1] - no_interval[0]]
        contrast = c * surprise
        contrast_interval = _product(c_bounds, surprise_interval)
        if (not _finite(surprise, *surprise_interval, contrast, *contrast_interval)
                or not _inside(contrast, contrast_interval)):
            ledger.append({**row, "status": "UNBOUND_NONFINITE_DERIVED_IMPACT",
                           "reason": "Derived return or scenario interval is not finite and internally consistent."})
            continue
        unit = response.get("state_unit")
        binding = {
            "event_id": event["event_id"], "contract_id": contract,
            "YES": f"{contract} settles YES under its complete source rules; settlement state {state['value']} {unit}.",
            "NO": f"{contract} settles NO: any other outcome of {card.get('source_event_ticker')} (including source exceptions).",
            "no_outcome_mixture": (f"Conditional mean of the other source atoms under the joint reference Q "
                                   f"({model.get('family')}): {no_mean:.6g} {unit}, LP range "
                                   f"[{no_interval[0]:.6g}, {no_interval[1]:.6g}] over the source polytope."),
            "price_basis": ("SPY same-session announcement return contrast YES minus NO = published slope x state "
                            "contrast; instruments via the original v2 engine's same-family event-day betas "
                            "with explicit daily fallback."),
            "event_window": f"Announcement session {model.get('event_date')} (close-to-close).",
            "assumption_basis": (f"{(response.get('source') or {}).get('title')}; {(response.get('source') or {}).get('locator')}. "
                                 "Declared historical transfer, not a causal or forecast certification."),
            "return_unit": "fractional_return", "includes_source_exceptions": True,
        }
        overrides[event["event_id"]] = {
            "benchmark_return_contrast": contrast, "benchmark_contrast_interval": contrast_interval,
            "binding": binding,
        }
        ledger.append({**row, "status": "BINDING_DERIVED_FROM_DEFAULT_SOURCE_MODEL",
                       "source_event_ticker": card.get("source_event_ticker"), "specification": specification,
                       "joint_q_reference": case.get("q_reference"), "q_interval": q_interval,
                       "event_probability_q_interval": (event.get("probability") or {}).get("pricing_measure_Q_interval"),
                       "yes_state": state["value"], "yes_state_interval": x_interval,
                       "no_mean_state": no_mean, "no_mean_state_interval": no_interval,
                       "state_contrast": surprise, "state_contrast_interval": surprise_interval,
                       "benchmark_return_contrast": contrast, "benchmark_contrast_interval": contrast_interval,
                       "default_model_delta_pnl_usd": case.get("delta_pnl_usd"),
                       "default_model_delta_pnl_interval_usd": case.get("delta_pnl_interval_usd")})
    body = json.dumps(overrides, sort_keys=True, allow_nan=False).encode()
    return {"schema": SCHEMA, "overrides": overrides, "ledger": ledger,
            "overrides_sha256": hashlib.sha256(body).hexdigest(),
            "bound": sum(r["status"] == "BINDING_DERIVED_FROM_DEFAULT_SOURCE_MODEL" for r in ledger),
            "unbound": sum(r["status"] != "BINDING_DERIVED_FROM_DEFAULT_SOURCE_MODEL" for r in ledger),
            "gate_changed": False,
            "interpretation": ("Each binding states one exact contract's YES and complementary NO as source states and "
                               "maps their contrast to a SPY return with the published slope. The original v2 engine then "
                               "computes positions, priced-in YES/NO and the complete-case gate unchanged.")}


def bound_case_summary(report, derived):
    """Per bound event: the v2 gate result and the engine's numbers, side by side with the default model."""
    from .economic_binding import complete_bound_case
    by_id = {e["event_id"]: e for e in report.get("events", [])}
    rows = []
    for entry in derived["ledger"]:
        event = by_id.get(entry["event_id"])
        if event is None:
            continue
        impact = event.get("impact") or {}
        priced = impact.get("priced_in") or {}
        rows.append({"event_id": entry["event_id"], "contract_id": entry["contract_id"], "title": event.get("title"),
                     "binding_status": (event.get("economic_binding") or {}).get("status"),
                     "complete_bound_case": complete_bound_case(event),
                     "q_interval": (event.get("probability") or {}).get("pricing_measure_Q_interval"),
                     "delta_pnl_usd": impact.get("delta_pnl_usd"),
                     "delta_pnl_interval_usd": deepcopy(impact.get("delta_pnl_interval_usd")),
                     "yes_pnl_usd": priced.get("yes_pnl_usd"), "no_pnl_usd": priced.get("no_pnl_usd"),
                     "yes_pnl_interval_usd": deepcopy(priced.get("yes_pnl_interval_usd")),
                     "no_pnl_interval_usd": deepcopy(priced.get("no_pnl_interval_usd")),
                     "engine_q_reference": priced.get("q_reference"),
                     "joint_q_reference": entry.get("joint_q_reference"),
                     "default_model_delta_pnl_usd": entry.get("default_model_delta_pnl_usd"),
                     "default_model_delta_pnl_interval_usd": entry.get("default_model_delta_pnl_interval_usd"),
                     "positions": [{k: p.get(k) for k in ("symbol", "weight", "return_contrast", "return_contrast_interval",
                                                          "delta_pnl_usd", "delta_pnl_interval_usd", "basis", "status")}
                                   for p in impact.get("positions", [])],
                     "derivation_status": entry["status"]})
    return rows


__all__ = ["SCHEMA", "bound_case_summary", "derive_exact_bindings"]


def close_exact_source_scenarios(report, *, build_kwargs, scenario_settings=None):
    """Admit only contract-specific YES/NO scenarios justified by an existing source polytope.

    ``build_kwargs`` are the *same archived inputs* used to build ``report``;
    the deterministic second assembly makes no provider/model request and
    cannot take a quote from a sibling contract or assign it to a broad event.
    Explicit existing per-event operator choices are preserved, never replaced.
    """
    from .live_assessment import build_assessment_from_evidence
    from .economic_binding import complete_bound_case

    derived = derive_exact_bindings(report, specification='announcement_day_economics')
    selected = {k: v for k, v in derived['overrides'].items()
                if k not in (scenario_settings or {})}
    if selected:
        settings = {**(scenario_settings or {}), **selected}
        if build_kwargs.get('scenario_settings') not in (None, scenario_settings):
            raise ValueError('A source scenario cannot overwrite an independently supplied specification')
        params = dict(build_kwargs)
        params['scenario_settings'] = settings
        replay = build_assessment_from_evidence(**params)
        def source_identity(r):
            return {e['event_id']: (e.get('contract_id'), e.get('title'), e.get('parent_event_id'),
                                    e.get('probability'), e.get('candidate_origin'))
                    for e in r['events']}
        if source_identity(replay) != source_identity(report):
            raise ValueError('Source-derived economic closure changed a settlement, quote or research event')
        # Source-defined model outcomes are unchanged, even if the portfolio's
        # event-specific branch assumptions become explicit in the second pass.
        def distribution_identities(r):
            return [(d.get('source_event_ticker'), d.get('status'),
                     d.get('distribution'))
                    for d in r.get('default_analysis', {}).get('events', [])]
        if distribution_identities(replay) != distribution_identities(report):
            raise ValueError('Admitted source outcome distributions unexpectedly changed')
        report = replay
    report['source_exact_binding_derivation'] = {
        'schema': derived['schema'],
        'applied_event_ids': sorted(selected),
        'bound_by_admitted_distribution': derived['bound'],
        'unbound_by_admitted_distribution': derived['unbound'],
        'ledger': derived['ledger'],
        'bound_case_results': bound_case_summary(report, derived),
        'completed_exact_scenarios': sum(complete_bound_case(e) for e in report['events']),
        'scope': ('The narrow Kalshi settlement question receives a declared scenario '
                  'from a coherent source polytope; the related broad research '
                  'question never inherits its Q or its economic contrast. '
                  'The portfolio P&L is a model scenario, not a causal prediction.'),
        'paid_inference_or_provider_requests_added': 0,
    }
    return report
