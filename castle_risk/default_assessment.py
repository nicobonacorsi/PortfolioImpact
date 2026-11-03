"""The primary Option 1 projection: outcome probabilities and declared economics.

Existing exact-contract reports remain intact. This projection explicitly
defines the narrower source outcome families, joins a documented default
economic model, and keeps incompatible or unavailable evidence visible.
No acquisition, clock read, or provider credential is used by this module.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
import re

from .default_event_economics import (
    ANNOUNCEMENT_DAY_SPECIFICATION, build_default_channel_sensitivity,
    build_default_macro_economics, build_tlt_duration_comparison,
)
from .discovery_context import aggregated_issuer_exposure, named_exposure_sensitivity
from .fund_composition import (FundCompositionError, analyze_holdings_snapshot,
                               portfolio_rate_sensitivity, review_rate_dollar_binding)
from .event_impact import load_duration_evidence
from .event_distributions import build_distributions_from_markets
from .outcome_table import build_outcome_table, holdings_quality_gaps, instrument_model_gate
from .instrument_exposures import build_instrument_exposures
from .instrument_identity import resolve_portfolio_types
from .model_admission import apply_model_admission_to_event, eligible_contract_scenarios
from .source_class_quality import review_portfolio_source_classes
from .contract_scenarios import contract_first_review
from .tariff_scenarios import is_tariff_risk, tariff_review
from .payroll_scenarios import FAMILY as PAYROLL_FAMILY, payroll_card, payroll_distributions, scheduled_payroll_events
from .source_grounding import _CALENDAR as OFFICIAL_MACRO_CALENDAR
from .event_subject_intent import explicit_non_tariff_agreement_title
from .scheduled_event_timing import reviewed_macro_window
from .portfolio import parse_portfolio, recover_exposures
from .business_interruption_assessment import build_business_interruption_component
from .mexico_tariff_transmission import build_mexico_tariff_component
from .event_factor_what_if import build_factor_what_if


_GLD_DOCUMENTARY_CLASS = {
    "symbol": "GLD", "instrument_class": "commodity_etf", "underlying_asset": "gold bullion",
    "source_url": "https://www.spdrgoldshares.com/usa/gld/",
    "source_locator": "US GLD page, ticker and Key Features investment objective",
    "review_date": "2026-10-21",
    "classification_basis": "ISSUER_DOCUMENTARY_OBJECTIVE_REVIEW_NOT_MASSIVE_TAXONOMY",
    "summary": "The issuer identifies GLD as shares intended to track gold bullion after trust expenses.",
    "holdings_quality_repaired": False,
    "economic_channel": "Gold prices may respond to real yields and the dollar; the model does not identify that separate elasticity.",
}


def _family(event):
    if explicit_non_tariff_agreement_title(str(event.get("title", ""))):
        return "OTHER"
    family = event.get("family")
    new_research_families = {"HORMUZ_ENERGY", "OPEC_SUPPLY", "RUSSIA_ENERGY_SANCTIONS",
                             "US_DRUG_PRICING", "FDA_DRUG_APPROVAL", "BANK_CREDIT_STRESS", "BANK_CAPITAL_RULES"}
    text = " ".join(str(event.get(key, "")) for key in
                    ("title", "description", "question_we_care_about", "event_id")).lower()
    from .geopolitical_market_search import _family as research_route
    routed = research_route(event)
    # A policy label supplied by an LLM is not proof of an interest-rate
    # decision. Basel/bank-capital rules do not settle with FOMC rate quotes.
    if family == "FED_POLICY" and routed == "BANK_CAPITAL_RULES":
        return "BANK_CAPITAL_RULES"
    # One binary question cannot be simultaneously a pharmaceutical import
    # tariff and an OR-alternative Medicare drug-pricing decision. It must be
    # decomposed and sourced before attaching either family's outcomes.
    mixed_pharma_predicate = bool(re.search(
        r"\b(?:drug[- ]pricing|medicare.{0,35}negotiat|fda.{0,35}(?:approval|application))\b", text))
    if (family == "TRADE_TARIFFS" and mixed_pharma_predicate and is_tariff_risk(text)
            and re.search(r"\b(?:or|either)\b", text)):
        return "OTHER"
    if family in new_research_families:
        return family
    if family in {"FED_POLICY", "US_INFLATION", "TAIWAN_SUPPLY", "EXPORT_CONTROLS", "TRADE_TARIFFS", "US_PAYROLLS"}:
        return family
    if "taiwan" in text:
        return "TAIWAN_SUPPLY"
    if is_tariff_risk(text):
        return "TRADE_TARIFFS"
    if re.search(r"export[ _-]control|chip[ _-]export|semiconductor|ai.chip", text):
        return "EXPORT_CONTROLS"
    if routed in new_research_families:
        return routed
    return family


def _ranking(model):
    cases = model.get("outcome_cases", [])
    candidates = []
    for case in cases:
        if (case.get("q_interval") or [0.0, 0.0])[1] <= 1e-10:
            continue
        source_bounds = case.get("state_model", {}).get("source_value_interval", [None, None])
        if any(bound is None for bound in source_bounds):
            continue
        value = case.get("delta_pnl_usd")
        bounds = case.get("delta_pnl_interval_usd")
        if value is None or bounds is None or not math.isfinite(value):
            continue
        candidates.append((abs(value), case["outcome_id"], bounds))
    if not candidates:
        return {"abs_delta_pnl_usd": None, "abs_delta_pnl_interval_usd": None,
                "selected_outcome_id": None, "basis": "NO_COMPLETE_MACRO_COMPONENT_CONTRAST"}
    point, selected, bounds = max(candidates, key=lambda row: (row[0], row[1]))
    low, high = bounds
    return {"abs_delta_pnl_usd": point,
            "abs_delta_pnl_interval_usd": [0.0 if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high))],
            "selected_outcome_id": selected,
            "basis": "LARGEST_FINITE_SOURCE_OUTCOME_WITH_POSITIVE_POSSIBLE_Q_VS_COMPLEMENT_COMPONENT_CONTRAST",
            "unbounded_source_tail_stresses_excluded_from_primary_order": True,
            "complement_expectation_may_still_require_declared_tail_assumptions": bool(model.get("unrestricted_support_review", {}).get("finite_economic_domain_is_declared_assumption")),
            "range_is_for_selected_outcome_not_a_rank_confidence_interval": True}


def _fund_compositions(inputs, *, evaluated_at):
    """Signed composition of every captured ETF batch; failures stay visible."""
    from .primary_treasury_transcription import terms_available_for_assessment
    verified_terms, terms_receipt = terms_available_for_assessment(evaluated_at)
    result = {}
    metadata = inputs.get("instrument_metadata") or {}
    for symbol, holdings in sorted((inputs.get("holdings") or {}).items()):
        if not isinstance(holdings, dict):
            continue
        name = str((metadata.get(symbol) or {}).get("name") or "").upper()
        try:
            result[symbol] = analyze_holdings_snapshot(
                holdings, fund_name_terms=[name] if name else [], instrument_terms=verified_terms)
            recovered = ((result[symbol].get("interest_rate_sensitivity") or {}).get(
                "coverage_status_weights") or {}).get("TERMS_FROM_PRIMARY_INSTRUMENT_SOURCE", 0.)
            if recovered:
                result[symbol]["primary_terms_recovery"] = {
                    **terms_receipt, "recovered_nav_weight": recovered,
                    "valuation_uses_original_provider_prices": True,
                    "not_a_new_provider_acquisition": True,
                    "unmodeled_event_to_yield_response_remains_unvalidated": True,
                }
        except (FundCompositionError, OSError, ValueError, KeyError, TypeError) as exc:
            result[symbol] = {"status": "SIGNED_COMPOSITION_UNAVAILABLE", "reason": str(exc)}
    return result


def _quotient_envelope(numerator, denominator):
    """Range of n/d over the rectangle numerator x denominator; None if d can be zero."""
    if denominator[0] <= 0 <= denominator[1]:
        return None
    values = [n / d for n in numerator for d in denominator]
    return [min(values), max(values)]


def _implied_yield_check(economics, rate):
    """Yield move implied by each announcement-day bond component, given covered duration.

A conditional diagnostic, not assumption-free arithmetic. The numerator is the
whole-ETF announcement component; the denominator is the dollar sensitivity of
the covered bullet bonds only, at the composition date. The main interval
propagates both the component interval and the DV01 interval over their
rectangle (independent inputs, no joint dependence modeled); a denominator
interval that reaches zero gives no finite bound. A separately labeled variant
scales the denominator to all fixed-income NAV under the assumption that
uncovered bonds share the covered average duration.
    """
    by_symbol = {row["symbol"]: row for row in rate.get("positions", [])}
    cases, rejected_bindings = [], []
    for case in economics.get("outcome_cases", []):
        rows = []
        for position in case.get("positions", []):
            sens = by_symbol.get(position["symbol"])
            if not sens or position.get("component_pnl_usd") is None or not sens["usd_per_bp_parallel_rise"]:
                continue
            binding = review_rate_dollar_binding(sens, symbol=position["symbol"],
                portfolio_notional_usd=economics.get("portfolio_notional_usd"),
                position_weight=position.get("weight"), position_asset_type=position.get("asset_type"))
            if not binding["binding_valid"]:
                rejected_bindings.append({"outcome_id": case["outcome_id"], "symbol": position["symbol"],
                                          "binding_review": binding})
                continue
            per_bp = sens["usd_per_bp_parallel_rise"]
            numerator = position["component_pnl_interval_usd"]
            per_bp_interval = sens.get("usd_per_bp_interval") or [per_bp, per_bp]
            envelope = _quotient_envelope(numerator, per_bp_interval)
            covered = sens.get("covered_fund_nav_weight") or 0.0
            uncovered = sens.get("uncovered_fixed_income_fund_nav_weight") or 0.0
            scale = (covered + uncovered) / covered if covered > 0 and covered + uncovered > 0 else None
            rows.append({"symbol": position["symbol"], "component_pnl_usd": position["component_pnl_usd"],
                         "implied_parallel_yield_move_bp": position["component_pnl_usd"] / per_bp,
                         "implied_parallel_yield_move_interval_bp": envelope,
                         "interval_status": ("COMPONENT_AND_DV01_RECTANGLE" if envelope is not None
                                             else "DV01_INTERVAL_CONTAINS_ZERO_NO_FINITE_BOUND"),
                         "point_dv01_only_interval_bp": sorted(x / per_bp for x in numerator),
                         "usd_per_bp_interval": list(per_bp_interval),
                         "covered_bond_basis": sens["basis"],
                         "coverage_alignment": {
                             "status": "MISALIGNED_NUMERATOR_WHOLE_ETF_DENOMINATOR_COVERED_BONDS",
                             "covered_fund_nav_weight": covered,
                             "uncovered_fixed_income_fund_nav_weight": uncovered,
                             "equal_duration_variant_bp": (None if scale is None else
                                 position["component_pnl_usd"] / (per_bp * scale)),
                             "equal_duration_variant_interval_bp": (None if scale is None or envelope is None else
                                 _quotient_envelope(numerator, [x * scale for x in per_bp_interval])),
                             "equal_duration_variant_assumption": "Uncovered fixed-income NAV has the covered average duration; not verified, and false for mortgage or inflation-linked holdings.",
                         }})
        if rows:
            cases.append({"outcome_id": case["outcome_id"], "positions": rows})
    return {"schema": "castle.implied_yield_consistency.v2", "outcome_cases": cases,
            "rejected_dollar_binding_positions": rejected_bindings,
            "interpretation": "If the bond ETF moved only through a parallel shift of its covered bullet bonds, this is the yield change the announcement-day component implies. It checks direction and scale against the rate channel under the stated coverage and independence assumptions; it is not a yield forecast or a confidence interval."}


# The live assessment's horizon: every FOMC decision and CPI release inside it is listed.
SCHEDULED_MACRO_HORIZON_DAYS = 90
_CALENDAR_FAMILY = {"US_FOMC": "FED_POLICY", "US_CPI": "US_INFLATION"}


def scheduled_macro_events(proposed, distributions, evaluated_at, *, horizon_days=SCHEDULED_MACRO_HORIZON_DAYS):
    """Every FOMC decision and CPI release within the horizon, whatever the model proposed.

Dates come from the reviewed official calendar transcription; a source ladder
whose close date is not in that calendar is added with a weaker, labelled
date basis. Model-proposed events for the same family and date are not duplicated.
    """
    official = {(_CALENDAR_FAMILY[family], day): (url, summary)
                for (family, _), (day, url, summary) in OFFICIAL_MACRO_CALENDAR.items()}
    candidates = dict(official)
    for distribution in distributions.values():
        key = (distribution.get("family"), distribution.get("event_date"))
        if key[0] in {"FED_POLICY", "US_INFLATION"} and key[1] and key not in candidates:
            candidates[key] = (None, None)
    present = {(_family(e), e.get("event_date")) for e in proposed}
    added = []
    for (family, day), (url, summary) in sorted(candidates.items(), key=lambda item: item[0][1]):
        timing = reviewed_macro_window(family, day, evaluated_at, horizon_days=horizon_days)
        if not timing["eligible"] or (family, day) in present:
            continue
        label = "FOMC policy decision" if family == "FED_POLICY" else "US CPI release"
        added.append({
            "event_id": ("fomc-" if family == "FED_POLICY" else "cpi-") + day + "-scheduled",
            "title": f"{label} on {day} (scheduled)", "event_date": day, "family": family,
            "question_we_care_about": f"How does the portfolio move across the settlement outcomes of the {label.lower()} on {day}?",
            "definition_origin": "OFFICIAL_CALENDAR_SCHEDULED_MACRO" if url else "SOURCE_LADDER_CLOSE_DATE_NOT_IN_REVIEWED_CALENDAR",
            "calendar_source": {"url": url, "summary": summary},
            "schedule_window": timing,
            "matching_comparisons": [],
        })
    return added


_SEARCH_TO_CARD_FAMILY = {"TAIWAN": "TAIWAN_SUPPLY", "US_CHIP_EXPORT_POLICY": "EXPORT_CONTROLS", "US_TARIFFS": "TRADE_TARIFFS"}


def reviewed_family_cards(originals, geopolitical_search):
    """Searched reviewed chip-supply risks that no model proposal covered (added from exposure)."""
    known = {event["event_id"] for event in originals}
    added = []
    for target in (geopolitical_search or {}).get("targets", []):
        if target.get("definition_origin") != "REVIEWED_FAMILY_FROM_DOCUMENTED_EXPOSURE" or target["event_id"] in known:
            continue
        added.append({
            "event_id": target["event_id"], "title": target["title"], "event_date": target["event_date"],
            "family": _SEARCH_TO_CARD_FAMILY.get(target["family"], target["family"]),
            "question_we_care_about": target["question_we_care_about"],
            "definition_origin": target["definition_origin"], "exposure_basis": deepcopy(target.get("exposure_basis")),
            "matching_comparisons": [],
        })
    return added


def resolve_assessment_identity(report, *, inputs, evaluated_at):
    """Retain original Auto/manual declarations and unresolved identity gates.

    The same pure reconciliation is used by projection and saved replay so
    canonical Portfolio serialization cannot remove an earlier safety block.
    """
    portfolio_data = report["portfolio"]
    declared_rows = deepcopy(portfolio_data["positions"])
    prior_identity = report.get("instrument_identity") or inputs.get("instrument_identity") or {}
    prior_by_symbol = {row.get("symbol"): row for row in prior_identity.get("positions", [])}
    for row in declared_rows:
        prior = prior_by_symbol.get(row.get("symbol")) or {}
        declared = prior.get("declared_asset_type")
        if "input_asset_type" not in row and declared in {"auto", "stock", "etf", "cash"}:
            # Portfolio.to_dict intentionally contains only canonical fields;
            # serialization must not turn unresolved Auto into a manual stock.
            row["input_asset_type"] = declared
    identity = resolve_portfolio_types(declared_rows,
        instrument_metadata=inputs.get("instrument_metadata") or {}, evaluated_at=evaluated_at,
        require_auto_resolution=False)
    verified_now = {row["symbol"] for row in identity["audit"]["positions"]
                    if row.get("source_verified") is True and not row.get("blocked")
                    and row.get("resolved_asset_type") in {"stock", "etf"}}
    active = {row["symbol"] for row in declared_rows if row["weight"]}
    inherited_blocked = (set(prior_identity.get("blocked_symbols") or ()) & active) - verified_now
    if inherited_blocked:
        identity["blocked_symbols"] = sorted(set(identity["blocked_symbols"]) | inherited_blocked)
        identity["audit"]["blocked_symbols"] = list(identity["blocked_symbols"])
        identity["audit"]["status"] = "IDENTITY_REVIEW_REQUIRED"
        identity["audit"]["prior_unresolved_identity_preserved"] = sorted(inherited_blocked)
        for row in identity["audit"]["positions"]:
            if row["symbol"] in inherited_blocked:
                row.update(blocked=True, resolved_asset_type=None,
                           status="PRIOR_IDENTITY_BLOCK_NOT_SOURCE_RESOLVED")
    identity["verified_supported_symbols"] = sorted(verified_now)
    return identity


def retain_reported_instrument_restrictions(exposures, prior_exposures, *, verified_identity_symbols):
    """Keep negative instrument-scope evidence until its own basis is resolved.

    A verified ticker type can resolve a declaration conflict, but it cannot
    disprove a previously observed leveraged/inverse mandate. This pure merge
    is shared with saved replay so replacing canonical exposure rows cannot
    silently drop that restriction during an offline type correction.
    """
    result = deepcopy(exposures)
    verified = set(verified_identity_symbols)
    prior_classes = {row.get("symbol"): row for row in (prior_exposures or {}).get("positions", [])}
    for row in result["positions"]:
        prior = prior_classes.get(row["symbol"]) or {}
        mandate_restricted = prior.get("classification_status") == "LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED"
        identity_not_reverified = (row["symbol"] not in verified
            and prior.get("instrument_class") in {"bond_etf", "unsupported", "unresolved_etf"})
        if mandate_restricted or identity_not_reverified:
            # Only restrictions survive. A prior positive stock/equity
            # classification is never promoted into fresh source evidence.
            for key in ("instrument_class", "classification_status", "provider_instrument_type"):
                if key in prior:
                    row[key] = deepcopy(prior[key])
            if row["symbol"] not in verified:
                row["instrument_identity_verified"] = False
            row["retained_original_constraint"] = (
                "ORIGINAL_REPORTED_MANDATE_NOT_CLEARED_BY_TICKER_IDENTITY" if mandate_restricted
                else "ORIGINAL_REPORTED_RESTRICTION_NOT_NEW_SOURCE_VERIFICATION")
    return result


def build_default_assessment(report, *, market_objects, inputs, evaluated_at,
                             geopolitical_search=None, allow_separate_search_clock=False,
                             distribution_overrides=None, payroll_markets=None):
    """Join already acquired inputs into a deterministic, inspectable main brief.

The original research questions, exact-contract verification, empirical fits
and authenticated acceptance record are not rewritten. Distribution assumptions
and source-defined question changes are visible in every macro event card.
    """
    if not isinstance(evaluated_at, datetime) or evaluated_at.utcoffset() is None:
        raise ValueError("Default analysis needs an explicit timezone-aware evaluation time.")
    portfolio_data = report["portfolio"]
    identity = resolve_assessment_identity(report, inputs=inputs, evaluated_at=evaluated_at)
    verified_now = set(identity["verified_supported_symbols"])
    portfolio = parse_portfolio(identity["portfolio_rows"], notional=portfolio_data["notional"])
    instrument_exposures = build_instrument_exposures(identity["portfolio_rows"],
        notional=portfolio.notional, holdings=inputs.get("holdings"),
        instrument_metadata=inputs.get("instrument_metadata"), evaluated_at=evaluated_at)
    instrument_exposures = retain_reported_instrument_restrictions(instrument_exposures,
        report.get("exposures"), verified_identity_symbols=verified_now)
    distributions = build_distributions_from_markets(market_objects, as_of=evaluated_at)
    for identifier, distribution in (distribution_overrides or {}).items():
        if not distribution.get("source") or distribution.get("event_ticker") != identifier:
            raise ValueError("Explicit distribution overrides require a verified source identity.")
        distributions[identifier] = deepcopy(distribution)
    empirical = report["empirical"]
    daily = empirical["daily_betas"]
    ledger = recover_exposures(portfolio, inputs.get("holdings") or {}, now=evaluated_at)
    compositions = _fund_compositions(inputs, evaluated_at=evaluated_at)
    rate_sensitivity = portfolio_rate_sensitivity(portfolio, compositions,
        instrument_exposures=instrument_exposures)
    two_factor = empirical.get("two_factor_responses") or {}
    identity_gate = instrument_model_gate(portfolio, instrument_exposures=instrument_exposures,
        identity_blocked_symbols=identity["blocked_symbols"], rate_sensitivity=rate_sensitivity,
        two_factor_fits=two_factor)
    fund_reviews = {row["fund_symbol"]: row for row in review_portfolio_source_classes(
        inputs.get("holdings") or {}, inputs.get("instrument_metadata") or {})}
    quality_gaps = holdings_quality_gaps(inputs, compositions, fund_reviews)
    issuers = aggregated_issuer_exposure(portfolio, inputs.get("holdings") or {}, now=evaluated_at)
    geo_by_id = {target["event_id"]: target for target in (geopolitical_search or {}).get("targets", [])}
    events = []
    originals = [e for e in report.get("events", []) if not e.get("parent_event_id")]
    originals += scheduled_macro_events(originals, distributions, evaluated_at)
    originals += reviewed_family_cards(originals, geopolitical_search)
    # Runs that acquired the KXPAYROLLS ladder list every jobs report in the horizon;
    # recorded runs from before the payroll family keep their original event list.
    payrolls = None
    if payroll_markets is not None:
        payrolls = payroll_distributions(payroll_markets, as_of=evaluated_at)
        originals += scheduled_payroll_events(originals, evaluated_at, horizon_days=SCHEDULED_MACRO_HORIZON_DAYS)
    for original in originals:
        if original.get("parent_event_id"):
            continue
        family = _family(original)
        searched_as = (geo_by_id.get(original["event_id"]) or {}).get("family")
        if family == "TRADE_TARIFFS" and searched_as == "US_CHIP_EXPORT_POLICY":
            # Recorded before the tariff family existed: the search ran as export controls.
            family = "EXPORT_CONTROLS"
        elif (searched_as == "US_TARIFFS"
              and not explicit_non_tariff_agreement_title(str(original.get("title", "")))):
            family = "TRADE_TARIFFS"
        card = {"event_id": original["event_id"], "title": original["title"],
                "event_date": original.get("event_date"), "family": family,
                "original_question": original.get("question_we_care_about", original.get("description", original["title"])),
                "original_question_probability_reassigned": False,
                "original_matching_comparisons": deepcopy(original.get("matching_comparisons", [])),
                "physical_probability_P": None,
                "definition_origin": original.get("definition_origin"),
                "calendar_source": deepcopy(original.get("calendar_source"))}
        if original.get("schedule_window"):
            card["schedule_window"] = deepcopy(original["schedule_window"])
        if original.get("exposure_basis"):
            card["exposure_basis"] = deepcopy(original["exposure_basis"])
        if family == PAYROLL_FAMILY:
            card.update(payroll_card(portfolio, original, payrolls or {}, inputs.get("prices_by_symbol"),
                                     evaluated_at=evaluated_at))
            # Measured original-fund returns need a valid supported identity,
            # but not the separate equity-versus-duration routing decision.
            payroll_gate = deepcopy(identity_gate)
            payroll_gate["blocked_symbols"] = list(identity_gate["identity_blocked_symbols"])
            payroll_gate["asset_class_blocked_symbols"] = []
            payroll_gate["status"] = ("BLOCKED_INSTRUMENT_IDENTITY" if payroll_gate["blocked_symbols"]
                                      else "IDENTITY_SCOPE_CHECKED")
            if payroll_gate["blocked_symbols"]:
                payroll_gate["reason"] = ("Supported instrument identity is unresolved for "
                    + ", ".join(payroll_gate["blocked_symbols"]) + "; a complete portfolio estimate is not admitted.")
            else:
                payroll_gate["reason"] = "The measured return model does not infer an underlying equity or duration channel."
            card["instrument_identity_gate"] = payroll_gate
        elif family in {"FED_POLICY", "US_INFLATION"}:
            same_date = [(identifier, distribution) for identifier, distribution in distributions.items()
                         if distribution.get("family") == family and
                         distribution.get("event_date") == original.get("event_date")]
            if len(same_date) != 1:
                card.update(status="SOURCE_OUTCOME_DISTRIBUTION_UNAVAILABLE",
                            reason="No unique reviewed same-date source outcome family was acquired; other dates are not substituted.",
                            distribution=None, economics=None,
                            candidate_distributions=[{"event_ticker": identifier, "status": row.get("status"),
                                "event_date": row.get("event_date"), "reason": row.get("reason")}
                                for identifier, row in distributions.items() if row.get("family") == family])
            else:
                identifier, distribution = same_date[0]
                card["distribution"] = deepcopy(distribution)
                card["source_event_ticker"] = identifier
                card["reviewed_question"] = distribution.get("probability_question")
                card["question_refinement"] = {
                    "decision": "SOURCE_DEFINED_OUTCOME_FAMILY_WITH_EXPLICIT_COMMON_SETTLEMENT_ASSUMPTIONS",
                    "difference": "The original broad research proposal is refined into the displayed dated settlement outcomes. Family membership does not certify its unspecified original metric or economic narrative.",
                    "original_question_preserved": True,
                }
                if distribution.get("atoms") and distribution.get("constraints"):
                    study = "FOMC" if family == "FED_POLICY" else "CPI"
                    try:
                        economics = build_default_macro_economics(portfolio, distribution, daily,
                            release_day_report=empirical.get("release_day_studies", {}).get(study),
                            prices_by_symbol=inputs.get("prices_by_symbol"), rate_sensitivity=rate_sensitivity)
                    except (ValueError, TypeError, KeyError) as exc:
                        card.update(status="ECONOMIC_COMPONENT_UNAVAILABLE", economics=None,
                                    reason="The documented model could not admit these source states: " + str(exc))
                    else:
                        card.update(status=economics["status"], economics=economics)
                        try:
                            announcement = build_default_macro_economics(portfolio, distribution, daily,
                                release_day_report=empirical.get("release_day_studies", {}).get(study),
                                prices_by_symbol=inputs.get("prices_by_symbol"),
                                beta_specification=ANNOUNCEMENT_DAY_SPECIFICATION,
                                rate_sensitivity=rate_sensitivity)
                        except (ValueError, TypeError, KeyError) as exc:
                            card["announcement_day_economics"] = None
                            card["announcement_day_reason"] = str(exc)
                        else:
                            card["announcement_day_economics"] = announcement
                            card["ranking_announcement_day"] = _ranking(announcement)
                            if family == "FED_POLICY" and rate_sensitivity.get("positions"):
                                card["implied_yield_consistency"] = _implied_yield_check(announcement, rate_sensitivity)
                        try:
                            card["outcome_table"] = build_outcome_table(portfolio, distribution, daily,
                                release_day_report=empirical.get("release_day_studies", {}).get(study),
                                rate_sensitivity=rate_sensitivity, holdings_quality=quality_gaps,
                                instrument_exposures=instrument_exposures,
                                identity_blocked_symbols=identity["blocked_symbols"],
                                two_factor_fits=two_factor)
                        except (ValueError, TypeError, KeyError) as exc:
                            card["outcome_table"] = {"status": "OUTCOME_TABLE_UNAVAILABLE", "reason": str(exc)}
                        if family == "FED_POLICY" and any(p.symbol == "TLT" for p in portfolio.positions):
                            try:
                                card["tlt_duration_comparison"] = build_tlt_duration_comparison(
                                    portfolio, economics, load_duration_evidence(), analysis_as_of=evaluated_at)
                            except (ValueError, TypeError, KeyError, OSError):
                                card["tlt_duration_comparison"] = {"status": "VERIFIED_DURATION_COMPARISON_UNAVAILABLE", "replaces_default": False}
                else:
                    card.update(status="SOURCE_DISTRIBUTION_REJECTED", economics=None,
                                reason=distribution.get("reason", "Source constraints did not admit a coherent distribution."))
        elif family in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS", "TRADE_TARIFFS"}:
            search = geo_by_id.get(card["event_id"])
            search_time = (geopolitical_search or {}).get("evaluated_at")
            scope_mismatch = search is not None and (
                search.get("question_we_care_about", "").strip() != card["original_question"].strip()
                or search.get("event_date") != card["event_date"])
            future_search = search_time is not None and datetime.fromisoformat(search_time.replace("Z", "+00:00")) > evaluated_at
            if scope_mismatch or (future_search and not allow_separate_search_clock):
                search = {"decision": "SEARCH_SCOPE_MISMATCH" if scope_mismatch else "SEARCH_AFTER_MACRO_EVALUATION_NOT_JOINED",
                          "closest_by_venue": {}, "candidates": [], "event_pricing_measure_Q_interval": None,
                          "reason": "The recorded search does not match this exact question/horizon or the requested as-of scope."}
            card["market_search"] = deepcopy(search) if search else {
                "decision": "NO_NEW_TWO_VENUE_SEARCH_IN_THIS_REPLAY",
                "closest_by_venue": {}, "candidates": [],
                "reason": "This offline replay cannot claim a search that was not recorded in the supplied evidence.",
                "event_pricing_measure_Q_interval": None,
            }
            card["market_search_context"] = {
                "search_evaluated_at": search_time,
                "capture_evaluated_at": (geopolitical_search or {}).get("capture_evaluated_at"),
                "macro_evaluated_at": evaluated_at.isoformat(),
                "separate_recorded_clock_explicitly_requested": bool(allow_separate_search_clock),
                "scope": "SEPARATE_DATED_MARKET_SEARCH_NOT_A_QUOTE_AT_MACRO_CAPTURE_TIME" if allow_separate_search_clock else "SEARCH_WITHIN_ASSESSMENT_AS_OF_SCOPE",
            }
            if family == "TRADE_TARIFFS":
                # A general-China rate family cannot become a pharmaceutical,
                # product-specific or other-country event's outcome table.
                # Related source scenarios remain inspectable under their own definition.
                card.update(economics=None, distribution=None)
                if search and search.get("candidates") and not scope_mismatch and card["event_date"]:
                    review = tariff_review(portfolio, daily, dict(search, event_date=card["event_date"]),
                                           prices_by_symbol=inputs.get("prices_by_symbol"))
                    card["tariff_review"] = review
                    if review["outcome_table"] is not None:
                        card.update(outcome_table=review["outcome_table"], distribution=deepcopy(review["distribution"]),
                                    status="SOURCE_BRACKET_DISTRIBUTION_WITH_MEASURED_ANCHOR_MODEL")
                    else:
                        card.update(status=("TARGET_TARIFF_SCOPE_NOT_MATCHED"
                                            if review["status"] == "TARGET_TARIFF_SCOPE_NOT_MATCHED"
                                            else "NO_PRICED_TARIFF_CONTRACT_WITHIN_SEARCH"),
                                    reason=review["reason"])
                else:
                    card.update(status="NO_RECORDED_TARIFF_CONTRACT_SEARCH",
                                reason="No recorded contract search matches this question and horizon.")
                card["named_exposure_sensitivity"] = named_exposure_sensitivity(portfolio, issuers, original)
                card["ranking"] = _ranking({})
                events.append(card)
                continue
            # These broad questions have no independently certified equivalence.
            # A price on a nearby candidate is never passed to this channel.
            card["economics"] = build_default_channel_sensitivity(portfolio, ledger,
                family=family, q_interval=None, event_definition=card["original_question"])
            card["distribution"] = None
            card["status"] = "IDENTIFIED_CHANNEL_SENSITIVITY_WITH_UNMATCHED_EVENT_PROBABILITY"
            # Contract-first: every searched contract is classified from its rules;
            # admitted ones become scenarios about exactly their own event, and the
            # broad risk above keeps its card and its unknown (or bounded) probability.
            if search and search.get("candidates") and not scope_mismatch and card["event_date"]:
                target = dict(search, event_date=card["event_date"],
                              family={"TAIWAN_SUPPLY": "TAIWAN", "EXPORT_CONTROLS": "US_CHIP_EXPORT_POLICY"}[family])
                card["contract_first"] = contract_first_review(portfolio, ledger, daily, target,
                    rate_sensitivity=rate_sensitivity, holdings_quality=quality_gaps)
            card["named_exposure_sensitivity"] = named_exposure_sensitivity(portfolio, issuers, original)
        else:
            matching_target = geo_by_id.get(card["event_id"])
            if matching_target and matching_target.get("event_date") == card.get("event_date"):
                card["market_search"] = deepcopy(matching_target)
            card.update(status="NO_REVIEWED_DEFAULT_ECONOMIC_CHANNEL", economics=None,
                        distribution=None, reason="No defensible default model has been reviewed for this event family.")
            card["named_exposure_sensitivity"] = named_exposure_sensitivity(portfolio, issuers, original)
        if family in {"FED_POLICY", "US_INFLATION"}:
            card["instrument_identity_gate"] = deepcopy(identity_gate)
        gate = card.get("instrument_identity_gate") or {}
        if gate.get("blocked_symbols"):
            card["legacy_fallback_allowed"] = False
            card["status_before_instrument_identity_gate"] = card.get("status")
            card["status"] = "BLOCKED_INSTRUMENT_IDENTITY"
            card["reason"] = gate["reason"]
            table = card.get("outcome_table")
            if isinstance(table, dict):
                table["instrument_identity_gate"] = deepcopy(gate)
                table["legacy_fallback_allowed"] = False
                # Payroll tables are direct measured regressions. Preserve
                # their diagnostic results, but do not certify a total whose
                # underlying supported instrument identity is contradictory.
                if family == PAYROLL_FAMILY:
                    card["identity_blocked_model_diagnostics"] = deepcopy(table)
                    card["outcome_table"] = {"status": "BLOCKED_INSTRUMENT_IDENTITY",
                        "instrument_identity_gate": deepcopy(gate), "legacy_fallback_allowed": False,
                        "reason": gate["reason"], "outcomes": [], "summary": {"q_standard_deviation_usd": None}}
            card["ranking"] = _ranking({})
        else:
            card["ranking"] = _ranking(card.get("economics") or {})
        events.append(card)
    original_by_id = {event["event_id"]: event for event in originals}
    for event in events:
        # This is an inspectable, separate what-if sensitivity layer. Its
        # arithmetic never upgrades the source probability, main event-impact
        # admission, or risk ranking. The original question is preserved.
        try:
            event["factor_only_what_if"] = build_factor_what_if(
                portfolio, original_by_id.get(event["event_id"], event), daily,
                inputs.get("instrument_metadata") or {}, evaluated_at=evaluated_at)
        except (ValueError, TypeError, KeyError) as exc:
            event["factor_only_what_if"] = {
                "schema": "castle.event_factor_what_if.v1", "status": "UNAVAILABLE",
                "reason": str(exc), "primary_economic_admission_allowed": False,
                "identified_event_conditional_portfolio_pnl": None,
                "economic_ranking_allowed": False,
            }
        if identity_gate["identity_blocked_symbols"] and event["family"] not in {
                "FED_POLICY", "US_INFLATION", PAYROLL_FAMILY}:
            # Non-macro models and early-return tariff cards must not regain a
            # complete-portfolio ranking after a hard instrument-scope block.
            gate = deepcopy(identity_gate)
            gate["blocked_symbols"] = list(identity_gate["identity_blocked_symbols"])
            gate["asset_class_blocked_symbols"] = []
            gate["reason"] = ("Supported instrument identity is unresolved for "
                + ", ".join(gate["blocked_symbols"]) + "; a complete portfolio estimate is not admitted.")
            event["instrument_identity_gate"] = gate
            event["status_before_instrument_identity_gate"] = event.get("status")
            event["status"] = "BLOCKED_INSTRUMENT_IDENTITY"
            event["reason"] = gate["reason"]
            if isinstance(event.get("outcome_table"), dict):
                event["identity_blocked_model_diagnostics"] = deepcopy(event["outcome_table"])
                event["outcome_table"] = {"status": "BLOCKED_INSTRUMENT_IDENTITY",
                    "instrument_identity_gate": deepcopy(gate), "legacy_fallback_allowed": False,
                    "reason": gate["reason"], "outcomes": [], "summary": {"q_standard_deviation_usd": None}}
        if (event.get("instrument_identity_gate") or {}).get("blocked_symbols"):
            event["legacy_fallback_allowed"] = False
            event["ranking"] = _ranking({})
            event["risk_ranking"] = {"q_standard_deviation_usd": None,
                "basis": "Instrument identity or required fund-class evidence does not admit a complete portfolio ranking."}
            apply_model_admission_to_event(event)
            continue
        table = event.get("outcome_table") or {}
        event["risk_ranking"] = {
            "q_standard_deviation_usd": (table.get("summary") or {}).get("q_standard_deviation_usd"),
            "basis": "Q-standard deviation of the not-yet-priced P&L across the event's settlement outcomes; events without a complete table follow.",
        }
        rankable_scenarios = eligible_contract_scenarios(event)
        if rankable_scenarios and event["risk_ranking"]["q_standard_deviation_usd"] is None:
            top = max(rankable_scenarios, key=lambda s: s["summary"]["q_standard_deviation_usd"])
            event["risk_ranking"] = {
                "q_standard_deviation_usd": top["summary"]["q_standard_deviation_usd"],
                "contract_id": top["contract"]["contract_id"],
                "basis": "Largest admitted contract scenario (declared stress profile): Q-standard deviation of its not-yet-priced YES/NO P&L.",
            }
        # A populated outcome table is not itself an admission decision. Keep
        # failed measured responses and non-equivalent source scenarios in the
        # audit trail, without letting their dollars drive the original target's
        # economic ordering. This never refits or selects a passing alternative.
        apply_model_admission_to_event(event)
    events.sort(key=lambda event: (event["risk_ranking"]["q_standard_deviation_usd"] is None,
                -(event["risk_ranking"]["q_standard_deviation_usd"] or 0.0),
                event["ranking"]["abs_delta_pnl_usd"] is None,
                -(event["ranking"]["abs_delta_pnl_usd"] or 0.0), event["event_date"] or "9999-12-31", event["event_id"]))
    for index, event in enumerate(events, 1):
        event["ranking"]["display_order"] = index
        component = build_business_interruption_component(event, portfolio=portfolio,
            inputs=inputs, evaluated_at=evaluated_at)
        if component and component.get('public_component'):
            event['business_interruption_component'] = component['public_component']
            event['private_business_interruption_inputs'] = component['private_business_interruption_inputs']
        elif component:
            event['business_interruption_review'] = component
        mexico = build_mexico_tariff_component(event, portfolio=portfolio,
            inputs=inputs, evaluated_at=evaluated_at)
        if mexico and mexico.get('public_component'):
            event['mexico_tariff_component'] = mexico['public_component']
            event['private_mexico_tariff_inputs'] = mexico['private_mexico_tariff_inputs']
        elif mexico:
            event['mexico_tariff_review'] = mexico
    classifications = []
    if any(position.symbol == "GLD" and position.asset_type == "etf" for position in portfolio.positions):
        classifications.append(deepcopy(_GLD_DOCUMENTARY_CLASS))
    return {
        "schema": "castle.option1.default_analysis.v1",
        "status": "DECLARED_DEFAULT_SCENARIOS_WITH_EXPLICIT_SOURCE_AND_MODEL_LIMITS",
        "evaluated_at_utc": evaluated_at.isoformat(), "events": events,
        "instrument_identity": deepcopy(identity["audit"]),
        "resolved_portfolio": portfolio.to_dict(),
        "instrument_identity_gate": deepcopy(identity_gate),
        "distribution_inventory": [{"event_ticker": name, "status": row.get("status"),
            "family": row.get("family"), "event_date": row.get("event_date"), "reason": row.get("reason")}
            for name, row in list(distributions.items()) + list((payrolls or {}).items())],
        "documentary_instrument_classes": classifications,
        "fund_signed_compositions": compositions,
        "fund_holdings_reviews": [{key: deepcopy(row.get(key)) for key in (
            "fund_symbol", "status", "holdings_accepted_for_assessment", "declared_tolerances_used",
            "original_issues_retained", "reason")} for row in fund_reviews.values()],
        "aggregated_issuer_exposure": issuers[:12],
        "portfolio_rate_sensitivity": rate_sensitivity,
        "economic_specifications": {
            "original_default": "DAILY: all-day SPY beta per instrument (unchanged Round 9 chain, field 'economics').",
            "announcement_day": ANNOUNCEMENT_DAY_SPECIFICATION + ": same-family announcement-day SPY beta where its frozen chronological check passed, else daily point with the hull of both intervals (field 'announcement_day_economics').",
            "presentation_rule": "Both are shown. The announcement-day specification is presented first because the modeled component is an announcement-day move; the daily chain remains the original reference and is never overwritten.",
        },
        "original_authenticated_acceptance_upgraded": False,
        "new_model_or_provider_calls_by_this_function": 0,
        "scope": "SOURCE_DEFINED_OUTCOME_DISTRIBUTIONS_AND_DECLARED_ANNOUNCEMENT_COMPONENTS_NOT_TOTAL_RETURN_FORECASTS",
        "limitations": [
            "Probability intervals describe a coherent pricing-measure model, not calibrated physical probabilities.",
            "Component ranges are conditional on historical coefficient transfer, instrument betas, state representations, and the common pricing convention; they are not future-return prediction intervals.",
            "A missing or incoherent distribution remains unavailable. No quote normalization or arbitrary probability repairs are used.",
            "Geopolitical partial channels never assign a zero return to the remainder or borrow a probability from a merely related contract.",
            "Original holdings quality failures, taxonomy 403s and an earlier failed forward test of coefficient transfer remain in the retained report.",
        ],
    }
