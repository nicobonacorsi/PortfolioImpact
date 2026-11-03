"""Primary presentation of recorded source distributions and declared economics."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .calibration_audit import historical_stress_audit, load_archived_calibration_sources
from .risk_robustness import RobustnessInputError, analyze_ranking
from .reviewer_overview import build_reviewer_overview


def _money(value):
    return "Unknown" if value is None else f"${value:,.2f}"


def _interval(value, *, probability=False):
    if value is None:
        return "Unknown"
    if probability:
        return f"{value[0]:.1%} to {value[1]:.1%}"
    return f"${value[0]:,.2f} to ${value[1]:,.2f}"


def _evidence_rows(rows):
    """Nested evidence as JSON text: a field such as a tolerance limit can be a number in one row and words in another."""
    return [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
             for key, value in row.items()} for row in rows]


def _probability_summary(event):
    contract_first = event.get("contract_first") or {}
    if contract_first:
        bounds = contract_first["broad_risk"]["probability"]["lower_bounds_by_venue"]
        scenarios = contract_first["contract_scenarios"]
        conditional = contract_first["broad_risk"]["probability"].get("conditional_lower_bounds_by_venue") or []
        text = ("Broad risk ≥ " + ", ".join(f"{b['lower_bound']:.1%} ({b['venue']})" for b in bounds) if bounds
                else "Broad risk: conditional bound only" if conditional else "Broad risk: unknown")
        return text + f"; {len(scenarios)} contract scenario(s) priced"
    distribution = event.get("distribution") or {}
    if event.get("family") == "TRADE_TARIFFS":
        if not distribution.get("atoms"):
            return "No priced tariff contract: probability unknown"
        disagree = [c for c in (event.get("tariff_review") or {}).get("cross_venue_checks", [])
                    if c.get("status") == "INCOMPATIBLE_QUOTES"]
        dated = [c for c in (event.get("tariff_review") or {}).get("cross_venue_checks", [])
                 if c.get("status") == "DATED_CROSS_CHECK_REQUIRES_RATE_CHANGE_OR_DIFFERENT_Q"]
        return (f"{len(distribution['atoms'])} brackets of the general US rate on China on {distribution['measurement_date']} "
                f"({distribution['venue']}); broad risk unknown" +
                "".join(f"; {c['venue']} ladder on {c['measurement_date']} disagrees" for c in disagree) +
                "".join(f"; {c['venue']} different-date ladder needs a rate change or a different Q" for c in dated))
    if event.get("family") == "US_PAYROLLS" and distribution.get("atoms"):
        return f"{len(distribution['atoms'])} payroll bands on {distribution['event_ticker']}, including both tails"
    groups = distribution.get("outcome_groups") or []
    if groups:
        return "; ".join(row["label"] + " " + _interval(row["q_interval"], probability=True) for row in groups)
    if distribution.get("atoms"):
        return f"{len(distribution['atoms'])} disjoint CPI bands, including both tails"
    return distribution.get("status", "No equivalent contract probability")


def _contract_largest_loss(event):
    """Worst row with reference Q of at least 1% across the event's admitted contracts: (amount, outcome)."""
    scenarios = event["contract_first"].get("contract_scenarios", [])
    priced = [s for s in scenarios if not s.get("summary", {}).get("no_portfolio_channel")]
    if scenarios and not priced:
        return _money(0.0), "No modelled channel to this portfolio"
    rows = [(row, scenario["contract"]["market_question"])
            for scenario in priced
            for row in scenario.get("outcomes", [])
            if row.get("pnl_if_outcome_usd") is not None and row["q_reference"] >= .01]
    if not rows:
        return "Unknown", ""
    row, question = min(rows, key=lambda item: item[0]["pnl_if_outcome_usd"])
    side = "YES" if row["outcome_id"].endswith(":YES") else "NO"
    return _money(row["pnl_if_outcome_usd"]), f"{side}: {question} (Q {row['q_reference']:.1%})"


def _outcome_label(row):
    """Display wording for a Kalshi bucket whose rules fix the direction but not the size; the export keeps the source label."""
    label = row["label"]
    for direction in ("Hike", "Cut"):
        if label == direction + " source bucket; magnitude ambiguous":
            return f"Larger-{direction.lower()} bucket (size not set by the contract)"
    return label


def _largest_loss(event):
    """The summary table's worst row with Q >= 1%, the same rule for macro tables and contract scenarios."""
    if event.get("contract_first"):
        return _contract_largest_loss(event)
    row = ((event.get("outcome_table") or {}).get("summary") or {}).get("largest_loss_outcome_reference_q_at_least_1pct")
    if not row:
        return "Unknown", ""
    return _money(row["pnl_if_outcome_usd"]), f"{_outcome_label(row)} (Q {row['q_reference']:.1%})"


_STATE_TEXT = {
    "SOURCE_OUTCOME_DISTRIBUTION_UNAVAILABLE": "No usable contract ladder: probability unknown",
    "SOURCE_DISTRIBUTION_REJECTED": "Contract ladder not usable: probability unknown",
    "ECONOMIC_COMPONENT_UNAVAILABLE": "Probabilities only: P&L unavailable",
    "NO_REVIEWED_DEFAULT_ECONOMIC_CHANNEL": "No reviewed P&L channel for this portfolio",
    "SOURCE_BRACKET_DISTRIBUTION_WITH_MEASURED_ANCHOR_MODEL": "Every rate bracket has Q and P&L; broad risk probability unknown",
    "SOURCE_BANDS_WITH_MEASURED_SURPRISE_RESPONSE": "Every outcome has Q and P&L",
    "SOURCE_BANDS_WITH_IDENTIFIED_NOMINAL_CASH_ZERO": "Each band has Q; fixed USD cash has an identified zero nominal price component",
    "NO_PRICED_TARIFF_CONTRACT_WITHIN_SEARCH": "Nothing fits: no priced bracket family on the China rate",
    "NO_RECORDED_TARIFF_CONTRACT_SEARCH": "No recorded contract search: probability unknown",
    "TARGET_TARIFF_SCOPE_NOT_MATCHED": "Nothing fits this tariff scope: probability and P&L unknown",
}


def _state_text(event):
    """Plain wording for the summary table; the export keeps the status code."""
    status = event["status"]
    contract_first = event.get("contract_first")
    if contract_first is not None:
        if not contract_first.get("contract_scenarios"):
            return "Nothing fits: no contract defines this event"
        if all(s.get("summary", {}).get("no_portfolio_channel") for s in contract_first["contract_scenarios"]):
            return "Contract matched; no modelled channel to this portfolio"
        bounds = contract_first["broad_risk"]["probability"]["lower_bounds_by_venue"]
        conditional = contract_first["broad_risk"]["probability"].get("conditional_lower_bounds_by_venue") or []
        return "Matched contracts priced; broad risk " + ("lower bound only" if bounds else
                    "conditional bound only" if conditional else "probability unknown")
    if status == "DECLARED_DEFAULT_COMPONENT_MODEL":
        complete = (event.get("outcome_table") or {}).get("outcomes")
        return "Every outcome has Q and P&L" if complete else "Probabilities only: P&L unavailable"
    return _STATE_TEXT.get(status, status)


def _transfer_warnings(st, economics):
    """Separate the selected scenario's check from the original daily diagnostic."""
    selected_recorded = 'scenario_transfer_failed_symbols' in economics
    daily_failed = economics.get('daily_transfer_failed_symbols', [])
    failed = economics.get('scenario_transfer_failed_symbols', daily_failed)
    if failed:
        label = 'Selected scenario' if selected_recorded else 'Daily'
        st.warning(label + ' chronological transfer check failed for ' + ', '.join(failed) +
                   '. The displayed historical sensitivity is a model assumption, not a validated forward forecast.')
    untested = economics.get('scenario_transfer_untested_symbols', [])
    if untested:
        st.warning('The selected scenario has no completed chronological transfer check for ' + ', '.join(untested) +
                   '. Its event-family transfer remains untested; a daily-series result does not establish that validation.')
    if selected_recorded and daily_failed:
        st.caption('Separate original daily-series diagnostic: chronological transfer failed for ' +
                   ', '.join(daily_failed) + '. This retained result is distinct from the selected scenario check.')


def _history_check(st, family):
    """One line from the archived event study next to the table it qualifies."""
    from .event_study_evidence import load_evidence
    try:
        evidence = load_evidence(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if family == "US_INFLATION":
        from .default_event_economics import published_response_model
        spy = next((r for r in evidence["cpi"]["surprise_response"] if r["symbol"] == "SPY" and r.get("status") == "ESTIMATED"), None)
        periods = evidence["cpi"]["period_slopes"].get("SPY") or []
        if spy is None or any(p["return_per_unit_surprise"] is None for p in periods):
            return
        used = published_response_model("US_INFLATION", "HEADLINE_CPI_MONTH_ON_MONTH")["coefficient"]
        st.caption(f"History check ({evidence['events']['CPI']} archived CPI releases, 15-minute windows): SPY moved "
                   f"{1e4 * spy['return_per_unit_surprise']:.0f} bp per percentage point of surprise against Q, and the Q-expected "
                   + ("part had no detectable effect (consistent with it being priced in, not proof of it)"
                      if abs(spy["kuttner"]["expected_t"]) < 1.96 else "part also moved prices")
                   + ". The size changed with the regime: "
                   + ", ".join(f"{p['period']} {1e4 * p['return_per_unit_surprise']:.0f} bp" for p in periods)
                   + f". The published coefficient in this table is {1e4 * used:.0f} bp per point: history supports the sign; the size depends on the regime.")
    elif family == "FED_POLICY" and evidence["fomc"]["n"]:
        fomc = evidence["fomc"]
        st.caption(f"History check ({fomc['n']} archived FOMC decisions): Q anticipated the decision each time "
                   f"(median surprise {fomc['median_abs_surprise']:.1f} bp, largest {fomc['max_abs_surprise']:.1f} bp). "
                   "Announcement-window moves came from the statement and guidance, which a decision contract does not "
                   "price: this table covers the decision only.")


_MACRO_TABLE_CAPTION = (
    "Each row is the move not already in prices if that outcome settles: the Q-weighted sum is zero by construction, "
    "because the contract weights are assumed to be the weights embedded in asset prices. Stocks and equity ETFs move with SPY "
    "through combined daily/announcement-day loadings; bond ETFs move with long yields through holdings-derived duration. "
    "Ranges propagate published-coefficient and loading uncertainty and span every Q the quotes allow; they are not forecast intervals. "
    "A narrow range reflects the declared published coefficients and pooled betas, not more evidence about this event. Q is not a physical probability.")


def _outcome_table(st, event, *, caption=_MACRO_TABLE_CAPTION, details=True):
    """Primary answer: every settlement outcome with its Q weight and not-yet-priced P&L."""
    table = event.get("outcome_table") or {}
    if not table.get("outcomes"):
        if table.get("reason"):
            st.info("Outcome table unavailable: " + table["reason"])
        return
    st.markdown("#### If each outcome happens, what does the portfolio gain or lose beyond what prices imply?")
    summary = table["summary"]
    left, middle, right = st.columns(3)
    likely = summary["most_likely_outcome"]
    worst = summary["largest_loss_outcome_reference_q_at_least_1pct"]
    left.metric("Most likely by market weight: " + _outcome_label(likely), _money(likely["pnl_if_outcome_usd"]))
    middle.metric("Largest loss with Q ≥ 1%: " + _outcome_label(worst), _money(worst["pnl_if_outcome_usd"]))
    right.metric("Q-std of unpriced P&L", _money(summary["q_standard_deviation_usd"]))
    st.dataframe([{"Outcome": _outcome_label(row),
                   "Market weight Q": f"{row['q_reference']:.1%}",
                   "Q interval": _interval(row["q_interval"], probability=True),
                   "P&L if it happens": _money(row["pnl_if_outcome_usd"]),
                   "Model range 5-95% (all feasible Q)": _interval(row["pnl_model_range_5_95_usd"]),
                   "Sign probability (model, reference Q)": "—" if row["sign_probability_under_model"] is None else f"{row['sign_probability_under_model']:.0%}",
                   "Q × P&L": _money(row["q_weighted_contribution_usd"])} for row in table["outcomes"]],
                 hide_index=True, width="stretch")
    for gap in table.get("data_quality_gaps", []):
        measured = ", ".join(f"{k} {v}" for k, v in gap.get("measured", {}).items())
        st.warning(f"{gap['symbol']} holdings snapshot failed quality checks: " + ", ".join(gap.get("issues", [])) +
                   (f" (measured: {measured})" if measured else "") + ". " + gap.get("treatment", ""))
    if table.get("missing_symbols"):
        st.warning("No sensitivity for " + ", ".join(table["missing_symbols"]) + f" ({table['missing_weight']:.1%} of NAV): "
                   "the table covers the known positions only. Missing is not zero.")
    st.caption(caption)
    _history_check(st, event.get("family"))
    if not details:
        return
    with st.expander("Loadings, factor responses and assumptions", expanded=False):
        st.dataframe([{"Instrument": p["symbol"], "Weight": f"{p['weight']:.1%}", "Channel": p["channel"],
                       "Loading": None if p.get("loading") is None else round(p["loading"], 3),
                       "SE used (dependence-robust)": None if p.get("se") is None else round(p["se"], 3),
                       "SE if samples were independent": None if p.get("se_if_independent") is None else round(p["se_if_independent"], 3),
                       "Basis": p["basis"], "Heterogeneity z": None if p.get("heterogeneity_z") is None else round(p["heterogeneity_z"], 2),
                       "Data quality": ", ".join((p.get("data_quality") or {}).get("issues", [])) or "No snapshot gap"}
                      for p in table["positions"]], hide_index=True, width="stretch")
        st.json(table["factors"])
        st.json(table["uncertainty"])
        for assumption in table["assumptions"]:
            st.write("- " + assumption)


def _macro(st, event, *, key_prefix):
    distribution = event.get("distribution") or {}
    st.write(event.get("reviewed_question") or event["original_question"])
    st.caption("Original research question: " + event["original_question"])
    if event.get("question_refinement"):
        st.caption(event["question_refinement"]["difference"])
    st.markdown("#### What probabilities do the contracts support?")
    if not distribution.get("atoms"):
        st.warning(event.get("reason", "No admissible common source distribution is available."))
        with st.expander("Distribution gate and source evidence", expanded=False):
            st.json(distribution or event.get("candidate_distributions", []))
        return
    st.caption("Quote-compatible pricing weights Q at the recorded capture time. Physical probabilities P are not estimated. "
               "The individual interval endpoints need not be jointly attainable; every displayed reference weight belongs to one feasible joint distribution.")
    if distribution.get("outcome_groups"):
        st.dataframe([{"Decision": row["label"], "Q interval": _interval(row["q_interval"], probability=True),
                       "Reference Q": f"{row['q_reference']:.1%}"} for row in distribution["outcome_groups"]],
                     hide_index=True, width="stretch")
    else:
        st.dataframe([{"Outcome": _outcome_label(row), "Q interval": _interval(row["q_interval"], probability=True),
                       "Reference Q": f"{row['q_reference']:.1%}"} for row in distribution["atoms"]],
                     hide_index=True, width="stretch")
    source = distribution.get("source", {})
    st.caption("Source capture: " + str(source.get("retrieved_at_utc", source.get("retrieved_at", distribution.get("evaluated_at", "Unknown")))) +
               " · " + distribution.get("event_ticker", "") + " · " + distribution.get("status", ""))
    if distribution.get("fee_model"):
        st.info("These Q bounds include an explicitly declared transaction-fee convention. The raw quote-basket result remains in the evidence.")
    _outcome_table(st, event)
    economics = event.get("economics")
    announcement = event.get("announcement_day_economics")
    if announcement and announcement.get("outcome_cases") and economics:
        choice = st.radio("Instrument transfer", ["Announcement-day beta where checked", "Daily beta (original chain)"],
                          horizontal=True, key=key_prefix + "_spec_" + event["event_id"])
        st.caption("The modeled component is an announcement-day SPY move, so each instrument uses its same-family "
                   "announcement-day beta when that fit passed its frozen chronological check. Otherwise the point "
                   "stays at the daily beta and the range covers both estimates. The daily chain is kept unchanged for comparison.")
        if choice.startswith("Announcement"):
            economics = announcement
    if not economics or not economics.get("outcome_cases"):
        st.info(event.get("reason", "A complete declared economic component is unavailable."))
        return
    cases = economics["outcome_cases"]
    by_id = {case["outcome_id"]: case for case in cases}
    # A reference mode is an understandable starting point, never a favorable P&L selection.
    ordered = sorted(by_id, key=lambda key: (-by_id[key]["q_reference"], key))
    selected = st.selectbox("Inspect an outcome", ordered,
                            format_func=lambda key: _outcome_label(by_id[key]), key=key_prefix + "_outcome_" + event["event_id"])
    case = by_id[selected]
    st.markdown("#### What would this portfolio gain or lose from this announcement component?")
    st.caption("Default chain: source outcome − same-distribution reference expectation → published benchmark response → "
               "estimated instrument betas → original portfolio positions. No manual coefficient entry is required.")
    if economics.get("unrestricted_support_review", {}).get("finite_economic_domain_is_declared_assumption"):
        st.warning("A source tail or ambiguous magnitude has positive possible mass. Without finite economic tail assumptions, "
                   "the reference expectation is not finitely bounded by these quotes. The dollar range below is conditional on the displayed finite state domain.")
    elif case.get("finite_state_scenario_required_for_this_case"):
        st.warning("The selected hypothetical outcome magnitude is unidentified; its dollar stress uses a declared finite domain. "
                   "A zero-Q state does not make this reference expectation unbounded.")
    left, middle, right = st.columns(3)
    left.metric("Modeled announcement component", _money(case.get("component_pnl_usd")))
    middle.metric("Conditional component range", _interval(case.get("component_pnl_interval_usd")))
    right.metric("Outcome pricing interval Q", _interval(case.get("q_interval"), probability=True))
    st.caption(case.get("sign_status", "") + " · This is a component scenario, not a prediction interval for the full future portfolio return.")
    if case.get("q_interval", [None, None])[1] == 0:
        st.info("This outcome has zero feasible Q under this recorded pricing model. Its hypothetical stress is retained; zero physical probability is not claimed.")
    if case.get("component_pnl_usd") is None:
        st.write("Known-position component: **" + _money(case.get("known_position_component_pnl_usd")) +
                 "**. Missing sensitivities or failed model-domain checks keep the portfolio component unknown.")
    st.write("Other news of **" + _money(case.get("unmodeled_news_break_even_usd")) +
             "** would offset the point component. Total future P&L remains unpredicted.")
    st.dataframe([{"Instrument": row["symbol"], "Portfolio weight": f"{row['weight']:.1%}",
                   "Beta to SPY": row.get("beta"), "Beta range": str([round(x, 3) for x in row.get("beta_interval") or []]),
                   "Beta basis": row.get("basis"), "Component (USD)": row.get("component_pnl_usd"),
                   "Selected transfer check": row.get("scenario_transfer_status", row.get("transfer_check", {}).get("status", "Not recorded")),
                   "Component range": _interval(row.get("component_pnl_interval_usd"))} for row in case["positions"]],
                 hide_index=True, width="stretch")
    implied = event.get("implied_yield_consistency")
    if implied and economics is announcement:
        rows = [{"Outcome": c["outcome_id"], "Fund": p["symbol"],
                 "Implied parallel yield move (bp)": round(p["implied_parallel_yield_move_bp"], 2),
                 "Range (bp)": str([round(x, 2) for x in p["implied_parallel_yield_move_interval_bp"]])}
                for c in implied["outcome_cases"] if c["outcome_id"] == selected for p in c["positions"]]
        if rows:
            st.markdown("##### Rate-channel consistency check")
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(implied["interpretation"] + " Kuttner (2001) reports about 19 bp of 30-year yield per 100 bp policy surprise; "
                       "the check is a scale comparison, not a fitted yield response.")
    st.caption("Each original position is counted once. Constituent exposure is not added to an ETF's whole-position return.")
    _transfer_warnings(st, economics)
    st.markdown("#### What is already priced in under this model?")
    priced = case["priced_in"]
    left, middle, right = st.columns(3)
    left.metric("Assumed priced component: q × ΔV", _money(priced.get("modeled_priced_component_usd")))
    middle.metric("YES residual: (1 − q) × ΔV", _money(priced.get("YES")))
    right.metric("NO residual: −q × ΔV", _money(priced.get("NO")))
    st.write(("NO means the conditional mixture of **all other outcomes**, under the same Q. "
              "It does not mean the opposite policy decision. YES range: " + _interval(priced.get("YES_interval")) +
              "; NO range: " + _interval(priced.get("NO_interval")) + ".").replace("$", "\\$"))
    st.caption("Using contract Q as the reference embedded in asset prices is an explicit transfer assumption. "
               "The centered weighted sum is zero by construction; it is not a measurement of actual equity-market pricing.")
    with st.expander("Model, uncertainty layers and source rules", expanded=False):
        st.warning("An earlier forward test of transferring daily coefficients to later announcements failed and stays on record. Neither historical coefficient transfer nor the future total return is certified.")
        st.write("Reference expectation: " + str(economics.get("reference_expectation")) +
                 "; conditional domain: " + str(economics.get("reference_expectation_interval")) + ".")
        st.json(economics["response_model"])
        st.json(economics["state_models"])
        st.json(case["interval_layers"])
        st.json(economics.get("historical_move_context", {}))
        st.json(economics.get("release_day_diagnostics", {}))
        for assumption in economics.get("assumptions", []):
            st.write("- " + assumption)
        st.json(distribution)
    alternative = event.get("tlt_duration_comparison")
    if alternative and alternative.get("outcome_cases"):
        with st.expander("Alternative TLT duration channel — replaces TLT contribution", expanded=False):
            st.caption("A separate model comparison. These amounts must not be added to the default portfolio component.")
            st.json(alternative)


def _documented_links(event):
    """'NVDA (direct, inside QQQ): statement' for each held issuer whose filing put this risk on the list."""
    grouped = {}
    for row in event.get("exposure_basis") or []:
        holders, statement = grouped.setdefault(row["instrument_symbol"], ([], row.get("statement") or ""))
        holders.append("direct" if row.get("original_symbol") == row["instrument_symbol"] else "inside " + str(row.get("original_symbol")))
    return [f"{symbol} ({', '.join(holders)}): {statement}" for symbol, (holders, statement) in grouped.items()]


def _why_listed(event):
    origin = event.get("definition_origin")
    if origin == "OFFICIAL_CALENDAR_SCHEDULED_MACRO":
        return "Scheduled (official calendar)"
    if origin == "SOURCE_LADDER_CLOSE_DATE_NOT_IN_REVIEWED_CALENDAR":
        return "Scheduled (contract close date)"
    if origin == "REVIEWED_FAMILY_FROM_DOCUMENTED_EXPOSURE":
        symbols = sorted({row["instrument_symbol"] for row in event.get("exposure_basis") or []})
        return "Always checked: " + (", ".join(symbols) or "a holding") + " filing documents the link"
    return "Proposed for this portfolio"


def _implication_text(bound):
    if not bound:
        return ""
    if bound["status"] == "LOWER_BOUND_FROM_IMPLYING_CONTRACT":
        return f"Yes: Q(broad risk) ≥ {bound['lower_bound']:.1%}"
    if bound["status"] == "CONDITIONAL_LOWER_BOUND_FROM_IMPLYING_CONTRACT":
        return f"Conditional only: ≥ {bound['conditional_lower_bound']:.1%}; no qualifying pre-window event assumed"
    return "Not proven: " + bound.get("reason", "")


def _contract_scenarios(st, event):
    """Contract-first view: each admitted contract is its own event and its own declared scenario."""
    review = event.get("contract_first") or {}
    if not review:
        return
    st.markdown("#### Which contracts define a priceable scenario, and what does each do to the portfolio?")
    if any(venue.get("queries_with_unfinished_pages")
           for venue in (event.get("market_search") or {}).get("closest_by_venue", {}).values()):
        st.caption("Some searches stopped with pages unread, so a later run can retrieve contracts this one did not "
                   "and the set of scenarios below can change between runs.")
    broad = review["broad_risk"]
    bounds = broad["probability"]["lower_bounds_by_venue"]
    conditional_bounds = broad["probability"].get("conditional_lower_bounds_by_venue") or []
    if bounds:
        for bound in bounds:
            st.success(f"Broad risk probability (pricing measure Q), for the reviewed definition below: at least "
                       f"{bound['lower_bound']:.1%}, implied by {bound['venue']} contract {bound['contract_id']}. Upper bound unknown.")
            with st.expander("Why this contract implies the broad risk", expanded=False):
                for step in bound["proof"]:
                    st.write("- " + step)
                st.caption(bound["not_claimed"])
    elif conditional_bounds:
        st.info("Broad risk probability: no unconditional bound. The following lower bounds require that no "
                "qualifying event occurred before the evaluation window; an open market does not establish this.")
        for bound in conditional_bounds:
            st.write(f"Conditional Q lower bound: **{bound['conditional_lower_bound']:.1%}** "
                     f"under {bound['venue']}'s pricing measure ({bound['contract_id']}).")
            with st.expander("Condition and implication evidence: " + bound['contract_id'], expanded=False):
                st.json(bound)
    else:
        st.info("Broad risk probability: unknown. No admitted contract is shown to imply it from its rules and deadline.")
    links = _documented_links(event)
    if links:
        st.caption("Listed without a model proposal, because a holding's annual report documents the channel. "
                   + " ".join(links))
    if broad.get("probability_definition"):
        st.caption("Reviewed definition used for the probability: " + broad["probability_definition"] +
                   (" No model proposal covered this risk." if links else
                    " Claude's proposal above chose which risk to search; its wording does not change this bound."))
    if len(review["contract_scenarios"]) > 1:
        st.info(f"These {len(review['contract_scenarios'])} contract scenarios can happen together (a blockade can be "
                "part of an invasion), so they are not separate risks. Read each one on its own: do not add their "
                "P&L figures or their probabilities.")
    for scenario in review["contract_scenarios"]:
        contract = scenario["contract"]
        st.markdown(f"**{contract['market_question']}** · {contract['venue']} {contract['contract_id']} · "
                    f"Q {_interval(contract['quote_interval'], probability=True)} · {scenario['scenario']['label']}")
        left, middle, right = st.columns(3)
        yes, no = scenario["outcomes"]
        left.metric("If YES (not yet in prices)", _money(yes["pnl_if_outcome_usd"]))
        middle.metric("If NO", _money(no["pnl_if_outcome_usd"]))
        right.metric("Q-std of unpriced P&L", _money(scenario["summary"]["q_standard_deviation_usd"]))
        st.dataframe([{"Outcome": row["label"], "Q interval": _interval(row["q_interval"], probability=True),
                       "P&L if it happens": _money(row["pnl_if_outcome_usd"]),
                       "Scenario range 5-95% (all quoted Q)": _interval(row["pnl_model_range_5_95_usd"])}
                      for row in scenario["outcomes"]], hide_index=True, width="stretch")
        st.dataframe([{"Component": part["component"], "YES minus NO": _money(part["pnl_usd"]),
                       "Data quality": ", ".join((part.get("data_quality") or {}).get("issues", [])) or "No snapshot gap"}
                      for part in scenario["positions"]], hide_index=True, width="stretch")
        profile = scenario["scenario"]
        issuer_channels = scenario.get("issuer_channels") or []
        if issuer_channels:
            st.markdown("**Issuer and commodity channels beyond the broad market move**")
            st.dataframe([{"Issuer / vehicle": row["issuer"], "Portfolio weight": f"{row['portfolio_weight']:.2%}",
                           "Declared return contrast": row["point"], "Stress domain": str(row["range"]),
                           "YES minus NO component": _money(row["declared_channel_pnl_usd"]),
                           "Treatment": row["mode"], "Evidence status": row["calibration_status"]}
                          for row in issuer_channels], hide_index=True, width="stretch")
            with st.expander("Issuer-channel sources and assumptions: " + contract["contract_id"], expanded=False):
                for row in issuer_channels:
                    st.write(row["issuer"] + ": " + row["basis"])
                    if row.get("anchors"):
                        st.caption(row["anchors"])
                    source = row.get("source") or {}
                    if source.get("url"):
                        st.markdown("[" + source.get("title", "Primary source") + "](" + source["url"] + ")")
                    if source.get("supports"):
                        st.caption("Documentary support: " + str(source["supports"]))
                    if source.get("does_not_support"):
                        st.caption("Not established by the source: " + str(source["does_not_support"]))
                st.caption("Documented exposure and assumed return magnitude are different claims. "
                           "The fund's common market sensitivity is counted once; issuer residuals are applied only to their recorded slices.")
        st.dataframe([{"Declared input": name, "Point": profile[key]["point"], "Range": str(profile[key]["range"]),
                       "Unit": profile[key]["unit"], "Basis": profile[key]["basis"], "Anchor": profile[key]["anchors"]}
                      for name, key in (("Broad market", "market_return"), ("NVDA supply add-on", "nvda_addon"),
                                        ("Long Treasury yield", "long_yield_bp"))], hide_index=True, width="stretch")
        summary = scenario["summary"]
        if summary.get("no_portfolio_channel"):
            channels = [name for name, key in (("the broad market", "market_return"),
                                               ("NVIDIA, held directly or inside a fund", "nvda_addon"),
                                               ("long Treasury yields", "long_yield_bp"))
                        if any(profile[key]["range"])]
            st.info("This contract defines the event, but the declared scenario moves none of this portfolio's positions "
                    f"(its only channel: {' and '.join(channels) or 'none'}), so YES and NO are $0 under this model. "
                    "That is a limit of the model, not evidence that the event cannot move these holdings.")
        elif summary.get("market_beta_only_symbols"):
            st.warning("Market beta only, no event-specific effect: " + ", ".join(summary["market_beta_only_symbols"])
                       + ". The model has no separately quantified issuer channel for these positions.")
        offset = summary.get("bond_return_that_offsets_equity_contrast")
        st.caption((f"Window: {profile['window']}. Per 1% market move: {_money(summary['usd_per_one_percent_market_move_through_betas'])}; "
                    f"per 1% NVDA add-on: {_money(summary['usd_per_one_percent_nvda_addon'])}"
                    + ("" if offset is None else f"; the bond sleeve would need {offset:+.1%} to offset the equity contrast") + ". "
                    "The contract price weights YES against NO; it does not set these shocks.").replace("$", "\\$"))
    with st.expander("Every searched contract and why it was admitted or rejected", expanded=False):
        st.dataframe([{"Venue": r["venue"], "Contract": r["contract_id"], "Market question": r["market_question"],
                       "Event class (from rules)": r["event_class"], "Decision": r["decision"],
                       "Reason": r.get("reason", ""), "Deadline": r.get("contract_end_time"),
                       "Q": _interval(r.get("quote_interval"), probability=True),
                       "Implies the broad risk?": _implication_text(r.get("implication_to_broad_risk"))}
                      for r in review["candidate_reviews"]],
                     hide_index=True, width="stretch")
        st.caption("Classes come from each contract's resolution rules, not its title. Overlapping scenarios "
                   "(for example a blockade and an invasion) must not be added.")
    st.caption(broad["kept_visible"])


def _geopolitical(st, event):
    st.write(event["original_question"])
    review = event.get("market_search") or {}
    st.markdown("#### Which markets were actually searched, and does anything fit?")
    st.caption(review.get("decision", "NO_RECORDED_SEARCH"))
    context = event.get("market_search_context", {})
    if context.get("search_evaluated_at"):
        st.caption("Search evaluation: " + context["search_evaluated_at"] + " · " + context.get("scope", ""))
    contract_first = bool(event.get("contract_first"))
    for venue, decision in review.get("closest_by_venue", {}).items():
        st.write("**" + venue + "** · " + decision.get("status", "Unknown search status") +
                 ("" if not decision.get("queries_with_unfinished_pages") else
                  " · unfinished pages for: " + ", ".join(decision["queries_with_unfinished_pages"])))
        if contract_first:
            continue  # every candidate is classified below; no single "closest" title is promoted
        candidate = decision.get("closest_candidate")
        if not candidate:
            st.write(decision.get("decision", "No admissible candidate in the recorded bounded search."))
            continue
        st.dataframe([{"Question we care about": event["original_question"],
                       "Market question": candidate["market_question"],
                       "Difference": candidate["difference"], "Decision": candidate["decision"]}],
                     hide_index=True, width="stretch")
        quote = candidate.get("contract_quote_reference", {})
        st.write("Reference for this different contract: **" + _interval(quote.get("interval"), probability=True) + "**.")
        st.caption(quote.get("scope", "No admissible quote") + " · " + candidate["contract_id"] +
                   " · captured " + str(candidate.get("source", {}).get("retrieved_at", "Unknown")) +
                   " · evaluated " + str(quote.get("evaluated_at", "Unknown")))
        with st.expander(venue + " complete rules and selection evidence", expanded=False):
            st.json(candidate)
    if review.get("decision") in {"SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE", "SEARCH_SCOPE_MISMATCH", "SEARCH_AFTER_MACRO_EVALUATION_NOT_JOINED", "NO_NEW_TWO_VENUE_SEARCH_IN_THIS_REPLAY"}:
        st.warning("The available search cannot establish a match for this question and evaluation scope. "
                   "Probability remains unknown; a failed or incompatible search is not evidence that no market fits.")
    elif not contract_first:
        st.info("No equivalent contract was admitted for this broad research question. Its probability remains unknown. "
                "The recorded search is bounded; it does not prove that no such market exists anywhere.")
    else:
        st.info("No contract is equivalent to this broad question. Contracts whose rules define a narrower event inside "
                "the horizon are priced below as their own scenarios. The search is bounded; it does not prove that no "
                "other market exists.")
    _contract_scenarios(st, event)
    economics = event.get("economics") or {}
    st.markdown("#### The broad risk itself: what can we quantify without a price?")
    multi = economics.get("multiissuer_sensitivity") or {}
    channel = economics.get("dollars_per_one_percent_nvda_return")
    if multi.get("channels"):
        st.dataframe([{"Issuer": row["issuer"], "Portfolio weight": f"{row['portfolio_weight']:.2%}",
                       "USD per 1% issuer return": _money(row["dollars_per_one_percent_return"]),
                       "Declared component": _money(row["declared_component_pnl_usd"]),
                       "Stress domain": _interval(row["declared_component_range_usd"])}
                      for row in multi["channels"]], hide_index=True, width="stretch")
        st.caption(multi.get("scope", "Documented exposure under declared stress assumptions; no broad-event probability is assigned."))
    elif channel is not None and abs(channel) < .005:
        # A zero channel would only print $0.00 metrics; say why instead.
        st.write("This declared stress covers the NVIDIA channel only, and no holding carries it, directly or through "
                 "a fund. The contract scenarios above show the market channel this portfolio does have.")
    else:
        left, right = st.columns(2)
        left.metric("Identified channel per 1% NVDA return", _money(channel))
        right.metric("Declared NVDA channel stress", _money(economics.get("known_component_delta_pnl_usd")))
        st.write("Channel stress range: " + _interval(economics.get("known_component_delta_pnl_interval_usd")) + ".")
        st.caption("The shock magnitude is an illustrative declared loss stress, not an empirical estimate of this event. "
                   "Direct NVDA and verified indirect NVDA exposure are counted once. The rest is unknown.")
        breakeven = economics.get("break_even", {}).get("remaining_return_contrast")
        if breakeven is not None:
            st.write(f"A return contrast of **{breakeven:+.2%}** on the remaining portfolio would offset this channel stress.")
    st.write("Full P&L of the broad risk and its already-priced residual: **unknown**. Only the admitted contracts above are priced, each for its own event.")
    with st.expander("Recorded search and channel details", expanded=False):
        st.json(review)
        st.json(economics)


_TARIFF_DECISION_TEXT = {
    "ADMITTED_OUTCOME_FAMILY_MEMBER": "Admitted: a bracket of the general China rate",
    "CROSS_CHECK_SAME_RATE_THRESHOLD": "Cross-check: the same rate as 'at least X%', on another date",
    "REJECTED_DEADLINE_AFTER_HORIZON": "Rejected: measured after the horizon",
    "REJECTED_NOT_OPEN_OR_NOT_QUOTED": "Rejected: no valid current quote",
    "REJECTED_DIFFERENT_EVENT": "Rejected: different event",
}
_TARIFF_TABLE_CAPTION = (
    "Each row is the move not already in prices if the general US tariff rate on Chinese goods settles in that bracket: "
    "the Q-weighted sum is zero by construction. Every position moves by its own return per point of the China rate on two "
    "dated announcements: 12 May 2025 (Geneva, 145% to 30%, China only) sets the point; 3 April 2025 (+34 points on China, "
    "announced with tariffs on every partner) sets the far end. The ranges span both anchors, the rate inside each bracket "
    "and every Q the quotes allow; they are not forecast intervals. Q is not a physical probability.")


def _tariff_cross_venue(st, review, table):
    """Each 'at least X%' ladder on the same rate against the bracket family; disagreement is stated, not resolved."""
    checks = review.get("cross_venue_checks") or []
    if not checks:
        return
    st.markdown("#### Do the venues agree on this rate?")
    if review.get("reviewed_again_from_recorded_search"):
        st.caption("Contract decisions on this card were re-read with the current rules from the recorded search; "
                   "quotes, brackets and P&L are as recorded.")
    for check in checks:
        if check["status"] == "LADDER_NOT_USABLE":
            st.caption(f"{check['venue']} ladder on {check['measurement_date']} not usable: {check['reason']}")
            continue
        st.dataframe([{"Rate at least": f"{row['threshold_pp']:g}%",
                       f"{check['venue']} on {check['measurement_date']}": _interval(row["ladder_interval"], probability=True),
                       f"{check['bracket_venue']} brackets on {check['bracket_measurement_date']} allow":
                           _interval(row["bracket_family_interval"], probability=True),
                       "Compatible": "yes" if row["compatible"] else "no"} for row in check["rows"]],
                     hide_index=True, width="stretch")
        low, high = check["ladder_mean_range_pp"]
        if check["status"] in {"COMPATIBLE", "DATED_CROSS_CHECK_CONSISTENT_WITH_UNCHANGED_RATE"}:
            st.caption((f"{check['venue']}'s quotes overlap the brackets at every threshold. "
                        + ("The different dates still prevent an exact event or pricing-measure match."
                           if not check.get("same_settlement_date", True) else "")))
            continue
        shift = check.get("pnl_shift_usd_range")
        if check.get("same_settlement_date", True):
            text = (f"Two quotes for the same settlement date have disjoint implied intervals. "
                    f"That is a pricing disagreement, not evidence that either venue gives physical P. "
                    f"The ladder's Q-expected rate is {low:.1f}% to {high:.1f}% (rate capped at 145%)")
        else:
            text = (f"These are NOT the same settlement event. {check['venue']}'s ladder is dated "
                    f"{check['measurement_date']}; {check['bracket_venue']}'s brackets are dated "
                    f"{check['bracket_measurement_date']}. Their quote intervals would conflict only "
                    f"under BOTH an unchanged rate across those dates and a transferable pricing weight Q. "
                    f"A rate change or a different Q can explain the difference. "
                    f"The ladder's Q-expected rate is {low:.1f}% to {high:.1f}% (rate capped at 145%)")
        if shift and table:
            envelope = check["pnl_shift_usd_range_envelope"]
            text += (f", against {table['reference_expectation']:.1f}% for the brackets. If the ladder's Q is the one in asset "
                     f"prices, every row of the table moves by {_interval(shift)} at the Geneva anchor "
                     f"({_interval(envelope)} at the April envelope) and their order does not change.")
        else:
            text += "."
        st.warning((text + " Which venue's Q asset prices embed is not tested here; Q is not a physical probability.")
                   .replace("$", "\\$"))


def _tariff(st, event):
    """Tariff risk: rules-first review of every searched contract, the bracket table, and the broad risk."""
    st.write(event["original_question"])
    search = event.get("market_search") or {}
    review = event.get("tariff_review") or {}
    st.markdown("#### Which tariff contracts were searched, and which define a priceable event?")
    context = event.get("market_search_context", {})
    if context.get("search_evaluated_at"):
        st.caption("Search evaluation: " + context["search_evaluated_at"] + " · " + context.get("scope", ""))
    for venue, decision in search.get("closest_by_venue", {}).items():
        st.write("**" + venue + "** · " + decision.get("status", "Unknown search status") +
                 ("" if not decision.get("queries_with_unfinished_pages") else
                  " · unfinished pages for: " + ", ".join(decision["queries_with_unfinished_pages"])))
    rows = review.get("candidate_reviews") or []
    if rows:
        st.dataframe([{"Venue": row["venue"], "Contract": row["market_question"],
                       "Decision": _TARIFF_DECISION_TEXT.get(row["decision"], row["decision"]),
                       "Why": row["reason"], "Quote (bid to ask)": _interval(row.get("quote_interval"), probability=True)}
                      for row in rows], hide_index=True, width="stretch",
                     column_config={"Contract": st.column_config.TextColumn(width=300),
                                    "Why": st.column_config.TextColumn(width=320)})
    if not review:
        st.warning(event.get("reason", "No recorded contract search for this question and horizon.") +
                   " Probability remains unknown; a missing search is not evidence that no market fits.")
        return
    distribution = review.get("distribution")
    if distribution is None:
        st.info("Nothing fits: " + (review.get("reason") or "no searched contract defines a priced bracket family.") +
                " The search is bounded; it does not prove that no such market exists.")
    else:
        st.caption(f"{distribution['venue']}: {len(distribution['atoms'])} brackets of the general US tariff rate on Chinese goods "
                   f"on {distribution['measurement_date']} cover the rate from 0% to an open top, so they form one outcome family. "
                   "Q comes from each bracket's bid/ask, projected onto distributions that sum to one; the Q interval is every "
                   "value those quotes allow. Quotes are indicative Gamma summaries, not a verified order book.")
        if review.get("outcome_table"):
            _outcome_table(st, event, caption=_TARIFF_TABLE_CAPTION, details=False)
        elif review.get("reason"):
            st.info("P&L unavailable: " + review["reason"])
    table = review.get("outcome_table") or {}
    if table:
        summary = table["summary"]
        st.write(f"Portfolio response: **{_money(summary['usd_per_point_rate_surprise'])}** per point of the China rate above what "
                 f"prices imply (Geneva anchor), **{_money(summary['usd_per_point_rate_surprise_envelope'])}** at the April 2025 envelope. "
                 f"Q-expected rate: {table['reference_expectation']:.1f}% (all quotes allow "
                 f"{table['reference_expectation_range_over_q_polytope'][0]:.1f}% to {table['reference_expectation_range_over_q_polytope'][1]:.1f}%)."
                 .replace("$", "\\$"))
        from .tariff_scenarios import compare_anchor_calibrations
        sensitivity = review.get("anchor_sensitivity") or compare_anchor_calibrations(table)
        st.markdown("#### What changes if we calibrate on the other real tariff announcement?")
        st.dataframe([{
            "Rate bracket": item["label"],
            "Q": f"{item['q_reference']:.1%}",
            "Geneva 12 May 2025 (USD)": round(item["pnl_geneva_usd"]),
            "April 3 2025 (USD)": round(item["pnl_april_usd"]),
            "Direction": item["direction_status"].replace("_", " ").capitalize(),
        } for item in sensitivity["outcomes"]], hide_index=True, width="stretch")
        if sensitivity["direction_reversal_count"]:
            st.info("Some outcome P&L signs reverse across the two observed announcement-day calibrations. "
                    "Their direction is not robust to the historical anchor; neither is a causal return estimate.")
        elif sensitivity["worst_case_identity_changes"]:
            st.info("The worst quoted outcome changes with the historical anchor. "
                    "Treat the ranking as specification-sensitive, not a forecast.")
        else:
            st.caption("Both anchor scenarios are shown with the same market-implied Q and no historical-session selection "
                       "based on which gives the preferred result. This is a two-scenario stress, not a confidence interval "
                       "or validation of a causal tariff sensitivity.")
        with st.expander("Measured responses, anchors and assumptions", expanded=False):
            st.dataframe([{"Instrument": p["symbol"], "Weight": f"{p['weight']:.1%}", "Channel": p["channel"],
                           "Return on 12 May 2025": None if not p.get("anchor_returns") else f"{p['anchor_returns']['GENEVA_2025_05_12']:+.2%}",
                           "Return on 3 Apr 2025": None if not p.get("anchor_returns") else f"{p['anchor_returns']['RECIPROCAL_2025_04_03']:+.2%}",
                           "bp per point (Geneva)": None if p.get("loading") is None else round(1e4 * p["loading"], 2),
                           "bp per point (April envelope)": None if p.get("loading_envelope") is None else round(1e4 * p["loading_envelope"], 2),
                           "Basis": p["basis"]} for p in table["positions"]], hide_index=True, width="stretch")
            for anchor in table["factors"]["anchors"]:
                st.write(f"- **{anchor['session']}** ({anchor['change_pp']:+.0f} points): {anchor['event']}")
                for source in anchor["sources"]:
                    st.caption(source["title"] + " · " + source["url"] + " · " + source["locator"])
            from .tariff_scenarios import CONSIDERED_SESSIONS
            st.markdown("**Why two sessions.** Every dated change in the general China rate since 2025, and why only "
                        "these two are used. Listed from the announcements; no return on these sessions was measured to list them.")
            st.dataframe([{"Session": s["session"], "Change": s["change"], "China only": "yes" if s["china_only"] else "no",
                           "Use": {"POINT": "Point", "ENVELOPE": "Far end"}.get(s["use"], "Not used"),
                           "Why": s["reason"], "Source": s["source"]["url"]} for s in CONSIDERED_SESSIONS],
                         hide_index=True, width="stretch")
            st.json(table["uncertainty"])
            for assumption in table["assumptions"]:
                st.write("- " + assumption)
    _tariff_cross_venue(st, review, table)
    related = review.get("related_general_rate_scenario") or {}
    if related:
        with st.expander("Related scenario: general US tariff rate on China", expanded=False):
            st.write("**" + related["title"] + "**")
            st.caption(related["reason"])
            st.caption("These prices and impacts belong to the general-rate scenario below. "
                       "They do not price the original sector-specific or country-specific event.")
            related_event = {
                **event,
                "title": related["title"],
                "distribution": related.get("distribution"),
                "outcome_table": related.get("outcome_table"),
            }
            if related.get("outcome_table"):
                _outcome_table(st, related_event, caption=_TARIFF_TABLE_CAPTION, details=False)
            else:
                st.info("No portfolio P&L is available for this related general-rate scenario.")
            _tariff_cross_venue(st, related, related.get("outcome_table") or {})
            closest_related = related.get("closest_exact_event")
            if closest_related:
                st.write("Exact related question: " + closest_related["definition"])
                st.caption("Quote-implied interval: " +
                           _interval(closest_related.get("q_interval"), probability=True))
            st.json({key: value for key, value in related.items() if key != "outcome_table"}, expanded=False)
    broad = review.get("broad_risk") or {}
    st.markdown("#### The broad risk itself: is it priced?")
    st.write(broad.get("probability_definition") or "")
    closest = broad.get("closest_exact_event")
    if closest:
        st.write(f"Closest exact event: **{closest['definition']}** Q {closest['q_reference']:.1%} "
                 f"(quotes allow {_interval(closest['q_interval'], probability=True)}).")
        for difference in closest["differences"]:
            st.caption("Difference: " + difference)
    st.write("Probability of the broad risk: **unknown**. " + (broad.get("probability") or {}).get("scope", ""))
    with st.expander("Recorded search and contract review", expanded=False):
        st.json(search)
        st.json({key: value for key, value in review.items() if key != "outcome_table"})


_PAYROLL_TABLE_CAPTION = (
    "Each row is the move not already in prices if the first-release payroll change settles in that band: the Q-weighted "
    "sum is zero by construction. The portfolio's response is its own close-to-close return on past release days per "
    "100,000 jobs of surprise, the surprise being the first release minus the Q-expected value of that release's own Kalshi "
    "ladder before 08:20 ET. Ranges span the slope's 90% interval, the value inside each band and every Q the quotes allow; "
    "they are not forecast intervals. Q is not a physical probability.")


def _payroll(st, event):
    """Scheduled jobs report: Kalshi payroll bands, measured surprise response, not-yet-priced P&L."""
    distribution = event.get("distribution") or {}
    st.write(event.get("reviewed_question") or event["original_question"])
    source = event.get("calendar_source") or {}
    if source.get("summary"):
        st.caption("Scheduled: " + source["summary"] + " · " + str(source.get("url")))
    if not distribution.get("atoms"):
        st.info(event.get("reason", "No usable payroll ladder: probability unknown."))
        return
    st.caption(f"Kalshi {distribution['event_ticker']}: {len(distribution['atoms'])} bands from the ladder's 'above X' strikes; "
               "Q weights are the quote-compatible distribution, with every value those quotes allow in the Q interval.")
    _outcome_table(st, event, caption=_PAYROLL_TABLE_CAPTION, details=False)
    if not (event.get("outcome_table") or {}).get("outcomes") and event.get("reason"):
        st.info(event["reason"])
    study = event.get("payroll_response_study") or {}
    fit = study.get("portfolio")
    if fit:
        table = event.get("outcome_table") or {}
        st.write(f"Measured response: **{1e4 * fit['slope']:+.1f} bp** per 100,000 jobs of surprise "
                 f"(standard error {1e4 * fit['se']:.1f} bp; {fit['n']} releases, {fit['first_release_date']} to {fit['last_release_date']})."
                 + ("" if not table else f" Q-expected payrolls: {table['reference_expectation'] / 1000:+,.0f}k."))
        if fit.get("slope_last_third") is not None:
            st.caption(f"Stability: {1e4 * fit['slope_first_two_thirds']:+.1f} bp in the first two thirds of the sample, "
                       f"{1e4 * fit['slope_last_third']:+.1f} bp in the last third"
                       + ("." if fit.get("same_sign_in_both_periods") else ": the sign changed, so the direction is not settled."))
    with st.expander("Payroll response by position, sample and assumptions", expanded=False):
        st.dataframe([{"Instrument": symbol, "bp per 100k surprise": None if row.get("slope") is None else round(1e4 * row["slope"], 1),
                       "Standard error (bp)": None if row.get("se") is None else round(1e4 * row["se"], 1),
                       "Releases": row.get("n"), "Status": row.get("status")}
                      for symbol, row in (study.get("positions") or {}).items()], hide_index=True, width="stretch")
        st.json(study.get("tape_source") or {})
        table = event.get("outcome_table") or {}
        if table:
            st.json(table["uncertainty"])
            for assumption in table["assumptions"]:
                st.write("- " + assumption)


def _lookthrough(st, analysis):
    """What the portfolio holds through its ETFs, from the captured constituent rows."""
    issuers = analysis.get("aggregated_issuer_exposure")
    funds = analysis.get("fund_signed_compositions")
    if not issuers and not funds:
        return
    st.markdown("#### What do you hold through your ETFs?")
    if issuers:
        st.dataframe([{"Exposure": row["symbol"], "Total weight": f"{row['portfolio_weight']:.2%}",
                       "Direct": f"{row['direct_weight']:.2%}",
                       "Through ETFs": ", ".join(f"{k} {v:.2%}" for k, v in row["via_etf"].items()) or "-"}
                      for row in issuers[:8]], hide_index=True, width="stretch")
        st.caption("Issuer look-through identifies direct and indirect exposure channels; constituent weights are NOT added to the ETF valuation a second time.")
    rows = []
    for symbol, comp in sorted((funds or {}).items()):
        if "segments" not in comp:
            rows.append({"Fund": symbol, "Status": comp.get("status"), "Note": comp.get("reason")})
            continue
        rate = comp.get("interest_rate_sensitivity") or {}
        top = sorted(comp["segments"].items(), key=lambda kv: -abs(kv[1]["net_weight"]))[:3]
        rows.append({"Fund": symbol, "Date": comp["effective_date"], "Rows": comp["rows"],
                     "Net reported": f"{comp['net_weight_included']:.4f}",
                     "Short positions": f"{comp['gross_short_weight']:.4%}",
                     "Unreported / residual": f"{comp['unreported_or_residual_nav_weight']:+.4%}",
                     "Main segments": ", ".join(f"{k} {v['net_weight']:.1%}" for k, v in top),
                     "Bond duration (covered)": (f"{rate['covered_average_modified_duration']:.2f}y on {rate['covered_nav_weight']:.0%}"
                                                 if rate else "-")})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Every reported row is kept with its sign. Rows sharing a ticker but carrying different ISINs are different "
                   "securities; a long and a short row of the same TBA CUSIP net out. A row whose ticker field equals the fund stays "
                   "an unresolved collision (kept, never given a duration) unless an authoritative fund identifier resolves it. "
                   "Nothing is clipped or renormalized.")
    rate = analysis.get("portfolio_rate_sensitivity") or {}
    if rate.get("usd_per_bp_parallel_rise_covered") is not None:
        # Escape "$" so Streamlit markdown does not read two amounts as LaTeX.
        st.write((f"Covered bond holdings change by **{_money(rate['usd_per_bp_parallel_rise_covered'])} per +1 bp** parallel yield rise "
                  f"(range {_interval(rate['usd_per_bp_interval_covered'])}); "
                  f"{rate['portfolio_weight_with_unknown_bond_duration']:.1%} of the portfolio is bonds without machine-readable terms.").replace("$", "\\$"))
    with st.expander("Fund composition evidence", expanded=False):
        st.json(funds or {})


_MATCH_DECISION_TEXT = {
    "NOTHING_FITS_WITHIN_RETRIEVED_EVIDENCE_NOT_GLOBAL_MARKET_ABSENCE":
        "Nothing fits in the retrieved evidence (the search is bounded; this does not prove no market exists)",
}


def _unmodeled(st, event):
    """An event outside the reviewed families still shows its question and its match decision."""
    st.write(event["original_question"])
    comparisons = event.get("original_matching_comparisons") or []
    if comparisons:
        st.markdown("#### Does any contract fit?")
        st.dataframe([{"Market question": row.get("market_question") or "No contract retained",
                       "Difference": row.get("difference", ""),
                       "Decision": _MATCH_DECISION_TEXT.get(row.get("decision"), row.get("decision", "Unknown")),
                       "Q": _interval(row.get("contract_pricing_measure_Q_interval"), probability=True)}
                      for row in comparisons], hide_index=True, width="stretch")
    search = event.get("market_search") or {}
    candidates = search.get("candidates") or []
    if candidates:
        st.markdown("#### Related markets discovered — not event probabilities")
        st.dataframe([{"Venue": row.get("venue"), "Contract": row.get("market_question"),
                       "Search status": "RELATED_ONLY_NOT_SETTLEMENT_EQUIVALENT"}
                      for row in candidates], hide_index=True, width="stretch")
        st.caption("Related markets do not identify the broader portfolio risk, its physical probability, "
                   "or an asset-price response. No quote is transferred to this event.")
    st.info(event.get("reason", "No reviewed default model covers this event.")
            + " Probability and P&L stay unknown; neither is set to zero.")


def _named_sensitivity(st, sensitivity):
    if not sensitivity or not sensitivity.get("per_symbol"):
        return
    st.markdown("#### Exposure named by the event rationale")
    st.dataframe([{"Exposure": row["symbol"], "Portfolio weight": f"{row['portfolio_weight']:.2%}",
                   "USD per 1% move": _money(row["usd_per_one_percent_move"])} for row in sensitivity["per_symbol"]],
                 hide_index=True, width="stretch")
    st.caption(sensitivity["interpretation"])


def _excluded_offsets_caption(st, report):
    rows = [row for row in report.get("holdings_quality_review") or [] if row.get("excluded_negligible_offset_rows")]
    if rows:
        st.caption("Kept out of look-through as negligible offsets (negative derivative or unclassified rows, at most 0.1% "
                   "of NAV in total; the fund's holdings are otherwise accepted): " + "; ".join(
                       f"{row['symbol']} {row['excluded_negligible_offset_rows']} row(s), {row['excluded_offset_nav']:.4%} of NAV"
                       for row in rows) + ".")


_TOLERANCE_TEXT = {
    "SHORT_ROWS_NETTED_BY_SAME_IDENTIFIER_LONGS": lambda m: (
        f"{m['rows']} short rows ({-m['gross_short_nav']:.2%} of NAV) net against long rows of the same identifier, "
        "none net short"),
    "REPEATED_ROWS_OF_ONE_IDENTIFIER_SUMMED": lambda m: f"repeated rows of one security summed ({m['nav']:.4%} of NAV)",
    "TICKER_FIELD_EQUALS_FUND_KEPT_UNRESOLVED": lambda m: (
        f"{m['rows']} row whose ticker field equals the fund but carries its own identifier ({m['nav']:.4%} of NAV), "
        "kept unresolved with no duration"),
    "UNIDENTIFIED_CASH_OR_SHORT_TERM_ROWS": lambda m: f"{m['nav']:.2%} of NAV in cash or short-term rows without identifier",
    "LISTED_NAV_SHORTFALL": lambda m: f"{m['unlisted_nav']:.2%} of NAV not listed, not renormalized",
    "SINGLE_LINE_WITHIN_NAV_BAND": lambda m: (
        f"one holding line at {m['line_nav']:.3%} of NAV without identifier: nothing to look through, "
        "priced as a whole through its own measured beta"),
    "DUST_LINES_SET_ASIDE": lambda m: f"{m['rows']} further line(s) of {m['abs_nav']:.1e} of NAV set aside as rounding dust",
}


def _fund_reviews_caption(st, analysis):
    """Which fund holdings the class-aware review accepted, and under which measured tolerances."""
    reviews = [row for row in analysis.get("fund_holdings_reviews") or []
               if row.get("holdings_accepted_for_assessment") and row.get("status") != "PLAIN_EQUITY_HOLDINGS_ALREADY_ADMITTED"]
    if not reviews:
        return
    strict = [row["fund_symbol"] for row in reviews if row["status"] == "SIGNED_FIXED_INCOME_SOURCE_CLASS_ACCEPTED"]
    parts = []
    if strict:
        parts.append(", ".join(strict) + ": bond holdings accepted (different ISINs under a shared ticker are different bonds; "
                     "a cash liability of at most 0.1% of NAV is kept)")
    for row in reviews:
        if row["fund_symbol"] in strict:
            continue
        texts = [_TOLERANCE_TEXT[t["tolerance"]](t["measured"]) for t in row.get("declared_tolerances_used") or []
                 if t.get("tolerance") in _TOLERANCE_TEXT]
        parts.append(row["fund_symbol"] + ": " + "; ".join(texts))
    st.caption("Fund holdings accepted under declared tolerances, each within its stated limit and never repaired. "
               + " · ".join(parts) + ".")


def _ranking_robustness_panel(st, report, events, key_prefix):
    """Failure-aware ranking audit over quote bounds and declared stress slopes."""
    source_data, source_receipts = load_archived_calibration_sources(Path(__file__).resolve().parents[1])
    calibration = historical_stress_audit(report, source_data)
    st.markdown("#### Can the portfolio's top risk change when the assumptions change?")
    severity = st.slider("Independent stress multiplier for each modeled event",
                         min_value=0.0, max_value=3.0, value=(0.5, 1.5), step=0.1,
                         key=key_prefix + "_robust_severity")
    st.caption("0.5×–1.5× is an illustrative stress domain, NOT a statistical confidence range. "
               "Each event may move independently; change the bounds above to test a different declared domain. "
               "Quote sensitivity uses a conservative outer relaxation of the source bid/ask constraints.")
    try:
        audit = analyze_ranking(events, severity_domain=severity,
                                alternative_vectors=calibration["alternative_vectors_by_event"])
    except (RobustnessInputError, KeyError, TypeError) as exc:
        st.warning("The rank stability audit cannot certify these records: " + str(exc))
        return
    if audit["visible_unquantifiable_risks"]:
        st.warning(f"{audit['unquantifiable_event_count']} risk(s) cannot receive a complete numerical rank. "
                   "They remain on the investigation queue, not a claim of low exposure or zero probability.")
        st.dataframe([{"Unquantifiable risk": row["title"], "Reason": row["reason"],
                       "Risk family": row.get("family")} for row in audit["visible_unquantifiable_risks"]],
                     hide_index=True, width="stretch")
    if audit["modeled_rank"]:
        leader = audit["modeled_rank"][0]
        if audit["status"] == "CONDITIONALLY_ROBUST_PRICED_SCENARIO_LEADER":
            st.success("The modeled leader remains first for every quote/calibration stress in the declared "
                       "outer domain, AMONG THE PRICED SCENARIOS ONLY: " + leader["title"])
        elif audit["ranking_reversal_witnesses"]:
            st.warning("A different priced scenario becomes the leader under an EXPLICIT, FEASIBLE "
                       "reference-Q / archived-calibration / independent-severity configuration. "
                       "This is a model stress witness, NOT a historical market outcome or a claim "
                       "that the stresses will occur together. Baseline leader: " + leader["title"])
        else:
            st.info("The lead is NOT certified over this stress domain. Overlapping conservative ranges are "
                    "not proof that an actual feasible rank reversal exists. Baseline leader: " + leader["title"])
        st.dataframe([{"Modeled event": row["title"],
                       "Reference Q-std": _money(row["reference_risk_usd"]),
                       "Q + historical specification outer bounds": _interval(row["quote_and_calibration_interval_usd"]),
                       "With severity stress": _interval(row["stress_interval_usd"]),
                       "Scenarios individually priced": len(row["priced_scenarios"]),
                       "Broad event Q proven": "No" if row["broad_event_probability_certified"] is False
                                               else "Not asserted"} for row in audit["modeled_rank"]],
                     hide_index=True, width="stretch")
        if audit["challenger_thresholds"]:
            st.caption("Break-even test: how much one challenger's severity must be multiplied to TIE the "
                       "reference leader if all other quote weights and severities stay fixed. It does not "
                       "certify that this multiplier is economically plausible.")
            st.dataframe([{"Challenger": row["title"],
                           "Multiplier to tie baseline leader": (
                               None if row["severity_multiplier_to_tie_reference_leader"] is None else
                               round(row["severity_multiplier_to_tie_reference_leader"], 3)),
                           "Quote/specification change alone ruled out": row["quote_or_calibration_only_can_be_ruled_out"],
                           "Severity change alone ruled out": row["severity_only_can_be_ruled_out"]}
                          for row in audit["challenger_thresholds"]], hide_index=True, width="stretch")
        if audit["ranking_reversal_witnesses"]:
            st.markdown("##### Witnessed changes to the top-ranked priced event")
            st.dataframe([{"New leader": w["new_leader_title"],
                           "Assumed top-event severity": w["new_leader_dominant_calibration"]["severity_multiplier"],
                           "Top-event calibration": w["new_leader_dominant_calibration"]["calibration"],
                           "New leader Q-std": _money(w["new_leader_stressed_q_std_usd"]),
                           "Highest other Q-std": _money(w["highest_other_stressed_q_std_usd"]),
                           "Strict lead": _money(w["strict_margin_usd"])}
                          for w in audit["ranking_reversal_witnesses"]],
                         hide_index=True, width="stretch")
            st.caption("Construction: reference Q for every scenario; the displayed archival model/calibration "
                       "and upper severity for the challenger; the documented minimum archival calibration "
                       "and lower severity for every other event. The next expander contains source receipts. "
                       "This witness certifies reversibility *only within the independently varying modeled inputs*.")
    st.caption("These bounds neither identify physical P nor the state prices embedded by equities. "
               "Unknown event probabilities, missing impact channels, and overlapping contract scenarios "
               "are not added to the ranked numbers.")
    with st.expander("Historical calibration checks and reproducible source receipts", expanded=False):
        st.caption("CPI: three observed subperiod SPY surprise slopes. Payrolls: early/late slopes only "
                   "when this is the exact archived portfolio. Tariffs: both 2025 announcement episodes. "
                   "FOMC: the policy-decision quote does NOT price path guidance or the press conference. "
                   "All are retrospective stress specifications, not causal estimates or predictive bands.")
        st.dataframe(_evidence_rows(calibration["diagnostics"]), hide_index=True, width="stretch")
        st.json(source_receipts)


def _navigate_results(st, key_prefix, selected=None, *, portfolio_details=False):
    st.session_state[key_prefix + "_selected_risk"] = selected
    st.session_state[key_prefix + "_portfolio_tools_open"] = portfolio_details
    st.session_state[key_prefix + "_risk_sources"] = False


def _risk_source_view(st, event, key_prefix):
    """The existing family inspector, evaluated only after a deliberate click."""
    if event["family"] in {"FED_POLICY", "US_INFLATION"}:
        _macro(st, event, key_prefix=key_prefix)
    elif event["family"] == "US_PAYROLLS":
        _payroll(st, event)
    elif event["family"] in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}:
        _geopolitical(st, event)
    elif event["family"] == "TRADE_TARIFFS":
        _tariff(st, event)
    else:
        _unmodeled(st, event)
    _named_sensitivity(st, event.get("named_exposure_sensitivity"))


def _simple_risk_detail(st, item, key_prefix):
    from .simple_results import date_label, impact, outcome_rows, probability, probability_scope, title, why_it_matters
    event = item["source_event"]
    st.subheader(title(event))
    st.caption(date_label(event))
    st.write(why_it_matters(event))
    linked = (event.get("named_exposure_sensitivity") or {}).get("per_symbol") or []
    if linked:
        st.caption("Linked holdings: " + ", ".join(
            f"{row['symbol']} {row['portfolio_weight']:.1%}" for row in linked[:5]))
    rows = outcome_rows(event)
    if rows:
        st.markdown("#### What could happen")
        st.dataframe([{"Outcome": row["display_label"],
                       "Market odds": probability(row),
                       "Portfolio gain / loss": impact(row)}
                      for row in rows], hide_index=True, width="stretch",
                     column_config={"Outcome": st.column_config.TextColumn(width="large"),
                                    "Market odds": st.column_config.TextColumn(width="small"),
                                    "Portfolio gain / loss": st.column_config.TextColumn(width="small")})
        st.caption(probability_scope(event) + " Market odds are pricing weights, not a calibrated forecast. "
                   "Dollar amounts use the recorded scenario assumptions.")
        if event.get("contract_first"):
            st.caption("Different contracts can overlap; their odds and impacts must not be added.")
        st.caption("Already in prices: the model subtracts its assumed market-weighted expectation from each outcome.")
    else:
        st.info("No usable probability or complete impact estimate for this event.")
        st.caption(_state_text(event))
    # Source probabilities and scenario assumptions are deliberately separate.
    # The optional factor-only tool still works when an original event has no
    # market settlement match; it cannot manufacture that missing linkage.
    factor_what_if = event.get("factor_only_what_if") or {}
    if factor_what_if.get("status", "").startswith(("COMPLETE_", "PARTIAL_")):
        with st.expander("Explore a portfolio what-if", expanded=False):
            st.caption("Choose a hypothetical combination of market, rate and gold shocks. "
                       "This is a sensitivity calculation, not an estimate of how the selected event will move prices.")
            cases = {case["regime"]: case for case in factor_what_if.get("assumption_stress_regimes", [])}
            if cases:
                labels = {"BROAD_RISK_OFF": "Broad market sell-off",
                          "RATE_SPIKE": "Rates rise",
                          "FLIGHT_TO_SAFETY": "Flight to safety"}
                selected_regime = st.selectbox("Assumed market regime", list(cases),
                    format_func=lambda key: labels.get(key, key), key=key_prefix + "_factor_regime")
                case = cases[selected_regime]
                assumptions = {row["name"]: row["return_range_pct"]
                               for row in case["factor_shock_assumptions"]}
                equity = assumptions.get("SPY_INDEX_RETURN")
                rates = assumptions.get("DURATION_ONE_BOND_PRICE_RETURN")
                gold = assumptions.get("GOLD_SPOT_RETURN")
                st.caption("Assumptions · SPY " + _interval(equity).replace("$", "") +
                           "% · yield " + (f"{-rates[1] * 100:g} to {-rates[0] * 100:g} bp" if rates else "unknown") +
                           " · gold " + _interval(gold).replace("$", "") + "%")
                amount = case.get("sharp_assumed_factor_pnl_range_usd")
                if amount is not None:
                    st.metric("Portfolio P&L under these assumptions", _interval(amount))
                else:
                    st.metric("Known-position contribution only", _interval(case.get("known_position_pnl_range_usd")))
                    st.caption(f"Unmodeled original portfolio weight: {factor_what_if.get('unmodeled_original_position_weight', 0):.1%}. "
                               "The missing position return is unknown, not zero.")
                st.caption(factor_what_if.get("missing_direct_event_mechanism", ""))
                with st.expander("Factor evidence and model checks", expanded=False):
                    st.dataframe(_evidence_rows(factor_what_if.get("positions", [])),
                                 hide_index=True, width="stretch")
                    if factor_what_if.get("issuer_documentary_factor_evidence"):
                        st.dataframe(_evidence_rows(factor_what_if["issuer_documentary_factor_evidence"]),
                                     hide_index=True, width="stretch")
    with st.expander("Exact question", expanded=False):
        st.write(event.get("title") or title(event))
        if event.get("original_question"):
            st.write(event["original_question"])
    if st.toggle("Sources & assumptions", key=key_prefix + "_risk_sources"):
        _risk_source_view(st, event, key_prefix + "_selected")


def render_reviewer_dashboard(st, report, *, key_prefix="opt1_reviewer"):
    """A small answer screen, followed by one risk or optional portfolio tools.

    Navigation changes presentation state only. Old reports and exports remain
    untouched, and the complete analytical controls execute only on request.
    """
    from html import escape
    from .simple_results import date_label, downside, has_partial_impact, money, probability, probability_scope, title
    from .tariff_scenarios import normalize_recorded_tariff_scopes
    report, scope_corrections = normalize_recorded_tariff_scopes(report)
    brief = build_reviewer_overview(report)
    # Legacy payroll/tariff reports can carry a finite known-position subtotal.
    # It cannot stand in for the entire portfolio on the simplified screen.
    for item in brief["rows"]:
        if has_partial_impact(item["source_event"]):
            item["full_outcome_model_available"] = False
            item["modeled_q_std_usd"] = None
            item["worst_priced_outcome_usd"] = None
    brief["unquantified"] = [item for item in brief["rows"] if not item["full_outcome_model_available"]]
    brief["modeled_count"] = len(brief["rows"]) - len(brief["unquantified"])
    by_id = {row["event_id"]: row for row in brief["rows"]}
    watchlist = [r for r in report.get('risk_watchlist', [])
                 if isinstance(r, dict)
                 and r.get('origin') == 'CLAUDE_RISK_HYPOTHESIS_NOT_INDEPENDENTLY_VERIFIED']
    token = (brief.get("evaluated_at_utc"), tuple(by_id),
             json.dumps(report.get("portfolio") or {}, sort_keys=True))
    if st.session_state.get(key_prefix + "_report_token") != token:
        _navigate_results(st, key_prefix)
        st.session_state[key_prefix + "_report_token"] = token
        st.session_state[key_prefix + "_show_all"] = False
        st.session_state[key_prefix + "_show_unpriced"] = False
    selected = st.session_state.get(key_prefix + "_selected_risk")
    tools_open = st.session_state.get(key_prefix + "_portfolio_tools_open", False)
    if selected is not None or tools_open:
        st.button("Back to results", key=key_prefix + "_back",
                  on_click=_navigate_results, args=(st, key_prefix))
        if tools_open:
            st.subheader("Portfolio details")
            st.caption("ETF exposures, model controls and the full evidence record.")
            if scope_corrections:
                st.caption("Scope reviewed with the current rules: the original tariff risk remains unpriced. "
                           "The recorded general-China rate calculation is a separate related scenario.")
            if watchlist:
                with st.expander("Research leads and issuer evidence", expanded=False):
                    st.dataframe(_evidence_rows(watchlist), hide_index=True, width="stretch")
            web_sources = (report.get('web_research') or {}).get('sources') or []
            if web_sources:
                from .live_web_research import public_web_research
                sources = public_web_research(report.get('web_research'))['sources']
                if sources:
                    with st.expander("Online risk research · source links", expanded=False):
                        st.caption("Web search leads are not independently confirmed events, "
                                   "probabilities or return estimates. Community posts are signals only.")
                        st.dataframe([{'Source': s['title'], 'Website': s['domain'],
                                       'Query': s['search_query'], 'URL': s['url']}
                                      for s in sources], hide_index=True, width="stretch")
            render_default_assessment(st, report, key_prefix=key_prefix)
        elif selected in by_id:
            _simple_risk_detail(st, by_id[selected], key_prefix)
        elif isinstance(selected, str) and selected.startswith("watch:"):
            try:
                item = watchlist[int(selected.split(":", 1)[1])]
            except (IndexError, ValueError):
                _navigate_results(st, key_prefix)
                st.rerun()
                return
            st.subheader(item.get("risk_title") or "Risk to investigate")
            st.write(item.get("economic_channel") or item.get("exposure_rationale") or "")
            st.info("Probability and portfolio impact are not estimated for this research lead.")
            if item.get("monitoring_signal"):
                st.markdown("**What to watch**")
                st.write(item["monitoring_signal"])
            with st.expander("Portfolio link and sources", expanded=False):
                st.write(item.get("exposure_rationale") or "")
                st.json(item)
        else:
            _navigate_results(st, key_prefix)
            st.rerun()
        return

    if brief["source_quality_partial"]:
        st.warning("Some provider or ETF data are incomplete. The available estimates cover only the supported exposures.")
    positions = (report.get("portfolio") or {}).get("positions") or []
    symbols = " / ".join(str(row.get("symbol", "")) for row in positions)
    value = brief["portfolio_value_usd"]
    st.markdown("### Your risk overview")
    st.caption((f"${value:,.0f}" if isinstance(value, (int, float)) else "Portfolio")
               + (" · " + symbols if symbols else "")
               + f" · {brief['modeled_count']} modeled scenarios")
    priced = [row for row in brief["rows"] if row["full_outcome_model_available"]]
    if priced:
        st.markdown("**Start with these risks**")
        st.caption("Each card shows one downside outcome. Open a risk to compare all outcomes.")
        show_all = st.session_state.get(key_prefix + "_show_all", False)
        visible = priced if show_all else priced[:3]
        # Three native containers per row keep actions attached to their event.
        for offset in range(0, len(visible), 3):
            for col, item in zip(st.columns(3), visible[offset:offset + 3]):
                event = item["source_event"]
                shown = downside(event)
                with col:
                    with st.container(border=True):
                        st.markdown('<div class="cr-risk-kicker">' + escape(date_label(event)) + '</div>'
                                    '<div class="cr-risk-title">' + escape(title(event)) + '</div>',
                                    unsafe_allow_html=True)
                        if shown:
                            st.caption(shown["display_label"])
                            st.metric("If this outcome happens", money(shown.get("pnl_if_outcome_usd")),
                                      help="Lowest modeled P&L among outcomes with reference market weight of at least 1%. "
                                           "It is one scenario, not a maximum-loss bound.")
                            st.markdown("**Market odds: " + probability(shown) + "**")
                        else:
                            st.metric("Portfolio impact", "Unknown")
                            st.caption("No complete priced outcome for this portfolio.")
                        if event.get("contract_first"):
                            st.caption(probability_scope(event))
                        st.button("View risk", key=key_prefix + "_open_" + str(item["event_id"]),
                                  width="stretch", on_click=_navigate_results,
                                  args=(st, key_prefix, item["event_id"]))
        if len(priced) > 3:
            st.toggle(f"Show all {len(priced)} modeled risks", key=key_prefix + "_show_all")
        st.caption("Ordered by modeled variation under market weights. Odds come from quotes; impacts use scenario assumptions. "
                   "Shown outcomes have reference weight ≥ 1%; rarer outcomes remain in the details.")
    else:
        st.info("No complete priced scenario is available. The risks found are listed below.")

    unpriced = [(row["event_id"], title(row["source_event"]), date_label(row["source_event"]))
                for row in brief["unquantified"]]
    unpriced += [("watch:" + str(index), item.get("risk_title") or "Research lead", "Research lead")
                for index, item in enumerate(watchlist)]
    if unpriced:
        st.markdown("**Also worth watching · estimates unavailable**")
        st.caption("These risks can matter even when their probability or impact cannot be estimated.")
        show_unpriced = st.session_state.get(key_prefix + "_show_unpriced", False)
        visible = unpriced if show_unpriced else unpriced[:3]
        for event_id, label, date in visible:
            left, right = st.columns([5, 1])
            left.write(label)
            left.caption(date)
            right.button("View", key=key_prefix + "_open_" + str(event_id),
                         on_click=_navigate_results, args=(st, key_prefix, event_id))
        if len(unpriced) > 3:
            st.toggle(f"Show all {len(unpriced)} unpriced events and research leads", key=key_prefix + "_show_unpriced")
    st.button("Portfolio details & model controls", key=key_prefix + "_portfolio_details",
              on_click=_navigate_results, args=(st, key_prefix), kwargs={"portfolio_details": True})


def render_default_assessment(st, report, *, key_prefix="opt1_default"):
    """Render derived values without reading a clock, acquiring data or changing evidence."""
    analysis = report["default_analysis"]
    st.markdown("### Portfolio event assessment")
    st.write("Compare event outcomes, quote-compatible probabilities, declared announcement components and the part assumed already priced in.")
    st.caption("Recorded macro evaluation: " + analysis["evaluated_at_utc"] +
               " · Separate public-search timestamps remain on their source cards.")
    events = analysis.get("events", [])
    if not events:
        st.info("No reviewed event survived this portfolio's investigation. This is not evidence of no risk.")
        return
    # The answer columns come first, so they are visible without scrolling sideways.
    st.dataframe([{"Order": e["ranking"]["display_order"], "Event": e["title"], "Date": e.get("event_date"),
                   "Largest loss (Q ≥ 1%)": _largest_loss(e)[0], "If this happens": _largest_loss(e)[1],
                   "State": _state_text(e),
                   "Q-std of unpriced P&L": _money((e.get("risk_ranking") or {}).get("q_standard_deviation_usd")),
                   "Probability evidence": _probability_summary(e),
                   "Why listed": _why_listed(e)} for e in events],
                 hide_index=True, width="stretch",
                 column_config={"Order": st.column_config.NumberColumn(width="small"),
                                "Event": st.column_config.TextColumn(width=270),
                                "Largest loss (Q ≥ 1%)": st.column_config.TextColumn(width=150),
                                "If this happens": st.column_config.TextColumn(width=250),
                                "State": st.column_config.TextColumn(width=250)})
    st.caption("Events are ordered by the Q-standard deviation of the P&L not already in prices: how far each scheduled outcome "
               "can move this portfolio beyond what contract prices imply. Events without a complete outcome table follow. "
               "Every FOMC decision, CPI release and jobs report inside the 90-day horizon is always listed, whatever the model proposed "
               "(jobs reports in runs that read the payroll ladder), and so "
               "are the reviewed Taiwan and chip-export risks whenever a holding's annual report documents the link (NVIDIA and AMD: "
               "TSMC foundry and China export licences; Micron: DRAM fabs in Taiwan; Apple: manufacturing partners in Taiwan; TSMC "
               "itself), held directly or through a fund; the US tariff rate on Chinese goods when a holding's report documents "
               "manufacturing in China (Apple). "
               "Overlapping events must not be added.")
    _ranking_robustness_panel(st, report, events, key_prefix)
    # Display a defensible YES-versus-all-NO settlement case before the
    # broader macro component tables; no narrower Q is transferred to them.
    derived = report.get('source_exact_binding_derivation') or {}
    complete = [case for case in derived.get('bound_case_results', [])
                if case.get('complete_bound_case') is True]
    if complete:
        exact_view = st.expander('Single-contract YES/NO view (same model family, binary contract settlement)', expanded=False)
        exact_view.dataframe([{
            'Contract': case['contract_id'], 'Exact question': case['title'],
            'Q interval (bid/ask)': (f"{case['q_interval'][0]:.1%}–{case['q_interval'][1]:.1%}"
                                     if case.get('q_interval') is not None else 'UNKNOWN'),
            'YES minus NO': _money(case.get('delta_pnl_usd')),
            'YES residual (π=Q assumed)': _money(case.get('yes_pnl_usd')),
            'NO residual (π=Q assumed)': _money(case.get('no_pnl_usd')),
            'Scope': 'Full original portfolio — declared scenario',
        } for case in complete], hide_index=True, width='stretch')
        exact_view.caption('YES is precisely the contract settlement. NO includes ALL remaining outcomes and source exceptions; '
                   'its modeled state is a joint-Q conditional mixture, not merely a rate hold. The observed market '
                   'bid/ask bounds a pricing weight, NOT physical event probability. The asset-implied π=Q assumption '
                   'is NOT empirically verified. Historical return transfer failed earlier forward-validation; '
                   'the numbers above are conditional scenario arithmetic, not a causal forecast or executable hedge.')
        exact_view.json([case for case in complete])
    elif derived.get('ledger'):
        st.caption('No source-defined exact contract has a complete admitted YES/NO economic binding. '
                   'Incoherent quotes and missing complement states are not silently repaired.')
    _lookthrough(st, analysis)
    _excluded_offsets_caption(st, report)
    _fund_reviews_caption(st, analysis)
    class_reviews = report.get('class_aware_fund_reviews') or []
    if class_reviews:
        with st.expander('ETF source-class validity (separate from equity look-through)', expanded=False):
            st.dataframe([{
                'ETF': row.get('fund_symbol'), 'Status': row.get('status'),
                'Original equity-style reliable': row.get('original_plain_holdings_reliable'),
                'Source-class accepted': row.get('eligible_for_source_class_attribution'),
                'Holdings accepted for assessment': row.get('holdings_accepted_for_assessment'),
                'Signed NAV': row.get('source_signed_nav_sum'),
                'Cash liability / NAV': row.get('signed_usd_cash_liability_nav'),
                'Non-cash short rows': row.get('noncash_short_rows'),
                'Original quality flags': ', '.join(row.get('original_issues_retained') or []),
            } for row in class_reviews], hide_index=True, width='stretch')
            st.caption('Bond source-class acceptance recognizes distinct ISINs under generic tickers and small negative cash; '
                       'the second tier adds the declared tolerances listed above with their measured values. Neither turns '
                       'bonds into equities or reweights holdings. A fund outside every tolerance keeps its structural gap, '
                       'shown here and on each table that uses it. Original equity look-through controls and all raw/source '
                       'hashes are preserved.')
    by_id = {e["event_id"]: e for e in events}
    selected = st.selectbox("Inspect an event", list(by_id), format_func=lambda key: by_id[key]["title"],
                            key=key_prefix + "_event")
    event = by_id[selected]
    if event["family"] in {"FED_POLICY", "US_INFLATION"}:
        _macro(st, event, key_prefix=key_prefix)
    elif event["family"] == "US_PAYROLLS":
        _payroll(st, event)
    elif event["family"] in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}:
        _geopolitical(st, event)
    elif event["family"] == "TRADE_TARIFFS":
        _tariff(st, event)
    else:
        _unmodeled(st, event)
    _named_sensitivity(st, event.get("named_exposure_sensitivity"))
    with st.expander("Portfolio evidence, coverage and retained limitations", expanded=False):
        st.dataframe(_evidence_rows(report.get("exposures", {}).get("positions", [])), hide_index=True, width="stretch")
        st.json(report.get("exposures", {}).get("coverage", {}))
        st.json(report.get("holdings_quality_review", []))
        st.json(analysis.get("documentary_instrument_classes", []))
        st.json(report.get("stages", {}))
        for limitation in analysis.get("limitations", []) + report.get("limitations", []):
            st.write("- " + limitation)


def render_captured_assessments(st, *, root: Path):
    """Open packaged, hashed derived examples without pretending they use sidebar inputs."""
    folder = Path(root) / "data" / "verified_assessments"
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest["reports"]
        if not entries:
            raise ValueError("Empty captured assessment inventory")
        labels = {row["label"]: row for row in entries}
        selected = st.selectbox("Captured portfolio", list(labels), key="opt1_captured_portfolio")
        entry = labels[selected]
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid captured report path")
        body = (folder / filename).read_bytes()
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Captured report hash mismatch")
        report = json.loads(body)
        if "default_analysis" not in report:
            raise ValueError("Captured report lacks the reviewed projection")
    except (OSError, ValueError, TypeError, KeyError):
        st.error("The packaged assessment cannot be verified. No captured result is displayed.")
        return
    st.caption("Recorded research example. Opening it makes no provider requests; portfolio editor inputs do not change this example.")
    st.dataframe(report["portfolio"]["positions"], hide_index=True, width="stretch")
    st.metric("Captured portfolio value", _money(report["portfolio"]["notional"]))
    st.caption("Quotes are historical evidence at their recorded timestamps. The original incomplete authenticated acceptance and Windows cold-start failure remain open.")
    render_reviewer_dashboard(st, report, key_prefix="opt1_captured_" + entry["id"])
    st.download_button("Export captured assessment (JSON)",
        json.dumps(report, indent=2, allow_nan=False), file_name=filename,
        mime="application/json", key="opt1_captured_export")


def render_r19_heldout_sector_captures(st, *, root: Path):
    """Replay the exact R19 new-sector derived evidence; never redo a live read."""
    folder = Path(root) / 'reports' / 'r19_live_holdout'
    try:
        from hashlib import sha256
        receipt = json.loads((Path(root) / 'benchmarks' / 'r19_frozen_receipt.json').read_text(encoding='utf-8'))
        batch = json.loads((folder / 'batch_manifest.json').read_text(encoding='utf-8'))
        score = json.loads((folder / 'frozen_score.json').read_text(encoding='utf-8'))
        if score['benchmark_sha256'] != receipt['benchmark_sha256']:
            raise ValueError('Holdout benchmark hash mismatch')
        available = {}
        for case in batch['runs']:
            name = case.get('report')
            if name and Path(name).name == name:
                raw = (folder / name).read_bytes()
                if sha256(raw).hexdigest() != case.get('report_sha256'):
                    raise ValueError('Recorded report failed hash verification')
                available[case['case_id']] = (json.loads(raw), raw)
        if len(available) != 3:
            raise ValueError('Incomplete R19 sector archive')
    except (OSError, ValueError, TypeError, KeyError):
        st.warning('The independent sector capture could not be fully verified. No stale or partial substitute is displayed.')
        return
    chosen = st.selectbox('Recorded sector portfolio', sorted(available),
                          key='opt1_r19_sector_choice',
                          format_func=lambda name: {
                              'datacenters': 'Data centers · EQIX / QQQ',
                              'pharma': 'Pharmaceuticals · PFE',
                              'logistics': 'Shipping & logistics · FDX',
                          }.get(name, name))
    report, original_body = available[chosen]
    st.caption('Saved sector study · 26 October 2026 · Original R19 capture')
    render_reviewer_dashboard(st, report, key_prefix='opt1_r19_' + chosen)
    with st.popover('About this recording'):
        st.caption('R19: three untouched sector portfolios, authenticated captures on 26 October 2026. '
                   'This mode makes no API calls. Business risk proposals are not verified market probabilities.')
        st.write(('Frozen risk-scout test: **' + str(score['thematic_hits']) + '/' +
                  str(score['thematic_denominator']) + '** issuer-risk families; ' +
                  str(score['fully_eligible_source_cases']) + '/3 source-quality eligible. '
                  'This does not validate event P&L.'))
        st.caption('Evidence status: ' + report.get('verification_status', 'UNKNOWN'))
        st.caption('The download preserves the original R19 output, including any scope mapping corrected in this view.')
    st.download_button('Download original R19 report (JSON)', original_body,
                       file_name='castle_r19_' + chosen + '.json', mime='application/json',
                       key='opt1_r19_export')


_LATEST_CAPTURE_CASES = {
    'technology_and_treasury': ('NVDA/QQQ/TLT', {'NVDA', 'QQQ', 'TLT'}),
    'bonds': ('BND/IEF', {'BND', 'IEF'}),
    'outside_old_registry': ('AAPL/JPM/GLD', {'AAPL', 'JPM', 'GLD'}),
}


EARLIER_CAPTURE_FOLDER = 'live_verified_20261022'
# Recorded on 23 October by the R9 pipeline; kept for audit.
RELEASE_CAPTURE_FOLDER = 'live_verified_20261023'
# Original R14 provider capture; the current interface presents its dated results.
FINAL_CAPTURE_FOLDER = 'live_verified_20261025'


def load_latest_verified_live_capture(root: Path, folder_name: str = EARLIER_CAPTURE_FOLDER):
    """Verify one delivery's three derived reports; retain their exact export bytes."""
    folder = Path(root) / 'reports' / folder_name
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'castle.verified_live_capture.v1':
        raise ValueError('Unknown verified-capture manifest')

    def timestamp(value):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError('Capture provenance requires UTC timestamps')
        return parsed

    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    if not digest(manifest.get('source_archive_sha256')):
        raise ValueError('Missing source-archive digest')
    recomputed = timestamp(manifest['recomputed_at_utc'])
    entries = manifest['reports']
    if (not isinstance(entries, list) or len(entries) != len(_LATEST_CAPTURE_CASES)
            or {entry['id'] for entry in entries} != set(_LATEST_CAPTURE_CASES)):
        raise ValueError('The complete three-portfolio inventory is required')
    loaded = {}
    for entry in entries:
        case_id = entry['id']
        filename = entry['file']
        label, symbols = _LATEST_CAPTURE_CASES[case_id]
        if filename != case_id + '.json' or Path(filename).name != filename or entry['label'] != label:
            raise ValueError('Invalid captured portfolio identity or report filename')
        body = (folder / filename).read_bytes()
        if not digest(entry.get('sha256')) or hashlib.sha256(body).hexdigest() != entry['sha256']:
            raise ValueError('Verified-capture report hash mismatch')
        report = json.loads(body)
        captured = timestamp(entry['captured_at_utc'])
        if (captured > recomputed
                or captured != timestamp(report['evaluated_at_utc'])
                or captured != timestamp(report['default_analysis']['evaluated_at_utc'])):
            raise ValueError('Capture and recomputation provenance do not reconcile')
        if {p['symbol'] for p in report['portfolio']['positions']} != symbols:
            raise ValueError('Captured report belongs to a different portfolio')
        if report.get('export_scope') != 'DERIVED_COEFFICIENTS_AGGREGATE_EXPOSURES_PUBLIC_QUOTES_AND_RESPONSE_HASHES_NO_MASSIVE_RAW_DATA':
            raise ValueError('Only derived public-assessment exports may be displayed')
        loaded[case_id] = (entry, report, body)
    # A three-portfolio batch may cross midnight; each report keeps its own clock.
    if min(timestamp(entry['captured_at_utc']) for entry in entries).date().isoformat() != manifest['capture_date']:
        raise ValueError('The batch start date does not match the first capture')
    return manifest, loaded


def render_latest_verified_live_capture(st, *, root: Path, release: bool = False, final: bool = False):
    """Read-only review of a delivered live capture, without provider calls.

    ``final`` selects the original R14 provider capture; ``release`` the
    23 October capture; the default keeps the earlier October 22 capture and its
    recomputation wording.
    """
    release = release or final
    prefix = 'opt1_final_capture_' if final else 'opt1_release_capture_' if release else 'opt1_latest_capture_'
    folder = FINAL_CAPTURE_FOLDER if final else RELEASE_CAPTURE_FOLDER if release else EARLIER_CAPTURE_FOLDER
    try:
        manifest, loaded = load_latest_verified_live_capture(root, folder)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        st.error(('The recorded live assessment' if release else 'The latest verified live capture')
                 + ' cannot be verified. No result or export is displayed.')
        return
    labels = {entry['label']: case_id for case_id, (entry, _, _) in loaded.items()}
    # Streamlit removes widget state while another page is open. Keep the
    # user's example separately so returning from Research preserves it.
    if prefix + 'portfolio' not in st.session_state:
        remembered = st.session_state.get(prefix + 'remembered_portfolio')
        if remembered in labels:
            st.session_state[prefix + 'portfolio'] = remembered
    selected = st.selectbox('Portfolio', list(labels), key=prefix + 'portfolio')
    st.session_state[prefix + 'remembered_portfolio'] = selected
    entry, report, body = loaded[labels[selected]]
    st.caption('Saved example · ' + entry['captured_at_utc'][:10] +
               (' · Original R14 capture' if final else ' · Recorded provider data'))
    render_reviewer_dashboard(st, report, key_prefix=prefix + entry['id'])
    with st.popover('About this recording'):
        if final:
            st.caption('R14 provider capture · ' + entry['captured_at_utc'] +
                       '. Holdings, quotes, discovery and event P&L retain that capture time. '
                       'The current interface adds the separate ranking-sensitivity review.')
        elif release:
            st.caption('Recorded on 23 October by the r9 pipeline. Quotes retain that capture time.')
        else:
            st.caption('The chronological beta-transfer correction is applied to this recomputation.')
        st.caption('Source capture: ' + entry['captured_at_utc'] +
                   ' · Packaged/recomputed timestamp: ' + manifest['recomputed_at_utc'])
        st.dataframe(report['portfolio']['positions'], hide_index=True, width='stretch')
        st.write('Original source archive SHA-256: ' + manifest['source_archive_sha256'])
        st.write('Derived report SHA-256: ' + entry['sha256'])
        st.caption('The download preserves the recorded report. These are dated examples, not current quotes.')
    st.download_button('Download recorded analysis (JSON)', body, file_name=entry['file'],
                       mime='application/json', key=prefix + 'export')


def _fresh_discovery(st, case):
    st.markdown("#### Fresh discovery check on the same holdings")
    statuses = sorted({s["status"] for e in case["events"] for s in e["market_search"]})
    st.caption("Separate, later Claude run (" + case["evaluated_at_utc"] + ", model " + case["model"] + ") on the same captured "
               "Massive inputs with the source exposure profile. It is not the discovery behind the probabilities above. "
               "Recorded Kalshi search statuses: " + (", ".join(statuses) or "none") + ". This record does not keep the "
               "failure cause, so no network explanation is claimed; these proposals have no matched market and no probability.")
    st.dataframe([{"Proposed event": e["title"], "Window or date": e["event_date"], "Routed family": e["routed_family"],
                   "Search": ", ".join(sorted({s["status"] for s in e["market_search"]})) or "not run"}
                  for e in case["events"]], hide_index=True, width="stretch")
    for event in case["events"]:
        if event["routed_family"] in {"TAIWAN_SUPPLY", "EXPORT_CONTROLS"}:
            with st.expander(event["title"], expanded=False):
                st.write(event["exposure_rationale"])
                _named_sensitivity(st, event["named_exposure_sensitivity"])


def render_round10_assessments(st, *, root: Path):
    """Authenticated October 21 capture, recomputed offline by the current engine."""
    folder = Path(root) / "data" / "round10_assessments"
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        labels = {row["label"]: row for row in manifest["reports"]}
        selected = st.selectbox("Captured portfolio", list(labels), key="opt1_round10_portfolio")
        entry = labels[selected]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Invalid report path")
        body = (folder / entry["file"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Report hash mismatch")
        report = json.loads(body)
        refresh_entry = manifest["discovery_refresh"]
        refresh_body = (folder / refresh_entry["file"]).read_bytes()
        if hashlib.sha256(refresh_body).hexdigest() != refresh_entry["sha256"]:
            raise ValueError("Discovery refresh hash mismatch")
        refresh = json.loads(refresh_body)
    except (OSError, ValueError, TypeError, KeyError):
        st.error("The packaged assessment cannot be verified. No captured result is displayed.")
        return
    st.info("RECORDED EVIDENCE — authenticated Massive, Kalshi and Claude responses captured on " + manifest["capture_date"] +
            ", recomputed offline by this release. Quotes are those at capture time. No API call is made by opening it.")
    st.dataframe(report["portfolio"]["positions"], hide_index=True, width="stretch")
    st.metric("Captured portfolio value", _money(report["portfolio"]["notional"]))
    render_default_assessment(st, report, key_prefix="opt1_round10_" + entry["id"])
    case = next((c for c in refresh.get("cases", []) if c["run_id"] == entry["id"]), None)
    if case:
        _fresh_discovery(st, {**case, "evaluated_at_utc": refresh["evaluated_at_utc"]})


def _round11_fresh_run(st, record, receipt):
    """Fresh Round 11 stages for one captured portfolio, with every status taken from the records."""
    st.markdown("#### Round 11 fresh run on the same holdings")
    proposals = record["stages"]["proposals"]
    envelopes = (proposals.get("envelope_receipt") or {}).get("envelopes", [])
    st.caption("Fresh Claude proposals at " + str(proposals.get("proposal_clock")) + " (" + proposals["status"] + "). "
               "Holdings, prices and retained quotes keep their capture clock " +
               record["clocks"]["holdings_prices_and_retained_quotes_capture"] + ".")
    if envelopes:
        st.dataframe([{"Model": e["returned_model"], "HTTP": e["http_status"], "Validated": e["schema_result"],
                       "Input tokens": e["input_tokens"], "Output tokens": e["output_tokens"],
                       "Envelope checks": ", ".join(e["problems"]) or "request hash, usage and model consistent"}
                      for e in envelopes], hide_index=True, width="stretch")
    outcomes = {o["target_id"]: o for o in record["target_outcomes"]}
    st.dataframe([{"Proposed question": t.get("branch_question") or t["original_proposal"]["title"],
                   "Target": t["target_kind"], "Routed to": t["routing_family"],
                   "Window or date": t["window"].get("date") or t["window"].get("end"),
                   "Result": outcomes.get(t["target_id"], {}).get("status", "-"),
                   "Unresolved": ", ".join(t["unresolved_issues"]) or "-"} for t in record["targets"]],
                 hide_index=True, width="stretch")
    venues = record["stages"]["venues"]
    if venues.get("failure_classes"):
        st.warning("Kalshi and Polymarket search for the geopolitical targets: " + venues["status"] + " (" +
                   ", ".join(venues["failure_classes"]) + "). Probability stays unknown; a failed search is not evidence that no market fits.")
    macro = record["stages"]["macro"]
    st.caption("Macro targets joined to their source events by family and scheduled date: " + macro["status"] +
               (" (" + ", ".join(macro["failure_classes"]) + ")" if macro.get("failure_classes") else "") + ".")
    bound = record.get("bound_cases_v2_engine") or []
    if bound:
        st.markdown("##### Exact YES/NO contracts evaluated by the original complete-case gate")
        st.dataframe([{"Contract": b["contract_id"], "Complete bound case": b["complete_bound_case"],
                       "Q interval": _interval(b["q_interval"], probability=True),
                       "YES minus NO P&L": _money(b["delta_pnl_usd"]), "Range": _interval(b["delta_pnl_interval_usd"]),
                       "Not yet priced if YES": _money(b["yes_pnl_usd"]), "Not yet priced if NO": _money(b["no_pnl_usd"])}
                      for b in bound], hide_index=True, width="stretch")
        st.caption("YES/NO states come from the joint source distribution of the same event; the original v2 engine computes "
                   "positions and priced-in amounts unchanged. Ranges are conditional on the declared model, not forecasts.")
    unbound = [b for b in record.get("exact_bindings", []) if b["status"] != "BINDING_DERIVED_FROM_DEFAULT_SOURCE_MODEL"]
    for row in unbound:
        st.caption(row["contract_id"] + ": " + row["status"] + ". " + row.get("reason", ""))
    holdout = record.get("holdout_beta_comparison") or {}
    rows = [{"Family": family, "Instrument": symbol, "Held-out sessions": v["n_holdout"],
             "Announcement beta": f"{v['event_day_beta']:.3f}", "Daily beta": f"{v['daily_beta']:.3f}",
             "Slope RMSE announcement": f"{v['slope_rmse']['event_day_slope']:.5f}",
             "Slope RMSE daily": f"{v['slope_rmse']['daily_slope']:.5f}", "Lower error": v["lower_slope_error"]}
            for family, report in holdout.items() for symbol, v in (report.get("instruments") or {}).items()]
    if rows:
        st.markdown("##### Announcement-day versus daily beta on the same held-out sessions")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(next(iter(holdout.values())).get("scope", ""))
    with st.expander("Round 11 records for this portfolio", expanded=False):
        st.json({"stages": record["stages"], "exact_bindings": record.get("exact_bindings"),
                 "fund_compositions": record.get("fund_compositions")})



def _round12_treasury_term_review(st, *, root: Path, portfolio):
    """Separately dated official-term review, not a rewrite of the captured holdings."""
    symbols = {position.get("symbol") for position in portfolio.get("positions", [])}
    relevant = symbols.intersection({"TLT", "IEF"})
    if not relevant:
        return
    folder = Path(root) / "data" / "round12_primary_treasury"
    try:
        from .primary_treasury_transcription import read_checked_transcription
        terms, evidence = read_checked_transcription(folder / "source_transcription.json")
        receipt_bytes = (folder / "recovery_receipt.json").read_bytes()
        expected = json.loads((Path(root) / "FILE_MANIFEST.json").read_text(encoding="utf-8"))["files"]
        relative = "data/round12_primary_treasury/recovery_receipt.json"
        if hashlib.sha256(receipt_bytes).hexdigest() != expected[relative]["sha256"]:
            raise ValueError("The Round 12 recovery receipt has changed")
        receipt = json.loads(receipt_bytes)
        if (receipt["source_review"]["transcription_sha256"] != evidence["transcription_sha256"]
                or receipt["status"] != "EIGHT_SOURCE_TERMS_REVIEWED_DURATION_ROWS_RECALCULATED"
                or len(terms) != 8):
            raise ValueError("The source review and recovery no longer reconcile")
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("The supplementary Treasury review cannot be verified and is not displayed.")
        return
    st.markdown("#### Supplementary official Treasury terms (separately reviewed)")
    st.caption("Static coupon/maturity terms independently transcribed from the US Treasury's October 21, 2026 "
               "FedInvest table. This is not a new Massive or Fiscal Data response and does not alter historical "
               "capture receipts or use subsequent TreasuryDirect prices.")
    rows = [r for r in receipt["rows"] if r["fund"] in relevant]
    st.dataframe([{"Fund": r["fund"], "CUSIP": r["cusip"], "Fund NAV weight": f"{r['nav_weight']:.3%}",
                   "Coupon": f"{r['terms']['coupon_percent']:.3f}%", "Maturity": r["terms"]["maturity"],
                   "Modified duration (model)": round(r["duration_result"]["modified_duration"], 3)}
                  for r in rows], hide_index=True, width="stretch")
    with st.expander("Primary source, limitation and review receipt", expanded=False):
        st.write(evidence["source_url"])
        st.write("Only these eight formerly uncovered Treasury rows are newly supported. Other unclassified "
                 "fund mass and total portfolio risk remain unresolved where the evidence is incomplete.")
        st.json({"source_review": receipt["source_review"],
                 "fund_totals": {k: v for k, v in receipt["fund_totals"].items() if k in relevant}})




def render_reviewer_answer_sheet(st, *, report, original=None, taiwan=None):
    """Make the original assessment's required decisions visible before appendices."""
    from .reviewer_decision import assemble_reviewer_answer_sheet

    rows = assemble_reviewer_answer_sheet(report, original=original, taiwan=taiwan)
    st.markdown("### Assessment answers — portfolio, event, probability, impact")
    st.caption("This is a compact index of separately dated verified sources, NOT a new synchronized "
               "quote snapshot or an additional risk model. Each row describes its own event/settlement. "
               "Market Q is a pricing weight/price proxy, not a calibrated physical probability P.")
    st.dataframe([{k: v for k, v in row.items() if not k.startswith('_')} for row in rows],
                 hide_index=True, width="stretch")
    st.caption("UNKNOWN is intentional: a nearby contract does not inherit the broader event's probability; "
               "a partial issuer channel is not full portfolio P&L; modeled announcement components are "
               "not forecasts of total return. Never sum overlapping event rows.")
    if original is not None:
        matching = [e for e in original['default_analysis']['events']
                    if e['family'] in {'TAIWAN_SUPPLY', 'EXPORT_CONTROLS'}]
        if matching:
            with st.expander("Challenge a prediction-market match: recorded rejected alternatives", expanded=False):
                audits = []
                for event in matching:
                    search = event.get('market_search') or {}
                    candidates = search.get('candidates') or []
                    for venue in ('Kalshi', 'Polymarket'):
                        related = next((c for c in candidates if c.get('venue', '').lower() == venue.lower()), None)
                        if related is None:
                            continue
                        audits.append({
                            'Event we care about': event['title'],
                            'Venue': venue,
                            'Actual contract question': related['market_question'],
                            'Decision': related['decision'],
                            'Why it does not match': related['difference'],
                        })
                if audits:
                    st.dataframe(audits, hide_index=True, width="stretch")
                st.caption("These rejection reasons come from the original, separately dated two-venue capture. "
                           "No rejected contract price constrains the broad event. Full source rules and "
                           "provider receipt hashes are retained in Captured portfolio assessments.")


def render_round11_assessments(st, *, root: Path):
    """October 21 capture recomputed by this release, plus the Round 11 fresh stages and their receipt."""
    folder = Path(root) / "data" / "round11_assessments"
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        bodies = {}
        for entry in manifest["files"]:
            if Path(entry["file"]).name != entry["file"]:
                raise ValueError("Invalid report path")
            body = (folder / entry["file"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != entry["sha256"]:
                raise ValueError("Report hash mismatch")
            bodies[entry["file"]] = json.loads(body)
        receipt = bodies["enhanced_path_receipt.json"]
        records = {r["run_id"]: r for r in bodies["pipeline_records.json"]["records"]}
        labels = {row["label"]: row for row in manifest["reports"]}
        selected = st.selectbox("Captured portfolio", list(labels), key="opt1_round11_portfolio")
        entry = labels[selected]
        report = bodies[entry["file"]]
        record = records[entry["id"]]
    except (OSError, ValueError, TypeError, KeyError):
        st.error("The packaged assessment cannot be verified. No captured result is displayed.")
        return
    # Do not conflate the authenticated older source capture with a later,
    # unsuccessful provider refresh. A recorded quote is NOT a live quote.
    st.info("HISTORICAL EVIDENCE — authenticated Massive, Kalshi and Claude capture: " +
            manifest["capture_date"] + ". Recomputed offline with exact YES/NO bindings. " +
            "Opening this view makes no provider call and does not refresh prices.")
    access = receipt.get("access") or []
    failures = [a for a in access if a.get("status") != "SUCCEEDED"]
    if failures:
        at = str(failures[0].get("at") or "")
        when = at[:19].replace("T", " ") + " UTC" if len(at) >= 19 else "a separately dated time"
        blocked = ", ".join(sorted({str(a.get("provider") or "unknown provider") for a in failures}))
        st.caption("SEPARATE REFRESH ATTEMPT — " + when + ": " + blocked +
                   " could not be refreshed in the restricted cloud environment. " +
                   "The older authenticated evidence above remains historical; provider failures " +
                   "do NOT show that no matching market exists. See technical audit below.")
    positions = report["portfolio"]["positions"]
    st.caption("Reference portfolio · " + _money(report["portfolio"]["notional"]) + " · " +
               " / ".join(p["symbol"] + " " + f"{p['weight']:.0%}" for p in positions) +
               ". This archived example is independent of the Portfolio tab allocation.")
    # Show only independently checked cross-capture references; never transfer
    # their quotes into the separate macro model or the broad event probability.
    original = None
    taiwan = None
    cross_capture_issue = None
    try:
        from .reviewer_decision import read_separately_dated_matching_record, _portfolio_key
        original = read_separately_dated_matching_record(root, selected, report['portfolio'])
        if _portfolio_key(report['portfolio']) == (100000.0,
                (('NVDA', 'stock', .35), ('QQQ', 'etf', .45), ('TLT', 'etf', .2))):
            from .round17_conditional_ui import verified_record
            taiwan = verified_record(root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        cross_capture_issue = type(exc).__name__
    try:
        render_reviewer_answer_sheet(st, report=report, original=original, taiwan=taiwan)
    except (ValueError, OSError, KeyError, TypeError):
        st.warning("A source-consistent reviewer summary could not be verified; use the detailed "
                   "primary report below. No cross-capture probability was admitted.")
    if cross_capture_issue:
        st.warning("Supplementary separately dated evidence unavailable or inconsistent "
                   "(" + cross_capture_issue + "). Its results were not imported.")
    # Newer positive exact-settlement case is a separate *dated*, derived-only
    # audit. It must never borrow the older source capture's quote or claim a
    # fresh investigation / full authenticated acceptance.
    from .round22_recorded_case import render_recorded_case
    render_recorded_case(st, root=root, portfolio=report['portfolio'])
    # Keep the original portfolio and provider receipts visible and auditable,
    # but do not force reviewers to scroll past raw engineering diagnostics
    # before reaching the assessment's event/probability/impact answers.
    with st.expander("Captured positions and portfolio valuation", expanded=False):
        st.dataframe(positions, hide_index=True, width="stretch")
        st.metric("Captured portfolio value", _money(report["portfolio"]["notional"]))
    with st.expander("Later provider refresh attempt — technical audit (NOT source capture)", expanded=False):
        st.caption("These are separate later access checks. Their outcome does not invalidate "
                   "the earlier authenticated records and does not establish current availability.")
        if access:
            st.dataframe([{"Provider": a["provider"], "Attempt (UTC)": a.get("at", "UNKNOWN"),
                           "Refresh result": a["status"],
                           "Failure classification": a.get("failure_class") or "None"}
                          for a in access], hide_index=True, width="stretch")
        else:
            st.caption("No separately recorded provider-access check.")
        st.caption("Enhanced-refresh status: " + receipt["status"] +
                   ". The dated full receipts remain in the packaged audit.")
    render_default_assessment(st, report, key_prefix="opt1_round11_" + entry["id"])
    _round12_treasury_term_review(st, root=root, portfolio=report["portfolio"])
    _round11_fresh_run(st, record, receipt)
    # Recorded historical conditional stress applies only to the reference tech portfolio.
    # It is assumption-based and is not injected into the live event acceptance engine.
    symbols={p.get('symbol') for p in report['portfolio']['positions']}
    if symbols=={'NVDA','QQQ','TLT'}:
        from .round17_conditional_ui import render_recorded_taiwan_stress
        render_recorded_taiwan_stress(st, root=root)
