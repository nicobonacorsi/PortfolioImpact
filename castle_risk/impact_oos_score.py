"""Forward-only price-return evaluation of event-conditional portfolio estimates.

This instrument DOES NOT identify an event's causal return effect: only one of
the YES/NO potential outcomes can be observed. It scores the prespecified
prediction for the branch that actually occurred against a later fixed-horizon
portfolio price return, plus a no-change benchmark. It does not score the
unobservable YES-minus-NO counterfactual gap.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, date, timezone
from math import fsum, isfinite, sqrt
from random import Random

LOCK_SCHEMA = "castle.prospective_impact_lock.v1"
SCORE_SCHEMA = "castle.prospective_impact_oos_score.v1"


class ImpactOosInputError(ValueError):
    """A prospective portfolio-return record is incomplete or leaky."""


def _dt(x, label):
    if not isinstance(x, str):
        raise ImpactOosInputError(label + " must be ISO timestamp.")
    try:
        d = datetime.fromisoformat(x.replace("Z", "+00:00"))
    except ValueError:
        raise ImpactOosInputError(label + " invalid timestamp.") from None
    if d.tzinfo is None or d.utcoffset() is None:
        raise ImpactOosInputError(label + " needs timezone.")
    return d


def _date(x, label):
    if not isinstance(x, str):
        raise ImpactOosInputError(label + " must be a date.")
    try:
        return date.fromisoformat(x)
    except ValueError:
        raise ImpactOosInputError(label + " invalid date.") from None


def _finite(x, label):
    if type(x) not in (float, int) or not isfinite(x):
        raise ImpactOosInputError(label + " must be finite number.")
    return float(x)


def score_forward_portfolio_impacts(lock: object, outcomes: object, *,
                                    minimum_independent_event_families: int = 20,
                                    bootstrap_draws: int = 2000,
                                    audit_time_utc: str | None = None,
                                    seed: int = 119) -> dict:
    """Pre-registered fixed-endpoint YES/NO price-return OOS evaluation.

    Position weights and opening split-adjusted close must be locked before
    the event. A realized close exists for every original noncash holding on
    the fixed future endpoint. Cash has exactly zero asset return. The solver
    chooses the precommitted YES or NO return vector only AFTER adjudication.
    """
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
        raise ImpactOosInputError("Expected prospective impact lock schema.")
    events = lock.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 10000:
        raise ImpactOosInputError("Expected 1-10000 locked cases.")
    if not isinstance(outcomes, list) or len(outcomes) > len(events):
        raise ImpactOosInputError("Outcomes must be a bounded list.")
    if (type(minimum_independent_event_families) is not int or
            minimum_independent_event_families < 2 or
            type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 10000):
        raise ImpactOosInputError("Invalid scoring gates.")
    now = _dt(audit_time_utc, "audit_time_utc") if audit_time_utc else datetime.now(timezone.utc)
    lock_time = _dt(lock.get("locked_at_utc"), "locked_at_utc")
    if lock_time > now:
        raise ImpactOosInputError("Lock is in future relative to audit.")
    entries, model_versions = {}, set()
    for e in events:
        if not isinstance(e, dict):
            raise ImpactOosInputError("Impact lock event must be object.")
        ident, target, family = e.get("case_id"), e.get("event_condition"), e.get("event_family_id")
        if (not isinstance(ident, str) or not ident or ident in entries
                or not isinstance(target, str) or len(target) < 10
                or not isinstance(family, str) or not family):
            raise ImpactOosInputError("Unique exact event, case ID and event family required.")
        entries[ident] = e
        status = e.get("status")
        if status not in ("MODEL_ASSUMPTION_UNCALIBRATED", "ABSTAIN"):
            raise ImpactOosInputError("Unknown conditional impact status.")
        version = e.get("model_version")
        if not isinstance(version, str) or not version:
            raise ImpactOosInputError("Model prompt/version must be frozen.")
        model_versions.add(version)
        modeled_at = _dt(e.get("model_response_received_at_utc"),
                         "model_response_received_at_utc")
        mark_at = _dt(e.get("opening_mark_recorded_at_utc"), "opening_mark_recorded_at_utc")
        fixed_end = _date(e.get("fixed_price_return_end_session_date"),
                          "fixed_price_return_end_session_date")
        if not mark_at <= modeled_at <= lock_time or fixed_end <= lock_time.date():
            raise ImpactOosInputError("Opening mark/response/lock/end chronology invalid.")
        nav = _finite(e.get("original_notional_usd"), "original_notional_usd")
        if nav <= 0:
            raise ImpactOosInputError("Original NAV must be positive.")
        rows = e.get("positions")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 30:
            raise ImpactOosInputError("All original positions required.")
        symbols = []
        weights = []
        for p in rows:
            if not isinstance(p, dict) or not isinstance(p.get("symbol"), str):
                raise ImpactOosInputError("Invalid original position.")
            symbol = p["symbol"]
            if not symbol or symbol in symbols:
                raise ImpactOosInputError("Duplicate or empty original holding.")
            symbols.append(symbol)
            weight = _finite(p.get("weight"), "funded original weight")
            if not 0 < weight <= 1:
                raise ImpactOosInputError("Original portfolio weights must be positive.")
            weights.append(weight)
            start = _finite(p.get("opening_split_adjusted_close"), "opening price")
            if symbol == "USD":
                if start != 1:
                    raise ImpactOosInputError("Nominal cash opening mark must equal one.")
            elif start <= 0:
                raise ImpactOosInputError("Noncash opening close must be positive.")
            if status == "MODEL_ASSUMPTION_UNCALIBRATED":
                for k in ("predicted_yes_return_pct", "predicted_no_return_pct"):
                    prediction = _finite(p.get(k), k)
                    if symbol == "USD" and prediction != 0.0:
                        raise ImpactOosInputError("No imaginary USD cash return allowed.")
                    if not -100 <= prediction <= 200:
                        raise ImpactOosInputError("Return exceeds declared stress safety limits.")
        if abs(fsum(weights)-1) > 1e-8:
            raise ImpactOosInputError("Original weights must sum to 1 exactly.")
    if len(model_versions) != 1:
        raise ImpactOosInputError("Different impact model versions cannot be pooled.")
    resolved = {}
    for row in outcomes:
        if not isinstance(row, dict) or row.get("case_id") not in entries:
            raise ImpactOosInputError("Unexpected impact outcome.")
        ident = row["case_id"]
        if ident in resolved:
            raise ImpactOosInputError("Duplicate impact outcome.")
        if row.get("event_condition") != entries[ident]["event_condition"]:
            raise ImpactOosInputError("Event condition substituted after lock.")
        y = row.get("outcome")
        if type(y) is not int or y not in (0,1):
            raise ImpactOosInputError("Outcome must be 0 or 1 integer.")
        if row.get("resolution_source_kind") not in (
                "MARKET_SETTLEMENT","OFFICIAL_EVENT_ADJUDICATION"):
            raise ImpactOosInputError("Independent resolution source required.")
        if not isinstance(row.get("resolution_source_url"), str) or not row["resolution_source_url"].startswith("https://"):
            raise ImpactOosInputError("Resolution reference required.")
        settled = _dt(row.get("resolved_at_utc"), "resolved_at_utc")
        observed_at = _dt(row.get("end_prices_observed_at_utc"), "end_prices_observed_at_utc")
        fixed_end = _date(entries[ident]["fixed_price_return_end_session_date"],"fixed date")
        actual_date = _date(row.get("end_close_session_date"),"observed close date")
        if (not lock_time < settled <= now or not settled <= observed_at <= now
                or actual_date != fixed_end or observed_at.date() < fixed_end):
            raise ImpactOosInputError("Adjudication or fixed end-price chronology invalid.")
        marks = row.get("observed_closing_prices")
        if not isinstance(marks, list):
            raise ImpactOosInputError("End price marks required.")
        keys = [p.get("symbol") for p in marks if isinstance(p,dict)]
        expected = [p["symbol"] for p in entries[ident]["positions"] if p["symbol"]!="USD"]
        if len(marks) != len(expected) or len(keys)!=len(expected) or set(keys)!=set(expected):
            raise ImpactOosInputError("All and only original noncash holdings must be priced.")
        if not isinstance(row.get("price_data_source"), str) or not row["price_data_source"]:
            raise ImpactOosInputError("Provider price provenance required.")
        for price in marks:
            if _finite(price.get("split_adjusted_close"), "realized close") <= 0:
                raise ImpactOosInputError("Missing or invalid end close.")
        resolved[ident] = row
    scored = []
    for ident, e in entries.items():
        if ident not in resolved or e["status"]=="ABSTAIN":
            continue
        actual = resolved[ident]
        marks = {p["symbol"]:p["split_adjusted_close"]
                 for p in actual["observed_closing_prices"]}
        y = actual["outcome"]
        nav = e["original_notional_usd"]
        predictions, realized = [], []
        for p in e["positions"]:
            symbol = p["symbol"]
            start = p["opening_split_adjusted_close"]
            end = 1.0 if symbol=="USD" else marks[symbol]
            predictions.append(p["weight"]*p["predicted_yes_return_pct" if y else
                                            "predicted_no_return_pct"]/100)
            realized.append(p["weight"]*(end/start-1))
        predicted_pct = 100*fsum(predictions)
        observed_pct = 100*fsum(realized)
        err_pct = predicted_pct-observed_pct
        scored.append({
            "case_id":ident,"event_family_id":e["event_family_id"],
            "realized_event_state":y,
            "predicted_price_return_pct":predicted_pct,
            "observed_price_return_pct":observed_pct,
            "predicted_pnl_usd":nav*predicted_pct/100,
            "observed_pnl_usd":nav*observed_pct/100,
            "absolute_error_pct_nav":abs(err_pct),
            "squared_error_pct_nav2":err_pct**2,
            "zero_change_baseline_abs_error_pct_nav":abs(observed_pct),
            "zero_change_baseline_squared_error_pct_nav2":observed_pct**2,
        })
    metrics = {
        "model_mae_pct_nav":(fsum(v["absolute_error_pct_nav"] for v in scored)/len(scored)
                             if scored else None),
        "zero_change_mae_pct_nav":(fsum(v["zero_change_baseline_abs_error_pct_nav"]
                                     for v in scored)/len(scored) if scored else None),
        "model_rmse_pct_nav":(sqrt(fsum(v["squared_error_pct_nav2"] for v in scored)/len(scored))
                              if scored else None),
        "zero_change_rmse_pct_nav":(sqrt(fsum(v["zero_change_baseline_squared_error_pct_nav2"]
                                      for v in scored)/len(scored)) if scored else None),
    }
    by_family = defaultdict(list)
    for row in scored:
        by_family[row["event_family_id"]].append(
            row["zero_change_baseline_abs_error_pct_nav"]-row["absolute_error_pct_nav"])
    family_means = [fsum(vals)/len(vals) for _,vals in sorted(by_family.items())]
    ci = None
    if len(family_means)>=10:
        rng = Random(seed)
        draws=[]
        for _ in range(bootstrap_draws):
            draws.append(fsum(family_means[rng.randrange(len(family_means))]
                              for _ in family_means)/len(family_means))
        draws.sort()
        ci=[draws[int(.025*len(draws))],draws[min(len(draws)-1,int(.975*len(draws)))]]
    return {
        "schema":SCORE_SCHEMA,
        "status":("PENDING_NO_RESOLVED_FIXED_HORIZONS" if not resolved
                  else "INSUFFICIENT_INDEPENDENT_EVENT_FAMILIES"
                  if len(by_family)<minimum_independent_event_families
                  else "DESCRIPTIVE_PREREGISTERED_ERRORS_REQUIRING_EXTERNAL_AUDIT"),
        "model_version":next(iter(model_versions)),
        "locked_cases":len(events),"resolved_cases":len(resolved),
        "scored_cases":len(scored),"scored_independent_event_families":len(by_family),
        "resolved_abstentions":len(resolved)-len(scored),
        **metrics,
        "cluster_bootstrap_95_percent_mae_improvement_interval":ci,
        "price_returns_exclude_dividends":True,
        "unobserved_causal_counterfactual_identified":False,
        "event_yes_minus_no_causal_effect_validated":False,
        "external_adjudication_and_provider_price_authenticity_certified":False,
        "realized_branch_only_is_scored":True,
        "score_rows":scored,
    }
