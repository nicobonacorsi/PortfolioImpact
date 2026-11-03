"""Dated factor sensitivities and named-holding components, separate from event P&L.

This layer does not estimate how an event moves factors. It exposes calculations
that remain meaningful without a prediction-market contract. Its numeric output
never increases event-conditional forecast coverage. Existing outcome models and
their admission decisions are not changed.

The pre-implementation policy is data/r103_scenario_protocol.json. The Federal
Reserve's scenario-design distinction is relevant context, not a validation of
this implementation: https://www.federalreserve.gov/publications/2026-stress-test-scenarios.htm
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
import math
from statistics import stdev

from .portfolio import Portfolio, parse_portfolio
from .scenario_proxy_response import FAMILY_PROXIES, PROXY_SOURCES, build_proxy_models

SCHEMA = "castle.scenario_response.v1"
MODEL_SCHEMA = "castle.scenario_response_model.v1"
_MACRO = {
    "FED_POLICY": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "FOMC": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "US_INFLATION": "https://www.bls.gov/cpi/",
    "CPI": "https://www.bls.gov/cpi/",
    "US_LABOR": "https://www.bls.gov/ces/",
    "US_PAYROLLS": "https://www.bls.gov/ces/",
    "PAYROLLS": "https://www.bls.gov/ces/",
    "LABOR_MARKET": "https://www.bls.gov/ces/",
}
_FACTORS = {
    "SPY_RETURN_PCT": ("SPY price", "percent"),
    "US_10Y_YIELD_CHANGE_BP": ("10-year Treasury yield", "basis_points"),
}
_FACTORS.update({symbol + "_RETURN_PCT": (info["label"], "percent")
                 for symbol, info in PROXY_SOURCES.items()})


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _day(value):
    if not isinstance(value, str):
        raise ValueError("A dated observation is required")
    return date.fromisoformat(value[:10]).isoformat()


def _source_metadata(provenance):
    source = {k: deepcopy(provenance[k]) for k in
              ("source_url", "source_response_sha256", "retrieved_at", "adjustment", "scope", "series")
              if k in provenance}
    source["receipts"] = [{k: receipt[k] for k in
                          ("url", "sha256", "source_url", "source_response_sha256", "retrieved_at", "fixture", "provider")
                          if k in receipt}
                          for receipt in provenance.get("receipts") or [] if isinstance(receipt, Mapping)]
    return source


def _observations(rows, key, through):
    """Do not bridge a missing observation silently or accept duplicate dates."""
    values = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed source row")
        day = _day(row.get("date"))
        value = row.get(key)
        if day >= through:
            continue
        if day in values or not _finite(value) or (key == "close" and value <= 0):
            raise ValueError("Duplicate or invalid source observation")
        values[day] = float(value)
    return values


def _training_scales(prices, yields, train_end, through):
    """Scales use the training portion only, never the admission period."""
    output = {}
    for factor, rows, key in (("SPY_RETURN_PCT", prices, "close"),
                              ("US_10Y_YIELD_CHANGE_BP", yields, "yield_pct")):
        try:
            values = _observations(rows, key, through)
            days = sorted(d for d in values if d <= train_end)
            changes = []
            for a, b in zip(days, days[1:]):
                # More than one normal holiday weekend is a data gap.
                if (date.fromisoformat(b) - date.fromisoformat(a)).days > 4:
                    continue
                change = ((values[b] / values[a] - 1) * 100 if key == "close"
                          else (values[b] - values[a]) * 100)
                if not math.isfinite(change):
                    raise ValueError("Nonfinite training interval")
                changes.append(change)
            if len(changes) < 250:
                continue
            scale = stdev(changes)
            if not math.isfinite(scale) or scale <= 0:
                continue
            output[factor] = {
                "value": scale, "unit": _FACTORS[factor][1],
                "basis": "ONE_TRAINING_DAILY_STANDARD_DEVIATION_NOT_EVENT_SEVERITY",
                "training_span": [days[0], days[-1]], "observations": len(changes),
                "probability_interval": False,
            }
        except (ValueError, TypeError, KeyError):
            continue
    return output


def build_response_model(portfolio: Portfolio, *, prices_by_symbol, treasury_yields,
                         two_factor_responses, evaluated_at, train_end,
                         price_sources=None, benchmark="SPY", proxy_prices=None,
                         proxy_sources=None, release_dates=(), treasury_source=None):
    """Retain admitted original-position responses with a strict as-of check.

    The existing two-factor admission rule is reused, not optimized on R103.
    The slopes describe contemporaneous co-movement, not causal responses.
    """
    if not isinstance(portfolio, Portfolio):
        raise ValueError("An original portfolio is required")
    now = evaluated_at if isinstance(evaluated_at, datetime) else datetime.fromisoformat(evaluated_at)
    if now.utcoffset() is None:
        raise ValueError("A timezone-aware analysis time is required")
    through = now.date().isoformat()
    split = _day(train_end)
    if split >= through:
        raise ValueError("Calibration must precede analysis")
    if benchmark != "SPY":
        return {"schema": MODEL_SCHEMA, "status": "UNSUPPORTED_BENCHMARK", "positions": [], "scales": {}}
    sources = price_sources or {}
    positions = []
    for p in portfolio.positions:
        row = {"symbol": p.symbol, "weight": p.weight, "asset_type": p.asset_type,
               "status": "UNAVAILABLE", "reason": "NO_ADMITTED_MEASURED_FACTOR_RESPONSE"}
        if p.asset_type == "cash":
            row.update(status="AVAILABLE", coefficients={"SPY_RETURN_PCT": 0., "US_10Y_YIELD_CHANGE_BP": 0.},
                       basis="USD_CASH_PRICE_HELD_FIXED_NO_CARRY", reason=None)
        elif p.symbol == benchmark and (prices_by_symbol or {}).get(benchmark):
            row.update(status="AVAILABLE", coefficients={"SPY_RETURN_PCT": .01, "US_10Y_YIELD_CHANGE_BP": 0.},
                       basis="EXACT_SAME_INSTRUMENT_PRICE_IDENTITY", reason=None)
        else:
            fit = (two_factor_responses or {}).get(p.symbol) or {}
            spans = [fit.get("training_span"), fit.get("holdout_span")]
            valid = (fit.get("admitted") is True and fit.get("status") == "ADMITTED_MEASURED_TWO_FACTOR"
                     and _finite(fit.get("spy_loading")) and _finite(fit.get("rate_loading_per_bp")))
            try:
                valid = valid and all(isinstance(s, (list, tuple)) and len(s) == 2 for s in spans)
                valid = valid and (_day(spans[0][0]) <= _day(spans[0][1]) <= split
                                   < _day(spans[1][0]) <= _day(spans[1][1]) < through)
            except (TypeError, ValueError, IndexError):
                valid = False
            if valid:
                row.update(status="AVAILABLE", reason=None,
                           coefficients={"SPY_RETURN_PCT": fit["spy_loading"] / 100,
                                         "US_10Y_YIELD_CHANGE_BP": fit["rate_loading_per_bp"]},
                           basis="ADMITTED_HISTORICAL_TWO_FACTOR_ASSOCIATION",
                           training_span=deepcopy(spans[0]), holdout_span=deepcopy(spans[1]),
                           holdout_rmse=fit.get("holdout_rmse"),
                           holdout_zero_rmse=fit.get("holdout_zero_rmse"),
                           holdout_oos_r2=fit.get("holdout_oos_r2"))
            else:
                row["reason"] = ("ADMITTED_RESPONSE_HAS_INVALID_TIME_OR_COEFFICIENTS" if fit.get("admitted")
                                 else str(fit.get("status") or "NO_MEASURED_FACTOR_RESPONSE"))
        provenance = sources.get(p.symbol) or {}
        row["source"] = _source_metadata(provenance)
        positions.append(row)
    proxy_models = build_proxy_models(portfolio, prices_by_symbol, proxy_prices or {}, treasury_yields,
        train_end=split, evaluated_at=now, release_dates=release_dates)
    return {
        "schema": MODEL_SCHEMA, "status": "COMPUTED", "as_of": now.isoformat(),
        "notional": portfolio.notional, "positions": positions,
        "scales": _training_scales((prices_by_symbol or {}).get(benchmark) or [],
                                   treasury_yields, split, through),
        "proxy_models": proxy_models,
        "proxy_sources": {symbol: {"issuer": PROXY_SOURCES[symbol],
            **_source_metadata((sources.get(symbol) or {}) if (prices_by_symbol or {}).get(symbol)
                               else ((proxy_sources or {}).get(symbol) or {}))}
            for symbol in proxy_models},
        "factor_sources": {"SPY": _source_metadata(sources.get("SPY") or {}),
                           "US10Y": _source_metadata(treasury_source or {})},
        "horizon": "One trading-day factor change; no carry or rebalancing.",
        "calibration_policy": "Existing R85 admission; fit before the final 365 calendar days, no holdout refit.",
        "factor_source_scope": "SPY split-adjusted closes and U.S. Treasury 10-year yield; contemporaneous association.",
        "scope": "Factor component only; idiosyncratic returns and event-to-factor transmission remain unmodeled.",
    }


def _factor_scenario(model, factor, value, *, scale=None):
    if factor not in _FACTORS or not _finite(value):
        raise ValueError("A supported factor and finite shock are required")
    if factor.endswith("_RETURN_PCT") and value < -100:
        raise ValueError("A long-only price cannot fall by more than 100 percent")
    contributions = []
    total = model["notional"]
    for row in model["positions"]:
        coefficient = (row.get("coefficients") or {}).get(factor)
        known = row.get("status") == "AVAILABLE" and _finite(coefficient)
        ret = coefficient * value if known else None
        # Do not let a linear approximation imply negative asset values.
        if ret is not None and (not math.isfinite(ret) or ret < -1
                                or not math.isfinite(total * row["weight"] * ret)):
            known, ret = False, None
        contributions.append({"symbol": row["symbol"], "weight": row["weight"],
                              "response_weight": row["weight"] if known else 0.,
                              "pnl": total * row["weight"] * ret if known else None,
                              "modeled_return_fraction": ret,
                              "basis": row.get("basis") if known else row.get("reason") or "LINEAR_DOMAIN_EXCEEDED"})
    covered = math.fsum(r["response_weight"] for r in contributions)
    try:
        known_pnl = math.fsum(r["pnl"] for r in contributions if r["pnl"] is not None)
    except OverflowError as exc:
        raise ValueError("The declared scenario aggregate is not numerically representable") from exc
    complete = all(r["pnl"] is not None or r["weight"] == 0 for r in contributions)
    label, unit = _FACTORS[factor]
    return {
        "id": factor.lower() + ("-up" if value >= 0 else "-down"),
        "label": f"{label} {'+' if value >= 0 else ''}{value:.3g} {'bp' if unit == 'basis_points' else '%'}",
        "shock": {"factor": factor, "value": value, "unit": unit,
                  "basis": (scale or {}).get("basis", "EXPLICIT_CALLER_ASSUMPTION"),
                  "training_span": (scale or {}).get("training_span")},
        "pnl": known_pnl if complete else None, "known_pnl": known_pnl,
        "covered_weight": covered, "unmodeled_weight": max(0., 1 - covered),
        "position_contributions": contributions, "probability": None,
        "scope": "FACTOR_COMPONENT_ONLY", "event_effect_estimated": False,
        "scope_note": "Factor response with the other factor fixed. Company-specific moves are excluded.",
        "horizon": model["horizon"],
    }


def _named_source_exposure(risk):
    named = {}
    for row in risk.get("exposure") or []:
        if not isinstance(row, Mapping) or not row.get("symbol") or not _finite(row.get("weight")) or not 0 <= row["weight"] <= 1:
            return {}, [{"reason": "INVALID_NAMED_SOURCE_EXPOSURE"}]
        symbol = str(row["symbol"])
        if symbol in named:
            return {}, [{"symbol": symbol, "reason": "DUPLICATE_NAMED_SOURCE_EXPOSURE"}]
        named[symbol] = row["weight"]
    return {symbol: weight for symbol, weight in named.items() if weight > 0}, []


def _proxy_scenarios(risk, model, portfolio, lookthrough=()):
    """Prespecified event-family hypothesis; a named portfolio link is required."""
    if risk.get("exposure_basis") != "NAMED_SOURCE_CHANNELS":
        return [], {}, []
    # Reuse original-NAV reconciliation before any measured factor response is
    # displayed for this event. A plausible family and an absent ticker are
    # not a documented link to this portfolio.
    linked_components, failures = _linked_components(risk, portfolio, lookthrough)
    if not linked_components:
        return [], {}, failures
    named = sorted(linked_components[0]["shock"]["symbols"])
    if not named:
        return [], {}, []
    scenarios, evidence, missing = [], {}, list(failures)
    for symbol in FAMILY_PROXIES.get(risk.get("family"), ()):
        source = PROXY_SOURCES[symbol]
        fitted = (model.get("proxy_models") or {}).get(symbol) or {}
        scale = fitted.get("scale_pct")
        evidence[symbol] = deepcopy(fitted)
        if not _finite(scale) or scale <= 0:
            missing.append({"proxy": symbol, "reason": "PROXY_TRAINING_SCALE_UNAVAILABLE"})
            continue
        factor = symbol + "_RETURN_PCT"
        positions = []
        for p in portfolio.positions:
            fit = (fitted.get("positions") or {}).get(p.symbol) or {}
            valid = fit.get("admitted") is True and _finite(fit.get("proxy_loading"))
            positions.append({"symbol": p.symbol, "weight": p.weight, "asset_type": p.asset_type,
                "status": "AVAILABLE" if valid else "UNAVAILABLE",
                "coefficients": {factor: fit["proxy_loading"]/100} if valid else {},
                "basis": fit.get("basis"), "reason": str(fit.get("status") or "NO_PROXY_RESPONSE")})
            if not valid:
                missing.append({"symbol": p.symbol, "proxy": symbol, "reason": positions[-1]["reason"]})
        if not any(r["status"] == "AVAILABLE" and r["weight"] > 0
                   and r["coefficients"].get(factor, 0) != 0 for r in positions):
            continue
        local_model = {"notional": portfolio.notional, "positions": positions,
                       "horizon": "One trading-day proxy-price change with SPY and 10-year Treasury yield held fixed."}
        scale_info = {"basis": fitted["scale_basis"], "training_span": fitted.get("training_span")}
        for sign in (-1, 1):
            try:
                row = _factor_scenario(local_model, factor, sign*scale, scale=scale_info)
            except ValueError:
                missing.append({"proxy": symbol, "reason": "SCENARIO_NUMERIC_DOMAIN_EXCEEDED"})
                continue
            row["shock"].update(proxy_symbol=symbol, source_url=source["url"], proxy_scope=source["scope"])
            row["scope_note"] = "Proxy-price response with SPY and Treasury yields fixed; other residual returns are excluded."
            scenarios.append(row)
    linkage = {"status": "DOCUMENTED_EVENT_PROXY_HYPOTHESIS", "event_to_factor_identified": False,
        "basis": "Prespecified fund-price proxy for the event family, with documented named portfolio exposure.",
        "exposure_basis": risk.get("exposure_basis"), "named_source_symbols": named,
        "proxy_sources": {s: PROXY_SOURCES[s] for s in FAMILY_PROXIES.get(risk.get("family"), ())},
        "source_url": (next(iter(PROXY_SOURCES[s]["url"] for s in FAMILY_PROXIES.get(risk.get("family"), ())), None))}
    return scenarios, {"linkage": linkage, "fits": evidence}, missing


def evaluate_factor_shock(model, factor, value):
    """Evaluate a supplied scenario, independently of event odds and severity."""
    if not isinstance(model, Mapping) or model.get("schema") != MODEL_SCHEMA or model.get("status") != "COMPUTED":
        raise ValueError("A dated scenario response model is required")
    return _factor_scenario(model, factor, value)


def _linked_components(risk, portfolio, lookthrough):
    """Named source exposure mapped back to original capital, once per slice."""
    if risk.get("exposure_basis") != "NAMED_SOURCE_CHANNELS":
        return [], []
    named, failures = _named_source_exposure(risk)
    if failures:
        return [], failures
    if not named:
        return [], []
    originals = {p.symbol: p for p in portfolio.positions}
    accepted = {str(r.get("symbol")): r for r in lookthrough or []
                if isinstance(r, Mapping) and not r.get("unresolved")
                and r.get("basis") == "REVIEWED_LEDGER_DIRECT_AND_LOOK_THROUGH"}
    slices = {p.symbol: {} for p in portfolio.positions}
    failures = []
    for symbol, expected in named.items():
        parts = {}
        if symbol in originals and originals[symbol].asset_type != "cash":
            parts[symbol] = originals[symbol].weight
        record = accepted.get(symbol) or {}
        for parent, weight in (record.get("via") or {}).items():
            if parent in originals and originals[parent].asset_type == "etf" and _finite(weight) and weight > 0:
                parts[parent] = weight
        observed = math.fsum(parts.values())
        # The named link must cover the same source-identified slices. Never
        # enlarge an event exposure from the parent investment's full NAV.
        if not parts or observed > expected + 1e-8:
            failures.append({"symbol": symbol, "reason": "NAMED_WEIGHT_AND_ORIGINAL_SLICES_DO_NOT_RECONCILE"})
            continue
        if observed < expected - 1e-8:
            failures.append({"symbol": symbol, "reason": "SOURCE_LINKED_WEIGHT_NOT_RECOVERED",
                             "unrecovered_weight": expected - observed})
        for parent, weight in parts.items():
            slices[parent][symbol] = weight
    for parent, values in slices.items():
        # If the whole ETF is itself named, constituent names are an alternate
        # view of the same capital and must not be added a second time.
        if parent in values:
            slices[parent] = {parent: originals[parent].weight}
        if math.fsum(slices[parent].values()) > originals[parent].weight + 1e-8:
            failures.append({"symbol": parent, "reason": "LINKED_SLICES_EXCEED_ORIGINAL_NAV"})
            slices[parent] = {}
    covered = math.fsum(math.fsum(v.values()) for v in slices.values())
    if covered <= 0:
        return [], failures
    scenarios = []
    selected_symbols = sorted({s for v in slices.values() for s in v})
    for shock in (-1., 1.):
        contributions = []
        for p in portfolio.positions:
            exposed = math.fsum(slices[p.symbol].values())
            pnl = portfolio.notional * exposed * shock / 100
            contributions.append({"symbol": p.symbol, "weight": p.weight,
                                  "response_weight": exposed, "pnl": pnl,
                                  "modeled_return_fraction": pnl / (portfolio.notional * p.weight) if p.weight else 0.,
                                  "linked_symbols": sorted(slices[p.symbol]),
                                  "basis": "FIXED_CONSTITUENT_WEIGHT_PRICE_COMPONENT" if exposed else "OTHER_PRICES_EXPLICITLY_HELD_FIXED"})
        subtotal = math.fsum(r["pnl"] for r in contributions)
        scenarios.append({
            "id": "linked-price-down" if shock < 0 else "linked-price-up",
            "label": f"Linked holdings {'−' if shock < 0 else '+'}1%",
            "shock": {"factor": "LINKED_HOLDING_RETURN_PCT", "value": shock, "unit": "percent",
                      "symbols": selected_symbols, "basis": "ONE_PERCENT_UNIT_SENSITIVITY_NOT_EVENT_SEVERITY"},
            "pnl": None, "known_pnl": subtotal,
            "covered_weight": covered, "unmodeled_weight": max(0., 1 - covered),
            "position_contributions": contributions, "probability": None,
            "scope": "LINKED_PRICE_COMPONENT_ONLY", "event_effect_estimated": False,
            "scope_note": "Named holdings move together by 1%; other prices stay fixed. This is the exposed component.",
            "horizon": "Instantaneous price change at fixed current holdings; no carry, fees or rebalancing.",
        })
    return scenarios, failures


def attach_scenario_responses(risks, report, *, lookthrough=()):
    """Add separate B/C diagnostics; never modify outcomes, probabilities or rank."""
    portfolio_data = report.get("portfolio") or {}
    portfolio = parse_portfolio(portfolio_data.get("positions") or [], notional=portfolio_data.get("notional"))
    model = report.get("scenario_response_model") or {}
    for risk in risks:
        response = {
            "schema": SCHEMA, "level": "C", "status": "NO_SUPPORTED_RESPONSE_CHANNEL",
            "title": "What if markets move?", "scenarios": [],
            "linkage": {"status": "UNRESOLVED", "event_to_factor_identified": False},
            "event_id": risk.get("id"), "event_conditional_coverage_increment": 0,
            "forecast_validated": False, "event_effect_estimated": False,
            "probability": None, "missing_channels": [],
            "existing_model_conditional": bool(risk.get("outcomes")),
            "assumptions": ["Shock size is a scenario input; its probability and event severity are not estimated.",
                            "Current portfolio weights are fixed. Source contract odds do not enter this calculation."],
            "source_as_of": report.get("evaluated_at_utc"),
        }
        quality = risk.get("quality") or {}
        if ((quality.get("definition_review") or {}).get("status") != "DEFINED_FOR_REVIEW"
                or (quality.get("instrument_identity_gate") or {}).get("blocked_symbols")):
            response["missing_channels"] = [{"reason": "EVENT_DEFINITION_OR_INSTRUMENT_IDENTITY_UNRESOLVED"}]
        elif risk.get("family") in _MACRO and model.get("schema") == MODEL_SCHEMA and model.get("status") == "COMPUTED":
            response["linkage"] = {"status": "SCHEDULED_MACRO_FACTOR_HYPOTHESIS",
                "basis": "The release is linked to market rates and equity prices; its factor moves are explicit hypotheses.",
                "source_url": _MACRO[risk["family"]], "event_to_factor_identified": False}
            for factor, scale in (model.get("scales") or {}).items():
                if factor not in ("SPY_RETURN_PCT", "US_10Y_YIELD_CHANGE_BP"):
                    continue
                # A fixed-cash or other exact zero alone is not a response
                # channel. Its zero remains useful inside a supported model.
                if not any(row.get("status") == "AVAILABLE" and row.get("weight", 0) > 0
                           and _finite((row.get("coefficients") or {}).get(factor))
                           and row["coefficients"][factor] != 0 for row in model.get("positions") or []):
                    continue
                for direction in (-1, 1):
                    try:
                        scenario = _factor_scenario(model, factor, direction * scale["value"], scale=scale)
                    except ValueError:
                        response["missing_channels"].append({"factor": factor, "reason": "SCENARIO_NUMERIC_DOMAIN_EXCEEDED"})
                        continue
                    if scenario["covered_weight"] > 0:
                        response["scenarios"].append(scenario)
            response["response_evidence"] = {"calibration_policy": model.get("calibration_policy"),
                "factor_source_scope": model.get("factor_source_scope"),
                "factor_sources": deepcopy(model.get("factor_sources") or {}),
                "positions": [{k: deepcopy(row[k]) for k in ("symbol", "status", "reason", "basis", "training_span",
                              "holdout_span", "holdout_rmse", "holdout_zero_rmse", "holdout_oos_r2", "source") if k in row}
                              for row in model.get("positions") or []]}
            response["missing_channels"].extend({"symbol": r["symbol"], "reason": r["reason"]}
                                                for r in model["positions"] if r.get("status") != "AVAILABLE")
            if not response["scenarios"]:
                reason = ("NO_NONZERO_ADMITTED_RESPONSE" if model.get("scales")
                          else "INSUFFICIENT_TRAINING_HISTORY_FOR_SCENARIO_SCALE")
                response["missing_channels"].append({"reason": reason})
        else:
            scenarios, proxy_evidence, failures = _proxy_scenarios(risk, model, portfolio, lookthrough)
            if scenarios:
                response["linkage"] = proxy_evidence["linkage"]
                response["response_evidence"] = {"proxy_fits_including_failed_positions": proxy_evidence["fits"],
                    "factor_sources": deepcopy(model.get("factor_sources") or {}),
                    "proxy_sources": {symbol: deepcopy((model.get("proxy_sources") or {}).get(symbol) or {})
                                      for symbol in proxy_evidence["fits"]},
                    "position_sources": {row["symbol"]: deepcopy(row.get("source") or {})
                                         for row in model.get("positions") or []},
                    "scope": "The admission period selected allowed responses; it is not independent post-selection validation."}
                response["assumptions"].append("Each proxy shock is a separate scenario; do not add their totals.")
            else:
                scenarios, link_failures = _linked_components(risk, portfolio, lookthrough)
                failures.extend(link_failures)
            response["scenarios"] = scenarios
            response["missing_channels"] = failures
            if scenarios and scenarios[0]["scope"] == "LINKED_PRICE_COMPONENT_ONLY":
                response["linkage"] = {"status": "DOCUMENTED_NAMED_HOLDING_COMPONENT",
                    "basis": "Named source-linked holdings and accepted original-position slices.",
                    "event_to_factor_identified": False, "exposure_basis": risk.get("exposure_basis")}
            elif not scenarios:
                response["missing_channels"].append({"reason": "NO_SOURCE_LINKED_POSITION_TRANSMISSION"})
        if response["scenarios"]:
            response.update(level="B", status="PARAMETRIC_SENSITIVITY_NOT_EVENT_EFFECT")
        risk["scenario_response"] = response
    return risks
