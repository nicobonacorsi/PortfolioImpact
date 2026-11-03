"""Separate fixed published-response FOMC pilot, using only the standard library.

This does not change historical_chain's empirical 24/12 protocol. Historical
and application replay use the same deterministic prediction function below.
Only daily equity loadings and prior-event residuals are estimated; the macro
response is the original declared published point, never a FOMC-outcome fit.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from .historical_chain import (
    ChainInputError, SCOPE_FIELDS, _aggregate, _checked_event, _distribution,
    _dot, _finite, _interval, _iso, _probability_scores, _score_forecast,
    _time, _weights,
)

MODEL_SCHEMA = "castle.published_fomc_response.r11.v1"
PREDICTION_SCHEMA = "castle.published_fomc_predictive_distribution.r11.v1"
REPORT_SCHEMA = "castle.historical_published_fomc_chain.r11.v1"
CONTEXT_SCHEMA = "castle.historical_daily_beta_contexts.r11.v1"
CALIBRATION_COUNT = 12
DAILY_PAIRS = 252
MIN_DAILY_PAIRS = 60
EQUITY_RESPONSE = -0.000468
YIELD_RESPONSE = 0.194
TLT_DURATION = 16.0
SYMBOLS = ("NVDA", "QQQ", "TLT")
PORTFOLIO_WEIGHTS = {"NVDA": 0.35, "QQQ": 0.45, "TLT": 0.20}
SCOPE = {
    "family": "FOMC", "metric": "scheduled_FOMC_target_rate_change", "state_unit": "basis_points",
    "cohort": "FOMC_LAGGED_REFERENCE_24H",
    "model_domain_id": "HELD_SCHEDULED_FOMC_QUARTER_POINT_ABS100_V1",
    "horizon_id": "FOMC_1359_CLOSE_1414_CLOSE",
}
ASSUMPTIONS = [
    "Fixed published one-day broad-equity response transferred to a 15-minute announcement window; not a validated 15-minute response.",
    "Trailing prior daily equity OLS slopes transfer to announcement exposure; daily intercept is not transferred.",
    "TLT duration is fixed at an assumed 16 years, not a measured historical duration; convexity and tracking are omitted.",
    "Contract Q is an observed reference, not certified equity pricing weights or physical probability.",
    "Whole prior-event residual vectors are transferred across states and time; conditional noise stability is an assumption.",
    "Up-to-24-hour quotes, retrospective rules and finite held-meeting/quarter-point domain assumptions remain upstream limitations.",
]


def canonical_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def published_source_receipt():
    """Execute exactly the original helper AST, without importing its dependencies.

    This checks the fixed point response mechanism only, not equality to the
    original Monte Carlo/combined-loading production model.
    """
    root = Path(__file__).parent
    namespace = {"deepcopy": deepcopy}
    hashes = {}
    for name, functions, constants in (
        ("default_event_economics.py", {"published_response_model"}, {"FOMC_SOURCE", "CPI_SOURCE"}),
        ("outcome_table.py", {"_factor_model"}, {"KUTTNER_2001", "BEECHEY_WRIGHT_2009", "CPI_RATE_PROXY_UPPER"}),
    ):
        raw = (root / name).read_bytes()
        hashes[name] = sha256(raw).hexdigest()
        tree = ast.parse(raw.decode("utf-8-sig"))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
    equity = namespace["published_response_model"]("FED_POLICY")
    factors = namespace["_factor_model"]("FED_POLICY")
    if equity["coefficient"] != EQUITY_RESPONSE or factors["equity_point"] != EQUITY_RESPONSE or factors["rate"]["point"] != YIELD_RESPONSE:
        raise ChainInputError("Fixed published coefficients disagree with original production helpers")
    return {"status": "EXACT_ORIGINAL_HELPER_POINT_MATCH", "source_module_sha256": hashes,
            "equity_response": EQUITY_RESPONSE, "equity_unit": equity["coefficient_unit"],
            "yield_response": YIELD_RESPONSE, "yield_unit": factors["rate"]["unit"],
            "equity_source": equity["source"], "yield_source": factors["rate"]["source"],
            "verification_scope": "Point response only; new historical daily-only loading and residual diagnostic, not equivalence to original full Monte Carlo model"}


def _daily_returns_from_archive(archive):
    """Read validated unadjusted bars and splits; never expose private URLs."""
    raw = Path(archive).read_bytes() if isinstance(archive, (str, Path)) else archive
    if not isinstance(raw, bytes) or len(raw) > 38_000_000:
        raise ChainInputError("Invalid or oversized daily source archive")
    with ZipFile(BytesIO(raw)) as z:
        files, total = {}, 0
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            total += info.file_size
            if (info.file_size > 24_000_000 or total > 38_000_000 or name in files or
                    "/" not in name or "\\" in name or ".." in name.split("/")):
                raise ChainInputError("Unsafe or duplicate daily archive member")
            files[name] = z.read(info)
    manifest = json.loads(files["meta/manifest.json"])
    if manifest.get("schema") != "CASTLE_P3_MASSIVE_V1":
        raise ChainInputError("Daily archive schema mismatch")
    returns, member_hashes, spans = {}, {}, {}
    for symbol in ("NVDA", "QQQ", "SPY"):
        sources = {}
        for kind in ("bars", "splits"):
            record = manifest.get("assets", {}).get(symbol, {}).get(kind, {})
            name = f"raw/{symbol}_{kind}.json"
            if record.get("status") != "VERIFIED_HTTP_200" or record.get("path") != name or name not in files:
                raise ChainInputError("Required verified daily source member is absent")
            content = files[name]
            digest = sha256(content).hexdigest()
            if len(content) != record.get("bytes") or digest != record.get("sha256"):
                raise ChainInputError("Daily source member checksum mismatch")
            data = json.loads(content)
            if not isinstance(data, dict) or not isinstance(data.get("results"), list) or data.get("next_url"):
                raise ChainInputError("Incomplete daily provider response")
            if kind == "bars" and (data.get("ticker") != symbol or data.get("adjusted") is not False):
                raise ChainInputError("Daily bars must be symbol-matched unadjusted prices")
            sources[kind] = data["results"]
            member_hashes[name] = digest
        closes, splits = {}, {}
        for row in sources["bars"]:
            day = datetime.fromtimestamp(_finite(row["t"], "bar timestamp") / 1000, timezone.utc).date().isoformat()
            close = _finite(row["c"], "daily close")
            if close <= 0 or day in closes:
                raise ChainInputError("Invalid or duplicate daily close")
            closes[day] = close
        for row in sources["splits"]:
            if row.get("ticker", symbol) != symbol:
                raise ChainInputError("Split ticker mismatch")
            day = date.fromisoformat(row["execution_date"]).isoformat()
            numerator, denominator = _finite(row["split_to"], "split numerator"), _finite(row["split_from"], "split denominator")
            if min(numerator, denominator) <= 0 or day in splits:
                raise ChainInputError("Invalid or duplicate split")
            splits[day] = numerator / denominator
        days = sorted(closes)
        if len(days) < MIN_DAILY_PAIRS + 1:
            raise ChainInputError("Insufficient raw daily prices")
        pairs = {}
        for prior, day in zip(days, days[1:]):
            if (date.fromisoformat(day) - date.fromisoformat(prior)).days > 7:
                continue
            # A split must coincide with a listed daily endpoint, never silently
            # disappear inside a missing-price gap.
            if any(prior < split_day < day for split_day in splits):
                continue
            value = closes[day] * splits.get(day, 1.0) / closes[prior] - 1.0
            if not math.isfinite(value):
                raise ChainInputError("Nonfinite daily return")
            pairs[(prior, day)] = value
        returns[symbol] = pairs
        spans[symbol] = {"bar_n": len(days), "first_price_date": days[0], "last_price_date": days[-1], "return_pair_n": len(pairs)}
    return returns, {"archive_sha256": sha256(raw).hexdigest(), "member_sha256": member_hashes,
                     "daily_spans": spans, "source_authentication": "USER_HELD_HASH_CONSISTENT_NOT_PROVIDER_SIGNED_NOT_HISTORICAL_VINTAGE_CERTIFIED"}


def daily_beta_context(event, returns, source_receipt):
    """Calculate historical loadings from strictly prior matched date endpoints."""
    cutoff = _time(event["cutoff_time_utc"], "daily beta cutoff")
    release = _time(event["release_time_utc"], "daily beta event")
    if cutoff >= release or cutoff.date() != release.date():
        raise ChainInputError("Daily beta cutoff must precede release on its event date")
    event_day = cutoff.date().isoformat()
    loadings = {}
    for symbol in ("NVDA", "QQQ"):
        matched = sorted(pair for pair in set(returns[symbol]) & set(returns["SPY"])
                         if pair[0] < pair[1] < event_day)[-DAILY_PAIRS:]
        if len(matched) < MIN_DAILY_PAIRS:
            raise ChainInputError("Fewer than 60 strictly prior matched daily return pairs")
        xs = [_finite(returns["SPY"][p], "benchmark daily return") for p in matched]
        ys = [_finite(returns[symbol][p], "asset daily return") for p in matched]
        xbar, ybar = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
        sxx = math.fsum((x - xbar) ** 2 for x in xs)
        if sxx <= len(xs) * (1e-12 * max(1.0, max(abs(x) for x in xs))) ** 2:
            raise ChainInputError("Daily beta design is rank deficient")
        beta = math.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
        digest_rows = [{"start": p[0], "end": p[1], "x": x, "y": y} for p, x, y in zip(matched, xs, ys)]
        loadings[symbol] = {"beta": beta, "daily_intercept_not_transferred": ybar - beta * xbar,
                            "n_pairs": len(matched), "first_pair": list(matched[0]), "last_pair": list(matched[-1]),
                            "matched_pair_dates": [list(p) for p in matched],
                            "matched_returns_sha256": canonical_sha256(digest_rows), "benchmark": "SPY"}
    return {"event_id": event["event_id"], "cutoff_time_utc": _iso(cutoff), "release_time_utc": _iso(release),
            "price_dates_strictly_before": event_day, "daily_loadings": loadings,
            "algorithm": "TRAILING_252_MATCHED_SPLIT_ADJUSTED_PRICE_RETURN_OLS_WITH_INTERCEPT_MIN60_SLOPE_ONLY",
            "source_archive_sha256": source_receipt["archive_sha256"],
            "source_member_sha256": source_receipt["member_sha256"]}


def build_beta_contexts(events, daily_archive):
    """Derive distributable beta receipts; no raw prices or private URLs returned."""
    returns, receipt = _daily_returns_from_archive(daily_archive)
    contexts, exclusions, seen = [], [], set()
    for event in events:
        if event["event_id"] in seen:
            raise ChainInputError("Duplicate beta context event ID")
        seen.add(event["event_id"])
        if event.get("eligible") is False:
            continue
        try:
            contexts.append(daily_beta_context(event, returns, receipt))
        except (ChainInputError, KeyError, TypeError, ValueError) as exc:
            exclusions.append({"event_id": event["event_id"], "reasons": ["DAILY_CONTEXT_UNAVAILABLE", str(exc)]})
    return {"schema": CONTEXT_SCHEMA, "source_receipt": receipt, "contexts": contexts, "exclusions": exclusions,
            "privacy": "Derived betas and dates only; private raw price archive and provider URLs omitted"}


def _context_map(contexts):
    if not isinstance(contexts, dict) or contexts.get("schema") != CONTEXT_SCHEMA:
        raise ChainInputError("Versioned historical beta contexts required")
    answer = {}
    for row in contexts.get("contexts", []):
        if row["event_id"] in answer:
            raise ChainInputError("Duplicate historical beta context")
        answer[row["event_id"]] = row
    return answer


def _context_coefficients(context, event):
    if context.get("event_id") != event["event_id"]:
        raise ChainInputError("Beta context belongs to a different event")
    context_cutoff = _time(context.get("cutoff_time_utc"), "context cutoff")
    context_release = _time(context.get("release_time_utc"), "context release")
    if context_cutoff >= context_release or context_cutoff.date() != context_release.date():
        raise ChainInputError("Beta context cutoff must precede its same-day release")
    for field in ("cutoff_time_utc", "release_time_utc"):
        if _time(context.get(field), "context time") != _time(event[field], "event time"):
            raise ChainInputError("Beta context historical timestamp mismatch")
    event_day = _time(event["cutoff_time_utc"], "event cutoff").date().isoformat()
    if context.get("price_dates_strictly_before") != event_day:
        raise ChainInputError("Beta context price cutoff mismatch")
    slopes = {"TLT": -TLT_DURATION * YIELD_RESPONSE / 10000}
    for symbol in ("NVDA", "QQQ"):
        fit = context["daily_loadings"][symbol]
        dates = fit.get("matched_pair_dates", [])
        if (not MIN_DAILY_PAIRS <= fit.get("n_pairs", 0) <= DAILY_PAIRS or len(dates) != fit["n_pairs"] or
                dates != sorted(dates) or len(set(tuple(x) for x in dates)) != len(dates) or
                fit.get("first_pair") != dates[0] or fit.get("last_pair") != dates[-1] or
                any(len(p) != 2 or not p[0] < p[1] < event_day or
                    (date.fromisoformat(p[1]) - date.fromisoformat(p[0])).days > 7 for p in dates)):
            raise ChainInputError("Daily loading dates/counts permit look-ahead or invalid matching")
        slopes[symbol] = _finite(fit["beta"], "asof daily beta") * EQUITY_RESPONSE
    return {s: {"slope": slopes[s], "intercept": 0.0} for s in SYMBOLS}


def _scope_check(row):
    if any(row.get(k) != value for k, value in SCOPE.items()) or not row.get("state_mapping_sha256"):
        raise ChainInputError("Published FOMC diagnostic scope mismatch")
    if row.get("eligible") is False:
        raise ChainInputError("Source-ineligible event cannot be predicted")


def fit_published_model(prior_events, beta_contexts, current_event):
    """Prepare fixed response plus last12 previous joint residuals; no macro fit."""
    current = _checked_event(current_event, outcome_required=False)
    _scope_check(current)
    cutoff, contexts = _time(current["cutoff_time_utc"], "forecast cutoff"), _context_map(beta_contexts)
    base = {"schema": MODEL_SCHEMA, "model_type": "FIXED_PUBLISHED_RESPONSE_ASOF_DAILY_BETA_JOINT_RESIDUALS",
            "status": "INSUFFICIENT_HISTORY", **{k: current[k] for k in ("family", "metric", "state_unit", *SCOPE_FIELDS)},
            "symbols": list(SYMBOLS), "forecast_event_id": current["event_id"], "fit_cutoff_time_utc": current["cutoff_time_utc"],
            "required_prior_n": CALIBRATION_COUNT, "macro_response_fit_n": 0,
            "fixed_response": {"equity": EQUITY_RESPONSE, "yield": YIELD_RESPONSE, "tlt_duration_assumed_years": TLT_DURATION},
            "assumptions": list(ASSUMPTIONS), "exclusions": []}
    if current["event_id"] not in contexts:
        base.update(status="MISSING_ASOF_DAILY_CONTEXT", prior_complete_n=0)
        return base
    context = contexts[current["event_id"]]
    base.update(daily_context=deepcopy(context), coefficients=_context_coefficients(context, current))
    valid, seen = [], set()
    for event in prior_events:
        eid = event.get("event_id")
        if eid in seen:
            raise ChainInputError("Duplicate prior event ID")
        seen.add(eid)
        reasons = []
        if event.get("eligible") is False:
            reasons.extend(event.get("exclusion_reasons") or ["SOURCE_INELIGIBLE"])
        if any(event.get(k) != current.get(k) for k in ("family", "metric", "state_unit", *SCOPE_FIELDS)):
            reasons.append("DIFFERENT_SCOPE")
        if reasons:
            base["exclusions"].append({"event_id": eid, "reasons": reasons})
            continue
        row = _checked_event(event, outcome_required=True)
        if eid == current["event_id"] or _time(row["available_time_utc"], "prior availability") >= cutoff:
            reasons.append("NOT_AVAILABLE_BEFORE_FORECAST")
        if any(s not in row["asset_returns"] for s in SYMBOLS):
            reasons.append("INCOMPLETE_ORIGINAL_POSITIONS")
        if eid not in contexts:
            reasons.append("MISSING_ASOF_DAILY_CONTEXT")
        if reasons:
            base["exclusions"].append({"event_id": eid, "reasons": reasons})
            continue
        coefficients = _context_coefficients(contexts[eid], row)
        valid.append((row, coefficients))
    valid.sort(key=lambda item: (item[0]["release_time_utc"], item[0]["event_id"]))
    if len({x[0]["release_time_utc"] for x in valid}) != len(valid):
        raise ChainInputError("Duplicate FOMC release in calibration candidates")
    base["prior_complete_n"] = len(valid)
    if len(valid) < CALIBRATION_COUNT:
        return base
    selected = valid[-CALIBRATION_COUNT:]
    residuals, actuals = [], []
    for row, coefficients in selected:
        surprise = row["realized_reference_surprise"]
        central = {s: coefficients[s]["slope"] * surprise for s in SYMBOLS}
        residuals.append({"event_id": row["event_id"], "returns": {s: row["asset_returns"][s] - central[s] for s in SYMBOLS},
                          "central_event_returns": central, "reference_surprise": surprise, "coefficients": coefficients,
                          "available_time_utc": row["available_time_utc"], "daily_context": deepcopy(contexts[row["event_id"]]),
                          "source_hashes": row.get("source_hashes", {})})
        actuals.append({"event_id": row["event_id"], "returns": {s: row["asset_returns"][s] for s in SYMBOLS},
                        "settlement_value": row.get("settlement_value"), "available_time_utc": row["available_time_utc"]})
    base.update(status="READY", calibration_n=CALIBRATION_COUNT,
                calibration_event_ids=[r["event_id"] for r in residuals], residual_vectors=residuals,
                calibration_returns=actuals,
                latest_training_available_time_utc=max(r["available_time_utc"] for r in actuals))
    return base


def predict_published_distribution(model, event, *, weights=None, notional_usd=100000.0):
    """Shared application/historical table: independent of forecast realization."""
    if model.get("schema") != MODEL_SCHEMA or model.get("status") != "READY":
        raise ChainInputError("Ready separate published-response model required")
    row = _checked_event(event, outcome_required=False)
    _scope_check(row)
    if model.get("forecast_event_id") != row["event_id"] or any(row.get(k) != model.get(k) for k in ("family", "metric", "state_unit", *SCOPE_FIELDS)):
        raise ChainInputError("Published model event or source scope mismatch")
    if model.get("macro_response_fit_n") != 0 or model.get("fixed_response") != {"equity": EQUITY_RESPONSE, "yield": YIELD_RESPONSE, "tlt_duration_assumed_years": TLT_DURATION}:
        raise ChainInputError("Published fixed response was altered")
    cutoff = _time(row["cutoff_time_utc"], "prediction cutoff")
    if (_time(model["fit_cutoff_time_utc"], "model cutoff") != cutoff or
            _time(model["latest_training_available_time_utc"], "training availability") >= cutoff):
        raise ChainInputError("Published model chronology violation")
    coefficients = _context_coefficients(model["daily_context"], row)
    if coefficients != model.get("coefficients"):
        raise ChainInputError("Published coefficients disagree with asof historical context")
    ids, residuals, actuals = model["calibration_event_ids"], model["residual_vectors"], model["calibration_returns"]
    if (model.get("calibration_n") != CALIBRATION_COUNT or len(ids) != CALIBRATION_COUNT or len(set(ids)) != CALIBRATION_COUNT or
            row["event_id"] in ids or [x["event_id"] for x in residuals] != ids or [x["event_id"] for x in actuals] != ids):
        raise ChainInputError("Exactly12 aligned prior complete calibration vectors required")
    calibration_availabilities = []
    for residual, actual in zip(residuals, actuals):
        available = _time(residual["available_time_utc"], "residual availability")
        if available >= cutoff:
            raise ChainInputError("Future calibration residual")
        if _time(actual["available_time_utc"], "actual return availability") != available:
            raise ChainInputError("Residual and actual return availability must match")
        calibration_availabilities.append(available)
        prior_context = residual["daily_context"]
        if prior_context.get("event_id") != residual["event_id"]:
            raise ChainInputError("Residual beta context must belong to its calibration event")
        if _time(prior_context.get("release_time_utc"), "calibration context release") > available:
            raise ChainInputError("Calibration beta context release cannot follow outcome availability")
        prior_coefficients = _context_coefficients(prior_context, prior_context)
        if prior_coefficients != residual["coefficients"]:
            raise ChainInputError("Calibration must use its own historical daily beta")
        for s in SYMBOLS:
            value = _finite(residual["returns"][s], "residual return")
            central = prior_coefficients[s]["slope"] * _finite(residual["reference_surprise"], "calibration surprise")
            observed = _finite(actual["returns"][s], "calibration observed return")
            if abs(value - (observed - central)) > 1e-14 or abs(residual["central_event_returns"][s] - central) > 1e-14:
                raise ChainInputError("Residual does not reconcile with its own historical central response")
    if _time(model["latest_training_available_time_utc"], "declared latest availability") != max(calibration_availabilities):
        raise ChainInputError("Latest training availability must match calibration receipts")
    weights = _weights(PORTFOLIO_WEIGHTS if weights is None else weights, SYMBOLS)
    nav = _finite(notional_usd, "notional")
    if nav <= 0:
        raise ChainInputError("Notional must be positive")
    scenarios, atoms = [], []
    for state in row["states"]:
        surprise = state["value"] - row["reference_expectation"]
        central = {s: coefficients[s]["slope"] * surprise for s in SYMBOLS}
        portfolio = _dot(weights, central)
        scenarios.append({"state_id": state["id"], "q": state["q"], "state_value": state["value"], "surprise": surprise,
                          "event_component_returns": central, "conditional_returns": dict(central),
                          "event_component_portfolio_return": portfolio, "conditional_portfolio_return": portfolio,
                          "event_component_pnl_usd": nav * portfolio, "conditional_pnl_usd": nav * portfolio})
        for residual in residuals:
            returns = {s: central[s] + residual["returns"][s] for s in SYMBOLS}
            portfolio = _dot(weights, returns)
            atoms.append({"state_id": state["id"], "calibration_event_id": residual["event_id"],
                          "probability": state["q"] / CALIBRATION_COUNT, "asset_returns": returns,
                          "portfolio_return": portfolio, "pnl_usd": nav * portfolio})
    values, masses = [a["portfolio_return"] for a in atoms], [a["probability"] for a in atoms]
    _distribution(values, masses)
    mean = math.fsum(v * p for v, p in zip(values, masses))
    variance = math.fsum(p * (v - mean) ** 2 for v, p in zip(values, masses))
    residual_mean = {s: math.fsum(r["returns"][s] for r in residuals) / CALIBRATION_COUNT for s in SYMBOLS}
    event_mean = math.fsum(s["q"] * s["event_component_portfolio_return"] for s in scenarios)
    if abs(event_mean) > 1e-12:
        raise ChainInputError("Published event contribution failed Q-centering")
    summary = {"mean_return": mean, "mean_pnl_usd": nav * mean,
               "standard_deviation_return": math.sqrt(variance), "standard_deviation_pnl_usd": nav * math.sqrt(variance),
               "q_weighted_event_component_return": event_mean, "intercept_portfolio_return": 0.0,
               "mean_residual_asset_returns": residual_mean, "mean_residual_portfolio_return": _dot(weights, residual_mean),
               **_interval(values, masses)}
    summary.update(lower_05_pnl_usd=nav * summary["lower_05"], upper_95_pnl_usd=nav * summary["upper_95"])
    return {"schema": PREDICTION_SCHEMA, "status": "READY", "event_id": row["event_id"],
            **{k: row[k] for k in ("family", "metric", "state_unit", *SCOPE_FIELDS)},
            "release_time_utc": row["release_time_utc"], "cutoff_time_utc": row["cutoff_time_utc"],
            "reference_expectation": row["reference_expectation"], "symbols": list(SYMBOLS),
            "weights": weights, "notional_usd": nav, "scenario_rows": scenarios, "atoms": atoms,
            "distribution_summary": summary, "physical_probability_P": None, "assumptions": list(ASSUMPTIONS),
            "distribution_interpretation": "Q-reference mixture with last12 joint residual vectors; separate published-response pilot",
            "event_component_interpretation": "Fixed published central response, zero fitted intercept, residual mean disclosed separately"}


def evaluate_published_chain(events, beta_contexts):
    """Full separate report retaining negative results and every exclusion."""
    source_receipt = published_source_receipt()
    accepted, exclusions, seen = [], [], set()
    for event in events:
        eid = event.get("event_id")
        if eid in seen:
            raise ChainInputError("Duplicate published diagnostic event ID")
        seen.add(eid)
        try:
            if event.get("eligible") is False:
                raise ChainInputError("SOURCE_INELIGIBLE: " + "; ".join(map(str, event.get("exclusion_reasons", []))))
            row = _checked_event(event, outcome_required=True)
            _scope_check(row)
            accepted.append(row)
        except (ChainInputError, KeyError, TypeError) as exc:
            exclusions.append({"event_id": eid, "reasons": [str(exc)], "source_receipt": dict(event)})
    accepted.sort(key=lambda e: (e["release_time_utc"], e["event_id"]))
    if len({e["release_time_utc"] for e in accepted}) != len(accepted):
        raise ChainInputError("Duplicate selected FOMC release")
    targets = [{"id": "symbol:" + s, "kind": "symbol", "weights": {s: 1.0}, "notional_usd": 100000.0} for s in SYMBOLS]
    targets.append({"id": "portfolio_A", "kind": "portfolio", "weights": dict(PORTFOLIO_WEIGHTS), "notional_usd": 100000.0})
    results = []
    for target in targets:
        forecasts = []
        for event in accepted:
            record = {"event_id": event["event_id"], "family": "FOMC", "release_time_utc": event["release_time_utc"],
                      "cutoff_time_utc": event["cutoff_time_utc"], "source_hashes": event.get("source_hashes", {}),
                      "normalized_event": deepcopy(event)}
            if any(s not in event["asset_returns"] for s in SYMBOLS):
                record.update(status="INCOMPLETE_ORIGINAL_POSITIONS")
                forecasts.append(record)
                continue
            prior = [e for e in accepted if _time(e["release_time_utc"], "prior event") < _time(event["cutoff_time_utc"], "cutoff")]
            model = fit_published_model(prior, beta_contexts, event)
            record.update(model=model, status=model["status"])
            if model["status"] == "READY":
                prediction = predict_published_distribution(model, event, weights=target["weights"], notional_usd=target["notional_usd"])
                record.update(status="EVALUATED", prediction=prediction, prediction_sha256=canonical_sha256(prediction),
                              scores=_score_forecast(event, model, prediction, target["weights"]),
                              probability_scores=_probability_scores(event, model["calibration_returns"]),
                              settled_state_id=event["settled_state_id"], settlement_value=event.get("settlement_value"))
            forecasts.append(record)
        summary = _aggregate(forecasts)
        summary.update(status="RETROSPECTIVE_PUBLISHED_RESPONSE_PILOT" if summary["n"] else "NO_ELIGIBLE_FORECASTS",
                       empirical_skill_supported=False, broad_empirical_endorsement=False,
                       interpretation="Pilot sample CRPS comparison only; no broad skill, historical duration or pricing-alpha certification")
        results.append({"target": target, **SCOPE, "state_mapping_sha256": accepted[0].get("state_mapping_sha256") if accepted else None,
                        "candidate_complete_source_n": len(accepted), "forecasts": forecasts,
                        "status_counts": dict(Counter(f["status"] for f in forecasts)), "summary": summary})
    n = sum(r["summary"]["n"] for r in results)
    return {"schema": REPORT_SCHEMA, "evaluation_type": "RETROSPECTIVE_FIXED_PUBLISHED_RESPONSE_PILOT_SEPARATE_FROM_EMPIRICAL_V1",
            "protocol": {"calibration_count": CALIBRATION_COUNT, "daily_pairs": DAILY_PAIRS, "minimum_daily_pairs": MIN_DAILY_PAIRS,
                         "macro_response_fit_n": 0, "fixed_tlt_duration_assumed": TLT_DURATION,
                         "document": "docs/R11_PUBLISHED_FOMC_PROTOCOL.md",
                         "document_sha256": sha256((Path(__file__).resolve().parents[1] / "docs/R11_PUBLISHED_FOMC_PROTOCOL.md").read_bytes()).hexdigest()},
            "input_event_n": len(events), "normalized_eligible_event_n": len(accepted), "scored_forecast_n": n,
            "source_exclusions": exclusions, "context_exclusions": beta_contexts.get("exclusions", []),
            "published_source_receipt": source_receipt, "daily_contexts_sha256": canonical_sha256(beta_contexts),
            "target_results": results, "event_receipts": accepted, "limitations": list(ASSUMPTIONS),
            "conclusions": {"pipeline_arithmetic_checked": True, "historical_chain_evaluated": n > 0,
                            "frozen_empirical_v1_requirements_unchanged": True, "universal_empirical_skill_claim": False,
                            "source_admission_verified_by_this_module": False,
                            "note": "Actual historical Q to published-response table to original-position P&L is evaluated;9meetings are a pilot and overlapping targets are not independent."}}
