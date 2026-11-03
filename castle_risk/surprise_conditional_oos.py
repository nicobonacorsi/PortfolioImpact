"""Continuous-surprise conditional return evidence screen (Castle R122).

Generalizes the R121 binary YES/NO screen to a scheduled event whose exact
outcome E fixes a pre-announcement *surprise* s(E) = D(E) - m, where m is a
market expectation observed before the release and D(E) is the decision under
E. Conditional on the realized decision, s(E_realized) is the recorded surprise.
Scoring uses the realized branch only; the counterfactual branch is never scored.

Nothing here estimates P(E), a venue quote Q, or an equity priced-in fraction q.
A research-only PASS is not promoted into live portfolio P&L: that requires a
separate forward test with pre-announcement inputs and independent resolution.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import math
import random
from typing import Any, Mapping, Sequence

SCHEMA = "castle.r122_surprise_conditional_historical_screen.v1"
_ALLOWED_ASSETS = {"stock", "etf", "cash"}
_MIN_TRAIN = 24
_MIN_TEST = 24
_SHRINKAGE_PSEUDO_COUNT = 10
_MIN_RMSE_IMPROVEMENT = .05
_MIN_WIN_FRACTION = .75
_BOOTSTRAP_SEED = 122
_BOOTSTRAP_REPLICATES = 2000
_BRANCH_THRESHOLD_PP = .025
_SECONDARY_ABS_SURPRISE_PP = .05
_RIDGE_RELATIVE_GRID = (0.,) + tuple(10. ** k for k in range(-6, 3))
MODELS = ("M0_zero", "M1_training_mean", "M2_ols_surprise", "M3_ridge_loo_surprise",
          "M4_categorical_shrunk", "M5_spy_factor_transmission")
BASELINES = ("M0_zero", "M1_training_mean")
CANDIDATES = tuple(m for m in MODELS if m not in BASELINES)


class SurpriseConditionalEvidenceError(ValueError):
    """Input provenance, chronology or original funded capital is inconsistent."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise SurpriseConditionalEvidenceError(name + " must be a finite real number.")
    return float(value)


def _date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise SurpriseConditionalEvidenceError(name + " must be an ISO date.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SurpriseConditionalEvidenceError(name + " must be an ISO date.") from None
    if parsed.isoformat() != value:
        raise SurpriseConditionalEvidenceError(name + " must be a canonical ISO date.")
    return parsed


def _as_of(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SurpriseConditionalEvidenceError("analysis_as_of_utc must have a timezone.")
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SurpriseConditionalEvidenceError("Invalid timezone-aware analysis timestamp.") from None
    if observed.utcoffset() is None:
        raise SurpriseConditionalEvidenceError("Analysis timestamp must have a timezone.")
    return observed.astimezone(timezone.utc)


def _specifications(portfolios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(portfolios, (list, tuple)) or not 1 <= len(portfolios) <= 128:
        raise SurpriseConditionalEvidenceError("One to 128 explicit original portfolio specifications required.")
    seen, out = set(), []
    for p in portfolios:
        if not isinstance(p, Mapping):
            raise SurpriseConditionalEvidenceError("Portfolio must be an explicit record.")
        pid = p.get("id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise SurpriseConditionalEvidenceError("Portfolio IDs must be unique and nonempty.")
        seen.add(pid)
        notional = _real(p.get("notional_usd"), "Original portfolio notional")
        if notional <= 0:
            raise SurpriseConditionalEvidenceError("Original notional must be positive.")
        positions = p.get("positions")
        if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= 128:
            raise SurpriseConditionalEvidenceError("Original positions must be nonempty.")
        weights, names, parsed = [], set(), []
        for row in positions:
            if not isinstance(row, Mapping):
                raise SurpriseConditionalEvidenceError("Original position must be a record.")
            symbol, typ = row.get("symbol"), row.get("asset_type")
            weight = _real(row.get("weight"), "Original position weight")
            if (not isinstance(symbol, str) or not symbol or symbol != symbol.strip().upper()
                    or symbol in names or typ not in _ALLOWED_ASSETS or weight < 0 or weight > 1):
                raise SurpriseConditionalEvidenceError("Duplicate, noncanonical or unsupported original holding.")
            names.add(symbol)
            weights.append(weight)
            parsed.append({"symbol": symbol, "weight": weight, "asset_type": typ})
        if not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12):
            raise SurpriseConditionalEvidenceError("Original weights must sum to one with no normalization.")
        out.append({"id": pid, "notional_usd": notional, "positions": parsed})
    return out


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-18:
        return my, 0.
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def _ridge_loo(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Single-regressor ridge with unpenalized intercept; LOO MSE on a fixed relative grid."""
    n = len(xs)
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    cx = [x - mx for x in xs]
    cy = [y - my for y in ys]
    sxx = math.fsum(c * c for c in cx)
    sxy = math.fsum(a * b for a, b in zip(cx, cy))
    if sxx <= 1e-18:
        return my, 0., 0.
    best = None
    for relative in _RIDGE_RELATIVE_GRID:
        lam = relative * sxx
        slope = sxy / (sxx + lam)
        loo = 0.
        for x, y in zip(cx, cy):
            h = 1. / n + x * x / (sxx + lam)
            if h >= 1 - 1e-12:
                loo = math.inf
                break
            loo += ((y - slope * x) / (1 - h)) ** 2
        if best is None or loo < best[0]:
            best = (loo, lam, slope)
    _, lam, slope = best
    return my - slope * mx, slope, lam


def _branch(s: float) -> str:
    if s < -_BRANCH_THRESHOLD_PP:
        return "dovish"
    if s > _BRANCH_THRESHOLD_PP:
        return "hawkish"
    return "neutral"


def _fit_asset(train_rows: Sequence[tuple[float, float]]) -> dict[str, Any]:
    xs = [s for s, _ in train_rows]
    ys = [y for _, y in train_rows]
    mean = math.fsum(ys) / len(ys)
    a2, b2 = _ols(xs, ys)
    a3, b3, lam = _ridge_loo(xs, ys)
    branch_means = {}
    for name in ("dovish", "neutral", "hawkish"):
        members = [y for s, y in train_rows if _branch(s) == name]
        n = len(members)
        branch_means[name] = (mean if n == 0 else
                              mean + (n / (n + _SHRINKAGE_PSEUDO_COUNT)) * (math.fsum(members) / n - mean))
    return {"n_train": len(train_rows), "training_mean": mean,
            "ols_intercept": a2, "ols_slope_per_pp": b2,
            "ridge_intercept": a3, "ridge_slope_per_pp": b3, "ridge_lambda": lam,
            "categorical_shrunk_means": branch_means,
            "training_branch_counts": {name: sum(_branch(s) == name for s, _ in train_rows)
                                       for name in ("dovish", "neutral", "hawkish")}}


def _predict(fit: dict[str, Any], model: str, s: float, spy_fit: dict[str, Any] | None,
             spy_beta: float | None) -> float | None:
    if model == "M0_zero":
        return 0.
    if model == "M1_training_mean":
        return fit["training_mean"]
    if model == "M2_ols_surprise":
        return fit["ols_intercept"] + fit["ols_slope_per_pp"] * s
    if model == "M3_ridge_loo_surprise":
        return fit["ridge_intercept"] + fit["ridge_slope_per_pp"] * s
    if model == "M4_categorical_shrunk":
        return fit["categorical_shrunk_means"][_branch(s)]
    if model == "M5_spy_factor_transmission":
        if spy_fit is None or spy_beta is None:
            return None
        return spy_beta * (spy_fit["ols_intercept"] + spy_fit["ols_slope_per_pp"] * s)
    raise SurpriseConditionalEvidenceError("Unknown model " + model)


def _rmse(sq: float, n: int) -> float:
    return math.sqrt(sq / n) if n else math.nan


def _confidence_interval(values: Sequence[float]) -> list[float] | None:
    ordered = sorted(v for v in values if math.isfinite(v))
    if len(ordered) < 100:
        return None
    return [ordered[int(.025 * (len(ordered) - 1))], ordered[int(.975 * (len(ordered) - 1))]]


def _pool(rows: Sequence[dict[str, Any]], unit_ids: Sequence[str]) -> dict[str, Any]:
    """Pooled RMSE per model with a date-cluster bootstrap and per-unit wins.

    ``rows``: {unit, date, actual, predictions:{model: value or None}}.
    Rows whose prediction is None for a model are excluded for that model only
    and the count is reported, so a model is never scored on a different row
    set silently without disclosure.
    """
    by_date_model: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_unit: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for model in MODELS:
            value = row["predictions"].get(model)
            if value is None:
                continue
            err = (row["actual"] - value) ** 2
            by_date_model[row["date"]][model].append(err)
            per_unit[row["unit"]][model].append(err)
    totals = {m: math.fsum(e for d in by_date_model.values() for e in d.get(m, [])) for m in MODELS}
    counts = {m: sum(len(d.get(m, [])) for d in by_date_model.values()) for m in MODELS}
    rmse = {m: _rmse(totals[m], counts[m]) for m in MODELS}
    ordered_dates = sorted(by_date_model)
    date_sums = [{m: math.fsum(by_date_model[d].get(m, [])) for m in MODELS} for d in ordered_dates]
    date_counts = [{m: len(by_date_model[d].get(m, [])) for m in MODELS} for d in ordered_dates]
    rng = random.Random(_BOOTSTRAP_SEED)
    boot = {c: {b: [] for b in BASELINES} for c in CANDIDATES}
    if ordered_dates:
        for _ in range(_BOOTSTRAP_REPLICATES):
            chosen = [rng.randrange(len(ordered_dates)) for _ in ordered_dates]
            sums = {m: math.fsum(date_sums[j][m] for j in chosen) for m in MODELS}
            ns = {m: sum(date_counts[j][m] for j in chosen) for m in MODELS}
            for c in CANDIDATES:
                if ns[c] == 0:
                    continue
                rc = _rmse(sums[c], ns[c])
                for b in BASELINES:
                    rb = _rmse(sums[b], ns[b]) if ns[b] else math.nan
                    if rb and math.isfinite(rb) and rb > 0:
                        boot[c][b].append(1 - rc / rb)
    unit_table = {}
    for unit in unit_ids:
        if unit not in per_unit:
            continue
        unit_rmse = {m: _rmse(math.fsum(v), len(v)) for m, v in per_unit[unit].items()}
        unit_table[unit] = {"rows": len(per_unit[unit].get("M0_zero", [])), "rmse_fraction": unit_rmse,
                            "beats_both_baselines": {
                                c: (c in unit_rmse and all(b in unit_rmse for b in BASELINES)
                                    and unit_rmse[c] < min(unit_rmse[b] for b in BASELINES))
                                for c in CANDIDATES}}
    summary = {}
    for c in CANDIDATES:
        improvement = {b: (1 - rmse[c] / rmse[b]) if counts[c] and rmse[b] > 0 else None for b in BASELINES}
        ci = {b: _confidence_interval(boot[c][b]) for b in BASELINES}
        eligible_units = [u for u in unit_table if c in unit_table[u]["rmse_fraction"]]
        wins = sum(unit_table[u]["beats_both_baselines"][c] for u in eligible_units)
        passed = bool(
            counts[c] and all(v is not None and v >= _MIN_RMSE_IMPROVEMENT for v in improvement.values())
            and all(ci[b] is not None and ci[b][0] > 0 for b in BASELINES)
            and eligible_units and wins / len(eligible_units) >= _MIN_WIN_FRACTION)
        summary[c] = {"scored_rows": counts[c], "pooled_rmse_fraction": rmse[c],
                      "relative_improvement": improvement,
                      "event_date_bootstrap_95pct": ci,
                      "units_beating_both": wins, "eligible_units": len(eligible_units),
                      "research_gate_passed": passed,
                      "status": ("RESEARCH_ONLY_RETRO_HOLDOUT_SCREEN_PASS_NOT_LIVE_VALIDATED" if passed
                                 else "RESEARCH_ONLY_RETRO_HOLDOUT_NO_INCREMENTAL_EVIDENCE")}
    return {"distinct_test_event_dates": len(ordered_dates),
            "baseline_pooled_rmse_fraction": {b: rmse[b] for b in BASELINES},
            "baseline_scored_rows": {b: counts[b] for b in BASELINES},
            "candidates": summary, "per_unit": unit_table}


def evaluate_surprise_conditional_screen(
    event_observations: Sequence[Mapping[str, Any]],
    portfolios: Sequence[Mapping[str, Any]],
    *,
    train_end: str,
    analysis_as_of_utc: str,
    evidence_class: str,
    surprise_field: str = "surprise_pp",
    event_variable_id: str = "FOMC_TARGET_DECISION_MINUS_FED_FUNDS_FUTURES_EXPECTATION",
    surprise_unit: str = "percentage_points",
    spy_beta_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    spy_symbol: str = "SPY",
) -> dict[str, Any]:
    """Compare conditional-surprise return estimates against two frozen baselines.

    Each event row carries the realized surprise and a per-symbol return map;
    a symbol absent from a row (not yet listed) is simply unavailable for that
    date, never imputed. Portfolios need every funded non-cash holding on a date.
    """
    cutoff = _date(train_end, "train_end")
    now = _as_of(analysis_as_of_utc)
    if cutoff >= now.date():
        raise SurpriseConditionalEvidenceError("Training cutoff must precede the analysis date.")
    if (not isinstance(surprise_field, str) or not surprise_field.isidentifier()
            or not isinstance(event_variable_id, str) or not 8 <= len(event_variable_id) <= 120
            or not isinstance(surprise_unit, str) or not 1 <= len(surprise_unit) <= 40
            or not isinstance(evidence_class, str) or not 8 <= len(evidence_class) <= 180):
        raise SurpriseConditionalEvidenceError("The source event variable, field, units and evidence class must be explicit.")
    specs = _specifications(portfolios)
    if not isinstance(event_observations, (list, tuple)) or not 1 <= len(event_observations) <= 5000:
        raise SurpriseConditionalEvidenceError("Bounded historical event observations required.")
    if spy_beta_by_symbol is not None and not isinstance(spy_beta_by_symbol, Mapping):
        raise SurpriseConditionalEvidenceError("spy_beta_by_symbol must be a mapping.")
    symbols = {h["symbol"] for p in specs for h in p["positions"] if h["asset_type"] != "cash"}
    rows, seen = [], set()
    for raw in event_observations:
        if not isinstance(raw, Mapping):
            raise SurpriseConditionalEvidenceError("Event row must be an explicit record.")
        day = _date(raw.get("date"), "Historical event date")
        if day >= now.date() or day in seen:
            raise SurpriseConditionalEvidenceError("Future, same-day or duplicate historical event date.")
        seen.add(day)
        s = _real(raw.get(surprise_field), "As-recorded pre-announcement surprise")
        if abs(s) > 5:
            raise SurpriseConditionalEvidenceError("Surprise magnitude outside a plausible policy domain.")
        returns = raw.get("return_fraction_by_symbol")
        if not isinstance(returns, Mapping):
            raise SurpriseConditionalEvidenceError("Every event row needs a symbol-return mapping.")
        accepted = {}
        for sym in symbols:
            if sym in returns and returns[sym] is not None:
                r = _real(returns[sym], "Historical split-adjusted close return")
                if not -1 <= r <= 2:
                    raise SurpriseConditionalEvidenceError("Reported unlevered close return is outside a conservative input domain.")
                accepted[sym] = r
        rows.append({"date": day, "surprise": s, "returns": accepted})
    rows.sort(key=lambda r: r["date"])
    training = [r for r in rows if r["date"] <= cutoff]
    heldout = [r for r in rows if r["date"] > cutoff]
    fits, ineligible = {}, {}
    for sym in sorted(symbols):
        tr = [(r["surprise"], r["returns"][sym]) for r in training if sym in r["returns"]]
        te = [(r["surprise"], r["returns"][sym]) for r in heldout if sym in r["returns"]]
        if len(tr) < _MIN_TRAIN or len(te) < _MIN_TEST:
            ineligible[sym] = {"n_train": len(tr), "n_test": len(te),
                               "reason": "FEWER_THAN_MINIMUM_TRAIN_OR_TEST_EVENTS"}
            continue
        fits[sym] = {**_fit_asset(tr), "n_test": len(te)}
    spy_fit = fits.get(spy_symbol)
    betas = {}
    if spy_beta_by_symbol is not None:
        for sym, rec in spy_beta_by_symbol.items():
            if not isinstance(rec, Mapping):
                raise SurpriseConditionalEvidenceError("Invalid SPY beta record.")
            betas[sym] = _real(rec.get("beta"), "SPY beta")
    if spy_symbol in symbols:
        betas.setdefault(spy_symbol, 1.)

    def holding_prediction(sym: str, model: str, s: float) -> float | None:
        if sym not in fits:
            return None
        return _predict(fits[sym], model, s, spy_fit, betas.get(sym))

    def scored_rows(subset: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        asset_rows, portfolio_rows = [], []
        for r in subset:
            for sym in fits:
                if sym not in r["returns"]:
                    continue
                asset_rows.append({"unit": sym, "date": r["date"], "actual": r["returns"][sym],
                                   "predictions": {m: holding_prediction(sym, m, r["surprise"]) for m in MODELS}})
            for spec in specs:
                ok, actual, preds = True, 0., {m: 0. for m in MODELS}
                for h in spec["positions"]:
                    if h["asset_type"] == "cash" or h["weight"] == 0:
                        continue
                    sym = h["symbol"]
                    if sym not in fits or sym not in r["returns"]:
                        ok = False
                        break
                    actual += h["weight"] * r["returns"][sym]
                    for m in MODELS:
                        value = holding_prediction(sym, m, r["surprise"])
                        if preds[m] is None or value is None:
                            preds[m] = None
                        else:
                            preds[m] += h["weight"] * value
                if ok:
                    portfolio_rows.append({"unit": spec["id"], "date": r["date"], "actual": actual,
                                           "predictions": preds})
        return asset_rows, portfolio_rows

    asset_rows, portfolio_rows = scored_rows(heldout)
    unit_ids = sorted(fits)
    portfolio_ids = [p["id"] for p in specs]
    portfolio_counts = defaultdict(int)
    for row in portfolio_rows:
        portfolio_counts[row["unit"]] += 1
    train_counts = defaultdict(int)
    for row in scored_rows(training)[1]:
        train_counts[row["unit"]] += 1
    eligible_portfolios = [p for p in portfolio_ids
                           if portfolio_counts[p] >= _MIN_TEST and train_counts[p] >= _MIN_TRAIN]
    portfolio_rows = [row for row in portfolio_rows if row["unit"] in eligible_portfolios]
    primary_assets = _pool(asset_rows, unit_ids)
    primary_portfolios = _pool(portfolio_rows, eligible_portfolios)
    large = [r for r in heldout if abs(r["surprise"]) >= _SECONDARY_ABS_SURPRISE_PP]
    sec_assets, sec_portfolios = scored_rows(large)
    sec_portfolios = [row for row in sec_portfolios if row["unit"] in eligible_portfolios]
    secondary = {"abs_surprise_threshold_pp": _SECONDARY_ABS_SURPRISE_PP,
                 "test_event_dates": len(large), "assets": _pool(sec_assets, unit_ids),
                 "portfolios": _pool(sec_portfolios, eligible_portfolios),
                 "declared_role": "DESCRIPTIVE_SECONDARY_NOT_A_GATE"}
    return {
        "schema": SCHEMA,
        "event_question": {"variable": event_variable_id, "surprise_field": surprise_field,
                           "unit": surprise_unit,
                           "conditional_surprise_definition": "s(E) = D(E) - m, m = pre-announcement market expectation; realized branch only"},
        "analysis_as_of_utc": now.isoformat(), "train_end": cutoff.isoformat(),
        "evidence_class": evidence_class,
        "counts": {"training_events": len(training), "heldout_events": len(heldout),
                   "eligible_assets": len(fits), "ineligible_assets": ineligible,
                   "eligible_portfolios": eligible_portfolios,
                   "ineligible_portfolios": sorted(set(portfolio_ids) - set(eligible_portfolios))},
        "horizon": "PREVIOUS_SESSION_CLOSE_TO_ANNOUNCEMENT_SESSION_CLOSE",
        "return_basis": "Massive split-adjusted prices; dividends excluded",
        "method": {
            "models": list(MODELS), "baselines": list(BASELINES),
            "ridge_relative_lambda_grid": list(_RIDGE_RELATIVE_GRID),
            "categorical_branch_threshold_pp": _BRANCH_THRESHOLD_PP,
            "regularization_pseudo_count": _SHRINKAGE_PSEUDO_COUNT,
            "min_train_events": _MIN_TRAIN, "min_test_events": _MIN_TEST,
            "bootstrap_unit": "shared event date across every asset or portfolio",
            "bootstrap_seed": _BOOTSTRAP_SEED, "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "minimum_relative_rmse_improvement_each_baseline": _MIN_RMSE_IMPROVEMENT,
            "minimum_unit_win_fraction": _MIN_WIN_FRACTION,
            "spy_factor_betas_supplied": sorted(betas),
        },
        "asset_fits": fits,
        "primary": {"assets": primary_assets, "portfolios": primary_portfolios},
        "secondary_large_surprise": secondary,
        "any_candidate_passed_primary_asset_gate": any(
            v["research_gate_passed"] for v in primary_assets["candidates"].values()),
        "empirically_identified_causal_effect": False,
        "ex_ante_live_forecast_verified": False,
        "expected_event_probability_estimated": False,
        "correlation_and_event_news_confounding_remain": True,
        "promoted_to_live_conditional_impact": False,
        "reason": ("A retrospective conditional association screen on the realized branch is not a pre-issue live forecast "
                   "or a causal event effect; no number is promoted into app P&L automatically."),
    }
