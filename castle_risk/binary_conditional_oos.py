"""Historical binary-event conditional-return evidence gate (Castle R121).

This is a deliberately narrower research diagnostic than a causal event-impact
model. It fits separate YES/NO historical price-return means from training
events, regularizes both toward the training unconditional mean, and scores
only the branch realized on later, held-out event dates. It never sees
contemporaneous test-day market factors as forecast inputs.

Returned numbers are NOT automatically promoted into the live portfolio P&L:
provenance of archived market expectations, security-level future returns,
event timing, selection, and causal confounding require separate verification.
The gate records a negative result rather than tuning against held-out dates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import math
import random
from typing import Any, Mapping, Sequence


SCHEMA = "castle.r121_binary_conditional_historical_screen.v1"
_ALLOWED_OPERATORS = {"lt", "le", "gt", "ge"}
_ALLOWED_ASSETS = {"stock", "etf", "cash"}
_MIN_BRANCH_TRAIN = 8
_MIN_BRANCH_TEST = 4
_SHRINKAGE_PSEUDO_COUNT = 10
_MIN_RMSE_IMPROVEMENT = .05
_BOOTSTRAP_SEED = 121
_BOOTSTRAP_REPLICATES = 1000


class BinaryConditionalEvidenceError(ValueError):
    """Input provenance, chronology or original funded capital is inconsistent."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise BinaryConditionalEvidenceError(name + " must be a finite real number.")
    return float(value)


def _date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise BinaryConditionalEvidenceError(name + " must be an ISO date.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise BinaryConditionalEvidenceError(name + " must be an ISO date.") from None
    if parsed.isoformat() != value:
        raise BinaryConditionalEvidenceError(name + " must be a canonical ISO date.")
    return parsed


def _as_of(value: Any) -> datetime:
    if not isinstance(value, str):
        raise BinaryConditionalEvidenceError("analysis_as_of_utc must have a timezone.")
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BinaryConditionalEvidenceError("Invalid timezone-aware analysis timestamp.") from None
    if observed.utcoffset() is None:
        raise BinaryConditionalEvidenceError("Analysis timestamp must have a timezone.")
    return observed.astimezone(timezone.utc)


def _eligible_specifications(portfolios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(portfolios, (list, tuple)) or not 1 <= len(portfolios) <= 64:
        raise BinaryConditionalEvidenceError("One to 64 explicit original portfolio specifications required.")
    seen = set()
    out = []
    for p in portfolios:
        if not isinstance(p, Mapping):
            raise BinaryConditionalEvidenceError("Portfolio must be an explicit record.")
        pid = p.get("id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise BinaryConditionalEvidenceError("Portfolio IDs must be unique and nonempty.")
        seen.add(pid)
        notional = _real(p.get("notional_usd"), "Original portfolio notional")
        if notional <= 0:
            raise BinaryConditionalEvidenceError("Original notional must be positive.")
        positions = p.get("positions")
        if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= 128:
            raise BinaryConditionalEvidenceError("Original positions must be nonempty.")
        weights, names, parsed = [], set(), []
        for row in positions:
            if not isinstance(row, Mapping):
                raise BinaryConditionalEvidenceError("Original position must be a record.")
            symbol = row.get("symbol")
            typ = row.get("asset_type")
            weight = _real(row.get("weight"), "Original position weight")
            if (not isinstance(symbol, str) or not symbol or symbol != symbol.strip().upper()
                    or symbol in names or typ not in _ALLOWED_ASSETS
                    or weight < 0 or weight > 1):
                raise BinaryConditionalEvidenceError("Duplicate, noncanonical or unsupported original holding.")
            names.add(symbol)
            weights.append(weight)
            parsed.append({"symbol": symbol, "weight": weight, "asset_type": typ})
        if not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12):
            raise BinaryConditionalEvidenceError("Original weights must sum to one with no normalization.")
        out.append({"id": pid, "notional_usd": notional, "positions": parsed})
    return out


def _flag(x: float, operator: str, threshold: float) -> bool:
    return {"lt": lambda: x < threshold, "le": lambda: x <= threshold,
            "gt": lambda: x > threshold, "ge": lambda: x >= threshold}[operator]()


def _rmse_squared(errors: Sequence[float]) -> float:
    return math.sqrt(math.fsum(e * e for e in errors) / len(errors))


def _confidence_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        raise BinaryConditionalEvidenceError("No finite bootstrap scores.")
    return [ordered[int(.025 * (len(ordered) - 1))],
            ordered[int(.975 * (len(ordered) - 1))]]


def evaluate_binary_conditional_screen(
    event_observations: Sequence[Mapping[str, Any]],
    portfolios: Sequence[Mapping[str, Any]],
    *,
    threshold_jobs: float | None = None,
    operator: str,
    train_end: str,
    analysis_as_of_utc: str,
    evidence_class: str,
    threshold_value: float | None = None,
    observed_field: str = "surprise_jobs",
    event_variable_id: str = "FIRST_RELEASE_PAYROLL_MINUS_KALSHI_IMPLIED_EXPECTATION",
    observed_unit: str = "jobs",
) -> dict[str, Any]:
    """Compare binary conditional return estimates against both frozen baselines.

    Prior-to-event disclosure of expected jobs is asserted by the separately
    archived evidence, not certified by this method. A 2026 held-out date
    is retrospective *now*, not a 2026-11-02 pre-issue forecast.
    """
    cutoff = _date(train_end, "train_end")
    now = _as_of(analysis_as_of_utc)
    if cutoff >= now.date():
        raise BinaryConditionalEvidenceError("Training cutoff must precede the analysis date.")
    if (threshold_jobs is None) == (threshold_value is None):
        raise BinaryConditionalEvidenceError("Exactly one explicit threshold unit/basis is required.")
    threshold = _real(threshold_jobs if threshold_value is None else threshold_value, "Event threshold")
    if (not isinstance(observed_field, str) or not observed_field.isidentifier()
            or len(observed_field) > 64 or not isinstance(event_variable_id, str)
            or not 8 <= len(event_variable_id) <= 120
            or not isinstance(observed_unit, str) or not 1 <= len(observed_unit) <= 40):
        raise BinaryConditionalEvidenceError("The source event variable, field and units must be explicit.")
    if operator not in _ALLOWED_OPERATORS:
        raise BinaryConditionalEvidenceError("Event predicate operator is unsupported.")
    if not isinstance(evidence_class, str) or not 8 <= len(evidence_class) <= 180:
        raise BinaryConditionalEvidenceError("Explicit historical evidence class required.")
    specs = _eligible_specifications(portfolios)
    if not isinstance(event_observations, (list, tuple)) or len(event_observations) > 5000:
        raise BinaryConditionalEvidenceError("Bounded historical event observations required.")
    rows, dates = [], set()
    symbols = {h["symbol"] for p in specs for h in p["positions"] if h["asset_type"] != "cash"}
    for raw in event_observations:
        if not isinstance(raw, Mapping):
            raise BinaryConditionalEvidenceError("Event row must be an explicit record.")
        day = _date(raw.get("date"), "Historical event date")
        if day >= now.date() or day in dates:
            raise BinaryConditionalEvidenceError("Future, same-day or duplicate historical event date.")
        dates.add(day)
        x = _real(raw.get(observed_field), "As-recorded historical event value")
        returns = raw.get("return_fraction_by_symbol")
        if not isinstance(returns, Mapping) or any(s not in returns for s in symbols):
            raise BinaryConditionalEvidenceError("Every dated historical return must cover every funded original security.")
        accepted = {s: _real(returns[s], "Historical split-adjusted close return") for s in symbols}
        if any(not -1 <= r <= 2 for r in accepted.values()):
            raise BinaryConditionalEvidenceError("Reported unlevered close return is outside a conservative input domain.")
        rows.append({"date": day, "yes": _flag(x, operator, threshold), "returns": accepted})
    rows.sort(key=lambda row: row["date"])
    training = [r for r in rows if r["date"] <= cutoff]
    heldout = [r for r in rows if r["date"] > cutoff]
    train_yes = sum(r["yes"] for r in training)
    test_yes = sum(r["yes"] for r in heldout)
    counts = {
        "training_events": len(training), "training_yes": train_yes,
        "training_no": len(training) - train_yes,
        "heldout_events": len(heldout), "heldout_yes": test_yes,
        "heldout_no": len(heldout) - test_yes,
    }
    result = {
        "schema": SCHEMA,
        "event_question": {"variable": event_variable_id,
                           "observed_field": observed_field, "unit": observed_unit,
                           "operator": operator, "threshold_value": threshold,
                           "threshold_jobs": threshold if threshold_jobs is not None else None,
                           "different_from_survey_consensus": observed_field == "surprise_jobs"},
        "analysis_as_of_utc": now.isoformat(),
        "train_end": cutoff.isoformat(),
        "evidence_class": evidence_class, "counts": counts,
        "horizon": "PREVIOUS_SESSION_CLOSE_TO_RELEASE_SESSION_CLOSE",
        "return_basis": "Massive split-adjusted prices; dividends excluded",
        "method": {
            "conditional_center": "mean_all_train + n_branch/(n_branch+10)*(mean_branch_train-mean_all_train)",
            "two_frozen_baselines": ["zero_return", "unconditional_training_mean"],
            "min_train_per_branch": _MIN_BRANCH_TRAIN,
            "min_heldout_per_branch": _MIN_BRANCH_TEST,
            "regularization_pseudo_count": _SHRINKAGE_PSEUDO_COUNT,
            "bootstrap_unit": "shared event date across every portfolio",
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "minimum_relative_rmse_improvement_each_baseline": _MIN_RMSE_IMPROVEMENT,
            "minimum_portfolio_win_fraction": .75,
        },
        "empirically_identified_causal_effect": False,
        "ex_ante_live_forecast_verified": False,
        "expected_event_probability_estimated": False,
        "correlation_and_event_news_confounding_remain": True,
        "historical_prices_and_expectations_independently_reauthenticated_by_this_screen": False,
        "promoted_to_live_conditional_impact": False,
    }
    if (min(counts["training_yes"], counts["training_no"]) < _MIN_BRANCH_TRAIN
            or min(counts["heldout_yes"], counts["heldout_no"]) < _MIN_BRANCH_TEST):
        return {
            **result, "status": "ABSTAIN_INSUFFICIENT_EVENT_BRANCH_SUPPORT",
            "reason": "Both training and later test must contain enough YES and NO events; event rarity is missing evidence, not a zero return.",
            "conditional_portfolio_centers_usd": None,
            "heldout_metrics": None,
        }
    scored = []
    errors_by_date: dict[date, list[tuple[float, float, float]]] = defaultdict(list)
    for spec in specs:
        no_event, yes_event, unconditional = {}, {}, {}
        for holding in spec["positions"]:
            symbol = holding["symbol"]
            if holding["asset_type"] == "cash":
                no_event[symbol] = yes_event[symbol] = unconditional[symbol] = 0.
                continue
            tr = [r["returns"][symbol] for r in training]
            mu = math.fsum(tr) / len(tr)
            unconditional[symbol] = mu
            for flag, to in ((True, yes_event), (False, no_event)):
                branch = [r["returns"][symbol] for r in training if r["yes"] == flag]
                n = len(branch)
                to[symbol] = mu + (n / (n + _SHRINKAGE_PSEUDO_COUNT)) * (math.fsum(branch) / n - mu)
        yes_pct = 100 * math.fsum(h["weight"] * yes_event[h["symbol"]] for h in spec["positions"])
        no_pct = 100 * math.fsum(h["weight"] * no_event[h["symbol"]] for h in spec["positions"])
        e_candidate, e_mean, e_zero = [], [], []
        for sample in heldout:
            observed = math.fsum(h["weight"] * (0 if h["asset_type"] == "cash" else
                                 sample["returns"][h["symbol"]]) for h in spec["positions"])
            predicted = math.fsum(h["weight"] * (yes_event if sample["yes"] else no_event)[h["symbol"]]
                                  for h in spec["positions"])
            base_mean = math.fsum(h["weight"] * unconditional[h["symbol"]] for h in spec["positions"])
            e_candidate.append(observed - predicted)
            e_mean.append(observed - base_mean)
            e_zero.append(observed)
            errors_by_date[sample["date"]].append(((observed - predicted)**2,
                                                  (observed - base_mean)**2, observed**2))
        candidate = _rmse_squared(e_candidate)
        mean = _rmse_squared(e_mean)
        zero = _rmse_squared(e_zero)
        scored.append({
            "portfolio_id": spec["id"], "original_notional_usd": spec["notional_usd"],
            "conditional_train_only_yes_price_return_pct": yes_pct,
            "conditional_train_only_no_price_return_pct": no_pct,
            "conditional_train_only_yes_pnl_usd": spec["notional_usd"] * yes_pct / 100,
            "conditional_train_only_no_pnl_usd": spec["notional_usd"] * no_pct / 100,
            "heldout_rmse_fraction": {"candidate": candidate, "unconditional_mean": mean, "zero": zero},
            "beats_both_baselines": candidate < min(mean, zero),
        })
    total = {k: 0. for k in ("candidate", "unconditional_mean", "zero")}
    for records in errors_by_date.values():
        for a, b, c in records:
            total["candidate"] += a
            total["unconditional_mean"] += b
            total["zero"] += c
    observations = math.fsum(len(v) for v in errors_by_date.values())
    rmse = {k: math.sqrt(v / observations) for k, v in total.items()}
    if min(rmse.values()) <= 0:
        return {**result, "status": "ABSTAIN_DEGENERATE_BENCHMARK", "heldout_metrics": None,
                "conditional_portfolio_centers_usd": None}
    improvement = {k: 1 - rmse["candidate"] / rmse[k] for k in ("unconditional_mean", "zero")}
    ordered_dates = sorted(errors_by_date)
    date_totals = [[math.fsum(e[i] for e in errors_by_date[d]) for i in range(3)] for d in ordered_dates]
    date_counts = [len(errors_by_date[d]) for d in ordered_dates]
    rng = random.Random(_BOOTSTRAP_SEED)
    bootstrap = {k: [] for k in ("unconditional_mean", "zero")}
    for _ in range(_BOOTSTRAP_REPLICATES):
        chosen = [rng.randrange(len(ordered_dates)) for _ in ordered_dates]
        sums = [math.fsum(date_totals[j][i] for j in chosen) for i in range(3)]
        denom = math.fsum(date_counts[j] for j in chosen)
        rmses = [math.sqrt(s / denom) for s in sums]
        if rmses[1] > 0 and rmses[2] > 0:
            bootstrap["unconditional_mean"].append(1 - rmses[0] / rmses[1])
            bootstrap["zero"].append(1 - rmses[0] / rmses[2])
    ci = {k: _confidence_interval(v) for k, v in bootstrap.items()}
    winning = sum(s["beats_both_baselines"] for s in scored)
    pass_gate = (
        all(value >= _MIN_RMSE_IMPROVEMENT for value in improvement.values())
        and all(bounds[0] > 0 for bounds in ci.values())
        and winning / len(scored) >= .75
    )
    return {
        **result, "status": ("RESEARCH_ONLY_RETRO_HOLDOUT_SCREEN_PASS_NOT_LIVE_VALIDATED"
                              if pass_gate else "RESEARCH_ONLY_RETRO_HOLDOUT_NO_INCREMENTAL_EVIDENCE"),
        "research_gate_passed": pass_gate,
        "conditional_portfolio_centers_usd": None,
        "research_only_training_branch_centers": scored,
        "heldout_metrics": {
            "distinct_later_event_dates": len(ordered_dates),
            "portfolio_event_rows": int(observations),
            "pooled_rmse_fraction": rmse,
            "relative_improvement": improvement,
            "event_date_bootstrap_95pct": ci,
            "portfolios_beating_both": winning,
            "total_portfolios": len(scored),
        },
        "reason": ("A retrospective conditional association screen is not a pre-issue live forecast or a causal event effect;"
                   "no number is promoted into app P&L automatically."),
    }


def screen_archived_r107_payroll(*, analysis_as_of_utc: str = "2026-11-02T00:00:00Z",
                                 threshold_jobs: int = -100_000,
                                 operator: str = "le") -> dict[str, Any]:
    """Replay the already archived seven-ETF payroll source in frozen R107 split.

    Source receipt hashes and 31 dated ETF returns are recorded, but this
    routine does not independently authenticate raw Kalshi quote timestamps.
    """
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "r107_payroll_context.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "castle.empirical_payroll_context.v1":
        raise BinaryConditionalEvidenceError("Wrong archived empirical source schema.")
    study = data.get("event_study", {})
    if study.get("event_dates_n") != len(study.get("observations", [])) or study.get("close_to_close") is not True:
        raise BinaryConditionalEvidenceError("Archived event and price-return basis mismatch.")
    symbols = study.get("market_symbols")
    provenance = data.get("provenance")
    if (not isinstance(symbols, list) or len(symbols) != 7 or not isinstance(provenance, Mapping)
            or set(symbols) - set(provenance)):
        raise BinaryConditionalEvidenceError("Seven source-bound ETFs and receipts required.")
    for symbol in symbols:
        record = provenance[symbol]
        if (not isinstance(record, Mapping) or record.get("provider") != "Massive"
                or not isinstance(record.get("raw_response_sha256"), str)
                or len(record["raw_response_sha256"]) != 64):
            raise BinaryConditionalEvidenceError("Archived provider receipt fingerprint absent.")
    one_etf_portfolios = [
        {"id": symbol, "notional_usd": 100_000, "positions": [
            {"symbol": symbol, "weight": 1., "asset_type": "etf"}]}
        for symbol in symbols
    ]
    return evaluate_binary_conditional_screen(
        study["observations"], one_etf_portfolios,
        threshold_jobs=threshold_jobs, operator=operator, train_end="2025-12-31",
        analysis_as_of_utc=analysis_as_of_utc,
        evidence_class="R107_KALSHI_DERIVED_NOT_SURVEY_CONSENSUS_MASSIVE_SPLIT_ADJUSTED",
    )
