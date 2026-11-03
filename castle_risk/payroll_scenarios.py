"""US jobs report (nonfarm payrolls): scheduled releases, Kalshi payroll bands, measured surprise response.

Every Employment Situation release inside the horizon is listed from the BLS schedule, like FOMC and CPI.
The Kalshi KXPAYROLLS ladder ("above X jobs") becomes disjoint payroll bands with quote-compatible Q
weights, through the same linear constraints as the CPI ladder. Each band's P&L is the move not already in
prices:

    PnL_b = notional * slope * (x_b - E_Q[x]) / 100,000

slope is the portfolio's own close-to-close return on past release days per 100,000 jobs of surprise,
where the surprise is the first-release value minus the Q-expected value read from that release's own
Kalshi ladder before 08:20 ET (PAYROLL_TAPE). Asset prices are assumed to embed Q, so the Q-weighted P&L
is zero. No network, clock read or model call happens here.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import t as student_t

from .event_distributions import (EventDistributionError, _attach_bounds, _base, _current_quotes,
                                  _relax_if_incoherent, coobserved_event_markets)
from .empirical_event_impact import _fit_association
from .scheduled_event_timing import scheduled_event_window

SCHEMA = "castle.option1.payroll_scenarios.v1"
FAMILY = "US_PAYROLLS"
SERIES = "KXPAYROLLS"
UNIT = 100_000.0
DEFAULT_DRAWS = 20000
DEFAULT_SEED = 20261024
MIN_EVENTS = 12
MIN_TRAIN_EVENTS = 8
MIN_HOLDOUT_EVENTS = 4

# BLS Employment Situation schedule, transcribed from the official page on 2026-10-24.
BLS_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"
BLS_SCHEDULE = {
    "2026-10": "2026-11-06", "2026-11": "2026-12-04", "2026-12": "2027-01-08", "2027-01": "2027-02-05",
    "2027-02": "2027-03-05", "2027-03": "2027-04-02", "2027-04": "2027-05-07", "2027-05": "2027-06-04",
    "2027-06": "2027-07-02", "2027-07": "2027-08-06", "2027-08": "2027-09-03", "2027-09": "2027-10-08",
    "2027-10": "2027-11-05", "2027-11": "2027-12-03",
}
# The open tails have no bound in the contracts. Their representatives are declared: one 50,000-job
# step beyond the last strike, with a range of 200,000 jobs beyond it.
TAIL_STEP = 50_000.0
TAIL_RANGE = 200_000.0
TAPE_PATH = Path(__file__).resolve().parents[1] / "data" / "payrolls" / "payroll_surprise_tape.json"

_RULES = re.compile(r"if the increase in total non-farm payroll employment is above (-?[\d,]+(?:\.\d+)?)\s+as reported by "
                    r"the bureau of labor statistics(?:'|’)? monthly employment situation report for the month of "
                    r"([a-z]+) (\d{4})", re.I)
_ET = ZoneInfo("America/New_York")


def _jobs(value):
    return f"{value / 1000:+,.0f}k".replace("+-", "-")


def scheduled_payroll_events(proposed, evaluated_at, *, horizon_days=90):
    """Every jobs report inside the horizon from the BLS schedule, whatever the model proposed."""
    present = {(event.get("family"), event.get("event_date")) for event in proposed}
    added = []
    for month, day in sorted(BLS_SCHEDULE.items(), key=lambda item: item[1]):
        timing = scheduled_event_window(day, evaluated_at, horizon_days=horizon_days,
                                        release_time_et=time(8, 30), clock_source_url=BLS_SCHEDULE_URL)
        if not timing["eligible"] or (FAMILY, day) in present:
            continue
        reference = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
        added.append({
            "event_id": "payrolls-" + day + "-scheduled",
            "title": f"US jobs report ({reference} payrolls) on {day} (scheduled)", "event_date": day, "family": FAMILY,
            "reference_month": month,
            "question_we_care_about": f"How does the portfolio move across the settlement outcomes of the {reference} "
                                      f"nonfarm payrolls first release on {day}?",
            "definition_origin": "OFFICIAL_CALENDAR_SCHEDULED_MACRO",
            "calendar_source": {"url": BLS_SCHEDULE_URL,
                                "summary": f"BLS Employment Situation for {reference}, released {day} at 08:30 ET"},
            "schedule_window": timing,
            "matching_comparisons": [],
        })
    return added


def _parse(market):
    raw = market.metadata.get("raw") or {}
    match = _RULES.search(market.rules_primary or "")
    if match is None:
        raise EventDistributionError("Payroll rules are outside the reviewed grammar: " + market.ticker)
    strike = float(match.group(1).replace(",", ""))
    floor = raw.get("floor_strike")
    if (raw.get("strike_type") != "greater" or isinstance(floor, bool) or not isinstance(floor, (int, float))
            or float(floor) != strike or raw.get("cap_strike") is not None):
        raise EventDistributionError("Payroll structured strike contradicts the rules: " + market.ticker)
    month = datetime.strptime(match.group(2).title() + " " + match.group(3), "%B %Y")
    return strike, month.strftime("%Y-%m")


def payroll_distribution(markets, *, as_of, max_age_seconds=300):
    """Disjoint payroll bands from one same-snapshot Kalshi ladder; fail closed otherwise."""
    cohort = coobserved_event_markets(markets)
    now, quotes = _current_quotes(cohort, as_of, max_age_seconds)
    if any(market.series_ticker != SERIES for market in cohort):
        raise EventDistributionError("Only the reviewed KXPAYROLLS series defines payroll bands.")
    parsed = sorted((_parse(market) + (market,) for market in cohort), key=lambda row: row[0])
    months = {month for _, month, _ in parsed}
    if len(months) != 1:
        raise EventDistributionError("The ladder mixes reference months.")
    month = months.pop()
    strikes = [strike for strike, _, _ in parsed]
    if len(set(strikes)) != len(strikes):
        raise EventDistributionError("Duplicate payroll strike.")
    release = datetime.fromisoformat(cohort[0].close_time.replace("Z", "+00:00")).astimezone(_ET).date().isoformat()
    scheduled = BLS_SCHEDULE.get(month)
    if scheduled is not None and scheduled != release:
        raise EventDistributionError("The ladder's close date contradicts the BLS schedule for " + month + ".")
    n = len(strikes) + 1
    atoms = []
    for index in range(n):
        lower = None if index == 0 else strikes[index - 1]
        upper = None if index == n - 1 else strikes[index]
        if lower is None:
            label, point, span = f"Payrolls {_jobs(upper)} or less", upper - TAIL_STEP, [upper - TAIL_RANGE, upper]
        elif upper is None:
            label, point, span = f"Payrolls above {_jobs(lower)}", lower + TAIL_STEP, [lower, lower + TAIL_RANGE]
        else:
            label, point, span = f"Payrolls above {_jobs(lower)} to {_jobs(upper)}", (lower + upper) / 2, [lower, upper]
        atoms.append({"outcome_id": f"PAYROLLS_BAND_{index}", "label": label, "unit": "jobs, first release",
                      "strict_threshold_bin": {"lower": lower, "upper": upper, "lower_inclusive": False, "upper_inclusive": True},
                      "value": point, "value_interval": span,
                      "source_contract_ids": [parsed[i][2].ticker for i in
                                              ([0] if index == 0 else [len(parsed) - 1] if index == n - 1
                                               else [index - 1, index])],
                      "representative_basis": ("BAND_MIDPOINT_UNIFORM_WITHIN_BAND" if lower is not None and upper is not None
                                               else "DECLARED_TAIL_ONE_50K_STEP_RANGE_200K")})
    constraints = {"atom_ids": [atom["outcome_id"] for atom in atoms], "A_ub": [], "b_ub": [],
                   "A_eq": [[1.0] * n], "b_eq": [1.0], "bounds": [[0.0, 1.0]] * n}
    quoted = {quote["ticker"]: quote for quote in quotes}
    for index, (_, _, market) in enumerate(parsed):
        row = [0.0] * (index + 1) + [1.0] * (n - index - 1)
        constraints["A_ub"].extend([row, [-value for value in row]])
        constraints["b_ub"].extend([quoted[market.ticker]["upper"], -quoted[market.ticker]["lower"]])
    relaxation = _relax_if_incoherent(constraints)
    reference = _attach_bounds(atoms, constraints)
    result = _base(cohort, now)
    result.update(family=FAMILY, metric_identity="BLS_EMPLOYMENT_SITUATION_TOTAL_NONFARM_PAYROLL_CHANGE_FIRST_RELEASE",
                  atoms=atoms, outcomes=atoms, outcome_groups=[], constraints=constraints, q_reference=reference,
                  source_quote_constraints=quotes, quote_relaxation=relaxation, reference_month=month,
                  event_date=release, release_date=release,
                  release_date_status=("SOURCE_CLOSE_DATE_MATCHES_BLS_SCHEDULE" if scheduled else "SOURCE_CLOSE_DATE_ONLY"),
                  probability_question="Distribution of the first-release change in total nonfarm payrolls across disjoint bands, including both tails.",
                  event_level_probability=None)
    result["assumptions"].extend([
        "Every strike settles on the same first-release BLS change in total nonfarm payrolls for the stated month.",
        "Both tails are open in the contracts; their representative values and ranges are declared, not implied by quotes.",
    ])
    return result


def payroll_distributions(markets_by_ticker, *, as_of):
    """One result per KXPAYROLLS event; failures stay visible."""
    grouped = {}
    for market in (markets_by_ticker or {}).values():
        if market.series_ticker == SERIES:
            grouped.setdefault(market.event_ticker, []).append(market)
    result = {}
    for event, markets in sorted(grouped.items()):
        try:
            result[event] = payroll_distribution(markets[:40], as_of=as_of)
        except (EventDistributionError, ValueError, TypeError, KeyError, OSError) as exc:
            result[event] = {"status": "DISTRIBUTION_UNAVAILABLE", "event_ticker": event, "family": FAMILY,
                             "reason": str(exc), "atoms": [], "physical_probability_P": None,
                             "event_date": datetime.fromisoformat(markets[0].close_time.replace("Z", "+00:00"))
                             .astimezone(_ET).date().isoformat() if markets[0].close_time else None}
    return result


def load_tape(path=TAPE_PATH):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _closes(rows):
    closes, duplicate_days = {}, set()
    for row in rows or []:
        try:
            day = date.fromisoformat(str(row["date"])[:10])
            value = float(row["close"])
            if isinstance(row["close"], bool) or not math.isfinite(value) or value <= 0:
                continue
            if day in closes:
                duplicate_days.add(day)
            closes[day] = value
        except (KeyError, TypeError, ValueError):
            continue
    return {day: value for day, value in closes.items() if day not in duplicate_days}


def _release_window(closes, day):
    """Observed start, release end and return; no exchange calendar is inferred."""
    if closes.get(day, 0) <= 0:
        return None
    earlier = [d for d in closes if day - timedelta(days=5) <= d < day]
    if not earlier or closes[max(earlier)] <= 0:
        return None
    start = max(earlier)
    return start, day, closes[day] / closes[start] - 1


def _release_return(closes, day):
    window = _release_window(closes, day)
    return None if window is None else window[2]


def _ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    if n <= 2 or not np.all(np.isfinite(design)) or not np.all(np.isfinite(y)) or np.linalg.matrix_rank(design) != 2:
        raise ValueError("A finite varying payroll surprise sample is required")
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    meat = design.T @ (design * (resid ** 2)[:, None])
    cov = bread @ meat @ bread * n / (n - 2)
    return float(coef[1]), float(math.sqrt(max(0.0, cov[1, 1])))


def _frozen_payroll_fit(sample):
    """Freeze the first two thirds; the final third never changes its coefficient.

    The split is a fixed sample-order rule, not selected by predictive scores.
    Per-position fits and the portfolio fit use the same complete sessions, so
    their weighted slopes/contributions agree, and portfolio covariance is
    measured directly rather than assuming independent securities.
    """
    if len(sample) < MIN_EVENTS:
        return None
    sample = sorted(sample)
    split = (2 * len(sample)) // 3
    rows = [(date.fromisoformat(day), x, y) for day, x, y in sample]
    fitted = _fit_association(rows, rows[split - 1][0], MIN_TRAIN_EVENTS, MIN_HOLDOUT_EVENTS)
    if fitted["beta"] is None:
        return None
    critical = float(student_t.ppf(.975, fitted["n_train"] - 2))
    se = (fitted["beta_interval"][1] - fitted["beta_interval"][0]) / (2 * critical)
    late = sample[split:]
    try:
        late_slope = _ols([s[1] for s in late], [s[2] for s in late])[0] if len(late) >= 6 else None
    except (ValueError, np.linalg.LinAlgError):
        late_slope = None
    full_slope, full_se = _ols([s[1] for s in sample], [s[2] for s in sample])
    specification = {
        "schema": "castle.payroll_frozen_response_record.v1",
        "fit_rule": "FIRST_TWO_THIRDS_OF_THIS_RECORDED_COMMON_SAMPLE",
        "training_event_ids": [row[0] for row in sample[:split]],
        "training_rows_sha256": hashlib.sha256(json.dumps(sample[:split],
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "slope": fitted["beta"], "intercept": fitted["intercept"], "se": se,
        "slope_unit": "fractional_release_session_return_per_100000_jobs_surprise",
    }
    specification["model_fingerprint_sha256"] = hashlib.sha256(json.dumps(
        specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    specification.update(
        diagnostic_event_ids=[row[0] for row in late],
        evaluation_scope="RETROSPECTIVE_CHRONOLOGICAL_ASSOCIATION_DIAGNOSTIC",
        fitted_once_per_record_not_persistently_frozen_across_new_runs=True,
        fresh_unseen_validation_established=False,
        diagnostic_estimand="Intercept plus slope times realized surprise versus full release-session return",
        scenario_estimand="Incremental slope times settlement state minus current source-Q expectation; no intercept",
    )
    return {
        "status": "ESTIMATED", "slope": fitted["beta"], "se": se,
        "intercept": fitted["intercept"], "slope_interval_95": fitted["beta_interval"],
        "n": len(sample), "n_train": fitted["n_train"], "n_holdout": fitted["n_holdout"],
        "train_end": fitted["train_end"], "training_span": fitted["training_span"],
        "holdout_span": fitted["holdout_span"], "transfer_check": fitted["transfer_check"],
        "first_release_date": sample[0][0], "last_release_date": sample[-1][0],
        "slope_first_two_thirds": fitted["beta"], "slope_last_third": late_slope,
        "same_sign_in_both_periods": None if late_slope is None else bool(fitted["beta"] * late_slope > 0),
        "all_sample_fit_for_comparison_only": {"slope": full_slope, "se": full_se, "n": len(sample)},
        "coefficient_source": "FROZEN_FIRST_TWO_THIRDS_NO_HOLDOUT_REFIT_OR_MODEL_SELECTION",
        "frozen_response_record": specification,
        "forward_prediction_certified": False,
    }


def payroll_response_study(portfolio, prices_by_symbol, tape, *, analysis_as_of):
    """Training-only response plus a chronological later-period diagnostic.

    Frozen within this recorded evaluation, not automatically a fresh blind
    test or a persistently frozen model across subsequent acquisitions.
    """
    if analysis_as_of.utcoffset() is None:
        raise ValueError("Payroll analysis_as_of must be timezone-aware")
    raw_events = (tape or {}).get("events", [])
    dates = Counter(str(row.get("release_date")) for row in raw_events if isinstance(row, dict))
    events, rejected = [], []
    for row in raw_events:
        try:
            day = date.fromisoformat(str(row["release_date"]))
            surprise = row.get("surprise")
            if dates[day.isoformat()] != 1:
                raise ValueError("Duplicate release date; every copy is excluded")
            if day >= analysis_as_of.astimezone(_ET).date():
                raise ValueError("Release session not completed before analysis")
            if (isinstance(surprise, bool) or not isinstance(surprise, (int, float))
                    or not math.isfinite(surprise) or row.get("excluded_reason")):
                raise ValueError("Missing, nonfinite or explicitly excluded surprise")
            events.append({**row, "surprise": float(surprise)})
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"release_date": row.get("release_date") if isinstance(row, dict) else None,
                             "reason": str(exc)})
    closes = {symbol: _closes(rows) for symbol, rows in (prices_by_symbol or {}).items()}
    held = [p for p in portfolio.positions if p.asset_type != "cash" and p.weight != 0]
    positions = {p.symbol: {"slope": 0.0, "se": 0.0, "n": 0,
                           "status": "KNOWN_NOMINAL_USD_CASH_ZERO",
                           "transfer_check": {"status": "NOT_APPLICABLE"},
                           "basis": "NOMINAL_CASH_VALUE_HELD_FIXED_OVER_RELEASE_WINDOW"}
                 for p in portfolio.positions if p.asset_type == "cash" or p.weight == 0}
    rows_by_symbol, windows_by_symbol = {}, {}
    for position in held:
        windows = {row["release_date"]: _release_window(closes.get(position.symbol, {}),
                                                       date.fromisoformat(row["release_date"]))
                   for row in events}
        windows_by_symbol[position.symbol] = windows
        rows = [(row["release_date"], row["surprise"] / UNIT, windows[row["release_date"]][2])
                for row in events if windows[row["release_date"]] is not None]
        rows_by_symbol[position.symbol] = {day: (x, y) for day, x, y in rows}
        fit = _frozen_payroll_fit(rows)
        positions[position.symbol] = fit or {
            "slope": None, "se": None, "n": len(rows),
            "status": "TOO_FEW_RELEASE_DAYS" if len(rows) < MIN_EVENTS else "UNIDENTIFIED_SURPRISE_RESPONSE",
            "transfer_check": {"status": "NOT_TESTED", "reason": "A finite chronological train/holdout fit is unavailable"},
        }
    known = [p for p in held if positions[p.symbol]["status"] == "ESTIMATED"]
    sample, admitted_windows, excluded_windows = [], [], []
    for row in sorted(events, key=lambda r: r["release_date"]):
        if not known:
            continue
        windows = [windows_by_symbol[p.symbol][row["release_date"]] for p in known]
        observed = [{"symbol": p.symbol, "start_date": window[0].isoformat() if window else None,
                     "end_date": window[1].isoformat() if window else None}
                    for p, window in zip(known, windows)]
        if any(window is None for window in windows):
            excluded_windows.append({"release_date": row["release_date"],
                                     "reason": "MISSING_OBSERVED_CLOSE_WINDOW", "positions": observed})
            continue
        if len({window[:2] for window in windows}) != 1:
            # Matching only the release end can combine a one-session equity
            # return with a different holding's missing-session return. Never
            # synthesize an earlier close or silently realign to a longer span.
            excluded_windows.append({"release_date": row["release_date"],
                                     "reason": "MISMATCHED_OBSERVED_RETURN_WINDOWS", "positions": observed})
            continue
        start, end = windows[0][:2]
        admitted_windows.append({"start_date": start.isoformat(), "end_date": end.isoformat()})
        sample.append((row["release_date"], row["surprise"] / UNIT,
                       math.fsum(p.weight * window[2] for p, window in zip(known, windows))))
    alignment = {
        "policy": "IDENTICAL_OBSERVED_START_AND_END_DATES_ACROSS_COVERED_NONCASH_HOLDINGS",
        "covered_symbols": [p.symbol for p in known], "admitted_windows": admitted_windows,
        "excluded_windows": excluded_windows, "maximum_observed_close_gap_days": 5,
        "exchange_calendar_authenticated": False,
        "limitation": "Identical observed date windows do not independently prove the prior date was the exchange's immediately preceding session.",
    }
    portfolio_fit = None
    if len(sample) >= MIN_EVENTS:
        portfolio_fit = _frozen_payroll_fit(sample)
        if portfolio_fit is not None:
            portfolio_fit["covered_weight"] = math.fsum(p.weight for p in known)
            portfolio_fit["return_alignment"] = deepcopy(alignment)
            for p in known:
                own = [(day, *rows_by_symbol[p.symbol][day]) for day, _, _ in sample]
                positions[p.symbol] = _frozen_payroll_fit(own)
            reconstructed = math.fsum(p.weight * positions[p.symbol]["slope"] for p in known)
            if not math.isclose(reconstructed, portfolio_fit["slope"], rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("Common-sample payroll position slopes do not sum to the portfolio slope")
    return {"positions": positions, "portfolio": portfolio_fit, "events_in_tape": len(events),
            "rejected_events": rejected, "return_alignment": alignment,
            "estimation_policy": "FROZEN_FIRST_TWO_THIRDS_WITH_UNTOUCHED_LAST_THIRD_CHRONOLOGICAL_HOLDOUT",
            "minimum_samples": {"total": MIN_EVENTS, "train": MIN_TRAIN_EVENTS, "holdout": MIN_HOLDOUT_EVENTS},
            "unit": "fractional close-to-close return on the release session per 100,000 jobs of surprise",
            "estimator": "Training-only OLS with HC3 standard errors and Student-t parameter uncertainty",
            "tape_source": deepcopy((tape or {}).get("source"))}


def _mean_range(atoms, constraints):
    from .event_distributions import distribution_linear_bounds
    return distribution_linear_bounds(constraints, [atom["value"] for atom in atoms])["interval"]


def _payroll_reliability(portfolio, study):
    diagnostics = []
    failed, untested = [], []
    for position in portfolio.positions:
        fit = study.get("positions", {}).get(position.symbol) or {}
        check = deepcopy(fit.get("transfer_check") or {"status": "NOT_TESTED"})
        if position.asset_type == "cash" or position.weight == 0:
            check = {"status": "NOT_APPLICABLE"}
        elif check.get("status") == "FAIL":
            failed.append(position.symbol)
        elif check.get("status") != "PASS":
            untested.append(position.symbol)
        diagnostics.append({"symbol": position.symbol, "weight": position.weight,
                            "training_span": deepcopy(fit.get("training_span")),
                            "holdout_span": deepcopy(fit.get("holdout_span")),
                            "transfer_check": check})
    portfolio_check = deepcopy((study.get("portfolio") or {}).get("transfer_check") or {"status": "NOT_TESTED"})
    all_cash = all(p.asset_type == "cash" or p.weight == 0 for p in portfolio.positions)
    if all_cash:
        status, portfolio_check = "NOT_APPLICABLE", {"status": "NOT_APPLICABLE"}
    elif failed or portfolio_check.get("status") == "FAIL":
        status = "FAIL"
    elif untested or portfolio_check.get("status") != "PASS":
        status = "NOT_TESTED"
    else:
        status = "PASS"
    return {"status": status, "portfolio_transfer_check": portfolio_check,
            "scenario_transfer_failed_symbols": failed, "scenario_transfer_untested_symbols": untested,
            "positions": diagnostics, "forward_prediction_certified": False,
            "scope": "Response fitted to the first two thirds of this recorded sample, checked on its chronological final third against zero-return and frozen training-mean baselines. Previously inspected dates are retrospective diagnostics, not a fresh blind holdout. Both portfolio and noncash holdings must pass for overall PASS. The diagnostic scores total release-session return including an intercept; the scenario displays incremental repricing without an intercept. Observed surprise is contemporaneous, not an advance predictor; neither PASS nor these model ranges certify future P&L."}


def build_payroll_table(portfolio, distribution, study, *, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """Not-yet-priced P&L for every payroll band from the portfolio's measured surprise response."""
    fit = study.get("portfolio")
    reliability = _payroll_reliability(portfolio, study)
    if all(p.asset_type == "cash" or p.weight == 0 for p in portfolio.positions):
        # A fixed nominal USD cash balance has no price component to estimate.
        # This result must not be generalized to bonds, FX cash or future carry.
        atoms = distribution["atoms"]
        if not atoms or abs(math.fsum(a["q_reference"] for a in atoms) - 1) > 1e-6:
            raise ValueError("An intact source pricing distribution is required")
        mean = math.fsum(a["q_reference"] * a["value"] for a in atoms)
        rows = [{"outcome_id": a["outcome_id"], "label": a["label"],
                 "q_interval": list(a["q_interval"]), "q_reference": a["q_reference"],
                 "state": a["value"], "state_interval": list(a["value_interval"]),
                 "surprise": a["value"] - mean, "pnl_if_outcome_usd": 0.0,
                 "pnl_model_range_5_95_usd": [0.0, 0.0],
                 "model_5_95_range_excludes_zero": False,
                 "source_contract_ids": deepcopy(a.get("source_contract_ids", [])),
                 "position_contributions": [{"symbol": p.symbol, "channel": "NONE", "pnl_usd": 0.0}
                                            for p in portfolio.positions],
                 "sign_probability_under_model": None, "q_weighted_contribution_usd": 0.0}
                for a in atoms]
        likely = max(rows, key=lambda r: r["q_reference"])
        summary_row = {"label": likely["label"], "q_reference": likely["q_reference"],
                       "pnl_if_outcome_usd": 0.0, "pnl_model_range_5_95_usd": [0.0, 0.0]}
        return {"schema": SCHEMA, "status": "IDENTIFIED_NOMINAL_USD_CASH_ZERO",
                "family": FAMILY, "event_date": distribution.get("event_date"),
                "notional_usd": portfolio.notional, "data_quality_gaps": [],
                "missing_symbols": [], "missing_weight": 0.0,
                "reference_expectation": mean,
                "reference_expectation_range_over_q_polytope": _mean_range(atoms, distribution["constraints"]),
                "state_unit": "jobs, first-release change in total nonfarm payrolls", "outcomes": rows,
                "summary": {"most_likely_outcome": summary_row,
                            "largest_loss_outcome_reference_q_at_least_1pct": summary_row,
                            "q_standard_deviation_usd": 0.0, "q_weighted_pnl_usd": 0.0,
                            "usd_per_100k_jobs_surprise": 0.0,
                            "usd_per_100k_jobs_surprise_90pct": [0.0, 0.0],
                            "ranking_basis": "Identified zero nominal price change for fixed USD cash."},
                "positions": [{"symbol": p.symbol, "weight": p.weight,
                               "status": "KNOWN_NOMINAL_USD_CASH_ZERO", "slope": 0.0, "se": 0.0}
                              for p in portfolio.positions],
                "factors": {"portfolio_response": None, "tape_source": None},
                "uncertainty": {"method": "IDENTITY_NO_EMPIRICAL_SLOPE_USED",
                                "interpretation": "Nominal USD cash only; excludes interest carry, inflation and FX."},
                "physical_probability_P": None,
                "model_reliability": reliability,
                "scenario_transfer_status": "NOT_APPLICABLE",
                "scenario_transfer_failed_symbols": [], "scenario_transfer_untested_symbols": [],
                "future_total_return_prediction_available": False,
                "assumptions": ["Every position is fixed nominal USD cash or has zero weight.",
                                "A payroll release cannot reprice that cash balance; income, FX and purchasing power are excluded."]}
    if fit is None:
        raise ValueError("No measured payroll surprise response for this portfolio.")
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
        raise ValueError("At least two parameter draws are required")
    if (not math.isfinite(float(fit["slope"])) or not math.isfinite(float(fit["se"]))
            or fit["se"] < 0):
        raise ValueError("A finite payroll slope and nonnegative uncertainty are required")
    from .default_event_economics import _polytope
    from .outcome_table import _expectation_ranges, _pnl_q_envelope
    atoms = distribution["atoms"]
    notional = portfolio.notional
    constraints, reference, _ = _polytope(distribution, atoms)
    q = np.array(reference)
    values = np.array([a["value"] for a in atoms])
    mean = float(q @ values)
    per_unit = notional * fit["slope"]
    degrees = fit.get("n_train", 0) - 2
    critical = float(student_t.ppf(.95, degrees)) if degrees > 0 else 1.645
    per_unit_bounds = (notional * (fit["slope"] - critical * fit["se"]),
                       notional * (fit["slope"] + critical * fit["se"]))
    rng = np.random.default_rng(seed)
    slopes = ((fit["slope"] + fit["se"] * rng.standard_t(degrees, n_draws))
              if degrees > 0 else rng.normal(fit["slope"], fit["se"], n_draws))
    low_states = np.array([a["value_interval"][0] for a in atoms])
    high_states = np.array([a["value_interval"][1] for a in atoms])
    x_draw = rng.uniform(low_states, high_states, (n_draws, len(atoms)))
    # All states and their expectation move together under the same Q. Fixed
    # representative expectations with independently moved outcome states would
    # omit this dependency and would not center each simulated partition.
    mean_lower, mean_upper, optimization = _expectation_ranges(constraints, np.vstack([values, x_draw]))
    mean_low, mean_high = float(mean_lower[0]), float(mean_upper[0])
    pnl_draw = notional * slopes[:, None] * (x_draw - (x_draw @ q)[:, None]) / UNIT
    low_q, high_q = _pnl_q_envelope(notional, slopes / UNIT, x_draw, mean_lower[1:], mean_upper[1:])
    rows = []
    pnl_point = per_unit * (values - mean) / UNIT
    for k, atom in enumerate(atoms):
        low, high = float(np.percentile(low_q[:, k], 5)), float(np.percentile(high_q[:, k], 95))
        point = float(pnl_point[k])
        at_ref = pnl_draw[:, k]
        sign = None if abs(point) < 1e-9 else float(np.mean(np.sign(at_ref) == np.sign(point)))
        contributions = []
        for position in portfolio.positions:
            own = study.get("positions", {}).get(position.symbol) or {}
            own_slope = own.get("slope")
            value = (notional * position.weight * own_slope * (atom["value"] - mean) / UNIT
                     if own_slope is not None else None)
            contributions.append({"symbol": position.symbol,
                                  "channel": "NONE" if position.asset_type == "cash" or position.weight == 0 else "PAYROLL_RESPONSE",
                                  "pnl_usd": value})
        contribution_sum = math.fsum(c["pnl_usd"] for c in contributions if c["pnl_usd"] is not None)
        reconciled = math.isclose(contribution_sum, point, rel_tol=1e-8, abs_tol=1e-7)
        if not reconciled:
            # Older externally supplied studies can use unequal per-position
            # samples. Do not invent allocations of a different portfolio fit.
            contributions = [{**c, "pnl_usd": None} for c in contributions]
        rows.append({"outcome_id": atom["outcome_id"], "label": atom["label"], "q_interval": list(atom["q_interval"]),
                     "q_reference": atom["q_reference"], "state": atom["value"], "state_interval": list(atom["value_interval"]),
                     "surprise": atom["value"] - mean, "pnl_if_outcome_usd": point, "pnl_model_range_5_95_usd": [low, high],
                     "model_5_95_range_excludes_zero": bool(low > 0 or high < 0),
                     "pnl_model_range_5_95_at_reference_q_usd": [float(v) for v in np.percentile(at_ref, [5, 95])],
                     "source_contract_ids": deepcopy(atom.get("source_contract_ids", [])),
                     "position_contributions": contributions,
                     "position_contribution_status": "RECONCILED_COMMON_SAMPLE" if reconciled else "UNAVAILABLE_UNEQUAL_STUDY_SAMPLES",
                     "sign_probability_under_model": sign, "q_weighted_contribution_usd": atom["q_reference"] * point})
    weighted = float(pnl_point @ q)
    if abs(weighted) > 1e-6 * max(1.0, notional):
        raise ValueError("Q-centering identity failed")
    material = [r for r in rows if r["q_reference"] >= .01]
    worst = min(material or rows, key=lambda r: r["pnl_if_outcome_usd"])
    likely = max(rows, key=lambda r: r["q_reference"])
    missing = [p.symbol for p in portfolio.positions
               if p.asset_type != "cash" and p.weight != 0 and study["positions"].get(p.symbol, {}).get("status") != "ESTIMATED"]
    missing_weight = math.fsum(p.weight for p in portfolio.positions if p.symbol in missing)
    return {
        "schema": SCHEMA, "status": "COMPLETE_MEASURED_SURPRISE_RESPONSE" if not missing else "PARTIAL_KNOWN_POSITIONS_ONLY",
        "family": FAMILY, "event_date": distribution.get("event_date"), "data_quality_gaps": [],
        "notional_usd": notional, "missing_symbols": missing, "missing_weight": missing_weight,
        "reference_expectation": mean, "reference_expectation_range_over_q_polytope": [mean_low, mean_high],
        "state_unit": "jobs, first-release change in total nonfarm payrolls",
        "outcomes": rows,
        "summary": {
            "most_likely_outcome": {"label": likely["label"], "q_reference": likely["q_reference"],
                                    "pnl_if_outcome_usd": likely["pnl_if_outcome_usd"]},
            "largest_loss_outcome_reference_q_at_least_1pct": {"label": worst["label"], "q_reference": worst["q_reference"],
                                                               "pnl_if_outcome_usd": worst["pnl_if_outcome_usd"],
                                                               "pnl_model_range_5_95_usd": worst["pnl_model_range_5_95_usd"]},
            "q_standard_deviation_usd": float(math.sqrt(max(0.0, float((pnl_point ** 2) @ q)))),
            "q_weighted_pnl_usd": weighted,
            "usd_per_100k_jobs_surprise": per_unit, "usd_per_100k_jobs_surprise_90pct": list(per_unit_bounds),
            "ranking_basis": "Q-standard deviation of the not-yet-priced P&L across the payroll bands.",
        },
        "positions": [{"symbol": p.symbol, "weight": p.weight, **deepcopy(study["positions"].get(p.symbol, {"status": "CASH"}))}
                      for p in portfolio.positions],
        "model_reliability": reliability,
        "scenario_transfer_status": reliability["status"],
        "scenario_transfer_failed_symbols": reliability["scenario_transfer_failed_symbols"],
        "scenario_transfer_untested_symbols": reliability["scenario_transfer_untested_symbols"],
        "scenario_transfer_check_scope": reliability["scope"],
        "future_total_return_prediction_available": False,
        "factors": {"portfolio_response": deepcopy(fit), "tape_source": deepcopy(study.get("tape_source"))},
        "uncertainty": {"method": "FROZEN_SLOPE_STUDENT_T_AND_JOINT_STATE_DRAWS_Q_POLYTOPE_ENVELOPE",
                        "draws": n_draws, "seed": seed,
                        "parameter_degrees_of_freedom": degrees if degrees > 0 else None,
                        "q_optimization": optimization,
                        "max_absolute_reference_q_centering_error_usd": float(np.max(np.abs(pnl_draw @ q))),
                        "includes": ["training-only portfolio slope uncertainty (HC3, Student-t when training size is recorded)",
                                     "joint payroll values inside all bands (uniform; tails as declared)",
                                     "every feasible Q expectation recomputed for each joint state draw"],
                        "excludes": ["other news on release sessions beyond the regression error",
                                     "revisions to earlier months, the unemployment rate and wages, which the contract does not settle on",
                                     "a change in the response since the sample"],
                        "interpretation": "5th percentile of per-draw feasible-Q minima to the 95th percentile of per-draw feasible-Q maxima. Conditional on a finite declared tail domain and linear response, not a simultaneous parameter bound or future-return prediction interval.",
                        "sign_probability_scope": "Under the reference Q and the stated training-slope parameter distribution."},
        "physical_probability_P": None,
        "assumptions": [
            "The band quotes are the outcome weights embedded in asset prices, so the Q-weighted P&L is zero and each row is the move not yet in prices if that band settles.",
            "The portfolio moves with the payroll surprise as it did on past release days (close to close), measured against the Q-expected value of each release's own pre-release ladder.",
            "The reported response uses the first two thirds of this run's common release sessions. The chronological final third does not choose, refit or tune that coefficient. Previously inspected dates remain retrospective diagnostics, not a new unseen holdout; a new run may have a different sample split.",
            "The response is linear in the surprise; the unemployment rate, wages and revisions released at the same time are part of the regression error, not modelled.",
        ],
    }


def payroll_card(portfolio, original, distributions, prices_by_symbol, *, evaluated_at, tape=None):
    """Same-date ladder, measured response and table for one scheduled jobs report."""
    tape = load_tape() if tape is None else tape
    same = [(ticker, d) for ticker, d in distributions.items() if d.get("event_date") == original.get("event_date")]
    card = {"distribution": None, "economics": None}
    if len(same) != 1 or not same[0][1].get("atoms"):
        card.update(status="SOURCE_OUTCOME_DISTRIBUTION_UNAVAILABLE" if not same else "SOURCE_DISTRIBUTION_REJECTED",
                    reason=("No same-date KXPAYROLLS ladder was acquired; other dates are not substituted." if not same
                            else same[0][1].get("reason", "The ladder did not admit a coherent distribution.")),
                    candidate_distributions=[{"event_ticker": t, "status": d.get("status"), "event_date": d.get("event_date"),
                                              "reason": d.get("reason")} for t, d in distributions.items()])
        return card
    ticker, distribution = same[0]
    card.update(distribution=deepcopy(distribution), source_event_ticker=ticker,
                reviewed_question=distribution.get("probability_question"))
    study = payroll_response_study(portfolio, prices_by_symbol, tape, analysis_as_of=evaluated_at)
    card["payroll_response_study"] = study
    try:
        card["outcome_table"] = build_payroll_table(portfolio, distribution, study)
    except (ValueError, KeyError, TypeError) as exc:
        card.update(status="ECONOMIC_COMPONENT_UNAVAILABLE", reason="Probabilities only: " + str(exc))
    else:
        card["status"] = ("SOURCE_BANDS_WITH_IDENTIFIED_NOMINAL_CASH_ZERO"
                          if card["outcome_table"]["status"] == "IDENTIFIED_NOMINAL_USD_CASH_ZERO"
                          else "SOURCE_BANDS_WITH_MEASURED_SURPRISE_RESPONSE")
    return card
