"""What the archived CPI and FOMC history says about the chain's own claims.

The R11 package (reports/r11_validation) holds, for every admitted event, the
contract reference Q before the release, the official first-release value and
the 15-minute return of each instrument. This module asks of those same events
the questions Option 1 depends on, instead of a forecasting contest:

1. Do prices move on the surprise measured against Q (event study)?
2. Is the anticipated part already in prices? Kuttner (2001) splits the outcome
   into its Q expectation and the surprise; only the surprise should matter.
3. Is Q calibrated on these events (mean Q of the realized bucket against its
   value if outcomes followed Q)?
4. Given the realized outcome, does a response slope fitted only on earlier
   events beat "no move" out of sample, and how stable is it across periods?
5. How do the scored ex-ante forecasts compare with the Q-blind baseline once
   forecasts that share an event are grouped (forecast_scores)?

The current Q cannot move the chain's predictive mean, because the Q-weighted
surprise is zero; the mean still differs from the Q-blind baseline by a term
fixed before Q is seen (see q_blind_identity and location_gap). Offline and
deterministic; numpy only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "castle.event_study_evidence.v1"
COHORT = "lagged_24h"
SYMBOLS = ("SPY", "QQQ", "NVDA", "TLT")
OOS_STARTS = ("2022", "2023", "2025")
PERIODS = (("2021", "2022"), ("2023", "2024"), ("2025", "2026"))
MIN_PRIOR = 10
BOOTSTRAP_DRAWS = 10000
SEED = 20261024
UNITS = {"CPI": "headline CPI month-on-month percentage point", "FOMC": "policy-rate basis point"}
KUTTNER = ("K. N. Kuttner (2001), Monetary policy surprises and interest rates: Evidence from the Fed funds "
           "futures market, Journal of Monetary Economics 47(3), 523-544")
GSS = ("R. S. Gurkaynak, B. Sack and E. T. Swanson (2005), Do Actions Speak Louder Than Words? The Response of "
       "Asset Prices to Monetary Policy Actions and Statements, International Journal of Central Banking 1(1), 55-93")


def event_rows(package, cohort=COHORT):
    """Admitted events with Q expectation, realized bucket, official value and returns, in release order."""
    rows = []
    for event in package["normalized_events"][cohort]:
        if event.get("eligible") is False:
            continue
        states = event["states"]
        expected = math.fsum(s["q"] * s["value"] for s in states)
        realized = next(s for s in states if s["id"] == event["settled_state_id"])
        official = float(event["settlement_value"])
        rows.append({"event_id": event["event_id"], "family": event["family"],
                     "release_time_utc": event["release_time_utc"], "horizon": event.get("horizon"),
                     "q_expectation": expected, "official_value": official, "surprise": official - expected,
                     "realized_state_value": realized["value"], "q_of_realized_state": realized["q"],
                     "q_sum_of_squares": math.fsum(s["q"] ** 2 for s in states), "states": len(states),
                     "returns": dict(event["asset_returns"])})
    return sorted(rows, key=lambda row: (row["release_time_utc"], row["event_id"]))


def _hc3(X, y):
    """OLS coefficients, HC3 covariance and R^2."""
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coefficients
    inverse = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, inverse, X)
    covariance = inverse @ ((X * (residuals / (1 - leverage))[:, None] ** 2).T @ X) @ inverse
    centered = y - y.mean()
    return coefficients, covariance, float(1 - residuals @ residuals / (centered @ centered))


def _one_sided_sign_p(successes, n):
    """P(at least this many sign agreements) under a fair coin."""
    return math.fsum(math.comb(n, k) for k in range(successes, n + 1)) / 2 ** n


def _series(rows, family, symbol):
    used = [row for row in rows if row["family"] == family and symbol in row["returns"]]
    return used, np.array([row["surprise"] for row in used]), np.array([row["returns"][symbol] for row in used])


def surprise_response(rows, family, symbol):
    """Return per unit of surprise, with the Kuttner split into Q expectation and surprise."""
    used, s, y = _series(rows, family, symbol)
    if len(used) < 8 or not np.any(s):
        return {"symbol": symbol, "n": len(used), "status": "TOO_FEW_EVENTS_OR_NO_SURPRISE"}
    expected = np.array([row["q_expectation"] for row in used])
    one = np.ones(len(used))
    b, v, r2 = _hc3(np.c_[one, s], y)
    k, kv, _ = _hc3(np.c_[one, s, expected], y)
    contrast = np.array([0., 1., -1.])
    opposite = int(np.sum(np.sign(s) == -np.sign(y)))
    same = int(np.sum(np.sign(s) == np.sign(y)))
    return {"symbol": symbol, "n": len(used), "status": "ESTIMATED", "first_event": used[0]["event_id"],
            "last_event": used[-1]["event_id"],
            "return_per_unit_surprise": float(b[1]), "t": float(b[1] / math.sqrt(v[1, 1])), "r_squared": r2,
            "kuttner": {"surprise_coefficient": float(k[1]), "surprise_t": float(k[1] / math.sqrt(kv[1, 1])),
                        "expected_coefficient": float(k[2]), "expected_t": float(k[2] / math.sqrt(kv[2, 2])),
                        "expected_95": [float(k[2] - 1.96 * math.sqrt(kv[2, 2])), float(k[2] + 1.96 * math.sqrt(kv[2, 2]))],
                        "difference_t": float(contrast @ k / math.sqrt(contrast @ kv @ contrast))},
            "moved_against_surprise": opposite, "moved_with_surprise": same,
            "sign_p_one_sided": _one_sided_sign_p(max(opposite, same), opposite + same)}


def period_slopes(rows, family, symbol, periods=PERIODS):
    """Through-origin slope by calendar period: the size of the response, regime by regime."""
    answer = []
    for first, last in periods:
        used, s, y = _series([r for r in rows if first <= r["release_time_utc"][:4] <= last], family, symbol)
        answer.append({"period": f"{first}-{last}", "n": len(used),
                       "return_per_unit_surprise": None if len(used) < 3 or not np.any(s) else float(s @ y / (s @ s)),
                       "moved_against_surprise": int(np.sum(np.sign(s) == -np.sign(y)))})
    return answer


def out_of_sample(rows, family, symbol, start, *, min_prior=MIN_PRIOR, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Given the realized outcome, a slope fitted on earlier events only, against a zero-move prediction."""
    used, s, y = _series(rows, family, symbol)
    model, zero, signs, detail = [], [], [], []
    for i, row in enumerate(used):
        if row["release_time_utc"][:4] < start or i < min_prior or not np.any(s[:i]):
            continue
        slope = s[:i] @ y[:i] / (s[:i] @ s[:i])
        model.append((slope * s[i] - y[i]) ** 2)
        zero.append(y[i] ** 2)
        signs.append(bool(np.sign(slope * s[i]) == np.sign(y[i])))
        detail.append({"event_id": row["event_id"], "earlier_events": i, "slope_from_earlier": float(slope),
                       "surprise": float(s[i]), "conditional_estimate": float(slope * s[i]), "observed": float(y[i])})
    if len(model) < 5:
        return {"symbol": symbol, "from": start, "n": len(model), "status": "TOO_FEW_EVENTS"}
    model, zero = np.array(model), np.array(zero)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(model), (draws, len(model)))
    ratios = np.sqrt(model[picks].mean(axis=1) / zero[picks].mean(axis=1))
    return {"symbol": symbol, "from": start, "n": len(model), "status": "EVALUATED", "per_event": detail,
            "rmse_model": float(math.sqrt(model.mean())), "rmse_zero": float(math.sqrt(zero.mean())),
            "rmse_ratio": float(math.sqrt(model.mean() / zero.mean())),
            "rmse_ratio_95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])],
            "sign_correct": int(sum(signs))}


def q_calibration(rows, family):
    used = [row for row in rows if row["family"] == family]
    return {"n": len(used), "mean_q_of_realized_state": float(np.mean([r["q_of_realized_state"] for r in used])),
            "mean_if_outcomes_followed_q": float(np.mean([r["q_sum_of_squares"] for r in used])),
            "median_q_of_realized_state": float(np.median([r["q_of_realized_state"] for r in used]))}


def q_blind_identity():
    """What the current Q can and cannot change in the chain's ex-ante prediction."""
    return ("With residuals e_k = r_k - a - b z_k from the twelve calibration events, every atom of the chain's "
            "prediction is a + b z_j + e_k = r_k + b (z_j - z_k): the Q-blind baseline's own returns r_k with the "
            "past surprise z_k replaced by a surprise z_j drawn from Q. Because sum_j q_j z_j = 0, the predictive mean "
            "is mean(r_k) - b mean(z_k) whatever the current Q, so Q's information enters only through the spread. "
            "The mean still differs from the baseline's mean(r_k) by -b mean(z_k), the slope times the average "
            "surprise of the calibration events, which is fixed before Q is seen (see location_gap).")


def location_gap(package, cohort=COHORT):
    """Distance between the chain's predictive mean and the Q-blind baseline's mean, per CPI target."""
    result = []
    for target in package["cohorts"][cohort].get("target_results", []):
        if target.get("family") != "CPI":
            continue
        gaps = [(1e4 * (f["prediction"]["distribution_summary"]["mean_return"] - f["scores"]["prior_mean_return"]),
                 f["event_id"]) for f in target.get("forecasts", []) if f.get("status") == "EVALUATED"]
        if gaps:
            largest = max(gaps, key=lambda g: abs(g[0]))
            result.append({"target": target["target"]["id"], "forecasts": len(gaps),
                           "median_abs_gap_bp": float(np.median([abs(g) for g, _ in gaps])),
                           "largest_gap_bp": float(largest[0]), "largest_gap_event": largest[1]})
    return result


def forecast_scores(package, cohort=COHORT, *, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Mean CRPS difference against the Q-blind baseline on the scored CPI forecasts.

    Forecasts of several targets share an event, so differences are averaged
    within each event and events are resampled, not forecasts.
    """
    forecasts = []
    for target in package["cohorts"][cohort].get("target_results", []):
        if target.get("family") != "CPI":
            continue
        ledger = {row["event_id"]: row for row in target.get("stability_feedback_ledger") or []}
        for forecast in target.get("forecasts", []):
            row = ledger.get(forecast["event_id"])
            if forecast.get("status") != "EVALUATED" or row is None:
                continue
            terms = row["terms"]
            forecasts.append({"target": target["target"]["id"], "event_id": forecast["event_id"],
                              "pool_weight": row["forecast_weight"], "model": terms["crps_F"],
                              "pool": terms["crps_pool_quadratic"], "baseline": terms["crps_G"]})
    rng = np.random.default_rng(seed)
    result = []
    for label, chosen in (("ALL_CPI_TARGETS", forecasts),
                          ("PORTFOLIO_A", [f for f in forecasts if f["target"] == "PORTFOLIO_A"])):
        events = sorted({f["event_id"] for f in chosen})
        if not events:
            continue
        entry = {"targets": label, "forecasts": len(chosen), "events": len(events),
                 "pool_used_baseline_only": sum(f["pool_weight"] == 0 for f in chosen)}
        for name in ("model", "pool"):
            per_event = 1e4 * np.array([np.mean([f[name] - f["baseline"] for f in chosen if f["event_id"] == e])
                                        for e in events])
            boot = per_event[rng.integers(0, len(per_event), (draws, len(per_event)))].mean(axis=1)
            entry[name + "_minus_baseline_bp"] = float(per_event.mean())
            entry[name + "_minus_baseline_95_bp"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        result.append(entry)
    return result


def _findings(cpi, oos, periods, fomc, scores=()):
    found = []
    strong = [r for r in cpi if r.get("status") == "ESTIMATED" and abs(r["t"]) >= 1.96]
    if strong:
        found.append(f"CPI surprises measured against Q move prices: significant at 5% for "
                     f"{', '.join(r['symbol'] for r in strong)} (t from {min(r['t'] for r in strong):.1f} to "
                     f"{max(r['t'] for r in strong):.1f}).")
    priced = [r for r in cpi if r.get("status") == "ESTIMATED" and abs(r["kuttner"]["expected_t"]) < 1.96
              and abs(r["kuttner"]["difference_t"]) >= 1.96]
    if priced:
        found.append("The Q-expected part of the CPI print has no detectable effect while the surprise does "
                     f"({', '.join(r['symbol'] for r in priced)}). This is consistent with the anticipated part being "
                     "already in prices, but does not prove it: the intervals for the expected part are wide and Q can "
                     "be up to 24 hours old.")
    for start in OOS_STARTS:
        rows = [r for r in oos if r["from"] == start and r["status"] == "EVALUATED"]
        better = [r["symbol"] for r in rows if r["rmse_ratio_95"][1] < 1]
        found.append(f"Out of sample from {start}: conditional error below a zero-move prediction with 95% confidence for "
                     + (", ".join(better) if better else "no instrument") + f" ({len(rows)} evaluated).")
    spy = periods.get("SPY") or []
    if len(spy) == 3 and all(p["return_per_unit_surprise"] is not None for p in spy):
        found.append("The size of the CPI response depends on the regime: SPY "
                     + ", ".join(f"{p['period']} {1e4 * p['return_per_unit_surprise']:.0f} bp" for p in spy)
                     + " per percentage point of surprise.")
    overall = next((row for row in scores if row["targets"] == "ALL_CPI_TARGETS"), None)
    if overall:
        tied = all(row[name + "_minus_baseline_95_bp"][0] <= 0 <= row[name + "_minus_baseline_95_bp"][1]
                   for row in (overall,) for name in ("model", "pool"))
        found.append(f"Ex-ante forecast scores over {overall['events']} CPI events: "
                     + ("neither the response model nor the stability pool differs from the Q-blind baseline with 95% confidence"
                        if tied else "at least one forecast differs from the Q-blind baseline with 95% confidence")
                     + ". Q cannot move the chain's predictive mean, so its information enters only through the spread "
                       "(identity below).")
    if fomc["n"]:
        found.append(f"Q anticipated the FOMC decisions: median absolute surprise {fomc['median_abs_surprise']:.1f} bp "
                     f"over {fomc['n']} decisions, so a decision contract cannot explain the announcement-window moves.")
    return found


def build_evidence(package, *, source_sha256, cohort=COHORT):
    rows = event_rows(package, cohort)
    cpi = [surprise_response(rows, "CPI", symbol) for symbol in SYMBOLS]
    oos = [out_of_sample(rows, "CPI", symbol, start) for start in OOS_STARTS for symbol in SYMBOLS]
    periods = {symbol: period_slopes(rows, "CPI", symbol) for symbol in SYMBOLS}
    scores = forecast_scores(package, cohort) if cohort in package.get("cohorts", {}) else []
    fomc_rows = [row for row in rows if row["family"] == "FOMC"]
    surprises = np.abs([row["surprise"] for row in fomc_rows]) if fomc_rows else np.array([])
    fomc = {"n": len(fomc_rows), "median_abs_surprise": float(np.median(surprises)) if len(surprises) else None,
            "max_abs_surprise": float(surprises.max()) if len(surprises) else None,
            "calibration": q_calibration(rows, "FOMC") if fomc_rows else None,
            "responses": [surprise_response(rows, "FOMC", symbol) for symbol in ("QQQ", "NVDA", "TLT")],
            "reading": "Equity moves in the 14:00-14:15 window come mostly from the statement and guidance, which a "
                       "decision contract does not price; see " + GSS + ". With decision surprises of a few basis "
                       "points the slopes below are not interpretable and are kept only for completeness."}
    return {
        "schema": SCHEMA, "source_report_sha256": source_sha256, "cohort": cohort,
        "units": UNITS, "returns": "simple instrument return over the archived 15-minute announcement window",
        "events": {"CPI": sum(r["family"] == "CPI" for r in rows), "FOMC": len(fomc_rows),
                   "first": rows[0]["event_id"] if rows else None, "last": rows[-1]["event_id"] if rows else None},
        "findings": _findings(cpi, oos, periods, fomc, scores),
        "cpi": {"q_calibration": q_calibration(rows, "CPI"), "surprise_response": cpi,
                "period_slopes": periods, "out_of_sample": oos},
        "fomc": fomc,
        "why_not_crps": q_blind_identity(),
        "location_gap": location_gap(package, cohort) if cohort in package.get("cohorts", {}) else [],
        "forecast_scores": scores,
        "method": {
            "surprise": "official first-release value minus the Q expectation of the contract partition before the cutoff",
            "regressions": "OLS with intercept, HC3 standard errors; Kuttner split adds the Q expectation as a regressor",
            "kuttner_reference": KUTTNER,
            "forecast_scores": "mean CRPS difference against the Q-blind baseline, averaged within each event; 95% interval "
                               f"by event bootstrap ({BOOTSTRAP_DRAWS} draws, seed {SEED})",
            "out_of_sample": f"expanding through-origin slope from earlier events only (at least {MIN_PRIOR}), applied to "
                             "the realized surprise; 95% interval of the RMSE ratio by event bootstrap "
                             f"({BOOTSTRAP_DRAWS} draws, seed {SEED})",
        },
        "limitations": [
            "Retrospective, not blind: the outcomes were public before this analysis.",
            "Exploratory quote cohort: reference quotes can precede the cutoff by up to 24 hours, so part of a measured "
            "surprise can be news that arrived after the quote.",
            "Few events per period (8 to 17); the out-of-sample result from 2022 leans on the 2022 inflation regime.",
            "Sign tests treat events as independent fair coins under the null.",
            "Q is a contract reference, not a certified physical probability.",
        ],
    }


def load_evidence(root):
    """Read the committed evidence and check that it was built from the hash-pinned R11 report."""
    folder = Path(root) / "reports" / "r11_validation"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    files = {entry["file"]: entry["sha256"] for entry in manifest["files"]}
    body = (folder / "event_study_evidence.json").read_bytes()
    evidence = json.loads(body)
    if evidence.get("schema") != SCHEMA:
        raise ValueError("Unsupported event-study evidence")
    if hashlib.sha256(body).hexdigest() != files.get("event_study_evidence.json"):
        raise ValueError("Event-study evidence hash mismatch")
    if evidence["source_report_sha256"] != files.get("historical_chain.json"):
        raise ValueError("Event-study evidence was built from a different historical report")
    return evidence
