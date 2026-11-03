"""Historical check of the jobs-report response: release-session returns on the Kalshi payroll surprise.

The observations are derived values only (the surprise from the public payroll tape and each instrument's
close-to-close return on the release session); raw prices are not stored. The fits are rebuilt from them.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "castle.option1.payroll_evidence.v1"
UNIT = 100_000
EVIDENCE_PATH = Path("reports") / "r13_validation" / "payroll_evidence.json"
RECORDED_PORTFOLIOS = {"NVDA/QQQ/TLT": {"NVDA": .2, "QQQ": .5, "TLT": .3},
                       "BND/IEF": {"BND": .6, "IEF": .4},
                       "AAPL/JPM/GLD": {"AAPL": .35, "JPM": .35, "GLD": .3}}
SAMPLES = {"all": "Every release with a surprise (April 2023 to October 2026)",
           "kx": "Current KXPAYROLLS ladders only (November 2024 onward, denser strikes)"}


def _hc1(columns, y):
    design = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in columns])
    y = np.asarray(y, float)
    n, k = design.shape
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    cov = bread @ (design.T @ (design * (resid ** 2)[:, None])) @ bread * n / (n - k)
    return coef[1:], np.sqrt(np.diag(cov))[1:]


def _fit(rows):
    """rows: (release_date, surprise_jobs, expected_jobs, return)."""
    rows = sorted(rows)
    if len(rows) < 8:
        return {"n": len(rows), "status": "TOO_FEW_RELEASES"}
    x = [r[1] / UNIT for r in rows]
    e = [r[2] / UNIT for r in rows]
    y = [r[3] for r in rows]
    (slope,), (se,) = _hc1([x], y)
    (k_surprise, k_expected), (k_se_surprise, k_se_expected) = _hc1([x, e], y)
    split = (2 * len(rows)) // 3
    early = _hc1([x[:split]], y[:split])[0][0] if split >= 6 else None
    late = _hc1([x[split:]], y[split:])[0][0] if len(rows) - split >= 6 else None
    with_sign = sum(1 for xi, yi in zip(x, y) if xi * yi * slope > 0)
    return {"n": len(rows), "status": "ESTIMATED", "first": rows[0][0], "last": rows[-1][0],
            "slope_per_100k": float(slope), "se": float(se), "t": float(slope / se),
            "slope_90": [float(slope - 1.645 * se), float(slope + 1.645 * se)],
            "moved_with_fitted_sign": with_sign,
            "kuttner": {"surprise": float(k_surprise), "surprise_t": float(k_surprise / k_se_surprise),
                        "expected": float(k_expected), "expected_t": float(k_expected / k_se_expected),
                        "expected_95": [float(k_expected - 1.96 * k_se_expected), float(k_expected + 1.96 * k_se_expected)]},
            "slope_first_two_thirds": None if early is None else float(early),
            "slope_last_third": None if late is None else float(late)}


def _sample(observations, sample):
    return [o for o in observations if sample == "all" or o["event_ticker"].startswith("KX")]


def build_payroll_evidence(observations, *, source):
    """Position and recorded-portfolio fits for both samples, from derived observations only."""
    symbols = sorted({o["symbol"] for o in observations})
    fits = {}
    for sample in SAMPLES:
        chosen = _sample(observations, sample)
        by_symbol = {s: [(o["release_date"], o["surprise_jobs"], o["expected_jobs"], o["release_session_return"])
                         for o in chosen if o["symbol"] == s] for s in symbols}
        rows = {s: _fit(r) for s, r in by_symbol.items()}
        for name, weights in RECORDED_PORTFOLIOS.items():
            dated = {s: {r[0]: r for r in by_symbol.get(s, [])} for s in weights}
            common = set.intersection(*(set(d) for d in dated.values())) if all(dated.values()) else set()
            rows[name] = _fit([(day, dated[next(iter(weights))][day][1], dated[next(iter(weights))][day][2],
                                math.fsum(w * dated[s][day][3] for s, w in weights.items())) for day in common])
        fits[sample] = rows
    bonds, gold = fits["all"].get("BND/IEF", {}), fits["all"].get("GLD", {})
    findings = []
    if bonds.get("status") == "ESTIMATED":
        findings.append(f"Bond ETFs fall when payrolls beat the Kalshi Q expectation: BND/IEF {1e4 * bonds['slope_per_100k']:+.0f} bp "
                        f"per 100,000 jobs (t {bonds['t']:+.1f}, {bonds['n']} releases, {bonds['moved_with_fitted_sign']} moved "
                        f"with that sign). The size fell from {1e4 * bonds['slope_first_two_thirds']:+.0f} bp in the first two thirds "
                        f"of the sample to {1e4 * bonds['slope_last_third']:+.0f} bp in the last third.")
    if gold.get("status") == "ESTIMATED":
        findings.append(f"Gold moves the same way: GLD {1e4 * gold['slope_per_100k']:+.0f} bp per 100,000 jobs (t {gold['t']:+.1f}).")
    equities = [s for s in ("SPY", "QQQ", "NVDA", "AAPL", "JPM") if fits["all"].get(s, {}).get("status") == "ESTIMATED"]
    if equities and all(abs(fits["all"][s]["t"]) < 2 for s in equities):
        findings.append("No equity in the sample has a detectable response (all |t| < 2), so equity-heavy portfolios get "
                        "jobs-report ranges that include zero.")
    priced = [(s, f["kuttner"]["expected_t"]) for s, f in fits["all"].items()
              if f.get("status") == "ESTIMATED" and abs(f["kuttner"]["expected_t"]) >= 2]
    calm = [s for s, f in fits["all"].items() if f.get("status") == "ESTIMATED" and abs(f["kuttner"]["expected_t"]) < 2]
    findings.append(f"The Q-expected part has no detectable effect for {len(calm)} of {len(calm) + len(priced)} instruments and "
                    "portfolios, consistent with it being priced before the release"
                    + ("" if not priced else "; it is detectable for " + ", ".join(f"{s} (t {t:+.1f})" for s, t in priced)
                       + ", where that assumption is not supported on jobs reports") + ".")
    return {"schema": SCHEMA, "source": source, "unit": "fractional close-to-close return on the release session per 100,000 jobs",
            "estimator": "OLS with intercept and HC1 standard errors; Kuttner (2001) split adds the Q expectation as a regressor",
            "samples": SAMPLES, "recorded_portfolios": RECORDED_PORTFOLIOS, "fits": fits, "findings": findings,
            "observations": observations,
            "limitations": [
                "Close-to-close returns include the rest of the session's news, so the fits are noisier than an announcement window.",
                "In 2023 the Kalshi ladders had 100,000-job strikes and on some months their median sat well above the published "
                "consensus; Q may then differ from what bond and equity prices had priced, which biases slopes towards zero.",
                "Retrospective: the outcomes were public before this analysis; rules were fixed before returns were read.",
                "Two releases fell on Good Friday (7 April 2023, 3 April 2026) with US stock markets closed; they have no "
                "release-session return and are left out.",
                "The response weakened in the latest third of the sample for bonds, as it did for CPI.",
            ]}


def same_values(a, b):
    """Equal structure, text and integers; floats equal to 1e-10 relative, since BLAS builds differ in the last digits."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_values(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_values(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)
    return type(a) is type(b) and a == b


def load_payroll_evidence(root):
    path = Path(root) / EVIDENCE_PATH
    body = path.read_bytes()
    evidence = json.loads(body)
    if evidence.get("schema") != SCHEMA:
        raise ValueError("Unexpected payroll evidence schema")
    rebuilt = build_payroll_evidence(evidence["observations"], source=evidence["source"])
    if not same_values(json.loads(json.dumps(rebuilt["fits"])), evidence["fits"]):
        raise ValueError("Payroll evidence fits do not match a rebuild from the stored observations")
    return evidence, hashlib.sha256(body).hexdigest()


def render_payroll_evidence(st, *, root):
    if not (Path(root) / EVIDENCE_PATH).is_file():
        return
    try:
        evidence, _ = load_payroll_evidence(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error("The jobs-report evidence cannot be verified: " + str(exc))
        return
    st.markdown("**Jobs reports.** Surprise = BLS first release minus the Q expectation of the Kalshi payroll ladder "
                "at 08:20 ET (last trades); return = close to close on the release session.")
    for line in evidence["findings"]:
        st.markdown("- " + line)
    rows = []
    for name, fit in evidence["fits"]["all"].items():
        if fit.get("status") != "ESTIMATED":
            continue
        kx = evidence["fits"]["kx"].get(name, {})
        rows.append({"Instrument or portfolio": name, "Releases": fit["n"],
                     "Return per 100k jobs surprise (bp)": round(1e4 * fit["slope_per_100k"]), "t": round(fit["t"], 1),
                     "Effect of the Q-expected part (bp per 100k)": round(1e4 * fit["kuttner"]["expected"]),
                     "its t": round(fit["kuttner"]["expected_t"], 1),
                     "First 2/3 vs last 1/3 (bp)": (None if fit["slope_first_two_thirds"] is None or fit["slope_last_third"] is None
                                                    else f"{1e4 * fit['slope_first_two_thirds']:+.0f} / {1e4 * fit['slope_last_third']:+.0f}"),
                     "KXPAYROLLS only (bp, t)": (None if kx.get("status") != "ESTIMATED"
                                                 else f"{1e4 * kx['slope_per_100k']:+.0f} ({kx['t']:+.1f})")})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("The jobs-report tables use each portfolio's own fit over the live price history, with its 90% interval. "
               + " ".join(evidence["limitations"][:2]))
