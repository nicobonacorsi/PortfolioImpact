"""R94: how much this exact portfolio typically moves on a release day, and what is already priced in.

For each scheduled macro family (FOMC, CPI, payrolls) the project ships a derived table of
past releases that Kalshi listed: the pre-release market-implied expected outcome E_Q[X], the
realized outcome X and the surprise s = X - E_Q[X] (data/r94_kalshi_release_surprises.json,
built by scripts/r94_kalshi_surprise_study.py from public Kalshi candles).

`announcement_band` replays TODAY's weights through each holding's own close-to-close return
on those past release days (funds through their own prices, so ETF look-through is implicit),
removes the part the shipped outcome model attributes to the surprise, and measures the
remaining release-day swing with exponentially weighted weights (half-life 12 releases).
That swing is the uncertainty a reader must add to any outcome-conditional number: in the
R94 study it dominated the outcome-conditional mean for every tested portfolio.

The 80% band is validated out of sample in the R94 study (coverage reported in
data/r94_priced_in_evidence.json). It is a release-day range, not a total-return forecast.
No network, clock read, or model call happens here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import json
import math
from pathlib import Path

HALF_LIFE_RELEASES = 12
MIN_RELEASES = 12
Z80 = 1.2815515655446004
FAMILY_KEYS = {"FED_POLICY": "FOMC", "FOMC": "FOMC", "US_INFLATION": "CPI", "CPI": "CPI",
               "US_LABOR": "PAYROLLS", "US_PAYROLLS": "PAYROLLS", "PAYROLLS": "PAYROLLS", "LABOR_MARKET": "PAYROLLS"}
# Shipped transfer slopes (fractional SPY return per unit surprise) used to centre each outcome.
# FOMC: Bernanke-Kuttner -4.68 %/pp; CPI: Knox-Timmer -1.42 %/pp. Payrolls has no admitted
# transfer in the product, so its band is centred at zero.
SHIPPED_SPY_SLOPE = {"FOMC": -0.0468, "CPI": -0.0142}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _finite(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name + " must be a finite number")
    return float(x)


def load_release_table(path: Path | None = None) -> dict:
    table = json.loads(Path(path or DATA_DIR / "r94_kalshi_release_surprises.json").read_text(encoding="utf-8"))
    if table.get("schema") != "castle.r94_kalshi_release_surprises.v1":
        raise ValueError("Unsupported release-surprise table")
    return table


def load_evidence(path: Path | None = None) -> dict:
    data = json.loads(Path(path or DATA_DIR / "r94_priced_in_evidence.json").read_text(encoding="utf-8"))
    if data.get("schema") != "castle.r94_priced_in_evidence.v1":
        raise ValueError("Unsupported priced-in evidence file")
    return data


def _sorted_closes(series: Mapping[str, float]) -> list[tuple[str, float]]:
    rows = []
    for day, value in series.items():
        date.fromisoformat(str(day)[:10])
        v = _finite(value, "close")
        if v <= 0:
            raise ValueError("Closes must be positive")
        rows.append((str(day)[:10], v))
    return sorted(rows)


def release_day_returns(closes: Mapping[str, float], days: Sequence[str]) -> dict[str, float]:
    """Close(release day) / close(previous available session) - 1 for each release day present."""
    rows = _sorted_closes(closes)
    index = {d: i for i, (d, _) in enumerate(rows)}
    out = {}
    for d in days:
        i = index.get(d)
        if i is None or i == 0:
            continue
        out[d] = rows[i][1] / rows[i - 1][1] - 1.0
    return out


def _daily_beta(closes_a: Mapping[str, float], closes_spy: Mapping[str, float]) -> float | None:
    a, s = dict(_sorted_closes(closes_a)), dict(_sorted_closes(closes_spy))
    days = sorted(set(a) & set(s))
    xs, ys = [], []
    for i in range(1, len(days)):
        xs.append(s[days[i]] / s[days[i - 1]] - 1); ys.append(a[days[i]] / a[days[i - 1]] - 1)
    if len(xs) < 120:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    return None if vx <= 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx


def announcement_band(weights: Mapping[str, float], closes: Mapping[str, Mapping[str, float]], family: str,
                      table: Mapping, *, as_of: str, notional: float, benchmark: str = "SPY") -> dict:
    """Release-day band for one family. Cash ('USD') has zero release-day return.

    Returns status 'MEASURED' with the 80% half-width in dollars, or 'UNAVAILABLE' with a reason.
    Weights must sum to 1 within 1e-6 (cash included).
    """
    key = FAMILY_KEYS.get(str(family).upper())
    if key is None:
        return {"status": "UNAVAILABLE", "reason": "No release-day table for this event family"}
    notional = _finite(notional, "notional")
    w = {str(k).upper(): _finite(v, "weight") for k, v in weights.items()}
    if abs(sum(w.values()) - 1) > 1e-6 or any(v < 0 for v in w.values()):
        raise ValueError("Weights must be non-negative and sum to one")
    rows = [r for r in table.get("families", {}).get(key, []) if r.get("date") and r["date"] < as_of]
    held = [s for s, v in w.items() if v > 0 and s not in ("USD", "CASH")]
    missing = [s for s in held if s not in closes]
    if missing:
        return {"status": "UNAVAILABLE", "reason": "No daily closes for " + ", ".join(sorted(missing))}
    days = [r["date"] for r in rows]
    per = {s: release_day_returns(closes[s], days) for s in held}
    usable = [r for r in rows if all(r["date"] in per[s] for s in held)]
    if len(usable) < MIN_RELEASES:
        return {"status": "UNAVAILABLE", "family": key,
                "reason": f"Only {len(usable)} past {key} releases with prices for every holding (need {MIN_RELEASES})"}
    slope = SHIPPED_SPY_SLOPE.get(key, 0.0)
    beta_p = 0.0
    if slope:
        if benchmark not in closes:
            return {"status": "UNAVAILABLE", "reason": "Benchmark closes unavailable"}
        for s in held:
            b = _daily_beta(closes[s], closes[benchmark])
            if b is None:
                return {"status": "UNAVAILABLE", "reason": "Too little history to measure the daily beta of " + s}
            beta_p += w[s] * b
    r = [sum(w[s] * per[s][row["date"]] for s in held) for row in usable]
    m = [beta_p * slope * _finite(row["surprise"], "surprise") for row in usable]
    k = len(usable)
    wts = [0.5 ** ((k - 1 - j) / HALF_LIFE_RELEASES) for j in range(k)]
    tw = sum(wts)
    sigma = math.sqrt(sum(x * (ri - mi) ** 2 for x, ri, mi in zip(wts, r, m)) / tw)
    raw_abs = sorted(abs(x) for x in r)
    return {"status": "MEASURED", "family": key, "n_releases": k,
            "first_release": usable[0]["date"], "last_release": usable[-1]["date"],
            "sigma_fraction": sigma, "half_width80_usd": Z80 * sigma * notional,
            "largest_past_move_usd": max(raw_abs) * notional,
            "median_abs_move_usd": raw_abs[len(raw_abs) // 2] * notional,
            "centre_model": ("SHIPPED_PUBLISHED_TRANSFER_TIMES_KALSHI_SURPRISE" if slope else "ZERO_NO_ADMITTED_TRANSFER"),
            "portfolio_daily_spy_beta": beta_p if slope else None,
            "method": ("Today's weights replayed through each holding's own close-to-close return on past Kalshi-listed "
                       "release days; exponentially weighted (half-life 12 releases) s.d. of the part not explained by "
                       "the shipped outcome model; 80% band = ±1.28 s.d. Release-day range, not a forecast of total return.")}


def attach_bands(risks: list, weights: Mapping[str, float], closes: Mapping[str, Mapping[str, float]], *,
                 as_of: str, notional: float, table: Mapping | None = None, evidence: Mapping | None = None) -> list:
    """Attach `announcement_band` and `priced_in_evidence` to each scheduled macro risk in place."""
    table = table or load_release_table()
    evidence = evidence or load_evidence()
    cache = {}
    for risk in risks:
        key = FAMILY_KEYS.get(str(risk.get("family") or "").upper())
        if key is None:
            continue
        if key not in cache:
            try:
                cache[key] = announcement_band(weights, closes, key, table, as_of=as_of, notional=notional)
            except ValueError as exc:
                cache[key] = {"status": "UNAVAILABLE", "reason": str(exc)[:200]}
        risk["announcement_band"] = dict(cache[key])
        fam = (evidence.get("families") or {}).get(key)
        if fam:
            risk["priced_in_evidence"] = dict(fam)
    return risks


def compute_report_bands(portfolio, prices_by_symbol: Mapping, *, as_of: str, notional: float,
                         table: Mapping | None = None) -> dict:
    """Bands for every scheduled family from the live pipeline's daily price rows (no new requests)."""
    try:
        table = table or load_release_table()
    except (OSError, ValueError) as exc:
        return {"status": "UNAVAILABLE", "reason": "Release table unavailable: " + str(exc)[:120]}
    weights = {}
    for p in portfolio.positions:
        key = "USD" if p.asset_type == "cash" else p.symbol.upper()
        weights[key] = weights.get(key, 0.0) + float(p.weight)
    closes = {}
    for symbol, rows in (prices_by_symbol or {}).items():
        try:
            closes[str(symbol).upper()] = {str(r["date"])[:10]: float(r["close"]) for r in rows
                                           if isinstance(r, Mapping) and r.get("close") is not None}
        except (KeyError, TypeError, ValueError):
            continue
    out = {"status": "COMPUTED", "as_of": as_of, "table_as_of": table.get("built_at_utc"), "families": {}}
    for fam in ("FOMC", "CPI", "PAYROLLS"):
        try:
            out["families"][fam] = announcement_band(weights, closes, fam, table, as_of=as_of, notional=notional)
        except ValueError as exc:
            out["families"][fam] = {"status": "UNAVAILABLE", "reason": str(exc)[:200]}
    return out
