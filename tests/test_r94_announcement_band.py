"""R94 release-day band: arithmetic, refusal cases, and the shipped derived tables."""
import json
import math
from datetime import date, timedelta

import pytest

from castle_risk import announcement_band as ab


def _calendar(n=900, start=date(2023, 1, 2)):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _closes(days, daily, jumps=None):
    out, level = {}, 100.0
    for i, d in enumerate(days):
        if i:
            level *= 1 + daily(i) + (jumps or {}).get(d, 0.0)
        out[d] = level
    return out


def _table(days, every=20, surprise=0.0, family="CPI"):
    rows = [{"date": d, "surprise": surprise, "expected": 0.3, "realized": 0.3 + surprise} for d in days[200::every]]
    return {"schema": "castle.r94_kalshi_release_surprises.v1", "families": {family: rows}}


def test_band_equals_ew_sd_of_release_day_moves_when_surprise_zero():
    days = _calendar()
    table = _table(days)
    rel = [r["date"] for r in table["families"]["CPI"]]
    jumps = {d: (0.01 if i % 2 else -0.01) for i, d in enumerate(rel)}
    spy = _closes(days, lambda i: 0.0005 * math.sin(i))
    stock = _closes(days, lambda i: 0.001 * math.sin(i), jumps)
    out = ab.announcement_band({"XYZ": 1.0}, {"XYZ": stock, "SPY": spy}, "US_INFLATION", table,
                               as_of=days[-1], notional=100000)
    assert out["status"] == "MEASURED" and out["family"] == "CPI"
    # surprise 0 => centre 0; every release-day move is about +-1% => sigma close to 1%
    assert 0.009 < out["sigma_fraction"] < 0.0115
    assert out["half_width80_usd"] == pytest.approx(ab.Z80 * out["sigma_fraction"] * 100000)


def test_cash_has_no_release_day_risk_and_scales_band():
    days = _calendar()
    table = _table(days)
    rel = [r["date"] for r in table["families"]["CPI"]]
    jumps = {d: (0.02 if i % 2 else -0.02) for i, d in enumerate(rel)}
    spy = _closes(days, lambda i: 0.0004 * math.cos(i))
    stock = _closes(days, lambda i: 0.0, jumps)
    full = ab.announcement_band({"XYZ": 1.0}, {"XYZ": stock, "SPY": spy}, "CPI", table, as_of=days[-1], notional=1e5)
    half = ab.announcement_band({"XYZ": .5, "USD": .5}, {"XYZ": stock, "SPY": spy}, "CPI", table, as_of=days[-1], notional=1e5)
    assert half["sigma_fraction"] == pytest.approx(full["sigma_fraction"] / 2, rel=1e-9)


def test_future_releases_are_never_used():
    days = _calendar()
    table = _table(days)
    rel = [r["date"] for r in table["families"]["CPI"]]
    spy = _closes(days, lambda i: 0.0003 * math.sin(i))
    stock = _closes(days, lambda i: 0.0002 * math.cos(i))
    cut = rel[20]
    out = ab.announcement_band({"A": 1.0}, {"A": stock, "SPY": spy}, "CPI", table, as_of=cut, notional=1e5)
    assert out["last_release"] < cut and out["n_releases"] == 20


def test_too_few_releases_or_missing_prices_refuse():
    days = _calendar()
    table = _table(days, every=200)
    spy = _closes(days, lambda i: 0.0)
    out = ab.announcement_band({"A": 1.0}, {"A": spy, "SPY": spy}, "CPI", table, as_of=days[-1], notional=1e5)
    assert out["status"] == "UNAVAILABLE" and "need 12" in out["reason"]
    out = ab.announcement_band({"B": 1.0}, {"SPY": spy}, "CPI", _table(days), as_of=days[-1], notional=1e5)
    assert out["status"] == "UNAVAILABLE" and "B" in out["reason"]
    assert ab.announcement_band({"A": 1.0}, {"A": spy}, "TAIWAN_SUPPLY", table, as_of=days[-1], notional=1e5)["status"] == "UNAVAILABLE"


def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        ab.announcement_band({"A": .7}, {}, "CPI", {"families": {}}, as_of="2026-10-01", notional=1e5)


def test_surprise_part_is_removed_with_shipped_transfer():
    days = _calendar()
    table = _table(days, surprise=0.5)
    rel = set(r["date"] for r in table["families"]["CPI"])
    spy_jump = {d: -0.0142 * 0.5 for d in rel}       # SPY moves exactly the shipped transfer
    spy = _closes(days, lambda i: 0.004 * math.sin(i * 1.7), spy_jump)
    out = ab.announcement_band({"SPY": 1.0}, {"SPY": spy}, "CPI", table, as_of=days[-1], notional=1e5)
    assert out["portfolio_daily_spy_beta"] == pytest.approx(1.0, abs=1e-9)
    raw = ab.announcement_band({"SPY": 1.0}, {"SPY": spy}, "PAYROLLS",
                               {"families": {"PAYROLLS": table["families"]["CPI"]}}, as_of=days[-1], notional=1e5)
    assert out["sigma_fraction"] < raw["sigma_fraction"]


def test_shipped_tables_are_consistent():
    table = ab.load_release_table()
    ev = ab.load_evidence()
    for fam in ("CPI", "FOMC", "PAYROLLS"):
        rows = table["families"][fam]
        assert len(rows) >= 20
        assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)
        for r in rows:
            assert r["surprise"] == pytest.approx(r["realized"] - r["expected"], abs=1e-9)
    assert set(ev["families"]) >= {"CPI", "FOMC", "PAYROLLS"}
    for key in ("FOMC", "CPI"):
        assert "preregistered test" in ev["families"][key]["long_sample_test"]
    import json as _json, pathlib as _pl
    docs = _pl.Path(ab.__file__).resolve().parents[1] / "docs"
    c = _json.loads((docs / "r94_addendum_c_result.json").read_text())
    d = _json.loads((docs / "r94_addendum_d_result.json").read_text())
    assert c["intraday"]["G7_MS"]["PASS"] is False and d["intraday"]["G8_MS"]["PASS"] is False
    for fam in ev["families"].values():
        assert fam["claim_scope"].startswith("Scheduled")
        assert "not" in fam["what_it_does_not_show"].lower()


def test_attach_bands_only_touches_macro_families():
    days = _calendar()
    table = _table(days)
    spy = _closes(days, lambda i: 0.001 * math.sin(i))
    risks = [{"family": "US_INFLATION"}, {"family": "TAIWAN_SUPPLY"}]
    ev = {"schema": "castle.r94_priced_in_evidence.v1", "families": {"CPI": {"claim_scope": "Scheduled", "what_it_does_not_show": "not"}}}
    ab.attach_bands(risks, {"SPY": 1.0}, {"SPY": spy}, as_of=days[-1], notional=1e5, table=table, evidence=ev)
    assert risks[0]["announcement_band"]["status"] == "MEASURED" and "priced_in_evidence" in risks[0]
    assert "announcement_band" not in risks[1]


class _SeriesClient:
    """Serves /series by category and empty /markets; records every URL."""
    def __init__(self, by_category):
        self.by_category, self.urls, self.request_count = by_category, [], 0

    def _get(self, url):
        from urllib.parse import parse_qs, urlparse
        self.urls.append(url); self.request_count += 1
        q = parse_qs(urlparse(url).query)
        meta = {"retrieved_at": "2026-10-31T00:00:00Z", "sha256": "0" * 64, "raw_path": "x", "url": url}
        if "/series" in url:
            return {"series": self.by_category.get(q["category"][0], [])}, meta
        return {"markets": [], "cursor": ""}, meta


def test_expanded_catalog_ranks_politics_series_for_an_unrouted_policy_query():
    # R94 audit: "NDAA ..." routed to Economics only, so the exact Kalshi series KXNDAA
    # (filed under Politics) was never read. Expanded runs rank both policy categories.
    from castle_risk.live_catalog import BoundedKalshiCatalog
    cats = {"Economics": [{"ticker": "KXCPIYOY", "title": "CPI year over year 2027"}],
            "Politics": [{"ticker": "KXNDAA", "title": "When will the FY2027 NDAA become law?"}]}
    client = _SeriesClient(cats)
    cat = BoundedKalshiCatalog(client, as_of=date(2026, 10, 31), expanded=True)
    assert "KXNDAA" in cat._category_series("NDAA fiscal year 2027 signed")
    assert cat._series_diagnostics["categories_ranked"][0] == "Economics"
    n = client.request_count
    cat._category_series("NDAA passed Senate")          # both categories cached: no new GET
    assert client.request_count == n
    legacy = BoundedKalshiCatalog(_SeriesClient(cats), as_of=date(2026, 10, 31), expanded=False)
    assert "KXNDAA" not in legacy._category_series("NDAA fiscal year 2027 signed")
