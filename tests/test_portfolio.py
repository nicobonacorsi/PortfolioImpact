"""Accounting invariants tested with synthetic inputs, not market evidence."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from castle_risk.portfolio import (
    Holding, HoldingsSnapshot, Portfolio, Position, parse_portfolio, recover_exposures,
)


NOW = datetime(2026, 10, 16, 16, tzinfo=timezone.utc)


def sample_portfolio():
    return parse_portfolio([
        {"symbol": "NVDA", "weight": 0.35, "asset_type": "stock"},
        {"symbol": "QQQ", "weight": 0.45, "asset_type": "etf"},
        {"symbol": "TLT", "weight": 0.20, "asset_type": "etf"},
    ])


def snapshot(**changes):
    base = HoldingsSnapshot(
        fund_symbol="QQQ", as_of="2026-10-15", retrieved_at="2026-10-16T15:00:00Z",
        source_url="https://example.test/fixtures/qqq-holdings",
        holdings=(Holding("NVDA", 0.08), Holding("MSFT", 0.12)),
    )
    return replace(base, **changes)


def test_explicit_asset_types_and_weight_validation():
    p = sample_portfolio()
    assert p.total_weight == pytest.approx(1)
    assert p.positions[1].asset_type == "etf"
    with pytest.raises(ValueError, match="sum to 1"):
        parse_portfolio([{"symbol": "NVDA", "weight": 0.7, "asset_type": "stock"}])
    with pytest.raises(ValueError, match="asset_type"):
        parse_portfolio([{"symbol": "NVDA", "weight": 1}])


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf"), True, 35])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(ValueError):
        Position("NVDA", weight)


def test_duplicate_positions_rejected_after_normalization():
    with pytest.raises(ValueError, match="Duplicate"):
        Portfolio((Position(" nvda ", 0.5), Position("NVDA", 0.5)))


def test_cash_is_supported_and_notional_is_positive():
    portfolio = Portfolio((Position("CASH", 1, "cash"),), 50_000)
    result = recover_exposures(portfolio, now=NOW)
    assert result.identified_by_symbol == {"CASH": 1}
    assert result.unresolved_weight == 0
    with pytest.raises(ValueError, match="positive"):
        Portfolio(portfolio.positions, 0)


def test_partial_holdings_preserve_unidentified_weight_without_renormalization():
    report = recover_exposures(sample_portfolio(), {"QQQ": snapshot()}, now=NOW)
    assert report.identified_by_symbol["NVDA"] == pytest.approx(0.35 + 0.45 * 0.08)
    assert report.identified_by_symbol["MSFT"] == pytest.approx(0.45 * 0.12)
    assert report.unresolved_weight == pytest.approx(0.45 * 0.80 + 0.20)
    assert report.total_weight == pytest.approx(1)
    assert sum(report.identified_by_symbol.values()) + report.unresolved_weight == pytest.approx(1)
    looked_through = [e for e in report.entries if e.status == "look_through"]
    assert all(e.as_of == "2026-10-15" for e in looked_through)
    assert all(e.source_url == "https://example.test/fixtures/qqq-holdings" for e in looked_through)


@pytest.mark.parametrize("changes, reason", [
    ({"as_of": "2026-01-01"}, "stale"),
    ({"as_of": "2026-10-17"}, "future"),
    ({"retrieved_at": "2026-10-17T15:00:00Z"}, "future"),
    ({"retrieved_at": "2026-10-16T15:00:00"}, "timezone"),
    ({"reliable": False}, "reliable"),
    ({"complete": False}, "incomplete"),
    ({"weight_basis": "percent"}, "fraction-of-NAV"),
    ({"holdings": (Holding("NVDA", 0.7), Holding("MSFT", 0.4))}, "exceed"),
    ({"holdings": (Holding("NVDA", 0.1), Holding("NVDA", 0.1))}, "Duplicate"),
    ({"holdings": (Holding("NVDA", -0.1),)}, "Negative"),
    ({"holdings": (Holding("NVDA", 0.1, "derivative"),)}, "Derivative"),
    ({"holdings": (Holding("QQQ", 0.1, "etf"),)}, "Self-referential"),
    ({"source_url": "https://example.test/data?apiKey=secret"}, "credentials or query"),
])
def test_invalid_snapshots_remain_unresolved(changes, reason):
    report = recover_exposures(sample_portfolio(), {"QQQ": snapshot(**changes)}, now=NOW)
    assert report.identified_by_symbol == {"NVDA": 0.35}
    assert report.unresolved_weight == pytest.approx(0.65)
    assert report.total_weight == pytest.approx(1)
    qqq = [e for e in report.entries if e.original_symbol == "QQQ"]
    assert len(qqq) == 1 and reason in qqq[0].reason


def test_nested_etf_is_not_mistaken_for_resolved_equities():
    report = recover_exposures(
        Portfolio((Position("QQQ", 1, "etf"),)),
        {"QQQ": snapshot(holdings=(Holding("VTI", 0.4, "etf"), Holding("CASH", 0.1, "cash"), Holding("NVDA", 0.5)))},
        now=NOW,
    )
    assert report.identified_by_symbol == {"CASH": 0.1, "NVDA": 0.5}
    assert report.unresolved_weight == pytest.approx(0.4)


def test_adapter_mapping_schema_is_accepted():
    data = {
        "fund_symbol": "QQQ", "as_of": "2026-10-15", "retrieved_at": "2026-10-16T15:00:00Z",
        "source_url": "https://api.massive.com/etf-global/v1/constituents", "reliable": True,
        "weight_basis": "fraction_of_nav", "complete": True,
        "holdings": [{"symbol": "NVDA", "weight": 0.5, "asset_type": "stock"}],
        "raw_paths": ["ignored-extra-adapter-metadata.json"],
    }
    result = recover_exposures(Portfolio((Position("QQQ", 1, "etf"),)), {"QQQ": data}, now=NOW)
    assert result.identified_by_symbol == {"NVDA": 0.5}
    assert result.unresolved_weight == 0.5


def test_holdings_age_limit_is_explicit_configurable_policy():
    report = recover_exposures(sample_portfolio(), {"QQQ": snapshot(as_of="2026-09-01")}, now=NOW, max_holdings_age_days=30)
    assert "30-day" in report.entries[1].reason
    assert report.max_holdings_age_days == 30


def test_missing_holdings_do_not_remove_etfs():
    report = recover_exposures(sample_portfolio(), now=NOW)
    assert report.unresolved_weight == pytest.approx(0.65)
    assert report.total_weight == 1


def test_missing_and_unclassified_holdings_preserve_nav_residual_and_valid_siblings():
    """Realistic normalized shape: missing weights are unknown, not known zeros."""
    data = {
        "fund_symbol": "QQQ", "as_of": "2026-10-15", "retrieved_at": "2026-10-16T15:00:00Z",
        "source_url": "https://api.massive.com/etf-global/v1/constituents", "reliable": True,
        "weight_basis": "fraction_of_nav", "complete": True,
        "holdings": [
            {"symbol": "NVDA", "weight": .3, "asset_type": "stock"},
            {"symbol": "FUTURE", "weight": .2, "asset_type": "other"},
            {"symbol": "MISSING", "weight": None, "asset_type": "stock"},
        ],
    }
    result = recover_exposures(Portfolio((Position("QQQ", 1, "etf"),)), {"QQQ": data}, now=NOW)
    assert result.identified_by_symbol == {"NVDA": .3}
    assert result.unresolved_weight == pytest.approx(.7)
    assert result.total_weight == pytest.approx(1)
    assert all(e.symbol != "MISSING" for e in result.entries)
    future = next(e for e in result.entries if e.symbol == "FUTURE")
    assert future.portfolio_weight == .2 and future.status == "unresolved_etf"
