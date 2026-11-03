"""R97: last-pass Castle requirements 1 and 3, including failure cases."""
from copy import deepcopy

import pytest

from castle_risk.portfolio_impact_adapter import (
    _documented_relevance_weight,
    _order_live_risks_by_documented_relevance,
    _position_accounting_audit,
)


def _risk(identifier, family, weight=None, source=False):
    return {
        "id": identifier, "family": family, "source_defined_scenario": source,
        "exposure_basis": "NAMED_SOURCE_CHANNELS" if weight is not None else "PORTFOLIO_CONTEXT_ONLY",
        "exposure": [] if weight is None else [{"symbol": "NVDA", "weight": weight}],
        "quality": {"definition_review": {"status": "DEFINED_FOR_REVIEW"}},
    }


def test_repeated_calendar_families_prioritize_the_distinct_supported_macro_risks():
    rows = [
        _risk("fed-a", "FED_POLICY"), _risk("fed-b", "FED_POLICY"),
        _risk("cpi-a", "US_INFLATION"), _risk("cpi-b", "US_INFLATION"),
        _risk("cpi-c", "US_INFLATION"), _risk("taiwan", "TAIWAN_SUPPLY"),
        _risk("tariff", "TRADE_TARIFFS"), _risk("payroll", "US_PAYROLLS"),
    ]
    ordered = _order_live_risks_by_documented_relevance(rows)
    assert [r["id"] for r in ordered[:3]] == ["fed-a", "cpi-a", "payroll"]
    assert [r["id"] for r in ordered[3:6]] == ["fed-b", "cpi-b", "cpi-c"]
    assert [r["id"] for r in ordered[6:]] == ["taiwan", "tariff"]
    assert len({id(r) for r in ordered}) == len(rows)


def test_material_documented_links_stay_ahead_of_unreviewed_and_repeated_calendar():
    rows = [_risk("fed", "FED_POLICY"), _risk("cpi", "US_INFLATION"),
            _risk("taiwan", "TAIWAN_SUPPLY", .6),
            _risk("taiwan-next", "TAIWAN_SUPPLY", .55),
            _risk("export", "EXPORT_CONTROLS", .4),
            _risk("source", "FED_POLICY", .9, source=True)]
    result = _order_live_risks_by_documented_relevance(rows)
    assert [r["id"] for r in result] == ["taiwan", "export", "taiwan-next", "fed", "cpi", "source"]
    assert result[0]["display_relevance"]["event_effect_identified"] is False
    assert result[0]["display_relevance"]["linked_weight"] == .6
    assert "display_relevance" not in result[-1]


def test_other_events_not_falsely_collapsed_into_one_causal_family():
    rows = [_risk("fed", "FED_POLICY"), _risk("auction", "OTHER"),
            _risk("sovereign", "OTHER"), _risk("fed-next", "FED_POLICY")]
    assert [r["id"] for r in _order_live_risks_by_documented_relevance(rows)] == [
        "fed", "fed-next", "auction", "sovereign"]


def test_small_equity_sleeve_does_not_promote_speculative_geopolitics_over_duration():
    rows = [_risk("fed1", "FED_POLICY"), _risk("fed2", "FED_POLICY"),
            _risk("cpi1", "US_INFLATION"), _risk("cpi2", "US_INFLATION"),
            _risk("payroll", "US_PAYROLLS"), _risk("chip_10pct", "EXPORT_CONTROLS", .1),
            _risk("other_unreviewed", "OTHER")]
    result = _order_live_risks_by_documented_relevance(rows)
    assert [r["id"] for r in result[:5]] == ["fed1", "cpi1", "payroll", "fed2", "cpi2"]
    assert result[-2]["id"] == "chip_10pct"


def test_duplicate_source_weight_cannot_promote_unverified_family():
    row = _risk("fake", "TAIWAN_SUPPLY", .75)
    row["exposure"].append({"symbol": "NVDA", "weight": .05})
    assert _documented_relevance_weight(row) is None
    assert _order_live_risks_by_documented_relevance([_risk("fed", "FED_POLICY"), row])[0]["id"] == "fed"


POSITIONS = [{"symbol": "NVDA", "weight": .2}, {"symbol": "QQQ", "weight": .5},
             {"symbol": "TLT", "weight": .3}]
CONTRIBUTIONS = [{"symbol": "NVDA", "weight": .2, "pnl": -110.},
                 {"symbol": "QQQ", "weight": .5, "pnl": -170.},
                 {"symbol": "TLT", "weight": .3, "pnl": 30.}]


def _audit(rows=None, total=-250.):
    return _position_accounting_audit(POSITIONS, {"pnl": total, "position_contributions": rows})


def test_full_accounting_only_when_each_original_position_and_total_reconcile():
    audit = _audit(deepcopy(CONTRIBUTIONS))
    assert audit["status"] == "FULL_RECONCILED"
    assert audit["covered_weight"] == pytest.approx(1.)
    assert audit["missing_symbols"] == []
    assert audit["checked_original_positions"] == 3
    assert "does not validate" in audit["scope"]


def test_partial_accounting_keeps_missing_fund_visible_not_zero():
    audit = _audit(deepcopy(CONTRIBUTIONS[:2]), total=-280.)
    assert audit["status"] == "PARTIAL_COVERAGE"
    assert audit["covered_weight"] == pytest.approx(.7)
    assert audit["missing_symbols"] == ["TLT"]


@pytest.mark.parametrize("rows,total", [
    (CONTRIBUTIONS, -249),
    (CONTRIBUTIONS + [CONTRIBUTIONS[0]], -360),
    ([{**CONTRIBUTIONS[0], "weight": .8}] + CONTRIBUTIONS[1:], -250),
    ([{**CONTRIBUTIONS[0], "symbol": "MSFT"}] + CONTRIBUTIONS[1:], -250),
    ([{**CONTRIBUTIONS[0], "pnl": float("nan")}] + CONTRIBUTIONS[1:], -250),
])
def test_bad_attribution_never_certifies_whole_portfolio(rows, total):
    assert _audit(deepcopy(rows), total)["status"] == "INVALID"


def test_undecomposed_model_total_is_not_mistaken_for_zero_contributions():
    audit = _audit([])
    assert audit["status"] == "NOT_AUDITABLE"
    assert "covered_weight" not in audit


def test_bad_original_portfolio_cannot_be_audited():
    duplicate = POSITIONS + [{"symbol": "TLT", "weight": 0}]
    assert _position_accounting_audit(duplicate, {"pnl": -250., "position_contributions": CONTRIBUTIONS})["status"] == "INVALID"
