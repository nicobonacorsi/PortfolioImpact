"""Synthetic source-byte fixtures exercise gates, never assert live provider proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

import pytest

from castle_risk import adapters
from castle_risk.adapters import MassiveClient
from castle_risk.instrument_exposures import build_instrument_exposures
from castle_risk.portfolio import parse_portfolio, recover_exposures


AT = datetime(2026, 10, 21, 14, tzinfo=timezone.utc)
RECEIVED = "2026-10-21T12:00:00+00:00"


@pytest.fixture(autouse=True)
def no_network_and_fixed_receipt(monkeypatch):
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 21, 12, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    def denied(*args, **kwargs):
        raise AssertionError("Only synthetic offline responses are allowed")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(adapters, "utc_now", lambda: RECEIVED)
    monkeypatch.setattr(adapters, "datetime", FixtureClock)


def row(symbol, weight, asset_class, security_type=None, **extra):
    return {"composite_ticker": "ARBFUND", "constituent_ticker": symbol,
        "weight": weight, "asset_class": asset_class, "security_type": security_type,
        "effective_date": "2026-10-19", "processed_date": "2026-10-20",
        "currency_traded": "USD", **extra}


def snapshot(tmp_path, monkeypatch, rows):
    """Replace native transport solely to exercise its non-fixture receipt gate.

    These are locally fabricated test bytes. fixture=False here is a gate input,
    not a claim of provider authenticity or a new HTTP acquisition.
    """
    def native_fixture(url, headers, timeout):
        return 200, {}, json.dumps({"status": "OK", "results": rows}).encode()
    monkeypatch.setattr(adapters, "_transport", native_fixture)
    service = MassiveClient(tmp_path, api_key="SYNTHETIC_TEST_VALUE", min_interval=0,
                            max_retries=0, cache_ttl=0)
    return service.holdings("ARBFUND", as_of_date="2026-10-21")


def report(snapshot_value, weight=1., other=None):
    portfolio = [{"symbol": "ARBFUND", "weight": weight, "asset_type": "etf"}]
    if other:
        portfolio.append(other)
    result = build_instrument_exposures(portfolio, holdings={"ARBFUND": snapshot_value}, evaluated_at=AT)
    original = recover_exposures(parse_portfolio(portfolio), {"ARBFUND": snapshot_value}, now=AT).to_dict()
    assert result["exposure_ledger"] == original
    return result


def test_structured_class_slices_survive_unresolved_security_types_without_reclassifying_fund(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("PLAIN", .61, "Equity", "Common Stock"),
        row("DEPOSITARY", .28, "Equity", "ADR"), row("NOTE", .1, "Fixed Income", "US GOVERNMENT"),
        row("UNKNOWN", .005, None)])
    result = report(source)
    composition = result["positions"][0]["holdings_composition"]
    assert composition["status"] == "LOCAL_SOURCE_WEIGHTED_COMPOSITION_VERIFIED"
    assert composition["reported_class_nav_weights"] == pytest.approx({
        "equity": .89, "fixed_income": .1, "commodity": 0, "cash": 0, "unclassified": .005})
    assert composition["unreported_nav_weight"] == pytest.approx(.005)
    assert composition["predominant_observed_class"] == "equity"
    assert composition["whole_fund_taxonomy_inferred"] is False
    assert composition["ledger_modified"] is False
    assert result["positions"][0]["instrument_class"] == "unresolved_etf"
    assert result["coverage"]["class_weights"]["equity_etf"] == 0
    # R85: an Equity-class ADR row is issuer equity, so only the note and the
    # unclassified row (0.105 of the fund's NAV) stay unresolved.
    assert result["exposure_ledger"]["unresolved_weight"] == pytest.approx(.11)
    assert result["coverage"]["source_observed_etf_class_portfolio_weights"]["equity"] == pytest.approx(.89)
    context = result["agent_exposure_input"]["source_verified_holdings_composition"]["positions"][0]
    assert context["source_class_portfolio_weights"]["fixed_income"] == pytest.approx(.1)
    assert context["unclassified_or_unobserved_portfolio_weight"] == pytest.approx(.01)
    assert "raw_path" not in json.dumps(composition)


def test_bond_class_evidence_does_not_turn_bond_identifiers_into_plain_stocks(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("TREASURY", .99, "Fixed Income", "US GOVERNMENT"),
        row("CASH", .006, "Cash"), row("FUNDUNIT", .004, "Equity", "Open-End Fund")])
    result = report(source, .4, {"symbol": "OTHER", "weight": .6, "asset_type": "stock"})
    assert result["coverage"]["source_observed_etf_class_portfolio_weights"]["fixed_income"] == pytest.approx(.396)
    assert result["coverage"]["source_observed_etf_class_portfolio_weights"]["cash"] == pytest.approx(.0024)
    assert result["coverage"]["unclassified_or_unobserved_etf_composition_weight"] == pytest.approx(0)
    assert result["coverage"]["class_weights"]["bond_etf"] == 0
    assert result["exposure_ledger"]["unresolved_weight"] == pytest.approx(.3976)
    assert result["positions"][0]["holdings_composition"]["predominant_observed_class"] == "fixed_income"


def test_old_revisions_are_not_added_and_latest_pair_is_verified_from_raw_bytes(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("OLD", 1, "Fixed Income", "US GOVERNMENT",
        processed_date="2026-10-19"), row("CURRENT", .9, "Equity", "ADR")])
    result = report(source)
    composition = result["positions"][0]["holdings_composition"]
    assert composition["processed_date"] == "2026-10-20"
    assert composition["reported_class_nav_weights"]["equity"] == .9
    assert composition["reported_class_nav_weights"]["fixed_income"] == 0
    assert composition["unreported_nav_weight"] == pytest.approx(.1)


@pytest.mark.parametrize("mutation", ["class", "weight", "hash", "fixture", "future", "truncated", "wrong_batch"])
def test_forged_normalization_or_unbound_source_cannot_supply_class_weights(tmp_path, monkeypatch, mutation):
    source = snapshot(tmp_path, monkeypatch, [row("HOLD", .8, "Equity", "ADR")])
    page = source["metadata"]["pages"][0]
    raw = Path(page["raw_path"])
    receipt_path = raw.with_suffix(".meta.json")
    receipt = json.loads(receipt_path.read_text())
    if mutation == "class":
        source["holdings"][0]["asset_class"] = "Fixed Income"
    elif mutation == "weight":
        source["holdings"][0]["weight"] = .9
    elif mutation == "hash":
        page["sha256"] = "0" * 64
    elif mutation == "fixture":
        receipt["fixture"] = True
    elif mutation == "future":
        receipt["retrieved_at"] = page["retrieved_at"] = "2026-10-22T12:00:00+00:00"
    elif mutation == "wrong_batch":
        source["metadata"]["selected_batch"] = ["2026-10-19", "2026-10-19"]
    else:
        payload = json.loads(raw.read_text())
        payload["next_url"] = "https://api.massive.com/etf-global/v1/constituents?cursor=missing"
        raw.write_text(json.dumps(payload))
        page["sha256"] = receipt["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    composition = report(source)["positions"][0]["holdings_composition"]
    assert composition["eligible_for_research"] is False
    assert "reported_class_nav_weights" not in composition


@pytest.mark.parametrize("rows", [
    [row("BOND", 1.0001, "Fixed Income"), row("CASH", -.0001, "Cash")],
    [row("BOND", .4, "Fixed Income"), row("BOND", .5, "Fixed Income")],
])
def test_invalid_actual_shape_stays_unresolved_even_with_typed_asset_classes(tmp_path, monkeypatch, rows):
    source = snapshot(tmp_path, monkeypatch, rows)
    assert source["reliable"] is False
    result = report(source)
    assert result["positions"][0]["holdings_composition"]["eligible_for_research"] is False
    assert result["exposure_ledger"]["unresolved_weight"] == 1


def test_missing_weight_stays_unknown_and_name_never_supplies_class(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("GOLDNAME", .7, None, constituent_name="A gold investment"),
        row("UNKNOWN", None, "Equity")])
    result = report(source)
    composition = result["positions"][0]["holdings_composition"]
    assert composition["reported_class_nav_weights"]["commodity"] == 0
    assert composition["reported_class_nav_weights"]["unclassified"] == .7
    assert composition["missing_weight_row_count"] == 1
    assert composition["unreported_nav_weight"] == pytest.approx(.3)


def test_etv_identity_is_source_verified_without_inventing_commodity_class(tmp_path):
    payload = {"results": {"ticker": "VEHICLE", "type": "ETV", "currency_name": "usd", "active": True}}
    raw = tmp_path / "metadata.json"
    raw.write_text(json.dumps(payload))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    url = "https://api.massive.com/v3/reference/tickers/VEHICLE?date=2026-10-21"
    raw.with_suffix(".meta.json").write_text(json.dumps({"url": url, "status": 200,
        "fixture": False, "retrieved_at": RECEIVED, "sha256": digest}))
    metadata = {**payload["results"], "raw_path": str(raw), "source_url": url,
        "retrieved_at": RECEIVED, "source_response_sha256": digest}
    result = build_instrument_exposures([{"symbol": "VEHICLE", "weight": 1, "asset_type": "etf"}],
        instrument_metadata={"VEHICLE": metadata}, evaluated_at=AT)
    position = result["positions"][0]
    assert position["instrument_identity_verified"] is True
    assert position["provider_instrument_type"] == "ETV"
    assert position["instrument_class"] == "unsupported"
    assert result["coverage"]["class_weights"]["commodity_etf"] == 0


def test_direct_and_accepted_constituent_nvda_overlap_is_unchanged(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("NVDA", .1, "Equity", "Common Stock"),
        row("OTHER", .89, "Equity", "ADR")])
    result = report(source, .7, {"symbol": "NVDA", "weight": .3, "asset_type": "stock"})
    assert result["coverage"]["taiwan_identified_nvda_channel_weight"] == pytest.approx(.37)
    assert len(result["channels_by_family"]["TAIWAN_SUPPLY"]) == 2
    assert result["exposure_ledger"]["total_weight"] == 1


def test_malformed_optional_composition_metadata_fails_closed(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("HOLD", .8, "Equity", "ADR")])
    source["metadata"] = []
    composition = report(source)["positions"][0]["holdings_composition"]
    assert composition["eligible_for_research"] is False


def test_r85_equity_class_aggregate_survives_duplicate_lines_without_lookthrough(tmp_path, monkeypatch):
    # An emerging-markets fund lists one issuer twice (local and depositary
    # line, different ISINs): the plain ledger rejects the snapshot, but a
    # reconciled NAV that is >= 98% Equity still establishes an equity fund.
    source = snapshot(tmp_path, monkeypatch, [
        row("TWSE1", .55, "Equity", "Common Stock", isin="TW0002330008"),
        row("TWSE1", .10, "Equity", "Common Stock", isin="US8740391003"),
        row("KRX1", .34, "Equity", "Common Stock", isin="KR7005930003"),
        row("USDCASH", .01, "Cash", None)])
    assert source["reliable"] is False
    composition = report(source)["positions"][0]["holdings_composition"]
    assert composition["status"] == "SIGNED_EQUITY_SOURCE_CLASS_AGGREGATE_ACCEPTED"
    assert composition["predominant_observed_class"] == "equity"
    assert composition["equity_class_aggregate_only"] is True
    assert composition["signed_source_class_attribution"] is False
    assert composition["equity_issuer_lookthrough_upgraded"] is False


def test_r85_mostly_equity_fund_below_threshold_stays_unresolved(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [
        row("TWSE1", .50, "Equity", "Common Stock", isin="TW0002330008"),
        row("TWSE1", .45, "Equity", "Common Stock", isin="US8740391003"),
        row("NOTE", .05, "Fixed Income", "US GOVERNMENT", isin="US912828XX11")])
    composition = report(source)["positions"][0]["holdings_composition"]
    assert composition["status"] == "SOURCE_VERIFIED_COMPOSITION_UNAVAILABLE"


def test_r85_equity_class_tolerates_tiny_fx_short_but_not_a_real_short(tmp_path, monkeypatch):
    base = [row("TWSE1", .55, "Equity", "Common Stock", isin="TW0002330008"),
            row("TWSE1", .10, "Equity", "Common Stock", isin="US8740391003"),
            row("KRX1", .3400004, "Equity", "Common Stock", isin="KR7005930003"),
            row("USDCASH", .01, "Cash", None)]
    tiny = snapshot(tmp_path, monkeypatch, base + [row("FXFWD", -4e-07, "FX", None)])
    composition = report(tiny)["positions"][0]["holdings_composition"]
    assert composition["status"] == "SIGNED_EQUITY_SOURCE_CLASS_AGGREGATE_ACCEPTED"
    big = [*base[:2], row("KRX1", .35, "Equity", "Common Stock", isin="KR7005930003"),
           row("USDCASH", .01, "Cash", None), row("SHORTX", -.01, "Equity", "Common Stock", isin="US0000000002")]
    composition = report(snapshot(tmp_path / "b", monkeypatch, big))["positions"][0]["holdings_composition"]
    assert composition["status"] == "SOURCE_VERIFIED_COMPOSITION_UNAVAILABLE"


def test_r85_composition_listing_slightly_above_nav_is_kept_without_rescaling(tmp_path, monkeypatch):
    source = snapshot(tmp_path, monkeypatch, [row("AAA", .6, "Equity", "Common Stock"),
        row("BBB", .4008, "Equity", "Common Stock")])
    composition = report(source)["positions"][0]["holdings_composition"]
    assert composition["status"] == "LOCAL_SOURCE_WEIGHTED_COMPOSITION_VERIFIED"
    assert composition["reported_nav_weight"] == pytest.approx(1.0008)
    assert composition["predominant_observed_class"] == "equity"
    over = snapshot(tmp_path / "o", monkeypatch, [row("AAA", .6, "Equity", "Common Stock"),
        row("BBB", .42, "Equity", "Common Stock")])
    assert report(over)["positions"][0]["holdings_composition"]["status"] == "SOURCE_VERIFIED_COMPOSITION_UNAVAILABLE"
