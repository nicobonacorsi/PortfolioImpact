"""Synthetic R31 identity, model-routing and offline-replay safety regressions.

Every provider-shaped receipt created below is a synthetic byte-binding test,
not evidence about the named security or a live provider request.
"""
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import socket

import pytest

from castle_risk import adapters
from castle_risk.default_assessment import build_default_assessment
from castle_risk.fund_composition import analyze_fund_rows, portfolio_rate_sensitivity
from castle_risk.instrument_exposures import build_instrument_exposures
from castle_risk.outcome_table import build_outcome_table, instrument_model_gate
from castle_risk.portfolio import parse_portfolio
from castle_risk.saved_assessment_replay import upgrade_saved_assessment
from test_r30_saved_replay import (CAPTURE, OLD_ROOT, _write_private, _write_report,
                                   saved_fixture)
from test_round8_default_integration import (NOW, add_macro, synthetic_distribution,
                                            synthetic_report, synthetic_search)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Identity/model replay regressions must remain offline")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(adapters, "_transport", denied)


def source_metadata(path, *, symbol="NVDA", kind="ETF", stamp=NOW.isoformat(), **fields):
    """Synthetic, locally bound identity bytes with an explicit test marker."""
    row = {"ticker": symbol, "type": kind, "active": True, "currency_name": "usd", **fields}
    body = json.dumps({"status": "OK", "results": row,
                       "test_data": "SYNTHETIC_IDENTITY_NOT_PROVIDER_EVIDENCE"}, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    url = "https://api.massive.com/v3/reference/tickers/" + symbol + "?date=" + stamp[:10]
    digest = sha256(body).hexdigest()
    path.with_suffix(".meta.json").write_text(json.dumps({
        "url": url, "retrieved_at": stamp, "sha256": digest, "status": 200,
        "fixture": False, "test_data": "SYNTHETIC_IDENTITY_NOT_PROVIDER_EVIDENCE",
    }))
    return {**row, "source_url": url, "source_response_sha256": digest,
            "raw_path": str(path), "retrieved_at": stamp, "metadata_as_of": stamp[:10]}


def model_inputs(kind="stock"):
    report = synthetic_report()
    report["portfolio"]["positions"][0]["asset_type"] = kind
    portfolio = parse_portfolio(report["portfolio"]["positions"], notional=100_000)
    return portfolio, synthetic_distribution(), report["empirical"]["daily_betas"]


def test_typed_etf_conflicting_with_stock_is_not_a_complete_equity_model(tmp_path):
    portfolio, distribution, daily = model_inputs()
    metadata = {"NVDA": source_metadata(tmp_path / "identity.json")}
    exposures = build_instrument_exposures(portfolio.to_dict()["positions"],
        instrument_metadata=metadata, evaluated_at=NOW)
    table = build_outcome_table(portfolio, distribution, daily,
        instrument_exposures=exposures, n_draws=100)
    assert table["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert table["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert table["legacy_fallback_allowed"] is False
    assert table["positions"][0]["channel"] == "UNKNOWN"
    assert all(row["pnl_if_outcome_usd"] is None for row in table["outcomes"])
    assert table["summary"]["q_standard_deviation_usd"] is None


def test_resolved_etf_without_class_or_holdings_never_falls_back_to_equity(tmp_path):
    portfolio, distribution, daily = model_inputs("etf")
    exposures = build_instrument_exposures(portfolio.to_dict()["positions"],
        instrument_metadata={"NVDA": source_metadata(tmp_path / "identity.json")}, evaluated_at=NOW)
    table = build_outcome_table(portfolio, distribution, daily,
        instrument_exposures=exposures, n_draws=100)
    assert table["instrument_identity_gate"]["identity_blocked_symbols"] == []
    assert table["instrument_identity_gate"]["asset_class_blocked_symbols"] == ["NVDA"]
    assert table["missing_weight"] == 1
    assert table["positions"][0]["channel"] == "UNKNOWN"
    assert all(row["position_contributions"][0]["pnl_usd"] is None for row in table["outcomes"])


def test_source_resolved_fund_with_supplied_duration_uses_rates_not_beta(tmp_path):
    portfolio, distribution, daily = model_inputs("etf")
    exposures = build_instrument_exposures(portfolio.to_dict()["positions"],
        instrument_metadata={"NVDA": source_metadata(tmp_path / "identity.json")}, evaluated_at=NOW)
    # A typed ETF plus a supplied DV01 alone is not whole-fund bond evidence.
    # Retain this positive routing test with an explicit synthetic source
    # composition, independent of the deliberately misleading stock ticker.
    composition = analyze_fund_rows([
        {"constituent_ticker": "T 4.0 05/15/2030", "constituent_name": "SYNTHETIC TREASURY",
         "asset_class": "FIXED INCOME", "security_type": "US GOVERNMENT", "weight": .9},
        {"constituent_ticker": "TERMS_UNKNOWN", "constituent_name": "SYNTHETIC TREASURY UNKNOWN TERMS",
         "asset_class": "FIXED INCOME", "security_type": "US GOVERNMENT", "weight": .1},
    ], "NVDA", as_of=NOW.date().isoformat())
    rate = portfolio_rate_sensitivity(portfolio, {"NVDA": composition})
    table = build_outcome_table(portfolio, distribution, daily,
        instrument_exposures=exposures, rate_sensitivity=rate, n_draws=100)
    assert table["status"] == "COMPLETE_DECLARED_MODEL"
    assert table["positions"][0]["channel"] == "RATES"
    assert table["instrument_identity_gate"]["blocked_symbols"] == []


@pytest.mark.parametrize("scope", [{"active": False}, {"currency_name": "eur"}])
def test_unsupported_scope_cannot_be_overridden_by_supplied_duration(tmp_path, scope):
    portfolio, distribution, daily = model_inputs("etf")
    exposures = build_instrument_exposures(portfolio.to_dict()["positions"],
        instrument_metadata={"NVDA": source_metadata(tmp_path / "identity.json", **scope)}, evaluated_at=NOW)
    table = build_outcome_table(portfolio, distribution, daily, instrument_exposures=exposures,
        rate_sensitivity={"positions": [{"symbol": "NVDA", "usd_per_bp_parallel_rise": -60.,
                                          "covered_fund_nav_weight": .9}]}, n_draws=100)
    assert table["positions"][0]["channel"] == "UNKNOWN"
    assert table["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]


def test_identity_gate_is_not_a_ticker_registry_and_unverified_declaration_is_not_verified():
    portfolio, _, _ = model_inputs("etf")
    exposures = build_instrument_exposures(portfolio.to_dict()["positions"], evaluated_at=NOW)
    assert exposures["positions"][0]["instrument_identity_verified"] is False
    assert exposures["positions"][0]["instrument_class"] == "unresolved_etf"
    gate = instrument_model_gate(portfolio, instrument_exposures=exposures)
    assert gate["blocked_symbols"] == []
    assert gate["no_class_inferred_from_ticker_or_name"] is True


def test_default_projection_corrects_verified_identity_without_mutating_report(tmp_path):
    report = synthetic_report()
    add_macro(report)
    original = deepcopy(report)
    inputs = {"instrument_metadata": {"NVDA": source_metadata(tmp_path / "identity.json")}, "holdings": {}}
    result = build_default_assessment(report, market_objects={}, inputs=inputs, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert report == original
    assert result["resolved_portfolio"]["positions"][0]["asset_type"] == "etf"
    assert result["instrument_identity"]["corrections"][0]["declared_asset_type"] == "stock"
    assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert card["legacy_fallback_allowed"] is False
    assert card["outcome_table"]["legacy_fallback_allowed"] is False
    assert card["ranking"]["abs_delta_pnl_usd"] is None
    assert card["risk_ranking"]["q_standard_deviation_usd"] is None
    # The old association remains a diagnostic, never the escape hatch for a
    # complete total after the identity-aware model has rejected its routing.
    assert card["economics"] is not None
    assert card["outcome_table"]["missing_symbols"] == ["NVDA"]


def test_unverified_conflicting_claim_blocks_but_does_not_retype(tmp_path):
    report = synthetic_report()
    add_macro(report)
    metadata = source_metadata(tmp_path / "identity.json")
    metadata.pop("raw_path")
    result = build_default_assessment(report, market_objects={},
        inputs={"instrument_metadata": {"NVDA": metadata}}, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert result["resolved_portfolio"]["positions"][0]["asset_type"] == "stock"
    assert result["instrument_identity"]["corrections"] == []
    assert card["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"


def test_saved_wrong_stock_type_is_reconciled_but_missing_fund_inputs_remain_missing(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    metadata = source_metadata(root / "massive_private" / "identity.json", stamp=CAPTURE,
                               asset_class="Fixed Income")
    metadata["raw_path"] = OLD_ROOT + "\\massive_private\\identity.json"
    private["inputs"]["instrument_metadata"] = {"NVDA": metadata}
    _write_private(root, private)
    _write_report(root, report)
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original = deepcopy(report)
    result = upgrade_saved_assessment(report, root)
    assert report == original
    assert result["offline_replay"]["engine_revision"] == "R31"
    assert result["offline_replay"]["new_provider_calls"] == 0
    assert result["evaluated_at_utc"] == CAPTURE
    assert result["empirical"] == report["empirical"]
    assert result["portfolio"]["positions"][0]["asset_type"] == "etf"
    assert result["original_portfolio_declaration"]["positions"][0]["asset_type"] == "stock"
    assert result["offline_replay"]["original_input_fingerprint"] == report["input_fingerprint"]
    assert result["input_fingerprint"] != report["input_fingerprint"]
    assert result["default_analysis"]["portfolio_rate_sensitivity"]["positions"] == []
    fed = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "fomc-2026-10-28-scheduled")
    assert fed["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert fed["legacy_fallback_allowed"] is False
    assert "NVDA" in fed["instrument_identity_gate"]["blocked_symbols"]
    retained = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "synthetic-taiwan")
    assert retained["legacy_fallback_allowed"] is False
    assert retained["instrument_identity_gate"]["blocked_symbols"] == ["NVDA"]
    assert retained["market_search"] == report["default_analysis"]["events"][0]["market_search"]
    assert {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_saved_replay_without_metadata_does_not_invent_an_identity(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    result = upgrade_saved_assessment(report, root)
    assert result["portfolio"] == report["portfolio"]
    assert result["instrument_identity"]["status"] == "PARTIAL_IDENTITY_VERIFICATION"
    assert result["instrument_identity"]["corrections"] == []
    assert result["instrument_identity"]["positions"][0]["source_verified"] is False


def test_explicit_empirical_opt_in_cannot_bypass_identity_failure():
    portfolio, distribution, daily = model_inputs()
    table = build_outcome_table(portfolio, distribution, daily,
        empirical_response={"deliberately_not_used": True}, identity_blocked_symbols=["NVDA"])
    assert table["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert table["legacy_fallback_allowed"] is False
    assert table["outcomes"] == []


def test_serialized_auto_identity_failure_is_not_downgraded_to_manual_stock():
    report = synthetic_report()
    add_macro(report)
    report["instrument_identity"] = {"status": "IDENTITY_REVIEW_REQUIRED", "blocked_symbols": ["NVDA"],
        "positions": [{"symbol": "NVDA", "weight": 1., "declared_asset_type": "auto",
                       "resolved_asset_type": None, "source_verified": False, "blocked": True}]}
    result = build_default_assessment(report, market_objects={}, inputs={}, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert result["instrument_identity"]["blocked_symbols"] == ["NVDA"]
    assert result["instrument_identity"]["positions"][0]["declared_asset_type"] == "auto"
    assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert card["outcome_table"]["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert card["risk_ranking"]["q_standard_deviation_usd"] is None


def test_prior_block_can_only_be_cleared_by_current_verified_supported_identity(tmp_path):
    report = synthetic_report()
    add_macro(report)
    report["instrument_identity"] = {"status": "IDENTITY_REVIEW_REQUIRED", "blocked_symbols": ["NVDA"],
        "positions": [{"symbol": "NVDA", "declared_asset_type": "auto", "blocked": True}]}
    record = source_metadata(tmp_path / "identity.json", kind="CS")
    result = build_default_assessment(report, market_objects={},
        inputs={"instrument_metadata": {"NVDA": record}}, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert result["instrument_identity"]["blocked_symbols"] == []
    assert card["outcome_table"]["status"] == "COMPLETE_DECLARED_MODEL"
    assert card["instrument_identity_gate"]["blocked_symbols"] == []


@pytest.mark.parametrize("kind", ["ETN", "ETV", "ADRC"])
def test_retained_unsupported_stock_identity_cannot_reenter_through_beta(kind):
    portfolio, distribution, daily = model_inputs()
    exposures = {"positions": [{"symbol": "NVDA", "asset_type": "stock",
        "instrument_class": "unsupported", "provider_instrument_type": kind,
        "classification_status": "DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED"}]}
    table = build_outcome_table(portfolio, distribution, daily,
        instrument_exposures=exposures, n_draws=100)
    assert table["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert table["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert table["legacy_fallback_allowed"] is False
    assert table["summary"]["q_standard_deviation_usd"] is None


def test_legacy_manually_declared_etv_fund_keeps_separate_source_class_review():
    portfolio, _, _ = model_inputs("etf")
    exposures = {"positions": [{"symbol": "NVDA", "asset_type": "etf",
        "instrument_class": "unsupported", "provider_instrument_type": "ETV",
        "classification_status": "DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED"}]}
    gate = instrument_model_gate(portfolio, instrument_exposures=exposures)
    assert gate["identity_blocked_symbols"] == []


@pytest.mark.parametrize("current_metadata", [{}, {"NVDA": {}}])
def test_original_exposure_conflict_is_not_erased_when_private_metadata_is_missing(current_metadata):
    report = synthetic_report()
    add_macro(report)
    report["exposures"] = {"positions": [{"symbol": "NVDA", "asset_type": "stock",
        "instrument_class": "unsupported", "provider_instrument_type": "ETF",
        "classification_status": "DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED"}]}
    result = build_default_assessment(report, market_objects={},
        inputs={"instrument_metadata": current_metadata}, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert card["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert card["outcome_table"]["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert card["risk_ranking"]["q_standard_deviation_usd"] is None


def test_saved_serialized_unresolved_auto_is_not_changed_to_manual_stock(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    report["instrument_identity"] = {"status": "IDENTITY_REVIEW_REQUIRED", "blocked_symbols": ["NVDA"],
        "positions": [{"symbol": "NVDA", "weight": 1., "declared_asset_type": "auto",
                       "resolved_asset_type": None, "source_verified": False, "blocked": True}]}
    _write_report(root, report)
    result = upgrade_saved_assessment(report, root)
    assert result["instrument_identity"]["blocked_symbols"] == ["NVDA"]
    assert result["instrument_identity"]["positions"][0]["declared_asset_type"] == "auto"
    fed = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "fomc-2026-10-28-scheduled")
    assert fed["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert fed["risk_ranking"]["q_standard_deviation_usd"] is None
    retained = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "synthetic-taiwan")
    assert retained["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert retained["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert retained["legacy_fallback_allowed"] is False
    assert retained["market_search"] == report["default_analysis"]["events"][0]["market_search"]


@pytest.mark.parametrize("declared_kind", ["stock", "etf"])
def test_verified_ticker_identity_cannot_erase_a_prior_leveraged_mandate(tmp_path, declared_kind):
    report = synthetic_report()
    add_macro(report)
    report["portfolio"]["positions"][0]["asset_type"] = declared_kind
    report["exposures"] = {"positions": [{"symbol": "NVDA", "asset_type": declared_kind,
        "instrument_class": "unsupported", "provider_instrument_type": "ETF",
        "classification_status": "LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED"}]}
    metadata = source_metadata(tmp_path / "identity.json", asset_class="Equity")
    result = build_default_assessment(report, market_objects={},
        inputs={"instrument_metadata": {"NVDA": metadata}}, evaluated_at=NOW,
        distribution_overrides={"SYNTHETIC-FED": synthetic_distribution()})
    card = next(row for row in result["events"] if row["event_id"] == "synthetic-fed")
    assert result["resolved_portfolio"]["positions"][0]["asset_type"] == "etf"
    assert result["instrument_identity"]["positions"][0]["source_verified"] is True
    assert card["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert card["outcome_table"]["positions"][0]["channel"] == "UNKNOWN"
    assert card["outcome_table"]["status"] == "PARTIAL_KNOWN_POSITIONS_ONLY"
    assert card["risk_ranking"]["q_standard_deviation_usd"] is None


def test_saved_type_reconciliation_keeps_prior_leverage_scope_restriction(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    report["exposures"] = {"positions": [{"symbol": "NVDA", "asset_type": "stock",
        "instrument_class": "unsupported", "provider_instrument_type": "ETF",
        "classification_status": "LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED"}]}
    metadata = source_metadata(root / "massive_private" / "identity.json", stamp=CAPTURE,
                               asset_class="Equity")
    metadata["raw_path"] = OLD_ROOT + "\\massive_private\\identity.json"
    private["inputs"]["instrument_metadata"] = {"NVDA": metadata}
    _write_private(root, private)
    _write_report(root, report)
    result = upgrade_saved_assessment(report, root)
    assert result["portfolio"]["positions"][0]["asset_type"] == "etf"
    assert result["exposures"]["positions"][0]["classification_status"] == "LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED"
    fed = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "fomc-2026-10-28-scheduled")
    assert fed["instrument_identity_gate"]["identity_blocked_symbols"] == ["NVDA"]
    assert fed["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
    assert fed["risk_ranking"]["q_standard_deviation_usd"] is None


def test_hard_identity_block_prevents_nonmacro_and_contract_ranking_fallback(monkeypatch):
    from castle_risk import default_assessment as module

    report = synthetic_report()
    report["instrument_identity"] = {"status": "IDENTITY_REVIEW_REQUIRED", "blocked_symbols": ["NVDA"],
        "positions": [{"symbol": "NVDA", "weight": 1., "declared_asset_type": "auto",
                       "resolved_asset_type": None, "source_verified": False, "blocked": True}]}
    report["events"].append({"event_id": "synthetic-tariff", "title": "Synthetic tariff risk",
        "event_date": "2027-01-19", "family": "TRADE_TARIFFS"})
    monkeypatch.setattr(module, "contract_first_review", lambda *args, **kwargs: {
        "contract_scenarios": [{"status": "DECLARED_CONTRACT_SCENARIO",
            "contract": {"contract_id": "SYNTHETIC_TEST"}, "summary": {"q_standard_deviation_usd": 500.}}]})
    result = build_default_assessment(report, market_objects={}, inputs={}, evaluated_at=NOW,
        geopolitical_search=synthetic_search(report))
    for identifier in ("synthetic-taiwan", "synthetic-export", "synthetic-tariff"):
        card = next(row for row in result["events"] if row["event_id"] == identifier)
        assert card["status"] == "BLOCKED_INSTRUMENT_IDENTITY"
        assert card["legacy_fallback_allowed"] is False
        assert card["ranking"]["abs_delta_pnl_usd"] is None
        assert card["risk_ranking"]["q_standard_deviation_usd"] is None
