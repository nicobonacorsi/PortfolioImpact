"""Offline integration with synthetic prices and original archived contract bytes.

These price histories/calendars are deterministic test fixtures, not empirical
calibration evidence, real market observations, or paid provider executions.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import math
import socket
import pytest

import castle_risk.live_assessment as assessment
from castle_risk.agentic_bridge import catalog_record
from castle_risk.semantics import make_contract_target
from castle_risk.workflow import load_real_snapshot


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Integration fixtures must not make network requests.")
    monkeypatch.setattr(socket.socket, "connect", blocked)


def synthetic_inputs():
    symbols = ("SPY", "AAPL", "JPM", "GLD", "NVDA", "QQQ", "TLT")
    prices = {symbol: [] for symbol in symbols}
    closes = dict.fromkeys(symbols, 100.0)
    event_dates = {"FOMC": {"dates": []}, "CPI": {"dates": []}}
    coefficients = {
        "FOMC": {"SPY": 1., "AAPL": 2., "JPM": .5, "GLD": -.2, "NVDA": 3., "QQQ": 1.4, "TLT": -.6},
        "CPI": {"SPY": 1., "AAPL": .4, "JPM": 1.4, "GLD": .8, "NVDA": 1.5, "QQQ": .8, "TLT": -.2},
        "OTHER": {"SPY": 1., "AAPL": 1., "JPM": .7, "GLD": .2, "NVDA": 2., "QQQ": 1., "TLT": -.3},
    }
    day = date(2024, 1, 2)
    i = 0
    while day <= date(2026, 10, 15):
        if day.weekday() < 5:
            family = "FOMC" if i and i % 10 == 0 else "CPI" if i % 10 == 5 else "OTHER"
            benchmark_return = .004 * math.sin(i * 1.37) + .002 * math.cos(i * .43)
            for symbol in symbols:
                if i:
                    closes[symbol] *= 1 + coefficients[family][symbol] * benchmark_return
                prices[symbol].append({"date": day.isoformat(), "close": closes[symbol]})
            if family in event_dates:
                event_dates[family]["dates"].append(day.isoformat())
            i += 1
        day += timedelta(days=1)
    return {
        "prices_by_symbol": prices, "benchmark_symbol": "SPY", "holdings": {},
        "instrument_metadata": {}, "price_sources": {}, "attempts": [],
        "limitations": ["SYNTHETIC_TEST_PRICES_AND_CALENDARS_NOT_EMPIRICAL_MARKET_EVIDENCE"],
    }, event_dates


@pytest.fixture
def evidence_case():
    cpi = load_real_snapshot("cpi_oct26")["markets"][1]
    fed = next(m for m in load_real_snapshot("oct26")["markets"] if m.ticker.endswith("-H25"))
    now = datetime.fromisoformat(cpi.retrieved_at)
    events, groups = [], {}
    for family, market in [("FED_POLICY", fed), ("US_INFLATION", cpi)]:
        record = catalog_record(market)
        event = make_contract_target(record)
        event.update(family=family, event_date=market.close_time[:10], title=market.title)
        events.append(event)
        groups[event["event_id"]] = [record]
    events.append({
        "event_id": "taiwan-commercial-blockade-fixture", "family": "TAIWAN_SUPPLY",
        "title": "Taiwan commercial blockade hypothesis", "description": "A commercial shipping disruption near Taiwan",
        "event_date": "2026-12-31", "definition_origin": "SYNTHETIC_TEST_HYPOTHESIS",
    })
    inputs, release_dates = synthetic_inputs()
    return {
        "portfolio_rows": [{"symbol": "AAPL", "weight": .35, "asset_type": "stock"},
                           {"symbol": "JPM", "weight": .35, "asset_type": "stock"},
                           {"symbol": "GLD", "weight": .30, "asset_type": "etf"}],
        "notional": 100000., "inputs": inputs,
        "discovery": {"status": "SCRIPTED_INTEGRATION_FIXTURE_NOT_CLAUDE", "events": events,
                      "markets_by_event": groups, "matches_by_event": {}},
        "market_objects": {fed.ticker: fed, cpi.ticker: cpi}, "evaluated_at": now,
        "release_dates": release_dates,
        "scenario_settings": {
            "FED_POLICY": {"benchmark_return_contrast": -.01, "benchmark_contrast_interval": [-.012, -.008]},
            "US_INFLATION": {"benchmark_return_contrast": -.02, "benchmark_contrast_interval": [-.022, -.018]},
            "TAIWAN_SUPPLY": {"nvda_return_contrast": -.2, "nvda_contrast_interval": [-.3, -.1]},
        },
    }


def by_family(report):
    return {event["family"]: event for event in report["events"]}


def test_full_build_uses_distinct_instruments_and_distinct_family_calibrations(evidence_case):
    report = assessment.build_assessment_from_evidence(**evidence_case)
    studies = report["empirical"]["release_day_studies"]
    expected = {"FOMC": {"AAPL": 2., "JPM": .5, "GLD": -.2},
                "CPI": {"AAPL": .4, "JPM": 1.4, "GLD": .8}}
    for family, values in expected.items():
        for symbol, beta in values.items():
            fit = studies[family]["instruments"][symbol]
            assert fit["beta"] == pytest.approx(beta, abs=1e-10)
            assert fit["n_train"] >= 12 and fit["n_holdout"] >= 4
            assert fit["transfer_check"]["status"] == "PASS"
            assert not fit["forward_prediction_certified"]
    events = by_family(report)
    assert events["FED_POLICY"]["impact"]["delta_pnl_usd"] == pytest.approx(-815, abs=1e-8)
    assert events["US_INFLATION"]["impact"]["delta_pnl_usd"] == pytest.approx(-1740, abs=1e-8)
    assert events["US_INFLATION"]["impact"]["scope"] == "FULL_ORIGINAL_PORTFOLIO"
    assert report["physical_probability_P"] is None
    assert report["empirical"]["prior_A6_forward_transfer_status"] == "FAIL_RETAINED_NOT_OVERRIDDEN"


def test_actual_archived_receipt_gate_separates_current_and_stale_q(evidence_case):
    report = assessment.build_assessment_from_evidence(**evidence_case)
    events = by_family(report)
    cpi, fed = events["US_INFLATION"], events["FED_POLICY"]
    assert cpi["probability"]["pricing_measure_Q_interval"] == pytest.approx([.19, .22])
    assert cpi["event_definition_scope"] == "EXACT_CONTRACT_YES_SETTLEMENT"
    assert not cpi["broad_macro_event_matched"]
    assert fed["probability"]["pricing_measure_Q_interval"] is None
    assert fed["impact"]["delta_pnl_usd"] is not None
    assert fed["impact"]["priced_in"]["yes_pnl_usd"] is None
    q = .205
    assert cpi["impact"]["priced_in"]["yes_pnl_usd"] == pytest.approx((1 - q) * -1740, abs=1e-8)
    assert cpi["impact"]["priced_in"]["no_pnl_usd"] == pytest.approx(-q * -1740, abs=1e-8)


def test_ranking_uses_economic_magnitude_with_unknown_total_last(evidence_case):
    report = assessment.build_assessment_from_evidence(**evidence_case)
    assert [event["family"] for event in report["events"]] == ["US_INFLATION", "FED_POLICY", "TAIWAN_SUPPLY"]
    cpi, fed, taiwan = report["events"]
    assert cpi["ranking"]["abs_delta_pnl_usd"] == pytest.approx(1740, abs=1e-8)
    assert fed["ranking"]["abs_delta_pnl_usd"] == pytest.approx(815, abs=1e-8)
    assert taiwan["ranking"]["abs_delta_pnl_usd"] is None


def test_unknown_instrument_never_gets_zero_or_a_full_portfolio_total(evidence_case):
    evidence_case["portfolio_rows"] = [
        {"symbol": "AAPL", "weight": .5, "asset_type": "stock"},
        {"symbol": "UNMODELED", "weight": .5, "asset_type": "stock"},
    ]
    report = assessment.build_assessment_from_evidence(**evidence_case)
    cpi = by_family(report)["US_INFLATION"]
    assert cpi["impact"]["delta_pnl_usd"] is None
    assert cpi["impact"]["delta_pnl_interval_usd"] is None
    assert cpi["impact"]["missing_symbols"] == ["UNMODELED"]
    assert cpi["impact"]["known_component_delta_pnl_usd"] == pytest.approx(-400, abs=1e-8)
    assert cpi["impact"]["priced_in"]["yes_pnl_usd"] is None
    assert cpi["impact"]["known_component_priced_in"]["yes_pnl_usd"] == pytest.approx(-318, abs=1e-8)


def test_no_prices_does_not_invent_betas_even_with_usable_contract_q(evidence_case):
    evidence_case["inputs"]["prices_by_symbol"] = {}
    report = assessment.build_assessment_from_evidence(**evidence_case)
    events = by_family(report)
    assert events["US_INFLATION"]["probability"]["pricing_measure_Q_interval"] is not None
    for family in ("FED_POLICY", "US_INFLATION"):
        impact = events[family]["impact"]
        assert impact["delta_pnl_usd"] is None
        assert impact["known_component_delta_pnl_usd"] is None
        assert impact["priced_in"]["yes_pnl_usd"] is None
    assert report["empirical"]["daily_betas"]["status"] == "NOT_ESTIMATED"


@pytest.mark.parametrize("mutation", ["quote_tamper", "future_receipt", "missing_source"])
def test_full_pipeline_rechecks_every_admitted_contract_against_its_source(evidence_case, mutation):
    market = next(m for m in evidence_case["market_objects"].values() if m.ticker.startswith("KXCPI"))
    if mutation == "quote_tamper":
        market.yes_bid = 0.01
    elif mutation == "future_receipt":
        market.retrieved_at = (evidence_case["evaluated_at"] + timedelta(seconds=1)).isoformat()
    else:
        market.raw_path = None
    report = assessment.build_assessment_from_evidence(**evidence_case)
    cpi = by_family(report)["US_INFLATION"]
    assert cpi["probability"]["pricing_measure_Q_interval"] is None
    assert cpi["impact"]["priced_in"]["yes_pnl_usd"] is None
    assert cpi["impact"]["delta_pnl_usd"] == pytest.approx(-1740, abs=1e-8)


def test_proposed_nonfinite_impact_cannot_become_a_verified_or_ranked_total(evidence_case):
    evidence_case["inputs"]["prices_by_symbol"] = {}
    for event in evidence_case["discovery"]["events"]:
        event["impact"] = {"status": "COMPLETE", "delta_pnl_usd": float("nan"),
                           "delta_pnl_interval_usd": [float("nan"), float("inf")]}
    report = assessment.build_assessment_from_evidence(**evidence_case)
    for event in report["events"]:
        assert event["impact"]["delta_pnl_usd"] is None
        assert event["ranking"]["abs_delta_pnl_usd"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_declared_shock_fails_instead_of_becoming_a_price(evidence_case, invalid):
    evidence_case["scenario_settings"]["US_INFLATION"]["benchmark_return_contrast"] = invalid
    with pytest.raises(ValueError, match="finite"):
        assessment.build_assessment_from_evidence(**evidence_case)


def test_missing_release_calendar_uses_explicit_daily_fallback_without_fake_event_samples(evidence_case):
    evidence_case["release_dates"] = {}
    report = assessment.build_assessment_from_evidence(**evidence_case)
    cpi = by_family(report)["US_INFLATION"]
    assert cpi["impact"]["delta_pnl_usd"] is not None
    assert all(p["basis"] == "DECLARED_BENCHMARK_SCENARIO_TIMES_DAILY_BETA_EVENT_SAMPLE_INSUFFICIENT" for p in cpi["impact"]["positions"])
    assert report["empirical"]["release_day_studies"]["CPI"]["instruments"]["AAPL"]["n_total"] == 0
    assert report["empirical"]["release_day_studies"]["CPI"]["instruments"]["AAPL"]["beta"] is None


def test_taiwan_full_flow_uses_real_ledger_mass_and_unknown_remainder(evidence_case):
    evidence_case["portfolio_rows"] = [
        {"symbol": "NVDA", "weight": .35, "asset_type": "stock"},
        {"symbol": "QQQ", "weight": .45, "asset_type": "etf"},
        {"symbol": "TLT", "weight": .2, "asset_type": "etf"},
    ]
    evidence_case["inputs"]["holdings"] = {"QQQ": {
        "fund_symbol": "QQQ", "as_of": "2026-10-01", "retrieved_at": "2026-10-15T12:00:00Z",
        "source_url": "https://example.org/synthetic-integration-holdings", "reliable": True,
        "complete": True, "weight_basis": "fraction_of_nav",
        "holdings": [{"symbol": "NVDA", "weight": .1, "asset_type": "stock"},
                     {"symbol": "AAPL", "weight": .8, "asset_type": "stock"}],
    }}
    report = assessment.build_assessment_from_evidence(**evidence_case)
    taiwan = by_family(report)["TAIWAN_SUPPLY"]
    impact = taiwan["impact"]
    assert impact["channel_weight"] == pytest.approx(.395)
    assert impact["unresolved_holdings_weight"] == pytest.approx(.245)
    assert impact["known_component_delta_pnl_usd"] == pytest.approx(-7900)
    assert impact["delta_pnl_usd"] is None
    assert impact["break_even"]["remaining_return_contrast"] == pytest.approx(.079 / .605)
    assert len(impact["positions"]) == 3
    assert taiwan["probability"]["pricing_measure_Q_interval"] is None
    assert impact["known_component_priced_in"]["yes_pnl_usd"] is None
    assert "NOTHING_FITS" in taiwan["matching_comparisons"][0]["decision"]


def test_pure_build_never_reads_the_wall_clock_and_does_not_mutate_evidence(evidence_case, monkeypatch):
    class ClockForbidden(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("The deterministic builder may not depend on today's clock.")
    monkeypatch.setattr(assessment, "datetime", ClockForbidden)
    before = deepcopy(evidence_case)
    first = assessment.build_assessment_from_evidence(**evidence_case)
    second = assessment.build_assessment_from_evidence(**evidence_case)
    assert first == second
    assert evidence_case == before
    assert first["evaluated_at_utc"] == evidence_case["evaluated_at"].isoformat()


def test_later_explicit_date_retains_candidates_and_unknowns_without_stale_quote_promotion(evidence_case):
    evidence_case["evaluated_at"] = datetime(2027, 2, 1, 12, tzinfo=timezone.utc)
    report = assessment.build_assessment_from_evidence(**evidence_case)
    assert len(report["events"]) == 3
    assert all(e["probability"]["pricing_measure_Q_interval"] is None for e in report["events"])
    assert all(e["event_time_status"] == "PAST_RELATIVE_TO_EVALUATION" for e in report["events"])
    assert not report["historical_archive_automatically_substituted"]


def test_forged_documentary_card_cannot_supply_an_admitted_q(evidence_case):
    forged = {
        "contract_id": "FORGED-NO-SOURCE", "event_pricing_measure_Q_interval": [.4, .6],
        "contract_pricing_measure_Q_interval": [.4, .6], "pricing_measure_Q_interval": [.4, .6],
        "decision": "CERTIFIED_EVENT_EQUIVALENCE", "event_relationship_status": "CERTIFIED",
        "question_we_care_about": "A supplied documentary assertion", "market_question": "A supplied documentary assertion",
    }
    candidate = evidence_case["discovery"]["events"][1]
    candidate["documentary_matching"] = forged
    evidence_case["discovery"]["markets_by_event"] = {}
    evidence_case["market_objects"] = {}
    report = assessment.build_assessment_from_evidence(**evidence_case)
    cpi = by_family(report)["US_INFLATION"]
    assert cpi["probability"]["pricing_measure_Q_interval"] is None
    assert cpi["impact"]["priced_in"]["yes_pnl_usd"] is None


def test_export_cannot_preserve_a_conflicting_model_proposed_priced_in_alias(evidence_case):
    evidence_case["discovery"]["events"][1]["priced_in"] = {
        "status": "FORGED_MODEL_CLAIM", "yes_pnl_usd": 999999999, "no_pnl_usd": -999999999,
    }
    report = assessment.build_assessment_from_evidence(**evidence_case)
    cpi = by_family(report)["US_INFLATION"]
    assert "priced_in" not in cpi or cpi["priced_in"] == cpi["impact"]["priced_in"]
    assert "FORGED_MODEL_CLAIM" not in json.dumps(report)


def fake_acceptance_report(*, claude_status="FAILED", usable_q=True):
    return {
        "verification_status": "PARTIAL_PUBLIC_RUN_AUTHENTICATED_ACCEPTANCE_NOT_COMPLETED",
        "evaluated_at_utc": "2026-10-21T12:00:00+00:00",
        "stages": {"Massive": {"status": "ATTEMPTED", "request_count": 1, "successful_operations": 1},
                   "Claude": {"status": claude_status, "request_count": 1},
                   "Kalshi": {"status": "READS_COMPLETED", "request_count": 1}},
        "events": [{"family": "US_INFLATION", "impact": {
                        "delta_pnl_usd": -100, "delta_pnl_interval_usd": [-110, -90],
                        "priced_in": {"yes_pnl_usd": -75 if usable_q else None, "no_pnl_usd": 25 if usable_q else None}},
                    "probability": {"pricing_measure_Q_interval": [.2, .3] if usable_q else None}}],
        "provider_receipts": [{"matches_receipt": True}], "acquisition_attempts": [{"status": "OK"}],
    }


def test_cli_cannot_count_a_failed_claude_request_as_authenticated_acceptance(tmp_path, monkeypatch):
    from scripts import run_live_assessment as cli
    monkeypatch.setattr(cli, "run_live_assessment", lambda **kwargs: fake_acceptance_report(claude_status="FAILED"))
    exit_code = cli.main(["--three-portfolios", "--output-dir", str(tmp_path)])
    receipt = json.loads((tmp_path / "acceptance_receipt.json").read_text())
    assert exit_code == 2
    assert not receipt["authenticated_acceptance_completed"]


def test_cli_with_no_admitted_event_q_cannot_claim_complete_acceptance(tmp_path, monkeypatch):
    from scripts import run_live_assessment as cli
    monkeypatch.setattr(cli, "run_live_assessment", lambda **kwargs: fake_acceptance_report(claude_status="COMPLETE_SOURCE_BACKED_REVIEW", usable_q=False))
    exit_code = cli.main(["--three-portfolios", "--output-dir", str(tmp_path)])
    receipt = json.loads((tmp_path / "acceptance_receipt.json").read_text())
    assert exit_code == 2
    assert not receipt["authenticated_acceptance_completed"]


def test_public_only_cannot_be_promoted_to_authenticated_acceptance(tmp_path, monkeypatch):
    from scripts import run_live_assessment as cli
    monkeypatch.setattr(cli, "run_live_assessment", lambda **kwargs: fake_acceptance_report(claude_status="COMPLETE_SOURCE_BACKED_REVIEW"))
    exit_code = cli.main(["--three-portfolios", "--public-only", "--output-dir", str(tmp_path)])
    receipt = json.loads((tmp_path / "acceptance_receipt.json").read_text())
    assert exit_code == 2
    assert receipt["public_only_requested"]
    assert not receipt["authenticated_acceptance_completed"]


def test_cli_positive_control_requires_all_documented_stage_and_event_gates(tmp_path, monkeypatch):
    from scripts import run_live_assessment as cli
    report = fake_acceptance_report(claude_status="COMPLETE_SOURCE_BACKED_REVIEW")
    report["verification_status"] = "LIVE_READS_RECORDED_REVIEW_EACH_COVERAGE_AND_MODEL_GATE"
    monkeypatch.setattr(cli, "run_live_assessment", lambda **kwargs: deepcopy(report))
    exit_code = cli.main(["--three-portfolios", "--output-dir", str(tmp_path)])
    receipt = json.loads((tmp_path / "acceptance_receipt.json").read_text())
    assert exit_code == 0
    assert receipt["authenticated_acceptance_completed"]
    assert len(receipt["cases"]) == 3

