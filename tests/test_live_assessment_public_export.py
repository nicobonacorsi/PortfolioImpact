"""Derived exports keep aggregate conclusions without private provider datasets."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from castle_risk.instrument_exposures import build_instrument_exposures
from castle_risk.live_assessment import build_assessment_from_evidence, public_assessment_view


NOW = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
PORTFOLIO = [{"symbol": "NVDA", "weight": .35, "asset_type": "stock"},
             {"symbol": "QQQ", "weight": .45, "asset_type": "etf"},
             {"symbol": "TLT", "weight": .2, "asset_type": "etf"}]
PRIVATE_PATH = "/SYNTHETIC_PRIVATE_ROOT_DO_NOT_EXPORT/private_data.json"
SOURCE_HASH = "a" * 64


def _private_report():
    # Synthetic rows intentionally include unique constituent markers so that a
    # blacklist that misses a duplicate representation cannot silently pass.
    holdings = {"QQQ": {
        "fund_symbol": "QQQ", "as_of": "2026-10-20", "retrieved_at": "2026-10-21T11:00:00+00:00",
        "source_url": "https://api.massive.com/etf-global/v1/holdings",
        "reliable": True, "complete": True, "weight_basis": "fraction_of_nav",
        "holdings": [{"symbol": "NVDA", "weight": .1, "asset_type": "stock"},
                     {"symbol": "VENDORPRIVATEA", "weight": .4, "asset_type": "stock"},
                     {"symbol": "VENDORPRIVATEB", "weight": .5, "asset_type": "stock"}],
    }}
    exposures = build_instrument_exposures(PORTFOLIO, holdings=holdings,
        notional=100000., instrument_metadata={}, evaluated_at=NOW)
    discovery = {"events": [{
        "event_id": "synthetic-taiwan", "title": "Synthetic Taiwan channel",
        "family": "TAIWAN_SUPPLY", "event_date": "2026-12-31",
        "description": "Synthetic documented channel, not a provider forecast",
        "economic_channel_components": exposures["channels_by_family"]["TAIWAN_SUPPLY"],
        "candidate_origin": "SYNTHETIC_UNIT_TEST",
    }, {
        "event_id": "synthetic-macro", "title": "Synthetic macro scenario",
        "family": "FED_POLICY", "event_date": "2026-10-28",
        "description": "Synthetic event with missing calibrated contrast",
        "economic_channel_components": exposures["channels_by_family"]["FED_POLICY"],
        "exposure_channels": exposures["channels_by_family"]["FED_POLICY"],
        "candidate_origin": "SYNTHETIC_UNIT_TEST",
    }], "markets_by_event": {}, "matches_by_event": {}}
    source = {"source_url": "https://api.massive.com/v2/aggs/ticker/QQQ/range/1/day/2025-01-01/2026-10-20",
              "source_response_sha256": SOURCE_HASH, "retrieved_at": "2026-10-21T11:00:00+00:00",
              "raw_path": PRIVATE_PATH, "pages": [{"raw_path": PRIVATE_PATH, "source_response_sha256": SOURCE_HASH}]}
    report = build_assessment_from_evidence(portfolio_rows=PORTFOLIO, notional=100000.,
        inputs={"holdings": holdings, "instrument_metadata": {}, "prices_by_symbol": {},
                "price_sources": {"QQQ": source}, "benchmark_symbol": "SPY", "attempts": [], "limitations": []},
        discovery=discovery, market_objects={}, evaluated_at=NOW,
        scenario_settings={"TAIWAN_SUPPLY": {"nvda_return_contrast": -.2}},
        release_dates={"FOMC": {"status": "OFFICIAL_LINK_DATES", "dates": [],
                              "sha256": SOURCE_HASH, "raw_path": PRIVATE_PATH}})
    report["run_directory"] = "/SYNTHETIC_PRIVATE_ROOT_DO_NOT_EXPORT"
    report["private_replay_inputs"] = {"holdings": holdings}
    report["additional_nested_receipt"] = {
        "raw_paths": [PRIVATE_PATH], "normalized_path": PRIVATE_PATH,
        "request_body": {"synthetic_private_payload": True},
        "prices_by_symbol": {"QQQ": [{"date": "2026-10-20", "close": 999.123456}]},
        "source_response_sha256": SOURCE_HASH,
    }
    return report


def test_public_view_omits_private_paths_and_constituents_in_all_known_representations():
    report = _private_report()
    before = deepcopy(report)
    view = public_assessment_view(report)
    encoded = json.dumps(view, allow_nan=False)
    assert report == before, "Preparing a shareable view must not destroy the local replay evidence"
    assert "SYNTHETIC_PRIVATE_ROOT_DO_NOT_EXPORT" not in encoded
    assert "VENDORPRIVATEA" not in encoded and "VENDORPRIVATEB" not in encoded
    assert "999.123456" not in encoded
    assert "agent_exposure_input" not in view["exposures"]
    ledger = view["exposures"]["exposure_ledger"]
    assert "entries" not in ledger and "identified_by_symbol" not in ledger
    assert ledger["constituent_rows_omitted"] is True
    aggregate = {row["original_symbol"]: row for row in ledger["aggregated_original_positions"]}
    assert sum(row["total_weight"] for row in aggregate.values()) == pytest.approx(1.)
    assert aggregate["QQQ"]["total_weight"] == pytest.approx(.45)
    assert aggregate["QQQ"]["identified_nvda_weight"] == pytest.approx(.045)
    assert aggregate["TLT"]["unresolved_weight"] == pytest.approx(.2)
    assert view["source_response_references"]["QQQ"]["source_response_sha256"] == SOURCE_HASH
    assert view["release_date_sources"]["FOMC"]["sha256"] == SOURCE_HASH
    assert "NO_MASSIVE_RAW_DATA" in view["export_scope"]


def test_public_view_preserves_unknowns_declared_scope_magnitude_and_research_channel_mass():
    report = _private_report()
    view = public_assessment_view(report)
    by_id = {event["event_id"]: event for event in view["events"]}
    private_events = {event["event_id"]: event for event in report["events"]}
    for event_id, event in by_id.items():
        assert event["impact"] == private_events[event_id]["impact"]
        assert event["probability"] == private_events[event_id]["probability"]
        assert event["ranking"] == private_events[event_id]["ranking"]
        assert event["probability"]["pricing_measure_Q_interval"] is None
        assert event["impact"]["delta_pnl_usd"] is None
    taiwan = by_id["synthetic-taiwan"]["impact"]
    assert taiwan["scope"] == "IDENTIFIED_NVDA_CHANNEL_ONLY"
    assert taiwan["known_component_delta_pnl_usd"] == pytest.approx(-7900.)
    for family, rows in view["exposures"]["channels_by_family"].items():
        assert sum(row["portfolio_weight"] for row in rows) == pytest.approx(
            sum(row["portfolio_weight"] for row in report["exposures"]["channels_by_family"][family]))
        assert all("instrument_symbol" not in row and "ledger_index" not in row for row in rows)
