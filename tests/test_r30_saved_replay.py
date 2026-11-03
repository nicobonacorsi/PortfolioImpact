"""Offline saved-run upgrade, with explicitly synthetic relocated input bundles.

The market rows are archival public fixtures wrapped in a synthetic test
receipt. Their timestamps are constructed here, never changed by replay, and
are not claims of a new provider capture or a calibrated forecast.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from castle_risk import adapters
from castle_risk.a8_offline_demo import archived_a8_demo
from castle_risk.adapters import normalize_market
from castle_risk.option1_brief import portfolio_input_fingerprint
from castle_risk.saved_assessment_replay import upgrade_saved_assessment
from test_round8_default_integration import synthetic_report


RUN = "20261028T001737685793Z"
CAPTURE = "2026-10-28T00:17:37.685793+00:00"
OLD_ROOT = "C:\\OldCastle\\runtime\\option1_live\\" + RUN


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Saved replay may never call a provider, network or LLM")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(adapters, "_transport", blocked)
    monkeypatch.setattr(adapters._ReadOnlyClient, "_get", blocked)


def _write_report(root, report):
    body = json.dumps(report, sort_keys=True, indent=2).encode()
    (root / "derived_assessment.json").write_bytes(body)
    (root / "derived_assessment.sha256").write_text(sha256(body).hexdigest() + "  derived_assessment.json\n")


def _write_private(root, private):
    (root / "private_replay_inputs.json").write_text(json.dumps(private, sort_keys=True, indent=2))


def saved_fixture(tmp_path, *, with_prices=False):
    root = tmp_path / "new-install" / "runtime" / "option1_live" / RUN
    (root / "kalshi_final_public").mkdir(parents=True)
    _, archived = archived_a8_demo()
    first = next(m for m in archived.values() if m.event_ticker == "KXFEDDECISION-26OCT")
    source = json.loads(Path(first.raw_path).read_bytes())
    rows = [row for row in source["markets"] if row.get("event_ticker") == first.event_ticker]
    body = json.dumps({"markets": rows, "cursor": ""}, sort_keys=True).encode()
    raw = root / "kalshi_final_public" / "test-cohort.json"
    raw.write_bytes(body)
    raw.with_suffix(".meta.json").write_text(json.dumps({
        "url": "https://external-api.kalshi.com/trade-api/v2/markets?event_ticker=" + first.event_ticker,
        "retrieved_at": CAPTURE, "sha256": sha256(body).hexdigest(), "fixture": False,
        "status": 200, "test_data": "SYNTHETIC_TEST_RECEIPT_NOT_LIVE_CAPTURE",
    }))
    old_path = OLD_ROOT + "\\kalshi_final_public\\test-cohort.json"
    markets = {row["ticker"]: normalize_market(row, CAPTURE, old_path).to_dict() for row in rows}
    report = synthetic_report()
    report.update(schema="castle.option1.live_assessment.v2", evaluated_at_utc=CAPTURE,
                  verification_status="ORIGINAL_SYNTHETIC_ACCEPTANCE_NOT_UPGRADED",
                  run_directory=OLD_ROOT, started_at_utc="2026-10-27T23:57:00+00:00")
    geo = {"event_id": "synthetic-taiwan", "family": "TAIWAN_SUPPLY", "title": "Retained research event",
           "event_date": "2027-01-19", "status": "ORIGINAL_RELATED_ONLY",
           "market_search": {"candidates": [{"source_quote": [.02, .04], "decision": "RELATED_ONLY"}]},
           "original_matching_comparisons": [{"decision": "UNKNOWN", "reason": "Original source rules"}],
           "ranking": {"abs_delta_pnl_usd": None, "display_order": 1},
           "risk_ranking": {"q_standard_deviation_usd": None}}
    report["default_analysis"] = {"events": [geo], "evaluated_at_utc": CAPTURE}
    report["geopolitical_search"] = {"targets": [{"event_id": "synthetic-taiwan", "status": "PRESERVED"}]}
    inputs = {"holdings": {}, "instrument_metadata": {}, "prices_by_symbol": {}, "price_sources": {},
              "evaluated_at_utc": "2026-10-27T23:57:00+00:00"}
    if with_prices:
        (root / "massive_private").mkdir()
        prices_raw = root / "massive_private" / "nvda.json"
        raw_bars = [{"t": int(datetime(2026, 10, 23, tzinfo=timezone.utc).timestamp() * 1000), "c": 100.0},
                    {"t": int(datetime(2026, 10, 26, tzinfo=timezone.utc).timestamp() * 1000), "c": 101.0}]
        price_body = json.dumps({"ticker": "NVDA", "adjusted": True, "status": "OK", "results": raw_bars}).encode()
        prices_raw.write_bytes(price_body)
        digest = sha256(price_body).hexdigest()
        prices_raw.with_suffix(".meta.json").write_text(json.dumps({"status": 200, "fixture": False,
            "sha256": digest, "url": "https://api.massive.com/v2/aggs/ticker/NVDA/range/1/day/2026-10-23/2026-10-26",
            "retrieved_at": inputs["evaluated_at_utc"], "test_data": "SYNTHETIC_PRICE_FIXTURE"}))
        inputs["prices_by_symbol"]["NVDA"] = [{"date": "2026-10-23", "close": 100.0},
                                                {"date": "2026-10-26", "close": 101.0}]
        inputs["price_sources"]["NVDA"] = {"receipts": [{"raw_path": OLD_ROOT + "\\massive_private\\nvda.json",
            "source_response_sha256": digest, "retrieved_at": inputs["evaluated_at_utc"]}]}
    private = {"inputs": inputs, "market_objects": markets, "payroll_markets": {},
               "discovery": {"ignored_original_discovery": True}}
    report["input_fingerprint"] = portfolio_input_fingerprint(report["portfolio"]["positions"],
        report["portfolio"]["notional"], inputs["holdings"])
    _write_report(root, report)
    _write_private(root, private)
    return root, report, private


def test_relocated_windows_capture_is_recomputed_without_writes_or_new_times(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original = deepcopy(report)
    result = upgrade_saved_assessment(report, root)
    assert report == original
    after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert result["offline_replay"]["status"] == "OFFLINE_SAVED_INPUTS_RECOMPUTED_R30"
    assert result["offline_replay"]["new_provider_calls"] == result["offline_replay"]["new_model_calls"] == 0
    assert result["evaluated_at_utc"] == CAPTURE
    assert result["verification_status"] == original["verification_status"]
    assert result["empirical"] == original["empirical"]
    assert result["geopolitical_search"] == original["geopolitical_search"]
    fed = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "fomc-2026-10-28-scheduled")
    assert fed["economics"] is not None
    assert fed["distribution"]["source"]["retrieved_at_utc"] == CAPTURE
    assert fed["distribution"]["evaluated_at"] == CAPTURE
    assert fed["schedule_window"]["scheduled_release_at_utc"] == "2026-10-28T18:00:00+00:00"
    retained = next(e for e in result["default_analysis"]["events"] if e["event_id"] == "synthetic-taiwan")
    assert retained["market_search"] == original["default_analysis"]["events"][0]["market_search"]
    assert retained["original_matching_comparisons"] == original["default_analysis"]["events"][0]["original_matching_comparisons"]


def test_missing_private_bundle_keeps_original_results(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    (root / "private_replay_inputs.json").unlink()
    result = upgrade_saved_assessment(report, root)
    assert result["default_analysis"] == report["default_analysis"]
    assert result["offline_replay"]["reason_code"] == "PRIVATE_INPUTS_NOT_SAVED"


def test_original_report_hash_mismatch_blocks_replay(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    (root / "derived_assessment.sha256").write_text("0" * 64 + "  derived_assessment.json")
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "ORIGINAL_DERIVED_REPORT_SHA256_MISMATCH"
    assert result["default_analysis"] == report["default_analysis"]


def test_expected_report_must_match_chosen_local_file(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    report["verification_status"] = "ALTERED_IN_MEMORY"
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "EXPECTED_REPORT_DOES_NOT_MATCH_SAVED_REPORT"


def test_missing_optional_original_digest_is_disclosed(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    (root / "derived_assessment.sha256").unlink()
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "OFFLINE_SAVED_INPUTS_RECOMPUTED_R30"
    assert result["offline_replay"]["original_digest_present"] is False


@pytest.mark.parametrize("path", ["/etc/passwd", OLD_ROOT + "\\..\\other-run\\secret.json"])
def test_arbitrary_and_traversing_paths_cannot_be_followed(tmp_path, path):
    root, report, private = saved_fixture(tmp_path)
    first = next(iter(private["market_objects"].values()))
    first["raw_path"] = path
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "ORIGINAL_RESULTS_RETAINED"
    assert result["offline_replay"]["reason_code"] in {"SAVED_PATH_NOT_ANCHORED_TO_THIS_RUN", "SAVED_PATH_TRAVERSAL"}
    assert result["default_analysis"] == report["default_analysis"]
    assert path not in json.dumps(result["offline_replay"])


def test_report_run_directory_is_never_used_to_find_data(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    report["run_directory"] = "/outside/should/not/be/read"
    _write_report(root, report)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "OFFLINE_SAVED_INPUTS_RECOMPUTED_R30"


def test_symlink_source_is_rejected(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    source = root / "kalshi_final_public" / "test-cohort.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit this test's symlink")
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SYMLINK_SAVED_MEMBER_REJECTED"


def test_source_bytes_hash_mismatch_keeps_original(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    source = root / "kalshi_final_public" / "test-cohort.json"
    source.write_bytes(source.read_bytes() + b" ")
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_SOURCE_RECEIPT_MISMATCH"


def test_altered_normalized_market_cannot_replace_original_quote(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    next(iter(private["market_objects"].values()))["yes_bid"] = 0.4321
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_MARKET_SOURCE_MISMATCH"


def test_saved_prices_are_rebound_to_source_bars(tmp_path):
    root, report, _ = saved_fixture(tmp_path, with_prices=True)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "OFFLINE_SAVED_INPUTS_RECOMPUTED_R30"
    assert result["offline_replay"]["local_source_receipts_verified"] == 2


def test_normalized_price_tampering_is_not_used_in_replay(tmp_path):
    root, report, private = saved_fixture(tmp_path, with_prices=True)
    private["inputs"]["prices_by_symbol"]["NVDA"][0]["close"] = 123.0
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_NORMALIZED_PRICES_CHANGED"


def test_changed_holdings_fail_original_fingerprint(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    private["inputs"]["holdings"] = {"UNRELATED": {"fixture": True}}
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "ORIGINAL_INPUT_FINGERPRINT_MISMATCH"


def test_later_inputs_do_not_acquire_original_capture_time(tmp_path):
    root, report, private = saved_fixture(tmp_path)
    private["inputs"]["evaluated_at_utc"] = "2026-10-29T00:00:00Z"
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_INPUTS_POSTDATE_CAPTURE"


def test_corrupt_private_json_returns_safe_status(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    (root / "private_replay_inputs.json").write_text('{"secret": "PRIVATE_TEST_VALUE",')
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "ORIGINAL_RESULTS_RETAINED"
    assert "PRIVATE_TEST_VALUE" not in json.dumps(result["offline_replay"])


def test_missing_source_receipts_do_not_trigger_a_network_refetch(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    (root / "kalshi_final_public" / "test-cohort.meta.json").unlink()
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["status"] == "ORIGINAL_RESULTS_RETAINED"
    assert result["offline_replay"]["new_provider_calls"] == 0


def test_later_source_receipt_cannot_be_used_at_an_earlier_capture(tmp_path):
    root, report, _ = saved_fixture(tmp_path, with_prices=True)
    meta_path = root / "massive_private" / "nvda.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["retrieved_at"] = "2026-10-29T00:00:00Z"
    meta_path.write_text(json.dumps(meta))
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_SOURCE_POSTDATES_CAPTURE"
    assert result["default_analysis"] == report["default_analysis"]


def test_saved_normalized_price_receipt_time_must_match_original(tmp_path):
    root, report, private = saved_fixture(tmp_path, with_prices=True)
    private["inputs"]["price_sources"]["NVDA"]["receipts"][0]["retrieved_at"] = CAPTURE
    _write_private(root, private)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "SAVED_PRICE_RECEIPT_TIME_MISMATCH"


def test_malformed_original_event_list_is_retained_without_crashing(tmp_path):
    root, report, _ = saved_fixture(tmp_path)
    report["default_analysis"]["events"] = [None]
    _write_report(root, report)
    result = upgrade_saved_assessment(report, root)
    assert result["offline_replay"]["reason_code"] == "UNSUPPORTED_SAVED_REPLAY_INPUTS"
    assert result["default_analysis"] == report["default_analysis"]
