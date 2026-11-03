"""Offline restart/replay checks: no real provider transports or API keys."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from castle_risk.adapters import KalshiClient
from castle_risk.agent import AnthropicClient
from castle_risk.restart_recovery import (
    find_matching_legacy_run, load_job_records, portfolio_fingerprint,
    prime_verified_http_cache, save_job_records,
)
from scripts.serve_portfolio_impact import PortfolioImpactServer, parse_positions


def _receipt(directory: Path, url: str, *, age_minutes: int = 3,
             content: dict | None = None, correct_hash: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    raw_path = directory / ("20261103T040000_" + key[:12] + ".json")
    raw = json.dumps(content or {"markets": [], "cursor": ""}).encode()
    raw_path.write_bytes(raw)
    metadata = {
        "url": url,
        "retrieved_at": (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat(),
        "status": 200,
        "sha256": hashlib.sha256(raw).hexdigest() if correct_hash else "0" * 64,
        "raw_path": str(raw_path),
        "fixture": False,
        "cache_hit": False,
    }
    raw_path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return raw_path


def test_replays_only_integrity_checked_original_get_without_network(tmp_path):
    url = "https://external-api.kalshi.com/trade-api/v2/markets?limit=100"
    folder = tmp_path / "kalshi_public"
    raw_path = _receipt(folder, url, age_minutes=20)
    def reject_network(*args):
        raise AssertionError("GET was not supposed to execute")
    client = KalshiClient(folder, cache_ttl=0, transport=reject_network)
    assert prime_verified_http_cache(client, folder) == 1
    payload, meta = client._get(url)
    assert payload == {"markets": [], "cursor": ""}
    assert meta["replayed_from_archive"] is True
    assert meta["raw_path"] == str(raw_path)
    assert meta["retrieved_at"] != datetime.now(timezone.utc).isoformat()
    assert client.request_count == 0


def test_corrupt_expired_and_other_origin_archives_are_not_replayed(tmp_path):
    folder = tmp_path / "saved"
    url = "https://external-api.kalshi.com/trade-api/v2/markets?limit=90"
    _receipt(folder, url, correct_hash=False)
    _receipt(folder, "https://example.org/api/markets")
    _receipt(folder, "https://external-api.kalshi.com/trade-api/v2/markets?limit=91",
             age_minutes=800)
    client = KalshiClient(folder, cache_ttl=0, transport=lambda *args: (_ for _ in ()).throw(
        AssertionError("network transport invoked")))
    assert prime_verified_http_cache(client, folder) == 0
    with pytest.raises(AssertionError, match="network transport"):
        client._get(url)


def test_portfolio_fingerprint_ignores_order_but_not_weights():
    rows, total = parse_positions({"positions": [
        {"symbol": "NVDA", "type": "auto", "amount": 30000},
        {"symbol": "TSM", "type": "auto", "amount": 25000},
        {"symbol": "LMT", "type": "auto", "amount": 25000},
        {"symbol": "TLT", "type": "auto", "amount": 20000},
    ]})
    assert total == 100_000
    fingerprint = portfolio_fingerprint(rows, total)
    assert fingerprint == portfolio_fingerprint(list(reversed(rows)), total)
    changed = [dict(row) for row in rows]
    changed[0]["weight"] = 0.31
    assert fingerprint != portfolio_fingerprint(changed, total)


def test_legacy_run_adoption_requires_all_four_tickers(tmp_path):
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    saved = tmp_path / run_name / "massive_private"
    saved.mkdir(parents=True)
    wanted = {"NVDA", "TSM", "LMT", "TLT"}
    for symbol in wanted:
        (saved / (symbol + ".meta.json")).write_text(json.dumps(
            {"url": "https://api.massive.com/v3/reference/tickers/" + symbol}),
            encoding="utf-8")
    assert find_matching_legacy_run(tmp_path, wanted) == run_name
    assert find_matching_legacy_run(tmp_path, wanted | {"GLD"}) is None
    (saved.parent / "derived_assessment.json").write_text("{}", encoding="utf-8")
    assert find_matching_legacy_run(tmp_path, wanted) is None


def test_server_restores_interrupted_job_and_reuses_its_identifier(tmp_path):
    rows, total = parse_positions({"positions": [
        {"symbol": "NVDA", "amount": 30000},
        {"symbol": "TSM", "amount": 25000},
        {"symbol": "LMT", "amount": 25000},
        {"symbol": "TLT", "amount": 20000},
    ]})
    job_id = "restart_test_job_abcdefghi"
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (tmp_path / run_name).mkdir()
    save_job_records(tmp_path, [{
        "id": job_id, "fingerprint": portfolio_fingerprint(rows, total),
        "run_name": run_name, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": "Researching event 5/8",
    }])
    finished = threading.Event()
    def fake_runner(_rows, _total, progress):
        progress("No external calls")
        finished.set()
        return {"risks": [1]}
    server = PortfolioImpactServer(("127.0.0.1", 0), asset_root=tmp_path,
                                   runner=fake_runner, converter=lambda result: result,
                                   saved_runs_root=tmp_path)
    try:
        assert server.job(job_id)["status"] == "interrupted"
        assert server.start_job(rows, total) == job_id
        assert finished.wait(timeout=5)
        for _ in range(100):
            if server.job(job_id)["status"] == "complete":
                break
            time.sleep(.01)
        assert server.job(job_id)["status"] == "complete"
        assert load_job_records(tmp_path)[0]["status"] == "complete"
        assert not any("api_key" in str(record) for record in load_job_records(tmp_path))
    finally:
        server.server_close()


class _FakeResponse:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self, *_args):
        return json.dumps({
            "content": [{"type": "text", "text": "offline example"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 2},
        }).encode()


def test_exact_anthropic_response_archive_prevents_second_request(tmp_path):
    saved = tmp_path / "claude"
    transmitted = []
    def first_opener(request, timeout):
        transmitted.append(request.data)
        return _FakeResponse()
    inputs = [{"role": "user", "content": "offline test"}]
    kwargs = dict(api_key="dummy-test-token", model="claude-sonnet-4-5",
                  archive_dir=saved, max_output_tokens=256,
                  min_interval_seconds=0)
    first = AnthropicClient(opener=first_opener, **kwargs)
    assert first.request(inputs, [])["content"][0]["text"] == "offline example"
    assert len(transmitted) == 1
    first_archive = next(saved.glob("claude_*.json"))
    saved_envelope = json.loads(first_archive.read_text(encoding="utf-8"))
    assert saved_envelope["request_sha256"] == hashlib.sha256(transmitted[0]).hexdigest()
    # Model a pre-existing REAL receipt without making a real provider call.
    saved_envelope["provenance"] = "REAL_ANTHROPIC_API_RESPONSE"
    first_archive.write_text(json.dumps(saved_envelope), encoding="utf-8")
    resumed = AnthropicClient(reuse_verified_archives=True, **kwargs)
    resumed._opener = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Anthropic HTTP POST must not execute"))
    result = resumed.request(inputs, [])
    assert result["content"][0]["text"] == "offline example"
    assert resumed.request_count == 0
    assert resumed.request_ledger[0]["status"] == "REPLAYED_VALIDATED_ARCHIVE"
    assert resumed.request_ledger[0]["cost_status"] == "NO_NEW_PROVIDER_REQUEST"


def test_private_v4_serves_additive_recovery_without_modifying_approved_assets(tmp_path):
    assets = Path(__file__).resolve().parents[1] / "web" / "portfolio_impact_v4"
    expected_source = (assets / "index.html").read_bytes()
    assert b"recovery.js" not in expected_source
    server = PortfolioImpactServer(("127.0.0.1", 0), asset_root=assets,
                                   runner=lambda *_args: {}, converter=lambda result: result,
                                   saved_runs_root=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def fetch(path):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", path, headers={"Host": f"127.0.0.1:{server.server_port}"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()
    try:
        status, page = fetch("/")
        assert status == 200
        assert b'<script src="./app.js" defer></script>' in page
        assert b'<script src="./recovery.js" defer></script>' in page
        status, addon = fetch("/recovery.js")
        assert status == 200
        assert b"runAnalysis = async function" in addon
        assert (assets / "index.html").read_bytes() == expected_source
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
