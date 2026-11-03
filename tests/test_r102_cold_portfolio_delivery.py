"""Check the portfolio editor's HTTP path with a new four-asset input.

The injected runner verifies request wiring without spending API credits. A
separate test verifies that the real server requests provider configuration.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from unittest.mock import patch

import pytest

from scripts.serve_portfolio_impact import PortfolioImpactServer


COLD_HOLDINGS = [
    {"symbol": "MSFT", "type": "auto", "amount": 30_000},
    {"symbol": "JPM", "type": "auto", "amount": 25_000},
    {"symbol": "TLT", "type": "auto", "amount": 25_000},
    {"symbol": "GLD", "type": "auto", "amount": 20_000},
]


def _request(server, method, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Content-Type": "application/json", "X-Portfolio-Impact": "1"}
    connection.request(
        method, path,
        body=json.dumps(payload) if payload is not None else None,
        headers=headers,
    )
    response = connection.getresponse()
    code = response.status
    body = response.read()
    connection.close()
    return code, json.loads(body)


@pytest.fixture
def running_server(tmp_path):
    accepted = []

    def runner(rows, total, progress):
        accepted.append({"rows": rows, "total": total})
        progress("Checking positions")
        return {
            "user_supplied": True,
            "positions": rows,
            "portfolio_value": total,
            "risks": [{"id": "test-event", "title": "Test event"}],
        }

    server = PortfolioImpactServer(
        ("127.0.0.1", 0), runner=runner, converter=lambda report: report,
        saved_runs_root=tmp_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, accepted
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_four_new_holdings_reach_the_async_runner_without_weight_loss(running_server):
    server, accepted = running_server
    status, started = _request(server, "POST", "/api/analyze", {"positions": COLD_HOLDINGS})
    assert status == 202
    assert started["job_id"]

    for _ in range(100):
        status, job = _request(server, "GET", "/api/jobs/" + started["job_id"])
        if job["status"] == "complete":
            break
        time.sleep(.01)

    assert status == 200
    assert job["status"] == "complete"
    assert len(accepted) == 1
    assert accepted[0]["total"] == 100_000
    assert [item["symbol"] for item in accepted[0]["rows"]] == [
        "MSFT", "JPM", "TLT", "GLD",
    ]
    assert [item["weight"] for item in accepted[0]["rows"]] == [.3, .25, .25, .2]
    assert all(item["input_asset_type"] == "auto" for item in accepted[0]["rows"])
    assert [item["symbol"] for item in job["case"]["positions"]] == [
        "MSFT", "JPM", "TLT", "GLD",
    ]


def test_live_server_requires_sources_before_starting_a_new_analysis(tmp_path):
    with patch.dict("os.environ", {
        "MASSIVE_API_KEY": "", "POLYGON_API_KEY": "", "ANTHROPIC_API_KEY": "",
    }):
        server = PortfolioImpactServer(("127.0.0.1", 0), saved_runs_root=tmp_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            code, response = _request(server, "POST", "/api/analyze", {
                "positions": COLD_HOLDINGS,
            })
            assert code == 428
            assert "provider" in response["error"].lower()
            assert server.jobs == {}
            assert server.active_job is None
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
