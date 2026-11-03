"""No-network HTTP checks for the private Codespaces setup and approved v4."""
from __future__ import annotations

import json
from http.client import HTTPConnection
import threading
import time

from castle_risk.codespaces_server import CodespacesPortfolioServer

HOST = "castle-test-8765.app.github.dev"
ORIGIN = "https://" + HOST


def request(server, method, path, payload=None, *, origin=None, csrf=None, host=HOST, forwarded_host=None, fetch_site=None):
    conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Host": host}
    encoded = None
    if forwarded_host is not None:
        headers["X-Forwarded-Host"] = forwarded_host
    if origin is not None:
        headers["Origin"] = origin
    if csrf is not None:
        headers["X-CSRF-Token"] = csrf
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site
    if payload is not None:
        encoded = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["X-Portfolio-Impact"] = "1"
    try:
        conn.request(method, path, body=encoded, headers=headers)
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        conn.close()


def test_private_codespaces_provider_setup_and_v4(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    called = []

    def mock_runner(rows, total, progress):
        called.append((rows, total))
        return {"ready": True}

    server = CodespacesPortfolioServer(
        ("127.0.0.1", 0), origin=ORIGIN,
        runner=mock_runner,
        converter=lambda _: {"user_supplied": True, "risks": []},
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        status, _, headers = request(server, "GET", "/")
        assert status == 302 and headers["Location"] == "/connections"
        assert request(server, "GET", "/", fetch_site="cross-site")[0] == 302
        assert request(server, "GET", "/", origin="https://github.dev", fetch_site="cross-site")[0] == 302
        assert request(server, "GET", "/connections")[0] == 200
        assert request(server, "GET", "/connections.js")[0] == 200
        assert request(server, "GET", "/index.html")[0] == 200
        assert request(server, "GET", "/styles.css")[0] == 200
        assert request(server, "GET", "/workspace.css")[0] == 200
        assert request(server, "GET", "/api/status", host="evil.example")[0] == 403
        assert request(server, "GET", "/api/status", host="evil.example", forwarded_host="evil.example")[0] == 403
        # Real Codespaces proxies can rewrite Host before forwarding the request.
        loopback_host = "127.0.0.1:" + str(server.server_port)
        assert request(server, "GET", "/connections", host=loopback_host)[0] == 200
        assert request(server, "GET", "/api/status", host=loopback_host, forwarded_host=HOST)[0] == 200
        assert request(server, "GET", "/api/status", host=loopback_host, forwarded_host="evil.example")[0] == 403
        status, content, _ = request(server, "GET", "/api/status")
        assert status == 200
        initial = json.loads(content)
        token = initial["csrf_token"]
        assert len(token) == 43 and initial["provider_setup_allowed"]
        assert not initial["data_provider_configured"]
        assert not initial["research_model_configured"]
        fake_keys = {"massive": "test-market-data-secret", "anthropic": "test-model-secret"}
        # Origin and Fetch Metadata may be rewritten by the private port proxy.
        # Neither they nor the Host header can replace the secret CSRF token.
        assert request(server, "POST", "/api/providers", fake_keys, origin=ORIGIN)[0] == 403
        assert request(server, "POST", "/api/providers", fake_keys, origin="https://evil.example", csrf="wrong", fetch_site="cross-site")[0] == 403
        assert request(server, "POST", "/api/providers", fake_keys, origin=ORIGIN, csrf="wrong")[0] == 403
        assert request(server, "POST", "/api/providers", fake_keys, origin=ORIGIN, csrf=token, host=loopback_host, forwarded_host="evil.example")[0] == 403
        # GitHub's private proxy can relay metadata from the editor origin.
        assert request(server, "POST", "/api/providers", fake_keys, origin="https://github.dev", csrf=token, host=loopback_host, forwarded_host=HOST, fetch_site="same-site")[0] == 200
        assert request(server, "POST", "/api/providers", fake_keys, origin=ORIGIN, csrf=token, fetch_site="same-origin")[0] == 200
        assert request(server, "POST", "/api/providers", fake_keys, csrf=token)[0] == 200
        status, content, _ = request(server, "GET", "/api/status")
        assert status == 200
        configured = json.loads(content)
        assert configured["data_provider_configured"] and configured["research_model_configured"]
        assert "test-market-data-secret" not in content.decode()
        assert "test-model-secret" not in content.decode()
        assert request(server, "GET", "/")[0] == 200
        status, content, _ = request(
            server, "POST", "/api/analyze",
            {"positions": [{"symbol": "AAPL", "type": "auto", "amount": 10000.0}]},
            origin=ORIGIN, csrf=token,
        )
        assert status == 202
        job_id = json.loads(content)["job_id"]
        assert job_id
        for _ in range(100):
            result = request(server, "GET", "/api/jobs/" + job_id)
            if result[0] == 200 and json.loads(result[1])["status"] == "complete":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Mock analysis did not complete")
        assert called and called[0][1] == 10000.0
        assert request(server, "GET", "/not-existing")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_codespaces_origin_detection(monkeypatch):
    from castle_risk.codespaces_server import codespaces_origin
    monkeypatch.setenv("CODESPACE_NAME", "demo-space")
    monkeypatch.setenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
    assert codespaces_origin(8765) == "https://demo-space-8765.app.github.dev"
    monkeypatch.delenv("CODESPACE_NAME")
    assert codespaces_origin(8765) is None
