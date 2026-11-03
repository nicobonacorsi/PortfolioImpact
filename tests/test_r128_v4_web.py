"""Verify the pinned application assets and private-preview HTTP boundary."""
from __future__ import annotations

import base64
import hashlib
from http.client import HTTPConnection
from pathlib import Path
import threading

from castle_risk.public_server import PublicPortfolioServer


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "web" / "portfolio_impact_v4"
SOURCE_BLOB_SHA = {
    "app.js": "bd972d2e8a2acffaf8b280aca6dc30a4c23e7744",
    "index.html": "87555a1a76c78c4f17f3105f072c7a4441a35ce4",
    "styles.css": "06bebb08f649203b6ccd5a198570804ad81e1019",
    "workspace.css": "fb753af34ac3556558ff5793a1ae729c3dda86b3",
    "market_scenarios_20261025.json": "29acd9638ad16fb91f27203ca1f0001230301390",
    "r125_defense_semis_japan_credit_acceptance_summary_20261102.json": "b4b7f18722feb393030211298a5d8b3a7ae56ed3",
    "fonts/InstrumentSerif-Italic.woff2": "e1f3123a86f0196825de76d34f32fc02b6bb6b12",
    "fonts/InstrumentSerif-OFL.txt": "669ce7917bec662fcf7244d37f49c4277f6fbd73",
    "fonts/PlusJakartaSans-OFL.txt": "ea69b0ca12a02556c436680ea0a159efb7a748fa",
    "fonts/PlusJakartaSans-variable.woff2": "eebac44580986b89ed54b1dfc23209ce82ebc073",
}


def _get(server, path, authorization=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Host": "castle-preview.example"}
    if authorization is not None:
        headers["Authorization"] = authorization
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


def test_application_assets_match_pinned_git_blobs():
    for path, expected in SOURCE_BLOB_SHA.items():
        content = (V4 / path).read_bytes()
        prefix = ("blob " + str(len(content))).encode() + bytes([0])
        assert hashlib.sha1(prefix + content).hexdigest() == expected, path


def test_authenticated_single_origin_v4_static_routes(monkeypatch):
    monkeypatch.setenv("CASTLE_PREVIEW_PASSWORD", "test-secret-not-for-production")
    server = PublicPortfolioServer(
        ("127.0.0.1", 0),
        origin="https://castle-preview.example",
        runner=lambda *args: {},
        asset_root=V4,
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert _get(server, "/healthz")[0] == 200
        assert _get(server, "/")[0] == 401
        assert _get(server, "/api/status")[0] == 401
        incorrect = "Basic " + base64.b64encode(b"castle:wrong").decode()
        assert _get(server, "/", incorrect)[0] == 401
        correct = "Basic " + base64.b64encode(b"castle:test-secret-not-for-production").decode()
        status, html, _ = _get(server, "/", correct)
        assert status == 200
        assert b"<title>Portfolio Impact</title>" in html
        for path in (
            "/app.js", "/styles.css", "/workspace.css",
            "/market_scenarios_20261025.json",
            "/r125_defense_semis_japan_credit_acceptance_summary_20261102.json",
            "/fonts/PlusJakartaSans-variable.woff2",
        ):
            assert _get(server, path, correct)[0] == 200, path
        assert _get(server, "/private.txt", correct)[0] == 404
        assert _get(server, "/api/status", correct)[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
