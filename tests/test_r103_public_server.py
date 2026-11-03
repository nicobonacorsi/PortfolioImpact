"""Real HTTP checks for anonymous-session isolation and paid-API controls."""
from contextlib import contextmanager
from dataclasses import replace
import http.client
import io
import json
import threading
import time
from zipfile import ZipFile

import pytest

from castle_risk.public_server import (
    InstrumentIdentityError, Limits, PublicPortfolioHandler, PublicPortfolioServer, PublicRequestError,
)

ORIGIN = "https://castle.example"
POSITIONS = [{"symbol": "MSFT", "type": "auto", "amount": 1000}]
TRANSPORT = "socket"


@pytest.fixture(autouse=True, params=["memory", "socket"])
def transport(request):
    """Keep socket evidence separate from the same parser's memory transport."""
    global TRANSPORT
    TRANSPORT = request.param


class MemoryConnection:
    def __init__(self, request):
        self.input, self.output = io.BytesIO(request), io.BytesIO()

    def makefile(self, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.write(data)


def case(rows, total, progress):
    progress("Private provider diagnostics")
    return {"user_supplied": True, "positions": rows, "portfolio_value": total,
            "risks": [], "capture": "2026-10-31T20:00:00Z"}


class Visitor:
    def __init__(self, server):
        self.server, self.cookie, self.csrf = server, None, None

    def request(self, method, path, payload=None, headers=None, raw=None):
        conn = None
        supplied = {"Host": "castle.example", "Origin": ORIGIN, "X-Portfolio-Impact": "1"}
        if self.cookie:
            supplied["Cookie"] = self.cookie
        if self.csrf:
            supplied["X-CSRF-Token"] = self.csrf
        body = raw
        if payload is not None:
            body = json.dumps(payload)
            supplied["Content-Type"] = "application/json"
        if headers:
            supplied.update(headers)
        if TRANSPORT == "memory":
            body_bytes = body.encode("utf-8") if isinstance(body, str) else body or b""
            if body is not None and "Content-Length" not in supplied:
                supplied["Content-Length"] = str(len(body_bytes))
            request = (f"{method} {path} HTTP/1.0\r\n" + "".join(
                f"{key}: {value}\r\n" for key, value in supplied.items()) + "\r\n").encode("latin-1") + body_bytes
            connection = MemoryConnection(request)
            PublicPortfolioHandler(connection, ("127.0.0.1", 1234), self.server)
            response = http.client.HTTPResponse(MemoryConnection(connection.output.getvalue()))
            response.begin()
        else:
            conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            conn.request(method, path, body=body, headers=supplied)
            response = conn.getresponse()
        code, result_headers, data = response.status, dict(response.getheaders()), response.read()
        if conn is not None:
            conn.close()
        if "Set-Cookie" in result_headers:
            self.cookie = result_headers["Set-Cookie"].split(";", 1)[0]
        content = json.loads(data) if "application/json" in result_headers.get("Content-Type", "") else data
        if path == "/api/status" and code == 200:
            self.csrf = content["csrf_token"]
        return code, content, result_headers

    def status(self):
        return self.request("GET", "/api/status")

    def start(self, positions=POSITIONS, **kwargs):
        return self.request("POST", "/api/analyze", {"positions": positions}, **kwargs)

    def finish(self, job):
        for _ in range(100):
            code, result, _ = self.request("GET", "/api/jobs/" + job)
            if code != 200 or result["status"] not in {"queued", "running"}:
                return code, result
            time.sleep(.01)
        raise AssertionError("Injected analysis did not finish")


@contextmanager
def running(runner=case, **kwargs):
    server = PublicPortfolioServer(("127.0.0.1", 0), origin=ORIGIN, runner=runner,
                                   bind_and_activate=TRANSPORT == "socket", **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    if TRANSPORT == "socket":
        thread.start()
    try:
        yield server
    finally:
        if TRANSPORT == "socket":
            server.shutdown()
        server.server_close()
        if TRANSPORT == "socket":
            thread.join(3)


def test_two_visitors_cannot_read_each_others_job_latest_or_export():
    with running() as server:
        first, second = Visitor(server), Visitor(server)
        code, status, headers = first.status()
        second.status()
        assert code == 200 and status["hosted"] and status["provider_setup_allowed"] is False
        assert all(flag in headers["Set-Cookie"] for flag in ("Secure", "HttpOnly", "SameSite=Strict"))
        code, task, _ = first.start()
        assert code == 202
        assert first.finish(task["job_id"])[1]["status"] == "complete"
        assert second.request("GET", "/api/jobs/" + task["job_id"])[0] == 404
        assert second.request("GET", "/api/latest")[0] == 404
        assert second.request("GET", "/api/export/latest")[0] == 404
        assert second.status()[1]["active_job_id"] is None
        assert first.request("GET", "/api/latest")[1]["case"]["positions"][0]["symbol"] == "MSFT"
        code, data, _ = first.request("GET", "/api/export/latest")
        assert code == 200
        with ZipFile(io.BytesIO(data)) as archive:
            exported = json.loads(archive.read("portfolio_data.json"))
        assert exported["positions"][0]["symbol"] == "MSFT"


def test_scenario_lab_uses_only_completed_owners_portfolio_with_csrf_and_no_paid_reanalysis():
    """User scenario writes must never read another visitor's holdings or restart paid research."""
    def runner(rows, total, progress):
        return {"user_supplied": True, "id": "separate-owner", "notional": total,
                "positions": [{"symbol": p["symbol"], "weight": p["weight"], "type": "stock"}
                              for p in rows], "risks": [], "capture": "2026-10-31T12:00:00Z"}

    with running(runner=runner) as server:
        first, second = Visitor(server), Visitor(server)
        first.status()
        second.status()
        scenario = {"hypothesis": "Tariffs on imports by 10 percentage points",
                    "mode": "today", "position_shocks": {"MSFT": -3}}
        endpoint = "/api/scenarios/evaluate"
        assert first.request("POST", endpoint, scenario)[0] == 409
        job = first.start()[1]["job_id"]
        assert first.finish(job)[1]["status"] == "complete"
        assert second.request("POST", endpoint, scenario)[0] == 409
        assert first.request("POST", endpoint, scenario, headers={"Origin": "https://wrong.example"})[0] == 403
        assert first.request("POST", endpoint, scenario, headers={"X-CSRF-Token": "invalid"})[0] == 403
        code, result, _ = first.request("POST", endpoint, scenario)
        assert code == 200 and result["schema"] == "castle.scenario_lab.v1"
        assert result["portfolio_change_usd"] == -30
        assert result["positions"][0]["symbol"] == "MSFT"
        assert len(server.jobs) == 1
        assert first.request("POST", endpoint, {**scenario, "pricing_weight": .5})[0] == 422
        assert len(server.jobs) == 1


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"}, {"Origin": "https://evil.example"},
    {"Origin": "null"}, {"Origin": ""}, {"X-CSRF-Token": "incorrect"},
    {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},
])
def test_foreign_origin_host_and_forged_csrf_do_not_start_jobs(headers):
    with running() as server:
        visitor = Visitor(server)
        visitor.status()
        assert visitor.start(headers=headers)[0] == 403
        assert server.jobs == {}


def test_provider_override_and_extra_payload_fields_are_rejected():
    with running() as server:
        visitor = Visitor(server)
        visitor.status()
        assert visitor.request("POST", "/api/providers", {"massive": "fake-key"})[0] == 404
        assert visitor.request("POST", "/api/analyze", {"positions": POSITIONS, "massive_key": "fake"})[0] == 422
        assert visitor.start(positions=[{"symbol": "MSFT", "amount": float("nan")}])[0] == 422
        assert visitor.start(positions=POSITIONS * 25)[0] == 422
        assert server.jobs == {}


def test_oversized_and_chunked_requests_are_rejected_before_engine():
    with running() as server:
        visitor = Visitor(server)
        visitor.status()
        assert visitor.request("POST", "/api/analyze", raw="x" * 13000,
                               headers={"Content-Type": "application/json"})[0] == 413
        assert visitor.start(headers={"Transfer-Encoding": "chunked"})[0] == 413
        assert server.jobs == {}


def test_no_session_cannot_start_paid_work():
    with running() as server:
        assert Visitor(server).start()[0] == 403
        assert server.jobs == {}


def test_limits_apply_across_cookie_rotation_and_spoofed_forwarded_ip():
    limits = replace(Limits(), starts_per_ip=1, starts_per_session=4)
    with running(limits=limits) as server:
        one, two = Visitor(server), Visitor(server)
        one.status()
        two.status()
        job = one.start()[1]["job_id"]
        one.finish(job)
        code, _, headers = two.start(headers={"X-Forwarded-For": "1.2.3.4"})
        assert code == 429 and int(headers["Retry-After"]) > 0
        assert len(server.jobs) == 1


def test_active_job_is_retained_and_queue_capacity_rejects_without_new_work():
    release, entered = threading.Event(), threading.Event()
    now = [0.]

    def blocked(rows, total, progress):
        entered.set()
        release.wait(5)
        return case(rows, total, progress)

    limits = replace(Limits(), pending_jobs=1, session_seconds=10, result_seconds=5)
    try:
        with running(blocked, limits=limits, clock=lambda: now[0]) as server:
            one, two, three = [Visitor(server) for _ in range(3)]
            for visitor in (one, two, three):
                visitor.status()
            first = one.start()[1]["job_id"]
            assert entered.wait(2)
            assert one.start()[0] == 409
            second = two.start()[1]["job_id"]
            assert two.request("GET", "/api/jobs/" + second)[1]["status"] == "queued"
            assert three.start()[0] == 503
            now[0] = 20_000  # TTL is never a timeout for running or queued work.
            assert one.request("GET", "/api/jobs/" + first)[1]["status"] == "running"
            assert two.request("GET", "/api/jobs/" + second)[1]["status"] == "queued"
            release.set()
            assert one.finish(first)[1]["status"] == "complete"
            assert two.finish(second)[1]["status"] == "complete"
    finally:
        release.set()


def test_result_expiry_clears_job_latest_and_export():
    now = [0.]
    with running(clock=lambda: now[0], limits=replace(Limits(), result_seconds=5)) as server:
        visitor = Visitor(server)
        visitor.status()
        job = visitor.start()[1]["job_id"]
        visitor.finish(job)
        now[0] = 6
        for path in ("/api/jobs/" + job, "/api/latest", "/api/export/latest"):
            assert visitor.request("GET", path)[0] == 404
        assert server.jobs == {}


def test_failures_and_progress_cannot_leak_provider_credentials(caplog):
    def broken(rows, total, progress):
        progress("credential-that-must-not-be-shared")
        raise RuntimeError("credential-that-must-not-be-shared")

    with running(broken) as server:
        visitor = Visitor(server)
        visitor.status()
        job = visitor.start()[1]["job_id"]
        code, result = visitor.finish(job)
        assert code == 200 and result["status"] == "failed"
        assert "credential-that" not in json.dumps(result)
        assert "credential-that" not in caplog.text
        assert result["error_id"] in caplog.text and "RuntimeError" in caplog.text


def test_environment_keys_and_credential_fields_are_removed_from_results(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "server-only-secret-123")

    def tainted(rows, total, progress):
        result = case(rows, total, progress)
        result.update(api_key="server-only-secret-123", note="value=server-only-secret-123",
                      nested={"Authorization": "Bearer fake-secret-123456"})
        return result

    with running(tainted) as server:
        visitor = Visitor(server)
        visitor.status()
        result = visitor.finish(visitor.start()[1]["job_id"])[1]
        assert result["status"] == "complete"
        assert "server-only-secret" not in json.dumps(result)
        assert "api_key" not in result["case"]
        assert result["case"]["nested"] == {}


def test_tuple_nested_credentials_and_dictionary_keys_are_redacted_from_result_and_export(monkeypatch):
    secret = "abc123DEF456ghi789JKL012"
    monkeypatch.setenv("MASSIVE_API_KEY", secret)

    def tainted(rows, total, progress):
        result = case(rows, total, progress)
        result["nested"] = (secret, {"note": "value=" + secret, "Authorization": secret})
        result["lookup"] = {"provider-" + secret: "public value", "source": "provider metadata"}
        return result

    with running(tainted) as server:
        visitor = Visitor(server)
        visitor.status()
        result = visitor.finish(visitor.start()[1]["job_id"])[1]
        assert result["status"] == "complete"
        assert secret not in json.dumps(result)
        assert "Authorization" not in result["case"]["nested"][1]
        assert result["case"]["lookup"]["source"] == "provider metadata"
        status, data, _ = visitor.request("GET", "/api/export/latest")
        assert status == 200
        with ZipFile(io.BytesIO(data)) as archive:
            assert all(secret.encode() not in archive.read(name) for name in archive.namelist())


def test_missing_server_secrets_return_service_error_and_do_not_create_job(monkeypatch):
    for key in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with running(runner=None) as server:
        visitor = Visitor(server)
        visitor.status()
        assert visitor.start()[0] == 503
        assert server.jobs == {}


def test_static_assets_and_security_headers_preserve_approved_ui():
    with running() as server:
        visitor = Visitor(server)
        for path in ("/", "/studio.js", "/studio.css", "/identity.css", "/r99.css", "/data.json"):
            code, _, headers = visitor.request("GET", path)
            assert code == 200
            assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
            assert headers["Strict-Transport-Security"].startswith("max-age=")
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert headers["Referrer-Policy"] == "no-referrer"
        for path in ("/../README.md", "/runtime/option1_live/report.json", "/.env", "/Dockerfile"):
            assert visitor.request("GET", path)[0] == 404


def test_idle_sessions_are_reclaimed_before_they_can_block_new_visitors():
    with running(limits=replace(Limits(), max_sessions=1)) as server:
        first, second = Visitor(server), Visitor(server)
        assert first.status()[0] == 200
        old_cookie = first.cookie
        assert second.status()[0] == 200
        assert second.cookie != old_cookie
        assert len(server.sessions) == 1
        assert first.start()[0] == 403


def test_session_reclamation_never_evicts_a_retained_result_owner():
    with running(limits=replace(Limits(), max_sessions=1)) as server:
        first, second = Visitor(server), Visitor(server)
        first.status()
        job = first.start()[1]["job_id"]
        assert first.finish(job)[1]["status"] == "complete"
        assert second.status()[0] == 503
        assert len(server.sessions) == 1
        assert first.request("GET", "/api/latest")[0] == 200


def test_new_session_creation_is_throttled_and_recovers_after_window():
    now = [0.]
    limits = replace(Limits(), max_sessions=2, session_creations_per_ip=2)
    with running(limits=limits, clock=lambda: now[0]) as server:
        first, second, third = [Visitor(server) for _ in range(3)]
        first.status()
        second.status()
        code, _, headers = third.request("GET", "/api/status", headers={"X-Forwarded-For": "1.2.3.4"})
        assert code == 429 and int(headers["Retry-After"]) > 0
        assert first.status()[0] == 200  # Existing session reads do not consume creation budget.
        now[0] = 61
        assert third.status()[0] == 200
        assert len(server.sessions) == 2


def test_evicted_session_cannot_start_paid_work_or_read_status():
    with running(limits=replace(Limits(), max_sessions=1)) as server:
        first = server.session(None, create=True)[0]
        server.session(None, create=True)
        for operation in (lambda: server.start_job(first, "127.0.0.1", POSITIONS, 1000),
                          lambda: server.status(first)):
            with pytest.raises(PublicRequestError) as rejected:
                operation()
            assert rejected.value.status == 403
        assert server.jobs == {} and server.rates == {}


@pytest.mark.parametrize("path", ["/api/status", "/api/analyze"])
def test_http_session_reclamation_race_returns_reload_without_orphan_work(monkeypatch, path):
    with running(limits=replace(Limits(), max_sessions=1)) as server:
        visitor = Visitor(server)
        visitor.status()
        original = server.session

        def reclaim_after_lookup(*args, **kwargs):
            selected = original(*args, **kwargs)
            original(None, create=True)
            return selected

        monkeypatch.setattr(server, "session", reclaim_after_lookup)
        response = visitor.status() if path == "/api/status" else visitor.start()
        assert response[0] == 403
        assert "Reload" in response[1]["error"]
        assert server.jobs == {} and server.rates == {}


def test_unverified_holdings_receive_actionable_input_error_without_raw_message():
    def invalid_identity(rows, total, progress):
        raise InstrumentIdentityError("private source detail")

    with running(invalid_identity) as server:
        visitor = Visitor(server)
        visitor.status()
        result = visitor.finish(visitor.start()[1]["job_id"])[1]
        assert result["status"] == "failed"
        assert "ticker symbols" in result["error"]
        assert "private source detail" not in json.dumps(result)


def test_completed_results_respect_total_byte_capacity():
    def larger(rows, total, progress):
        result = case(rows, total, progress)
        result["note"] = "x" * 650
        return result

    limits = replace(Limits(), max_result_bytes=1024, max_total_result_bytes=1024)
    with running(larger, limits=limits) as server:
        one, two = Visitor(server), Visitor(server)
        one.status()
        two.status()
        first = one.start()[1]["job_id"]
        assert one.finish(first)[1]["status"] == "complete"
        second = two.start()[1]["job_id"]
        assert two.finish(second)[1]["status"] == "complete"
        assert one.request("GET", "/api/latest")[0] == 404
        assert sum(job.get("bytes", 0) for job in server.jobs.values()) <= 1024


def test_oversized_result_becomes_failure_without_retaining_payload():
    def too_large(rows, total, progress):
        result = case(rows, total, progress)
        result["note"] = "x" * 2000
        return result

    with running(too_large, limits=replace(Limits(), max_result_bytes=512)) as server:
        visitor = Visitor(server)
        visitor.status()
        job = visitor.start()[1]["job_id"]
        response = visitor.finish(job)[1]
        assert response["status"] == "failed" and "case" not in response
        assert visitor.request("GET", "/api/latest")[0] == 404


def test_session_expiry_removes_results_and_issues_a_new_random_cookie():
    now = [0.]
    with running(clock=lambda: now[0], limits=replace(Limits(), session_seconds=5)) as server:
        visitor = Visitor(server)
        visitor.status()
        original_cookie = visitor.cookie
        first = visitor.start()[1]["job_id"]
        visitor.finish(first)
        now[0] = 6
        assert visitor.status()[0] == 200
        assert visitor.cookie != original_cookie
        assert visitor.request("GET", "/api/jobs/" + first)[0] == 404
        assert server.jobs == {}


def test_completed_job_count_is_bounded():
    limits = replace(Limits(), max_jobs=2, pending_jobs=1, starts_per_session=4)
    with running(limits=limits) as server:
        visitor = Visitor(server)
        visitor.status()
        jobs = []
        for _ in range(3):
            identifier = visitor.start()[1]["job_id"]
            assert visitor.finish(identifier)[1]["status"] == "complete"
            jobs.append(identifier)
        assert len(server.jobs) == 2
        assert visitor.request("GET", "/api/jobs/" + jobs[0])[0] == 404


def test_export_concurrency_is_bounded_and_releases_slot_after_completion(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = PublicPortfolioHandler._send_export

    def blocked(handler, portfolio):
        entered.set()
        release.wait(5)
        return original(handler, portfolio)

    monkeypatch.setattr(PublicPortfolioHandler, "_send_export", blocked)
    try:
        with running(limits=replace(Limits(), max_exports=1)) as server:
            visitor = Visitor(server)
            visitor.status()
            visitor.finish(visitor.start()[1]["job_id"])
            results = []
            request = threading.Thread(target=lambda: results.append(visitor.request("GET", "/api/export/latest")))
            request.start()
            assert entered.wait(2)
            code, _, headers = visitor.request("GET", "/api/export/latest")
            assert code == 503 and headers["Retry-After"] == "5"
            release.set()
            request.join(3)
            assert results[0][0] == 200
            assert visitor.request("GET", "/api/export/latest")[0] == 200
    finally:
        release.set()


@pytest.mark.parametrize("origin", ["http://castle.example", "https://castle.example/path",
                                   "https://user:password@castle.example", "https://castle.example?x=y"])
def test_public_origin_rejects_unsafe_configuration(origin):
    with pytest.raises(ValueError):
        PublicPortfolioServer(("127.0.0.1", 0), origin=origin, runner=case)
