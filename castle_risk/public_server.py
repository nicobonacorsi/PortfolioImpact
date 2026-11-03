"""Anonymous HTTPS adapter for the existing Portfolio Impact analysis engine.

Run behind a TLS reverse proxy with CASTLE_PUBLIC_ORIGIN set to the exact
HTTPS origin. One process owns bounded, expiring sessions and analysis jobs.
Provider credentials are read from the server environment only.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookies import SimpleCookie, CookieError
from http.server import ThreadingHTTPServer
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import re
import secrets
import socket
import tempfile
import threading
import time
from urllib.parse import urlsplit

from castle_risk.instrument_identity import InstrumentIdentityError

from scripts.serve_portfolio_impact import (
    ASSETS, MAX_INPUT_BYTES, PortfolioImpactHandler, PortfolioInputError,
    _safe_browser_case, parse_positions,
)

COOKIE = "castle_session"
ACTIVE = {"queued", "running"}
SECRET_FIELD = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)", re.I)
SECRET_VALUE = re.compile(r"(?:sk-ant-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{12,})", re.I)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    workers: int = 1
    pending_jobs: int = 3
    max_sessions: int = 512
    max_jobs: int = 24
    session_seconds: int = 86_400
    result_seconds: int = 3600
    starts_per_session: int = 2
    starts_per_ip: int = 6
    starts_global: int = 12
    rate_seconds: int = 3600
    max_result_bytes: int = 8 * 1024 * 1024
    max_total_result_bytes: int = 64 * 1024 * 1024
    max_http_connections: int = 32
    session_creations_per_ip: int = 30
    session_creations_global: int = 120
    session_creation_seconds: int = 60
    max_exports: int = 2

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in vars(self).values()):
            raise ValueError("Public server limits must be positive integers")
        if self.max_jobs < self.workers + self.pending_jobs:
            raise ValueError("Job capacity must include running and queued jobs")


class PublicRequestError(Exception):
    def __init__(self, status, message, retry_after=None):
        self.status, self.message, self.retry_after = status, message, retry_after
        super().__init__(message)


def _public_case(rows, total, progress):
    from castle_risk.live_assessment import run_live_assessment
    # Each paid run owns its receipts. They never enter static serving or a
    # different visitor's replay path, and disappear after browser conversion.
    with tempfile.TemporaryDirectory(prefix="castle-public-") as work:
        report = run_live_assessment(
            portfolio_rows=rows, notional=total, root=Path(work),
            massive_key=os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY"),
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY"),
            include_geopolitical_search=True, include_payrolls=True, progress=progress,
        )
        return _safe_browser_case(report)


def _safe_result(case, maximum):
    """Reject malformed results and strip credential-shaped fields recursively."""
    credentials = tuple(v for k in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "ANTHROPIC_API_KEY")
                        if (v := os.environ.get(k)))

    def clean(value):
        if isinstance(value, dict):
            return {clean(str(k)): clean(v) for k, v in value.items()
                    if not SECRET_FIELD.search(str(k))}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for credential in credentials:
                value = value.replace(credential, "[redacted]")
            return SECRET_VALUE.sub("[redacted]", value)
        return value

    if not isinstance(case, dict) or case.get("user_supplied") is not True:
        raise ValueError("Expected a user portfolio result")
    result = clean(case)
    encoded = json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError("Analysis result exceeds retention capacity")
    return result, len(encoded)


class PublicPortfolioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, address, *, origin, runner=None, limits=None, clock=time.monotonic,
                 asset_root=ASSETS, allow_local_http=False, bind_and_activate=True):
        parsed = urlsplit(origin)
        local_http = (allow_local_http and parsed.scheme == "http" and
                      parsed.hostname in {"127.0.0.1", "localhost", "::1"})
        if (not parsed.netloc or parsed.username or parsed.password or parsed.query or
                parsed.fragment or parsed.path not in {"", "/"} or
                (parsed.scheme != "https" and not local_http)):
            raise ValueError("Use one exact HTTPS public origin")
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.host = parsed.netloc.lower()
        self.secure = parsed.scheme == "https"
        self.asset_root = Path(asset_root).resolve()
        self.limits = limits or Limits()
        self.clock = clock
        self.runner = runner or _public_case
        self.require_sources = runner is None
        self.lock = threading.RLock()
        self.sessions, self.jobs, self.rates, self.creation_rates = {}, {}, {}, {}
        self.pending = queue.Queue(maxsize=self.limits.workers + self.limits.pending_jobs)
        self.stop_workers = threading.Event()
        self.http_slots = threading.BoundedSemaphore(self.limits.max_http_connections)
        self.export_slots = threading.BoundedSemaphore(self.limits.max_exports)
        super().__init__(address, PublicPortfolioHandler, bind_and_activate=bind_and_activate)
        self.workers = [threading.Thread(target=self._work, daemon=True,
                                        name=f"castle-analysis-{i}")
                        for i in range(self.limits.workers)]
        for worker in self.workers:
            worker.start()

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(15)
        return connection, address

    def process_request(self, request, client_address):
        if not self.http_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.http_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.http_slots.release()

    def handle_error(self, request, client_address):
        # Never print HTTP contents, portfolio values, or provider exceptions.
        pass

    def server_close(self):
        self.stop_workers.set()
        super().server_close()

    def source_status(self):
        return {
            "data_provider_configured": bool(os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")),
            "research_model_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        }

    def _prune(self):
        now = self.clock()
        for key, job in list(self.jobs.items()):
            if job["status"] not in ACTIVE and now - job["finished"] >= self.limits.result_seconds:
                del self.jobs[key]
        active_owners = {job["owner"] for job in self.jobs.values() if job["status"] in ACTIVE}
        for key, session in list(self.sessions.items()):
            if key not in active_owners and now - session["seen"] >= self.limits.session_seconds:
                del self.sessions[key]
                for job_id, job in list(self.jobs.items()):
                    if job["owner"] == key:
                        del self.jobs[job_id]
        for key, starts in list(self.rates.items()):
            while starts and now - starts[0] >= self.limits.rate_seconds:
                starts.popleft()
            if not starts:
                del self.rates[key]
        for key, starts in list(self.creation_rates.items()):
            while starts and now - starts[0] >= self.limits.session_creation_seconds:
                starts.popleft()
            if not starts:
                del self.creation_rates[key]

    def session(self, identifier, *, create=False, peer_ip="unknown"):
        with self.lock:
            self._prune()
            if identifier in self.sessions:
                self.sessions[identifier]["seen"] = self.clock()
                return identifier, False
            if not create:
                return None, False
            creation_keys = [("ip:" + peer_ip, self.limits.session_creations_per_ip),
                             ("global", self.limits.session_creations_global)]
            for key, limit in creation_keys:
                previous = self.creation_rates.get(key, ())
                if len(previous) >= limit:
                    retry = max(1, int(self.limits.session_creation_seconds -
                                       (self.clock() - previous[0])) + 1)
                    raise PublicRequestError(429, "Too many new sessions. Try again shortly.", retry)
            if len(self.sessions) >= self.limits.max_sessions:
                # Empty sessions carry no paid work or result. Reclaim them so
                # repeated anonymous status reads cannot fill a 24-hour store.
                protected = {job["owner"] for job in self.jobs.values()}
                idle = min((key for key in self.sessions if key not in protected),
                           key=lambda key: self.sessions[key]["seen"], default=None)
                if idle is None:
                    raise PublicRequestError(503, "The service is busy. Try again later.", 60)
                del self.sessions[idle]
            identifier = secrets.token_urlsafe(32)
            self.sessions[identifier] = {"seen": self.clock(), "csrf": secrets.token_urlsafe(32)}
            for key, _ in creation_keys:
                self.creation_rates.setdefault(key, deque()).append(self.clock())
            return identifier, True

    def _owned(self, session, identifier):
        job = self.jobs.get(identifier)
        return job if job and job["owner"] == session else None

    def status(self, session):
        with self.lock:
            self._prune()
            if session not in self.sessions:
                raise PublicRequestError(403, "Reload the page before starting an analysis.")
            active = next(((key, job) for key, job in self.jobs.items()
                           if job["owner"] == session and job["status"] in ACTIVE), None)
            return {"schema": "castle.portfolio_impact.api.v1", "presentation_build": "R103",
                    "available": True, "hosting_mode": "public", "hosted": True,
                    "provider_setup_allowed": False, "csrf_token": self.sessions[session]["csrf"],
                    "busy": active is not None, "active_job_id": active[0] if active else None,
                    "active_job_progress": active[1]["progress"] if active else None,
                    "result_retention_seconds": self.limits.result_seconds,
                    "limits": {"max_positions": 24, "max_notional_usd": 1_000_000_000},
                    **self.source_status()}

    def job(self, session, identifier):
        with self.lock:
            self._prune()
            job = self._owned(session, identifier)
            return ({k: v for k, v in job.items() if k in
                     {"status", "progress", "created_at", "case", "error", "error_id"}} if job else None)

    def latest(self, session):
        with self.lock:
            self._prune()
            completed = [job for job in self.jobs.values()
                         if job["owner"] == session and job["status"] == "complete"]
            return max(completed, key=lambda j: j["finished"])["case"] if completed else None

    def start_job(self, session, ip, rows, total):
        with self.lock:
            self._prune()
            # Empty-session reclamation can occur between HTTP CSRF validation
            # and paid admission. Do not create work without a retained owner.
            if session not in self.sessions:
                raise PublicRequestError(403, "Reload the page before starting an analysis.")
            if any(j["owner"] == session and j["status"] in ACTIVE for j in self.jobs.values()):
                raise PublicRequestError(409, "Your analysis is already running. Wait for its results.")
            if self.require_sources and not all(self.source_status().values()):
                raise PublicRequestError(503, "Live analysis is temporarily unavailable. Try again later.", 60)
            active_count = sum(j["status"] in ACTIVE for j in self.jobs.values())
            if active_count >= self.limits.workers + self.limits.pending_jobs:
                raise PublicRequestError(503, "The analysis queue is full. Try again later.", 60)
            rate_keys = [("session:" + session, self.limits.starts_per_session),
                         ("ip:" + ip, self.limits.starts_per_ip), ("global", self.limits.starts_global)]
            for key, limit in rate_keys:
                previous = self.rates.get(key, ())
                if len(previous) >= limit:
                    retry = max(1, int(self.limits.rate_seconds - (self.clock() - previous[0])) + 1)
                    raise PublicRequestError(429, "The analysis limit has been reached. Try again later.", retry)
            # Evict oldest completed jobs before admitting another paid run.
            while len(self.jobs) >= self.limits.max_jobs:
                oldest = min((k for k, j in self.jobs.items() if j["status"] not in ACTIVE),
                             key=lambda k: self.jobs[k]["finished"], default=None)
                if oldest is None:
                    raise PublicRequestError(503, "The service is busy. Try again later.", 60)
                del self.jobs[oldest]
            identifier = secrets.token_urlsafe(32)
            self.jobs[identifier] = {"owner": session, "status": "queued", "progress": "Waiting to start",
                                     "created_at": datetime.now(timezone.utc).isoformat()}
            self.pending.put_nowait((identifier, rows, total))
            for key, _ in rate_keys:
                self.rates.setdefault(key, deque()).append(self.clock())
            return identifier

    def _work(self):
        while not self.stop_workers.is_set():
            try:
                identifier, rows, total = self.pending.get(timeout=.25)
            except queue.Empty:
                continue
            with self.lock:
                self.jobs[identifier].update(status="running", progress="Analyzing your portfolio")

            def progress(_message):
                # Provider diagnostics can contain URLs or credentials. Keep the
                # public progress vocabulary independent of provider responses.
                with self.lock:
                    self.jobs[identifier]["progress"] = "Checking sources and calculating scenarios"

            try:
                case, size = _safe_result(self.runner(rows, total, progress), self.limits.max_result_bytes)
                with self.lock:
                    while sum(j.get("bytes", 0) for j in self.jobs.values()) + size > self.limits.max_total_result_bytes:
                        oldest = min((k for k, j in self.jobs.items() if j["status"] not in ACTIVE),
                                     key=lambda k: self.jobs[k]["finished"], default=None)
                        if oldest is None:
                            raise ValueError("Result retention is full")
                        del self.jobs[oldest]
                    self.jobs[identifier].update(status="complete", progress="Portfolio analysis ready",
                                                 case=case, bytes=size, finished=self.clock())
            except Exception as exc:
                error_id = secrets.token_hex(6)
                LOGGER.error("analysis_failed error_id=%s error_type=%s", error_id, type(exc).__name__)
                message = ("Some holdings could not be verified. Check their ticker symbols and use supported USD stocks or funds."
                           if isinstance(exc, InstrumentIdentityError) else
                           "The analysis could not finish. Try again later.")
                with self.lock:
                    self.jobs[identifier].update(status="failed", progress="Analysis stopped", finished=self.clock(),
                        error=message, error_id=error_id)
            finally:
                self.pending.task_done()


class PublicPortfolioHandler(PortfolioImpactHandler):
    server: PublicPortfolioServer
    server_version = "PortfolioImpact"
    sys_version = ""

    def _preview_authorized(self):
        """Require a preview password when the deployment is not yet released."""
        password = os.environ.get("CASTLE_PREVIEW_PASSWORD")
        if not password:
            return True
        header = self.headers.get("Authorization", "")
        authorized = False
        if header.startswith("Basic ") and len(header) <= 1024:
            try:
                credentials = base64.b64decode(header[6:], validate=True).decode("utf-8")
                username, delimiter, candidate = credentials.partition(":")
                authorized = bool(delimiter) and hmac.compare_digest(username, "castle") and hmac.compare_digest(candidate, password)
            except (ValueError, UnicodeError):
                pass
        if not authorized:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Portfolio Impact Private Preview"')
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        return authorized

    def end_headers(self):
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if self.server.secure:
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        if getattr(self, "new_session", None):
            secure = "; Secure" if self.server.secure else ""
            self.send_header("Set-Cookie", f"{COOKIE}={self.new_session}; Path=/; HttpOnly; SameSite=Strict; Max-Age={self.server.limits.session_seconds}{secure}")
        super().end_headers()

    def _error(self, exc):
        if exc.retry_after is not None:
            self.retry_after = str(exc.retry_after)
        return self._send(exc.status, {"error": exc.message})

    def send_response(self, code, message=None):
        super().send_response(code, message)
        if getattr(self, "retry_after", None):
            self.send_header("Retry-After", self.retry_after)

    def _request_session(self, *, create=False, write=False):
        if (len(self.path) > 2048 or len(self.headers.get_all("Host", [])) != 1 or
                self.headers.get("Host", "").lower() != self.server.host):
            raise PublicRequestError(403, "This request was not accepted.")
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != self.server.origin):
            raise PublicRequestError(403, "Open this page directly and try again.")
        if self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
            raise PublicRequestError(403, "Open this page directly and try again.")
        if write and origins != [self.server.origin]:
            raise PublicRequestError(403, "Open this page directly and try again.")
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            pass
        value = cookie[COOKIE].value if COOKIE in cookie else None
        session, _ = self.server.session(value, create=create, peer_ip=self.client_address[0])
        # Refresh the browser lifetime together with the server's sliding TTL.
        # Polling a long job therefore keeps its anonymous session usable.
        if session:
            self.new_session = session
        if write:
            if not session:
                raise PublicRequestError(403, "Reload the page before starting an analysis.")
            with self.server.lock:
                retained = self.server.sessions.get(session)
                if retained is None:
                    raise PublicRequestError(403, "Reload the page before starting an analysis.")
                expected = retained["csrf"]
            supplied = self.headers.get("X-CSRF-Token", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", supplied) or not hmac.compare_digest(supplied, expected):
                raise PublicRequestError(403, "Reload the page before starting an analysis.")
        return session

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            if path == "/healthz":
                return self._send(200, {"status": "ok"})
            if not self._preview_authorized():
                return
            session = self._request_session(create=path == "/api/status")
            if path == "/api/status":
                return self._send(200, self.server.status(session))
            if path in {"/api/latest", "/api/export/latest"}:
                case = self.server.latest(session) if session else None
                if case is None:
                    return self._send(404, {"error": "No completed analysis in this session"})
                if path == "/api/export/latest":
                    if not self.server.export_slots.acquire(blocking=False):
                        raise PublicRequestError(503, "Report downloads are busy. Try again shortly.", 5)
                    try:
                        return self._send_export(case)
                    finally:
                        self.server.export_slots.release()
                return self._send(200, {"case": case})
            if path.startswith("/api/jobs/"):
                task = self.server.job(session, path[len("/api/jobs/"):]) if session else None
                return self._send(200, task) if task else self._send(404, {"error": "Unknown analysis"})
            return self._static(path)
        except PublicRequestError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception:
            self._send(500, {"error": "This request could not finish. Try again."})

    def do_POST(self):
        try:
            if not self._preview_authorized():
                return
            session = self._request_session(write=True)
            path = urlsplit(self.path).path
            if path not in {"/api/analyze", "/api/what-if/treasury", "/api/scenarios/evaluate"}:
                return self._send(404, {"error": "Not found"})
            if (self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json" or
                    self.headers.get("X-Portfolio-Impact") != "1"):
                return self._send(415, {"error": "Expected the app's JSON request"})
            lengths = self.headers.get_all("Content-Length", [])
            if (self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit() or
                    not 0 < int(lengths[0]) <= MAX_INPUT_BYTES):
                return self._send(413, {"error": "Portfolio input is too large or empty"})
            try:
                payload = json.loads(self.rfile.read(int(lengths[0])))
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "Invalid portfolio JSON"})
            if path == "/api/what-if/treasury":
                return self._send(422, {"error": "This Treasury research case is unavailable for this portfolio."})
            if path == "/api/scenarios/evaluate":
                case = self.server.latest(session)
                if case is None:
                    return self._send(409, {"error": "Run a portfolio analysis before building a personal scenario."})
                from castle_risk.scenario_lab import ScenarioInputError, evaluate_scenario
                try:
                    result = evaluate_scenario(case, payload)
                except ScenarioInputError as exc:
                    return self._send(422, {"error": str(exc)})
                return self._send(200, result)
            if not isinstance(payload, dict) or set(payload) != {"positions"}:
                return self._send(422, {"error": "Send holdings and dollar amounts only."})
            try:
                rows, total = parse_positions(payload)
            except PortfolioInputError as exc:
                return self._send(422, {"error": str(exc)})
            # Forwarded client-IP headers are untrusted. The peer IP is an
            # additional limiter; per-session and global budgets always apply.
            ip = str(ipaddress.ip_address(self.client_address[0]))
            identifier = self.server.start_job(session, ip, rows, total)
            return self._send(202, {"job_id": identifier, "status": "queued"})
        except PublicRequestError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception:
            self._send(500, {"error": "This request could not finish. Try again."})


def main():
    origin = os.environ.get("CASTLE_PUBLIC_ORIGIN") or os.environ.get("RENDER_EXTERNAL_URL")
    if not origin:
        raise SystemExit("Set CASTLE_PUBLIC_ORIGIN to the public HTTPS origin.")
    changes = {key: int(os.environ["CASTLE_" + key.upper()]) for key in Limits.__dataclass_fields__
               if "CASTLE_" + key.upper() in os.environ}
    variant = os.environ.get("CASTLE_UI_VARIANT", "r127").lower()
    if variant == "v4":
        asset_root = Path(__file__).resolve().parents[1] / "web" / "portfolio_impact_v4"
    elif variant == "r127":
        asset_root = ASSETS
    else:
        raise SystemExit("Unknown CASTLE_UI_VARIANT")
    if not asset_root.is_dir():
        raise SystemExit("Selected web assets are missing")
    server = PublicPortfolioServer(("0.0.0.0", int(os.environ.get("PORT", "10000"))),
                                  origin=origin, limits=Limits(**changes), asset_root=asset_root)
    print("Portfolio Impact public server started", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
