"""Private, owner-operated Portfolio Impact server for Codespaces and local use.

The approved v4 assets remain unchanged. Provider keys are kept in process
memory, and GitHub Codespaces must keep the forwarded port PRIVATE.
"""
from __future__ import annotations

import argparse
from http.server import HTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit
import webbrowser

from scripts.serve_portfolio_impact import (
    PortfolioImpactHandler,
    PortfolioImpactServer,
    ROOT,
)

V4_ASSETS = ROOT / "web" / "portfolio_impact_v4"
SETUP_ASSETS = {
    "/connections": ("connections.html", "text/html; charset=utf-8"),
    "/connections.html": ("connections.html", "text/html; charset=utf-8"),
    "/connections.js": ("connections.js", "application/javascript; charset=utf-8"),
}


class CodespacesPortfolioServer(PortfolioImpactServer):
    """Single-user engine; Codespaces' PRIVATE port is the access boundary."""

    def __init__(self, address, *, origin, **kwargs):
        self.origin = origin.rstrip("/")
        parsed = urlsplit(self.origin)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("Provide an exact origin without a path")
        self.allowed_host = parsed.netloc.lower()
        self.csrf_token = secrets.token_urlsafe(32)
        super().__init__(address, asset_root=V4_ASSETS, **kwargs)
        self.RequestHandlerClass = CodespacesPortfolioHandler
        # Secrets configured in Codespaces settings are supported as an alternative
        # to the in-app setup form. Neither source is written to a project file.
        if key := (os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")):
            self.provider_keys["massive"] = key
        if key := os.environ.get("ANTHROPIC_API_KEY"):
            self.provider_keys["anthropic"] = key


class CodespacesPortfolioHandler(PortfolioImpactHandler):
    server: CodespacesPortfolioServer

    def _local_request(self):
        hosts = self.headers.get_all("Host", [])
        forwarded_hosts = self.headers.get_all("X-Forwarded-Host", [])
        if len(hosts) != 1 or not hosts[0].strip() or len(forwarded_hosts) > 1:
            return False

        # Codespaces authenticates the public HTTPS port, but its forwarding
        # proxy may replace Host with localhost:PORT before reaching Python.
        backend_host = hosts[0].strip().lower()
        forwarded_host = forwarded_hosts[0].strip().lower() if forwarded_hosts else None
        expected_host = self.server.allowed_host
        internal_hosts = {
            "localhost:" + str(self.server.server_port),
            "127.0.0.1:" + str(self.server.server_port),
            "[::1]:" + str(self.server.server_port),
        }
        if forwarded_host is not None and forwarded_host != expected_host:
            return False
        if backend_host != expected_host and backend_host not in internal_hosts and forwarded_host != expected_host:
            return False

        # A private Codespaces port authenticates visitors, but its reverse
        # proxy can rewrite Origin / Fetch Metadata independently of the
        # browser-visible URL. Never use those headers as a write gate here.
        # All POSTs instead require our unpredictable, process-local CSRF token
        # and an application-specific JSON header; CORS is not enabled.
        return True

    def end_headers(self):
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
        )
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _static(self, path):
        if path not in SETUP_ASSETS:
            return super()._static(path)
        filename, content_type = SETUP_ASSETS[path]
        content = (V4_ASSETS / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self._local_request():
            return self._send(403, {"error": "This page is available only on its own origin."})
        path = urlsplit(self.path).path
        if path == "/healthz":
            return self._send(200, {"status": "ok"})
        if path == "/" and not all(self.server.source_status().values()):
            self.send_response(302)
            self.send_header("Location", "/connections")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/api/status":
            with self.server.job_lock:
                job_id = self.server.active_job
                active = self.server.jobs.get(job_id, {}) if job_id else {}
            return self._send(200, {
                "schema": "castle.portfolio_impact.api.v1",
                "presentation_build": "R129-Codespaces",
                "available": True,
                "hosting_mode": "private_codespaces",
                "hosted": True,
                "provider_setup_allowed": True,
                "csrf_token": self.server.csrf_token,
                "busy": bool(job_id),
                "active_job_id": job_id,
                "active_job_progress": active.get("progress"),
                **self.server.source_status(),
            })
        return super().do_GET()

    def do_POST(self):
        if not self._local_request():
            return self._send(403, {"error": "This request must come from this application."})
        supplied = self.headers.get("X-CSRF-Token", "")
        if not (len(supplied) == len(self.server.csrf_token) and
                hmac.compare_digest(supplied, self.server.csrf_token)):
            return self._send(403, {"error": "Reload the page and try again."})
        return super().do_POST()


def codespaces_origin(port):
    """Codespaces' private port proxy uses this HTTPS browser origin."""
    name = os.environ.get("CODESPACE_NAME", "")
    if not name:
        return None
    if not all(char.isascii() and (char.isalnum() or char == "-") for char in name):
        raise SystemExit("Invalid CODESPACE_NAME")
    domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
    if not domain or not all(char.isascii() and (char.isalnum() or char in ".-") for char in domain):
        raise SystemExit("Invalid Codespaces forwarding domain")
    return "https://" + name + "-" + str(port) + "." + domain


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portfolio Impact private v4")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535")
    forwarded_origin = codespaces_origin(args.port)
    origin = forwarded_origin or "http://127.0.0.1:" + str(args.port)
    address = "0.0.0.0" if forwarded_origin else "127.0.0.1"
    server = CodespacesPortfolioServer((address, args.port), origin=origin)
    print("Portfolio Impact ready: " + origin, flush=True)
    print("Provider setup: " + origin + "/connections", flush=True)
    if forwarded_origin:
        # A localhost URL in terminal output tells Codespaces to forward this port.
        print("Port forwarding: http://localhost:" + str(args.port), flush=True)
    print("Keep the Codespaces forwarded port PRIVATE.", flush=True)
    if args.open and not forwarded_origin:
        webbrowser.open(origin)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
