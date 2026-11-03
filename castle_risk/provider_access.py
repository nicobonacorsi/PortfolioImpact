"""Failure classes and an egress circuit breaker for the original read-only clients.

``adapters.py`` is byte-protected (Build 01.1 engine hashes), so its clients are
extended here instead of edited. ``AuditedClientMixin`` wraps the client's
transport to observe what actually happened on the wire, then labels every
``DataAccessError`` the original ``_get`` raises with a fixed ``failure_class``:

  EGRESS_PROXY_POLICY_DENIED_PROVIDER_NOT_REACHED   the proxy refused the tunnel
  NOT_ATTEMPTED_AFTER_EGRESS_POLICY_DENIAL          not sent: the provider is unreachable
  PROVIDER_HTTP_AUTH_OR_ENTITLEMENT / _RATE_LIMIT / _NOT_FOUND / _SERVER_ERROR / _CLIENT_ERROR
  NETWORK_TIMEOUT / DNS_RESOLUTION_FAILED / NETWORK_FAILURE
  LOCAL_REQUEST_BUDGET_EXHAUSTED / SCHEMA_REJECTED / RESPONSE_SIZE_CAP / UNTRUSTED_HOST_REFUSED_LOCALLY

Labels never contain server text. Request retries, budgets, caching, redaction
and raw response bytes are the original client's, unchanged. Newly created
receipts record the underlying transport, not the identity of its audit wrapper;
historical receipts are never relabeled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit

from .adapters import DataAccessError, KalshiClient, MassiveClient, _transport, sanitize_url, utc_now

EGRESS_DENIED = "EGRESS_PROXY_POLICY_DENIED_PROVIDER_NOT_REACHED"
NOT_SENT = "NOT_ATTEMPTED_AFTER_EGRESS_POLICY_DENIAL"


def classify_network_failure(exc: BaseException) -> str:
    """Fixed label for a transport exception; reads only the local reason text."""
    reason = str(getattr(exc, "reason", exc))
    if "Tunnel connection failed" in reason and ("403" in reason or "407" in reason):
        return EGRESS_DENIED
    if isinstance(exc, TimeoutError) or "timed out" in reason.lower():
        return "NETWORK_TIMEOUT"
    if "Name or service not known" in reason or "getaddrinfo" in reason:
        return "DNS_RESOLUTION_FAILED"
    return "NETWORK_FAILURE"


def classify_http_status(code: int) -> str:
    if code in (401, 403):
        return "PROVIDER_HTTP_AUTH_OR_ENTITLEMENT"
    if code == 429:
        return "PROVIDER_HTTP_RATE_LIMIT"
    if code == 404:
        return "PROVIDER_HTTP_NOT_FOUND"
    if code >= 500:
        return "PROVIDER_HTTP_SERVER_ERROR"
    return "PROVIDER_HTTP_CLIENT_ERROR"


def _local_label(message: str) -> str | None:
    if message.startswith("Refused an untrusted data host"):
        return "UNTRUSTED_HOST_REFUSED_LOCALLY"
    if "budget exhausted" in message:
        return "LOCAL_REQUEST_BUDGET_EXHAUSTED"
    if "16 MB" in message:
        return "RESPONSE_SIZE_CAP"
    if ("not a JSON object" in message or "invalid JSON" in message
            or message.startswith("Kalshi response schema rejected")):
        return "SCHEMA_REJECTED"
    if message.startswith("Unable to finalize transport provenance receipt"):
        return "LOCAL_RECEIPT_WRITE_FAILED"
    return None


class AuditedClientMixin:
    """Adds ``failure_ledger``, ``egress_denied`` and ``DataAccessError.failure_class``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_ledger: list[dict] = []
        self.egress_denied = False
        self._wire_event: str | None = None
        inner = self.transport
        self._audited_inner_transport = inner

        def audited(url, headers, timeout):
            if self.egress_denied:
                # A retry inside the original loop after a denial is not sent.
                self._wire_event = NOT_SENT
                raise URLError("Not sent after egress policy denial")
            try:
                status, response_headers, body = inner(url, headers, timeout)
            except (URLError, TimeoutError, OSError) as exc:
                self._wire_event = classify_network_failure(exc)
                if self._wire_event == EGRESS_DENIED:
                    self.egress_denied = True
                raise
            self._wire_event = classify_http_status(status) if status >= 400 else None
            return status, response_headers, body

        self._audit_transport_wrapper = audited
        self.transport = audited

    def _record(self, url, label):
        parts = urlsplit(url)
        self.failure_ledger.append({"host": parts.hostname, "path": parts.path, "failure_class": label, "at": utc_now()})

    def _finalize_new_receipt(self, cache_key, prior_meta, *, fixture):
        """Finalize only a receipt created by this call, never an older capture.

        The original adapter identifies fixtures by callable identity. Its
        audit wrapper is a different callable even when the underlying wire
        transport is unchanged. Preserve that actual transport origin in both
        the new archived receipt and the cache before any consumer receives it.
        """
        entry = self._cache.get(cache_key)
        if entry is None or entry[2] is prior_meta:
            return
        meta = entry[2]
        finalized = {**meta, "fixture": fixture,
                     "transport_origin": "INJECTED_TRANSPORT" if fixture else "DEFAULT_HTTPS_TRANSPORT"}
        try:
            Path(meta["raw_path"]).with_suffix(".meta.json").write_text(
                json.dumps(finalized, indent=2), encoding="utf-8")
        except (OSError, KeyError, TypeError, ValueError):
            # A failed metadata write must not later be served from cache as a
            # successful, correctly labeled source. Raw response bytes remain.
            self._cache.pop(cache_key, None)
            raise DataAccessError("Unable to finalize transport provenance receipt; source not admitted") from None
        meta.update(finalized)

    def _get(self, url: str, headers: dict | None = None):
        if self.egress_denied:
            self._record(url, NOT_SENT)
            error = DataAccessError("Not sent: an earlier request on this client was denied by the egress proxy")
            error.failure_class = NOT_SENT
            raise error
        self._wire_event = None
        cache_key = hashlib.sha256(sanitize_url(url).encode()).hexdigest()
        previous = self._cache.get(cache_key)
        prior_meta = previous[2] if previous is not None else None
        inner = (self._audited_inner_transport if self.transport is self._audit_transport_wrapper
                 else self.transport)
        fixture = inner is not _transport
        try:
            try:
                return super()._get(url, headers)
            finally:
                # Also runs after a schema rejection: the HTTP receipt must
                # describe the actual transport even if its payload is invalid.
                self._finalize_new_receipt(cache_key, prior_meta, fixture=fixture)
        except DataAccessError as exc:
            label = _local_label(str(exc)) or self._wire_event or "UNCLASSIFIED"
            if label == NOT_SENT:
                label = EGRESS_DENIED
            exc.failure_class = label
            self._record(url, label)
            raise


def _validate_market_record(row):
    """Reject malformed containers without inventing missing settlement facts."""
    if not isinstance(row, dict):
        raise DataAccessError("Kalshi response schema rejected: market record is not an object")
    ticker = row.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise DataAccessError("Kalshi response schema rejected: market identity is missing or invalid")
    # An absent optional rule remains absent; the original semantic gate decides
    # whether that incomplete record supports a question. Wrong-type values must
    # not reach the original string and timestamp consumers.
    for field in ("title", "subtitle", "yes_sub_title", "no_sub_title",
                  "rules_primary", "rules_secondary", "event_ticker",
                  "series_ticker", "close_time", "expiration_time", "status"):
        if field in row and not isinstance(row[field], str):
            raise DataAccessError("Kalshi response schema rejected: market text field is invalid")


class ValidatedKalshiClient(KalshiClient):
    """Validate external shapes before the byte-protected adapter consumes them.

    Transport, credentials, receipts, limits and normalization are inherited
    unchanged. A malformed HTTP-200 payload becomes a safe acquisition failure,
    so the live workflow can retain other sources and report its partial status.
    """

    def _get(self, url: str, headers: dict | None = None):
        data, meta = super()._get(url, headers)
        path = urlsplit(url).path.rstrip("/")
        if path.endswith("/markets"):
            rows = data.get("markets")
            if not isinstance(rows, list):
                raise DataAccessError("Kalshi response schema rejected: markets is not an array")
            cursor = data.get("cursor")
            if cursor is not None and not isinstance(cursor, str):
                raise DataAccessError("Kalshi response schema rejected: cursor is not text")
            for row in rows:
                _validate_market_record(row)
        elif "/markets/" in path and path.rsplit("/markets/", 1)[1].count("/") == 0:
            _validate_market_record(data.get("market"))
        return data, meta


class AuditedKalshiClient(AuditedClientMixin, ValidatedKalshiClient):
    pass


class AuditedMassiveClient(AuditedClientMixin, MassiveClient):
    pass


__all__ = ["AuditedClientMixin", "ValidatedKalshiClient", "AuditedKalshiClient", "AuditedMassiveClient", "EGRESS_DENIED", "NOT_SENT",
           "classify_http_status", "classify_network_failure"]
