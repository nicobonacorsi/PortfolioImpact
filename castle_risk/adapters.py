"""Read-only data clients with provenance, bounded requests and secret-safe archives.

Kalshi timestamps are HTTP receipt times, not exchange quote-update timestamps.
Massive API keys are sent only in Authorization headers. No trading methods exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
MASSIVE_BASE = "https://api.massive.com"
_SECRET_KEYS = {"apikey", "api_key", "authorization", "token", "access_token", "secret", "password", "x-api-key"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def sanitize_url(url: str) -> str:
    p = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in _SECRET_KEYS]
    return urlunsplit((p.scheme, p.netloc.split("@")[-1], p.path, urlencode(query), ""))


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in _SECRET_KEYS else redact(v, secrets)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(api[_-]?key|access_token|token)=([^&\s\"']+)", r"\1=[REDACTED]", value)
    return value


NAV_WEIGHT_SUM_TOLERANCE = 0.01

class DataAccessError(RuntimeError):
    """Public safe error; never includes server body, headers or credentials."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Avoid forwarding Authorization to a redirected origin.
        return None


def _transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    req = Request(url, headers=headers, method="GET")
    with build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
        body = response.read(16_000_001)
        if len(body) > 16_000_000:
            raise DataAccessError("Response exceeds the 16 MB safety cap")
        return response.status, dict(response.headers), body


@dataclass
class Market:
    ticker: str
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    event_ticker: str = ""
    series_ticker: str = ""
    close_time: str = ""
    expiration_time: str = ""
    settlement_timer_seconds: int | None = None
    source_url: str = ""
    retrieved_at: str = ""
    raw_path: str = ""
    yes_bid: float | None = None
    yes_ask: float | None = None
    quote_timestamp: str = ""
    status: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        val = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return val if math.isfinite(val) else None


def _price(raw: dict, name: str) -> float | None:
    if name + "_dollars" in raw:
        return _number(raw[name + "_dollars"])
    old = _number(raw.get(name))
    return None if old is None else old / 100


def normalize_market(raw: dict, retrieved_at: str, raw_path: str = "") -> Market:
    """Normalize dollar fields; do not infer freshness from metadata updated_time."""
    ticker = str(raw.get("ticker", ""))
    if not ticker:
        raise DataAccessError("Kalshi market response has no ticker")
    bid, ask = _price(raw, "yes_bid"), _price(raw, "yes_ask")
    for side in ("bid", "ask"):
        size_key = f"yes_{side}_size_fp"
        size = _number(raw.get(size_key))
        side_price = bid if side == "bid" else ask
        # Zero is commonly a missing-side sentinel. Preserve a zero quote only
        # when an explicit positive displayed size proves an executable side.
        if (size_key in raw and (size is None or size <= 0)) or (side_price == 0 and (size is None or size <= 0)):
            if side == "bid":
                bid = None
            else:
                ask = None
    event = str(raw.get("event_ticker", ""))
    return Market(
        ticker=ticker, title=raw.get("title", ""), subtitle=raw.get("subtitle", ""),
        yes_sub_title=raw.get("yes_sub_title", ""), no_sub_title=raw.get("no_sub_title", ""),
        rules_primary=raw.get("rules_primary", ""), rules_secondary=raw.get("rules_secondary", ""),
        event_ticker=event, series_ticker=raw.get("series_ticker", event.split("-")[0]),
        close_time=raw.get("close_time", ""), expiration_time=raw.get("expiration_time", ""),
        settlement_timer_seconds=raw.get("settlement_timer_seconds"),
        source_url=f"{KALSHI_BASE}/markets/{quote(ticker, safe='')}",
        retrieved_at=retrieved_at, raw_path=raw_path, yes_bid=bid, yes_ask=ask,
        quote_timestamp=retrieved_at, status=raw.get("status", ""),
        metadata={"raw": raw, "quote_timestamp_kind": "retrieval_receipt", "quote_exchange_timestamp": None,
                  "venue": "kalshi", "market_type": raw.get("market_type"),
                  "notional_value_dollars": raw.get("notional_value_dollars"),
                  "expected_expiration_time": raw.get("expected_expiration_time"),
                  "metadata_updated_time": raw.get("updated_time"),
                  "zero_size_quotes_removed": True},
    )


@dataclass
class NormalizedQuote:
    lower: float = 0.0
    upper: float = 1.0
    usable: bool = False
    exclusion_reason: str = ""
    observed_yes_bid: float | None = None
    observed_yes_ask: float | None = None
    quote_timestamp: str = ""
    timestamp_kind: str = "retrieval_receipt"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _best_bid(levels: list | None, cents: bool = False) -> float | None:
    valid = []
    for level in levels or []:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        p, size = _number(level[0]), _number(level[1])
        if p is not None and size is not None and size > 0:
            valid.append(p / 100 if cents else p)
    return max(valid) if valid else None


def normalize_quote(market: Market, orderbook: dict | None = None, *, now: datetime | str | None = None,
                    max_age_seconds: float = 300, payout: float = 1.0,
                    discount_factor: float = 1.0) -> NormalizedQuote:
    """Quote-consistent Q bounds under user-declared common numeraire assumptions.

    Standard unit-dollar binary payouts only. For a direct YES quote at T,
    normalize by D(0,T). For orderbook complements D must be one, so
    YES ask = 1 - NO bid. Missing executable sides give trivial 0/1 bounds.
    Neither quote-update age nor actual available size beyond top levels is certified.
    """
    result = NormalizedQuote(quote_timestamp=market.quote_timestamp)
    if not math.isfinite(payout) or not math.isfinite(discount_factor) or payout <= 0 or discount_factor <= 0:
        result.exclusion_reason = "Invalid payout or discount factor"
        return result
    if abs(payout - 1.0) > 1e-12:
        result.exclusion_reason = "Only standard unit-dollar binary payouts are supported"
        return result
    if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
        result.exclusion_reason = "Invalid maximum quote age"
        return result
    if market.metadata.get("market_type") not in (None, "", "binary"):
        result.exclusion_reason = "Only binary contracts are supported"
        return result
    if market.status not in ("open", "active"):
        result.exclusion_reason = "Market is not open"
        return result
    reported_payout = _number(market.metadata.get("notional_value_dollars"))
    if reported_payout is not None and abs(reported_payout - payout) > 1e-9:
        result.exclusion_reason = "Configured payout differs from contract notional"
        return result
    if orderbook is not None and abs(discount_factor - 1.0) > 1e-12:
        result.exclusion_reason = "Orderbook complement normalization requires the declared unit-discount collateral convention"
        return result
    bid, ask = market.yes_bid, market.yes_ask
    if orderbook is not None:
        provenance = orderbook.get("_provenance", {})
        result.quote_timestamp = provenance.get("retrieved_at", market.quote_timestamp)
        if "orderbook_fp" in orderbook:
            book = orderbook["orderbook_fp"] or {}
            bid = _best_bid(book.get("yes_dollars"))
            no_bid = _best_bid(book.get("no_dollars"))
        elif "orderbook" in orderbook:
            book = orderbook["orderbook"] or {}
            if "yes_dollars" in book or "no_dollars" in book:
                bid = _best_bid(book.get("yes_dollars"))
                no_bid = _best_bid(book.get("no_dollars"))
            else:
                bid = _best_bid(book.get("yes"), cents=True)
                no_bid = _best_bid(book.get("no"), cents=True)
        else:
            result.exclusion_reason = "Orderbook response has no recognized book"
            return result
        ask = None if no_bid is None else payout - no_bid
        result.notes.append("YES ask reconstructed from executable NO bid; full-set redemption at stated payout assumed")
    result.observed_yes_bid, result.observed_yes_ask = bid, ask
    try:
        current = parse_time(now) if isinstance(now, str) else (now or datetime.now(timezone.utc))
        age = (current - parse_time(result.quote_timestamp)).total_seconds()
    except (ValueError, TypeError):
        result.exclusion_reason = "Missing or invalid quote receipt timestamp"
        return result
    if age < -5:
        result.exclusion_reason = "Quote receipt timestamp is in the future"
        return result
    if age > max_age_seconds:
        result.exclusion_reason = "Stale quotation receipt"
        return result
    if bid is None and ask is None:
        result.exclusion_reason = "Insufficient price evidence: no executable bid or ask"
        return result
    denom = payout * discount_factor
    if any(v is not None and (not math.isfinite(v) or v < 0 or v > payout) for v in (bid, ask)):
        result.exclusion_reason = "Quotation outside payout range"
        return result
    lower = 0.0 if bid is None else bid / denom
    upper = 1.0 if ask is None else ask / denom
    if lower > upper + 1e-12 or lower > 1 + 1e-12 or upper < 0:
        result.exclusion_reason = "Crossed quotes or incompatible payout/discount assumptions"
        return result
    result.lower, result.upper = max(0.0, lower), min(1.0, upper)
    result.usable = True
    if bid is None or ask is None:
        result.notes.append("One-sided price evidence: the missing side uses only the mathematical [0,1] restriction")
    result.notes.append("Receipt freshness checked; exchange quote-update timestamp is not supplied")
    result.notes.append("Spread-consistent reference pricing measure; not a physical-probability confidence interval; fees excluded")
    return result


class _ReadOnlyClient:
    def __init__(self, data_dir: str | Path, *, timeout: float = 20, min_interval: float = 0.35,
                 max_requests: int = 30, max_retries: int = 2, cache_ttl: float = 30,
                 transport: Callable | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.min_interval = timeout, min_interval
        self.max_requests, self.max_retries, self.cache_ttl = max_requests, max_retries, cache_ttl
        self.transport = transport or _transport
        self.request_count = 0
        self._last_request = 0.0
        self._cache: dict = {}
        self._secrets: tuple[str, ...] = ()

    def _get(self, url: str, headers: dict | None = None) -> tuple[dict, dict]:
        url = sanitize_url(url)
        cache_key = hashlib.sha256(url.encode()).hexdigest()
        if cache_key in self._cache:
            cached_at, payload, meta = self._cache[cache_key]
            if time.monotonic() - cached_at < self.cache_ttl:
                return payload, {**meta, "cache_hit": True}
        if urlsplit(url).hostname not in self.allowed_hosts or urlsplit(url).scheme != "https":
            raise DataAccessError("Refused an untrusted data host")
        headers = {"Accept": "application/json", "User-Agent": "CastleRiskAgent/0.1 (read-only)", **(headers or {})}
        for attempt in range(self.max_retries + 1):
            if self.request_count >= self.max_requests:
                raise DataAccessError("Data request budget exhausted")
            delay = self.min_interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self.request_count += 1
            self._last_request = time.monotonic()
            try:
                status, response_headers, body = self.transport(url, headers, self.timeout)
                if status >= 400:
                    raise HTTPError(url, status, "Data service rejected request", response_headers, None)
                if len(body) > 16_000_000:
                    raise DataAccessError("Response exceeds the 16 MB safety cap")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise DataAccessError("Data response is not a JSON object")
                safe_payload = redact(payload, self._secrets)
                redacted = safe_payload != payload
                saved_body = json.dumps(safe_payload, ensure_ascii=False, indent=2).encode() if redacted else body
                received = utc_now()
                name = f"{received.replace(':', '').replace('.', '')}_{cache_key[:12]}"
                raw_path = self.data_dir / (name + ".json")
                raw_path.write_bytes(saved_body)
                meta = {"url": url, "retrieved_at": received, "status": status,
                        "server_date": response_headers.get("Date") or response_headers.get("date"),
                        "sha256": hashlib.sha256(saved_body).hexdigest(), "raw_path": str(raw_path),
                        "transport": "https GET", "fixture": self.transport is not _transport,
                        "redacted": redacted, "cache_hit": False}
                raw_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                self._cache[cache_key] = (time.monotonic(), safe_payload, meta)
                return safe_payload, meta
            except HTTPError as exc:
                # Discard error body and exception text: either may echo credentials.
                code = exc.code
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise DataAccessError(f"Data service returned HTTP {code}; credentials and server body omitted") from None
            except (URLError, TimeoutError, OSError):
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise DataAccessError("Data service network request failed; details omitted") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise DataAccessError("Data service returned invalid JSON") from None
        raise DataAccessError("Data request failed")


class KalshiClient(_ReadOnlyClient):
    allowed_hosts = {"external-api.kalshi.com"}

    def list_markets(self, series_ticker: str | None = None, status: str | None = "open", *,
                     event_ticker: str | None = None, limit: int = 100, max_pages: int = 3) -> list[Market]:
        if not 1 <= limit <= 1000 or not 1 <= max_pages <= 10:
            raise ValueError("Invalid market retrieval limits")
        params: dict = {"limit": min(limit, 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        markets, seen = [], set()
        for _ in range(max_pages):
            data, meta = self._get(KALSHI_BASE + "/markets?" + urlencode(params))
            for raw in data.get("markets", []):
                markets.append(normalize_market(raw, meta["retrieved_at"], meta["raw_path"]))
                if len(markets) >= limit:
                    return sorted(markets, key=lambda m: (m.close_time, m.ticker))
            cursor = data.get("cursor")
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
            params["cursor"] = cursor
            params["limit"] = min(1000, limit - len(markets))
        return sorted(markets, key=lambda m: (m.close_time, m.ticker))

    def get_market(self, ticker: str) -> Market:
        data, meta = self._get(f"{KALSHI_BASE}/markets/{quote(ticker, safe='')}")
        return normalize_market(data.get("market", {}), meta["retrieved_at"], meta["raw_path"])

    def get_orderbook(self, ticker: str) -> dict:
        data, meta = self._get(f"{KALSHI_BASE}/markets/{quote(ticker, safe='')}/orderbook")
        return {**data, "_provenance": meta}

    def get_event(self, ticker: str) -> dict:
        data, meta = self._get(f"{KALSHI_BASE}/events/{quote(ticker, safe='')}")
        return {**data, "_provenance": meta}

    def get_series(self, ticker: str) -> dict:
        data, meta = self._get(f"{KALSHI_BASE}/series/{quote(ticker, safe='')}")
        return {**data, "_provenance": meta}

    def search_markets(self, query: str, limit: int = 15) -> list[Market]:
        """Bounded local keyword search over official series metadata and open markets.

        This is not an undocumented venue full-text API. A budget limits discovery
        coverage; caller must not interpret an empty result as no such market exists.
        """
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be 1..100")
        if query.strip().upper().startswith("KX") and re.fullmatch(r"[A-Za-z0-9-]+", query.strip()):
            ticker = query.strip().upper()
            if "-" in ticker:
                return self.list_markets(event_ticker=ticker, limit=limit)
            return self.list_markets(series_ticker=ticker, limit=limit)
        tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "a", "in", "at", "of", "will", "by"}
        series, _ = self._get(KALSHI_BASE + "/series")
        ranked = []
        for item in series.get("series", []):
            text = " ".join(str(item.get(k, "")) for k in ("title", "ticker", "category", "tags")).lower()
            score = sum(token in text for token in tokens)
            if score:
                ranked.append((score, str(item.get("ticker", ""))))
        markets = []
        for _, ticker in sorted(ranked, reverse=True)[:3]:
            markets.extend(self.list_markets(series_ticker=ticker, limit=100, max_pages=1))
        def score(m):
            text = " ".join((m.title, m.subtitle, m.rules_primary)).lower()
            return (-sum(token in text for token in tokens), m.close_time, m.ticker)
        return sorted(markets, key=score)[:limit]


class MassiveClient(_ReadOnlyClient):
    allowed_hosts = {"api.massive.com"}

    def __init__(self, data_dir: str | Path, *, api_key: str | None = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self._api_key = api_key if api_key is not None else os.environ.get("MASSIVE_API_KEY", "")
        self._secrets = (self._api_key,)

    def _authenticated_get(self, url: str) -> tuple[dict, dict]:
        if not self._api_key:
            raise DataAccessError("MASSIVE_API_KEY is not configured; ETF exposure remains unidentified")
        return self._get(url, {"Authorization": "Bearer " + self._api_key})

    def holdings(self, ticker: str, *, effective_date: str | None = None, max_pages: int = 5,
                 as_of_date: str | None = None, lookback_days: int = 30) -> dict:
        """Retrieve a bounded range, then select one exact effective/processed pair.

        No server-side sort is requested. Pagination must finish before the latest
        eligible observed batch is trusted. Even then, ``complete`` means the API
        query completed: it does not prove full fund disclosure or point-in-time
        availability. No intraday publication timestamp or immutable batch ID is
        supplied by this endpoint. Missing weights/identities remain unresolved.
        """
        if not isinstance(ticker, str) or not re.fullmatch(r"[A-Z0-9.\-]{1,20}", ticker.upper()):
            raise ValueError("Invalid ETF ticker")
        ticker = ticker.upper()
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 20:
            raise ValueError("Invalid holdings page limit")
        if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or not 1 <= lookback_days <= 366:
            raise ValueError("Holdings lookback must be 1..366 calendar days")

        def calendar_date(value, name):
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError(f"{name} must be YYYY-MM-DD")
            return date.fromisoformat(value)

        cutoff = calendar_date(as_of_date, "as_of_date") if as_of_date else datetime.now(timezone.utc).date()
        if cutoff > datetime.now(timezone.utc).date():
            raise ValueError("as_of_date cannot be in the future")
        if effective_date is not None:
            start = end = calendar_date(effective_date, "effective_date")
            if end > cutoff:
                raise ValueError("effective_date cannot exceed as_of_date")
        else:
            end, start = cutoff, cutoff - timedelta(days=lookback_days - 1)
        base = MASSIVE_BASE + "/etf-global/v1/constituents"
        params = {"composite_ticker": ticker, "effective_date.gte": start.isoformat(),
                  "effective_date.lte": end.isoformat(), "limit": 5000}
        url = base + "?" + urlencode(params)
        raw_paths, page_records, rows, seen = [], [], [], set()
        complete, issues, warnings = False, [], []
        initial_requests = self.request_count
        for _ in range(max_pages):
            if url in seen:
                issues.append("PAGINATION_CYCLE")
                break
            seen.add(url)
            try:
                data, meta = self._authenticated_get(url)
            except DataAccessError as exc:
                if not raw_paths:
                    raise
                issues.append("PAGINATION_FETCH_FAILED")
                warnings.append(str(exc))  # DataAccessError is deliberately secret-safe.
                break
            raw_paths.append(meta["raw_path"])
            page_records.append({k: meta[k] for k in ("raw_path", "sha256", "retrieved_at", "fixture", "cache_hit")})
            page_rows = data.get("results")
            if data.get("status", "OK") != "OK" or not isinstance(page_rows, list) or not all(isinstance(r, dict) for r in page_rows):
                issues.append("INVALID_RESPONSE_SHAPE_OR_STATUS")
                break
            rows.extend(page_rows)
            next_url = data.get("next_url")
            if not next_url:
                complete = True
                break
            if not isinstance(next_url, str):
                issues.append("INVALID_PAGINATION_LINK")
                break
            p = urlsplit(next_url)
            try:
                valid_port = p.port in (None, 443)
            except ValueError:
                valid_port = False
            if p.scheme != "https" or p.hostname != "api.massive.com" or p.path != "/etf-global/v1/constituents" or p.username or p.password or not valid_port:
                raise DataAccessError("Refused untrusted ETF pagination URL")
            url = sanitize_url(next_url)
        if not complete:
            issues.append("INCOMPLETE_QUERY")

        # Dates identify distinct observed revisions, never a pooled snapshot.
        candidates: dict[tuple[str, str], list[dict]] = {}
        excluded_after_cutoff = 0
        for row in rows:
            try:
                effective = calendar_date(row.get("effective_date"), "effective_date")
                processed = calendar_date(row.get("processed_date"), "processed_date")
            except ValueError:
                issues.append("MISSING_OR_INVALID_BATCH_DATES")
                continue
            if row.get("composite_ticker") != ticker or not start <= effective <= end:
                issues.append("ROW_OUTSIDE_REQUESTED_FUND_OR_DATE_RANGE")
                continue
            if processed < effective:
                issues.append("PROCESSED_DATE_PRECEDES_EFFECTIVE_DATE")
                continue
            if processed > cutoff:
                excluded_after_cutoff += 1
                continue
            candidates.setdefault((effective.isoformat(), processed.isoformat()), []).append(row)
        batch = max(candidates) if candidates else None
        selected = candidates.get(batch, [])
        if not selected:
            issues.append("NO_ELIGIBLE_DATED_BATCH_IN_BOUNDED_WINDOW")
        normalized, fingerprints, identifiers, ranks = [], set(), {}, {}
        duplicate_rows = 0
        for row in selected:
            fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if fingerprint in fingerprints:
                duplicate_rows += 1
                continue
            fingerprints.add(fingerprint)
            item = _normalize_constituent(row, ticker)
            normalized.append(item)
            for issue in item["quality_issues"]:
                issues.append(issue)
            # An exact duplicate is harmless. Conflicting versions of the same
            # security in one date-pair are not additive economic positions.
            identity_keys = [("ticker", item["source_ticker"])] if item["source_ticker"] else []
            identity_keys += [(kind, value) for kind, value in item["identifiers"].items() if value]
            for key in identity_keys:
                if key in identifiers:
                    issues.append("CONFLICTING_DUPLICATE_SECURITY")
                identifiers[key] = fingerprint
            rank = row.get("constituent_rank")
            if rank is not None:
                if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                    issues.append("INVALID_CONSTITUENT_RANK")
                else:
                    if rank in ranks:
                        issues.append("CONFLICTING_CONSTITUENT_RANK")
                    ranks[rank] = fingerprint
        weights = [h["weight"] for h in normalized if h["weight"] is not None]
        weight_sum = math.fsum(weights)
        # Issuer files routinely report 100.0-100.5% of NAV (rounding, accrued
        # cash, settlement timing). Up to 1% excess is kept as a visible warning
        # without renormalization; anything larger still fails closed.
        if weight_sum > 1 + NAV_WEIGHT_SUM_TOLERANCE:
            issues.append("REPORTED_NAV_WEIGHT_EXCEEDS_ONE")
        elif weight_sum > 1 + 1e-12:
            warnings.append(f"Reported constituent weights sum to {weight_sum:.4%} of NAV; "
                            "the excess is within the rounding tolerance and is not renormalized")
        known = math.fsum(h["weight"] for h in normalized if h["weight"] is not None and h["weight"] >= 0 and h["asset_type"] in {"stock", "cash"})
        unclassified = math.fsum(h["weight"] for h in normalized if h["weight"] is not None and h["weight"] >= 0 and h["asset_type"] == "other")
        nested = math.fsum(h["weight"] for h in normalized if h["weight"] is not None and h["weight"] >= 0 and h["asset_type"] == "etf")
        missing_weights = sum(h["weight"] is None for h in normalized)
        if missing_weights:
            warnings.append("Some constituent weights are unavailable; their NAV share is not inferred")
        if unclassified:
            warnings.append("Unsupported or unidentified instruments retain their stated NAV mass as unresolved exposure")
        issues = sorted(set(issues))
        retrieved_at = page_records[-1]["retrieved_at"] if page_records else utc_now()
        result = {"fund_symbol": ticker, "as_of": batch[0] if batch else end.isoformat(),
                  "processed_date": batch[1] if batch else None, "retrieved_at": retrieved_at,
                  "source_url": base, "holdings": normalized,
                  "reliable": complete and bool(selected) and not issues,
                  "weight_basis": "fraction_of_nav", "complete": complete,
                  "metadata": {"raw_paths": raw_paths, "pages": page_records, "query": params,
                               "selection_cutoff_date": cutoff.isoformat(), "selected_batch": list(batch) if batch else None,
                               "selection_provisional": not complete, "candidate_batches": [
                                   {"effective_date": e, "processed_date": p, "row_count": len(rs)}
                                   for (e, p), rs in sorted(candidates.items())],
                               "excluded_processed_after_cutoff": excluded_after_cutoff,
                               "request_count": self.request_count - initial_requests,
                               "raw_row_count": len(rows), "selected_raw_row_count": len(selected),
                               "exact_duplicate_rows_removed": duplicate_rows, "weight_sum": weight_sum,
                               "identified_weight": known, "unclassified_weight": unclassified,
                               "nested_etf_weight": nested, "missing_weight_row_count": missing_weights,
                               "unreported_nav_weight": max(0.0, 1.0 - weight_sum),
                               "quality_issues": issues, "quality_warnings": warnings,
                               "point_in_time_verified": False,
                               "completeness_scope": "pagination-complete bounded API query, not independently reconciled fund NAV",
                               "note": "No weight renormalization. Effective/processed dates are not publication timestamps. "
                                       "The endpoint supplies no immutable intraday batch ID; conflicting same-date revisions fail closed."}}
        # Persist normalized provenance separately from archived (sanitized) wire
        # responses. All unselected revisions remain available in raw_paths.
        stamp = retrieved_at.replace(":", "").replace(".", "")
        digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        normalized_path = self.data_dir / f"holdings_{ticker}_{stamp}_{digest}.normalized.json"
        result["metadata"]["normalized_path"] = str(normalized_path)
        normalized_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result


def _normalize_constituent(row: dict, fund_ticker: str) -> dict:
    """Conservative typed identity; identifier syntax is not external resolution."""
    raw_symbol = row.get("constituent_ticker")
    raw_name = row.get("constituent_name")
    # Preserve a provider-supplied name verbatim. It can carry economic-class
    # evidence (e.g. CASH FUNDS) absent from the generic legal OPEN-END FUND
    # label. No ticker-to-name guess or external identity verification occurs.
    source_name = raw_name if isinstance(raw_name, str) else None
    symbol = str(raw_symbol or "").strip().upper()
    valid_symbol = bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.\-_/]{0,31}", symbol))
    identifiers, notes, issues = {}, [], []
    for kind, pattern in (("isin", r"[A-Z]{2}[A-Z0-9]{9}[0-9]"),
                          ("figi", r"[A-Z0-9]{11}[0-9]")):
        value = str(row.get(kind) or "").strip().upper()
        if value and not re.fullmatch(pattern, value):
            notes.append(f"Malformed {kind.upper()} retained in the raw response; not used as identity")
            value = ""
        identifiers[kind] = value
    weight = _number(row.get("weight"))
    if row.get("weight") not in (None, "") and weight is None:
        issues.append("INVALID_WEIGHT")
    if weight is not None and not 0 <= weight <= 1:
        issues.append("NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT")
    asset = (str(row.get("asset_class", "")) + " " + str(row.get("security_type", ""))).lower()
    security_type = str(row.get("security_type", "")).lower()
    asset_type = "other"
    if any(term in asset for term in ("swap", "future", "option", "derivative", "warrant", "leveraged", "inverse", "forward", "contract for difference")):
        notes.append("Derivative/leverage exposure is outside the plain-instrument look-through model")
    elif "cash" in asset or security_type == "currency":
        if row.get("currency_traded") == "USD":
            asset_type, symbol, valid_symbol = "cash", "CASH", True
        else:
            notes.append("Cash currency is unknown or not USD")
    elif "exchange traded fund" in asset or "etf" in asset:
        asset_type = "etf"
    elif security_type.startswith(("common stock", "common equity", "ordinary share", "preferred stock", "reit")):
        # Listed REIT shares are exchange-traded equity of the named issuer.
        asset_type = "stock"
    elif str(row.get("asset_class") or "").strip().lower() == "equity" and \
            security_type.strip() in {"preference", "preferred", "adr", "gdr", "nvdr", "unit", "right",
                                      "depositary receipt"}:
        # Preferred shares, depositary receipts, Thai NVDRs, Brazilian share units
        # and rights are equity claims on the named issuer.
        asset_type = "stock"
        notes.append(f"Equity security type {row.get('security_type')} treated as issuer equity")
    elif (not security_type.strip() or security_type == "none") and \
            str(row.get("asset_class") or "").strip().lower() == "equity" and \
            any(str(row.get(kind) or "").strip() for kind in ("isin", "figi")):
        # Some issuer rows omit security_type but carry an Equity class and a
        # security identifier; they are shares, not unclassified instruments.
        asset_type = "stock"
        notes.append("Security type missing; classified as stock from asset_class Equity plus an identifier")
    if not valid_symbol:
        asset_type = "other"
        symbol = next((kind.upper() + "_" + value for kind, value in identifiers.items() if value),
                      "UNIDENTIFIED_" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12].upper())
        notes.append("No usable ticker: identifier-only row remains unresolved; no ticker is invented")
    if symbol == fund_ticker:
        issues.append("SELF_REFERENTIAL_FUND_HOLDING")
    if not any(identifiers.values()):
        notes.append("Ticker-only identity; no valid-format ISIN or FIGI supplied")
    return {"symbol": symbol, "source_ticker": str(raw_symbol or "").strip().upper(),
            "constituent_name": source_name, "name": source_name,
            "weight": weight, "asset_type": asset_type, "identifier": identifiers["figi"] or identifiers["isin"],
            "identifiers": identifiers, "identifier_assurance": "syntax_only_not_security_master_verified",
            "effective_date": row.get("effective_date"), "processed_date": row.get("processed_date"),
            "security_type": row.get("security_type"), "asset_class": row.get("asset_class"),
            "currency_traded": row.get("currency_traded"), "constituent_rank": row.get("constituent_rank"),
            "quality_issues": issues, "notes": notes}


def market_from_raw(record: dict, retrieved_at: str, raw_path: str = "", source_url: str | None = None) -> Market:
    """Public constructor for authenticated, archived, or independently supplied records."""
    market = normalize_market(record, retrieved_at, raw_path)
    if source_url is not None:
        market.source_url = sanitize_url(source_url)
    return market


def load_snapshot(path: str | Path) -> list[Market]:
    """Read a previously retrieved API JSON and its adjacent .meta.json provenance.

    Quotes retain their original receipt time and are not made live by loading.
    Event, market-list, and single-market response formats are recognized.
    """
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("sha256") != digest:
        raise DataAccessError("Archived API response SHA256 mismatch")
    if meta.get("timestamp_kind") == "request_started_at":
        raise DataAccessError("Snapshot records request start, not a usable quote receipt timestamp")
    records = payload.get("markets", [])
    if "market" in payload:
        records = [payload["market"]]
    result = [normalize_market(record, meta["retrieved_at"], str(path)) for record in records]
    for market in result:
        market.metadata["archived"] = True
        market.metadata["fixture"] = meta.get("fixture", False)
    return result
