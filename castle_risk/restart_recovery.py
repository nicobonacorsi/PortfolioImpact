"""Private, restart-safe job metadata and integrity-checked HTTP receipt reuse.

An interrupted analysis may reuse only exact-URL, SHA256-verified, dated
responses. Original receipt timestamps are never rewritten. In particular,
the final macro-quote refresh must use a fresh client and never call this helper.
No API keys, request headers or model prompts are persisted here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit

from .adapters import sanitize_url

JOB_FILE = ".portfolio_jobs.json"
RUN_NAME = re.compile(r"^\d{8}T\d{12}Z$")
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
MAX_ARCHIVE_AGE_SECONDS = 12 * 3600
MAX_ARCHIVE_FILES = 1200


def portfolio_fingerprint(rows: list[dict], total: float) -> str:
    """Tie a restart to precisely the submitted, normalized holdings."""
    canonical = {
        "notional": round(float(total), 2),
        "positions": sorted([
            {"symbol": str(row["symbol"]),
             "asset_type": str(row["asset_type"]),
             "weight": round(float(row["weight"]), 12),
             "input_asset_type": str(row.get("input_asset_type", ""))}
            for row in rows
        ], key=lambda row: row["symbol"]),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_job_records(folder: Path) -> list[dict]:
    path = Path(folder) / JOB_FILE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128_000:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        return []
    records = []
    for entry in payload.get("jobs", [])[:50]:
        if not isinstance(entry, dict):
            continue
        if not (isinstance(entry.get("id"), str) and JOB_ID.fullmatch(entry["id"]) and
                isinstance(entry.get("fingerprint"), str) and DIGEST.fullmatch(entry["fingerprint"]) and
                isinstance(entry.get("run_name"), str) and RUN_NAME.fullmatch(entry["run_name"]) and
                entry.get("status") in {"running", "interrupted", "complete", "failed"}):
            continue
        records.append({key: entry.get(key) for key in
                        ("id", "fingerprint", "run_name", "status", "progress",
                         "created_at", "updated_at", "error")})
    return records


def save_job_records(folder: Path, records: list[dict]) -> None:
    """Atomic, owner-only replacement. No raw input or provider secrets."""
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    path = root / JOB_FILE
    if path.is_symlink():
        raise OSError("Refusing a symlinked job registry")
    tmp = root / ("." + secrets.token_hex(12) + ".job.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump({"schema": 1, "jobs": records[-50:]}, handle,
                      ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _utc_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else None
    except ValueError:
        return None


def prime_verified_http_cache(client, directory: Path, *, max_age_seconds: int = MAX_ARCHIVE_AGE_SECONDS) -> int:
    """Reuse successful saved GETs under their *original* receipt timestamps.

    The normal adapter still validates the host and provider schema on cache
    hits. A missing, modified, expired, cross-host or symlinked file is skipped.
    No transport method is replaced; a cache miss performs the original GET.
    """
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        return 0
    allowed = getattr(client, "allowed_hosts", frozenset())
    cache = getattr(client, "_cache", None)
    if not isinstance(cache, dict):
        return 0
    now = datetime.now(timezone.utc)
    admitted = {}
    for index, metadata_path in enumerate(sorted(root.glob("*.meta.json"))):
        if index >= MAX_ARCHIVE_FILES:
            break
        if metadata_path.is_symlink():
            continue
        raw_path = metadata_path.with_name(metadata_path.name.removesuffix(".meta.json") + ".json")
        if raw_path.is_symlink() or not raw_path.is_file():
            continue
        try:
            if raw_path.stat().st_size > 16_000_000 or metadata_path.stat().st_size > 16_000:
                continue
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or meta.get("status") != 200 or meta.get("fixture") is True:
                continue
            if meta.get("timestamp_kind") == "request_started_at":
                continue
            url = meta.get("url")
            if not isinstance(url, str) or sanitize_url(url) != url:
                continue
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname not in allowed:
                continue
            when = _utc_time(meta.get("retrieved_at"))
            if when is None or not 0 <= (now - when).total_seconds() <= max_age_seconds:
                continue
            saved_hash = meta.get("sha256")
            raw = raw_path.read_bytes()
            if not isinstance(saved_hash, str) or not DIGEST.fullmatch(saved_hash):
                continue
            if hashlib.sha256(raw).hexdigest() != saved_hash:
                continue
            recorded_path = meta.get("raw_path")
            if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != raw_path.resolve():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
            key = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if key not in admitted or admitted[key][0] < when:
                # The original path/time/response hash are retained for audits.
                metadata = {**meta, "cache_hit": True, "replayed_from_archive": True}
                admitted[key] = (when, payload, metadata)
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            continue
    for key, (_when, payload, metadata) in admitted.items():
        age_seconds = (now - _when).total_seconds()
        cache[key] = (time.monotonic() - age_seconds, payload, metadata)
    if admitted:
        # Cache only for this explicit resumed attempt; final quotes use a
        # separate, unprimed client and are always refreshed.
        client.cache_ttl = max(float(client.cache_ttl), float(max_age_seconds))
    return len(admitted)


def find_matching_legacy_run(root: Path, symbols: set[str],
                             *, max_age_seconds: int = MAX_ARCHIVE_AGE_SECONDS) -> str | None:
    """Adopt a pre-upgrade incomplete run only if its Massive URLs show every ticker.

    Portfolio weights are never inferred from saved provider responses. The
    supplied portfolio is authoritative; these files are merely exact-URL
    replay candidates for those holdings.
    """
    root = Path(root)
    if not root.is_dir() or not symbols:
        return None
    now = datetime.now(timezone.utc)
    for folder in sorted(root.iterdir(), reverse=True):
        if folder.is_symlink() or not folder.is_dir() or not RUN_NAME.fullmatch(folder.name):
            continue
        if (folder / "derived_assessment.json").exists():
            continue
        try:
            started = datetime.strptime(folder.name, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if started is None or not 0 <= (now - started).total_seconds() <= max_age_seconds:
            continue
        evidence = folder / "massive_private"
        if not evidence.is_dir() or evidence.is_symlink():
            continue
        seen = set()
        for index, path in enumerate(evidence.glob("*.meta.json")):
            if index >= 800:
                break
            if path.is_symlink() or path.stat().st_size > 16_000:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                url = data.get("url") if isinstance(data, dict) else None
                if not isinstance(url, str):
                    continue
                for ticker in symbols - seen:
                    if re.search(r"(?<![A-Z0-9])" + re.escape(ticker) + r"(?![A-Z0-9])", url.upper()):
                        seen.add(ticker)
            except (OSError, ValueError, TypeError):
                continue
        if symbols <= seen:
            return folder.name
    return None
