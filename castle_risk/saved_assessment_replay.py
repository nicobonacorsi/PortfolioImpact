"""Read-only R30 upgrade of a user's own already-acquired local assessment.

This is not a live data fetch, an authenticated acceptance upgrade or a new
historical observation. Only the default projection is recomputed, at the
original capture instant, from the original empirical reports and verified
local inputs. No acquisition client or model is invoked and nothing is saved.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .adapters import Market, parse_time
from .agentic_bridge import AgenticEvidenceError, _receipt
from .default_assessment import (build_default_assessment, resolve_assessment_identity,
                                 retain_reported_instrument_restrictions)
from .option1_brief import portfolio_input_fingerprint
from .instrument_exposures import build_instrument_exposures
from .portfolio import parse_portfolio


_MAX_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_RUN = re.compile(r"\d{8}T\d{12}Z")
_HASH = re.compile(r"[0-9a-f]{64}")
_PATH_KEYS = {"raw_path", "raw_paths", "normalized_path", "run_directory"}
_MACRO = {"FED_POLICY", "US_INFLATION", "US_PAYROLLS"}


class _ReplayIssue(ValueError):
    """Safe machine-readable cause; never expose provider bodies or paths."""


def _fail(code):
    raise _ReplayIssue(code)


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def finite(_):
        _fail("NONFINITE_JSON_INPUT")

    return json.loads(body, object_pairs_hook=unique, parse_constant=finite)


def _member(root, parts):
    if not parts or any(not part or part in {".", ".."} or ":" in part or "\x00" in part for part in parts):
        _fail("UNSAFE_SAVED_MEMBER_PATH")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            _fail("SYMLINK_SAVED_MEMBER_REJECTED")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _fail("SAVED_MEMBER_NOT_WITHIN_SUPPLIED_RUN")
    if not current.is_file():
        _fail("MISSING_SAVED_MEMBER")
    return current


def _read(path, *, maximum=_MAX_BYTES):
    size = path.stat().st_size
    if not 0 <= size <= maximum:
        _fail("SAVED_MEMBER_SIZE_LIMIT")
    with path.open("rb") as stream:
        body = stream.read(maximum + 1)
    if len(body) != size or len(body) > maximum:
        _fail("SAVED_MEMBER_CHANGED_OR_TOO_LARGE")
    return body


def _relocated_path(value, root, run_id, *, directory=False):
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("INVALID_SAVED_PATH")
    parts = re.split(r"[\\/]+", value)
    if any(part in {".", ".."} for part in parts):
        _fail("SAVED_PATH_TRAVERSAL")
    anchors = [i for i in range(len(parts) - 2)
               if parts[i:i + 3] == ["runtime", "option1_live", run_id]]
    if len(anchors) != 1:
        _fail("SAVED_PATH_NOT_ANCHORED_TO_THIS_RUN")
    tail = parts[anchors[0] + 3:]
    if tail and tail[-1] == "":
        tail.pop()
    if directory:
        if tail:
            _fail("SAVED_RUN_DIRECTORY_MISMATCH")
        return root
    return _member(root, tail)


def _relocate(value, root, run_id, raw_sources):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _PATH_KEYS and item not in (None, ""):
                if key == "raw_paths":
                    if not isinstance(item, list):
                        _fail("INVALID_SAVED_PATH_LIST")
                    paths = [_relocated_path(path, root, run_id) for path in item]
                    raw_sources.update(paths)
                    result[key] = [str(path) for path in paths]
                else:
                    path = _relocated_path(item, root, run_id, directory=key == "run_directory")
                    result[key] = str(path)
                    if key == "raw_path":
                        raw_sources.add(path)
            else:
                result[key] = _relocate(item, root, run_id, raw_sources)
        return result
    if isinstance(value, list):
        return [_relocate(item, root, run_id, raw_sources) for item in value]
    return value


def _verify_sources(root, paths, *, evaluated_at):
    if len(paths) > 2048:
        _fail("SAVED_SOURCE_COUNT_LIMIT")
    total = 0
    verified = {}
    for source in sorted(paths):
        body = _read(source)
        total += len(body)
        if total > _MAX_SOURCE_BYTES:
            _fail("SAVED_SOURCE_TOTAL_SIZE_LIMIT")
        meta_path = _member(root, source.with_suffix(".meta.json").relative_to(root).parts)
        meta = _json(_read(meta_path, maximum=1024 * 1024))
        digest = hashlib.sha256(body).hexdigest()
        if (not isinstance(meta, dict) or meta.get("sha256") != digest
                or meta.get("fixture") is not False or meta.get("status") != 200):
            _fail("SAVED_SOURCE_RECEIPT_MISMATCH")
        if parse_time(meta.get("retrieved_at")) > evaluated_at:
            _fail("SAVED_SOURCE_POSTDATES_CAPTURE")
        verified[str(source)] = {"body": body, "sha256": digest, "meta": meta}
    return verified


def _verify_price_normalization(inputs, verified):
    """Bind saved normalized closes to the original adjusted Massive bars."""
    prices = inputs.get("prices_by_symbol") or {}
    sources = inputs.get("price_sources") or {}
    if not isinstance(prices, dict) or not isinstance(sources, dict):
        _fail("INVALID_SAVED_PRICE_MAP")
    for symbol, bars in prices.items():
        if not bars:
            continue
        source = sources.get(symbol)
        if not isinstance(source, dict) or not isinstance(source.get("receipts"), list) or not source["receipts"]:
            _fail("MISSING_SAVED_PRICE_LINEAGE")
        normalized = {}
        for receipt in source["receipts"]:
            if not isinstance(receipt, dict):
                _fail("INVALID_SAVED_PRICE_RECEIPT")
            checked = verified.get(receipt.get("raw_path"))
            if checked is None or receipt.get("source_response_sha256") != checked["sha256"]:
                _fail("SAVED_PRICE_RECEIPT_MISMATCH")
            if receipt.get("retrieved_at") != checked["meta"].get("retrieved_at"):
                _fail("SAVED_PRICE_RECEIPT_TIME_MISMATCH")
            payload = _json(checked["body"])
            if (not isinstance(payload, dict) or payload.get("ticker") != symbol
                    or payload.get("adjusted") is not True or payload.get("status", "OK") != "OK"
                    or not isinstance(payload.get("results"), list)):
                _fail("SAVED_PRICE_SOURCE_IDENTITY_MISMATCH")
            for row in payload["results"]:
                if not isinstance(row, dict) or isinstance(row.get("t"), bool) or isinstance(row.get("c"), bool):
                    _fail("INVALID_SAVED_PRICE_BAR")
                day = datetime.fromtimestamp(float(row["t"]) / 1000, tz=timezone.utc).date().isoformat()
                close = float(row["c"])
                if not math.isfinite(close) or close <= 0:
                    _fail("INVALID_SAVED_PRICE_BAR")
                if day in normalized and normalized[day]["close"] != close:
                    _fail("CONFLICTING_SAVED_PRICE_BARS")
                normalized[day] = {"date": day, "close": close}
        if bars != [normalized[day] for day in sorted(normalized)]:
            _fail("SAVED_NORMALIZED_PRICES_CHANGED")


def _market_objects(rows):
    if not isinstance(rows, dict):
        _fail("INVALID_SAVED_MARKET_MAP")
    result = {}
    for key, row in rows.items():
        if not isinstance(row, dict):
            _fail("INVALID_SAVED_MARKET")
        market = Market(**row)
        if key != market.ticker:
            _fail("SAVED_MARKET_IDENTITY_MISMATCH")
        try:
            _receipt(market)
        except AgenticEvidenceError:
            _fail("SAVED_MARKET_SOURCE_MISMATCH")
        result[key] = market
    return result


def upgrade_saved_assessment(expected_report, run_directory):
    """Return a copied report plus ``offline_replay``; never modify source files.

    The caller supplies the exact folder it already chose, not a path taken
    from report contents. Missing or inconsistent evidence keeps original
    values and marks the replay unavailable. The caller must still apply its
    usual public-export filter before transmitting the returned report.
    """
    original = deepcopy(expected_report)
    provenance = {
        "schema": "castle.saved_assessment_replay.v1", "engine_revision": "R31",
        "status": "ORIGINAL_RESULTS_RETAINED", "new_provider_calls": 0, "new_model_calls": 0,
        "original_capture_time_unchanged": True, "source_files_modified": False,
        "quote_scope": "AT_ORIGINAL_CAPTURE_INSTANT_NOT_CURRENT_QUOTES",
        "original_acceptance_was_not_upgraded": True,
    }
    if not isinstance(original, dict):
        raise ValueError("A saved assessment object is required")
    identity_reconciliation = {}
    try:
        root = Path(run_directory)
        if root.is_symlink() or not root.is_dir() or not _RUN.fullmatch(root.name):
            _fail("INVALID_OR_SYMLINK_SAVED_RUN_FOLDER")
        root = root.resolve(strict=True)
        run_id = root.name
        report_path = _member(root, ["derived_assessment.json"])
        report_bytes = _read(report_path)
        captured = _json(report_bytes)
        if captured != expected_report:
            _fail("EXPECTED_REPORT_DOES_NOT_MATCH_SAVED_REPORT")
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        digest_path = root / "derived_assessment.sha256"
        provenance["original_derived_report_sha256"] = report_hash
        provenance["original_digest_present"] = digest_path.exists() or digest_path.is_symlink()
        if provenance["original_digest_present"]:
            digest = _read(_member(root, ["derived_assessment.sha256"]), maximum=256).decode("ascii").strip()
            if digest != report_hash + "  derived_assessment.json":
                _fail("ORIGINAL_DERIVED_REPORT_SHA256_MISMATCH")
        if not (root / "private_replay_inputs.json").exists():
            _fail("PRIVATE_INPUTS_NOT_SAVED")
        private_bytes = _read(_member(root, ["private_replay_inputs.json"]))
        private = _json(private_bytes)
        if (original.get("schema") != "castle.option1.live_assessment.v2"
                or not isinstance(private, dict) or not isinstance(private.get("inputs"), dict)
                or not isinstance(original.get("empirical"), dict)
                or not isinstance(original.get("default_analysis"), dict)
                or not isinstance(original["default_analysis"].get("events"), list)
                or any(not isinstance(event, dict) for event in original["default_analysis"]["events"])):
            _fail("UNSUPPORTED_SAVED_REPLAY_INPUTS")
        evaluated = parse_time(original["evaluated_at_utc"])
        portfolio = original["portfolio"]
        fingerprint = portfolio_input_fingerprint(portfolio["positions"], portfolio["notional"],
                                                  private["inputs"].get("holdings"))
        if not _HASH.fullmatch(str(original.get("input_fingerprint", ""))) or fingerprint != original["input_fingerprint"]:
            _fail("ORIGINAL_INPUT_FINGERPRINT_MISMATCH")
        input_time = private["inputs"].get("evaluated_at_utc")
        if input_time is not None and parse_time(input_time) > evaluated:
            _fail("SAVED_INPUTS_POSTDATE_CAPTURE")
        raw_sources = set()
        # Discovery messages are not replayed or reinterpreted. Only documented
        # price/holdings pointers and original normalized market rows are needed.
        inputs = _relocate(private["inputs"], root, run_id, raw_sources)
        market_rows = _relocate(private.get("market_objects"), root, run_id, raw_sources)
        payroll_rows = _relocate(private.get("payroll_markets"), root, run_id, raw_sources)
        verified = _verify_sources(root, raw_sources, evaluated_at=evaluated)
        _verify_price_normalization(inputs, verified)
        markets = _market_objects(market_rows)
        payrolls = _market_objects(payroll_rows) if payroll_rows is not None else None
        identity = resolve_assessment_identity(original, inputs=inputs, evaluated_at=evaluated)
        resolved = parse_portfolio(identity["portfolio_rows"], notional=portfolio["notional"])
        # The original source report/fingerprint is checked before reconciling
        # its declaration. Changes below exist only in this returned copy; raw
        # observations, empirical studies, quotes and capture times stay put.
        original["instrument_identity"] = deepcopy(identity["audit"])
        if resolved != parse_portfolio(portfolio["positions"], notional=portfolio["notional"]):
            original["original_portfolio_declaration"] = deepcopy(portfolio)
            original["portfolio"] = resolved.to_dict()
            rebuilt_exposures = build_instrument_exposures(identity["portfolio_rows"],
                notional=resolved.notional, holdings=inputs.get("holdings"),
                instrument_metadata=inputs.get("instrument_metadata"), evaluated_at=evaluated)
            original["exposures"] = retain_reported_instrument_restrictions(rebuilt_exposures,
                original.get("exposures"), verified_identity_symbols=identity["verified_supported_symbols"])
            original["input_fingerprint"] = portfolio_input_fingerprint(
                resolved.to_dict()["positions"], resolved.notional, inputs.get("holdings"))
            identity_reconciliation.update(original_input_fingerprint=fingerprint,
                resolved_input_fingerprint=original["input_fingerprint"],
                portfolio_declaration_reconciled_from_saved_source=True)
        rebuilt = build_default_assessment(original, market_objects=markets, inputs=inputs,
                                           evaluated_at=evaluated, payroll_markets=payrolls)
        previous = original["default_analysis"]
        retained = [deepcopy(event) for event in previous.get("events", []) if event.get("family") not in _MACRO]
        changed_symbols = sorted({row["symbol"] for row in identity["audit"].get("corrections", [])
                                  if row.get("weight")})
        hard_blocked = set((rebuilt.get("instrument_identity_gate") or {}).get("identity_blocked_symbols") or [])
        retained_blocked = sorted(set(changed_symbols) | hard_blocked)
        if retained_blocked:
            for event in retained:
                # Preserve the original investigation and source quotes, not
                # an implication that its old instrument-dependent numerical
                # projection was revalidated after the declaration changed.
                event["instrument_identity_gate"] = {
                    "status": "BLOCKED_INSTRUMENT_IDENTITY", "blocked_symbols": retained_blocked,
                    "identity_blocked_symbols": sorted(hard_blocked),
                    "asset_class_blocked_symbols": sorted(set(changed_symbols) - hard_blocked),
                    "reason": "The original non-macro projection has not been revalidated for corrected or unresolved instrument identity for "
                              + ", ".join(retained_blocked) + ". Original source evidence remains available.",
                    "scope": "PRESERVED_ORIGINAL_CARD_NOT_REVALIDATED_FOR_CORRECTED_IDENTITY",
                }
                event["legacy_fallback_allowed"] = False
                event["status_before_instrument_identity_gate"] = event.get("status")
                event["status"] = "BLOCKED_INSTRUMENT_IDENTITY"
                if isinstance(event.get("outcome_table"), dict):
                    event["identity_blocked_model_diagnostics"] = deepcopy(event["outcome_table"])
                    event["outcome_table"] = {"status": "BLOCKED_INSTRUMENT_IDENTITY",
                        "instrument_identity_gate": deepcopy(event["instrument_identity_gate"]),
                        "legacy_fallback_allowed": False, "outcomes": [],
                        "summary": {"q_standard_deviation_usd": None}}
                event["ranking"] = {**(event.get("ranking") or {}), "abs_delta_pnl_usd": None,
                                    "abs_delta_pnl_interval_usd": None, "selected_outcome_id": None}
                event["risk_ranking"] = {**(event.get("risk_ranking") or {}), "q_standard_deviation_usd": None}
        rebuilt["events"] = [event for event in rebuilt["events"] if event.get("family") in _MACRO] + retained
        rebuilt["events"].sort(key=lambda event: (
            (event.get("risk_ranking") or {}).get("q_standard_deviation_usd") is None,
            -((event.get("risk_ranking") or {}).get("q_standard_deviation_usd") or 0.0),
            (event.get("ranking") or {}).get("abs_delta_pnl_usd") is None,
            -((event.get("ranking") or {}).get("abs_delta_pnl_usd") or 0.0),
            event.get("event_date") or "9999-12-31", event.get("event_id", "")))
        for index, event in enumerate(rebuilt["events"], 1):
            event.setdefault("ranking", {})["display_order"] = index
        original["default_analysis"] = rebuilt
        provenance.update(
            status="OFFLINE_SAVED_INPUTS_RECOMPUTED_R30",
            capture_evaluated_at_utc=original["evaluated_at_utc"],
            private_inputs_sha256=hashlib.sha256(private_bytes).hexdigest(),
            input_fingerprint_verified=True, local_source_receipts_verified=len(verified),
            source_integrity_scope="LOCAL_BYTES_RECEIPT_AND_NORMALIZATION_CONSISTENCY_NOT_PROVIDER_SIGNATURE",
            original_empirical_reports_preserved=True, non_macro_source_cards_preserved=True,
            original_event_count=len(previous.get("events", [])), updated_event_count=len(rebuilt["events"]),
            instrument_identity=deepcopy(identity["audit"]),
            instrument_identity_gate=deepcopy(rebuilt.get("instrument_identity_gate")),
            **identity_reconciliation,
        )
    except _ReplayIssue as exc:
        provenance["reason_code"] = str(exc)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        # Keep the prior report, not a partly built projection. Raw exceptions
        # can contain a provider body or host path and are not browser-safe.
        original = deepcopy(expected_report)
        provenance["reason_code"] = "SAVED_INPUTS_COULD_NOT_BE_SAFELY_RECOMPUTED"
    original["offline_replay"] = provenance
    return original
