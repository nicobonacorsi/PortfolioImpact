"""Class-based research inputs over the unchanged portfolio accounting engine.

This module does not acquire data. Provider metadata, accepted holdings and user
declarations have separate evidence labels. A symbol or fund name never supplies
an asset class. Research channels are hypotheses, not estimated event returns.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from .agent_budgeted_discovery import concise_exposures
from .adapters import NAV_WEIGHT_SUM_TOLERANCE, _normalize_constituent, sanitize_url
from .nonmacro_primary_desk_case import ISSUER_URL
from .portfolio import _validate_snapshot, parse_portfolio, recover_exposures
from .issuer_overlap import audit_issuer_overlap
from .source_class_quality import review_source_class_snapshot


INSTRUMENT_CLASSES = (
    "stock", "equity_etf", "bond_etf", "commodity_etf", "cash",
    "unresolved_etf", "unsupported",
)
_FUND_CLASSES = {
    "equity": "equity_etf", "equities": "equity_etf",
    "fixed_income": "bond_etf", "fixed income": "bond_etf",
    "bond": "bond_etf", "bonds": "bond_etf",
    "commodity": "commodity_etf", "commodities": "commodity_etf",
}
_MACRO_CHANNELS = {
    "stock": {
        "FED_POLICY": "Discount rates, financing costs and earnings expectations.",
        "US_INFLATION": "Discount rates, input costs and pricing power.",
    },
    "equity_etf": {
        "FED_POLICY": "The original equity fund has discount-rate and earnings channels.",
        "US_INFLATION": "The original equity fund has inflation, margin and valuation channels.",
    },
    "bond_etf": {
        "FED_POLICY": "Policy surprises may change yields and the original bond fund price.",
        "US_INFLATION": "Inflation surprises may change nominal yields, real yields and credit spreads.",
    },
    "commodity_etf": {
        "FED_POLICY": "Real rates, dollar prices and financing costs may affect commodity exposure.",
        "US_INFLATION": "Inflation news may affect commodity prices; direction depends on the underlying asset.",
    },
}
_OBSERVED_ASSET_CLASSES = {
    "equity": "equity", "equities": "equity",
    "fixed income": "fixed_income", "fixed_income": "fixed_income", "debt": "fixed_income",
    "commodity": "commodity", "commodities": "commodity", "cash": "cash",
}


def _holdings_composition(snapshot, symbol, now, max_age):
    """Reverify constituent asset-class fields without relaxing the NAV ledger.

    Broad provider asset_class is useful research evidence even when a bond,
    depositary receipt or identifier-only holding is outside the plain-stock
    accounting model. It is not whole-fund taxonomy or security-master identity.
    """
    unavailable = {
        "status": "SOURCE_VERIFIED_COMPOSITION_UNAVAILABLE",
        "reason": "No accepted source-bound constituent composition is available.",
        "fund_symbol": symbol, "eligible_for_research": False,
        "whole_fund_taxonomy_inferred": False, "ledger_modified": False,
    }
    if not isinstance(snapshot, Mapping):
        return unavailable
    try:
        metadata = snapshot.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Constituent metadata must be a record.")
        # The signed fixed-income source view has a different evidence domain
        # from equity issuer look-through. Recheck the original source-class
        # gate, then retain all chronology and page-chain checks below. Never
        # rewrite the snapshot's reliable flag or the authoritative NAV ledger.
        signed_review = (review_source_class_snapshot(snapshot, symbol=symbol)
                         if snapshot.get("reliable") is not True else None)
        signed_attribution = bool(signed_review and signed_review.get(
            "eligible_for_source_class_attribution") is True)
        equity_class_only = bool(signed_review and signed_review.get(
            "eligible_for_equity_class_composition") is True)
        signed_accepted = signed_attribution or equity_class_only
        if signed_accepted:
            source = urlsplit(str(snapshot.get("source_url", "")))
            retrieved = evaluation_instant(snapshot["retrieved_at"])
            effective = datetime.strptime(snapshot["as_of"], "%Y-%m-%d").date()
            processed = datetime.strptime(snapshot["processed_date"], "%Y-%m-%d").date()
            if (source.scheme != "https" or source.hostname != "api.massive.com"
                    or source.path != "/etf-global/v1/constituents"
                    or source.username or source.password or source.query or source.fragment
                    or source.port not in (None, 443) or retrieved > now
                    or not effective <= processed <= retrieved.date()
                    or effective > now.date() or (now.date() - effective).days > max_age):
                raise ValueError("Signed composition identity or evaluation chronology is invalid.")
        else:
            _validate_snapshot(snapshot, symbol, now, max_age)
        pages = metadata.get("pages")
        if (not isinstance(pages, list) or not 1 <= len(pages) <= 20
                or metadata.get("quality_issues") and not signed_accepted
                or metadata.get("selection_provisional") is True):
            raise ValueError("Source pages or reliable selection evidence are missing.")
        cutoff = datetime.strptime(metadata["selection_cutoff_date"], "%Y-%m-%d").date()
        if cutoff > now.date():
            raise ValueError("Source selection cutoff is in the future.")
        rows, references, seen_urls = [], [], set()
        expected_url = None
        start = end = requested_processed = None
        for index, page in enumerate(pages):
            path = Path(page["raw_path"])
            body = path.read_bytes()
            receipt = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            payload = json.loads(body)
            digest = sha256(body).hexdigest()
            retrieved = evaluation_instant(receipt["retrieved_at"])
            url = receipt["url"]
            parts = urlsplit(url)
            if (parts.scheme != "https" or parts.hostname != "api.massive.com"
                    or parts.path != "/etf-global/v1/constituents" or parts.username or parts.password
                    or parts.port not in (None, 443) or parts.fragment
                    or not _allowed_query(parts.query, {"composite_ticker", "effective_date", "effective_date.gte",
                        "effective_date.lte", "processed_date", "processed_date.gte", "processed_date.lte", "limit", "cursor"})
                    or url in seen_urls or retrieved > now or cutoff > retrieved.date()
                    or receipt.get("status") != 200 or receipt.get("fixture") is not False
                    or page.get("fixture") is not False or page.get("retrieved_at") != receipt["retrieved_at"]
                    or receipt.get("sha256") != digest or page.get("sha256") != digest
                    or not isinstance(payload, dict) or payload.get("status", "OK") != "OK"):
                raise ValueError("Constituent source receipt, identity or hash verification failed.")
            seen_urls.add(url)
            query = dict(parse_qsl(parts.query))
            if index == 0:
                if query.get("composite_ticker") != symbol:
                    raise ValueError("Constituent request identifies another fund.")
                start = datetime.strptime(query.get("effective_date.gte", query.get("effective_date", "")), "%Y-%m-%d").date()
                end = datetime.strptime(query.get("effective_date.lte", query.get("effective_date", "")), "%Y-%m-%d").date()
                requested_processed = query.get("processed_date")
                if not start <= end <= cutoff:
                    raise ValueError("Constituent request has invalid date bounds.")
            elif expected_url is None or url != expected_url:
                raise ValueError("Constituent source pages do not form the recorded page chain.")
            page_rows = payload.get("results")
            if not isinstance(page_rows, list) or not all(isinstance(row, dict) for row in page_rows):
                raise ValueError("Constituent source result shape is invalid.")
            rows.extend(page_rows)
            next_url = payload.get("next_url")
            if next_url is not None and not isinstance(next_url, str):
                raise ValueError("Constituent pagination link is invalid.")
            expected_url = sanitize_url(next_url) if next_url else None
            references.append({"source_url": url, "source_response_sha256": digest,
                               "retrieved_at": receipt["retrieved_at"]})
        if expected_url is not None or snapshot.get("retrieved_at") != references[-1]["retrieved_at"]:
            raise ValueError("Constituent pagination is incomplete or retrieval identity differs.")
        batches = {}
        for row in rows:
            effective = datetime.strptime(row["effective_date"], "%Y-%m-%d").date()
            processed = datetime.strptime(row["processed_date"], "%Y-%m-%d").date()
            if row.get("composite_ticker") != symbol or not start <= effective <= end or processed < effective:
                raise ValueError("Constituent row is outside its source request.")
            if requested_processed is not None and row["processed_date"] != requested_processed:
                raise ValueError("Constituent source did not honor its exact processing-date filter.")
            if processed <= cutoff:
                batches.setdefault((effective.isoformat(), processed.isoformat()), []).append(row)
        batch = max(batches) if batches else None
        if (batch is None or list(batch) != metadata.get("selected_batch")
                or tuple(batch) != (snapshot.get("as_of"), snapshot.get("processed_date"))):
            raise ValueError("Constituent selection does not match the latest complete source batch.")
        selected, seen_rows, identities, ranks = [], set(), set(), set()
        for row in batches[batch]:
            fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen_rows:
                continue
            seen_rows.add(fingerprint)
            normalized = _normalize_constituent(row, symbol)
            keys = [("ticker", normalized["source_ticker"])] if normalized["source_ticker"] else []
            keys += [(kind, value) for kind, value in normalized["identifiers"].items() if value]
            rank = row.get("constituent_rank")
            if ((not signed_accepted and
                    (normalized["quality_issues"] or any(key in identities for key in keys)))
                    or rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 or rank in ranks)):
                raise ValueError("Constituent quality or duplicate-identity conflict prevents reliable composition.")
            identities.update(keys)
            if rank is not None:
                ranks.add(rank)
            selected.append(normalized)
        canonical = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
        if Counter(map(canonical, selected)) != Counter(map(canonical, snapshot["holdings"])):
            raise ValueError("Normalized constituent fields differ from the original response.")
        by_class = {name: [] for name in ("equity", "fixed_income", "commodity", "cash", "unclassified")}
        missing = 0
        for row in selected:
            weight = row["weight"]
            if weight is None:
                missing += 1
                continue
            if signed_accepted and weight > 1:
                raise ValueError("A single source position above fund NAV is outside signed research support.")
            category = _OBSERVED_ASSET_CLASSES.get(str(row.get("asset_class") or "").strip().lower(), "unclassified")
            by_class[category].append(weight)
        weights = {category: math.fsum(values) for category, values in by_class.items()}
        reported = math.fsum(weights.values())
        # Issuer files can list slightly more than NAV (SCHD on 2026-10-29: 100.08%);
        # the same 1% tolerance as the holdings ledger applies, without rescaling.
        if not signed_accepted and reported > 1 + NAV_WEIGHT_SUM_TOLERANCE:
            raise ValueError("Reported composition exceeds the original NAV basis.")
        majority = [category for category, weight in weights.items() if category != "unclassified" and weight > .5]
        return {
            "status": (signed_review["status"] if signed_accepted
                       else "LOCAL_SOURCE_WEIGHTED_COMPOSITION_VERIFIED"), "fund_symbol": symbol,
            "eligible_for_research": True, "effective_date": batch[0], "processed_date": batch[1],
            "source_references": references, "reported_class_nav_weights": weights,
            "reported_nav_weight": reported,
            "unreported_nav_weight": 1. - reported if signed_accepted else max(0., 1. - reported),
            "recorded_nav_residual": 1. - reported,
            "signed_source_class_attribution": signed_attribution,
            "equity_class_aggregate_only": equity_class_only,
            "equity_issuer_lookthrough_upgraded": False,
            "original_quality_issues_retained": list(metadata.get("quality_issues") or []),
            "source_class_review": signed_review,
            "missing_weight_row_count": missing, "selected_constituent_count": len(selected),
            "predominant_observed_class": majority[0] if majority else None,
            "predominant_label_rule": "A structured source class accounts for strictly more than half of original fund NAV; diagnostic only.",
            "whole_fund_taxonomy_inferred": False, "ledger_modified": False,
            "scope": "Structured constituent asset_class at the selected date, without renormalization; not full-fund taxonomy, plain-security identity, mandate, beta or causal sensitivity.",
            "source_assurance": "Local response bytes and fields verified; not a provider signature or historical publication-time verification.",
        }
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {**unavailable, "reason": "Holdings quality, date, complete page chain, source receipt or normalized-field verification failed."}


def evaluation_instant(value: datetime | str) -> datetime:
    """Require an explicit UTC-convertible clock; never fall back to today."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Evaluation time must be a timezone-aware ISO timestamp.") from None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Evaluation time must be an explicit timezone-aware timestamp.")
    return value.astimezone(timezone.utc)


def _allowed_query(query, allowed):
    """Retain credential-free documented request parameters for exact binding."""
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        return False
    names = [key for key, _ in pairs]
    return (len(set(names)) == len(names) and all(key in allowed and value for key, value in pairs))


def _metadata_fields(record: Mapping[str, Any], symbol: str, now: datetime) -> dict[str, Any]:
    """Validate a typed adapter record and, when available, its original bytes.

    A hash without the original response is retained as reported provenance, not
    independently verified metadata. A boolean named ``verified`` has no effect.
    Raw local data and credential-bearing URLs are never copied to the report.
    """
    invalid = {"status": "METADATA_REJECTED", "reason": "Metadata identity or provenance is invalid."}
    # The normalized record cannot erase or contradict an explicit fixture
    # declaration merely because the separately stored receipt says otherwise.
    # Older adapter records without this optional field remain compatible.
    if record.get("fixture") is True:
        return {**invalid, "reason": "Provider metadata is explicitly fixture-labelled."}
    record_symbol = record.get("symbol", record.get("ticker"))
    if not isinstance(record_symbol, str) or record_symbol.upper() != symbol:
        return invalid
    kind = record.get("instrument_type", record.get("type"))
    if not isinstance(kind, str) or not kind.strip():
        return {**invalid, "reason": "Provider metadata does not contain a typed instrument kind."}
    url = record.get("source_url")
    try:
        parts = urlsplit(url) if isinstance(url, str) else None
        if (parts is None or parts.scheme != "https" or parts.hostname != "api.massive.com"
                or parts.path != "/v3/reference/tickers/" + quote(symbol, safe="")
                or parts.username or parts.password or parts.fragment
                or not _allowed_query(parts.query, {"date"})
                or parts.port not in (None, 443)):
            return invalid
        retrieved = evaluation_instant(record.get("retrieved_at"))
    except (ValueError, TypeError):
        return invalid
    if retrieved > now:
        return {**invalid, "reason": "Provider metadata was retrieved after the evaluation time."}
    digest = record.get("source_response_sha256", record.get("response_sha256"))
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return {**invalid, "reason": "Provider metadata has no valid response SHA-256."}
    asset_class = record.get("asset_class")
    if asset_class is not None and not isinstance(asset_class, str):
        return {**invalid, "reason": "Provider asset class must be a structured string field."}
    result = {
        "status": "ADAPTER_REPORTED_METADATA_NOT_REVERIFIED",
        "reason": "Typed adapter metadata retained; original response bytes were not reverified here.",
        "provider": str(record.get("provider", "unspecified"))[:80],
        "source_url": url, "retrieved_at": retrieved.isoformat(),
        "source_response_sha256": digest, "instrument_type": kind.strip().upper(),
        "asset_class": asset_class, "currency_name": record.get("currency_name"),
        "active": record.get("active"), "local_response_verified": False,
    }
    path = record.get("raw_path")
    if not path:
        return result
    try:
        source = Path(path)
        body = source.read_bytes()
        receipt = json.loads(source.with_suffix(".meta.json").read_text(encoding="utf-8"))
        payload = json.loads(body)
    except (OSError, ValueError, TypeError):
        return {**invalid, "reason": "Provider metadata response or local receipt is unreadable."}
    if (sha256(body).hexdigest() != digest or receipt.get("sha256") != digest
            or receipt.get("status") != 200 or receipt.get("fixture") is not False
            or receipt.get("retrieved_at") != record.get("retrieved_at")
            or receipt.get("url") != url):
        return {**invalid, "reason": "Provider metadata response and local receipt disagree or are fixture-labelled."}
    if not isinstance(payload, dict) or payload.get("status", "OK") != "OK":
        return {**invalid, "reason": "Provider metadata response reports an unsuccessful status."}
    rows = payload.get("results") if isinstance(payload, dict) else None
    rows = [rows] if isinstance(rows, dict) else rows
    if not isinstance(rows, list):
        return {**invalid, "reason": "Provider metadata response has no supported results object."}
    matches = [row for row in rows if isinstance(row, dict)
               and str(row.get("ticker", row.get("composite_ticker", ""))).upper() == symbol]
    if len(matches) != 1:
        return {**invalid, "reason": "Provider metadata does not uniquely identify the supplied symbol."}
    raw = matches[0]
    raw_kind = raw.get("type", raw.get("instrument_type"))
    if (not isinstance(raw_kind, str) or raw_kind.strip().upper() != result["instrument_type"]
            or any(key in record and raw.get(key) != record[key]
                   for key in ("asset_class", "currency_name", "active"))):
        return {**invalid, "reason": "Normalized metadata fields differ from the original provider response."}
    result.update(
        status="LOCAL_RESPONSE_BYTES_AND_FIELDS_VERIFIED",
        reason="Local response hash, receipt, symbol and typed fields agree; this is not a provider signature.",
        local_response_verified=True,
        asset_class=raw.get("asset_class"), currency_name=raw.get("currency_name"), active=raw.get("active"),
    )
    return result


def _taxonomy_fields(record, symbol, now):
    """Keep ETF taxonomy evidence separate from ticker identity evidence."""
    invalid = {"status": "TAXONOMY_REJECTED", "reason": "ETF taxonomy identity, date or provenance is invalid."}
    if not isinstance(record, Mapping) or record.get("composite_ticker") != symbol:
        return invalid
    asset_class = record.get("asset_class")
    if not isinstance(asset_class, str) or not asset_class.strip():
        return invalid
    try:
        retrieved = evaluation_instant(record.get("retrieved_at"))
        effective = datetime.strptime(record["effective_date"], "%Y-%m-%d").date()
        processed = datetime.strptime(record["processed_date"], "%Y-%m-%d").date()
        parts = urlsplit(record["source_url"])
        if (parts.scheme != "https" or parts.hostname != "api.massive.com"
                or parts.path != "/etf-global/v1/taxonomies" or parts.username or parts.password
                or parts.fragment or parts.port not in (None, 443)
                or not _allowed_query(parts.query, {"composite_ticker", "effective_date", "effective_date.gte",
                    "effective_date.lte", "processed_date", "processed_date.gte", "processed_date.lte",
                    "limit", "cursor", "sort", "order"})
                or not effective <= processed <= retrieved.date() <= now.date() or retrieved > now):
            return invalid
    except (ValueError, KeyError, TypeError):
        return invalid
    digest = record.get("source_response_sha256", record.get("response_sha256"))
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return invalid
    result = {
        "status": "ADAPTER_REPORTED_TAXONOMY_NOT_REVERIFIED",
        "reason": "Structured taxonomy retained; original response bytes were not reverified here.",
        "asset_class": asset_class, "composite_ticker": symbol,
        "effective_date": effective.isoformat(), "processed_date": processed.isoformat(),
        "retrieved_at": retrieved.isoformat(), "source_url": record["source_url"],
        "source_response_sha256": digest, "local_response_verified": False,
        "leverage_style": record.get("leverage_style"), "levered_amount": record.get("levered_amount"),
    }
    if not record.get("raw_path"):
        return result
    try:
        path = Path(record["raw_path"])
        body = path.read_bytes()
        receipt = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        payload = json.loads(body)
    except (OSError, ValueError, TypeError):
        return {**invalid, "reason": "ETF taxonomy response or local receipt is unreadable."}
    if (sha256(body).hexdigest() != digest or receipt.get("sha256") != digest
            or receipt.get("status") != 200 or receipt.get("fixture") is not False
            or receipt.get("url") != record["source_url"]
            or receipt.get("retrieved_at") != record.get("retrieved_at")):
        return {**invalid, "reason": "ETF taxonomy receipt and response disagree or are fixture-labelled."}
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {**invalid, "reason": "ETF taxonomy response has no results list."}
    matching = [row for row in rows if isinstance(row, dict)
                and all(row.get(k) == record[k] for k in ("composite_ticker", "effective_date", "processed_date"))]
    if len(matching) != 1 or any(matching[0].get(k) != record.get(k)
                                 for k in ("asset_class", "leverage_style", "levered_amount")):
        return {**invalid, "reason": "Selected ETF taxonomy fields do not uniquely match the original response."}
    result.update(status="LOCAL_TAXONOMY_BYTES_AND_FIELDS_VERIFIED", local_response_verified=True,
                  reason="Selected taxonomy fields, date pair and local receipt agree; no provider signature or historical publication time is implied.")
    return result


def _leveraged_or_inverse(taxonomy):
    style = str(taxonomy.get("leverage_style") or "").strip().lower()
    if "inverse" in style or "short" in style or style in {"leveraged", "levered"}:
        return True
    amount = taxonomy.get("levered_amount")
    if amount is not None:
        try:
            value = float(amount)
            if not math.isfinite(value) or value < 0 or value > 1:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _classify(position, entries, metadata, now):
    declared = position.asset_type
    result = {
        "symbol": position.symbol, "weight": position.weight, "asset_type": declared,
        "instrument_class": "cash" if declared == "cash" else "stock" if declared == "stock" else "unresolved_etf",
        "classification_status": "CASH_DECLARATION" if declared == "cash" else "USER_DECLARED_STOCK_UNVERIFIED" if declared == "stock" else "ETF_ASSET_CLASS_UNRESOLVED",
        "classification_basis": "Explicit portfolio declaration; no class inferred from symbol or name.",
        "instrument_identity_verified": False, "metadata_provenance": None,
        "taxonomy_provenance": None, "provider_instrument_type": None,
        "unresolved_holdings_weight": math.fsum(e.portfolio_weight for e in entries if e.status == "unresolved_etf"),
    }
    if declared == "cash":
        return result
    typed = _metadata_fields(metadata, position.symbol, now) if isinstance(metadata, Mapping) else None
    result["metadata_provenance"] = typed
    taxonomy = (_taxonomy_fields(metadata["taxonomy"], position.symbol, now)
                if isinstance(metadata, Mapping) and "taxonomy" in metadata else None)
    result["taxonomy_provenance"] = taxonomy
    accepted = typed is not None and typed["status"] != "METADATA_REJECTED"
    if accepted:
        kind = typed["instrument_type"]
        result["provider_instrument_type"] = kind
        result["instrument_identity_verified"] = typed["local_response_verified"]
        if kind == "ETV":
            result["provider_type_description"] = "Source-reported exchange-traded vehicle; underlying asset class is not established by this type."
        currency = typed.get("currency_name")
        bad_currency = currency is not None and str(currency).lower() not in {"usd", "us dollar", "united states dollar"}
        if typed.get("active") is False or bad_currency:
            result.update(instrument_class="unsupported", classification_status="UNSUPPORTED_PROVIDER_INSTRUMENT",
                          classification_basis="Inactive or non-USD metadata is outside the supported valuation scope.")
            return result
        expected = {"CS", "ADRC"} if declared == "stock" else {"ETF"}
        if kind not in expected:
            result.update(instrument_class="unsupported", classification_status="DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED",
                          classification_basis="Provider instrument type conflicts with the declaration or is outside common-stock/ETF support.")
            return result
        if declared == "stock":
            result.update(classification_status="PROVIDER_TYPED_COMMON_STOCK" if typed["local_response_verified"] else "ADAPTER_REPORTED_COMMON_STOCK_UNVERIFIED",
                          classification_basis=f"Structured provider instrument type {kind}; metadata evidence status is shown separately.")
            return result
        asset_source = taxonomy if taxonomy and taxonomy["status"] != "TAXONOMY_REJECTED" else typed
        if _leveraged_or_inverse(asset_source):
            result.update(instrument_class="unsupported", classification_status="LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED",
                          classification_basis="Source taxonomy indicates leverage or an inverse mandate; simple class channels do not model it.")
            return result
        provider_class = (asset_source.get("asset_class") or "").strip().lower()
        if provider_class in _FUND_CLASSES:
            result.update(instrument_class=_FUND_CLASSES[provider_class],
                          classification_status="PROVIDER_TYPED_ETF_ASSET_CLASS" if asset_source["local_response_verified"] else "ADAPTER_REPORTED_ETF_CLASS_UNVERIFIED",
                          classification_basis="Structured provider asset_class; ticker and fund name are not classification evidence.")
            return result
    # Composition can support an equity research channel, never a mandate, beta
    # or provider-authorship claim. Every NAV slice must be accounted for.
    positive = [entry for entry in entries if entry.portfolio_weight > 0]
    # Holdings of 99.5%+ of NAV in stock/cash support an equity channel; the
    # small remainder (e.g. a CVR) stays listed as unresolved in the ledger.
    unresolved_mass = math.fsum(e.portfolio_weight for e in positive
                                if not (e.status == "look_through" and e.asset_type in {"stock", "cash"}))
    if (declared == "etf" and positive
            and unresolved_mass <= 0.005 * position.weight
            and any(entry.asset_type == "stock" for entry in positive)
            and math.isclose(math.fsum(e.portfolio_weight for e in positive), position.weight, rel_tol=0, abs_tol=1e-9)):
        result.update(instrument_class="equity_etf", classification_status="ACCEPTED_HOLDINGS_EQUITY_COMPOSITION",
                      classification_basis="Accepted complete stock/cash holdings support an equity channel at their effective date; fund mandate and provider authorship are not inferred.")
    return result


def _channels(entries, positions, tsmc_source_ledger_indices=frozenset()):
    by_family = {"FED_POLICY": [], "US_INFLATION": [], "TAIWAN_SUPPLY": []}
    for index, entry in enumerate(entries):
        parent = positions[entry.original_symbol]
        if entry.portfolio_weight <= 0 or entry.asset_type == "cash" and entry.status != "unresolved_etf":
            continue
        base = {
            "ledger_index": index, "instrument_symbol": entry.symbol,
            "original_symbol": entry.original_symbol, "portfolio_weight": entry.portfolio_weight,
            "identity_status": entry.status, "asset_type": entry.asset_type,
            "instrument_class": parent["instrument_class"],
            "classification_status": parent["classification_status"],
            "holdings_as_of": entry.as_of, "holdings_source_url": entry.source_url,
            "estimated_return_sensitivity": None,
        }
        research_class = parent["instrument_class"]
        if research_class == "unresolved_etf" and entry.status == "look_through" and entry.asset_type == "stock":
            research_class = "stock"
        for family in ("FED_POLICY", "US_INFLATION"):
            if research_class in _MACRO_CHANNELS:
                reason = _MACRO_CHANNELS[research_class][family]
                status = "CLASS_BASED_RESEARCH_CHANNEL_NOT_MEASURED_SENSITIVITY"
            else:
                reason = "Instrument identity or underlying asset class is unresolved; investigate exposure before estimating sensitivity."
                status = "UNCLASSIFIED_INSTRUMENT_INVESTIGATION_REQUIRED"
            by_family[family].append({**base, "channel_status": status, "reason": reason})
        if (entry.symbol == "NVDA" and entry.asset_type == "stock" and entry.status != "unresolved_etf"
                and parent["instrument_class"] != "unsupported"):
            by_family["TAIWAN_SUPPLY"].append({
                **base, "channel_status": "DOCUMENTARY_ISSUER_LINK_NOT_ESTIMATED_RETURN",
                "reason": "NVIDIA's reviewed filing identifies TSMC among foundries. This supports a supply-chain hypothesis, not a fitted stock loss.",
                "evidence_reference": ISSUER_URL,
                "source_assurance": "PREVIOUS_PRIMARY_FILING_PAGE_REVIEW_NOT_CURRENT_RAW_PROVIDER_CAPTURE",
            })
        if index in tsmc_source_ledger_indices and entry.asset_type == "stock" and entry.status != "unresolved_etf":
            by_family["TAIWAN_SUPPLY"].append({
                **base, "channel_status": "PRIMARY_ISSUER_LISTING_LINK_NOT_ESTIMATED_RETURN",
                "reason": "TSMC's own issuer FAQ identifies ordinary Taiwan 2330 and US TSM ADR as its listings. Taiwan production/logistics is a potential event channel, not a fitted price sensitivity; local shares and ADR have different security returns.",
                "evidence_reference": "https://investor.tsmc.com/english/faq",
                "source_assurance": "SOURCE_MATCHED_LOCAL_ISIN_OR_PRIMARY_ISSUER_ADR_TICKER_NOT_INDEPENDENT_SECURITY_MASTER",
            })
    return by_family


def build_instrument_exposures(
    portfolio_rows: list[dict[str, Any]], *, notional: float = 100_000.0,
    holdings: Mapping[str, Any] | None = None,
    instrument_metadata: Mapping[str, Any] | None = None,
    evaluated_at: datetime | str, max_holdings_age_days: int = 90,
    issuer_siblings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a general USD-stock/ETF research brief without a ticker registry.

    The unchanged ``recover_exposures`` ledger is authoritative. Classifying an
    original ETF does not resolve its constituents, and original NAV is never
    added to its component NAV. Unsupported instruments remain visible.
    """
    if not isinstance(portfolio_rows, (list, tuple)) or not 1 <= len(portfolio_rows) <= 128:
        raise ValueError("One to 128 explicit portfolio rows are required.")
    if holdings is not None and not isinstance(holdings, Mapping):
        raise ValueError("Holdings must be a keyed snapshot mapping or None.")
    if instrument_metadata is not None and not isinstance(instrument_metadata, Mapping):
        raise ValueError("Instrument metadata must be a keyed mapping or None.")
    if issuer_siblings is not None and not isinstance(issuer_siblings, Mapping):
        raise ValueError("Issuer sibling records must be a keyed mapping or None.")
    now = evaluation_instant(evaluated_at)
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    ledger = recover_exposures(portfolio, holdings or {}, now=now, max_holdings_age_days=max_holdings_age_days)
    metadata = instrument_metadata or {}
    positions = [
        _classify(position, [e for e in ledger.entries if e.original_symbol == position.symbol],
                  metadata.get(position.symbol), now)
        for position in portfolio.positions
    ]
    for position in positions:
        if position["asset_type"] == "etf":
            position["holdings_composition"] = _holdings_composition(
                (holdings or {}).get(position["symbol"]), position["symbol"], now, max_holdings_age_days)
    issuer_overlap = audit_issuer_overlap(portfolio, ledger, holdings, issuer_siblings=issuer_siblings)
    tsmc_indices = frozenset(c["ledger_index"] for c in issuer_overlap["tsmc_source_identified_slices"])
    channels = _channels(ledger.entries, {row["symbol"]: row for row in positions}, tsmc_indices)
    class_weights = {kind: math.fsum(p["weight"] for p in positions if p["instrument_class"] == kind)
                     for kind in INSTRUMENT_CLASSES}
    context = concise_exposures(portfolio, ledger)
    context["source_supported_issuer_overlap"] = issuer_overlap
    selected = {p["symbol"] for p in context["portfolio"]["positions"]}
    context["instrument_classification"] = {
        "positions": [{key: p[key] for key in (
            "symbol", "weight", "instrument_class", "classification_status",
            "classification_basis", "instrument_identity_verified", "unresolved_holdings_weight",
        )} for p in positions if p["symbol"] in selected],
        "all_original_position_class_weights": class_weights,
        "no_classification_from_symbol_or_name": True,
    }
    composition_inputs = []
    observed_weights = {category: 0. for category in ("equity", "fixed_income", "commodity", "cash")}
    unresolved_composition = 0.
    for position in positions:
        if position["asset_type"] != "etf":
            continue
        composition = position["holdings_composition"]
        observed = composition.get("reported_class_nav_weights", {})
        portfolio_weights = {category: position["weight"] * observed.get(category, 0.) for category in observed_weights}
        for category, weight in portfolio_weights.items():
            observed_weights[category] += weight
        unclassified = position["weight"] - math.fsum(portfolio_weights.values())
        unresolved_composition += unclassified
        if position["symbol"] in selected:
            composition_inputs.append({
                "original_symbol": position["symbol"], "original_position_weight": position["weight"],
                "status": composition["status"], "effective_date": composition.get("effective_date"),
                "processed_date": composition.get("processed_date"),
                "reported_class_nav_weights": observed or None,
                "source_class_portfolio_weights": portfolio_weights,
                "unclassified_or_unobserved_portfolio_weight": unclassified,
                "recorded_nav_residual_portfolio_weight": (
                    position["weight"] * composition["recorded_nav_residual"]
                    if "recorded_nav_residual" in composition else None),
                "predominant_observed_class": composition.get("predominant_observed_class"),
                "whole_fund_taxonomy_inferred": False, "ledger_modified": False,
            })
    context["source_verified_holdings_composition"] = {
        "positions": composition_inputs,
        "scope": "Additional research view of the same ETF NAV. Signed cash and signed NAV discrepancies are retained; a negative reconciliation residual is not a negative unknown holding. Structured source classes do not certify full-fund taxonomy or resolve plain-security identity; never add these weights to the existing ledger.",
    }
    context["economic_research_channels"] = {
        family: {"candidate_weight": math.fsum(c["portfolio_weight"] for c in components),
                 "unclassified_candidate_weight": math.fsum(c["portfolio_weight"] for c in components
                     if c["channel_status"] == "UNCLASSIFIED_INSTRUMENT_INVESTIGATION_REQUIRED"),
                 "estimated_sensitivity": None}
        for family, components in channels.items()
    }
    context["research_instruction"] = (
        "Propose concrete events relevant to these original positions and accepted constituents. "
        "Class channels are qualitative hypotheses; estimate no return from class alone. "
        "Use source-weighted constituent composition only for its documented NAV slices, not as whole-fund taxonomy. "
        "Unknown metadata and ETF residuals remain unknown. Event definitions require independent source matching. "
        "For issuer-level event relevance, use the evidence-bounded overlap view but never add its NAV twice; "
        "US ADRs and local ordinary shares do not have identical security returns."
    )
    return {
        "status": "CLASS_BASED_RESEARCH_WITH_EXPLICIT_EVIDENCE_LIMITS",
        "evaluated_at_utc": now.isoformat(), "portfolio": portfolio.to_dict(),
        "positions": positions, "exposure_ledger": ledger.to_dict(),
        "channels_by_family": channels, "agent_exposure_input": context,
        "issuer_overlap_audit": issuer_overlap,
        "coverage": {
            "original_position_weight": portfolio.total_weight,
            "authoritative_ledger_weight": ledger.total_weight,
            "class_weights": class_weights,
            "source_observed_etf_class_portfolio_weights": observed_weights,
            "unclassified_or_unobserved_etf_composition_weight": unresolved_composition,
            "observed_composition_scope": "Research-only constituent asset classes; separate view of the original ETF NAV, not additional exposure or a revision of the authoritative ledger.",
            "locally_verified_instrument_identity_weight": math.fsum(p["weight"] for p in positions if p["instrument_identity_verified"]),
            "unresolved_etf_identity_weight": ledger.unresolved_weight,
            "taiwan_identified_nvda_channel_weight": math.fsum(c["portfolio_weight"] for c in channels["TAIWAN_SUPPLY"] if c["instrument_symbol"] == "NVDA"),
            "taiwan_source_supported_tsmc_channel_weight": math.fsum(c["portfolio_weight"] for c in channels["TAIWAN_SUPPLY"] if c["instrument_symbol"] in ("TSM", "2330")),
            "mass_preserved_without_renormalization": math.isclose(ledger.total_weight, portfolio.total_weight, rel_tol=0, abs_tol=1e-12),
            "all_tickers_or_instrument_classes_supported": False,
        },
        "limitations": [
            "Class channels are economic research hypotheses, not calibrated sensitivities or event probabilities.",
            "User-declared stocks without provider metadata retain an unverified identity label.",
            "Parent classes and the constituent ledger are two views of the same NAV; never add them together.",
            "Issuer overlaps are annotated accepted slices, not additional NAV, security-return parity, or verified corporate identity for arbitrary tickers.",
            "Unknown ETF classes, unsupported security types, rejected holdings and residual NAV remain explicit.",
            "A local response hash establishes stored-byte consistency, not provider authorship or historical publication time.",
            *ledger.limitations,
        ],
    }
