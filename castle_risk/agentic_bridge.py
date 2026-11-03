"""Evidence-preserving bridge between Claude discovery and the existing Castle risk engine.

This module never calls a provider.  Claude's proposed event is *not* the same as
an exchange contract's YES settlement: only the latter has a self-identity
relation.  A price quote is a conditional pricing-measure interval, not an
estimated real-world probability or portfolio return.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .adapters import Market, normalize_quote, normalize_market
from .impact import analyze_scenario
from .portfolio import parse_portfolio
from .semantics import make_contract_target, verify_relationship


class AgenticEvidenceError(ValueError):
    """An actual read-only source does not meet the declared evidence gate."""


def _receipt(market: Market) -> dict[str, Any]:
    """Check original response bytes against the adjacent local HTTP receipt.

    This checks stored-byte consistency and the observed request metadata only;
    it is not a Kalshi signature or independently witnessed publication time.
    """
    if (not isinstance(market, Market) or not isinstance(market.raw_path, (str, Path))
            or not market.raw_path):
        raise AgenticEvidenceError("No local original response file for this market.")
    source = Path(market.raw_path)
    receipt_file = source.with_suffix(".meta.json")
    try:
        body = source.read_bytes()
        metadata = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgenticEvidenceError("Source response and receipt are not both readable.") from None
    if not isinstance(metadata, dict):
        raise AgenticEvidenceError("Source receipt is not a metadata object.")
    digest = hashlib.sha256(body).hexdigest()
    if (metadata.get("sha256") != digest or metadata.get("status") != 200
            or metadata.get("fixture") is not False
            or metadata.get("retrieved_at") != market.retrieved_at):
        raise AgenticEvidenceError("Raw source body or receipt is inconsistent or fixture-labelled.")
    url = metadata.get("url")
    parts = urlsplit(url) if isinstance(url, str) else None
    if (not parts or parts.scheme != "https" or parts.hostname != "external-api.kalshi.com"
            or not parts.path.startswith("/trade-api/v2/")):
        raise AgenticEvidenceError("Source receipt is not a supported Kalshi HTTPS request.")
    if (not isinstance(market.source_url, str)
            or not market.source_url.startswith("https://external-api.kalshi.com/trade-api/v2/markets/")):
        raise AgenticEvidenceError("Market's source URL is not an allowed exact-contract URL.")
    if not isinstance(market.metadata, dict) or not isinstance(market.metadata.get("raw"), dict):
        raise AgenticEvidenceError("Original market rules and price fields are not available.")
    try:
        body_json = json.loads(body)
        if not isinstance(body_json, dict):
            raise AgenticEvidenceError("Stored source response is not a provider object.")
        raw = market.metadata["raw"]
        candidates = body_json.get("markets", [])
        if "market" in body_json:
            candidates = [body_json["market"]]
        if not any(c == {k: v for k, v in raw.items() if k != "_reviewed_context"} for c in candidates):
            raise AgenticEvidenceError("The normalized market is not present in the stored source response.")
    except (TypeError, ValueError):
        raise AgenticEvidenceError("Stored source response is invalid or omits this exact market.") from None
    # A matching raw JSON record is not enough if a normalized Python object
    # has been altered afterwards. Recompute every settlement and quote input
    # from the locally matched original provider row before admitting ANY Q.
    try:
        rebuilt = normalize_market(raw, market.retrieved_at, market.raw_path)
    except (ValueError, TypeError, KeyError) as exc:
        raise AgenticEvidenceError("Market could not be re-normalized from original provider bytes.") from None
    for name in ("ticker", "event_ticker", "series_ticker", "title", "subtitle",
                 "rules_primary", "rules_secondary", "close_time", "expiration_time",
                 "settlement_timer_seconds", "status", "yes_bid", "yes_ask",
                 "quote_timestamp", "source_url"):
        if getattr(market, name) != getattr(rebuilt, name):
            raise AgenticEvidenceError("Normalized source attribute has changed since the original HTTP response: " + name)
    return {"request_url": url, "source_response_sha256": digest,
            "retrieved_at_utc": market.retrieved_at,
            "assurance": "LOCAL_HTTP_RECEIPT_BYTES_VERIFIED_NOT_PROVIDER_SIGNATURE",
            "exchange_quote_update_time_verified": False}


def catalog_record(market: Market) -> dict[str, Any]:
    """Supply Claude with complete, real rules plus provenance; never tool-authored claims.

    An unsuccessful source verification does *not* disappear: it creates an
    explicit non-admission state; callers should show the failed receipt gate.
    """
    if not isinstance(market, Market):
        raise AgenticEvidenceError("Expected a normalized provider market, not model-authored text.")
    raw = market.metadata.get("raw") if isinstance(market.metadata, dict) else None
    if not isinstance(raw, dict) or raw.get("ticker") != market.ticker:
        raise AgenticEvidenceError("Normalized contract identity disagrees with raw provider data.")
    result = {key: value for key, value in raw.items() if not key.startswith("_")}
    # These are trusted adapter-derived attributes, never supplied by Claude.
    result.update(source_url=market.source_url, retrieved_at=market.retrieved_at,
                  quote_timestamp=market.quote_timestamp,
                  quote_timestamp_kind="retrieval_receipt", venue="kalshi")
    try:
        receipt = _receipt(market)
    except AgenticEvidenceError as exc:
        result["source_receipt_status"] = "UNVERIFIED"
        result["source_receipt_reason"] = str(exc)
    else:
        result["source_receipt_status"] = "LOCAL_BYTES_VERIFIED"
        result["source_response_sha256"] = receipt["source_response_sha256"]
        result["source_request_url"] = receipt["request_url"]
    return result


def inspect_exact_contract(
    market: Market, *, portfolio_rows: list[dict], notional: float,
    stresses: Mapping[str, float] | None = None,
    observed_at: datetime | str | None = None,
    max_age_seconds: int = 300,
) -> dict[str, Any]:
    """Inspect a source-backed contract directly, without a second HTTP request.

    A model-proposed macro event is not admitted.  Contract identity is verified
    from original complete rules, and a Q bound is permitted only when the source
    bytes, live/open status and observation-time quote gates all pass.
    """
    if not isinstance(market, Market):
        raise AgenticEvidenceError("A real normalized provider Market is required.")
    portfolio = parse_portfolio(portfolio_rows, notional)
    receipt = _receipt(market)  # no caller-supplied 'verified' flag can replace it
    raw = market.metadata["raw"]
    target = make_contract_target({**raw, "venue": "kalshi"})
    semantic = verify_relationship(target, {**raw, "venue": "kalshi"})
    if semantic["status"] != "CERTIFIED" or semantic["relationship"] != "EQUIVALENT":
        raise AgenticEvidenceError("The exact contract identity did not pass independent verification.")
    if isinstance(observed_at, str):
        observed_at = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if observed_at is None:
        observed_at = datetime.now(timezone.utc)
    quote = normalize_quote(market, now=observed_at, max_age_seconds=max_age_seconds)
    assumption_stress = analyze_scenario(portfolio, stresses or {},
        name="User-defined conditional stress, NOT fitted event impact",
        assumption_note="The inspected contract's YES payoff need not equal the macro event proposed by Claude.")
    return {
        "status": "CONTRACT_IDENTITY_Q_BOUND" if quote.usable else "CONTRACT_IDENTITY_NO_ADMISSIBLE_QUOTE",
        "scope": "EXACT_KALSHI_CONTRACT_YES_SETTLEMENT_ONLY",
        "model_proposed_macro_event_matched": False,
        "contract_id": market.ticker,
        "event_ticker": market.event_ticker,
        "title": market.title,
        "rules_sha256": target["rules_sha256"],
        "source": receipt,
        "quote": quote.to_dict(),
        "pricing_measure_Q_interval": [quote.lower, quote.upper] if quote.usable else None,
        "physical_probability_P": None,
        "equity_priced_in_q": None,
        "historical_price_response_estimate": None,
        "portfolio_stress": assumption_stress.to_dict(),
        "portfolio_stress_status": "OPERATOR_ASSUMPTIONS_NOT_EVENT_CAUSALITY",
        "notional_usd": portfolio.notional,
        "limitations": [
            "Only the exact retrieved contract YES settlement is verified; Claude's macro-event hypothesis remains unverified.",
            "A quote receipt is not an exchange-side quote-update timestamp or a probability confidence interval.",
            "Real-world probability P and equity pricing weight q are not inferred from Kalshi Q.",
            "Portfolio stresses are editable operator assumptions, not causal or learned event responses.",
            "ETF constituent exposures must not be added on top of parent ETF P&L.",
        ],
    }


def annotate_discovery_coverage(discovery: Mapping[str, Any]) -> dict[str, Any]:
    """Truthful source/coverage labels OUTSIDE the hash-frozen Anthropic engine.

    Never upgrades the agent's semantic statements or source evidence: it only
    demotes misleading success when read-only catalog searches all failed, and
    explains that finite query coverage cannot show market-universe absence.
    """
    report = dict(discovery)
    searches = list(report.get('searches') or [])
    failed = sum(isinstance(row, dict) and row.get('status') == 'ADAPTER_ERROR' for row in searches)
    successful = len(searches) - failed
    coverage = ('NO_MARKET_SEARCH_EXECUTED' if not searches else
                'ALL_MARKET_SEARCHES_FAILED' if successful == 0 else
                'PARTIAL_MARKET_SEARCH_FAILURES' if failed else
                'BOUNDED_QUERIES_ONLY_NOT_COMPLETE_MARKET_UNIVERSE')
    if failed and successful == 0:
        report['status'] = 'SOURCE_SEARCH_FAILED'
    elif failed:
        report['status'] = 'PARTIAL_SOURCE_FAILURE'
    elif not searches and report.get('status') == 'COMPLETE':
        report['status'] = 'NO_MARKET_SEARCH_EXECUTED'
    report['coverage_status'] = coverage
    report['all_market_availability_verified'] = False
    report['summary_evidence_status'] = 'LLM_TEXT_NOT_INDEPENDENTLY_VERIFIED'
    return report
