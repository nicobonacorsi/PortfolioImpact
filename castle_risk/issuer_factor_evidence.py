"""Source-identified original-fund stress factors, separate from holdings DV01.

The three entries are *dated manual transcriptions* of public primary-issuer
facts reviewed on 2026-10-29.  They are NOT archived GET receipts, an API
entitlement, a calibrated event response, or constituent-level look-through.
The time gate is important: old R59 captures cannot silently acquire facts
reviewed later.  A future review should replace these records with a hashed
issuer snapshot and an independently tested extractor.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import math
import re
from typing import Any, Mapping


REVIEWED_AT_UTC = "2026-10-29T20:09:24+00:00"
MAX_REVIEW_AGE_DAYS = 60

# URLs are official issuers; figures and objective are manually reviewed
# transcriptions, not claims of a locally hash-verified web download.
_REVIEWED = {
    "AGG": {
        "kind": "BOND_EFFECTIVE_DURATION",
        "provider_type": "ETF",
        "issuer_name_tokens": ("ishares", "aggregate", "bond"),
        "duration_years": 5.70,
        "fact_as_of": "2026-10-28",
        "official_source_url": "https://www.ishares.com/us/products/239458/ISHARES-CORE-US-AGGREGATE-BOND-ETF",
        "official_locator": "Portfolio Characteristics / Effective Duration",
        "issuer_security_name": "iShares Core U.S. Aggregate Bond ETF",
    },
    "BND": {
        "kind": "BOND_AVERAGE_DURATION",
        "provider_type": "ETF",
        "issuer_name_tokens": ("vanguard", "total", "bond", "market"),
        "duration_years": 5.70,
        "fact_as_of": "2026-09-30",
        "official_source_url": "https://advisors.vanguard.com/investments/products/bnd/vanguard-total-bond-market-etf",
        "official_locator": "Fund profile / Average duration",
        "issuer_security_name": "Vanguard Total Bond Market ETF",
    },
    "GLD": {
        "kind": "PHYSICAL_GOLD_TRUST_SPOT_PRICE_OBJECTIVE",
        "provider_type": "ETV",
        "issuer_name_tokens": ("spdr", "gold"),
        "fact_as_of": "2026-10-29",
        "official_source_url": "https://www.spdrgoldshares.com/usa/gld/",
        "official_locator": "Key Features / investment objective / gold bullion less expenses",
        "issuer_security_name": "SPDR Gold Shares",
        "spot_loading": 1.0,
    },
}


def review_issuer_factor(symbol: str, provider_metadata: Mapping[str, Any] | None,
                         as_of: str | datetime) -> dict[str, Any]:
    """Review a fund-level risk factor without upgrading constituent evidence.

    A verified source *description* supports a local instantaneous sensitivity
    assumption.  It does not establish the event's shock, spread change,
    maturity-specific curve repricing, tracking error, or conditional P&L.
    """
    facts = _REVIEWED.get(symbol)
    result = {
        "symbol": symbol,
        "status": "UNAVAILABLE",
        "source_kind": "MANUALLY_REVIEWED_OFFICIAL_ISSUER_DISPLAY_NOT_ARCHIVED_BYTES",
        "reviewed_at_utc": REVIEWED_AT_UTC,
        "provider_identity_checked": False,
        "event_to_factor_response_identified": False,
        "causal_or_predictive_validation": False,
        "full_event_pnl_identified": False,
    }
    if not facts:
        result["reason"] = "No independently reviewed original-fund issuer factor for this symbol."
        return result
    result.update({k: deepcopy(v) for k, v in facts.items() if k != "issuer_name_tokens"})
    try:
        now = as_of if isinstance(as_of, datetime) else datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Factor assessment requires an ISO timestamp") from exc
    if now.utcoffset() is None:
        raise ValueError("Factor assessment requires a timezone-aware timestamp")
    now = now.astimezone(timezone.utc)
    if now < datetime.fromisoformat(REVIEWED_AT_UTC):
        result["reason"] = "Issuer review took place after the original assessment; replay look-ahead prohibited."
        return result
    age = (now.date() - date.fromisoformat(facts["fact_as_of"])).days
    result["fact_age_days"] = age
    if age < 0 or age > MAX_REVIEW_AGE_DAYS:
        result["reason"] = "Issuer factor fact is outside its declared 60-day review-age scope."
        return result
    meta = provider_metadata or {}
    if (meta.get("ticker") != symbol or meta.get("type") != facts["provider_type"]
            or meta.get("active") is not True or meta.get("provider") != "Massive"
            or not isinstance(meta.get("source_response_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", meta["source_response_sha256"])):
        result["reason"] = "Original position ticker/type lacks a supported dated provider identity receipt."
        return result
    name = str(meta.get("name") or "").lower()
    if not all(token in name for token in facts["issuer_name_tokens"]):
        result["reason"] = "Current provider instrument name contradicts or fails to identify the issuer product."
        return result
    try:
        retrieved = datetime.fromisoformat(str(meta["retrieved_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError):
        result["reason"] = "Provider identity receipt has no valid retrieval timestamp."
        return result
    if retrieved.utcoffset() is None or retrieved.astimezone(timezone.utc) > now:
        result["reason"] = "Provider identity is not available as of the stress assessment."
        return result
    result["provider_identity_checked"] = True
    result["provider_identity_receipt_sha256"] = meta["source_response_sha256"]
    result["status"] = "ISSUER_FACT_AND_PROVIDER_IDENTITY_REVIEWED"
    result["scope"] = (
        "Dated original-fund effective/average duration; first-order parallel-yield price sensitivity only."
        if facts["kind"].startswith("BOND_") else
        "Physical-gold trust spot-tracking objective, less expenses; an instantaneous unit spot loading is a declared approximation."
    )
    if facts["kind"].startswith("BOND_") and not (
            isinstance(facts["duration_years"], (int, float))
            and math.isfinite(facts["duration_years"]) and 0 < facts["duration_years"] <= 50):
        raise ValueError("Invalid manually reviewed issuer duration")
    return result


def reviewed_symbols() -> tuple[str, ...]:
    """Only these named issuer products are reviewed; no ticker guessing."""
    return tuple(sorted(_REVIEWED))
