"""Audited, manually transcribed primary Treasury terms for the Round 11 gaps.

This is NOT a Fiscal Data API acquisition or a raw-page receipt. It is a
checked transcription of the US Treasury's 2026-10-21 FedInvest security table.
The source lists the coupon and maturity by CUSIP; it is not a price source for
portfolio valuation. The original fund and scenario engines are unchanged.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

from .treasury_terms import cusip_from_isin

SCHEMA = "castle.option1.round12_primary_treasury_transcription.v1"
SOURCE_URL = "https://www.savingsbonds.gov/GA-FI/FedInvest/todaySecurityPriceDetail"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def read_checked_transcription(path, *, evaluation_date="2026-10-21"):
    """Return (ISIN -> bullet terms, evidence receipt) only for exact reviewed rows.

    Rejects a changed transcription digest, unsupported security types, malformed
    identifiers, dates after the evaluation date, duplicated identities and
    malformed or out-of-domain coupons. The digest is of the LOCAL transcription;
    no original website bytes are asserted or reconstructed.
    """
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA:
        raise ValueError("Unexpected primary Treasury transcription schema")
    source = body.get("source") or {}
    if (source.get("url") != SOURCE_URL or source.get("primary_publisher") != "US Department of the Treasury"
            or source.get("capture_kind") != "MANUALLY_REVIEWED_OFFICIAL_WEB_TABLE"
            or source.get("raw_http_bytes_captured") is not False):
        raise ValueError("Unexpected primary Treasury source representation")
    source_date = date.fromisoformat(source["displayed_as_of_date"])
    if source_date > date.fromisoformat(evaluation_date):
        raise ValueError("A later source observation cannot certify an earlier replay")
    rows = body.get("securities")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("The eight explicitly reviewed CUSIPs are required")
    expected_digest = body.get("transcription_sha256")
    unsigned = {k: v for k, v in body.items() if k != "transcription_sha256"}
    actual_digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("Primary Treasury transcription hash mismatch")
    result = {}
    for row in rows:
        isin = row.get("isin")
        cusip = row.get("cusip")
        if not isinstance(isin, str) or cusip_from_isin(isin) != cusip:
            raise ValueError("CUSIP/ISIN association is not valid")
        if isin in result:
            raise ValueError("Duplicate instrument identity")
        if row.get("security_type") not in ("MARKET BASED BOND", "MARKET BASED NOTE"):
            raise ValueError("No verified fixed-coupon bullet Treasury type")
        coupon = row.get("coupon_percent")
        if isinstance(coupon, bool) or not isinstance(coupon, (float, int)) or not 0 < coupon < 20:
            raise ValueError("Coupon is not admissible")
        maturity = date.fromisoformat(row["maturity"])
        if maturity <= date.fromisoformat(evaluation_date):
            raise ValueError("Matured Treasury instrument")
        result[isin] = {"coupon_percent": float(coupon), "maturity": maturity.isoformat(),
                        "bullet_fixed_coupon": True,
                        "source_kind": source["capture_kind"],
                        "source_url": SOURCE_URL,
                        "transcription_sha256": actual_digest}
    return result, {"status": "PRIMARY_TRANSCRIPTION_REVIEWED_NOT_PROVIDER_API",
                    "source_url": SOURCE_URL, "source_displayed_as_of_date": source_date.isoformat(),
                    "transcription_sha256": actual_digest, "number_of_securities": len(result),
                    "raw_http_bytes_captured": False, "provider_api_request_performed": False}


def terms_available_for_assessment(evaluated_at):
    """Load only the reviewed, dated primary terms, without a provider request.

    Never time-travel the 21 October transcription into an earlier assessment.
    A missing, malformed or changed evidence file cannot create new duration
    coverage; the old uncovered-bond calculation remains available.
    """
    when = evaluated_at.date() if hasattr(evaluated_at, "date") else date.fromisoformat(str(evaluated_at)[:10])
    if when < date(2026, 10, 21):
        return {}, {"status": "PRIMARY_TERMS_NOT_YET_AVAILABLE_AT_EVALUATION"}
    path = Path(__file__).resolve().parents[1] / "data/round12_primary_treasury/source_transcription.json"
    try:
        return read_checked_transcription(path, evaluation_date=when.isoformat())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {}, {"status": "PRIMARY_TERMS_NOT_ADMITTED", "reason": str(exc)}
