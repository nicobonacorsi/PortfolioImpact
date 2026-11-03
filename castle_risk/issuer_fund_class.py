"""Narrow, reproducible issuer-document evidence for original-fund class.

This is not a general ticker registry, holdings substitution, or return model.
The deliberately single reviewed document is distributed with the private
assessment so its original bytes can be rechecked at execution time.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path

from .instrument_exposures import evaluation_instant


_XLE_FACTSHEET = Path(__file__).resolve().parent.parent / "data/official_issuer/xle_ssga_20260930.pdf"
_XLE_SHA256 = "7a7c83b7f0ae43a57f320e83866ddf12a68b9d00f4f6dbdd1da9b8a57d209454"
_XLE_URL = "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xle.pdf"
_FIRST_REVIEW_UTC = datetime(2026, 10, 29, 1, 15, 0, tzinfo=timezone.utc)


def reviewed_original_fund_equity_class(symbol: str, *, evaluated_at, document_path: Path | None = None):
    """Return a primary-issuer equity *class* only with the reviewed bytes.

    The September 2026 factsheet says XLE / Equity, identifies CUSIP
    81369Y506 and 21 holdings. The PDF was actually captured and reviewed on
    2026-10-29 01:14 UTC (October 28 in New York), so it is deliberately
    unavailable to earlier as-of analyses.
    A class does not reveal each constituent or validate beta/market shocks.
    """
    if symbol != "XLE":
        return None
    try:
        instant = evaluation_instant(evaluated_at)
        if instant < _FIRST_REVIEW_UTC or instant.date() > date(2026, 12, 29):
            return None
        path = Path(document_path) if document_path is not None else _XLE_FACTSHEET
        body = path.read_bytes()
    except (OSError, TypeError, ValueError):
        return None
    if not body.startswith(b"%PDF-") or sha256(body).hexdigest() != _XLE_SHA256:
        return None
    return {
        "symbol": "XLE", "issuer_name": "State Street Investment Management",
        "issuer_asset_class": "equity", "issuer_cusip_documented": "81369Y506",
        "fund_holdings_count_as_reported": 21, "factsheet_as_of": "2026-09-30",
        "first_verified_for_assessment": _FIRST_REVIEW_UTC.isoformat(),
        "source_url": _XLE_URL, "reviewed_source_pdf_sha256": _XLE_SHA256,
        "source_byte_status": "LOCAL_OFFICIAL_PDF_BYTES_MATCH_REVIEWED_SHA256",
        "scope": "ORIGINAL_FUND_EQUITY_MANDATE_ONLY_NOT_CONSTITUENT_RECONCILIATION",
        "does_not_validate_beta_or_event_response": True,
        "does_not_resolve_holdings": True,
        "provider_identity_must_be_verified_separately": True,
    }
