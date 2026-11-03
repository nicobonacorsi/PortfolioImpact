"""Primary-source instrument evidence for the captured fund rows that lack terms.

Two bounded acquisitions, both through the application's read-only client:

* Treasury terms. Seven TLT rows ("TREASURY BOND OTR", "WI TREASURY BOND") and
  one IEF row ("TREASURY NOTE OTR") carry an ISIN but no coupon or maturity, so
  their NAV weight has unknown duration. The CUSIP is the ISIN's middle nine
  characters (both check digits are verified first). The U.S. Treasury Fiscal
  Data "Treasury Securities Auctions Data" API is queried by CUSIP. Terms are
  accepted only if every auction record of that CUSIP agrees on maturity date
  and interest rate, the security is a fixed-rate Note or Bond, and it is not
  inflation-indexed, floating-rate or callable. Anything else stays uncovered
  with its reason.
* Fund self-ticker identity. A row that carries the fund's own ticker is
  checked by asking Massive which ticker, if any, its CUSIP belongs to.
  An empty answer is not evidence either way and leaves the row unresolved.

The response schema used here (``data`` rows with ``cusip``, ``security_type``,
``maturity_date``, ``int_rate``, ``inflation_index_security``,
``floating_rate``) follows the public Fiscal Data documentation; a response
without those fields is rejected, never guessed.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from .adapters import MASSIVE_BASE, DataAccessError, _ReadOnlyClient
from .provider_access import AuditedClientMixin
from .fund_composition import _cusip_check, _isin_check, _segment, _text, selected_rows_from_holdings, _bullet_terms

SCHEMA = "castle.option1.instrument_terms_evidence.v1"
FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
AUCTIONS_ENDPOINT = FISCALDATA_BASE + "/v1/accounting/od/auctions_query"
REQUIRED_FIELDS = ("cusip", "security_type", "maturity_date", "int_rate")


class FiscalDataClient(AuditedClientMixin, _ReadOnlyClient):
    allowed_hosts = {"api.fiscaldata.treasury.gov"}


def cusip_from_isin(isin):
    """US ISIN -> CUSIP, only when both check digits hold."""
    value = _text(isin).upper()
    if len(value) != 12 or not value.startswith("US") or not _isin_check(value):
        return None
    cusip = value[2:11]
    return cusip if _cusip_check(cusip) else None


def uncovered_treasury_rows(holdings_by_symbol, symbols=("TLT", "IEF")):
    """Treasury rows of the named funds with no machine-readable bullet terms."""
    rows = []
    for symbol in symbols:
        holdings = (holdings_by_symbol or {}).get(symbol)
        if not holdings:
            continue
        for row in selected_rows_from_holdings(holdings):
            if _segment(row, symbol) != "US_TREASURY" or _bullet_terms(row) is not None:
                continue
            isin = _text(row.get("isin")).upper()
            rows.append({"fund": symbol, "isin": isin, "cusip": cusip_from_isin(isin),
                         "constituent_name": _text(row.get("constituent_name")),
                         "constituent_ticker": _text(row.get("constituent_ticker")),
                         "weight": row.get("weight"), "market_value": row.get("market_value"),
                         "shares_held": row.get("shares_held"),
                         "implied_price_per_100_par": (100.0 * row["market_value"] / row["shares_held"]
                                                       if row.get("market_value") and row.get("shares_held") else None)})
    return rows


def parse_auction_terms(payload, cusip):
    """Fixed-coupon bullet terms from all auction records of one CUSIP, or a reason."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {"status": "SCHEMA_REJECTED_NO_DATA_ARRAY"}
    records = [r for r in data if isinstance(r, dict) and _text(r.get("cusip")).upper() == cusip]
    if not records:
        return {"status": "NO_RECORD_FOR_CUSIP"}
    if any(any(field not in r for field in REQUIRED_FIELDS) for r in records):
        return {"status": "SCHEMA_REJECTED_MISSING_FIELDS", "records": len(records)}
    maturities = {_text(r["maturity_date"]) for r in records}
    rates = {_text(r["int_rate"]) for r in records if _text(r["int_rate"]) not in {"", "null"}}
    types = {_text(r["security_type"]).upper() for r in records}
    flags = {k: {_text(r.get(k)).upper() for r in records} for k in ("inflation_index_security", "floating_rate", "callable")}
    base = {"auction_records": len(records), "security_types": sorted(types), "maturity_dates": sorted(maturities),
            "interest_rates": sorted(rates), "issue_dates": sorted({_text(r.get("issue_date")) for r in records}),
            "flags": {k: sorted(v) for k, v in flags.items()}}
    if not types <= {"NOTE", "BOND"}:
        return {**base, "status": "NOT_A_NOTE_OR_BOND"}
    if "YES" in flags["inflation_index_security"] or "YES" in flags["floating_rate"] or "YES" in flags["callable"]:
        return {**base, "status": "NOT_FIXED_COUPON_BULLET"}
    if len(maturities) != 1 or len(rates) != 1:
        return {**base, "status": "INCONSISTENT_TERMS_ACROSS_RECORDS"}
    try:
        maturity = date.fromisoformat(next(iter(maturities)))
        coupon = float(next(iter(rates)))
    except ValueError:
        return {**base, "status": "SCHEMA_REJECTED_UNPARSABLE_TERMS"}
    if not 0 <= coupon <= 20:
        return {**base, "status": "COUPON_OUT_OF_RANGE"}
    return {**base, "status": "TERMS_ACCEPTED", "maturity": maturity.isoformat(), "coupon_percent": coupon,
            "bullet_fixed_coupon": True}


def acquire_treasury_terms(rows, client):
    """One Fiscal Data query per distinct CUSIP; returns evidence rows and an ISIN->terms map."""
    evidence, terms, seen = [], {}, {}
    for row in rows:
        cusip = row["cusip"]
        if cusip is None:
            evidence.append({**row, "status": "IDENTIFIER_CHECK_DIGIT_FAILED"})
            continue
        if cusip not in seen:
            url = AUCTIONS_ENDPOINT + "?" + urlencode({"filter": "cusip:eq:" + cusip, "page[size]": 100})
            try:
                payload, meta = client._get(url)
                parsed = parse_auction_terms(payload, cusip)
                seen[cusip] = {**parsed, "source_url": url, "retrieved_at": meta["retrieved_at"],
                               "source_sha256": meta["sha256"], "http_status": meta["status"]}
            except DataAccessError as exc:
                seen[cusip] = {"status": "ACQUISITION_FAILED", "failure_class": getattr(exc, "failure_class", None),
                               "source_url": url}
        result = seen[cusip]
        evidence.append({**row, **result})
        if result.get("status") == "TERMS_ACCEPTED":
            terms[row["isin"]] = {k: result[k] for k in ("maturity", "coupon_percent", "bullet_fixed_coupon",
                                                         "source_sha256", "retrieved_at")}
    return evidence, terms


def resolve_self_ticker_rows(fund_symbol, rows, massive):
    """Ask Massive which ticker each self-labelled row's CUSIP belongs to."""
    out = []
    for row in rows:
        cusip = cusip_from_isin(row.get("isin"))
        record = {"fund": fund_symbol, "isin": row.get("isin"), "cusip": cusip, "weight": row.get("weight")}
        if cusip is None:
            out.append({**record, "status": "IDENTIFIER_CHECK_DIGIT_FAILED"})
            continue
        url = MASSIVE_BASE + "/v3/reference/tickers?" + urlencode({"cusip": cusip, "limit": 10})
        try:
            payload, meta = massive._authenticated_get(url)
        except DataAccessError as exc:
            out.append({**record, "status": "ACQUISITION_FAILED", "failure_class": getattr(exc, "failure_class", None)})
            continue
        tickers = sorted({_text(r.get("ticker")).upper() for r in payload.get("results") or [] if isinstance(r, dict)})
        status = ("CONFIRMED_SELF_REFERENCE_BY_PROVIDER_CUSIP" if tickers == [fund_symbol] else
                  "TICKER_FIELD_COLLISION_PROVIDER_MAPS_CUSIP_ELSEWHERE" if tickers else
                  "UNRESOLVED_PROVIDER_HAS_NO_TICKER_FOR_CUSIP")
        out.append({**record, "status": status, "provider_tickers": tickers, "retrieved_at": meta["retrieved_at"],
                    "source_sha256": meta["sha256"]})
    return out


def self_ticker_rows(holdings, fund_symbol):
    return [r for r in selected_rows_from_holdings(holdings) if _text(r.get("constituent_ticker")).upper() == fund_symbol]


def evidence_document(treasury_evidence, terms, self_ticker_evidence):
    body = {"schema": SCHEMA, "treasury_terms": treasury_evidence, "accepted_terms_by_isin": terms,
            "self_ticker_identity": self_ticker_evidence,
            "accepted_weight_by_fund": {}, "requested_weight_by_fund": {}}
    for row in treasury_evidence:
        fund = row["fund"]
        body["requested_weight_by_fund"][fund] = body["requested_weight_by_fund"].get(fund, 0.0) + (row["weight"] or 0.0)
        if row.get("status") == "TERMS_ACCEPTED":
            body["accepted_weight_by_fund"][fund] = body["accepted_weight_by_fund"].get(fund, 0.0) + (row["weight"] or 0.0)
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


__all__ = ["AUCTIONS_ENDPOINT", "FiscalDataClient", "acquire_treasury_terms", "cusip_from_isin", "evidence_document",
           "parse_auction_terms", "resolve_self_ticker_rows", "self_ticker_rows", "uncovered_treasury_rows"]
