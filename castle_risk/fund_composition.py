"""Identity-aware, signed composition of bond and commodity ETF constituent rows.

The original holdings validator rejects a snapshot on any repeated ticker, any
negative weight, or a row whose ticker equals the fund. That protects equity
look-through, but for bond funds it rejects valid structure: agency pools share
short pool tickers, TBA dollar rolls are reported as a long and a short row of
the same CUSIP, a fund may carry a negative cash balance, and a bond can carry
the fund's own ticker as a mislabeled field. This module reads the same rows and
keeps every one of them. It never deletes a row, clips a weight, renormalizes to
100%, or recurses into the fund itself; unresolved mass stays explicit.

The original validator and its ``reliable`` flag are not changed. The output is
a separate signed composition view of the fund NAV and an interest-rate
sensitivity for the bullet bonds whose coupon and maturity are machine-readable.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "castle.fund_signed_composition.v1"
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_FIGI = re.compile(r"[A-Z0-9]{11}[0-9]")
_CUSIP = re.compile(r"[A-Z0-9]{8}[0-9]")
_TICKER_BOND = re.compile(
    r"^(?P<issuer>[A-Z0-9./]+) (?P<coupon>\d{1,2}(?:\.\d+)?) (?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2}|\d{4})$")
_NAME_BOND = re.compile(
    r"(?P<coupon>\d{1,2}(?:\.\d+)?)%?\s+(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4}|\d{2})\s*$")
# Amortizing, prepayable or inflation-indexed securities need a cash-flow model
# that a coupon and final maturity do not supply. They are never given a
# bullet-bond duration.
_NO_BULLET_DURATION = ("MBS", "TBA", "CMBS", "ABS", "ARM", "POOL", "TIPS", "INFLATION")
# Declared yield domain for funds that report no par amount (so no price).
# Endpoints are a stress domain for a duration interval, not a yield estimate.
ASSUMED_YIELD_DOMAIN = (0.03, 0.07)


UNRESOLVED_SEGMENTS = ("CONFLICTING_IDENTITY_METADATA", "UNRESOLVED_SELF_TICKER_COLLISION")

# These are accounting/model-scope tolerances, not estimated return bounds.
# The cash-liability limit is the existing source-class review's tolerance.
RATE_SCOPE_SCHEMA = "castle.rate_model_composition_scope.v1"
RATE_DOLLAR_BINDING_SCHEMA = "castle.rate_dollar_position_binding.v1"
RATE_SCOPE_NAV_TOLERANCE = 1e-8
RATE_SCOPE_REPORTED_NAV_ROUNDING_TOLERANCE = 1e-6
RATE_SCOPE_CASH_LIABILITY_LIMIT = 0.001
_BULLET_RATE_SEGMENTS = {"US_TREASURY", "CORPORATE_AND_OTHER_BONDS"}
_CASH_RATE_SEGMENTS = {"CASH_AND_EQUIVALENTS", "MONEY_MARKET_FUND"}
_SPECIALIZED_RATE_SEGMENTS = {"AGENCY_MBS_TBA_FORWARD", "AGENCY_MBS_POOL", "CMBS", "ABS",
                              "INFLATION_LINKED_BONDS", "FLOATING_RATE_BONDS",
                              "FIXED_INCOME_DERIVATIVE", "CONVERTIBLE_OR_CALLABLE_BONDS"}


def review_rate_dollar_binding(row, *, symbol, portfolio_notional_usd, position_weight,
                              position_asset_type):
    """Dollar DV01 is reusable only for the position whose dollars it measures.

    This is a dimensional-consistency check, not source authentication. Missing
    legacy context cannot establish the amount used to compute a dollar DV01;
    fresh calculations must regenerate it from the retained composition.
    """
    result = {"status": "DOLLAR_DV01_PORTFOLIO_BINDING_MISMATCH_OR_MISSING",
              "binding_valid": False,
              "reason": "Dollar DV01 lacks a matching original portfolio notional and position weight/type."}
    if not isinstance(row, Mapping):
        return result
    bound = row.get("dollar_sensitivity_binding")
    expected = {"schema": RATE_DOLLAR_BINDING_SCHEMA, "portfolio_notional_usd": portfolio_notional_usd,
                "position_symbol": symbol, "position_weight": position_weight,
                "position_asset_type": position_asset_type, "currency": "USD"}
    if not isinstance(bound, Mapping) or set(bound) != set(expected) or row.get("symbol") != symbol:
        return result
    for field, positive in (("portfolio_notional_usd", True), ("position_weight", False)):
        supplied, current = bound[field], expected[field]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (supplied, current)):
            return result
        if supplied != current or (supplied <= 0 if positive else not 0 <= supplied <= 1):
            return result
    if (position_asset_type not in {"stock", "etf", "cash"}
            or any(bound[key] != value for key, value in expected.items())):
        return result
    result.update(status="BOUND_TO_ORIGINAL_PORTFOLIO_POSITION", binding_valid=True,
                  reason="Original dollar sensitivity notional, position weight and instrument declaration match.")
    return result


def rate_model_scope_inputs(composition: Mapping) -> dict:
    """Retain source-ledger facts needed to recheck an ETF's rate-model scope.

    A covered Treasury inside an equity or balanced ETF does not make the
    entire fund a bond. Keep the class ledger, not just a derived approval
    flag, so consumers can recompute the restriction when serialized metadata
    is missing or inconsistent. This is local source-ledger accounting, not
    independent authentication of an arbitrary supplied JSON.
    """
    if not isinstance(composition, Mapping):
        return {}
    fields = ("fund_symbol", "effective_date", "reported_weight_sum", "net_weight_included",
              "unreported_or_residual_nav_weight", "rows_missing_weight", "segments",
              "conflicting_identity_group_count", "fixed_income_duration_coverage")
    return {key: json.loads(json.dumps(composition[key])) for key in fields if key in composition}


def review_rate_model_scope(composition: Mapping) -> dict:
    """Decide whether a covered-bond channel can represent this entire ETF.

    The complete declared proxy is restricted to reconciled, non-short,
    ordinary fixed-income/cash-like ledgers. Missing bullet terms may be
    bridged only on the explicitly recorded remaining FIXED-INCOME NAV, under
    the already-declared equal-duration assumption. Equity, commodity,
    unclassified NAV, specialized bond structures and noncash shorts are not
    relabeled as bonds. Known partial DV01 remains useful when this gate fails.
    """
    out = {"schema": RATE_SCOPE_SCHEMA, "status": "PARTIAL_COVERED_BONDS_ONLY",
           "whole_fund_rate_proxy_allowed": False,
           "source_class_accounting_only": True,
           "uncovered_fixed_income_equal_duration_is_assumed": True,
           "cash_like_return_treatment": "DECLARED_UNCHANGED_NOMINAL_VALUE_OVER_ANNOUNCEMENT_WINDOW",
           "cash_like_return_assumption": ("Cash balances and source-identified money-market funds are "
                                          "approximated as unchanged nominal NAV over the announcement window. "
                                          "This is not an assertion of zero money-market-fund duration or risk; "
                                          "cash accrual and MMF repricing are not modeled."),
           "reported_nav_rounding_tolerance": RATE_SCOPE_REPORTED_NAV_ROUNDING_TOLERANCE,
           "small_residual_treatment": "DECLARED_SOURCE_ROUNDING_RESIDUAL_RETAINED_NOT_REALLOCATED",
           "whole_fund_nav_used_as_bond_nav": False}
    try:
        if not isinstance(composition, Mapping):
            raise ValueError("Source composition ledger is unavailable")
        segments = composition.get("segments")
        coverage = composition.get("fixed_income_duration_coverage")
        if not isinstance(segments, Mapping) or not segments or not isinstance(coverage, Mapping):
            raise ValueError("Source segment and fixed-income coverage ledgers are required")
        def number(value, name):
            result = _finite(value)
            if result is None:
                raise ValueError(name + " is missing or nonfinite")
            return result
        rows_missing = composition.get("rows_missing_weight")
        conflicts = composition.get("conflicting_identity_group_count")
        if type(rows_missing) is not int or type(conflicts) is not int or rows_missing < 0 or conflicts < 0:
            raise ValueError("Source missing-weight/conflict counts are required")
        reported = number(composition.get("reported_weight_sum"), "Reported NAV")
        included = number(composition.get("net_weight_included"), "Included NAV")
        residual = number(composition.get("unreported_or_residual_nav_weight"), "Unreported NAV")
        fixed = number(coverage.get("fixed_income_net_nav_weight"), "Fixed-income NAV")
        covered = number(coverage.get("covered_nav_weight"), "Covered bond NAV")
        uncovered = number(coverage.get("uncovered_fixed_income_nav_weight"), "Uncovered fixed-income NAV")
        ledger = {}
        for name, row in segments.items():
            if not isinstance(name, str) or not isinstance(row, Mapping):
                raise ValueError("Malformed source segment ledger")
            net, long, short = (number(row.get(key), name + " " + key)
                                for key in ("net_weight", "gross_long", "gross_short"))
            if long < 0 or short > 0 or abs(net - long - short) > RATE_SCOPE_NAV_TOLERANCE:
                raise ValueError("Inconsistent signed source segment ledger")
            ledger[name] = {"net": net, "long": long, "short": short}
        def total(names, field="net"):
            return math.fsum(row[field] for name, row in ledger.items() if name in names)
        known_names = _BULLET_RATE_SEGMENTS | _CASH_RATE_SEGMENTS | _SPECIALIZED_RATE_SEGMENTS
        noncash_short = -math.fsum(row["short"] for name, row in ledger.items()
                                  if name not in _CASH_RATE_SEGMENTS)
        cash_short = -total(_CASH_RATE_SEGMENTS, "short")
        unmodeled_gross = math.fsum(row["long"] - row["short"] for name, row in ledger.items()
                                  if name not in _BULLET_RATE_SEGMENTS | _CASH_RATE_SEGMENTS)
        cash = total(_CASH_RATE_SEGMENTS)
        source_fixed = total(_BULLET_RATE_SEGMENTS | _SPECIALIZED_RATE_SEGMENTS)
        segment_total = math.fsum(row["net"] for row in ledger.values())
        out.update(fixed_income_nav_weight=fixed, covered_bond_nav_weight=covered,
                   uncovered_fixed_income_nav_weight=uncovered, cash_like_nav_weight=cash,
                   noncash_short_gross_nav_weight=noncash_short,
                   cash_liability_gross_nav_weight=cash_short,
                   unmodeled_nonbullet_or_nonrate_gross_nav_weight=unmodeled_gross,
                   unreported_nav_weight=residual, reported_nav_weight=reported,
                   source_rows_missing_weight=rows_missing, source_identity_conflicts=conflicts)
        if rows_missing or conflicts:
            raise ValueError("Missing source weights or conflicting identities prohibit a whole-fund rate proxy")
        if (abs(reported - 1.) > RATE_SCOPE_REPORTED_NAV_ROUNDING_TOLERANCE
                or abs(residual) > RATE_SCOPE_REPORTED_NAV_ROUNDING_TOLERANCE
                or abs(residual - (1. - included)) > RATE_SCOPE_NAV_TOLERANCE
                or abs(included - reported) > RATE_SCOPE_NAV_TOLERANCE
                or abs(segment_total - included) > RATE_SCOPE_NAV_TOLERANCE):
            raise ValueError("Unreconciled or unreported NAV cannot be assigned a bond duration")
        if (covered <= 0 or fixed <= 0 or uncovered < -RATE_SCOPE_NAV_TOLERANCE
                or abs(covered + uncovered - fixed) > RATE_SCOPE_NAV_TOLERANCE
                or abs(source_fixed - fixed) > RATE_SCOPE_NAV_TOLERANCE):
            raise ValueError("Fixed-income and covered-bond source NAV do not reconcile")
        if noncash_short > RATE_SCOPE_NAV_TOLERANCE:
            raise ValueError("Noncash short positions require a separate signed multi-channel model")
        if cash_short > RATE_SCOPE_CASH_LIABILITY_LIMIT + RATE_SCOPE_NAV_TOLERANCE:
            raise ValueError("Cash borrowing exceeds the declared source-class liability tolerance")
        if unmodeled_gross > RATE_SCOPE_NAV_TOLERANCE or set(ledger) - known_names:
            raise ValueError("Equity, commodity, unresolved or specialized bond NAV cannot be replaced by a whole-fund bond proxy")
        out.update(status="DECLARED_FIXED_INCOME_AND_CASH_PROXY", whole_fund_rate_proxy_allowed=True,
                   reason="Only recorded fixed-income NAV may receive the covered bonds' average duration; recorded cash-like NAV is separate.")
    except (ValueError, TypeError, OverflowError) as exc:
        out["reason"] = str(exc)
    return out


def _count(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


class FundCompositionError(ValueError):
    """Rows or provenance cannot support a signed composition view."""


def _finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value):
    return str(value or "").strip()


SOURCE_IDENTIFIER_KINDS = ("isin", "figi", "us_code")


def _luhn_digits_ok(digits):
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _isin_check(value):
    """ISO 6166 check digit: letters expand to two digits, then Luhn."""
    return _luhn_digits_ok("".join(str(int(ch, 36)) for ch in value))


def _cusip_check(value):
    total = 0
    for i, ch in enumerate(value[:8]):
        v = int(ch) if ch.isdigit() else ord(ch) - 55
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10 == int(value[8])


def identifier_evidence(kind, value):
    """What an identifier string can and cannot establish without a security master.

A source identifier with a valid check digit shows the provider supplied a
well-formed code; it does not show which instrument the code denotes. Only a
caller-supplied authoritative registry can do that, so no tier here is
"authoritative".
    """
    if kind == "isin":
        return "SOURCE_IDENTIFIER_CHECK_DIGIT_VALID" if _isin_check(value) else "SOURCE_IDENTIFIER_CHECK_DIGIT_INVALID"
    if kind == "us_code":
        return "SOURCE_IDENTIFIER_CHECK_DIGIT_VALID" if _cusip_check(value) else "SOURCE_IDENTIFIER_CHECK_DIGIT_INVALID"
    if kind == "figi":
        return "SOURCE_IDENTIFIER_SYNTAX_ONLY"
    if kind == "ticker_and_name":
        return "TICKER_AND_NAME_FALLBACK"
    return "ROW_HASH_FALLBACK"


def _identity(row):
    """Strongest syntactically valid source identifier; never a bare ticker when one exists."""
    for kind, pattern in (("isin", _ISIN), ("figi", _FIGI), ("us_code", _CUSIP)):
        value = _text(row.get(kind)).upper()
        if value and pattern.fullmatch(value):
            return kind, value
    name = _text(row.get("constituent_name")).upper()
    ticker = _text(row.get("constituent_ticker")).upper()
    if name or ticker:
        return "ticker_and_name", ticker + "|" + name
    return "row_hash", hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]


def _segment(row, fund_symbol):
    asset = _text(row.get("asset_class")).upper()
    security = _text(row.get("security_type")).upper()
    name = _text(row.get("constituent_name")).upper()
    ticker = _text(row.get("constituent_ticker")).upper()
    if "TBA" in name or "TBA" in security:
        return "AGENCY_MBS_TBA_FORWARD"
    if security.startswith("MBS"):
        return "AGENCY_MBS_POOL"
    if "CMBS" in security:
        return "CMBS"
    if security.startswith("ABS"):
        return "ABS"
    specialized = _specialized_bond_segment(row)
    if specialized:
        return specialized
    if security == "US GOVERNMENT":
        return "US_TREASURY"
    if "MONEY MARKET" in security or "MONEY MARKET" in name or "CASH FUNDS" in name:
        return "MONEY_MARKET_FUND"
    if asset in {"SHORT_TERM", "CASH"} or ticker in {"USD", "CASH"} or name in {"CASH", "USD CASH"}:
        return "CASH_AND_EQUIVALENTS"
    # Open-end is a legal fund structure, not an economic cash classification.
    # An equity mutual fund must not disappear into the unchanged-cash sleeve;
    # absent a positive source class, keep its NAV unresolved.
    if "OPEN-END FUND" in security:
        return "EQUITY" if asset == "EQUITY" else "UNCLASSIFIED"
    if asset in {"FIXED INCOME", "DEBT"}:
        return "CORPORATE_AND_OTHER_BONDS"
    if "GOLD" in name and fund_symbol == "GLD":
        return "PHYSICAL_GOLD"
    if asset == "EQUITY" or security.startswith(("COMMON", "ORDINARY")):
        return "EQUITY"
    return "UNCLASSIFIED"


def _specialized_bond_segment(row):
    """Recognized non-bullet structures must not re-enter via missing terms.

    The source's explicit structure, rather than the fund ticker, determines
    this exclusion. A bond with unknown coupon/maturity may use the declared
    remaining fixed-income assumption; a known swap, TIPS or floating-rate
    holding requires a different response channel and cannot use that escape.
    """
    asset = _text(row.get("asset_class")).upper()
    security = _text(row.get("security_type")).upper()
    name = _text(row.get("constituent_name")).upper()
    ticker = _text(row.get("constituent_ticker")).upper()
    words = security + " " + name
    if (asset not in {"FIXED INCOME", "DEBT"} and security != "US GOVERNMENT"
            and not re.search(r"\b(?:BOND|TREASURY|INTEREST RATE)\b", security)):
        return None
    if re.search(r"\b(?:FUTURES?|SWAPS?|OPTIONS?|FORWARDS?|DERIVATIVES?)\b", words):
        return "FIXED_INCOME_DERIVATIVE"
    if ticker.startswith("TII") or "TIPS" in words or "INFLATION" in words:
        return "INFLATION_LINKED_BONDS"
    if "FLOATING" in words or "VARIABLE RATE" in words or re.search(r"\bFRN\b", words):
        return "FLOATING_RATE_BONDS"
    if "CONVERTIBLE" in words or "CALLABLE" in words:
        return "CONVERTIBLE_OR_CALLABLE_BONDS"
    return None


def _maturity(year, month, day):
    year = int(year)
    if year < 100:
        year += 2000
    try:
        return date(year, int(month), int(day))
    except ValueError:
        return None


def _bullet_terms(row, instrument_terms=None):
    """Coupon (percent) and final maturity from the source ticker, else the name.

``instrument_terms`` maps a row's ISIN to terms acquired from a primary
instrument source (see ``treasury_terms``); it is consulted only when the row
itself carries no machine-readable terms, and only for fixed-coupon bullet
securities that the source itself classified as such.
    """
    security = (_text(row.get("security_type")) + " " + _text(row.get("constituent_name"))).upper()
    if any(word in security for word in _NO_BULLET_DURATION) or _specialized_bond_segment(row):
        return None
    if _text(row.get("constituent_ticker")).upper().startswith("TII"):
        return None
    for text, pattern, basis in ((_text(row.get("constituent_ticker")).upper(), _TICKER_BOND, "SOURCE_TICKER"),
                                 (_text(row.get("constituent_name")).upper(), _NAME_BOND, "SOURCE_NAME")):
        match = pattern.search(text)
        if match:
            maturity = _maturity(match["y"], match["m"], match["d"])
            coupon = float(match["coupon"])
            if maturity is not None and 0 <= coupon <= 20:
                return {"coupon_percent": coupon, "maturity": maturity, "terms_basis": basis}
    acquired = (instrument_terms or {}).get(_text(row.get("isin")).upper())
    if acquired and acquired.get("bullet_fixed_coupon") is True:
        maturity = date.fromisoformat(acquired["maturity"])
        coupon = float(acquired["coupon_percent"])
        if 0 <= coupon <= 20:
            return {"coupon_percent": coupon, "maturity": maturity, "terms_basis": "PRIMARY_SOURCE_INSTRUMENT_TERMS",
                    "terms_source_sha256": acquired.get("source_sha256"),
                    "terms_transcription_sha256": acquired.get("transcription_sha256")}
    return None


def _add_months(day, months):
    total = day.year * 12 + day.month - 1 + months
    year, month = divmod(total, 12)
    month += 1
    for candidate in (day.day, 30, 29, 28):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    raise FundCompositionError("Invalid coupon schedule")


def _cash_flows(settlement, maturity, coupon_percent):
    """Semiannual bullet schedule after settlement, per 100 par (ACT/365.25 times)."""
    flows, step = [], 0
    while True:
        when = _add_months(maturity, -6 * step)
        if when <= settlement:
            break
        flows.append(((when - settlement).days / 365.25, coupon_percent / 2 + (100.0 if step == 0 else 0.0)))
        step += 1
        if step > 240:
            raise FundCompositionError("Coupon schedule exceeds 120 years")
    return sorted(flows)


def _price(flows, y):
    return math.fsum(cf / (1 + y / 2) ** (2 * t) for t, cf in flows)


def _modified_duration(flows, y):
    price = _price(flows, y)
    macaulay = math.fsum(t * cf / (1 + y / 2) ** (2 * t) for t, cf in flows) / price
    return macaulay / (1 + y / 2)


def _yield_from_price(flows, price):
    low, high = -0.05, 0.40
    if not _price(flows, high) <= price <= _price(flows, low):
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if _price(flows, mid) > price:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _bond_sensitivity(row, settlement, instrument_terms=None):
    terms = _bullet_terms(row, instrument_terms)
    if terms is None:
        return None
    if terms["maturity"] <= settlement:
        return {"status": "MATURED_OR_SETTLING", **terms}
    flows = _cash_flows(settlement, terms["maturity"], terms["coupon_percent"])
    market_value, par = _finite(row.get("market_value")), _finite(row.get("shares_held"))
    result = {"maturity": terms["maturity"].isoformat(), "coupon_percent": terms["coupon_percent"],
              "terms_basis": terms["terms_basis"], "years_to_maturity": flows[-1][0]}
    if terms.get("terms_source_sha256"):
        result["terms_source_sha256"] = terms["terms_source_sha256"]
    if terms.get("terms_transcription_sha256"):
        result["terms_transcription_sha256"] = terms["terms_transcription_sha256"]
    if market_value is not None and par is not None and par != 0 and market_value * par > 0:
        price = 100.0 * market_value / par
        y = _yield_from_price(flows, price) if 20.0 <= price <= 200.0 else None
        if y is not None:
            duration = _modified_duration(flows, y)
            return {**result, "status": "PRICE_IMPLIED_YIELD", "price_per_100_par": price,
                    "yield": y, "modified_duration": duration,
                    "modified_duration_interval": [duration, duration]}
        result["price_per_100_par"] = price
    # No usable par/price: a duration interval over a declared yield domain.
    durations = [_modified_duration(flows, y) for y in ASSUMED_YIELD_DOMAIN]
    return {**result, "status": "DECLARED_YIELD_DOMAIN",
            "modified_duration": _modified_duration(flows, sum(ASSUMED_YIELD_DOMAIN) / 2),
            "modified_duration_interval": [min(durations), max(durations)]}


def selected_rows_from_holdings(holdings):
    """Raw constituent rows of the selected batch, verified against stored page hashes.

    ``holdings`` is a normalized snapshot produced by the original Massive
    adapter. Raw pages are read from its recorded paths only when each byte
    hash matches; the selected (effective, processed) batch is reused unchanged.
    """
    metadata = holdings.get("metadata") or {}
    pages = metadata.get("pages") or []
    batch = metadata.get("selected_batch")
    if not pages or not isinstance(batch, list) or len(batch) != 2:
        raise FundCompositionError("Raw page provenance or the selected batch is missing")
    rows = []
    for page in pages:
        path = Path(str(page.get("raw_path", "")))
        if not path.is_file():
            raise FundCompositionError("A recorded raw holdings page is not available")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != page.get("sha256"):
            raise FundCompositionError("A raw holdings page does not match its recorded hash")
        results = json.loads(body).get("results")
        if not isinstance(results, list):
            raise FundCompositionError("A raw holdings page has no result list")
        rows.extend(results)
    symbol = holdings.get("fund_symbol")
    selected = [r for r in rows if isinstance(r, dict) and r.get("composite_ticker") == symbol
                and [r.get("effective_date"), r.get("processed_date")] == batch]
    # Exact duplicate rows across overlapping pages are one record (same rule
    # as the original adapter); distinct rows are all kept.
    unique, seen = [], set()
    for row in selected:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if len(unique) != metadata.get("selected_raw_row_count", len(unique)) - metadata.get("exact_duplicate_rows_removed", 0):
        raise FundCompositionError("Raw selected rows differ from the recorded selected batch")
    return unique


def analyze_fund_rows(rows: Iterable[Mapping], fund_symbol: str, *, as_of: str | None = None,
                      fund_name_terms: Iterable[str] = (),
                      fund_identifiers: Mapping[str, str] | None = None,
                      instrument_terms: Mapping[str, Mapping] | None = None) -> dict:
    """Signed, identity-grouped composition of one fund batch (fractions of fund NAV).

``fund_identifiers`` maps identifier kinds (isin, figi, us_code) to the fund's
own codes from an authoritative source. Without a same-kind code, a row that
carries the fund's ticker stays an unresolved collision: its mass is kept in a
separate segment, never excluded and never assigned a bond duration.
    """
    fund_symbol = _text(fund_symbol).upper()
    rows = [dict(r) for r in rows]
    if not rows:
        raise FundCompositionError("No constituent rows")
    dates = {(_text(r.get("effective_date")), _text(r.get("processed_date"))) for r in rows}
    if len(dates) != 1:
        raise FundCompositionError("Rows from more than one dated batch cannot be combined")
    effective = as_of or next(iter(dates))[0]
    try:
        settlement = date.fromisoformat(effective)
    except ValueError as exc:
        raise FundCompositionError("Invalid effective date") from exc
    terms = [t.upper() for t in fund_name_terms if t]
    own_ids = {str(k): _text(v).upper() for k, v in (fund_identifiers or {}).items() if _text(v)}
    groups = defaultdict(list)
    missing_weight, records = 0, []
    for row in rows:
        weight = _finite(row.get("weight"))
        if weight is None:
            missing_weight += 1
            continue
        kind, value = _identity(row)
        ticker = _text(row.get("constituent_ticker")).upper()
        name = _text(row.get("constituent_name")).upper()
        self_label = ticker == fund_symbol
        evidence = identifier_evidence(kind, value)
        if self_label:
            # A row is the fund itself when its identifier equals the fund's
            # authoritative code or its name is the fund's name; it is another
            # instrument only when a valid same-kind fund code differs. A
            # syntactically valid code alone cannot tell the two apart.
            own = own_ids.get(kind)
            if (own and own == value) or any(term in name for term in terms) or kind in {"ticker_and_name", "row_hash"}:
                self_status = "CONFIRMED_OR_UNRESOLVED_SELF_REFERENCE"
            elif own and evidence == "SOURCE_IDENTIFIER_CHECK_DIGIT_VALID":
                self_status = "TICKER_FIELD_COLLISION_RESOLVED_BY_AUTHORITATIVE_FUND_IDENTIFIER"
            else:
                self_status = "UNRESOLVED_TICKER_FIELD_COLLISION_NO_AUTHORITATIVE_FUND_IDENTIFIER"
        else:
            self_status = None
        segment = ("UNRESOLVED_SELF_TICKER_COLLISION"
                   if self_status == "UNRESOLVED_TICKER_FIELD_COLLISION_NO_AUTHORITATIVE_FUND_IDENTIFIER"
                   else _segment(row, fund_symbol))
        record = {"identity_kind": kind, "identity": value, "identifier_evidence": evidence, "weight": weight,
                  "segment": segment, "self_reference": self_status, "row": row}
        records.append(record)
        groups[(kind, value)].append(record)

    resolved = []
    group_reviews = defaultdict(int)
    examples = defaultdict(list)
    conflicts = []
    for (kind, value), members in sorted(groups.items()):
        weights = [m["weight"] for m in members]
        member_segments = sorted({m["segment"] for m in members})
        member_terms = {json.dumps({k: str(v) for k, v in (_bullet_terms(m["row"], instrument_terms) or {}).items()}, sort_keys=True)
                        for m in members if m["segment"] in {"US_TREASURY", "CORPORATE_AND_OTHER_BONDS"}}
        conflicting = len(member_segments) > 1 or len(member_terms) > 1
        if len(members) == 1:
            review = "SINGLE_ROW"
        elif conflicting:
            review = "CONFLICTING_METADATA_SAME_IDENTITY"
        elif any(w < 0 for w in weights) and any(w > 0 for w in weights):
            review = "OFFSETTING_LONG_SHORT_SAME_IDENTITY"
        else:
            # Summed mechanically; the provider does not say these are
            # independent lots rather than a duplicated record.
            review = "MULTIPLE_ROWS_SAME_IDENTITY_ADDITIVE"
        group_reviews[review] += 1
        if review != "SINGLE_ROW" and len(examples[review]) < 3:
            examples[review].append({"identity_kind": kind, "identity": value,
                                     "description": _text(members[0]["row"].get("constituent_name")),
                                     "row_weights": weights, "net_weight": math.fsum(weights)})
        if conflicting:
            conflicts.append({"identity_kind": kind, "identity": value, "member_segments": member_segments,
                              "distinct_bullet_terms": len(member_terms), "net_weight": math.fsum(weights),
                              "gross_weight": math.fsum(abs(w) for w in weights)})
        excluded = any(m["self_reference"] == "CONFIRMED_OR_UNRESOLVED_SELF_REFERENCE" for m in members)
        # Order-invariant: a group whose rows disagree is not classified by
        # whichever row came first; its mass stays visible as unresolved.
        resolved.append({"identity_kind": kind, "identity": value, "members": members,
                         "net_weight": math.fsum(weights), "group_review": review,
                         "excluded_self_reference": excluded,
                         "segment": "CONFLICTING_IDENTITY_METADATA" if conflicting else member_segments[0]})

    # A repeated ticker is split into separate securities only by distinct
    # source identifiers; ticker/name fallbacks do not distinguish anything.
    by_ticker = defaultdict(set)
    for record in records:
        by_ticker[_text(record["row"].get("constituent_ticker")).upper()].add(
            (record["identity_kind"], record["identity"], record["identifier_evidence"]))
    repeated_ticker_groups = {t: ids for t, ids in by_ticker.items() if len(ids) > 1 and t}
    repeated_review = defaultdict(int)
    for ids in repeated_ticker_groups.values():
        tiers = {tier for _, _, tier in ids}
        if tiers == {"SOURCE_IDENTIFIER_CHECK_DIGIT_VALID"}:
            repeated_review["DISTINCT_SOURCE_IDENTIFIERS_CHECK_DIGIT_VALID"] += 1
        elif all(kind in SOURCE_IDENTIFIER_KINDS for kind, _, _ in ids):
            repeated_review["DISTINCT_SOURCE_IDENTIFIERS_NOT_ALL_CHECK_DIGIT_VALID"] += 1
        else:
            repeated_review["NOT_DISTINGUISHED_BY_SOURCE_IDENTIFIER"] += 1

    included = [g for g in resolved if not g["excluded_self_reference"]]
    long_rows = [m["weight"] for g in included for m in g["members"] if m["weight"] > 0]
    short_rows = [m["weight"] for g in included for m in g["members"] if m["weight"] < 0]
    net = math.fsum(g["net_weight"] for g in included)
    segments = defaultdict(lambda: {"net_weight": 0.0, "gross_long": 0.0, "gross_short": 0.0, "identities": 0})
    for group in sorted(included, key=lambda g: (g["identity_kind"], g["identity"])):
        seg = segments[group["segment"]]
        seg["identities"] += 1
        for member in group["members"]:
            seg["net_weight"] += member["weight"]
            seg["gross_long" if member["weight"] > 0 else "gross_short"] += member["weight"]

    # Interest-rate sensitivity of machine-readable bullet bonds only.
    covered, covered_dw, covered_dw_low, covered_dw_high = 0.0, 0.0, 0.0, 0.0
    statuses = defaultdict(float)
    for group in included:
        if group["segment"] not in {"US_TREASURY", "CORPORATE_AND_OTHER_BONDS"}:
            continue
        for member in group["members"]:
            sensitivity = _bond_sensitivity(member["row"], settlement, instrument_terms)
            if sensitivity is None or "modified_duration" not in sensitivity:
                statuses["NO_MACHINE_READABLE_BULLET_TERMS"] += member["weight"]
                continue
            statuses[sensitivity["status"]] += member["weight"]
            if sensitivity.get("terms_basis") == "PRIMARY_SOURCE_INSTRUMENT_TERMS":
                statuses["TERMS_FROM_PRIMARY_INSTRUMENT_SOURCE"] += member["weight"]
            covered += member["weight"]
            covered_dw += member["weight"] * sensitivity["modified_duration"]
            low, high = sensitivity["modified_duration_interval"]
            covered_dw_low += member["weight"] * (low if member["weight"] >= 0 else high)
            covered_dw_high += member["weight"] * (high if member["weight"] >= 0 else low)
    fixed_income_net = math.fsum(v["net_weight"] for k, v in segments.items()
                                 if k not in {"CASH_AND_EQUIVALENTS", "MONEY_MARKET_FUND", "PHYSICAL_GOLD", "EQUITY",
                                              "UNCLASSIFIED", *UNRESOLVED_SEGMENTS})
    unresolved_net = math.fsum(v["net_weight"] for k, v in segments.items() if k in UNRESOLVED_SEGMENTS)
    unresolved_gross = math.fsum(v["gross_long"] - v["gross_short"] for k, v in segments.items() if k in UNRESOLVED_SEGMENTS)
    coverage = {
        "fixed_income_net_nav_weight": fixed_income_net,
        "covered_nav_weight": covered,
        "uncovered_fixed_income_nav_weight": fixed_income_net - covered,
        "unresolved_classification_net_nav_weight": unresolved_net,
        "unresolved_classification_gross_nav_weight": unresolved_gross,
        "duration_available": covered > 0,
        "interpretation": ("Fixed-income NAV without machine-readable bullet terms has unknown duration. Rows whose "
                           "identity or classification is unresolved are not assumed to be bonds or cash; their mass "
                           "is reported separately and counted as possibly rate-sensitive."),
    }
    rate = None
    if covered > 0:
        price_implied = statuses.get("PRICE_IMPLIED_YIELD", 0.0) >= covered - 1e-12
        rate = {
            "status": "SOURCE_PRICE_IMPLIED" if price_implied else "PARTLY_OR_WHOLLY_DECLARED_YIELD_DOMAIN",
            "covered_nav_weight": covered,
            "uncovered_fixed_income_nav_weight": fixed_income_net - covered,
            "coverage_status_weights": dict(statuses),
            "covered_average_modified_duration": covered_dw / covered,
            "covered_contribution_to_fund_duration": covered_dw,
            "covered_contribution_interval": [covered_dw_low, covered_dw_high],
            "fund_return_per_bp_parallel_rise_covered_part": -covered_dw / 10000.0,
            "method": ("Bullet semiannual cash flows from the source coupon and maturity; yield solved from "
                       "market value per par when both are reported (treated as a full price), else a duration "
                       "interval over the declared yield domain " + str(list(ASSUMED_YIELD_DOMAIN)) + ". "
                       "Mortgage, TBA, ABS, CMBS and inflation-linked rows are not given a bullet duration."),
            "limits": ["Uncovered fixed-income weight has unknown duration and is not set to zero in any total.",
                       "Parallel-shift first-order sensitivity: convexity, curve shape, spread and prepayment are omitted.",
                       "Coupon and maturity are parsed from source text fields, not a security master."],
        }
        if statuses.get("TERMS_FROM_PRIMARY_INSTRUMENT_SOURCE"):
            rate["limits"][-1] = ("Coupon and maturity are parsed from source text fields, except rows marked "
                                  "PRIMARY_SOURCE_INSTRUMENT_TERMS, whose terms come from the hashed primary "
                                  "instrument record matched by ISIN-derived CUSIP.")
    reported = math.fsum(m["weight"] for m in records)
    return {
        "schema": SCHEMA, "fund_symbol": fund_symbol, "effective_date": effective,
        "rows": len(rows), "rows_missing_weight": missing_weight,
        "identities": len(resolved),
        "reported_weight_sum": reported,
        "gross_long_weight": math.fsum(long_rows), "gross_short_weight": math.fsum(short_rows),
        "net_weight_included": net,
        "unreported_or_residual_nav_weight": 1.0 - net,
        "residual_interpretation": ("Reported positions exceed NAV; the negative residual is unidentified net "
                                    "liabilities or a reporting basis difference, not verified and not removed."
                                    if net > 1.0 + 1e-9 else
                                    "Unreported NAV share; not renormalized and not assigned to any segment."),
        "segments": {k: dict(v) for k, v in sorted(segments.items())},
        "identity_group_reviews": dict(group_reviews),
        "identity_group_examples": {k: v for k, v in examples.items()},
        "repeated_ticker_groups": len(repeated_ticker_groups),
        "repeated_ticker_groups_all_distinguished_by_identifier": all(
            all(kind in SOURCE_IDENTIFIER_KINDS for kind, _, _ in ids) for ids in repeated_ticker_groups.values()),
        "repeated_ticker_group_review": dict(repeated_review),
        "identifier_evidence_rows": dict(sorted(_count(r["identifier_evidence"] for r in records).items())),
        "identifiers_authoritatively_verified": False,
        "identifier_evidence_scope": ("Check digits show a well-formed provider code, not which security it denotes; "
                                      "no security master was consulted."),
        "conflicting_identity_groups": conflicts[:20],
        "conflicting_identity_group_count": len(conflicts),
        "fixed_income_duration_coverage": coverage,
        "self_reference_rows": [{"status": r["self_reference"], "identity_kind": r["identity_kind"],
                                 "identity": r["identity"], "description": _text(r["row"].get("constituent_name")),
                                 "weight": r["weight"]} for r in records if r["self_reference"]],
        "excluded_self_reference_weight": math.fsum(g["net_weight"] for g in resolved if g["excluded_self_reference"]),
        "interest_rate_sensitivity": rate,
        "original_validator_flags_unchanged": True,
        "rows_deleted_clipped_or_renormalized": False,
        "scope": "Signed composition of one dated provider batch as fractions of fund NAV; not a security-master "
                 "reconciliation, not a point-in-time publication record, and not a forecast.",
    }


def analyze_holdings_snapshot(holdings: Mapping, *, fund_name_terms: Iterable[str] = (),
                              fund_identifiers: Mapping[str, str] | None = None,
                              instrument_terms: Mapping[str, Mapping] | None = None) -> dict:
    rows = selected_rows_from_holdings(holdings)
    result = analyze_fund_rows(rows, holdings.get("fund_symbol"), fund_name_terms=fund_name_terms,
                               fund_identifiers=fund_identifiers, instrument_terms=instrument_terms)
    result["original_snapshot_reliable_flag"] = holdings.get("reliable")
    result["original_quality_issues"] = list((holdings.get("metadata") or {}).get("quality_issues", []))
    return result


IMMATERIAL_FIXED_INCOME_SHARE = 0.001


def uncovered_bond_symbols(rate_sensitivity: Mapping | None) -> set[str]:
    """Known fixed-income ETFs with no usable duration, never an equity-beta proxy.

    An unresolved ETF with no signed composition is *not* silently classified
    as a bond. A provider-classified bond ETF, however, remains unpriceable
    through the duration channel even when its constituent page is missing.
    """
    evidence = rate_sensitivity or {}
    return ({row["symbol"] for row in evidence.get("positions_with_unknown_duration_only", [])}
            | set(evidence.get("known_bond_etf_without_duration_symbols", [])))


def portfolio_rate_sensitivity(portfolio, compositions: Mapping[str, Mapping],
                               *, instrument_exposures: Mapping | None = None) -> dict:
    """Dollars per +1bp parallel rise from covered bullet bonds, by original position.

    Each original ETF position is valued once. The constituent duration explains
    the fund's own rate channel; it is never added to another ETF component.
    """
    rows, total, low, high, uncovered = [], 0.0, 0.0, 0.0, 0.0
    unknown_rows, no_composition = [], []
    for position in portfolio.positions:
        comp = compositions.get(position.symbol) or {}
        rate = comp.get("interest_rate_sensitivity")
        if not rate:
            coverage = comp.get("fixed_income_duration_coverage")
            if coverage:
                # Wholly uncovered fixed income and unresolved rows stay unknown.
                share = coverage["fixed_income_net_nav_weight"] + max(coverage["unresolved_classification_gross_nav_weight"], 0.0)
                if share:
                    uncovered += position.weight * share
                # A trace of bonds (IEMG on 2026-10-29: 0.00016% of NAV) does not make an
                # equity fund a bond fund; the share stays in the uncovered total above.
                if share >= IMMATERIAL_FIXED_INCOME_SHARE:
                    unknown_rows.append({"symbol": position.symbol, "basis": "NO_MACHINE_READABLE_BULLET_TERMS",
                                         "unknown_duration_fund_nav_weight": share,
                                         "portfolio_weight_with_unknown_duration": position.weight * share})
            elif position.asset_type != "stock" and position.asset_type != "cash" and position.weight:
                no_composition.append({"symbol": position.symbol, "portfolio_weight": position.weight,
                                       "status": comp.get("status", "NO_SIGNED_COMPOSITION"),
                                       "classification": "UNKNOWN_COMPOSITION_RATE_EXPOSURE_NOT_CLASSIFIED"})
            continue
        dollars = -portfolio.notional * position.weight * rate["covered_contribution_to_fund_duration"] / 10000.0
        interval = sorted(-portfolio.notional * position.weight * x / 10000.0 for x in rate["covered_contribution_interval"])
        rows.append({"symbol": position.symbol, "usd_per_bp_parallel_rise": dollars,
                     "usd_per_bp_interval": interval, "basis": rate["status"],
                     "dollar_sensitivity_binding": {
                         "schema": RATE_DOLLAR_BINDING_SCHEMA, "portfolio_notional_usd": portfolio.notional,
                         "position_symbol": position.symbol, "position_weight": position.weight,
                         "position_asset_type": position.asset_type, "currency": portfolio.currency},
                     "covered_fund_nav_weight": rate["covered_nav_weight"],
                     "uncovered_fixed_income_fund_nav_weight": rate["uncovered_fixed_income_nav_weight"],
                     "fund_composition_scope_inputs": rate_model_scope_inputs(comp),
                     "whole_fund_rate_model_scope": review_rate_model_scope(comp)})
        total += dollars
        low += interval[0]
        high += interval[1]
        unresolved = max(((comp.get("fixed_income_duration_coverage") or {})
                          .get("unresolved_classification_gross_nav_weight") or 0.0), 0.0)
        uncovered += position.weight * (rate["uncovered_fixed_income_nav_weight"] + unresolved)
    provider_bonds = {row["symbol"] for row in (instrument_exposures or {}).get("positions", [])
                      if row.get("instrument_class") == "bond_etf"
                      and row.get("classification_status") in {
                          "PROVIDER_TYPED_ETF_ASSET_CLASS", "ADAPTER_REPORTED_ETF_CLASS_UNVERIFIED"}}
    covered_symbols = {row["symbol"] for row in rows}
    typed_without_duration = sorted(provider_bonds - covered_symbols)
    return {"positions": rows, "portfolio_notional_usd": portfolio.notional,
            "original_positions": [{"symbol": p.symbol, "weight": p.weight, "asset_type": p.asset_type}
                                   for p in portfolio.positions],
            "usd_per_bp_parallel_rise_covered": total if rows else None,
            "usd_per_bp_interval_covered": [low, high] if rows else None,
            "portfolio_weight_with_unknown_bond_duration": uncovered,
            "positions_with_unknown_duration_only": unknown_rows,
            "known_bond_etf_without_duration_symbols": typed_without_duration,
            "positions_without_signed_composition": no_composition,
            "scope": "First-order parallel-yield sensitivity of covered bullet bonds inside bond ETFs. It maps a yield "
                     "move to dollars; it does not say how far yields move after any event."}
