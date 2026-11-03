"""Evidence-limited issuer-overlap view over the unchanged funded NAV ledger.

Identical stock tickers in independently accepted ETF holdings can be grouped.
The sole cross-listing rule below is constrained to an accepted ETF local
Taiwan Semiconductor constituent (2330, ISIN TW0002330008) and a direct TSM ADR.
The issuer's own FAQ identifies these two listings: 
https://investor.tsmc.com/english/faq

Neither grouping changes the security ledger, adds a second dollar of NAV,
or asserts identical security prices, currency exposure, or returns. A source
record is not a provider signature or a security-master verification.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .portfolio import ExposureReport, Portfolio

_TSMC_ISSUER_KEY = "TSMC:TW0002330008"
_TSMC_PRIMARY_SOURCE = "https://investor.tsmc.com/english/faq"
_ALLOWED_STATUSES = frozenset({"direct", "look_through"})
_CIK_SCOPE = "MASSIVE_REFERENCE_SAME_SEC_CIK_NOT_INDEPENDENT_SECURITY_MASTER"


def _accepted_cik_siblings(issuer_siblings: Mapping[str, Any] | None, portfolio: Portfolio) -> dict[str, str]:
    """Map US tickers to a registrant CIK, only for directly held stocks with a self-consistent record.

    A record is used only when it names a ten-digit CIK, lists the direct
    holding itself among the registrant's tickers, and is marked complete.
    """
    if not isinstance(issuer_siblings, Mapping):
        return {}
    direct_stocks = {p.symbol for p in portfolio.positions if p.asset_type == "stock"}
    ticker_to_cik: dict[str, str] = {}
    for symbol, record in issuer_siblings.items():
        if symbol not in direct_stocks or not isinstance(record, Mapping):
            continue
        cik = record.get("cik")
        rows = record.get("tickers")
        if (not isinstance(cik, str) or not cik.isdigit() or len(cik) != 10
                or record.get("complete") is not True or not isinstance(rows, (list, tuple))):
            continue
        tickers = {row.get("ticker") for row in rows if isinstance(row, Mapping) and isinstance(row.get("ticker"), str)}
        if symbol not in tickers:
            continue
        for ticker in tickers:
            if ticker_to_cik.get(ticker, cik) != cik:
                return {}  # conflicting registrant records: fail closed for all
            ticker_to_cik[ticker] = cik
    return ticker_to_cik


def _accepted_local_tsmc(
    snapshots: Mapping[str, Any],
    fund_symbol: str,
    source_weight: float,
    original_etf_weight: float,
) -> bool:
    """Cross-reference exact local ISIN with an *already accepted* ledger row."""
    record = snapshots.get(fund_symbol)
    if not isinstance(record, Mapping) or record.get("fund_symbol") != fund_symbol:
        return False
    holdings = record.get("holdings")
    if not isinstance(holdings, (list, tuple)):
        return False
    matches = [holding for holding in holdings if isinstance(holding, Mapping)
               and holding.get("symbol") == "2330"]
    if len(matches) != 1:
        return False
    holding = matches[0]
    identifiers = holding.get("identifiers")
    if not isinstance(identifiers, Mapping) or identifiers.get("isin") != "TW0002330008":
        return False
    if (holding.get("asset_type") != "stock"
            or holding.get("source_ticker", "2330") != "2330"):
        return False
    fraction = holding.get("weight")
    if (not isinstance(fraction, (float, int)) or isinstance(fraction, bool)
            or not math.isfinite(fraction) or not 0. <= fraction <= 1.):
        return False
    # Prevent an unrelated or inconsistent raw row from rebinding an
    # accepted entry's source identity. Ledger itself is never rewritten.
    return math.isclose(source_weight, original_etf_weight * fraction,
                        rel_tol=0., abs_tol=1e-11)


def audit_issuer_overlap(
    portfolio: Portfolio,
    ledger: ExposureReport,
    snapshots: Mapping[str, Any] | None = None,
    *,
    max_groups: int = 8,
    issuer_siblings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report additional concentration in the SAME original capital.

    A group only appears when at least two separate original portfolio
    instruments contribute an accepted stock slice. The local-vs-ADR issuer
    link is deliberately narrow and demands the exact local security ISIN.
    """
    if not isinstance(max_groups, int) or isinstance(max_groups, bool) or not 1 <= max_groups <= 64:
        raise ValueError("max_groups must be an integer between 1 and 64")
    if snapshots is not None and not isinstance(snapshots, Mapping):
        raise ValueError("snapshots must be a mapping")
    snapshots = snapshots or {}
    original = {position.symbol: position for position in portfolio.positions}
    ticker_to_cik = _accepted_cik_siblings(issuer_siblings, portfolio)
    # A CIK group is active only when at least two DISTINCT accepted security
    # tickers of that registrant appear among the stock ledger rows.
    cik_tickers: dict[str, set[str]] = defaultdict(set)
    for entry in ledger.entries:
        if entry.status in _ALLOWED_STATUSES and entry.asset_type == "stock" and entry.symbol in ticker_to_cik:
            cik_tickers[ticker_to_cik[entry.symbol]].add(entry.symbol)
    active_ciks = {cik for cik, tickers in cik_tickers.items() if len(tickers) >= 2}
    by_issuer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ledger_index, entry in enumerate(ledger.entries):
        if entry.status not in _ALLOWED_STATUSES or entry.asset_type != "stock":
            continue
        key = "EXACT_SYMBOL:" + entry.symbol
        identity_basis = "ACCEPTED_SAME_STOCK_TICKER_NOT_INDEPENDENT_SECURITY_MASTER"
        if entry.symbol in ticker_to_cik and ticker_to_cik[entry.symbol] in active_ciks:
            key = "CIK:" + ticker_to_cik[entry.symbol]
            identity_basis = _CIK_SCOPE
        elif entry.status == "direct" and entry.symbol == "TSM":
            key = _TSMC_ISSUER_KEY
            identity_basis = "TSM_ADR_LISTING_IDENTIFIED_BY_ISSUER_PRIMARY_FAQ"
        elif (entry.status == "look_through" and entry.symbol == "2330"
              and entry.original_symbol in original
              and original[entry.original_symbol].asset_type == "etf"
              and _accepted_local_tsmc(
                  snapshots, entry.original_symbol, entry.portfolio_weight,
                  original[entry.original_symbol].weight)):
            key = _TSMC_ISSUER_KEY
            identity_basis = "ETF_LOCAL_SECURITY_EXACT_TICKER_AND_ISIN_PLUS_ISSUER_FAQ"
        by_issuer[key].append({
            "ledger_index": ledger_index,
            "original_position": entry.original_symbol,
            "source_security_ticker": entry.symbol,
            "source_kind": entry.status,
            "portfolio_weight": entry.portfolio_weight,
            "identity_basis": identity_basis,
            "holdings_effective_date": entry.as_of if entry.status == "look_through" else None,
            "holdings_retrieved_at_utc": entry.retrieved_at if entry.status == "look_through" else None,
        })
    groups = []
    for key, components in by_issuer.items():
        origin_symbols = {component["original_position"] for component in components}
        if len(origin_symbols) < 2:
            continue
        direct = math.fsum(c["portfolio_weight"] for c in components if c["source_kind"] == "direct")
        lookthrough = math.fsum(c["portfolio_weight"] for c in components if c["source_kind"] == "look_through")
        total = math.fsum([direct, lookthrough])
        if not 0. < total <= portfolio.total_weight + 1e-10:
            continue
        cross_listing = key == _TSMC_ISSUER_KEY
        cik_group = key.startswith("CIK:")
        distinct_tickers = sorted({c["source_security_ticker"] for c in components})
        groups.append({
            "issuer_group_id": key,
            "scope": ("SAME_ISSUER_DIFFERENT_SECURITIES" if cross_listing else
                      "SAME_SEC_REGISTRANT_SHARE_CLASSES" if cik_group else
                      "SAME_ACCEPTED_SOURCE_SECURITY_TICKER"),
            "issuer_cik": key[4:] if cik_group else None,
            "distinct_security_tickers": distinct_tickers,
            "combined_portfolio_nav_fraction": total,
            "direct_portfolio_nav_fraction": direct,
            "etf_lookthrough_portfolio_nav_fraction": lookthrough,
            "contributing_original_positions": sorted(origin_symbols),
            "source_components": sorted(components, key=lambda c: (-c["portfolio_weight"], c["original_position"]))[:16],
            "source_component_count": len(components),
            "source_components_truncated": len(components) > 16,
            "identity_support": (
                "Exact source ETF 2330 ISIN TW0002330008, an accepted holding, and TSMC primary issuer FAQ; "
                "TSM is an ADR and 2330 an ordinary local share, NOT price-return-equivalent."
                if cross_listing else
                "US-listed share classes reported under one SEC registrant CIK by the provider's reference data "
                "(for example Alphabet GOOGL and GOOG); different securities with different prices, votes and "
                "possibly different returns, NOT an independent security master."
                if cik_group else
                "Identical reported stock ticker in separate accepted portfolio slices; "
                "not an independently certified same-issuer cross-listing."
            ),
            "issuer_primary_reference": _TSMC_PRIMARY_SOURCE if cross_listing else None,
            "new_nav_created": 0.,
            "authoritative_ledger_modified": False,
        })
    groups.sort(key=lambda group: (-group["combined_portfolio_nav_fraction"], group["issuer_group_id"]))
    return {
        "status": "SOURCE_BOUNDED_ISSUER_OVERLAP" if groups else "NO_SOURCE_SUPPORTED_CROSS_POSITION_STOCK_OVERLAP",
        "groups": groups[:max_groups],
        "tsmc_source_identified_slices": list(by_issuer.get(_TSMC_ISSUER_KEY, [])),
        "sec_cik_registrant_groups": sorted(active_ciks),
        "issuer_sibling_records_used": sorted(symbol for symbol in (issuer_siblings or {})
                                              if isinstance(issuer_siblings, Mapping)
                                              and any(ticker_to_cik.get(symbol) == cik for cik in active_ciks)),
        "all_overlapping_group_count": len(groups),
        "groups_truncated": len(groups) > max_groups,
        "capital_accounting": "Issuer overlap is an annotation of the SAME accepted NAV entries; do not sum group NAV with original position NAV.",
        "holdings_identity_assurance": "ETF holdings accepted by the existing dated ledger; not proof of provider authorship or live fund composition.",
        "cross_listing_return_warning": "A local ordinary share and US ADR have different units, trading currencies, venues and potential price returns.",
        "authoritative_ledger_modified": False,
        "new_nav_created": 0.,
    }
