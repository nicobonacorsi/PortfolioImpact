"""Bounded two-venue geopolitical search with receipt-checked, explicit no-fit cards.

Search relevance is not semantic equivalence. Broad Taiwan/export-control events
remain unmatched; a separately source-defined alternative retains its exact rules.
The existing read-only transport, timeouts, retries and request caps are reused.
No network activity occurs at import or in replay. No quote is transferred to a
broad economic event, and Gamma summary prices are only indicative references.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from .adapters import DataAccessError, KALSHI_BASE, KalshiClient, _ReadOnlyClient, normalize_market, normalize_quote, parse_time
from .tariff_scenarios import REVIEWED_RISK as TARIFF_REVIEWED_RISK, REVIEWED_TITLE as TARIFF_REVIEWED_TITLE, is_tariff_risk
from .event_subject_intent import explicit_non_tariff_agreement_title

GAMMA_BASE = "https://gamma-api.polymarket.com"
CATEGORIES = ("World", "Politics", "Companies", "Economics")
# Kalshi documents Commodities as an additional (not necessarily primary)
# discovery category. Do not spend the extra series request for a portfolio
# whose targets contain no energy supply risk.
ENERGY_CATEGORIES = ("Commodities",)
ENERGY_FAMILIES = frozenset({"HORMUZ_ENERGY", "OPEC_SUPPLY", "RUSSIA_ENERGY_SANCTIONS"})
SECTOR_RISK_FAMILIES = frozenset({"US_DRUG_PRICING", "FDA_DRUG_APPROVAL",
                                 "BANK_CREDIT_STRESS", "BANK_CAPITAL_RULES"})


def _series_categories(families):
    """One additional documented category read only for relevant energy targets."""
    return CATEGORIES + ENERGY_CATEGORIES if ENERGY_FAMILIES.intersection(families) else CATEGORIES


QUERIES = {
    # "China invade Taiwan" follows the contracts' own wording without a year: on
    # 2026-10-23 "Taiwan invasion" (3 pages) and "China invade Taiwan 2027" both
    # missed "Will China invade Taiwan by end of 2026?".
    "TAIWAN": ("Taiwan blockade", "Taiwan military", "Taiwan invasion", "China invade Taiwan"),
    "US_CHIP_EXPORT_POLICY": ("chip export", "H200 China", "export controls"),
    # On 2026-10-24 "China tariff" returned the bracket event on the general US rate
    # on Chinese goods; "tariff" (10 events) adds the other tariff contracts, which
    # the rules review rejects with a reason.
    "US_TARIFFS": ("China tariff", "tariff"),
    # These are retrieval prompts, never evidence that a contract is logically
    # equivalent to the broad shipping, energy or sanctions target.
    "HORMUZ_ENERGY": ("Strait of Hormuz", "Hormuz shipping", "Iran oil blockade"),
    "OPEC_SUPPLY": ("OPEC oil production", "oil supply cut", "OPEC meeting"),
    "RUSSIA_ENERGY_SANCTIONS": ("Russian oil sanctions", "Russia oil exports", "Ukraine Russia energy"),
    # Source-specific identity remains essential: a generic FDA contract cannot
    # be priced as a particular holding's clinical outcome without drug evidence.
    "US_DRUG_PRICING": ("Medicare drug price negotiation", "drug price cap", "pharmaceutical pricing"),
    "FDA_DRUG_APPROVAL": ("FDA drug approval", "FDA clinical trial", "FDA drug application"),
    "BANK_CREDIT_STRESS": ("bank failure", "bank credit losses", "bank deposit insurance"),
    "BANK_CAPITAL_RULES": ("bank capital requirements", "Basel III endgame", "bank capital regulation"),
}
MAX_SERIES_PER_FAMILY = 4
MAX_MARKET_PAGES = 2
MAX_GAMMA_PAGES = 2
# Gamma caps limit_per_type at 50 (a probe on 2026-10-23 returned the same 50
# events for 100). At 8 per page "Taiwan invasion" (35 results) and two export
# queries (82 each) still had pages after the third; at 50, three pages read
# every result of all three. Order is stable: smaller pages are its prefix.
GAMMA_RESULTS_PER_PAGE = 50
MAX_CANDIDATES_PER_VENUE = 6
# A bracket family is one event: all its members must reach the rules review.
CANDIDATES_PER_VENUE_BY_FAMILY = {"US_TARIFFS": 12}
MAX_RECEIPT_AGE_SECONDS = 300
# The expanded live profile permits 60 Kalshi and 150 Polymarket reads (Gamma
# queries are read to their last page). Retain that complete bounded capture;
# a smaller replay cap could reject it after acquisition had completed.
MAX_SOURCE_RECORDS = 220


from .provider_access import AuditedClientMixin, classify_http_status

class GeopoliticalSearchError(ValueError):
    """Invalid bounded input or mismatched local source evidence."""


class GammaSearchClient(AuditedClientMixin, _ReadOnlyClient):
    allowed_hosts = {"gamma-api.polymarket.com"}


class AuditedKalshiClient(AuditedClientMixin, KalshiClient):
    """The original Kalshi client; failures carry a fixed failure_class."""


# A transient failure that outlasts the client's own retries gets one more
# attempt after a pause. On 2026-10-23 (r9) one Kalshi read failed three
# attempts within four seconds and succeeded when repeated minutes later.
TRANSIENT_FAILURES = frozenset({"PROVIDER_HTTP_RATE_LIMIT", "PROVIDER_HTTP_SERVER_ERROR",
                                "NETWORK_TIMEOUT", "NETWORK_FAILURE"})
TRANSIENT_RETRY_PAUSE_SECONDS = 10
MAX_TRANSIENT_PAUSES = 2


def _failure_class(exc) -> str:
    """Fixed label from the exception's own class or its sanitized message; never server text."""
    label = getattr(exc, "failure_class", None)
    if label:
        return label
    if not isinstance(exc, DataAccessError):
        return "SCHEMA_REJECTED"
    message = str(exc)
    code = re.search(r"HTTP (\d{3})", message)
    if code:
        return classify_http_status(int(code.group(1)))
    if "network request failed" in message:
        return "NETWORK_FAILURE"
    if "invalid JSON" in message:
        return "SCHEMA_REJECTED"
    return "UNCLASSIFIED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _family(event: dict) -> str | None:
    if explicit_non_tariff_agreement_title(str(event.get("title", ""))):
        return None
    if event.get("family") in QUERIES and event.get("family") not in {"US_TARIFFS", "US_CHIP_EXPORT_POLICY"}:
        return event["family"]
    text = " ".join(str(event.get(k, "")) for k in ("event_id", "title", "description", "question_we_care_about", "family")).lower()
    if "taiwan" in text:
        return "TAIWAN"
    if re.search(r"\bhormuz\b|\bstrait\b.{0,12}\bhormuz\b", text):
        return "HORMUZ_ENERGY"
    if (re.search(r"\b(?:russia|russian|ukraine|ukrainian)\b", text)
            and re.search(r"\b(?:sanctions?|oil|energy|pipeline|embargo|crude|lng|export restriction|war)\b", text)):
        return "RUSSIA_ENERGY_SANCTIONS"
    if re.search(r"\bopec\b|\b(?:oil|crude)\b.{0,30}\b(?:production|output|supply cut)\b", text):
        return "OPEC_SUPPLY"
    if re.search(r"\b(?:medicare|drug pricing|prescription drug|pharma pricing)\b", text) and re.search(
            r"negotiat|price|cost|reimburse|medicare|cap", text):
        return "US_DRUG_PRICING"
    if re.search(r"\bfda\b", text) and re.search(r"drug|approval|trial|application|therapy", text):
        return "FDA_DRUG_APPROVAL"
    if re.search(r"\b(?:bank|banking|depositor|deposit insurance)\b", text) and re.search(
            r"failure|liquidity|credit losses|defaults?|bankrun|bank run|deposit|collapse", text):
        return "BANK_CREDIT_STRESS"
    if re.search(r"\b(?:bank|banking|basel)\b", text) and re.search(r"capital|basel|regulat|stress test", text):
        if re.search(r"basel|regulat|capital (?:requirement|adequacy|ratio|rule|standard)|stress test", text):
            return "BANK_CAPITAL_RULES"
    if event.get("family") in QUERIES:
        return event["family"]
    if is_tariff_risk(text):
        return "US_TARIFFS"
    if re.search(r"export[ _-]control|chip[ _-]export|semiconductor|ai.chip", text):
        return "US_CHIP_EXPORT_POLICY"
    return None


def _targets(events) -> list[dict]:
    if not isinstance(events, list) or len(events) > 30:
        raise GeopoliticalSearchError("Expected at most 30 event objects")
    result, seen = [], set()
    for event in events:
        if not isinstance(event, dict):
            raise GeopoliticalSearchError("Event must be an object")
        family = _family(event)
        if family is None:
            continue
        identifier = event.get("event_id")
        question = event.get("question_we_care_about") or event.get("description") or event.get("title")
        if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(question, str):
            raise GeopoliticalSearchError("Event identity and question must be unambiguous")
        seen.add(identifier)
        row = {"event_id": identifier, "question_we_care_about": question,
               "event_date": event.get("event_date"), "family": family}
        if event.get("definition_origin") == REVIEWED_FAMILY_ORIGIN:
            row.update(title=event["title"], definition_origin=REVIEWED_FAMILY_ORIGIN,
                       exposure_basis=deepcopy(event.get("exposure_basis")))
        queries = event.get("search_queries")
        if isinstance(queries, list) and all(isinstance(q, str) and 2 <= len(q) <= 120 for q in queries) and len(queries) <= 6:
            row["search_queries"] = list(queries)
        result.append(row)
    return result


REVIEWED_FAMILY_ORIGIN = "REVIEWED_FAMILY_FROM_DOCUMENTED_EXPOSURE"
_REVIEWED_FAMILY_TITLES = {
    "TAIWAN": "Taiwan: PRC blockade, quarantine or military action disrupting shipping or chips",
    "US_CHIP_EXPORT_POLICY": "US rule or action newly restricting AI-chip exports to China",
    "US_TARIFFS": TARIFF_REVIEWED_TITLE,
}
# The chip-supply families of the original list; the live run also passes US_TARIFFS.
REVIEWED_CHIP_FAMILIES = ("TAIWAN", "US_CHIP_EXPORT_POLICY")
REVIEWED_SEARCH_FAMILIES = REVIEWED_CHIP_FAMILIES + ("US_TARIFFS",)


# Reviewed statements from held issuers' latest annual reports. A documented link
# only puts the reviewed risk on the search list; it never sets a shock or a Q.
_NVDA_10K = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
_AAPL_10K = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
DOCUMENTED_ISSUER_LINKS = {
    "NVDA": {
        "TAIWAN": ('NVIDIA 10-K, fiscal 2026, Item 1: "We utilize foundries, such as Taiwan Semiconductor '
                   'Manufacturing Company Limited, or TSMC".', _NVDA_10K),
        "US_CHIP_EXPORT_POLICY": ('NVIDIA 10-K, fiscal 2026, Item 1: "In April 2025, the USG informed us that it '
                                  'requires a license for export to China".', _NVDA_10K),
    },
    "AAPL": {
        "TAIWAN": ("Apple 10-K, fiscal 2025, Item 1A: a significant majority of its manufacturing is performed by "
                   "outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan "
                   "and Vietnam.", _AAPL_10K),
        "US_TARIFFS": ('Apple 10-K, fiscal 2025, Item 1A: "Beginning in the second quarter of 2025, new tariffs were '
                       'announced on imports to the U.S."; its manufacturing partners are located primarily in China '
                       'mainland, India, Japan, South Korea, Taiwan and Vietnam.', _AAPL_10K),
    },
    "AMD": {
        "TAIWAN": ("AMD 2025 Form 10-K: TSMC produces its microprocessor and GPU wafers at 7nm or smaller nodes; foundry disruption can impair supply.",
                   "https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm"),
        "US_CHIP_EXPORT_POLICY": ("AMD 2025 Form 10-K: the April 2025 US export license requirement affected MI308 products; future sales depend on licensing and import rules.",
                                  "https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm"),
    },
    "TSM": {
        "TAIWAN": ("TSMC 2025 Form 20-F, Item 3: principal production facilities and most operating revenue are in Taiwan; military disruption is an identified operational risk.",
                   "https://investor.tsmc.com/sites/ir/sec-filings/2025_20F%20Report.pdf"),
    },
    "MU": {
        "TAIWAN": ('Micron 10-K, fiscal 2025, Item 1A: "A majority of our DRAM production output in 2025 was from our '
                   'fabrication facilities in Taiwan".',
                   "https://www.sec.gov/Archives/edgar/data/723125/000072312525000028/mu-20250828.htm"),
    },
}


def documented_issuer_links(holding_rows) -> dict[str, list[dict]]:
    """Held stocks, directly or inside an accepted fund, whose annual report documents a reviewed family's channel."""
    from .geopolitical_channels import SOURCES
    links: dict[str, list[dict]] = {}
    for row in holding_rows:
        if (row.get("asset_type") != "stock" or row.get("identity_status") == "unresolved_etf"
                or row.get("instrument_class") == "unsupported" or not (row.get("portfolio_weight") or 0) > 0):
            continue
        for family, (statement, url) in DOCUMENTED_ISSUER_LINKS.get(row.get("instrument_symbol"), {}).items():
            links.setdefault(family, []).append({
                "original_symbol": row.get("original_symbol"), "instrument_symbol": row["instrument_symbol"],
                "portfolio_weight": row["portfolio_weight"], "identity_status": row.get("identity_status"),
                "statement": statement, "evidence_reference": url,
                "source_published_on": SOURCES.get(row.get("instrument_symbol"), {}).get("published_on")})
    return links


def reviewed_family_events(proposed_events, links_by_family, *, start: str, horizon_days: int,
                           families=REVIEWED_CHIP_FAMILIES) -> list[dict]:
    """Reviewed supply-chain risks that the portfolio's documented exposure always puts on the list.

    Like the next FOMC and CPI, these do not depend on what the model proposed:
    a family is searched whenever a held issuer's annual report documents its
    channel (``documented_issuer_links``). A family the model already proposed is
    not added twice. The definition is the reviewed family text, so the
    probability bound never follows model wording.
    """
    from .contract_scenarios import REVIEWED_RISKS
    covered = {_family(event) for event in proposed_events if isinstance(event, dict)}
    first = datetime.fromisoformat(start).date()
    end = (first + timedelta(days=horizon_days)).isoformat()
    added = []
    for family in families:
        available_links = [link for link in links_by_family.get(family, [])
                           if not link.get("source_published_on") or link["source_published_on"] <= first.isoformat()]
        if family in covered or not available_links:
            continue
        added.append({
            "event_id": family.lower().replace("_", "-") + "-reviewed-family-by-" + end,
            "title": _REVIEWED_FAMILY_TITLES[family] + " (by " + end + ")",
            "event_date": end, "family": family,
            "question_we_care_about": f"Between {first.isoformat()} and {end}: "
                                      + (TARIFF_REVIEWED_RISK if family == "US_TARIFFS" else REVIEWED_RISKS[family]),
            "definition_origin": REVIEWED_FAMILY_ORIGIN, "exposure_basis": deepcopy(available_links),
        })
    return added


def _gamma_event_text(event: dict) -> str:
    markets = event.get("markets") if isinstance(event.get("markets"), list) else []
    return " ".join([str(event.get("title", ""))] + [str(m.get("question", "")) for m in markets if isinstance(m, dict)])


def _relevance(text: str, family: str) -> int:
    """Deterministic lexical retrieval rank, independent of prices and P&L."""
    text = text.lower()
    if family == "TAIWAN":
        if "taiwan" not in text:
            return 0
        score = 10 + 30 * bool(re.search(r"blockad|invad|invasion|military (?:clash|encounter)|armed", text))
        score += 10 * bool(re.search(r"travel (?:warning|advisory)|level 4", text))
        score -= 8 * bool(re.search(r"election|mayor|earnings|mention|visit|recogniz", text))
        return score
    if family == "US_TARIFFS":
        if not re.search(r"\btariffs?\b|\bimport dut(?:y|ies)\b|\bsection (?:232|301)\b", text):
            return 0
        score = 10 + 25 * bool(re.search(r"\bchina\b|\bchinese\b", text))
        score += 10 * bool(re.search(r"\brate\b|in effect|increase|raise", text))
        score -= 15 * bool(re.search(r"dividend|rebate|refund|\bstake\b", text))
        return max(1, score)
    if family == "HORMUZ_ENERGY":
        if not re.search(r"\bhormuz\b", text):
            return 0
        return 10 + 25 * bool(re.search(r"shipping|strait|oil|blockad|close|traffic|war|attack", text))
    if family == "RUSSIA_ENERGY_SANCTIONS":
        if not re.search(r"\b(?:russia|russian|ukraine|ukrainian)\b", text):
            return 0
        if not re.search(r"sanction|oil|gas|energy|pipeline|export|embargo|ceasefire|war", text):
            return 0
        return 10 + 20 * bool(re.search(r"sanction|embargo|oil|pipeline|export", text))
    if family == "OPEC_SUPPLY":
        if not re.search(r"\bopec\b|\b(?:oil|crude)\b.{0,25}\b(?:output|production|supply)\b", text):
            return 0
        if not re.search(r"\b(?:oil|crude|production|output|supply|quota|barrels?|cut|increase)\b", text):
            return 0
        return 10 + 20 * bool(re.search(r"cut|increase|quota|production|output|supply", text))
    if family == "US_DRUG_PRICING":
        return 0 if not re.search(r"\b(?:drug|pharmaceutical|medicare|prescription)\b", text) or not re.search(
            r"pric|negotiat|reimburse|cost|cap", text) else 10 + 20 * bool(re.search(r"medicare|negotiat|price", text))
    if family == "FDA_DRUG_APPROVAL":
        return 0 if not re.search(r"\bfda\b", text) or not re.search(r"drug|approv|trial|therap|application", text) else 25
    if family == "BANK_CREDIT_STRESS":
        return 0 if not re.search(r"bank|deposit|lender", text) or not re.search(
            r"failur|credit|liquid|deposit|collapse|default|stress", text) else 25
    if family == "BANK_CAPITAL_RULES":
        return 0 if not re.search(r"\b(?:bank|banking|basel)\b", text) or not re.search(
            r"basel|regulat|capital (?:requirement|adequacy|ratio|rule|standard)|stress test", text) else 25
    if not re.search(r"\b(?:chips?|semiconductors?|nvidia|h200|h20)\b|export control|gain ai", text):
        return 0
    score = 10
    score += 25 * bool(re.search(r"export|restrict|ban\b|licen[cs]|gain ai", text))
    score += 10 * bool(re.search(r"deliver|sell|sale", text))
    score += 5 * ("china" in text)
    score -= 18 * bool(re.search(r"earnings|revenue|margin|market cap|job|groq|rental|stake in", text))
    return max(1, score)


def _record(meta: dict, *, root: Path, venue: str, kind: str, query: str, page: int, families: list[str]) -> dict:
    raw = Path(meta["raw_path"]).resolve()
    if not raw.is_relative_to(root):
        raise GeopoliticalSearchError("Client receipt must be inside the chosen evidence directory")
    return {"venue": venue, "kind": kind, "query": query, "page": page, "families": families,
            "raw_file": raw.relative_to(root).as_posix(),
            "receipt_file": raw.with_suffix(".meta.json").relative_to(root).as_posix(),
            "source_url": meta["url"], "source_response_sha256": meta["sha256"],
            "retrieved_at": meta["retrieved_at"], "fixture": meta["fixture"]}


def _validated_response(document: dict, kind: str) -> tuple[list[dict], bool]:
    """Reject malformed successful responses before admitting their records.

    In particular, null nested arrays and truthy strings are not pagination
    evidence. The same gate applies to capture and replay so a failed page
    remains a source/schema failure rather than an empty or completed search.
    """
    if not isinstance(document, dict):
        raise GeopoliticalSearchError("Provider response must be an object")
    key = {"series": "series", "markets": "markets", "public_search": "events"}.get(kind)
    if key is None:
        raise GeopoliticalSearchError("Unknown provider response kind")
    rows = document.get(key, [] if kind == "public_search" else None)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise GeopoliticalSearchError("Provider result array is malformed")
    if kind == "public_search":
        pagination = document.get("pagination")
        if not isinstance(pagination, dict) or type(pagination.get("hasMore")) is not bool:
            raise GeopoliticalSearchError("Provider pagination must contain a boolean hasMore")
        for event in rows:
            markets = event.get("markets", [])
            if not isinstance(markets, list) or any(not isinstance(row, dict) for row in markets):
                raise GeopoliticalSearchError("Provider event market array is malformed")
        return rows, pagination["hasMore"]
    cursor = document.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise GeopoliticalSearchError("Provider cursor must be a string")
    if kind == "markets":
        for row in rows:
            if any(key in row and not isinstance(row[key], str)
                   for key in ("title", "rules_primary", "rules_secondary")):
                raise GeopoliticalSearchError("Provider market question and rules must be strings")
    return rows, bool(cursor)


def _read_record(root: Path, record: dict) -> tuple[dict, dict]:
    paths = []
    for field in ("raw_file", "receipt_file"):
        name = record.get(field)
        # Captures store POSIX-style relative filenames even when replayed on
        # Windows.  Path(name).is_absolute() alone is platform-dependent:
        # Windows regards "/tmp/secret.json" as drive-relative, rather than an
        # absolute POSIX path.  Reject both path grammars before filesystem IO.
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise GeopoliticalSearchError("Unsafe source evidence path")
        posix_name = PurePosixPath(name)
        windows_name = PureWindowsPath(name)
        if (posix_name.is_absolute() or windows_name.is_absolute()
                or bool(windows_name.drive) or ".." in posix_name.parts):
            raise GeopoliticalSearchError("Unsafe source evidence path")
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise GeopoliticalSearchError("Missing or unsafe source evidence file")
        paths.append(path)
    if paths[0].stat().st_size > 16_000_000 or paths[1].stat().st_size > 64_000:
        raise GeopoliticalSearchError("Oversized source evidence")
    body = paths[0].read_bytes()
    meta = json.loads(paths[1].read_text(encoding="utf-8"))
    document = json.loads(body)
    if not isinstance(document, dict) or not isinstance(meta, dict):
        raise GeopoliticalSearchError("Source evidence must be JSON objects")
    if (sha256(body).hexdigest() != record.get("source_response_sha256")
            or meta.get("sha256") != record.get("source_response_sha256")
            or meta.get("url") != record.get("source_url")
            or meta.get("retrieved_at") != record.get("retrieved_at")
            or meta.get("fixture") is not record.get("fixture")
            or meta.get("status") != 200 or meta.get("transport") != "https GET"):
        raise GeopoliticalSearchError("Source receipt/hash mismatch")
    url = urlsplit(meta["url"])
    expected_host = "external-api.kalshi.com" if record.get("venue") == "Kalshi" else "gamma-api.polymarket.com"
    if url.scheme != "https" or url.hostname != expected_host or url.username or url.password:
        raise GeopoliticalSearchError("Source receipt has an untrusted provider URL")
    params = parse_qs(url.query)
    kind = record.get("kind")
    expected = {("Kalshi", "series"): ("/trade-api/v2/series", "category"),
                ("Kalshi", "markets"): ("/trade-api/v2/markets", "series_ticker"),
                ("Polymarket", "public_search"): ("/public-search", "q")}.get((record.get("venue"), kind))
    if expected is None or url.path != expected[0] or params.get(expected[1]) != [record.get("query")]:
        raise GeopoliticalSearchError("Source query does not match its receipt")
    parse_time(meta["retrieved_at"])
    _validated_response(document, kind)
    return document, meta


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def _candidate(row: dict, event: dict | None, record: dict, now: str) -> dict | None:
    venue = record["venue"]
    if venue == "Kalshi":
        if not isinstance(row.get("ticker"), str) or not row["ticker"]:
            return None
        market = normalize_market(row, record["retrieved_at"])
        quote = normalize_quote(market, now=now)
        identifier, question = market.ticker, market.title
        rules = market.rules_primary + "\n" + market.rules_secondary
        end, is_open = market.close_time, market.status in ("active", "open")
        link = market.source_url
        interval = [quote.lower, quote.upper] if quote.usable else None
        quote_scope = "KALSHI_TOP_OF_BOOK_REFERENCE_NO_DEPTH_OR_UPDATE_TIME_CERTIFICATION"
        quote_reasons = [] if quote.usable else [quote.exclusion_reason]
        description = market.rules_primary
    else:
        identifier, question = str(row.get("id", "")), row.get("question")
        if not identifier or not isinstance(question, str):
            return None
        rules = row.get("description") or ""
        if not isinstance(rules, str):
            return None
        description = rules
        end = row.get("endDate")
        is_open = (row.get("active") is True and row.get("closed") is False
                   and row.get("archived") is False and row.get("acceptingOrders") is True
                   and row.get("enableOrderBook") is True
                   and row.get("umaResolutionStatus") not in ("resolved", "disputed"))
        slug = (event or {}).get("slug") or row.get("slug")
        link = "https://polymarket.com/event/" + str(slug) if re.fullmatch(r"[a-z0-9-]{1,160}", str(slug)) else None
        outcomes = row.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except ValueError:
                outcomes = None
        bid, ask = _number(row.get("bestBid")), _number(row.get("bestAsk"))
        interval = [bid, ask] if outcomes == ["Yes", "No"] and bid is not None and ask is not None and bid <= ask else None
        quote_scope = "INDICATIVE_GAMMA_FIRST_OUTCOME_YES_SUMMARY_NOT_CLOB_VERIFIED"
        quote_reasons = [] if interval is not None else ["INVALID_OR_UNSUPPORTED_GAMMA_QUOTE"]
    if not isinstance(rules, str) or not rules.strip():
        return None
    age = (parse_time(now) - parse_time(record["retrieved_at"])).total_seconds()
    if age < -5 or age > MAX_RECEIPT_AGE_SECONDS:
        interval = None
        quote_reasons.append("FUTURE_OR_STALE_RECEIPT")
    if record.get("fixture") is not False:
        interval = None
        quote_reasons.append("FIXTURE_NOT_LIVE_CAPTURE")
    try:
        time_active = parse_time(end) > parse_time(now)
    except (ValueError, TypeError, AttributeError):
        time_active = False
        quote_reasons.append("UNVERIFIED_CONTRACT_END")
    if not is_open or not time_active:
        interval = None
        quote_reasons.append("CONTRACT_NOT_OPEN_AT_EVALUATION")
    return {"venue": venue, "contract_id": identifier, "market_question": question,
            "rules_text": rules, "rules_sha256": sha256(rules.encode()).hexdigest(),
            "source_url": link, "contract_end_time": end,
            "market_open_at_evaluation": bool(is_open and time_active),
            "source": {k: record[k] for k in ("source_url", "source_response_sha256", "retrieved_at", "fixture")},
            "source_receipt_status": "LOCAL_RESPONSE_BYTES_AND_RECEIPT_VERIFIED_NOT_PROVIDER_SIGNATURE",
            "contract_quote_reference": {"interval": interval, "scope": quote_scope,
                "evaluated_at": now, "reasons": sorted(set(quote_reasons)),
                "physical_probability_P": None, "transferred_to_parent_event": False},
            "_search_text": question + " " + description}


def _tariff_scope_difference(target_text: str, question: str, rules: str) -> str:
    """Explain *this* tariff target, not a hard-coded China tariff hypothesis.

    A related contract is deliberately never equated to the parent event.
    Country, covered products, settlement action and time must each match.
    """
    target_text, question, rules = target_text.lower(), question.lower(), rules.lower()
    source = question + ' ' + rules
    if re.search(r'dividend|rebate|refund', question):
        return ('This contract settles a tariff dividend, rebate or refund, which '
                'is a fiscal transfer rather than the specified import-tariff action.')
    country_patterns = {'Mexico': r'\bmexic(?:o|an)\b',
                        'China': r'\bchin(?:a|ese)\b',
                        'Canada': r'\bcanad(?:a|ian)\b'}
    target_countries = {name for name, pattern in country_patterns.items()
                        if re.search(pattern, target_text)}
    contract_countries = {name for name, pattern in country_patterns.items()
                          if re.search(pattern, question)}
    if target_countries and contract_countries and not target_countries.intersection(contract_countries):
        return ('The target specifies imports from ' + ', '.join(sorted(target_countries)) +
                ', while this contract concerns ' + ', '.join(sorted(contract_countries)) +
                '; the affected countries and settlement tests differ.')
    product_groups = {'steel/aluminum': r'\bsteel\b|\balumin(?:um|ium)\b',
                      'machinery': r'\bmachinery\b|\bconstruction equipment\b',
                      'solar/batteries': r'\bsolar\b|\bbatter(?:y|ies)\b|\bphotovoltaic\b',
                      'electronics/chips': r'\belectronics?\b|\bsemiconductors?\b|\bchips?\b'}
    required = {name for name, pattern in product_groups.items() if re.search(pattern, target_text)}
    represented = {name for name, pattern in product_groups.items() if re.search(pattern, question)}
    if required and not (required & represented):
        return ('The target concerns tariff actions on ' + ', '.join(sorted(required)) +
                '; this contract does not specify those products as its settlement '
                'trigger. A general tariff rate is not an equivalent product action.')
    if 'usmca' in target_text and 'usmca' not in source:
        return ('A new import tariff and a formal USMCA withdrawal notice are '
                'different triggers; this contract does not establish both.')
    if 'rate' in question and re.search(r'on (?:dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov)|\bon \d{1,2} ', question):
        return ('This contract settles a stated tariff rate on one specified date; '
                'the target asks whether a new or increased tariff action occurs '
                'within its own window. Levels and actions are not equivalent.')
    return ('A tariff-related market is not sufficient: the target requires its '
            'specified country, products, government action, effective/announcement '
            'test and deadline. Exact settlement equivalence was not established.')


def _comparison(target: dict, candidate: dict) -> dict:
    card = deepcopy(candidate)
    score = _relevance(card.pop("_search_text"), target["family"])
    question = card["market_question"].lower()
    rules = card["rules_text"].lower()
    target_text = target["question_we_care_about"].lower()
    for action in ("blockade", "quarantine", "invasion", "export", "restriction"):
        if action in target_text and action in question:
            score += 10
    try:
        target_date = datetime.fromisoformat(str(target.get("event_date"))).date()
        distance = abs((parse_time(card["contract_end_time"]).date() - target_date).days)
    except (ValueError, TypeError, AttributeError):
        distance = None
    differences = []
    if target["family"] == "TAIWAN":
        if "blockad" in question:
            differences.append("The market requires its specifically defined blockade; the target also includes quarantine or armed action and economic supply interruption is not itself the settlement test.")
        elif re.search(r"invad|invasion", question):
            differences.append("The market requires an offensive intended to establish territorial control; a blockade, quarantine, military clash or supply shock need not satisfy it.")
        elif re.search(r"military (?:clash|encounter)|armed", question):
            differences.append("The market requires its defined military encounter; a nonviolent blockade/quarantine or supply interruption can fail that test.")
        elif re.search(r"travel|level 4", question):
            differences.append("A government travel advisory is a policy proxy; it is neither necessary nor sufficient for the target military/supply disruption.")
        else:
            differences.append("The market settles on a different Taiwan-related political or corporate action, not the target military/supply disruption.")
    elif target["family"] == "US_TARIFFS":
        differences.append(_tariff_scope_difference(target_text, question, rules))
    elif target["family"] in ENERGY_FAMILIES:
        differences.append(
            "This contract settles on its own specified military, supply, sanctions or policy outcome; "
            "a related energy-market question does not certify the portfolio's broader economic-loss event.")
        differences.append("Any transmission into oil prices and this portfolio remains uncalibrated here.")
    elif target["family"] in SECTOR_RISK_FAMILIES:
        differences.append("The market settles a specific regulatory, approval, credit or bank-capital predicate, "
                           "not this holding's entire sector risk; issuer-specific equivalence is unverified.")
        differences.append("Portfolio loss magnitude and any nontrivial source-defined implication remain unidentified.")
    else:
        if re.search(r"deliver|sell|sale", question):
            differences.append("Delivery, sales or licensing of a named chip is not the announcement of a new or tightened US export restriction; direction and covered products differ.")
        elif "china" in question and re.search(r"china(?:'s|’s)|chinese", question):
            differences.append("The market concerns a Chinese policy action; the target requires a US government restriction on advanced chips or semiconductors.")
        elif re.search(r"bill|act|law|gain ai", question):
            differences.append("Passage of this named legislation is narrower than all new US executive, regulatory or legislative export-control actions.")
        else:
            differences.append("The market's corporate, sector or policy outcome does not specify the same new US export-control action and affected products.")
    if target.get("event_date"):
        differences.append("Target horizon is " + str(target["event_date"]) + "; the market's full rules and end time " + str(card["contract_end_time"]) + " define its own window. No deadline equivalence is certified.")
    relevant_trigger = (bool(re.search(r"blockad|invad|invasion|military|travel|level 4", question)) if target["family"] == "TAIWAN"
                        else bool(re.search(r"\btariffs?\b", question + " " + rules)) if target["family"] == "US_TARIFFS"
                        else bool(_relevance(question + " " + rules, target["family"])) if target["family"] in ENERGY_FAMILIES | SECTOR_RISK_FAMILIES
                        else bool(re.search(r"export|restrict|ban\b|deliver|gain ai", question + " " + rules)))
    if not card["market_open_at_evaluation"]:
        differences.append("The contract is not open at the evaluation time; its price cannot serve as a current event probability.")
    card.update(question_we_care_about=target["question_we_care_about"],
        decision="RELATED_ONLY" if score > 0 and relevant_trigger and card["market_open_at_evaluation"] else "REJECT",
        difference=" ".join(differences), differences=differences,
        relevance_score=score, ranking_basis="SUBJECT_ACTION_RULES_ONLY_NEVER_QUOTE_OR_PNL",
        horizon_distance_days=distance,
        broad_macro_event_matched=False, event_pricing_measure_Q_interval=None,
        physical_probability_P=None,
        exact_source_alternative={"definition": "THIS_IDENTIFIED_CONTRACT_SETTLES_YES_UNDER_ITS_COMPLETE_RULES",
            "matching_decision": "ACCEPT_EXACT_SOURCE_CONTRACT_IDENTITY_ONLY",
            "venue": card["venue"], "contract_id": card["contract_id"],
            "rules_sha256": card["rules_sha256"], "question": card["market_question"],
            "same_as_parent": False, "economic_effect_identified": False})
    return card


def replay_geopolitical_search(bundle: dict, *, evaluated_at: str | None = None, data_dir: str | Path | None = None) -> dict:
    """Verify retained public response bytes and rebuild cards without HTTP calls."""
    if not isinstance(bundle, dict) or bundle.get("schema") != "castle.geopolitical_search.capture.v1":
        raise GeopoliticalSearchError("Unsupported geopolitical search capture")
    root = Path(data_dir or bundle["data_dir"]).resolve()
    now = evaluated_at or bundle["evaluated_at"]
    parse_time(now)
    targets = _targets(bundle["targets"])
    all_candidates, conflicts = {}, set()
    records = bundle.get("source_records")
    if not isinstance(records, list) or len(records) > MAX_SOURCE_RECORDS:
        raise GeopoliticalSearchError("Invalid bounded source record list")
    for record in records:
        doc, _ = _read_record(root, record)
        if record["kind"] == "series":
            continue
        if record["venue"] == "Kalshi":
            rows = [(row, None) for row in doc.get("markets", []) if isinstance(row, dict)]
        else:
            rows = [(row, event) for event in doc.get("events", []) if isinstance(event, dict)
                    for row in event.get("markets", []) if isinstance(row, dict)]
        for row, event in rows:
            candidate = _candidate(row, event, record, now)
            if candidate is None:
                continue
            key = (candidate["venue"], candidate["contract_id"])
            old = all_candidates.get(key)
            if old and old["rules_sha256"] != candidate["rules_sha256"]:
                conflicts.add(key)
            if old is None or old["source"]["retrieved_at"] < candidate["source"]["retrieved_at"]:
                all_candidates[key] = candidate
    reviewed = []
    for target in targets:
        cards = [_comparison(target, card) for key, card in all_candidates.items()
                 if key not in conflicts and _relevance(card["_search_text"], target["family"]) > 0]
        cards.sort(key=lambda card: (card["decision"] != "RELATED_ONLY", -card["relevance_score"],
            card["horizon_distance_days"] if card["horizon_distance_days"] is not None else 1000000,
            card["venue"], card["contract_id"]))
        cap = CANDIDATES_PER_VENUE_BY_FAMILY.get(target["family"], MAX_CANDIDATES_PER_VENUE)
        kept = [card for venue in ("Kalshi", "Polymarket")
                for card in [x for x in cards if x["venue"] == venue][:cap]]
        venues = {}
        for venue in ("Kalshi", "Polymarket"):
            checks = [row for row in bundle["search_ledger"] if row["venue"] == venue and target["family"] in row.get("families", [])]
            possible = [card for card in kept if card["venue"] == venue]
            successes = any(row["status"] == "OK" for row in checks)
            failures = any(row["status"] == "FAILED" for row in checks)
            last_pages = {}
            for row in checks:
                last_pages[(row["kind"], row["query"])] = row
            truncated = [row["query"] for row in last_pages.values()
                         if row.get("remaining_pages") and row.get("stop_reason") != "RELEVANCE_TAIL_NO_FAMILY_MATCH_ON_PAGE"]
            tail_stops = [row["query"] for row in last_pages.values()
                          if row.get("remaining_pages") and row.get("stop_reason") == "RELEVANCE_TAIL_NO_FAMILY_MATCH_ON_PAGE"]
            venues[venue] = {"status": ("PARTIAL_SEARCH" if successes and (failures or truncated) else "SEARCH_COMPLETED_WITHIN_BOUNDS" if successes else "SEARCH_FAILED" if failures else "NOT_SEARCHED"),
                "closest_candidate": possible[0] if possible else None,
                "decision": "NO_EXACT_FIT_WITHIN_SEARCHED_SCOPE" if possible else "NONE_FOUND_WITHIN_SCOPE" if successes else "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE",
                "successful_requests": sum(row["status"] == "OK" for row in checks),
                "failed_requests": sum(row["status"] == "FAILED" for row in checks),
                "queries_with_unfinished_pages": sorted(set(truncated)),
                "queries_stopped_at_relevance_tail": sorted(set(tail_stops)),
                "failure_classes": sorted({row.get("failure_class", "UNCLASSIFIED") for row in checks if row["status"] == "FAILED"})}
        available = any(value["successful_requests"] for value in venues.values())
        reviewed.append({**target, "decision": "NO_EXACT_FIT_WITHIN_SEARCHED_SCOPE" if available else "SEARCH_UNAVAILABLE_NOT_EVIDENCE_OF_ABSENCE",
            "candidates": kept, "closest_by_venue": venues,
            "event_pricing_measure_Q_interval": None, "physical_probability_P": None,
            "no_match_is_not_global_absence": True})
    return {"schema": "castle.geopolitical_search.review.v1", "evaluated_at": now,
        "capture_evaluated_at": bundle["evaluated_at"], "targets": reviewed,
        "search_ledger": deepcopy(bundle["search_ledger"]),
        "source_records": deepcopy(records), "verified_response_records": len(records),
        "conflicting_rules_contracts": [list(key) for key in sorted(conflicts)],
        "coverage": "BOUNDED_CATEGORY_AND_QUERY_SEARCH_NOT_ALL_LISTED_OR_POSSIBLE_MARKETS",
        "new_http_calls_in_replay": 0,
        "limitations": ["Local receipt integrity is not a provider signature.",
            "No broad event probability is inferred from merely related contracts.",
            "A narrower exact source question is a distinct event, not a repaired broad hypothesis.",
            "Receipt times are distinct from exchange quote-update times; Gamma references are indicative.",
            "Search failures and remaining pages prevent any claim of exhaustive absence."]}


def public_geopolitical_search_view(report: dict) -> dict:
    """Return a derived report without local evidence paths."""
    result = deepcopy(report)
    for record in result.get("source_records", []):
        record.pop("raw_file", None)
        record.pop("receipt_file", None)
    result.pop("data_dir", None)
    result.pop("capture_bundle_path", None)
    return result


def acquire_geopolitical_search(events: list[dict], *, data_dir: str | Path,
                                kalshi_client=None, gamma_client=None,
                                evaluated_at: str | None = None,
                                max_series_per_family: int = MAX_SERIES_PER_FAMILY,
                                max_market_pages: int = MAX_MARKET_PAGES,
                                max_gamma_pages: int = MAX_GAMMA_PAGES,
                                include_target_queries: bool = False) -> dict:
    """Explicitly acquire both public venues, save receipts, and return review + replay bundle.

    By default each family uses at most four Kalshi series and two market pages
    per series, and three or four fixed Gamma queries with at most two pages each. An explicit
    expanded execution profile may raise these page/series limits together and
    add each target's own proposed queries to the Gamma search; the clients'
    own request caps still bound the total and are never increased here.
    """
    if (any(type(value) is not int for value in (max_series_per_family, max_market_pages, max_gamma_pages))
            or not (1 <= max_series_per_family <= 12 and 1 <= max_market_pages <= 10 and 1 <= max_gamma_pages <= 10)
            or type(include_target_queries) is not bool):
        raise GeopoliticalSearchError("Search limits are outside the reviewed bounds")
    targets = _targets(events)
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    kalshi = kalshi_client or AuditedKalshiClient(root / "kalshi")
    gamma = gamma_client or GammaSearchClient(root / "polymarket")
    families = sorted({target["family"] for target in targets})
    ledger, records = [], []
    start_counts = {"Kalshi": kalshi.request_count, "Polymarket": gamma.request_count}

    pauses = []

    def get(client, url, *, venue, kind, query, page, kinds, retried=False):
        before = client.request_count
        try:
            if len(records) >= MAX_SOURCE_RECORDS:
                error = GeopoliticalSearchError("Retained source record budget exhausted")
                error.failure_class = "LOCAL_SOURCE_RECORD_BUDGET_EXHAUSTED"
                raise error
            data, meta = client._get(url)
            _, remaining = _validated_response(data, kind)
            records.append(_record(meta, root=root, venue=venue, kind=kind, query=query, page=page, families=kinds))
            ledger.append({"venue": venue, "query": query, "kind": kind, "page": page,
                "families": kinds, "status": "OK", "request_attempts": client.request_count - before,
                "source_url": meta["url"], "source_response_sha256": meta["sha256"],
                "retrieved_at": meta["retrieved_at"], "fixture": meta["fixture"],
                "remaining_pages": remaining})
            return data
        except (DataAccessError, GeopoliticalSearchError, ValueError, TypeError, KeyError) as exc:
            failure = _failure_class(exc)
            retry = failure in TRANSIENT_FAILURES and not retried and len(pauses) < MAX_TRANSIENT_PAUSES
            # The failed attempt stays in the ledger; only a read that fails after
            # the pause, or that is not transient, counts as a failed search read.
            ledger.append({"venue": venue, "query": query, "kind": kind, "page": page,
                "families": kinds, "status": "RETRIED_AFTER_PAUSE" if retry else "FAILED",
                "request_attempts": client.request_count - before,
                "reason": "PUBLIC_GET_OR_SOURCE_SCHEMA_FAILED_DETAILS_OMITTED",
                "failure_class": failure, "at": _now()})
            if not retry:
                return None
            pauses.append(TRANSIENT_RETRY_PAUSE_SECONDS)
            time.sleep(TRANSIENT_RETRY_PAUSE_SECONDS)
            return get(client, url, venue=venue, kind=kind, query=query, page=page, kinds=kinds, retried=True)

    series = {}
    if families:
        for category in _series_categories(families):
            doc = get(kalshi, KALSHI_BASE + "/series?" + urlencode({"category": category}),
                venue="Kalshi", kind="series", query=category, page=1, kinds=families)
            if doc:
                for row in doc.get("series", []):
                    if isinstance(row, dict) and re.fullmatch(r"[A-Z0-9]+", str(row.get("ticker", ""))):
                        series[row["ticker"]] = row
    selected = {}
    for family in families:
        ranked = []
        for ticker, row in series.items():
            # Broad discovery tags such as "Jobs & Economy" are not the event's
            # predicate and must not penalize a chip-delivery series as a jobs
            # market. Rank the provider-authored series title and identifier.
            score = _relevance(" ".join(str(row.get(k, "")) for k in ("title", "ticker")), family)
            if score:
                ranked.append((score, ticker))
        for _, ticker in sorted(ranked, key=lambda value: (-value[0], value[1]))[:max_series_per_family]:
            selected.setdefault(ticker, []).append(family)
    for ticker, kinds in sorted(selected.items()):
        params = {"series_ticker": ticker, "status": "open", "limit": 100, "mve_filter": "exclude"}
        seen_cursors = set()
        for page in range(1, max_market_pages + 1):
            doc = get(kalshi, KALSHI_BASE + "/markets?" + urlencode(params),
                venue="Kalshi", kind="markets", query=ticker, page=page, kinds=kinds)
            # A bounded unfiltered fallback exposes recently closed related
            # contracts as rejected evidence, never as live probabilities.
            if doc is not None and not doc.get("markets") and page == 1:
                params.pop("status", None)
                continue
            cursor = doc.get("cursor") if doc else None
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            params["cursor"] = cursor
    plans = {family: list(QUERIES[family]) for family in families}
    if include_target_queries:
        for target in targets:
            for query in target.get("search_queries", []):
                if query.lower() not in {q.lower() for q in plans[target["family"]]}:
                    plans[target["family"]].append(query)
    for family in families:
        for query in plans[family]:
            for page in range(1, max_gamma_pages + 1):
                params = {"q": query, "limit_per_type": GAMMA_RESULTS_PER_PAGE, "page": page, "events_status": "active",
                          "search_profiles": "false", "search_tags": "false", "keep_closed_markets": 0}
                doc = get(gamma, GAMMA_BASE + "/public-search?" + urlencode(params),
                    venue="Polymarket", kind="public_search", query=query, page=page, kinds=[family])
                if not doc or not doc.get("pagination", {}).get("hasMore"):
                    break
                # Gamma's search is fuzzy and relevance-ranked: "new chip export restrictions"
                # matched 8,126 events on 2026-10-30 because of the word "new". Once a later page
                # holds no event that the family's relevance rule accepts, the ranked tail is
                # not read further; the ledger records why the query stopped.
                # An empty page that still claims more results is not a tail; it stays unfinished.
                events = doc.get("events", [])
                if page > 1 and events and not any(_relevance(_gamma_event_text(event), family) for event in events):
                    ledger[-1]["stop_reason"] = "RELEVANCE_TAIL_NO_FAMILY_MATCH_ON_PAGE"
                    break
    now = evaluated_at or _now()
    bundle = {"schema": "castle.geopolitical_search.capture.v1", "targets": targets,
              "data_dir": str(root), "evaluated_at": now, "source_records": records,
              "search_ledger": ledger, "request_counts": {
                  "Kalshi": kalshi.request_count - start_counts["Kalshi"],
                  "Polymarket": gamma.request_count - start_counts["Polymarket"]},
              "existing_request_caps": {"Kalshi": kalshi.max_requests, "Polymarket": gamma.max_requests},
              "search_limits": {"max_series_per_family": max_series_per_family, "max_market_pages": max_market_pages,
                                "max_gamma_pages": max_gamma_pages, "include_target_queries": include_target_queries},
              "query_plan": plans}
    (root / "search_capture.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = replay_geopolitical_search(bundle)
    report["capture_bundle_path"] = str(root / "search_capture.json")
    report["request_counts"] = bundle["request_counts"]
    report["existing_request_caps"] = bundle["existing_request_caps"]
    report["search_limits"] = bundle["search_limits"]
    report["query_plan"] = bundle["query_plan"]
    return report
