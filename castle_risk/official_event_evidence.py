"""Read-only, bounded first-party evidence excerpts for live event forecasts.

ONLY pre-discovered government/central-bank public pages are fetched. No
credentials, user-supplied arbitrary URLs, redirects, JavaScript, or claims of
historical content availability. The resulting visible text is DATA, not trusted
instructions or a certified publication-time archive.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from typing import Callable

from .live_web_research import public_source_url

EVIDENCE_SCOPE = "DIRECT_PRIMARY_PAGE_TEXT_RETRIEVED_AT_ISSUE_TIME_NOT_FACT_CERTIFICATION"
PRIMARY_HOSTS = frozenset({
    "www.federalreserve.gov", "federalreserve.gov",
    "www.bls.gov", "bls.gov",
    "www.eia.gov", "eia.gov",
    "www.defense.gov", "defense.gov",
    "www.treasury.gov", "home.treasury.gov",
    "www.sec.gov", "sec.gov",
    "www.bea.gov", "bea.gov",
    "www.census.gov", "census.gov",
    "www.bis.gov", "bis.gov",
})
MAX_PRIMARY_FETCHES = 3
# R125: tiered first-party and specialized hosts for direct page text. The
# OFFICIAL tier is any government, central-bank, intergovernmental or rating
# agency host (suffix rule plus explicit hosts); the SPECIALIZED tier is a
# closed list of reputable financial/security reporting and research hosts.
# A tier describes the publisher class only; it is NOT fact certification.
SPECIALIZED_EVIDENCE_SCOPE = ("DIRECT_SPECIALIZED_SOURCE_PAGE_TEXT_RETRIEVED_AT_ISSUE_TIME_"
                              "NOT_FACT_CERTIFICATION")
OFFICIAL_HOST_SUFFIXES = (".gov", ".mil", ".gov.uk", ".europa.eu", ".gov.tw", ".gov.cn",
                          ".go.jp", ".gc.ca", ".gov.au", ".gov.sg", ".gov.kr", ".gov.in",
                          ".admin.ch", ".gouv.fr", ".bund.de", ".int")
OFFICIAL_HOSTS = frozenset({
    "www.ecb.europa.eu", "www.bankofengland.co.uk", "www.boj.or.jp", "www.imf.org",
    "www.worldbank.org", "www.bis.org", "www.oecd.org", "www.iea.org", "www.opec.org",
    "www.imo.org", "www.wto.org", "www.un.org", "press.un.org", "www.ukmto.org",
    "www.moodys.com", "ratings.moodys.com", "www.spglobal.com", "disclosure.spglobal.com",
    "www.fitchratings.com", "www.federalregister.gov", "crsreports.congress.gov",
    "www.congress.gov", "ustr.gov", "www.whitehouse.gov", "www.cbp.gov", "www.state.gov",
    "www.commerce.gov", "www.bis.doc.gov", "www.nhc.noaa.gov", "www.noaa.gov",
    "www.fiscal.treasury.gov", "www.treasurydirect.gov",
    "www.ndc.gov.tw", "www.mnd.gov.tw", "www.mofa.gov.tw", "english.www.gov.cn",
    "www.fmprc.gov.cn", "english.mofcom.gov.cn", "www.customs.gov.cn",
    "www.pbc.gov.cn", "www.stats.gov.cn",
})
SPECIALIZED_HOSTS = frozenset({
    "www.reuters.com", "apnews.com", "www.bloomberg.com", "www.ft.com", "www.wsj.com",
    "www.economist.com", "www.cnbc.com", "www.politico.com", "www.axios.com",
    "www.bbc.com", "www.bbc.co.uk", "www.theguardian.com", "www.nytimes.com",
    "www.washingtonpost.com", "asia.nikkei.com", "www.scmp.com", "focustaiwan.tw",
    "www.taipeitimes.com", "www.aljazeera.com", "www.lloydslist.com",
    "www.csis.org", "www.piie.com", "www.cfr.org",
    "www.brookings.org", "www.iiss.org", "www.rand.org", "www.atlanticcouncil.org",
    "www.marinetraffic.com", "gcaptain.com", "www.tradewindsnews.com",
    "www.hellenicshippingnews.com", "www.argusmedia.com", "oilprice.com",
    "tradingeconomics.com", "www.investing.com", "www.marketwatch.com",
    "www.barrons.com", "www.foxbusiness.com", "www.npr.org", "www.pbs.org",
    "www.usni.org", "news.usni.org", "www.defensenews.com", "breakingdefense.com",
    "www.janes.com", "www.supplychaindive.com", "www.freightwaves.com",
})
MAX_SOURCE_FETCHES = 6
SPECIALIZED_TIMEOUT_SECONDS = 5
# R125 live lesson (issuer_overlap, 2 November 2026): 14 of 42 page fetches exceeded the
# former 160 KB cap on ordinary news and agency pages. Still bounded, now 600 KB.
MAX_BYTES = 600_000
MAX_EXCERPT_CHARS = 2400
_SKIP = frozenset({"head", "script", "style", "noscript", "svg", "nav", "footer", "header", "form"})
_WORDS_TO_SKIP = frozenset({
    "about", "above", "after", "before", "between", "change", "commercial", "december",
    "exactly", "federal", "greater", "hours", "january", "meeting", "month", "monthly",
    "october", "original", "percent", "points", "rates", "release", "scheduled",
    "shipping", "target", "their", "through", "under", "within", "would", "year",
})


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _BodyText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.excluded: list[str] = []
        self.visible: list[str] = []
        self.body_seen = False

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.body_seen = True
        if tag in _SKIP:
            self.excluded.append(tag)
        if tag in ("p", "h1", "h2", "h3", "li", "tr", "br", "div"):
            self.visible.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and tag in self.excluded:
            self.excluded.remove(tag)
        if tag in ("p", "h1", "h2", "h3", "li", "tr", "div"):
            self.visible.append("\n")

    def handle_data(self, data):
        if not self.excluded and data.strip():
            self.visible.append(re.sub(r"\s+", " ", data).strip() + " ")


def _extract_visible(body: bytes, content_type: str) -> str:
    value = body.decode("utf-8", errors="replace")
    if "text/plain" in content_type:
        return value[:MAX_BYTES]
    parser = _BodyText()
    parser.feed(value)
    return "".join(parser.visible)[:MAX_BYTES]


def _excerpt(visible: str, event_title: str, source_title: str) -> str:
    """Rank bounded text fragments by lexical relevance, retaining exact text."""
    words = {
        w for w in re.findall(r"[a-z][a-z0-9]{3,}", (event_title + " " + source_title).lower())
        if w not in _WORDS_TO_SKIP
    }
    # Synonyms bridge policy-statement phrasing with a question about a rate hike.
    if "hike" in words or "funds" in words:
        words.update({"raise", "range", "inflation", "rate", "hike", "voted"})
    if "cpi" in words or "consumer" in words:
        words.update({"consumer", "price", "index", "adjusted", "increased", "inflation"})
    if "hormuz" in words:
        words.update({"hormuz", "flow", "tanker", "transit", "conflict"})
    if "taiwan" in words:
        words.update({"taiwan", "coast", "guard", "blockade", "maritime"})
    # R125 generic synonym groups so a precise predicate finds the page's own
    # wording; these are retrieval hints for excerpt ranking, not event rules.
    if words & {"blockade", "quarantine", "naval"}:
        words.update({"blockade", "quarantine", "naval", "vessels", "ports", "shipping", "exercise"})
    if words & {"tariff", "tariffs", "duties", "duty"}:
        words.update({"tariff", "tariffs", "duties", "imports", "section", "trade", "goods", "rate"})
    if words & {"downgrade", "downgraded", "rating", "ratings", "sovereign"}:
        words.update({"downgrade", "rating", "ratings", "outlook", "negative", "sovereign",
                      "moody", "fitch", "credit", "debt", "deficit"})
    if words & {"sanctions", "sanction", "embargo"}:
        words.update({"sanctions", "embargo", "designated", "ofac", "restrictions"})
    if words & {"strike", "walkout", "union"}:
        words.update({"strike", "union", "workers", "contract", "negotiations"})
    if words & {"export", "exports", "controls", "license", "licence"}:
        words.update({"export", "controls", "license", "licenses", "restrictions", "chips"})
    fragments = []
    for paragraph in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", visible):
        text = re.sub(r"\s+", " ", paragraph).strip()
        if not 30 <= len(text):
            continue
        for start in range(0, min(len(text), 4000), 800):
            part = text[start:start + 1000]
            hits = set(re.findall(r"[a-z][a-z0-9]{3,}", part.lower())) & words
            fragments.append((sum(1 for w in hits if w in words), len(fragments), part))
        if len(fragments) > 1200:
            break
    ranked = [row for row in sorted(fragments, key=lambda v: (-v[0], v[1]))
              if row[0] >= 1][:6]
    if not ranked:
        return ""
    out = []
    for _, _, sentence in sorted(ranked, key=lambda v: v[1]):
        if len("\n".join(out)) + len(sentence) + 1 > MAX_EXCERPT_CHARS:
            break
        out.append(sentence)
    return "\n".join(out)


def source_tier(url: object) -> str | None:
    """Publisher class of a canonical public URL: OFFICIAL, SPECIALIZED or None.

    The suffix rule admits government and intergovernmental hosts generically;
    the explicit lists cover central banks, rating agencies and reputable
    reporting. The tier never asserts that a page's content is true.
    """
    if not isinstance(url, str) or public_source_url(url) != url:
        return None
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return None
    if host in PRIMARY_HOSTS or host in OFFICIAL_HOSTS:
        return "OFFICIAL"
    if any(host.endswith(suffix) for suffix in OFFICIAL_HOST_SUFFIXES):
        return "OFFICIAL"
    if host in SPECIALIZED_HOSTS:
        return "SPECIALIZED"
    return None


def _fetch_excerpt(open_url: Callable, url: str, sid: str, event_title: str,
                   source_title: str, *, timeout: float, attempt: dict | None = None) -> dict | None:
    attempt = attempt if attempt is not None else {}
    try:
        request = Request(url, headers={"User-Agent": "CastleEvidenceResearch/1.0",
                                        "Accept": "text/html,text/plain"},
                          method="GET")
        with open_url(request, timeout=timeout) as response:
            if getattr(response, "geturl", lambda: url)() != url:
                attempt["outcome"] = "REDIRECTED_REFUSED"
                return None
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if not ("text/html" in content_type or "text/plain" in content_type):
                attempt["outcome"] = "NON_TEXT_CONTENT_TYPE"
                return None
            if int(response.headers.get("Content-Length", "0") or "0") > MAX_BYTES:
                attempt["outcome"] = "BODY_TOO_LARGE"
                return None
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                attempt["outcome"] = "BODY_TOO_LARGE"
                return None
    except HTTPError as exc:
        attempt["outcome"] = "HTTP_" + str(exc.code)
        return None
    except (URLError, TimeoutError, OSError) as exc:
        attempt["outcome"] = "TIMEOUT" if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() else "NETWORK_FAILURE"
        return None
    except (ValueError, TypeError):
        attempt["outcome"] = "INVALID_RESPONSE"
        return None
    try:
        excerpt = _excerpt(_extract_visible(body, content_type), event_title, source_title)
    except (ValueError, TypeError, AssertionError):
        attempt["outcome"] = "TEXT_EXTRACTION_FAILED"
        return None
    if len(excerpt) < 40:
        attempt["outcome"] = "NO_RELEVANT_VISIBLE_TEXT"
        return None
    attempt["outcome"] = "EXCERPT_RETRIEVED"
    return {
        "source_id": sid,
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "excerpt": excerpt,
        "evidence_scope": EVIDENCE_SCOPE,
        "publication_time_independently_verified": False,
        "source_text_is_untrusted_data": True,
    }


def retrieve_source_excerpts(leads: list[dict], event_title: str,
                             *, opener: Callable | None = None,
                             max_fetches: int = MAX_SOURCE_FETCHES,
                             tiers: tuple[str, ...] = ("OFFICIAL", "SPECIALIZED"),
                             attempt_ledger: list | None = None) -> list[dict]:
    """R125 tiered direct page text: official hosts first, then specialized.

    Same transport rules as the official-only fetcher (canonical HTTPS URL,
    no redirects, HTML/plain text only, bounded body, bounded attempts). Each
    excerpt records its publisher tier and tier-specific evidence scope. A
    fetch failure is simply absent; it is never replaced by a lead title.
    """
    if not isinstance(leads, list) or not isinstance(event_title, str):
        return []
    if type(max_fetches) is not int or not 0 <= max_fetches <= MAX_SOURCE_FETCHES:
        raise ValueError("Invalid source evidence fetch limit.")
    if not isinstance(tiers, tuple) or not set(tiers) <= {"OFFICIAL", "SPECIALIZED"}:
        raise ValueError("Invalid source tiers.")
    ordered = []
    for wanted in ("OFFICIAL", "SPECIALIZED"):
        if wanted not in tiers:
            continue
        for lead in leads:
            if isinstance(lead, dict) and source_tier(lead.get("url")) == wanted:
                ordered.append((wanted, lead))
    open_url = opener or build_opener(_RejectRedirect()).open
    results, seen, attempts = [], set(), 0
    for tier, lead in ordered:
        if attempts >= max_fetches:
            break
        url, sid = lead.get("url"), lead.get("id")
        if not isinstance(sid, str) or not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        attempts += 1
        attempt = {"url": url, "source_id": sid, "tier": tier, "outcome": "UNKNOWN"}
        row = _fetch_excerpt(open_url, url, sid, event_title, str(lead.get("title", "")),
                             timeout=3 if tier == "OFFICIAL" else SPECIALIZED_TIMEOUT_SECONDS, attempt=attempt)
        if attempt_ledger is not None:
            attempt_ledger.append(attempt)
        if row is None:
            continue
        row["source_tier"] = tier
        if tier == "SPECIALIZED":
            row["evidence_scope"] = SPECIALIZED_EVIDENCE_SCOPE
        results.append(row)
    return results


def retrieve_primary_excerpts(leads: list[dict], event_title: str,
                              *, opener: Callable | None = None,
                              max_fetches: int = MAX_PRIMARY_FETCHES) -> list[dict]:
    """Fetch independently observed HTML/plain text from limited exact lead URLs.

    A tool- or user-authored source lead does not grant arbitrary URL access:
    we require a canonical public HTTPS URL on an explicit first-party host.
    No PDF or script execution; no partial/truncated body is treated as complete.
    A 3-second limit per page keeps the live pipeline bounded and recoverable.
    """
    if not isinstance(leads, list) or not isinstance(event_title, str):
        return []
    if type(max_fetches) is not int or not 0 <= max_fetches <= MAX_PRIMARY_FETCHES:
        raise ValueError("Invalid primary evidence fetch limit.")
    open_url = opener or build_opener(_RejectRedirect()).open
    results, seen, attempts = [], set(), 0
    for lead in leads:
        if attempts >= max_fetches:
            break
        if not isinstance(lead, dict):
            continue
        url = lead.get("url")
        sid = lead.get("id")
        if (not isinstance(sid, str) or not isinstance(url, str)
                or url in seen or public_source_url(url) != url
                or urlsplit(url).hostname not in PRIMARY_HOSTS):
            continue
        seen.add(url)
        attempts += 1
        try:
            request = Request(url, headers={"User-Agent": "CastleEvidenceResearch/1.0",
                                            "Accept": "text/html,text/plain"},
                              method="GET")
            with open_url(request, timeout=3) as response:
                # Defense in depth for injected clients and HTTP-3xx relocation.
                if getattr(response, "geturl", lambda: url)() != url:
                    continue
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if not ("text/html" in content_type or "text/plain" in content_type):
                    continue
                if int(response.headers.get("Content-Length", "0") or "0") > MAX_BYTES:
                    continue
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    continue
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError):
            continue
        try:
            excerpt = _excerpt(_extract_visible(body, content_type), event_title,
                               str(lead.get("title", "")))
        except (ValueError, TypeError, AssertionError):
            continue
        if len(excerpt) < 40:
            continue
        results.append({
            "source_id": sid,
            "url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "excerpt": excerpt,
            "evidence_scope": EVIDENCE_SCOPE,
            "publication_time_independently_verified": False,
            "source_text_is_untrusted_data": True,
        })
    return results
