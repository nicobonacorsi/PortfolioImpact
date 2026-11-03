"""R125 tiered direct page text: official before specialized, same transport guards."""
from __future__ import annotations

from castle_risk.official_event_evidence import (
    EVIDENCE_SCOPE, SPECIALIZED_EVIDENCE_SCOPE, retrieve_primary_excerpts, retrieve_source_excerpts, source_tier,
)


_HTML = (b"<html><body><main><p>Fitch Ratings affirmed the United States at AA+ and revised the outlook "
         b"to negative, citing the fiscal deficit and rising debt interest costs.</p>"
         b"<p>The agency said a downgrade could follow if the deficit trajectory worsens.</p></main></body></html>")


class _Response:
    def __init__(self, url):
        self.url = url
        self.headers = {"Content-Type": "text/html", "Content-Length": str(len(_HTML))}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def geturl(self):
        return self.url

    def read(self, n):
        return _HTML[:n]


def test_tiers_cover_government_suffixes_agencies_and_named_outlets_only():
    assert source_tier("https://ustr.gov/about/policy-offices") == "OFFICIAL"
    assert source_tier("https://www.mnd.gov.tw/english/news") == "OFFICIAL"
    assert source_tier("https://www.fitchratings.com/research/sovereigns/us") == "OFFICIAL"
    assert source_tier("https://www.reuters.com/markets/rates-bonds/story") == "SPECIALIZED"
    assert source_tier("https://evil.gov.example.com/x") is None
    assert source_tier("https://www.federalreserve.gov/path?token=x") is None
    assert source_tier("http://www.reuters.com/plain") is None
    assert source_tier("https://randomblog.example/post") is None


def test_official_first_then_specialized_with_tier_specific_scopes_and_bounded_attempts():
    opened = []

    def opener(request, timeout):
        opened.append((request.full_url, timeout))
        return _Response(request.full_url)

    leads = [{"id": "T01", "url": "https://www.reuters.com/markets/us-rating", "title": "US rating outlook"},
             {"id": "T02", "url": "https://www.fitchratings.com/research/us-affirmed", "title": "Fitch affirms US"},
             {"id": "T03", "url": "https://randomblog.example/us-rating", "title": "Blog"},
             {"id": "T04", "url": "https://www.cnbc.com/us-rating", "title": "CNBC on rating"}]
    rows = retrieve_source_excerpts(leads, "US sovereign credit rating downgrade by Fitch or Moody's", opener=opener)
    assert [row["source_id"] for row in rows] == ["T02", "T01", "T04"]
    assert rows[0]["source_tier"] == "OFFICIAL" and rows[0]["evidence_scope"] == EVIDENCE_SCOPE
    assert rows[1]["source_tier"] == "SPECIALIZED" and rows[1]["evidence_scope"] == SPECIALIZED_EVIDENCE_SCOPE
    assert opened[0][1] == 3 and opened[1][1] == 5
    assert all("randomblog" not in url for url, _ in opened)
    assert "affirmed the United States" in rows[0]["excerpt"]
    limited = retrieve_source_excerpts(leads, "US rating", opener=opener, max_fetches=1)
    assert [row["source_id"] for row in limited] == ["T02"]
    official_only = retrieve_source_excerpts(leads, "US rating", opener=opener, tiers=("OFFICIAL",))
    assert [row["source_id"] for row in official_only] == ["T02"]
    legacy = retrieve_primary_excerpts(leads, "US rating", opener=opener)
    assert legacy == []  # fitchratings.com is not in the legacy nine-host list; the legacy path is unchanged
