"""Direct read-only US BLS CPI-U data snapshot for precise inflation forecasts.

A fixed public BLS API series, not a user-controlled URL or a model claim.
Raw index levels are observed before model issue; relative percent changes
are calculated deterministically. No inference about the future release, no
independent publication-time certification, and no guessed event probability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from typing import Callable

SERIES_ID = "CUSR0000SA0"
API_HOST = "api.bls.gov"
MAX_BYTES = 100_000
EVIDENCE_SCOPE = "DIRECT_BLS_PUBLIC_API_CPI_INDEX_OBSERVATION_NOT_FUTURE_OUTCOME"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def retrieve_cpi_snapshot(target_condition: str, *, opener: Callable | None = None,
                          now_utc: datetime | None = None) -> list[dict]:
    """Retrieve at most the last 2 published monthly seasonally adjusted levels.

    This is a live-only read: never call it to simulate a historical
    information set, which would require dated archival provider snapshots.
    """
    if not isinstance(target_condition, str) or not re.search(
            r"\bCPI\b|\bconsumer price index\b", target_condition, flags=re.I):
        return []
    observed_now = now_utc or datetime.now(timezone.utc)
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        return []
    year = observed_now.year
    if not 2020 <= year <= 2100:
        return []
    url = (f"https://{API_HOST}/publicAPI/v2/timeseries/data/{SERIES_ID}"
           f"?startyear={year-1}&endyear={year}")
    try:
        req = Request(url, headers={"Accept": "application/json",
                                     "User-Agent": "CastleEventEvidence/1.0"}, method="GET")
        with (opener or build_opener(_NoRedirect()).open)(req, timeout=5) as r:
            if getattr(r, "geturl", lambda: url)() != url:
                return []
            if "application/json" not in str(r.headers.get("Content-Type", "")).lower():
                return []
            if int(r.headers.get("Content-Length", "0") or "0") > MAX_BYTES:
                return []
            body = r.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                return []
        response = json.loads(body)
        if response.get("status") != "REQUEST_SUCCEEDED":
            return []
        series = response.get("Results", {}).get("series", [])
        if (not isinstance(series, list) or len(series) != 1
                or series[0].get("seriesID") != SERIES_ID):
            return []
        vals = []
        for d in series[0].get("data", []):
            if not isinstance(d, dict):
                continue
            y, period = d.get("year"), d.get("period")
            if (not isinstance(y, str) or not re.fullmatch(r"\d{4}", y)
                    or not isinstance(period, str) or not re.fullmatch(r"M(0[1-9]|1[0-2])", period)):
                continue
            ym = (int(y), int(period[1:]))
            # A public API should never introduce a future-month value here.
            if ym > (observed_now.year, observed_now.month):
                continue
            try:
                level = float(d.get("value"))
            except (TypeError, ValueError):
                continue
            if isfinite(level) and level > 0:
                vals.append((ym, level))
        vals.sort(reverse=True)
        if len(vals) < 2:
            return []
        (year1, month1), latest = vals[0]
        (year0, month0), previous = vals[1]
        if (year1 * 12 + month1) - (year0 * 12 + month0) != 1:
            return []
        return [{
            "series_id": SERIES_ID,
            "series_definition": "CPI-U All Items US City Average Seasonally Adjusted",
            "latest_index_period": f"{year1}-{month1:02d}",
            "latest_index_value": latest,
            "prior_index_period": f"{year0}-{month0:02d}",
            "prior_index_value": previous,
            "latest_month_over_month_change_pct_computed": round(100*(latest/previous-1), 6),
            "source_url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": EVIDENCE_SCOPE,
            "published_not_future_target": True,
            "source_text_is_untrusted_data": True,
            "publication_time_independently_verified": False,
        }]
    except (HTTPError, URLError, OSError, ValueError, TypeError, AttributeError, KeyError,
            json.JSONDecodeError):
        return []
