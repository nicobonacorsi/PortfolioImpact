"""Explicit New York clocks for upcoming scheduled U.S. macro releases.

Calendar dates are civil dates at the releasing institution, not UTC dates.
Source-only dates with no reviewed clock remain watchlist candidates through
their New York calendar day; no publication time is fabricated for them.
Source quote expiry remains a separate, potentially earlier boundary.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .source_grounding import _CALENDAR


NEW_YORK = ZoneInfo("America/New_York")
FED_RELEASE_TIME_SOURCE = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130313a.htm"
_DATED_FED_CLOCK_SOURCES = {
    "2026-10-28": "https://www.federalreserve.gov/newsevents/2026-october.htm?view=calendar",
}
_FAMILY_KEY = {"FED_POLICY": "US_FOMC", "US_INFLATION": "US_CPI"}
_RELEASE_TIMES = {"FED_POLICY": time(14), "US_INFLATION": time(8, 30)}


def _aware(value, label):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.utcoffset() is None:
        raise ValueError(label + " must be an explicit timezone-aware datetime")
    return parsed


def scheduled_event_window(day, evaluated_at, *, horizon_days=None, release_time_et=None,
                           source_close_at=None, clock_source_url=None):
    """Review date horizon and future status without observing the current clock.

    A known release expires exactly at its scheduled release instant. A source
    quote may expire earlier; when supplied it bounds numeric forward use but
    does not assert that the underlying event has occurred. The caller should
    omit that quote boundary when listing an as-yet unreleased watchlist event.
    """
    evaluated = _aware(evaluated_at, "Evaluation time")
    event_day = date.fromisoformat(day)
    if horizon_days is not None and (isinstance(horizon_days, bool)
            or not isinstance(horizon_days, int) or horizon_days < 0):
        raise ValueError("Horizon must be a nonnegative integer number of local calendar days")
    if release_time_et is not None and (not isinstance(release_time_et, time)
            or release_time_et.tzinfo is not None):
        raise ValueError("A reviewed release time must be an unzoned New York civil time")
    local_now = evaluated.astimezone(NEW_YORK)
    days_ahead = (event_day - local_now.date()).days
    scheduled = (datetime.combine(event_day, release_time_et, tzinfo=NEW_YORK)
                 if release_time_et is not None else None)
    expiry = scheduled or datetime.combine(event_day + timedelta(days=1), time(), tzinfo=NEW_YORK)
    quote_close = _aware(source_close_at, "Source market close") if source_close_at is not None else None
    if quote_close is not None:
        expiry = min(expiry, quote_close)
    pending = days_ahead >= 0 and evaluated < expiry
    in_horizon = days_ahead >= 0 and (horizon_days is None or days_ahead <= horizon_days)
    return {
        "pending": pending,
        "within_horizon": in_horizon,
        "eligible": pending and in_horizon,
        "evaluation_timezone": "America/New_York",
        "evaluation_date_local": local_now.date().isoformat(),
        "days_ahead": days_ahead,
        "scheduled_release_at_utc": scheduled.astimezone(timezone.utc).isoformat() if scheduled else None,
        "source_market_close_at_utc": quote_close.astimezone(timezone.utc).isoformat() if quote_close else None,
        "forward_window_end_at_utc": expiry.astimezone(timezone.utc).isoformat(),
        "timing_basis": ("REVIEWED_SCHEDULED_RELEASE_TIME" if scheduled is not None
                         else "DATE_ONLY_RETAINED_THROUGH_LOCAL_DAY_NOT_A_RELEASE_TIME"),
        "clock_source_url": clock_source_url,
        "expired_quote_implies_event_occurred": False,
    }


def reviewed_macro_window(family, day, evaluated_at, *, horizon_days=None, source_close_at=None):
    """Use a clock only for dates in the independently reviewed Fed/BLS calendar."""
    subject = _FAMILY_KEY.get(family)
    anchors = [value for (key, _), value in _CALENDAR.items()
               if key == subject and value[0] == day]
    clock = _RELEASE_TIMES.get(family) if anchors else None
    url = ((_DATED_FED_CLOCK_SOURCES.get(day, FED_RELEASE_TIME_SOURCE)
            if family == "FED_POLICY" else anchors[0][1]) if anchors else None)
    return scheduled_event_window(day, evaluated_at, horizon_days=horizon_days,
                                  release_time_et=clock, source_close_at=source_close_at,
                                  clock_source_url=url)
