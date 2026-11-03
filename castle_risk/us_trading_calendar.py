"""NYSE regular-session calendar for fixed common price endpoints (R122).

Holiday dates are transcribed from the official NYSE hours-and-calendars page
(https://www.nyse.com/trade/hours-calendars, read 2026-11-02 UTC) for 2026, 2027
and 2028. Early-close days are regular sessions with a 4:00 p.m. close moved to
1:00 p.m.; they still produce a settled daily close and count as sessions.
Dates outside the transcribed years fail closed instead of guessing.
"""
from __future__ import annotations

from datetime import date, timedelta

CALENDAR_SOURCE = "https://www.nyse.com/trade/hours-calendars"
CALENDAR_READ_UTC = "2026-11-02"
NYSE_HOLIDAYS = frozenset({
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
    # 2028 (New Year's Day falls on Saturday and is not observed)
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29), date(2028, 6, 19),
    date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23), date(2028, 12, 25),
})
COVERED_YEARS = frozenset({2026, 2027, 2028})
ENDPOINT_RULES = {
    # Matches the R120 convention for one-day scheduled releases (CPI 11-10 -> 11-12).
    "SCHEDULED_RELEASE_PLUS_2_SESSIONS": 2,
    # A window event may occur at any time up to the deadline; the common
    # endpoint is a fixed number of sessions after the last possible onset.
    "WINDOW_DEADLINE_PLUS_20_SESSIONS": 20,
}


class TradingCalendarError(ValueError):
    """The requested date is outside the transcribed official calendar."""


def is_trading_session(day: date) -> bool:
    if not isinstance(day, date):
        raise TradingCalendarError("A calendar date is required.")
    if day.year not in COVERED_YEARS:
        raise TradingCalendarError("Date outside the transcribed NYSE calendar years.")
    return day.weekday() < 5 and day not in NYSE_HOLIDAYS


def sessions_after(day: date, count: int) -> date:
    """The ``count``-th regular session strictly after ``day``."""
    if type(count) is not int or not 1 <= count <= 260:
        raise TradingCalendarError("Session count must be an integer between 1 and 260.")
    cursor, found = day, 0
    while found < count:
        cursor += timedelta(days=1)
        if is_trading_session(cursor):
            found += 1
    return cursor


def sessions_strictly_between(start: date, end: date) -> int:
    """Number of regular sessions in (start, end]; zero when end <= start."""
    if end <= start:
        return 0
    total, cursor = 0, start
    while cursor < end:
        cursor += timedelta(days=1)
        if is_trading_session(cursor):
            total += 1
    return total


def common_endpoint_for_event(deadline: date, rule: str) -> date:
    """Deterministic common price endpoint for YES and NO branches of one event."""
    if rule not in ENDPOINT_RULES:
        raise TradingCalendarError("Unknown common endpoint rule.")
    return sessions_after(deadline, ENDPOINT_RULES[rule])
