"""
NYSE / Nasdaq trading calendar.

Regular session: 09:30-16:00 America/New_York. Half-days close at 13:00.
Holidays and half-days are computed from rules; verified against the
NYSE published schedule for 2000-2030.

Limitations
-----------
* Pre-1998 the MLK and Juneteenth holidays did not exist; this calendar
  treats them as never-observed before their introduction year.
* One-off closures (hurricanes, presidential funerals, the WTC week) are
  not modelled. Add to ``NYSECalendar.extra_closures`` if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

NY_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_HALF_DAY_CLOSE = time(13, 0)

_MLK_FIRST_YEAR = 1998
_JUNETEENTH_FIRST_YEAR = 2022


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th occurrence (1-indexed) of ``weekday`` (Mon=0) in (year, month)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last occurrence of ``weekday`` in (year, month)."""
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    last = first_next - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(holiday: date) -> date | None:
    """Apply NYSE Sat→Fri and Sun→Mon observance.

    NYSE observance rule: if the holiday falls on a Saturday, observance
    moves to the preceding Friday; if it falls on a Sunday, observance moves
    to the following Monday. Returns ``None`` if the holiday is not observed
    in that year (currently always returns a date — kept as Optional for
    forward-compat with rules that may drop observance).
    """
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _new_years_day(year: int) -> date | None:
    """NYSE: if Jan 1 falls on Sat, no observance (Fri is regular session)."""
    nyd = date(year, 1, 1)
    if nyd.weekday() == 5:
        return None
    if nyd.weekday() == 6:
        return nyd + timedelta(days=1)
    return nyd


@lru_cache(maxsize=128)
def _holidays_for_year(year: int) -> frozenset[date]:
    """Full set of NYSE-closed dates in ``year`` (closures only, not half-days)."""
    days: list[date] = []
    if (d := _new_years_day(year)) is not None:
        days.append(d)
    if year >= _MLK_FIRST_YEAR:
        days.append(_nth_weekday(year, 1, weekday=0, n=3))
    days.append(_nth_weekday(year, 2, weekday=0, n=3))  # Washington's Birthday
    days.append(_easter_sunday(year) - timedelta(days=2))  # Good Friday
    days.append(_last_weekday(year, 5, weekday=0))  # Memorial Day
    if year >= _JUNETEENTH_FIRST_YEAR:
        if (d := _observed(date(year, 6, 19))) is not None:
            days.append(d)
    if (d := _observed(date(year, 7, 4))) is not None:
        days.append(d)
    days.append(_nth_weekday(year, 9, weekday=0, n=1))  # Labor Day
    days.append(_nth_weekday(year, 11, weekday=3, n=4))  # Thanksgiving
    if (d := _observed(date(year, 12, 25))) is not None:
        days.append(d)
    return frozenset(days)


@lru_cache(maxsize=128)
def _half_days_for_year(year: int) -> frozenset[date]:
    """Days when NYSE closes early at 13:00."""
    days: list[date] = []

    thanksgiving = _nth_weekday(year, 11, weekday=3, n=4)
    days.append(thanksgiving + timedelta(days=1))  # Black Friday

    july_4 = date(year, 7, 4)
    if july_4.weekday() <= 3:
        days.append(july_4 - timedelta(days=1))

    christmas = date(year, 12, 25)
    christmas_eve = date(year, 12, 24)
    if christmas.weekday() in (1, 2, 3, 4) and christmas_eve.weekday() in (0, 1, 2, 3):
        days.append(christmas_eve)

    holidays = _holidays_for_year(year)
    return frozenset(d for d in days if d not in holidays)


@dataclass(slots=True)
class NYSECalendar:
    """NYSE / Nasdaq regular-session calendar.

    Sessions run 09:30-16:00 America/New_York (13:00 on half-days).
    """

    tz: ZoneInfo = NY_TZ
    extra_closures: frozenset[date] = field(default_factory=frozenset)

    def sessions(self, start: date, end: date) -> pl.DataFrame:
        """Return one row per session in ``[start, end]``."""
        if end < start:
            raise ValueError("end must be >= start")
        dates: list[date] = []
        opens: list[datetime] = []
        closes: list[datetime] = []
        half_days: list[bool] = []
        cursor = start
        one_day = timedelta(days=1)
        while cursor <= end:
            if self._is_trading_day(cursor):
                half = cursor in _half_days_for_year(cursor.year)
                open_local = datetime.combine(cursor, _REGULAR_OPEN, tzinfo=self.tz)
                close_local = datetime.combine(
                    cursor, _HALF_DAY_CLOSE if half else _REGULAR_CLOSE, tzinfo=self.tz
                )
                dates.append(cursor)
                opens.append(open_local.astimezone(UTC))
                closes.append(close_local.astimezone(UTC))
                half_days.append(half)
            cursor += one_day
        return pl.DataFrame(
            {
                "date": dates,
                "open": opens,
                "close": closes,
                "half_day": half_days,
            },
            schema={
                "date": pl.Date,
                "open": pl.Datetime("ns", time_zone="UTC"),
                "close": pl.Datetime("ns", time_zone="UTC"),
                "half_day": pl.Boolean,
            },
        )

    def is_session(self, ts: datetime) -> bool:
        """True if ``ts`` falls within a regular session window."""
        local = ts.astimezone(self.tz)
        day = local.date()
        if not self._is_trading_day(day):
            return False
        half = day in _half_days_for_year(day.year)
        end_t = _HALF_DAY_CLOSE if half else _REGULAR_CLOSE
        return _REGULAR_OPEN <= local.time() < end_t

    def next_close(self, ts: datetime) -> datetime:
        """First session close that is ``>= ts`` (UTC)."""
        local = ts.astimezone(self.tz)
        day = local.date()
        for _ in range(15):
            if self._is_trading_day(day):
                half = day in _half_days_for_year(day.year)
                close_local = datetime.combine(
                    day, _HALF_DAY_CLOSE if half else _REGULAR_CLOSE, tzinfo=self.tz
                )
                if close_local >= local:
                    return close_local.astimezone(UTC)
            day += timedelta(days=1)
        raise RuntimeError(f"could not find a session close within 15 days of {ts}")

    def align(self, frame: pl.DataFrame, *, drop_outside: bool = True) -> pl.DataFrame:
        """Filter ``frame`` rows by session membership.

        ``frame`` must carry a ``timestamp`` column (tz-aware UTC). When
        ``drop_outside`` is True (the default) rows outside any regular
        session window are removed; when False, a boolean ``in_session``
        column is added and no rows are dropped.
        """
        if "timestamp" not in frame.columns:
            raise ValueError("frame must contain a 'timestamp' column")
        if frame.is_empty():
            if drop_outside:
                return frame
            return frame.with_columns(pl.lit(False).alias("in_session"))

        ts_min = frame["timestamp"].min()
        ts_max = frame["timestamp"].max()
        # narrow the candidate window to ±1 day to cover tz boundary effects
        start = (ts_min.astimezone(self.tz) - timedelta(days=1)).date()  # type: ignore[union-attr]
        end = (ts_max.astimezone(self.tz) + timedelta(days=1)).date()  # type: ignore[union-attr]
        sess = self.sessions(start, end)
        if sess.is_empty():
            mask = pl.Series("in_session", [False] * frame.height, dtype=pl.Boolean)
            if drop_outside:
                return frame.head(0)
            return frame.with_columns(mask)

        intervals = sess.select(
            pl.col("open").cast(pl.Int64).alias("o"),
            pl.col("close").cast(pl.Int64).alias("c"),
        )
        opens = intervals["o"].to_numpy()
        closes = intervals["c"].to_numpy()
        ts_ns = frame["timestamp"].cast(pl.Int64).to_numpy()
        # For each ts, the rightmost session whose open <= ts; in-session if also < close.
        idx = np.searchsorted(opens, ts_ns, side="right") - 1
        in_session = np.zeros(ts_ns.shape, dtype=np.bool_)
        valid = idx >= 0
        in_session[valid] = ts_ns[valid] < closes[idx[valid]]
        if drop_outside:
            return frame.filter(pl.Series(in_session))
        return frame.with_columns(pl.Series("in_session", in_session))

    def _is_trading_day(self, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        if day in _holidays_for_year(day.year):
            return False
        if day in self.extra_closures:
            return False
        return True
