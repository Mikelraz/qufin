"""NYSE calendar tests — holidays, half-days, sessions, alignment."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from qufin.data.calendars import ExchangeCalendar, NYSECalendar, Session

NY = ZoneInfo("America/New_York")


def test_calendar_satisfies_protocol() -> None:
    cal = NYSECalendar()
    assert isinstance(cal, ExchangeCalendar)


def test_session_dataclass_validates_order() -> None:
    open_ = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    close = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    s = Session(date=date(2024, 1, 2), open=open_, close=close, half_day=False)
    assert s.close > s.open
    with pytest.raises(ValueError):
        Session(date=date(2024, 1, 2), open=close, close=open_, half_day=False)


def test_sessions_omits_weekends() -> None:
    cal = NYSECalendar()
    sessions = cal.sessions(date(2024, 1, 1), date(2024, 1, 7))
    days = sessions["date"].to_list()
    assert date(2024, 1, 6) not in days
    assert date(2024, 1, 7) not in days


def test_known_holidays_2024_excluded() -> None:
    cal = NYSECalendar()
    sessions = cal.sessions(date(2024, 1, 1), date(2024, 12, 31))
    closed = set(sessions["date"].to_list())
    expected_closed = {
        date(2024, 1, 1),  # New Year's Day (Mon)
        date(2024, 1, 15),  # MLK Day
        date(2024, 2, 19),  # Washington's Birthday
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day (Thu)
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    }
    for d in expected_closed:
        assert d not in closed, f"{d} should be closed"


def test_known_half_days_2024_flagged() -> None:
    cal = NYSECalendar()
    sessions = cal.sessions(date(2024, 1, 1), date(2024, 12, 31))
    half_dates = set(sessions.filter(pl.col("half_day"))["date"].to_list())
    assert date(2024, 7, 3) in half_dates  # July 4 is Thu
    assert date(2024, 11, 29) in half_dates  # Black Friday
    assert date(2024, 12, 24) in half_dates  # Christmas Eve (Tue)


def test_observance_when_holiday_on_weekend() -> None:
    cal = NYSECalendar()
    # July 4, 2026 is a Saturday → observed Fri July 3
    sessions = cal.sessions(date(2026, 7, 1), date(2026, 7, 6))
    days = set(sessions["date"].to_list())
    assert date(2026, 7, 3) not in days
    # Christmas Day 2027 is a Saturday → observed Fri Dec 24
    sessions = cal.sessions(date(2027, 12, 20), date(2027, 12, 31))
    days = set(sessions["date"].to_list())
    assert date(2027, 12, 24) not in days


def test_mlk_introduced_in_1998() -> None:
    cal = NYSECalendar()
    sessions_1997 = cal.sessions(date(1997, 1, 13), date(1997, 1, 24))
    days_97 = set(sessions_1997["date"].to_list())
    assert date(1997, 1, 20) in days_97  # MLK Day 1997 — still trading
    sessions_1998 = cal.sessions(date(1998, 1, 12), date(1998, 1, 24))
    days_98 = set(sessions_1998["date"].to_list())
    assert date(1998, 1, 19) not in days_98


def test_juneteenth_introduced_in_2022() -> None:
    cal = NYSECalendar()
    sessions_2021 = cal.sessions(date(2021, 6, 14), date(2021, 6, 25))
    days_21 = set(sessions_2021["date"].to_list())
    assert date(2021, 6, 18) in days_21
    sessions_2022 = cal.sessions(date(2022, 6, 13), date(2022, 6, 24))
    days_22 = set(sessions_2022["date"].to_list())
    assert date(2022, 6, 20) not in days_22  # June 19 was Sun → observed Mon


def test_is_session_during_regular_hours() -> None:
    cal = NYSECalendar()
    ts = datetime(2024, 6, 3, 10, 0, tzinfo=NY).astimezone(UTC)
    assert cal.is_session(ts)
    pre = datetime(2024, 6, 3, 9, 0, tzinfo=NY).astimezone(UTC)
    assert not cal.is_session(pre)
    post = datetime(2024, 6, 3, 16, 30, tzinfo=NY).astimezone(UTC)
    assert not cal.is_session(post)


def test_is_session_on_holiday() -> None:
    cal = NYSECalendar()
    ts = datetime(2024, 7, 4, 12, 0, tzinfo=NY).astimezone(UTC)
    assert not cal.is_session(ts)


def test_is_session_on_half_day_after_one_pm() -> None:
    cal = NYSECalendar()
    ts = datetime(2024, 11, 29, 14, 0, tzinfo=NY).astimezone(UTC)
    assert not cal.is_session(ts)
    earlier = datetime(2024, 11, 29, 12, 0, tzinfo=NY).astimezone(UTC)
    assert cal.is_session(earlier)


def test_next_close_within_session() -> None:
    cal = NYSECalendar()
    ts = datetime(2024, 6, 3, 11, 0, tzinfo=NY).astimezone(UTC)
    nc = cal.next_close(ts)
    assert nc == datetime(2024, 6, 3, 16, 0, tzinfo=NY).astimezone(UTC)


def test_next_close_skips_weekend() -> None:
    cal = NYSECalendar()
    # Saturday morning → next close is Monday's
    ts = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
    nc = cal.next_close(ts)
    assert nc.astimezone(NY).date() == date(2024, 6, 3)


def test_align_drops_pre_and_post_market() -> None:
    cal = NYSECalendar()
    # 09:00 NY (pre), 10:00 NY (in), 16:30 NY (post) on 2024-06-03
    ts_utc = [
        datetime(2024, 6, 3, 9, 0, tzinfo=NY).astimezone(UTC),
        datetime(2024, 6, 3, 10, 0, tzinfo=NY).astimezone(UTC),
        datetime(2024, 6, 3, 16, 30, tzinfo=NY).astimezone(UTC),
    ]
    frame = pl.DataFrame(
        {"timestamp": ts_utc, "x": [1, 2, 3]},
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC"), "x": pl.Int64},
    )
    out = cal.align(frame)
    assert out.height == 1
    assert out["x"].to_list() == [2]


def test_align_with_drop_outside_false_adds_flag_column() -> None:
    cal = NYSECalendar()
    ts_utc = [
        datetime(2024, 6, 3, 9, 0, tzinfo=NY).astimezone(UTC),
        datetime(2024, 6, 3, 10, 0, tzinfo=NY).astimezone(UTC),
    ]
    frame = pl.DataFrame(
        {"timestamp": ts_utc},
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC")},
    )
    out = cal.align(frame, drop_outside=False)
    assert out.height == 2
    assert out["in_session"].to_list() == [False, True]


def test_align_handles_empty_frame() -> None:
    cal = NYSECalendar()
    frame = pl.DataFrame(
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC"), "x": pl.Int64}
    )
    assert cal.align(frame).is_empty()


def test_extra_closure_takes_effect() -> None:
    one_off = date(2024, 6, 4)
    cal = NYSECalendar(extra_closures=frozenset({one_off}))
    sessions = cal.sessions(date(2024, 6, 3), date(2024, 6, 5))
    days = set(sessions["date"].to_list())
    assert one_off not in days
    assert date(2024, 6, 3) in days
    assert date(2024, 6, 5) in days


def test_sessions_full_year_count_2024() -> None:
    cal = NYSECalendar()
    sessions = cal.sessions(date(2024, 1, 1), date(2024, 12, 31))
    # NYSE published 2024 schedule: 252 trading days.
    assert sessions.height == 252


def test_half_day_close_time_is_one_pm_local() -> None:
    cal = NYSECalendar()
    sessions = cal.sessions(date(2024, 11, 29), date(2024, 11, 29))
    close = sessions["close"][0]
    assert close.astimezone(NY).time() == datetime(2024, 11, 29, 13, 0).time()  # type: ignore[union-attr]


def test_align_across_two_sessions() -> None:
    cal = NYSECalendar()
    ts_utc = [
        datetime(2024, 6, 3, 10, 0, tzinfo=NY).astimezone(UTC),
        datetime(2024, 6, 4, 10, 0, tzinfo=NY).astimezone(UTC),
        datetime(2024, 6, 4, 18, 0, tzinfo=NY).astimezone(UTC),  # post-market
        datetime(2024, 6, 5, 10, 0, tzinfo=NY).astimezone(UTC),
    ]
    frame = pl.DataFrame(
        {"timestamp": ts_utc, "x": list(range(len(ts_utc)))},
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC"), "x": pl.Int64},
    )
    out = cal.align(frame)
    assert out["x"].to_list() == [0, 1, 3]


def test_align_treats_open_as_inclusive_and_close_as_exclusive() -> None:
    cal = NYSECalendar()
    open_ny = datetime(2024, 6, 3, 9, 30, tzinfo=NY)
    close_ny = datetime(2024, 6, 3, 16, 0, tzinfo=NY)
    ts_utc = [open_ny.astimezone(UTC), close_ny.astimezone(UTC)]
    frame = pl.DataFrame(
        {"timestamp": ts_utc, "x": [0, 1]},
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC"), "x": pl.Int64},
    )
    out = cal.align(frame)
    assert out["x"].to_list() == [0]


def test_new_year_observance_no_friday_holiday_when_saturday() -> None:
    cal = NYSECalendar()
    # Jan 1, 2022 is a Saturday → NYSE was open on Fri Dec 31, 2021
    sessions = cal.sessions(date(2021, 12, 30), date(2022, 1, 5))
    days = set(sessions["date"].to_list())
    assert date(2021, 12, 31) in days
    assert date(2022, 1, 3) in days  # Monday after, normal session
    assert date(2022, 1, 1) not in days  # Saturday is just a weekend


def test_sessions_raises_for_bad_range() -> None:
    cal = NYSECalendar()
    with pytest.raises(ValueError):
        cal.sessions(date(2024, 6, 5), date(2024, 6, 1))


def test_align_intraday_minute_bars(tmp_path: None = None) -> None:
    cal = NYSECalendar()
    base = datetime(2024, 6, 3, 9, 0, tzinfo=NY)
    timestamps = [base + timedelta(minutes=i) for i in range(0, 480, 1)]
    frame = pl.DataFrame(
        {"timestamp": [t.astimezone(UTC) for t in timestamps]},
        schema={"timestamp": pl.Datetime("ns", time_zone="UTC")},
    )
    out = cal.align(frame)
    # 09:30 inclusive to 16:00 exclusive = 390 minute bars
    assert out.height == 390
