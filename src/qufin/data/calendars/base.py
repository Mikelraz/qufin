"""Exchange-calendar protocol and the shared ``Session`` dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import polars as pl


@dataclass(slots=True, frozen=True)
class Session:
    """A single trading session: ``[open, close)`` in UTC."""

    date: date
    open: datetime
    close: datetime
    half_day: bool = False

    def __post_init__(self) -> None:
        if self.close <= self.open:
            raise ValueError("Session close must be strictly greater than open")


@runtime_checkable
class ExchangeCalendar(Protocol):
    """Trading-session arithmetic for one venue."""

    tz: ZoneInfo

    def sessions(self, start: date, end: date) -> pl.DataFrame:
        """Return one row per session in ``[start, end]``.

        Columns: ``(date, open, close, half_day)``. ``open`` and ``close``
        are tz-aware UTC datetimes.
        """
        ...

    def is_session(self, ts: datetime) -> bool:
        """True if ``ts`` falls within any session of this calendar."""
        ...

    def next_close(self, ts: datetime) -> datetime:
        """Return the first session close that is ``>= ts``."""
        ...

    def align(self, frame: pl.DataFrame, *, drop_outside: bool = True) -> pl.DataFrame:
        """Drop / keep rows of ``frame`` whose ``timestamp`` falls outside sessions."""
        ...
