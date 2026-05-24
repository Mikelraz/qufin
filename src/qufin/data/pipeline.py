"""
End-to-end data pipeline: vendor → store (delta fetch) → adjust → align.

Composes the rest of ``qufin.data`` into a single user-facing call. Each
``load(...)`` invocation:

1. Asks the store for the sub-ranges of ``[start, end)`` not yet on disk.
2. Fetches each gap from the vendor and writes it to the store.
3. Reads the full ``[start, end)`` slice back out of the store.
4. Optionally back-adjusts for corporate actions.
5. Optionally drops bars outside the exchange's regular sessions.

The vendor is only called for genuinely-missing ranges, so repeated
``load`` calls for overlapping windows are cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from ._types import OHLCV
from .adjustments import back_adjust
from .calendars.base import ExchangeCalendar
from .store import Store
from .vendors.base import ActionsSource, OHLCSource


@dataclass(slots=True)
class DataPipeline:
    """Compose vendor, store, calendar, and actions into one ``load`` call."""

    vendor: OHLCSource
    store: Store
    calendar: ExchangeCalendar | None = None
    actions: ActionsSource | None = None

    def load(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        interval: str,
        adjust: bool = True,
        align_calendar: bool = True,
    ) -> OHLCV:
        """Materialise bars for one symbol; delta-fetches any missing window."""
        if end <= start:
            raise ValueError("end must be strictly greater than start")

        gaps = self.store.missing_ranges(symbol, interval, start, end)
        for gap in gaps:
            fetched = self.vendor.fetch(
                symbol, start=gap.start, end=gap.end, interval=interval
            )
            if not fetched.data.is_empty():
                self.store.put(fetched, interval)
            # Record the *requested* gap, not just what came back — otherwise
            # the gap before the first available bar would be re-fetched on
            # every overlapping call.
            self.store.manifest.record(symbol, interval, gap.start, gap.end)
            self.store.manifest.save()

        bars = self.store.get(symbol, interval, start, end)
        if bars is None:
            return OHLCV.from_records(
                pl.DataFrame(schema=self.store.scan(symbol, interval).schema)  # type: ignore[union-attr]
                if self.store.scan(symbol, interval) is not None
                else pl.DataFrame(),
                symbol=symbol,
            )

        if adjust and self.actions is not None:
            action_frame = self.actions.fetch([symbol])
            if not action_frame.is_empty():
                bars = back_adjust(bars, action_frame)

        if align_calendar and self.calendar is not None:
            aligned = self.calendar.align(bars.data)
            bars = OHLCV.from_records(aligned, symbol=symbol)

        return bars

    def load_many(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
        adjust: bool = True,
        align_calendar: bool = True,
    ) -> dict[str, OHLCV]:
        """Materialise bars for many symbols. Returns an ``{symbol: OHLCV}`` dict."""
        return {
            sym: self.load(
                sym,
                start=start,
                end=end,
                interval=interval,
                adjust=adjust,
                align_calendar=align_calendar,
            )
            for sym in symbols
        }
