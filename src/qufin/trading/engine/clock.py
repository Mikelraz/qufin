"""
Bar-resolution event clock.

The clock owns the union of timestamps across all subscribed symbols and
yields one ``(timestamp, {symbol -> bar})`` tuple per step. Symbols that
have no bar at a given timestamp are simply absent from the dict — the
engine treats the missing symbol as "no new data" for that step.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from ...wyckoff._types import BAR_SCHEMA
from .._types import BarEvent


def _ns_to_datetime(ns_value: object) -> datetime:
    """Convert a numpy ``datetime64[ns]`` (or its int representation) to UTC datetime."""
    if isinstance(ns_value, datetime):
        return ns_value if ns_value.tzinfo is not None else ns_value.replace(tzinfo=UTC)
    # numpy datetime64[ns].item() returns an int (nanoseconds since epoch).
    # us-precision is safer: it returns a naive datetime which we re-stamp UTC.
    ts_us = np.datetime64(ns_value, "us") if not isinstance(ns_value, np.datetime64) \
        else ns_value.astype("datetime64[us]")
    naive = ts_us.item()
    if isinstance(naive, datetime):
        return naive.replace(tzinfo=UTC)
    # Fallback for very old numpy: nanoseconds → datetime via epoch.
    return datetime.fromtimestamp(int(ns_value) / 1e9, tz=UTC)


@dataclass(slots=True)
class Clock:
    """Discrete event clock backed by polars frames.

    Construct with a dict mapping symbol → BAR_SCHEMA frame. The clock
    validates schema and timestamp ordering on construction.
    """

    bars: dict[str, pl.DataFrame]

    def __post_init__(self) -> None:
        if not self.bars:
            raise ValueError("Clock requires at least one symbol")
        for symbol, frame in self.bars.items():
            missing = set(BAR_SCHEMA) - set(frame.columns)
            if missing:
                raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
            if frame.height >= 2 and not frame["timestamp"].is_sorted():
                raise ValueError(f"{symbol}: timestamps must be sorted ascending")

    def unique_timestamps(self) -> np.ndarray:
        """The sorted union of timestamps across all symbols."""
        all_ts = pl.concat([f.select("timestamp") for f in self.bars.values()])
        return all_ts.unique().sort("timestamp")["timestamp"].to_numpy()

    def iter_steps(self) -> Iterator[tuple[datetime, dict[str, BarEvent]]]:
        """Yield ``(timestamp, {symbol: BarEvent})`` for every distinct timestamp.

        For each step, the dict contains a ``BarEvent`` for every symbol that
        has a bar at that exact timestamp. Symbols absent from the dict are
        idle for that step.
        """
        cursors: dict[str, int] = {s: 0 for s in self.bars}
        timestamps = self.unique_timestamps()
        # Precompute numpy column views for hot-loop access.
        cols: dict[str, dict[str, np.ndarray]] = {}
        for sym, frame in self.bars.items():
            cols[sym] = {
                "timestamp": frame["timestamp"].to_numpy(),
                "open": frame["open"].to_numpy().astype(np.float64, copy=False),
                "high": frame["high"].to_numpy().astype(np.float64, copy=False),
                "low": frame["low"].to_numpy().astype(np.float64, copy=False),
                "close": frame["close"].to_numpy().astype(np.float64, copy=False),
                "volume": frame["volume"].to_numpy().astype(np.float64, copy=False),
            }
        for ts in timestamps:
            step: dict[str, BarEvent] = {}
            for sym in self.bars:
                i = cursors[sym]
                col = cols[sym]
                if i < len(col["timestamp"]) and col["timestamp"][i] == ts:
                    step[sym] = BarEvent(
                        symbol=sym,
                        timestamp=_ns_to_datetime(col["timestamp"][i]),
                        open=float(col["open"][i]),
                        high=float(col["high"][i]),
                        low=float(col["low"][i]),
                        close=float(col["close"][i]),
                        volume=float(col["volume"][i]),
                        index=i,
                    )
                    cursors[sym] = i + 1
            yield _ns_to_datetime(ts), step
