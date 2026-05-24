"""
Time-based resampling of OHLCV bars.

Aggregates finer-grained bars into a coarser interval using the standard
``open = first, high = max, low = min, close = last, volume = sum`` rule.
The window is closed on the left and labelled on the left ``open`` timestamp.
"""

from __future__ import annotations

import polars as pl

from .._types import OHLCV


def time_bars(bars: OHLCV, *, every: str, offset: str | None = None) -> OHLCV:
    """Resample ``bars`` into windows of length ``every`` (polars duration string).

    Examples
    --------
    ``every='5m'`` → 5-minute bars; ``every='1h'``; ``every='1d'``.
    ``offset`` shifts the window grid (e.g. ``'9h30m'`` for market-open anchored).
    """
    grouped = bars.data.group_by_dynamic(
        index_column="timestamp",
        every=every,
        offset=offset,
        closed="left",
        label="left",
    ).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    )
    return OHLCV.from_records(grouped, symbol=bars.symbol)
