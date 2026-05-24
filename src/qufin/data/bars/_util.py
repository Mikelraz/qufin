"""Shared helpers for materialising bars from tick boundaries."""

from __future__ import annotations

import numpy as np
import polars as pl

from .._types import BAR_SCHEMA, OHLCV, TICK_SCHEMA


def validate_ticks(ticks: pl.DataFrame) -> None:
    missing = set(TICK_SCHEMA) - set(ticks.columns)
    if missing:
        raise ValueError(f"tick frame is missing columns: {sorted(missing)}")
    if ticks.height >= 2 and not ticks["timestamp"].is_sorted():
        raise ValueError("tick frame must be sorted by timestamp ascending")


def bars_from_tick_ends(
    ticks: pl.DataFrame, ends: np.ndarray, *, symbol: str
) -> OHLCV:
    """Materialise an ``OHLCV`` from a tick frame and inclusive bar-end indices."""
    if ends.size == 0:
        return OHLCV(data=pl.DataFrame(schema=BAR_SCHEMA), symbol=symbol)

    prices = ticks["price"].to_numpy()
    sizes = ticks["size"].to_numpy()
    timestamps = ticks["timestamp"].to_numpy()

    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    n_bars = ends.size
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)
    volumes = np.empty(n_bars, dtype=np.float64)
    for k in range(n_bars):
        s, e = starts[k], ends[k]
        chunk = prices[s : e + 1]
        opens[k] = chunk[0]
        highs[k] = chunk.max()
        lows[k] = chunk.min()
        closes[k] = chunk[-1]
        volumes[k] = sizes[s : e + 1].sum()

    out = pl.DataFrame(
        {
            "timestamp": timestamps[ends],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        schema=BAR_SCHEMA,
    )
    return OHLCV(data=out, symbol=symbol)
