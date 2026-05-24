"""
Volume bars and dollar bars (de Prado, ch. 2).

A *volume bar* closes whenever the cumulative trade volume since the last
bar crosses a fixed threshold ``V``; a *dollar bar* closes on cumulative
``price * size`` crossing ``D``. Both produce more statistically homogeneous
samples than time bars on heavy-tail intraday flow.
"""

from __future__ import annotations

import polars as pl

from .._types import OHLCV, TICK_SCHEMA
from ._kernels import _threshold_bars
from ._util import bars_from_tick_ends, validate_ticks


def volume_bars(ticks: pl.DataFrame, *, threshold: float, symbol: str = "") -> OHLCV:
    """Construct volume bars from a tick frame matching ``TICK_SCHEMA``."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    validate_ticks(ticks)
    prices = ticks["price"].to_numpy()
    sizes = ticks["size"].to_numpy()
    ends = _threshold_bars(prices, sizes, sizes, threshold)
    return bars_from_tick_ends(ticks, ends, symbol=symbol)


def dollar_bars(ticks: pl.DataFrame, *, threshold: float, symbol: str = "") -> OHLCV:
    """Construct dollar bars (cumulative ``price * size``) from a tick frame."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    validate_ticks(ticks)
    prices = ticks["price"].to_numpy()
    sizes = ticks["size"].to_numpy()
    weights = prices * sizes
    ends = _threshold_bars(prices, sizes, weights, threshold)
    return bars_from_tick_ends(ticks, ends, symbol=symbol)


__all__ = ["TICK_SCHEMA", "dollar_bars", "volume_bars"]
