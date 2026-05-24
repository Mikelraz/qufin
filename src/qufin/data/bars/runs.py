"""
Tick runs bars (de Prado, ch. 2.3.3).

A bar closes when the longer of the (positive, negative) consecutive
same-sign tick runs since the last bar crosses an EMA-adaptive threshold.
"""

from __future__ import annotations

import polars as pl

from .._types import OHLCV
from ._kernels import _runs_bars, _tick_signs
from ._util import bars_from_tick_ends, validate_ticks


def tick_runs_bars(
    ticks: pl.DataFrame,
    *,
    initial_threshold: float,
    ema_alpha: float = 0.1,
    min_bar_size: int = 1,
    symbol: str = "",
) -> OHLCV:
    """Construct tick runs bars from a tick frame matching ``TICK_SCHEMA``."""
    if initial_threshold <= 0:
        raise ValueError("initial_threshold must be positive")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1]")
    if min_bar_size < 1:
        raise ValueError("min_bar_size must be >= 1")
    validate_ticks(ticks)
    prices = ticks["price"].to_numpy()
    signs = _tick_signs(prices)
    ends = _runs_bars(signs, initial_threshold, ema_alpha, min_bar_size)
    return bars_from_tick_ends(ticks, ends, symbol=symbol)
