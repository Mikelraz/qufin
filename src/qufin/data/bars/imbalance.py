"""
Imbalance bars (de Prado, ch. 2.3.2).

Tick / volume / dollar imbalance bars close when the absolute signed
accumulator crosses an EMA-adaptive threshold. The sign comes from the
Lee-Ready tick rule (sign-carry on zero ticks).
"""

from __future__ import annotations

import polars as pl

from .._types import OHLCV
from ._kernels import _imbalance_bars, _tick_signs
from ._util import bars_from_tick_ends, validate_ticks


def tick_imbalance_bars(
    ticks: pl.DataFrame,
    *,
    initial_threshold: float,
    ema_alpha: float = 0.1,
    min_bar_size: int = 1,
    symbol: str = "",
) -> OHLCV:
    """Imbalance bars where the signed weight is the tick sign itself."""
    return _imbalance(
        ticks,
        weight_kind="tick",
        initial_threshold=initial_threshold,
        ema_alpha=ema_alpha,
        min_bar_size=min_bar_size,
        symbol=symbol,
    )


def volume_imbalance_bars(
    ticks: pl.DataFrame,
    *,
    initial_threshold: float,
    ema_alpha: float = 0.1,
    min_bar_size: int = 1,
    symbol: str = "",
) -> OHLCV:
    """Imbalance bars where the signed weight is signed trade size."""
    return _imbalance(
        ticks,
        weight_kind="volume",
        initial_threshold=initial_threshold,
        ema_alpha=ema_alpha,
        min_bar_size=min_bar_size,
        symbol=symbol,
    )


def dollar_imbalance_bars(
    ticks: pl.DataFrame,
    *,
    initial_threshold: float,
    ema_alpha: float = 0.1,
    min_bar_size: int = 1,
    symbol: str = "",
) -> OHLCV:
    """Imbalance bars where the signed weight is signed ``price * size``."""
    return _imbalance(
        ticks,
        weight_kind="dollar",
        initial_threshold=initial_threshold,
        ema_alpha=ema_alpha,
        min_bar_size=min_bar_size,
        symbol=symbol,
    )


def _imbalance(
    ticks: pl.DataFrame,
    *,
    weight_kind: str,
    initial_threshold: float,
    ema_alpha: float,
    min_bar_size: int,
    symbol: str,
) -> OHLCV:
    if initial_threshold <= 0:
        raise ValueError("initial_threshold must be positive")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1]")
    if min_bar_size < 1:
        raise ValueError("min_bar_size must be >= 1")
    validate_ticks(ticks)
    prices = ticks["price"].to_numpy()
    sizes = ticks["size"].to_numpy()
    signs = _tick_signs(prices)
    match weight_kind:
        case "tick":
            signed = signs
        case "volume":
            signed = signs * sizes
        case "dollar":
            signed = signs * sizes * prices
        case _:
            raise ValueError(f"unknown weight_kind: {weight_kind!r}")
    ends = _imbalance_bars(signed, initial_threshold, ema_alpha, min_bar_size)
    return bars_from_tick_ends(ticks, ends, symbol=symbol)
