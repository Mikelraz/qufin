"""
Feature builders over OHLCV frames.

Wraps the indicators in ``qufin.indicators`` so feature engineering is a
single import in user code. A ``FeatureSet`` carries the column names and
the callable that materialises them; downstream code (Pipeline,
MLSignalStrategy) consumes the matrix without caring how it was built.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ....indicators.momentum import rsi
from ....indicators.moving_averages import sma
from ....indicators.volatility import atr

FeatureFn = Callable[[pl.DataFrame], np.ndarray]


@dataclass(slots=True)
class FeatureSet:
    """A named bundle of feature columns to compute from a BAR_SCHEMA frame.

    The ``transform`` method returns a 2-D float64 matrix shaped
    ``(n_bars, len(names))`` aligned to the input frame's row order.
    Rows with insufficient history (warm-up period) are filled with NaN;
    callers should drop them before fitting.
    """

    names: list[str]
    funcs: list[FeatureFn] = field(default_factory=list)

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        cols = [fn(frame) for fn in self.funcs]
        n = max(c.shape[0] for c in cols) if cols else frame.height
        out = np.full((n, len(cols)), np.nan, dtype=np.float64)
        for j, col in enumerate(cols):
            out[: col.shape[0], j] = col
        return out


def rsi_feature(window: int = 14) -> FeatureFn:
    """RSI-of-close feature."""
    def fn(frame: pl.DataFrame) -> np.ndarray:
        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        return rsi(close, window=window)
    return fn


def sma_ratio_feature(short: int = 10, long: int = 50) -> FeatureFn:
    """Ratio of short SMA to long SMA on close."""
    def fn(frame: pl.DataFrame) -> np.ndarray:
        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        s = sma(close, window=short)
        l_ = sma(close, window=long)
        with np.errstate(divide="ignore", invalid="ignore"):
            return s / l_
    return fn


def atr_feature(window: int = 14) -> FeatureFn:
    """ATR(window) on the bar frame's H/L/C columns."""
    def fn(frame: pl.DataFrame) -> np.ndarray:
        high = frame["high"].to_numpy().astype(np.float64, copy=False)
        low = frame["low"].to_numpy().astype(np.float64, copy=False)
        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        return atr(high, low, close, window=window)
    return fn


def build_default_features(
    *,
    rsi_window: int = 14,
    sma_short: int = 10,
    sma_long: int = 50,
    atr_window: int = 14,
    extra: Sequence[tuple[str, FeatureFn]] = (),
) -> FeatureSet:
    """Reasonable default feature set: RSI, SMA ratio, ATR, plus user extras."""
    names = [f"rsi_{rsi_window}", f"sma_ratio_{sma_short}_{sma_long}", f"atr_{atr_window}"]
    funcs: list[FeatureFn] = [
        rsi_feature(rsi_window),
        sma_ratio_feature(sma_short, sma_long),
        atr_feature(atr_window),
    ]
    for name, fn in extra:
        names.append(name)
        funcs.append(fn)
    return FeatureSet(names=names, funcs=funcs)
