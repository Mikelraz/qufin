"""
Hull Suite — Hull Moving Average family and dual-band ribbon.

The Hull Moving Average (HMA) reduces lag relative to SMA/EMA while keeping a
smooth output. The Hull Suite renders **two concurrent HMA bands** of different
length/variant ("fast" and "slow") on the chart, forming a coloured ribbon.
Trading decisions are driven by:

* the slope (rising/falling) of each band — colour-coded green/red
* the position of price relative to the ribbon — above, below, or inside.

Three HMA variants are provided:

* ``hma``  — standard HMA: ``WMA(2·WMA(n/2) − WMA(n), √n)``.  Fastest, most
  responsive, lowest lag.
* ``thma`` — Triple-HMA (LazyBear).  Less twitchy; better for swing filters.
  ``WMA(3·WMA(n/3) − WMA(n/2) − WMA(n), n)``.
* ``ehma`` — Exponential-smoothed HMA.  Same Hull skeleton but EMA instead of
  WMA: ``EMA(2·EMA(n/2) − EMA(n), √n)``.  Slower, cleaner.

The companion module :mod:`qufin.strategies.hull_strategy` implements signal
generation and multi-timeframe / VWAP / momentum filters on top of the ribbon.

Usage
-----
    >>> import numpy as np
    >>> from qufin.strategies.hull_suite import hull_ribbon, price_vs_ribbon
    >>> close = np.linspace(100.0, 120.0, 400) + np.random.randn(400) * 0.3
    >>> ribbon = hull_ribbon(close, fast_length=50, slow_length=60)
    >>> ribbon.fast.color[-1], ribbon.slow.color[-1]
    ('green', 'green')
    >>> price_vs_ribbon(close, ribbon.fast.values, ribbon.slow.values)[-1]
    'above'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..indicators._kernels import ema_kernel, wma_kernel
from ..indicators._types import to_numpy_1d

HullVariant = Literal["hma", "thma", "ehma"]
RibbonPosition = Literal["above", "below", "inside"]
SlopeColor = Literal["green", "red", "flat"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class HullBand:
    """
    A single Hull band: the smoothed series plus its per-bar slope/colour.

    Attributes
    ----------
    values : np.ndarray
        The Hull-smoothed series (length T, NaN-padded warm-up).
    slope : np.ndarray
        +1 where ``values[t] > values[t-1]`` (rising), −1 where falling, 0 flat.
        NaN where either bar is undefined.
    color : np.ndarray[str]
        ``'green'`` (rising), ``'red'`` (falling), ``'flat'`` (equal), or
        empty string ``''`` during warm-up.
    length : int
        Lookback window used to build this band.
    variant : str
        ``'hma'`` | ``'thma'`` | ``'ehma'``.
    """

    values: np.ndarray
    slope: np.ndarray
    color: np.ndarray
    length: int
    variant: HullVariant


@dataclass(slots=True, frozen=True)
class HullRibbon:
    """
    Two-band Hull ribbon plus the per-bar price-vs-ribbon position.

    Attributes
    ----------
    fast : HullBand
        The faster (shorter / more responsive) band.
    slow : HullBand
        The slower (longer / steadier) band.
    position : np.ndarray[str]
        ``'above'`` if close > max(fast, slow), ``'below'`` if close <
        min(fast, slow), ``'inside'`` otherwise.  Empty string during warm-up.
    """

    fast: HullBand
    slow: HullBand
    position: np.ndarray


# ---------------------------------------------------------------------------
# Moving-average primitives (kernels reused from qufin.indicators)
# ---------------------------------------------------------------------------


def wma(values: np.ndarray, period: int) -> np.ndarray:
    """Linearly-weighted moving average; weights 1..period (most recent highest)."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return wma_kernel(to_numpy_1d(values), period)


def hma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Standard Hull MA: ``WMA(2·WMA(price, n/2) − WMA(price, n), √n)``.

    Fastest of the three variants; the lowest-lag option but also the most
    sensitive to noise.  Use this for the fast (entry-timing) band.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    x = to_numpy_1d(values)
    half = max(1, period // 2)
    root = max(1, int(math.floor(math.sqrt(period))))
    inner = 2.0 * wma_kernel(x, half) - wma_kernel(x, period)
    cleaned = np.where(np.isnan(inner), 0.0, inner)
    out = wma_kernel(cleaned, root)
    warmup = (period - 1) + (root - 1)
    out[:warmup] = np.nan
    return out


def thma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Triple-HMA (LazyBear):
    ``WMA(3·WMA(price, n/3) − WMA(price, n/2) − WMA(price, n), n)``.

    Smoother than ``hma``; less prone to whipsaws.  Good choice for the slow
    (trend-filter) band when one wants a calmer ribbon.
    """
    if period < 3:
        raise ValueError(f"period must be >= 3, got {period}")
    x = to_numpy_1d(values)
    n1 = max(1, period // 3)
    n2 = max(1, period // 2)
    w1 = wma_kernel(x, n1)
    w2 = wma_kernel(x, n2)
    w3 = wma_kernel(x, period)
    inner = 3.0 * w1 - w2 - w3
    cleaned = np.where(np.isnan(inner), 0.0, inner)
    out = wma_kernel(cleaned, period)
    warmup = (period - 1) + (period - 1)
    out[:warmup] = np.nan
    return out


def ehma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential-smoothed Hull:
    ``EMA(2·EMA(price, n/2) − EMA(price, n), √n)``.

    Replaces the WMA stages with EMAs.  Slightly more lag than ``hma`` but
    visually cleaner, with fewer slope flips on noisy bars.  Good default
    for the slow band of the ribbon.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    x = to_numpy_1d(values)
    half = max(1, period // 2)
    root = max(1, int(math.floor(math.sqrt(period))))
    e_half = ema_kernel(x, half)
    e_full = ema_kernel(x, period)
    inner = 2.0 * e_half - e_full
    cleaned = np.where(np.isnan(inner), 0.0, inner)
    out = ema_kernel(cleaned, root)
    warmup = (period - 1) + (root - 1)
    out[:warmup] = np.nan
    return out


_HULL_DISPATCH = {"hma": hma, "thma": thma, "ehma": ehma}


def _hull(variant: HullVariant, values: np.ndarray, period: int) -> np.ndarray:
    match variant:
        case "hma" | "thma" | "ehma":
            return _HULL_DISPATCH[variant](values, period)
        case _:
            raise ValueError(f"unknown Hull variant '{variant}'; choose hma|thma|ehma")


# ---------------------------------------------------------------------------
# Slope / colour / ribbon helpers
# ---------------------------------------------------------------------------


def hull_slope(series: np.ndarray) -> np.ndarray:
    """
    Per-bar slope direction of a Hull-smoothed series.

    Returns +1 where the value rose vs. the prior bar, −1 where it fell, 0 if
    unchanged, and NaN where either bar is undefined.  This is the signal
    the Hull Suite uses to colour each band green or red.
    """
    x = to_numpy_1d(series)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    diff = np.diff(x)
    valid = ~np.isnan(diff) & ~np.isnan(x[:-1])
    sl = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    out[1:] = np.where(valid, sl, np.nan)
    return out


def _slope_to_color(slope: np.ndarray) -> np.ndarray:
    """Map the +1/−1/0/NaN slope array to 'green'/'red'/'flat'/'' labels."""
    color = np.empty(slope.shape[0], dtype=object)
    for i in range(slope.shape[0]):
        v = slope[i]
        if np.isnan(v):
            color[i] = ""
        elif v > 0.0:
            color[i] = "green"
        elif v < 0.0:
            color[i] = "red"
        else:
            color[i] = "flat"
    return color


def price_vs_ribbon(price: np.ndarray, fast_band: np.ndarray, slow_band: np.ndarray) -> np.ndarray:
    """
    Classify each bar's price relative to the ribbon.

    The ribbon is the band whose edges are ``min(fast, slow)`` and
    ``max(fast, slow)`` at each bar — the two HMA lines bound the ribbon
    irrespective of which is on top.

    Returns an object array of length T with entries in
    ``{'above', 'below', 'inside', ''}``.  Empty string indicates either
    band is undefined at that bar (warm-up).
    """
    p = to_numpy_1d(price)
    f = to_numpy_1d(fast_band)
    s = to_numpy_1d(slow_band)
    if not (p.shape[0] == f.shape[0] == s.shape[0]):
        raise ValueError("price, fast_band, slow_band must share length")
    n = p.shape[0]
    out = np.empty(n, dtype=object)
    top = np.maximum(f, s)
    bot = np.minimum(f, s)
    for i in range(n):
        if np.isnan(top[i]) or np.isnan(bot[i]) or np.isnan(p[i]):
            out[i] = ""
        elif p[i] > top[i]:
            out[i] = "above"
        elif p[i] < bot[i]:
            out[i] = "below"
        else:
            out[i] = "inside"
    return out


# ---------------------------------------------------------------------------
# Public assembly
# ---------------------------------------------------------------------------


def _build_band(
    price: np.ndarray, length: int, variant: HullVariant, length_multiplier: float
) -> HullBand:
    eff_len = max(2, int(round(length * length_multiplier)))
    values = _hull(variant, price, eff_len)
    slope = hull_slope(values)
    color = _slope_to_color(slope)
    return HullBand(values=values, slope=slope, color=color, length=eff_len, variant=variant)


def hull_ribbon(
    price_series: np.ndarray,
    fast_length: int = 50,
    fast_type: HullVariant = "hma",
    slow_length: int = 60,
    slow_type: HullVariant = "ehma",
    length_multiplier: float = 1.0,
) -> HullRibbon:
    """
    Compute the two-band Hull ribbon and the bar-by-bar price-vs-ribbon position.

    Parameters
    ----------
    price_series : np.ndarray or polars.Series
        Close (or any 1-D price series) of length T.
    fast_length, slow_length : int
        Lookback windows for the fast and slow bands.  Defaults of 50 / 60
        track the canonical Hull Suite settings; 55 is a common swing tweak
        for the fast band.
    fast_type, slow_type : {'hma', 'thma', 'ehma'}
        Hull variant per band.  Defaults: fast = ``hma`` (responsive),
        slow = ``ehma`` (cleaner).
    length_multiplier : float
        Multiplies both lengths.  Useful for previewing higher-timeframe
        ribbons on a single chart (e.g. ``length_multiplier=4`` on 4-hour
        bars roughly approximates a daily ribbon overlay).

    Returns
    -------
    HullRibbon
        Both bands (each with values, slope, colour) plus the per-bar
        position label of the price relative to the ribbon.
    """
    price = to_numpy_1d(price_series)
    if fast_length < 2 or slow_length < 2:
        raise ValueError("fast_length and slow_length must both be >= 2")
    if length_multiplier <= 0:
        raise ValueError("length_multiplier must be > 0")

    fast = _build_band(price, fast_length, fast_type, length_multiplier)
    slow = _build_band(price, slow_length, slow_type, length_multiplier)
    position = price_vs_ribbon(price, fast.values, slow.values)
    return HullRibbon(fast=fast, slow=slow, position=position)
