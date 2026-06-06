"""
Volume delta and cumulative volume delta (CVD).

Trade aggressor side is inferred with the *tick rule* (uptick → buy,
downtick → sell, zero tick carries the previous side) by default, since neither
IBKR ``AllLast`` ticks nor a basic exchange trade feed carry an explicit
aggressor flag. When ``bid`` / ``ask`` columns are present the Lee-Ready
classification can be requested instead.

For bar-only data (no ticks), :func:`bar_delta` approximates the buy/sell split
from each bar's OHLC geometry.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._kernels import cvd_kernel, tick_rule_sign_kernel
from ._types import DeltaProfile, SignMethod, check_lengths, to_numpy_1d

__all__ = [
    "bar_delta",
    "cumulative_volume_delta",
    "delta_divergence",
    "delta_profile",
    "signed_tick_volume",
]


def signed_tick_volume(ticks: pl.DataFrame, *, method: SignMethod = "tick_rule") -> np.ndarray:
    """
    Per-trade signed volume: ``+size`` for aggressor buys, ``-size`` for sells.

    ``method="tick_rule"`` (default) uses uptick/downtick classification on the
    trade price. ``method="lee_ready"`` requires ``bid`` and ``ask`` columns:
    a trade above the mid is a buy, below is a sell, and at the mid falls back
    to the tick rule.
    """
    price = to_numpy_1d(ticks["price"])
    size = to_numpy_1d(ticks["size"])
    check_lengths(price, size)
    match method:
        case "tick_rule":
            sign = tick_rule_sign_kernel(price)
        case "lee_ready":
            missing = {"bid", "ask"} - set(ticks.columns)
            if missing:
                raise ValueError(f"lee_ready requires columns {sorted(missing)}")
            bid = to_numpy_1d(ticks["bid"])
            ask = to_numpy_1d(ticks["ask"])
            mid = 0.5 * (bid + ask)
            sign = tick_rule_sign_kernel(price)  # fallback for at-mid trades
            sign = np.where(price > mid, 1.0, np.where(price < mid, -1.0, sign))
        case _:  # pragma: no cover - exhaustive Literal
            raise ValueError(f"unknown method: {method}")
    return sign * size


def cumulative_volume_delta(
    ticks: pl.DataFrame | None = None,
    *,
    signed_volume: np.ndarray | None = None,
    method: SignMethod = "tick_rule",
) -> np.ndarray:
    """
    Cumulative volume delta — running sum of signed trade volume.

    Pass either a ``TICK_SCHEMA`` frame (signs are computed with ``method``) or
    a pre-computed ``signed_volume`` array.
    """
    if signed_volume is None:
        if ticks is None:
            raise ValueError("provide either ticks or signed_volume")
        signed_volume = signed_tick_volume(ticks, method=method)
    sv = to_numpy_1d(signed_volume)
    return cvd_kernel(sv)


def bar_delta(
    open: np.ndarray,  # noqa: A002
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Approximate per-bar volume delta from OHLC geometry (no tick data).

    The fraction of a bar's volume attributed to buyers is the close's position
    within the bar's range, ``(close - low) / (high - low)``; sellers get the
    complement. Delta is ``(buy_frac - sell_frac) · volume = (2·buy_frac - 1)·v``.
    Zero-range bars contribute zero delta.
    """
    o = to_numpy_1d(open)
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(o, h, l, c, v)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_frac = np.where(rng > 0.0, (c - l) / rng, 0.5)
    return (2.0 * buy_frac - 1.0) * v


def delta_divergence(price: np.ndarray, cvd: np.ndarray, *, window: int = 14) -> np.ndarray:
    """
    Sign of price/CVD divergence over a trailing ``window``.

    Returns ``+1`` for bullish divergence (price fell while CVD rose), ``-1``
    for bearish divergence (price rose while CVD fell), and ``0`` otherwise.
    The first ``window`` entries are ``0`` (warm-up).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    p = to_numpy_1d(price)
    d = to_numpy_1d(cvd)
    check_lengths(p, d)
    n = p.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n <= window:
        return out
    dp = p[window:] - p[:-window]
    dd = d[window:] - d[:-window]
    seg = np.where((dp < 0.0) & (dd > 0.0), 1.0, np.where((dp > 0.0) & (dd < 0.0), -1.0, 0.0))
    out[window:] = seg
    return out


def delta_profile(
    ticks: pl.DataFrame, *, n_bins: int = 50, method: SignMethod = "tick_rule"
) -> DeltaProfile:
    """
    Buy/sell volume split by price (footprint-style) from a tick frame.

    Each trade's size is added to the buy or sell bucket of the price bin it
    falls in, per the inferred aggressor side.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    price = to_numpy_1d(ticks["price"])
    size = to_numpy_1d(ticks["size"])
    if price.shape[0] == 0:
        raise ValueError("tick frame is empty")
    signed = signed_tick_volume(ticks, method=method)
    buy_w = np.where(signed > 0.0, size, 0.0)
    sell_w = np.where(signed < 0.0, size, 0.0)
    p_lo = float(price.min())
    p_hi = float(price.max())
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    edges = np.linspace(p_lo, p_hi, n_bins + 1)
    buy_hist, _ = np.histogram(price, bins=edges, weights=buy_w)
    sell_hist, _ = np.histogram(price, bins=edges, weights=sell_w)
    buy_hist = buy_hist.astype(np.float64, copy=False)
    sell_hist = sell_hist.astype(np.float64, copy=False)
    return DeltaProfile(
        price_bins=edges,
        buy_volume=buy_hist,
        sell_volume=sell_hist,
        delta=buy_hist - sell_hist,
    )
