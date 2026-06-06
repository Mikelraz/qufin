"""
Numba-jitted numerical kernels for the microstructure subpackage.

Convention (shared with :mod:`qufin.timeseries._kernels`)
---------------------------------------------------------
* Kernels accept only primitive types: ``np.ndarray`` (float64 contiguous) and
  scalars.  No dataclasses, polars objects, or Python objects cross the boundary.
* Every kernel is decorated with ``@njit(cache=True)`` so compilation is
  amortised across runs.
* Trade-sign kernels emit ``+1.0`` (buyer-initiated), ``-1.0`` (seller-initiated)
  and ``0.0`` (indeterminate / unclassifiable).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def tick_signs(prices: np.ndarray) -> np.ndarray:
    """Lee-Ready tick rule with sign-carry on zero ticks (first tick = 0)."""
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last = 0.0
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0.0:
            last = 1.0
        elif diff < 0.0:
            last = -1.0
        out[i] = last
    return out


@njit(cache=True)
def quote_signs(prices: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Quote rule: sign of (price − midpoint); 0 exactly at the midpoint."""
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mid = 0.5 * (bid[i] + ask[i])
        if prices[i] > mid:
            out[i] = 1.0
        elif prices[i] < mid:
            out[i] = -1.0
    return out


@njit(cache=True)
def lee_ready_signs(prices: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """
    Lee-Ready (1991): quote rule away from the midpoint, tick-rule tiebreak at it.

    Trades strictly above (below) the prevailing midpoint are buys (sells);
    trades exactly at the midpoint inherit the contemporaneous tick-rule sign.
    """
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last = 0.0
    for i in range(n):
        if i > 0:
            diff = prices[i] - prices[i - 1]
            if diff > 0.0:
                last = 1.0
            elif diff < 0.0:
                last = -1.0
        mid = 0.5 * (bid[i] + ask[i])
        if prices[i] > mid:
            out[i] = 1.0
        elif prices[i] < mid:
            out[i] = -1.0
        else:
            out[i] = last
    return out


@njit(cache=True)
def emo_signs(prices: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """
    Ellis-Michaely-O'Hara (2000): trades at the ask are buys, at the bid sells,
    everything inside the spread is signed by the tick rule.
    """
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last = 0.0
    for i in range(n):
        if i > 0:
            diff = prices[i] - prices[i - 1]
            if diff > 0.0:
                last = 1.0
            elif diff < 0.0:
                last = -1.0
        if prices[i] >= ask[i]:
            out[i] = 1.0
        elif prices[i] <= bid[i]:
            out[i] = -1.0
        else:
            out[i] = last
    return out


@njit(cache=True)
def accumulate_buckets(
    buy_bar: np.ndarray, sell_bar: np.ndarray, bucket_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack per-bar buy/sell volume into equal-volume buckets of size ``bucket_size``.

    A bar whose volume straddles a bucket boundary is split proportionally (the
    bar's buy-fraction is constant, so each chunk carries the same ratio).  The
    trailing partial bucket is discarded, matching the VPIN convention.
    """
    n = buy_bar.shape[0]
    total = 0.0
    for i in range(n):
        total += buy_bar[i] + sell_bar[i]
    n_buckets = int(total // bucket_size)
    out_buy = np.zeros(n_buckets, dtype=np.float64)
    out_sell = np.zeros(n_buckets, dtype=np.float64)
    if n_buckets == 0:
        return out_buy, out_sell

    b = 0
    cur_vol = 0.0
    for i in range(n):
        v = buy_bar[i] + sell_bar[i]
        if v <= 0.0:
            continue
        b_rate = buy_bar[i] / v
        s_rate = sell_bar[i] / v
        remaining = v
        while remaining > 0.0 and b < n_buckets:
            space = bucket_size - cur_vol
            take = space if space < remaining else remaining
            out_buy[b] += take * b_rate
            out_sell[b] += take * s_rate
            cur_vol += take
            remaining -= take
            if cur_vol >= bucket_size - 1e-9:
                b += 1
                cur_vol = 0.0
        if b >= n_buckets:
            break
    return out_buy, out_sell
