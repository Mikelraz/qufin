"""
Numba-jitted accumulators for information-driven bars.

All kernels share the same input shape: contiguous float64 arrays of
``(timestamp_ns, price, size)`` plus a per-tick signed indicator (the tick
rule) where appropriate. They return parallel arrays of bar boundaries that
the polars wrapper materialises into an OHLCV frame.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _tick_signs(prices: np.ndarray) -> np.ndarray:
    """Lee-Ready style tick rule with sign-carry on zero ticks."""
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
def _threshold_bars(
    prices: np.ndarray,
    sizes: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Generic threshold accumulator.

    Emits a bar whenever cumulative ``weights[i]`` since the last bar
    crosses ``threshold``. Returns an int64 array of bar-end indices
    (inclusive).
    """
    n = prices.shape[0]
    ends = np.empty(n, dtype=np.int64)
    k = 0
    acc = 0.0
    for i in range(n):
        acc += weights[i]
        if acc >= threshold:
            ends[k] = i
            k += 1
            acc = 0.0
    return ends[:k]


@njit(cache=True)
def _imbalance_bars(
    signed_weights: np.ndarray,
    initial_threshold: float,
    ema_alpha: float,
    min_bar_size: int,
) -> np.ndarray:
    """Imbalance bars with EMA-adaptive expected-imbalance threshold.

    A bar ends when ``|cum signed_weights since last bar| >= threshold``.
    The threshold is updated as ``ema_alpha * |bar_imbalance| + (1-alpha) * threshold``.
    ``min_bar_size`` prevents pathological back-to-back single-tick bars.
    """
    n = signed_weights.shape[0]
    ends = np.empty(n, dtype=np.int64)
    k = 0
    acc = 0.0
    bar_start = 0
    threshold = initial_threshold
    for i in range(n):
        acc += signed_weights[i]
        if abs(acc) >= threshold and (i - bar_start + 1) >= min_bar_size:
            ends[k] = i
            k += 1
            threshold = ema_alpha * abs(acc) + (1.0 - ema_alpha) * threshold
            if threshold < 1e-9:
                threshold = initial_threshold
            acc = 0.0
            bar_start = i + 1
    return ends[:k]


@njit(cache=True)
def _runs_bars(
    signs: np.ndarray,
    initial_threshold: float,
    ema_alpha: float,
    min_bar_size: int,
) -> np.ndarray:
    """Tick-runs bars: emit when max(pos_run, neg_run) crosses the threshold."""
    n = signs.shape[0]
    ends = np.empty(n, dtype=np.int64)
    k = 0
    pos_run = 0.0
    neg_run = 0.0
    bar_start = 0
    threshold = initial_threshold
    for i in range(n):
        s = signs[i]
        if s > 0.0:
            pos_run += 1.0
        elif s < 0.0:
            neg_run += 1.0
        worst = pos_run if pos_run >= neg_run else neg_run
        if worst >= threshold and (i - bar_start + 1) >= min_bar_size:
            ends[k] = i
            k += 1
            threshold = ema_alpha * worst + (1.0 - ema_alpha) * threshold
            if threshold < 1e-9:
                threshold = initial_threshold
            pos_run = 0.0
            neg_run = 0.0
            bar_start = i + 1
    return ends[:k]
