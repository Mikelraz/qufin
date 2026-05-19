"""
Momentum oscillators.

Implementations
---------------
* ``rsi``         — Wilder's Relative Strength Index
* ``macd``        — Moving Average Convergence Divergence
* ``stochastic``  — Stochastic Oscillator (%K, %D)
* ``roc``         — Rate of Change (percent)
* ``cci``         — Commodity Channel Index
* ``williams_r``  — Williams %R
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    ema_kernel,
    rolling_max_kernel,
    rolling_min_kernel,
    rsi_kernel,
)
from ._types import MACDResult, StochasticResult, check_lengths, to_numpy_1d
from .moving_averages import sma


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index over ``window`` bars."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    c = to_numpy_1d(close)
    return rsi_kernel(c, window)


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDResult:
    """
    Moving Average Convergence Divergence.

    ``macd = EMA(close, fast) - EMA(close, slow)``;
    ``signal = EMA(macd, signal)``;
    ``hist = macd - signal``.
    """
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError(f"all windows must be >= 1, got ({fast}, {slow}, {signal})")
    if fast >= slow:
        raise ValueError(f"fast must be < slow, got fast={fast}, slow={slow}")
    c = to_numpy_1d(close)
    ema_fast = ema_kernel(c, fast)
    ema_slow = ema_kernel(c, slow)
    macd_line = ema_fast - ema_slow
    # Signal EMA cannot start until slow EMA is valid.
    seed_start = slow - 1
    n = c.shape[0]
    signal_line = np.full(n, np.nan, dtype=np.float64)
    if n - seed_start >= signal:
        sig_tail = ema_kernel(macd_line[seed_start:], signal)
        signal_line[seed_start:] = sig_tail
    hist = macd_line - signal_line
    return MACDResult(macd=macd_line, signal=signal_line, hist=hist)


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_window: int = 14,
    d_window: int = 3,
) -> StochasticResult:
    """Stochastic Oscillator. %K is the close's position within the rolling
    ``[lowest_low, highest_high]`` over ``k_window``; %D is the SMA of %K."""
    if k_window < 1 or d_window < 1:
        raise ValueError(f"windows must be >= 1, got ({k_window}, {d_window})")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    hh = rolling_max_kernel(h, k_window)
    ll = rolling_min_kernel(l, k_window)
    rng = hh - ll
    with np.errstate(invalid="ignore"):
        k = np.where(rng > 0.0, 100.0 * (c - ll) / rng, 50.0)
    k[: k_window - 1] = np.nan
    d = sma(k, d_window)
    return StochasticResult(k=k, d=d)


def roc(close: np.ndarray, window: int = 12) -> np.ndarray:
    """Rate of Change in percent: ``100 · (close_t / close_{t-window} - 1)``."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    c = to_numpy_1d(close)
    n = c.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = 100.0 * (c[window:] / c[:-window] - 1.0)
    return out


def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Commodity Channel Index.

    ``CCI = (TP - SMA(TP)) / (0.015 · MAD(TP))`` where ``TP = (H+L+C)/3`` and
    ``MAD`` is the mean absolute deviation over ``window`` bars.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    tp = (h + l + c) / 3.0
    tp_sma = sma(tp, window)
    n = tp.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        m = tp_sma[i]
        s = 0.0
        for k in range(i - window + 1, i + 1):
            s += abs(tp[k] - m)
        mad = s / window
        out[i] = (tp[i] - m) / (0.015 * mad) if mad > 0.0 else 0.0
    return out


def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Williams %R: ``-100 · (highest_high - close) / (highest_high - lowest_low)``."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    hh = rolling_max_kernel(h, window)
    ll = rolling_min_kernel(l, window)
    rng = hh - ll
    with np.errstate(invalid="ignore"):
        out = np.where(rng > 0.0, -100.0 * (hh - c) / rng, 0.0)
    out[: window - 1] = np.nan
    return out
