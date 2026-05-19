"""
Moving averages.

Implementations
---------------
* ``sma``  — Simple Moving Average (windowed mean)
* ``ema``  — Exponential MA, alpha = 2 / (window + 1)
* ``wma``  — Linearly-Weighted MA, weights 1..window
* ``dema`` — Double EMA: 2·EMA - EMA(EMA)
* ``tema`` — Triple EMA: 3·EMA - 3·EMA(EMA) + EMA(EMA(EMA))
* ``hma``  — Hull MA: WMA(2·WMA(n/2) - WMA(n), sqrt(n))
* ``kama`` — Kaufman's Adaptive MA

All routines NaN-pad the warm-up region so output length matches input length.
"""

from __future__ import annotations

import math

import numpy as np

from ._kernels import ema_kernel, kama_kernel, wma_kernel
from ._types import to_numpy_1d


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average over ``window`` samples."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = to_numpy_1d(values)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    csum = np.cumsum(x)
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def ema(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average; alpha = 2/(window + 1); seeded with SMA."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = to_numpy_1d(values)
    return ema_kernel(x, window)


def wma(values: np.ndarray, window: int) -> np.ndarray:
    """Linearly-weighted moving average — weights 1..window."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = to_numpy_1d(values)
    return wma_kernel(x, window)


def dema(values: np.ndarray, window: int) -> np.ndarray:
    """Double Exponential MA — reduces EMA lag by extrapolating its trend."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = to_numpy_1d(values)
    e1 = ema_kernel(x, window)
    e2 = ema_kernel(np.where(np.isnan(e1), 0.0, e1), window)
    # Mask warm-up: e2 needs (window - 1) more bars after e1 first becomes valid.
    out = 2.0 * e1 - e2
    warmup = 2 * (window - 1)
    out[:warmup] = np.nan
    return out


def tema(values: np.ndarray, window: int) -> np.ndarray:
    """Triple Exponential MA — further lag reduction beyond DEMA."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = to_numpy_1d(values)
    e1 = ema_kernel(x, window)
    e2 = ema_kernel(np.where(np.isnan(e1), 0.0, e1), window)
    e3 = ema_kernel(np.where(np.isnan(e2), 0.0, e2), window)
    out = 3.0 * e1 - 3.0 * e2 + e3
    warmup = 3 * (window - 1)
    out[:warmup] = np.nan
    return out


def hma(values: np.ndarray, window: int) -> np.ndarray:
    """Hull Moving Average — WMA(2·WMA(n/2) - WMA(n), sqrt(n))."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    x = to_numpy_1d(values)
    half = max(1, window // 2)
    root = max(1, int(round(math.sqrt(window))))
    w_half = wma_kernel(x, half)
    w_full = wma_kernel(x, window)
    inner = 2.0 * w_half - w_full
    # Replace NaN with 0 for the second WMA pass, then re-mask.
    cleaned = np.where(np.isnan(inner), 0.0, inner)
    result = wma_kernel(cleaned, root)
    warmup = window - 1 + root - 1
    result[:warmup] = np.nan
    return result


def kama(values: np.ndarray, window: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Kaufman's Adaptive Moving Average.

    The smoothing constant adapts each bar with the efficiency ratio
    ``ER = |price_t - price_{t-w}| / sum_{k=t-w+1..t} |price_k - price_{k-1}|``.
    ``ER`` is mapped to a smoothing constant between the ``fast`` and ``slow``
    EMA equivalents and squared, producing more responsive averages in trends
    and flatter averages in noisy ranges.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if fast < 1 or slow < 1 or fast >= slow:
        raise ValueError(f"need 1 <= fast < slow, got fast={fast}, slow={slow}")
    x = to_numpy_1d(values)
    return kama_kernel(x, window, fast, slow)
