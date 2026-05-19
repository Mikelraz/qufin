"""
Volatility indicators.

Implementations
---------------
* ``true_range``         — Wilder's per-bar true range
* ``atr``                — Wilder's smoothed Average True Range
* ``bollinger_bands``    — SMA midline +/- n_std rolling standard deviations
* ``keltner_channels``   — EMA midline +/- multiplier · ATR
* ``donchian_channels``  — Rolling high / low / midline
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    ema_kernel,
    rolling_max_kernel,
    rolling_min_kernel,
    true_range_kernel,
    wilder_smooth_kernel,
)
from ._types import (
    BollingerBands,
    DonchianChannels,
    KeltnerChannels,
    check_lengths,
    to_numpy_1d,
)
from .moving_averages import sma


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``."""
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    return true_range_kernel(h, l, c)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder's smoothed Average True Range over ``window`` bars."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    tr = true_range(high, low, close)
    return wilder_smooth_kernel(tr, window)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Population rolling standard deviation over ``window`` samples."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cumsum = np.cumsum(x)
    cumsum_sq = np.cumsum(x * x)
    s = np.empty(n - window + 1, dtype=np.float64)
    sq = np.empty(n - window + 1, dtype=np.float64)
    s[0] = cumsum[window - 1]
    sq[0] = cumsum_sq[window - 1]
    s[1:] = cumsum[window:] - cumsum[:-window]
    sq[1:] = cumsum_sq[window:] - cumsum_sq[:-window]
    mean = s / window
    var = sq / window - mean * mean
    var = np.maximum(var, 0.0)
    out[window - 1 :] = np.sqrt(var)
    return out


def bollinger_bands(close: np.ndarray, window: int = 20, n_std: float = 2.0) -> BollingerBands:
    """
    Bollinger Bands.

    ``middle = SMA(close, window)``; ``upper = middle + n_std·sigma``;
    ``lower = middle - n_std·sigma``; sigma is the population rolling std.
    ``bandwidth = (upper - lower) / middle``;
    ``percent_b = (close - lower) / (upper - lower)``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if n_std <= 0.0:
        raise ValueError(f"n_std must be > 0, got {n_std}")
    c = to_numpy_1d(close)
    mid = sma(c, window)
    sd = _rolling_std(c, window)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        bandwidth = np.where(mid != 0.0, (upper - lower) / mid, np.nan)
        width = upper - lower
        percent_b = np.where(width > 0.0, (c - lower) / width, np.nan)
    return BollingerBands(
        middle=mid, upper=upper, lower=lower, bandwidth=bandwidth, percent_b=percent_b
    )


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    atr_window: int = 10,
    atr_mult: float = 2.0,
) -> KeltnerChannels:
    """
    Keltner Channels.

    Midline is ``EMA(close, window)``; envelopes are ``middle ± atr_mult·ATR``
    using Wilder ATR over ``atr_window`` bars.
    """
    if window < 1 or atr_window < 1:
        raise ValueError(f"windows must be >= 1, got ({window}, {atr_window})")
    if atr_mult <= 0.0:
        raise ValueError(f"atr_mult must be > 0, got {atr_mult}")
    c = to_numpy_1d(close)
    mid = ema_kernel(c, window)
    a = atr(high, low, close, window=atr_window)
    upper = mid + atr_mult * a
    lower = mid - atr_mult * a
    return KeltnerChannels(middle=mid, upper=upper, lower=lower)


def donchian_channels(high: np.ndarray, low: np.ndarray, window: int = 20) -> DonchianChannels:
    """Highest high and lowest low over ``window`` bars; midline = (upper+lower)/2."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    check_lengths(h, l)
    upper = rolling_max_kernel(h, window)
    lower = rolling_min_kernel(l, window)
    middle = 0.5 * (upper + lower)
    return DonchianChannels(upper=upper, lower=lower, middle=middle)
