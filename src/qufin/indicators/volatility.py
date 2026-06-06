"""
Volatility indicators.

Implementations
---------------
* ``true_range``         — Wilder's per-bar true range
* ``atr``                — Wilder's smoothed Average True Range
* ``bollinger_bands``    — SMA midline +/- n_std rolling standard deviations
* ``keltner_channels``   — EMA midline +/- multiplier · ATR
* ``donchian_channels``  — Rolling high / low / midline

Range-based realized-volatility estimators (annualised)
* ``parkinson``          — Parkinson (1980) high-low
* ``garman_klass``       — Garman-Klass (1980) OHLC
* ``rogers_satchell``    — Rogers-Satchell (1991), drift-robust
* ``yang_zhang``         — Yang-Zhang (2000), drift- + gap-robust
"""

from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# Range-based realized-volatility estimators
# ---------------------------------------------------------------------------
#
# These exploit the full OHLC bar (not just close-to-close) to estimate
# volatility far more efficiently than the classical close-only estimator.
# Each returns a trailing ``window``-bar volatility, annualised by
# √``trading_periods``.

_LN2 = math.log(2.0)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over ``window`` samples; first ``window − 1`` entries NaN."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cumsum = np.cumsum(x)
    s = np.empty(n - window + 1, dtype=np.float64)
    s[0] = cumsum[window - 1]
    s[1:] = cumsum[window:] - cumsum[:-window]
    out[window - 1 :] = s / window
    return out


def _rolling_var_sample(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample variance (ddof = 1) over ``window`` samples."""
    mean = _rolling_mean(x, window)
    mean_sq = _rolling_mean(x * x, window)
    var_pop = np.maximum(mean_sq - mean * mean, 0.0)
    return var_pop * (window / (window - 1.0))


def parkinson(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 20,
    trading_periods: float = 252.0,
) -> np.ndarray:
    """
    Parkinson (1980) high-low range volatility (annualised).

    ``σ² = 1/(4 ln2) · mean[ ln(H/L)² ]`` over the trailing ``window``.  Uses
    the intraday range only; ~5× more efficient than the close-to-close
    estimator but assumes no drift and continuous trading.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    check_lengths(h, l)
    hl2 = np.log(h / l) ** 2
    var = _rolling_mean(hl2, window) / (4.0 * _LN2)
    return np.sqrt(var * trading_periods)


def garman_klass(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    trading_periods: float = 252.0,
) -> np.ndarray:
    """
    Garman-Klass (1980) OHLC volatility (annualised).

    ``σ² = mean[ ½ ln(H/L)² − (2 ln2 − 1) ln(C/O)² ]`` over the trailing
    ``window``.  More efficient than Parkinson by adding the open-close term.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    o = to_numpy_1d(open_)
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(o, h, l, c)
    term = 0.5 * np.log(h / l) ** 2 - (2.0 * _LN2 - 1.0) * np.log(c / o) ** 2
    var = _rolling_mean(term, window)
    return np.sqrt(np.maximum(var, 0.0) * trading_periods)


def rogers_satchell(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    trading_periods: float = 252.0,
) -> np.ndarray:
    """
    Rogers-Satchell (1991) OHLC volatility (annualised).

    ``σ² = mean[ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]``.  Unlike Parkinson /
    Garman-Klass it is unbiased in the presence of drift.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    o = to_numpy_1d(open_)
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(o, h, l, c)
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    var = _rolling_mean(term, window)
    return np.sqrt(np.maximum(var, 0.0) * trading_periods)


def yang_zhang(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    trading_periods: float = 252.0,
) -> np.ndarray:
    """
    Yang-Zhang (2000) OHLC volatility (annualised).

    The minimum-variance, drift-independent estimator that also accounts for
    overnight gaps:

        σ²_YZ = σ²_overnight + k·σ²_open-close + (1 − k)·σ²_Rogers-Satchell,
        k = 0.34 / (1.34 + (n + 1)/(n − 1)).

    Needs the previous close, so the first ``window`` bars are ``NaN``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    o = to_numpy_1d(open_)
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    n = check_lengths(o, h, l, c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return out

    overnight = np.log(o[1:] / c[:-1])
    open_close = np.log(c[1:] / o[1:])
    rs = np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(l[1:] / c[1:]) * np.log(
        l[1:] / o[1:]
    )

    var_o = _rolling_var_sample(overnight, window)
    var_c = _rolling_var_sample(open_close, window)
    var_rs = _rolling_mean(rs, window)
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    out[1:] = np.sqrt(np.maximum(var_yz, 0.0) * trading_periods)
    return out
