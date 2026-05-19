"""
OHLCV bar utilities — validation, true-range, ATR, resampling, volume normalisation.

Polars-first for tabular ops; numba-jitted scalar loops for the rolling
statistics used downstream by event detection.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from ._types import BAR_SCHEMA, OHLCV


def validate_ohlcv(bars: OHLCV, *, require_positive_volume: bool = False) -> None:
    """
    Run thorough invariant checks on an ``OHLCV`` frame.

    Checks
    ------
    * Schema matches ``BAR_SCHEMA``.
    * No NaN / inf in any numeric column.
    * ``high >= max(open, close)`` and ``low <= min(open, close)`` per bar.
    * Timestamps strictly monotonic.
    * Optionally, ``volume >= 0`` (always) or ``volume > 0`` (strict).
    """
    df = bars.data
    missing = set(BAR_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    for col in ("open", "high", "low", "close", "volume"):
        if df[col].is_nan().any() or df[col].is_null().any():
            raise ValueError(f"column {col!r} contains NaN/null")
        if not np.isfinite(df[col].to_numpy()).all():
            raise ValueError(f"column {col!r} contains inf")

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()  # noqa: E741 — financial convention
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()

    if not (h >= np.maximum(o, c)).all():
        raise ValueError("high must be >= max(open, close) for every bar")
    if not (l <= np.minimum(o, c)).all():
        raise ValueError("low must be <= min(open, close) for every bar")
    if not (h >= l).all():
        raise ValueError("high must be >= low for every bar")

    if (v < 0).any():
        raise ValueError("volume must be non-negative")
    if require_positive_volume and (v <= 0).any():
        raise ValueError("volume must be strictly positive")

    ts = df["timestamp"]
    if df.height >= 2 and not ts.is_sorted():
        raise ValueError("timestamps must be sorted ascending")
    if df.height >= 2:
        diffs = ts.diff().drop_nulls()
        if (diffs.dt.total_nanoseconds() <= 0).any():
            raise ValueError("timestamps must be strictly increasing")


@njit(cache=True)
def _true_range_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = high.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        m = hl
        if hc > m:
            m = hc
        if lc > m:
            m = lc
        out[i] = m
    return out


def true_range(bars: OHLCV) -> np.ndarray:
    """
    Wilder's true range per bar.

    ``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``.
    The first element is the bar's high-low (no prior close available).
    """
    return _true_range_kernel(bars.high(), bars.low(), bars.close())


@njit(cache=True)
def _wilder_atr_kernel(tr: np.ndarray, window: int) -> np.ndarray:
    n = tr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    s = 0.0
    for i in range(window):
        s += tr[i]
    atr = s / window
    out[window - 1] = atr
    for i in range(window, n):
        atr = (atr * (window - 1) + tr[i]) / window
        out[i] = atr
    return out


def atr(bars: OHLCV, window: int = 14) -> np.ndarray:
    """
    Wilder's smoothed Average True Range over ``window`` bars.

    The first ``window - 1`` elements are NaN. Index ``window - 1`` is the
    simple mean of the first ``window`` true ranges; subsequent elements use
    Wilder's exponential smoothing ``ATR_t = (ATR_{t-1}·(w-1) + TR_t)/w``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    tr = true_range(bars)
    return _wilder_atr_kernel(tr, window)


def resample(bars: OHLCV, every: str) -> OHLCV:
    """
    Aggregate bars to a coarser timestep using a polars duration string.

    ``every`` follows polars semantics: e.g. ``'1h'``, ``'1d'``, ``'1w'``.
    Aggregations: open = first, high = max, low = min, close = last,
    volume = sum. Empty windows are dropped.
    """
    grouped = (
        bars.data.sort("timestamp")
        .group_by_dynamic("timestamp", every=every, closed="left", label="left")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
    )
    return OHLCV.from_records(grouped, symbol=bars.symbol)


def normalize_volume(bars: OHLCV, window: int = 50) -> np.ndarray:
    """
    Rolling z-score of volume over a trailing window.

    Useful for spotting volume climaxes regardless of absolute size.
    Returns NaN for the first ``window - 1`` bars.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    v = bars.volume()
    return rolling_zscore(v, window)


def bar_range_zscore(bars: OHLCV, window: int = 50) -> np.ndarray:
    """Rolling z-score of (high - low) bar range. NaN-padded for the warmup."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    rng = bars.high() - bars.low()
    return rolling_zscore(rng, window)


@njit(cache=True)
def rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    s = 0.0
    s2 = 0.0
    for i in range(window):
        s += x[i]
        s2 += x[i] * x[i]
    for i in range(window - 1, n):
        if i >= window:
            old = x[i - window]
            s += x[i] - old
            s2 += x[i] * x[i] - old * old
        mean = s / window
        var = s2 / window - mean * mean
        if var <= 0.0:
            out[i] = 0.0
        else:
            out[i] = (x[i] - mean) / np.sqrt(var)
    return out


@njit(cache=True)
def _rolling_slope(x: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of y = a + b·t over a trailing window of length ``window``."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    # Pre-compute time-axis stats which are window-invariant.
    t_mean = (window - 1) / 2.0
    sst = 0.0
    for k in range(window):
        d = k - t_mean
        sst += d * d
    for i in range(window - 1, n):
        y_mean = 0.0
        for k in range(window):
            y_mean += x[i - window + 1 + k]
        y_mean /= window
        num = 0.0
        for k in range(window):
            num += (k - t_mean) * (x[i - window + 1 + k] - y_mean)
        out[i] = num / sst
    return out


def rolling_slope(x: np.ndarray, window: int) -> np.ndarray:
    """
    Trailing OLS slope of ``x`` over ``window`` samples.

    NaN-padded for the first ``window - 1`` positions. Used by event detectors
    to qualify the trend leading into a candidate climax bar.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    return _rolling_slope(np.ascontiguousarray(x, dtype=np.float64), window)
