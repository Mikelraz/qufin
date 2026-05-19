"""
Volume-weighted price indicators.

Implementations
---------------
* ``obv``                       — On-Balance Volume
* ``vwap``                      — Cumulative Volume-Weighted Average Price
* ``rolling_vwap``              — VWAP over a trailing window
* ``mfi``                       — Money Flow Index
* ``cmf``                       — Chaikin Money Flow
* ``accumulation_distribution`` — A/D line (Chaikin)
"""

from __future__ import annotations

import numpy as np

from ._kernels import obv_kernel
from ._types import check_lengths, to_numpy_1d


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume — cumulative signed volume keyed off close direction."""
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(c, v)
    return obv_kernel(c, v)


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Cumulative Volume-Weighted Average Price using the typical price."""
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    tp = (h + l + c) / 3.0
    cum_vp = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    out = np.full(cum_v.shape[0], np.nan, dtype=np.float64)
    mask = cum_v > 0.0
    out[mask] = cum_vp[mask] / cum_v[mask]
    return out


def rolling_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """VWAP over a trailing window of ``window`` bars."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    tp = (h + l + c) / 3.0
    pv = tp * v
    n = h.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cpv = np.cumsum(pv)
    cv = np.cumsum(v)
    win_pv = np.empty(n - window + 1, dtype=np.float64)
    win_v = np.empty(n - window + 1, dtype=np.float64)
    win_pv[0] = cpv[window - 1]
    win_v[0] = cv[window - 1]
    win_pv[1:] = cpv[window:] - cpv[:-window]
    win_v[1:] = cv[window:] - cv[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(win_v > 0.0, win_pv / win_v, np.nan)
    out[window - 1 :] = result
    return out


def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Money Flow Index over ``window`` bars."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    n = c.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out
    tp = (h + l + c) / 3.0
    mf = tp * v
    pos = np.zeros(n, dtype=np.float64)
    neg = np.zeros(n, dtype=np.float64)
    diff = np.diff(tp)
    pos[1:][diff > 0.0] = mf[1:][diff > 0.0]
    neg[1:][diff < 0.0] = mf[1:][diff < 0.0]
    cpos = np.cumsum(pos)
    cneg = np.cumsum(neg)
    for i in range(window, n):
        p = cpos[i] - cpos[i - window]
        ng = cneg[i] - cneg[i - window]
        if ng == 0.0:
            out[i] = 100.0 if p > 0.0 else 50.0
        else:
            ratio = p / ng
            out[i] = 100.0 - 100.0 / (1.0 + ratio)
    return out


def accumulation_distribution(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Chaikin Accumulation/Distribution line — cumulative money-flow volume."""
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = np.where(rng > 0.0, ((c - l) - (h - c)) / rng, 0.0)
    return np.cumsum(mfm * v)


def cmf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Chaikin Money Flow over ``window`` bars: rolling sum of money-flow
    volume divided by rolling sum of volume."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = np.where(rng > 0.0, ((c - l) - (h - c)) / rng, 0.0)
    mfv = mfm * v
    n = h.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cmfv = np.cumsum(mfv)
    cv = np.cumsum(v)
    win_mfv = np.empty(n - window + 1, dtype=np.float64)
    win_v = np.empty(n - window + 1, dtype=np.float64)
    win_mfv[0] = cmfv[window - 1]
    win_v[0] = cv[window - 1]
    win_mfv[1:] = cmfv[window:] - cmfv[:-window]
    win_v[1:] = cv[window:] - cv[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(win_v > 0.0, win_mfv / win_v, np.nan)
    out[window - 1 :] = result
    return out
