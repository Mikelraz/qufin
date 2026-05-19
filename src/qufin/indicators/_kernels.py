"""
Numba-jitted hot loops for the indicators subpackage.

Only inherently-recursive or hard-to-vectorise routines live here. Anything
that maps cleanly to a windowed numpy/polars expression stays in the
respective public module.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def ema_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Exponential MA with seed = SMA of first ``window`` values; alpha = 2/(w+1)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    s = 0.0
    for i in range(window):
        s += x[i]
    prev = s / window
    out[window - 1] = prev
    alpha = 2.0 / (window + 1)
    one_minus = 1.0 - alpha
    for i in range(window, n):
        prev = alpha * x[i] + one_minus * prev
        out[i] = prev
    return out


@njit(cache=True)
def wilder_smooth_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Wilder's exponential smoothing: seed = mean of first ``window``; alpha = 1/w."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    s = 0.0
    for i in range(window):
        s += x[i]
    prev = s / window
    out[window - 1] = prev
    for i in range(window, n):
        prev = (prev * (window - 1) + x[i]) / window
        out[i] = prev
    return out


@njit(cache=True)
def wilder_smooth_sum_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Wilder cumulative form: TR_sum_t = TR_sum_{t-1} - TR_sum_{t-1}/w + x_t."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    s = 0.0
    for i in range(window):
        s += x[i]
    out[window - 1] = s
    for i in range(window, n):
        s = s - s / window + x[i]
        out[i] = s
    return out


@njit(cache=True)
def rsi_kernel(close: np.ndarray, window: int) -> np.ndarray:
    """Wilder's RSI."""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window, n)):
        out[i] = np.nan
    if n <= window:
        return out
    gain = 0.0
    loss = 0.0
    for i in range(1, window + 1):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            gain += diff
        else:
            loss -= diff
    avg_gain = gain / window
    avg_loss = loss / window
    if avg_loss == 0.0:
        out[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[window] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(window + 1, n):
        diff = close[i] - close[i - 1]
        g = diff if diff > 0.0 else 0.0
        l_ = -diff if diff < 0.0 else 0.0
        avg_gain = (avg_gain * (window - 1) + g) / window
        avg_loss = (avg_loss * (window - 1) + l_) / window
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


@njit(cache=True)
def true_range_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
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


@njit(cache=True)
def adx_kernel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute +DI, -DI, ADX in a single pass with Wilder smoothing."""
    n = high.shape[0]
    plus_di = np.empty(n, dtype=np.float64)
    minus_di = np.empty(n, dtype=np.float64)
    adx_out = np.empty(n, dtype=np.float64)
    for i in range(n):
        plus_di[i] = np.nan
        minus_di[i] = np.nan
        adx_out[i] = np.nan
    if n < 2 * window + 1:
        return plus_di, minus_di, adx_out
    tr_sum = 0.0
    plus_dm_sum = 0.0
    minus_dm_sum = 0.0
    for i in range(1, window + 1):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm = up if up > dn and up > 0.0 else 0.0
        minus_dm = dn if dn > up and dn > 0.0 else 0.0
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr = hl
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        tr_sum += tr
        plus_dm_sum += plus_dm
        minus_dm_sum += minus_dm
    pdi_w = 100.0 * plus_dm_sum / tr_sum if tr_sum > 0.0 else 0.0
    mdi_w = 100.0 * minus_dm_sum / tr_sum if tr_sum > 0.0 else 0.0
    plus_di[window] = pdi_w
    minus_di[window] = mdi_w
    dx_buf = np.empty(window, dtype=np.float64)
    denom = pdi_w + mdi_w
    dx_buf[0] = 100.0 * abs(pdi_w - mdi_w) / denom if denom > 0.0 else 0.0
    dx_idx = 1
    for i in range(window + 1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm = up if up > dn and up > 0.0 else 0.0
        minus_dm = dn if dn > up and dn > 0.0 else 0.0
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr = hl
        if hc > tr:
            tr = hc
        if lc > tr:
            tr = lc
        tr_sum = tr_sum - tr_sum / window + tr
        plus_dm_sum = plus_dm_sum - plus_dm_sum / window + plus_dm
        minus_dm_sum = minus_dm_sum - minus_dm_sum / window + minus_dm
        pdi = 100.0 * plus_dm_sum / tr_sum if tr_sum > 0.0 else 0.0
        mdi = 100.0 * minus_dm_sum / tr_sum if tr_sum > 0.0 else 0.0
        plus_di[i] = pdi
        minus_di[i] = mdi
        denom = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / denom if denom > 0.0 else 0.0
        if dx_idx < window:
            dx_buf[dx_idx] = dx
            dx_idx += 1
            if dx_idx == window:
                s = 0.0
                for k in range(window):
                    s += dx_buf[k]
                adx_val = s / window
                adx_out[i] = adx_val
        else:
            adx_val = (adx_out[i - 1] * (window - 1) + dx) / window
            adx_out[i] = adx_val
    return plus_di, minus_di, adx_out


@njit(cache=True)
def parabolic_sar_kernel(
    high: np.ndarray, low: np.ndarray, af_start: float, af_step: float, af_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wilder's Parabolic SAR. Returns (sar, direction) with direction +1/-1."""
    n = high.shape[0]
    sar = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.float64)
    if n == 0:
        return sar, direction
    # Initialise: guess trend from first two bars.
    if n == 1:
        sar[0] = low[0]
        direction[0] = 1.0
        return sar, direction
    up = high[1] >= high[0]
    if up:
        sar_val = low[0]
        ep = high[1]
    else:
        sar_val = high[0]
        ep = low[1]
    af = af_start
    sar[0] = sar_val
    direction[0] = 1.0 if up else -1.0
    sar[1] = sar_val
    direction[1] = 1.0 if up else -1.0
    for i in range(2, n):
        prev_sar = sar_val
        sar_val = prev_sar + af * (ep - prev_sar)
        if up:
            # SAR cannot exceed prior two lows.
            if sar_val > low[i - 1]:
                sar_val = low[i - 1]
            if i >= 2 and sar_val > low[i - 2]:
                sar_val = low[i - 2]
            if low[i] < sar_val:
                up = False
                sar_val = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af += af_step
                    if af > af_max:
                        af = af_max
        else:
            if sar_val < high[i - 1]:
                sar_val = high[i - 1]
            if i >= 2 and sar_val < high[i - 2]:
                sar_val = high[i - 2]
            if high[i] > sar_val:
                up = True
                sar_val = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af += af_step
                    if af > af_max:
                        af = af_max
        sar[i] = sar_val
        direction[i] = 1.0 if up else -1.0
    return sar, direction


@njit(cache=True)
def supertrend_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Supertrend line and direction given a pre-computed ATR series."""
    n = high.shape[0]
    line = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.float64)
    for i in range(n):
        line[i] = np.nan
        direction[i] = np.nan
    # Find first index with a valid ATR.
    start = -1
    for i in range(n):
        if not np.isnan(atr[i]):
            start = i
            break
    if start < 0:
        return line, direction
    hl2 = 0.5 * (high[start] + low[start])
    upper = hl2 + multiplier * atr[start]
    lower = hl2 - multiplier * atr[start]
    # Initial trend assumption: long.
    dir_up = True
    line[start] = lower
    direction[start] = 1.0
    prev_upper = upper
    prev_lower = lower
    for i in range(start + 1, n):
        hl2 = 0.5 * (high[i] + low[i])
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]
        final_upper = (
            basic_upper if basic_upper < prev_upper or close[i - 1] > prev_upper else prev_upper
        )
        final_lower = (
            basic_lower if basic_lower > prev_lower or close[i - 1] < prev_lower else prev_lower
        )
        if dir_up:
            if close[i] < final_lower:
                dir_up = False
                line[i] = final_upper
            else:
                line[i] = final_lower
        else:
            if close[i] > final_upper:
                dir_up = True
                line[i] = final_lower
            else:
                line[i] = final_upper
        direction[i] = 1.0 if dir_up else -1.0
        prev_upper = final_upper
        prev_lower = final_lower
    return line, direction


@njit(cache=True)
def kama_kernel(x: np.ndarray, window: int, fast: int, slow: int) -> np.ndarray:
    """Kaufman's Adaptive Moving Average."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window, n)):
        out[i] = np.nan
    if n <= window:
        return out
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    out[window] = x[window]
    prev = x[window]
    for i in range(window + 1, n):
        change = abs(x[i] - x[i - window])
        vol = 0.0
        for k in range(i - window + 1, i + 1):
            vol += abs(x[k] - x[k - 1])
        er = change / vol if vol > 0.0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev = prev + sc * (x[i] - prev)
        out[i] = prev
    return out


@njit(cache=True)
def obv_kernel(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


@njit(cache=True)
def rolling_max_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling max; NaN-padded for the warm-up."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        for k in range(i - window + 2, i + 1):
            if x[k] > m:
                m = x[k]
        out[i] = m
    return out


@njit(cache=True)
def rolling_min_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling min; NaN-padded for the warm-up."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        for k in range(i - window + 2, i + 1):
            if x[k] < m:
                m = x[k]
        out[i] = m
    return out


@njit(cache=True)
def rolling_argmax_offset_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """For each i, bars-since-max over the trailing ``window``. NaN-padded warm-up."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        idx = 0
        for k in range(1, window):
            v = x[i - window + 1 + k]
            if v >= m:
                m = v
                idx = k
        out[i] = window - 1 - idx
    return out


@njit(cache=True)
def rolling_argmin_offset_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """For each i, bars-since-min over the trailing ``window``. NaN-padded warm-up."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        idx = 0
        for k in range(1, window):
            v = x[i - window + 1 + k]
            if v <= m:
                m = v
                idx = k
        out[i] = window - 1 - idx
    return out


@njit(cache=True)
def wma_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """Linearly-weighted moving average — weights 1..window."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(min(window - 1, n)):
        out[i] = np.nan
    if n < window:
        return out
    denom = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for k in range(window):
            s += (k + 1) * x[i - window + 1 + k]
        out[i] = s / denom
    return out
