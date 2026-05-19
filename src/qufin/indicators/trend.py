"""
Trend / directional indicators.

Implementations
---------------
* ``adx``             — Average Directional Index with +DI and -DI
* ``aroon``           — Aroon Up, Aroon Down, and Oscillator
* ``parabolic_sar``   — Wilder's Parabolic SAR
* ``supertrend``      — ATR-anchored Supertrend line and trend direction
* ``ichimoku``        — Ichimoku Kinko Hyo (Tenkan / Kijun / Senkou A,B / Chikou)
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    adx_kernel,
    parabolic_sar_kernel,
    rolling_argmax_offset_kernel,
    rolling_argmin_offset_kernel,
    rolling_max_kernel,
    rolling_min_kernel,
    supertrend_kernel,
)
from ._types import (
    ADXResult,
    AroonResult,
    IchimokuResult,
    SupertrendResult,
    check_lengths,
    to_numpy_1d,
)
from .volatility import atr


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> ADXResult:
    """
    Wilder's Average Directional Index with ``+DI`` and ``-DI``.

    The ADX series starts ``2·window`` bars in (one window to seed DI, another
    to average the DX values), per Wilder.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    plus_di, minus_di, adx_arr = adx_kernel(h, l, c, window)
    return ADXResult(adx=adx_arr, plus_di=plus_di, minus_di=minus_di)


def aroon(high: np.ndarray, low: np.ndarray, window: int = 25) -> AroonResult:
    """
    Aroon indicator. ``up = 100·(window - bars_since_high)/window``;
    ``down = 100·(window - bars_since_low)/window``; oscillator = up - down.

    The trailing window includes the current bar (``window + 1`` distinct bars
    considered when computing bars-since-extreme, the standard convention).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    check_lengths(h, l)
    span = window + 1
    bars_since_high = rolling_argmax_offset_kernel(h, span)
    bars_since_low = rolling_argmin_offset_kernel(l, span)
    up = 100.0 * (window - bars_since_high) / window
    down = 100.0 * (window - bars_since_low) / window
    return AroonResult(up=up, down=down, oscillator=up - down)


def parabolic_sar(
    high: np.ndarray,
    low: np.ndarray,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> SupertrendResult:
    """
    Wilder's Parabolic SAR.

    Returns a :class:`SupertrendResult` whose ``line`` holds the SAR price and
    ``direction`` holds the trend (+1 long, -1 short).
    """
    if not (0.0 < af_start <= af_max):
        raise ValueError(f"need 0 < af_start <= af_max, got {af_start}, {af_max}")
    if af_step <= 0.0:
        raise ValueError(f"af_step must be > 0, got {af_step}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    check_lengths(h, l)
    sar, direction = parabolic_sar_kernel(h, l, af_start, af_step, af_max)
    return SupertrendResult(line=sar, direction=direction)


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 10,
    multiplier: float = 3.0,
) -> SupertrendResult:
    """
    Supertrend indicator.

    Bands centred on ``(high + low) / 2`` and offset by ``multiplier · ATR``.
    The line tracks the lower band in uptrends and the upper band in
    downtrends, flipping when ``close`` crosses the active band.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if multiplier <= 0.0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    a = atr(h, l, c, window=window)
    line, direction = supertrend_kernel(h, l, c, a, multiplier)
    return SupertrendResult(line=line, direction=direction)


def ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> IchimokuResult:
    """
    Ichimoku Kinko Hyo.

    * Tenkan-sen: midpoint of the last ``tenkan`` bars' high/low.
    * Kijun-sen: midpoint of the last ``kijun`` bars' high/low.
    * Senkou Span A: ``(tenkan + kijun) / 2``, shifted forward by ``displacement``.
    * Senkou Span B: midpoint of the last ``senkou_b`` bars, shifted forward.
    * Chikou Span: close shifted *back* by ``displacement``.

    Senkou spans returned arrays have ``displacement`` trailing entries that
    project the cloud into the future; Chikou's last ``displacement`` entries
    are NaN since they correspond to future closes.
    """
    if min(tenkan, kijun, senkou_b, displacement) < 1:
        raise ValueError("all parameters must be >= 1")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    n = h.shape[0]

    def mid(w: int) -> np.ndarray:
        return 0.5 * (rolling_max_kernel(h, w) + rolling_min_kernel(l, w))

    tenkan_line = mid(tenkan)
    kijun_line = mid(kijun)
    senkou_a_raw = 0.5 * (tenkan_line + kijun_line)
    senkou_b_raw = mid(senkou_b)

    span_len = n + displacement
    senkou_a = np.full(span_len, np.nan, dtype=np.float64)
    senkou_b_arr = np.full(span_len, np.nan, dtype=np.float64)
    senkou_a[displacement:] = senkou_a_raw
    senkou_b_arr[displacement:] = senkou_b_raw

    chikou = np.full(n, np.nan, dtype=np.float64)
    if n > displacement:
        chikou[: n - displacement] = c[displacement:]

    return IchimokuResult(
        tenkan=tenkan_line,
        kijun=kijun_line,
        senkou_a=senkou_a,
        senkou_b=senkou_b_arr,
        chikou=chikou,
    )
