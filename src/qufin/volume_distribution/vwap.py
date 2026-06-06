"""
VWAP-family indicators: cumulative VWAP with volume-weighted standard-deviation
bands, per-session VWAP, and anchored VWAP.

All VWAPs use the typical price ``(high + low + close) / 3`` as the per-bar
contribution, matching :func:`qufin.indicators.vwap`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..data._types import OHLCV
from ._types import VWAPBands, check_lengths, coerce_ohlcv, to_numpy_1d

__all__ = ["anchored_vwap", "session_vwap", "vwap_bands"]


def vwap_bands(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    std_mults: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> VWAPBands:
    """
    Cumulative VWAP with volume-weighted standard-deviation envelopes.

    The band width at bar ``t`` is ``k · σ_t`` where ``σ_t`` is the
    volume-weighted standard deviation of the typical price up to ``t``:
    ``σ_t² = Σ w·(tp - vwap)² / Σ w`` with ``w`` the per-bar volume.
    """
    if not std_mults:
        raise ValueError("std_mults must be non-empty")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    v = to_numpy_1d(volume)
    check_lengths(h, l, c, v)
    tp = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    cum_vp = np.cumsum(tp * v)
    cum_vp2 = np.cumsum(tp * tp * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cum_v > 0.0, cum_vp / cum_v, np.nan)
        var = np.where(cum_v > 0.0, cum_vp2 / cum_v - vwap * vwap, np.nan)
    var = np.maximum(var, 0.0)  # guard tiny negative values from rounding
    sigma = np.sqrt(var)
    mults = np.asarray(std_mults, dtype=np.float64)
    upper = vwap[:, None] + mults[None, :] * sigma[:, None]
    lower = vwap[:, None] - mults[None, :] * sigma[:, None]
    return VWAPBands(vwap=vwap, upper=upper, lower=lower, std_mults=tuple(std_mults))


def session_vwap(bars: OHLCV | pl.DataFrame, period: str = "1d") -> np.ndarray:
    """
    VWAP that resets at the start of every ``period`` session.

    ``period`` is a polars duration string (``"1d"``, ``"1h"`` …). Returns an
    array of length ``len(bars)`` aligned to the input bar order.
    """
    ohlcv = coerce_ohlcv(bars)
    n = ohlcv.n_bars
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    frame = ohlcv.data.with_row_index("__i__").with_columns(
        pl.col("timestamp").dt.truncate(period).alias("__session__")
    )
    h = ohlcv.high()
    l = ohlcv.low()  # noqa: E741
    c = ohlcv.close()
    v = ohlcv.volume()
    tp = (h + l + c) / 3.0
    for session in frame.partition_by("__session__", maintain_order=True):
        idx = session["__i__"].to_numpy()
        cum_vp = np.cumsum(tp[idx] * v[idx])
        cum_v = np.cumsum(v[idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(cum_v > 0.0, cum_vp / cum_v, np.nan)
    return out


def anchored_vwap(bars: OHLCV | pl.DataFrame, anchor_idx: int) -> np.ndarray:
    """
    Running volume-weighted average price from ``anchor_idx`` forward.

    Uses the typical price ``(high + low + close) / 3`` as the per-bar
    contribution. Returns an array of length ``len(bars)``; entries before
    ``anchor_idx`` are NaN.
    """
    ohlcv = coerce_ohlcv(bars)
    n = ohlcv.n_bars
    if not 0 <= anchor_idx < n:
        raise ValueError(f"anchor_idx out of range: {anchor_idx} for {n} bars")
    h = ohlcv.high()
    l = ohlcv.low()  # noqa: E741
    c = ohlcv.close()
    v = ohlcv.volume()
    tp = (h + l + c) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(anchor_idx, n):
        cum_pv += tp[i] * v[i]
        cum_v += v[i]
        out[i] = cum_pv / cum_v if cum_v > 0.0 else np.nan
    return out
