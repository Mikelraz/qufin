"""
Trading-range detection.

A *trading range* in Wyckoff terms is a lateral price consolidation where
supply and demand are roughly balanced — accumulation or distribution can
occur inside it. We detect ranges by sliding a window over the bars and
flagging segments whose price extreme is narrow relative to local ATR.

Overlapping windows that pass the test are merged into a single maximal range.
"""

from __future__ import annotations

import numpy as np

from ._types import OHLCV, TradingRange
from .bars import atr


def detect_trading_ranges(
    bars: OHLCV,
    *,
    min_bars: int = 20,
    max_width_atr: float = 6.0,
    atr_window: int = 14,
) -> list[TradingRange]:
    """
    Detect lateral consolidations in the bar series.

    A sliding window of ``min_bars`` bars qualifies as a range if its
    high-low width is at most ``max_width_atr × median(ATR)`` over the window.
    Adjacent qualifying windows are merged into the longest contiguous run
    that still respects the width constraint over the merged extent.

    Parameters
    ----------
    bars            OHLCV bar sequence.
    min_bars        Minimum number of bars for a range; default 20.
    max_width_atr   Maximum range width as a multiple of ATR; default 6.0.
    atr_window      Window for ATR baseline; default 14.

    Returns
    -------
    list of TradingRange in chronological order, non-overlapping.
    """
    if min_bars < 2:
        raise ValueError(f"min_bars must be >= 2, got {min_bars}")
    if max_width_atr <= 0.0:
        raise ValueError(f"max_width_atr must be > 0, got {max_width_atr}")
    n = bars.n_bars
    if n < min_bars:
        return []

    high = bars.high()
    low = bars.low()
    a = atr(bars, window=atr_window)

    qualifying = np.zeros(n, dtype=np.bool_)
    for i in range(min_bars - 1, n):
        start = i - min_bars + 1
        window_high = float(high[start : i + 1].max())
        window_low = float(low[start : i + 1].min())
        window_atr = float(np.nanmedian(a[start : i + 1]))
        if not np.isfinite(window_atr) or window_atr <= 0.0:
            continue
        if (window_high - window_low) <= max_width_atr * window_atr:
            qualifying[start : i + 1] = True

    ranges: list[TradingRange] = []
    i = 0
    while i < n:
        if not qualifying[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and qualifying[j + 1]:
            j += 1
        if (j - i + 1) >= min_bars:
            r_high = float(high[i : j + 1].max())
            r_low = float(low[i : j + 1].min())
            if r_high > r_low:
                ranges.append(
                    TradingRange(
                        start_idx=int(i),
                        end_idx=int(j + 1),
                        support=r_low,
                        resistance=r_high,
                    )
                )
        i = j + 1
    return ranges


def is_in_range(tr: TradingRange, bar_idx: int) -> bool:
    """Return True iff ``bar_idx`` lies within ``[tr.start_idx, tr.end_idx)``."""
    return tr.contains_idx(bar_idx)
