"""Trading-range detection."""

from __future__ import annotations

import numpy as np

from qufin.wyckoff import detect_trading_ranges
from tests.wyckoff.conftest import make_ohlcv


def test_flat_segment_in_trend_yields_a_range() -> None:
    # 30 trending up, 60 lateral, 30 trending up again — lateral segment
    # should produce a detected range.
    n1, n_flat, n3 = 30, 60, 30
    trend1 = np.linspace(50.0, 100.0, n1)
    flat = 100.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_flat))
    trend3 = np.linspace(100.0, 150.0, n3)
    closes = np.concatenate([trend1, flat, trend3])
    opens = closes
    highs = closes + 0.3
    lows = closes - 0.3
    vols = np.full(closes.shape[0], 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    ranges = detect_trading_ranges(bars, min_bars=20, max_width_atr=6.0)
    assert len(ranges) >= 1
    # The detected range should overlap the lateral segment substantially.
    flat_start, flat_end = n1, n1 + n_flat
    tr = ranges[0]
    overlap = max(0, min(tr.end_idx, flat_end) - max(tr.start_idx, flat_start))
    assert overlap >= 0.7 * n_flat
    assert tr.resistance > tr.support


def test_pure_trend_yields_no_range() -> None:
    n = 100
    closes = np.linspace(50.0, 200.0, n)
    opens = closes
    highs = closes + 0.5
    lows = closes - 0.5
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    ranges = detect_trading_ranges(bars, min_bars=20, max_width_atr=2.0)
    assert ranges == []
