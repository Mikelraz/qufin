"""VWAP bands, session VWAP, anchored VWAP."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from qufin.volume_distribution import anchored_vwap, session_vwap, vwap_bands
from tests.volume_distribution.conftest import make_ohlcv, synthetic_ohlcv


def test_bands_straddle_vwap_and_widen_with_mult() -> None:
    bars = synthetic_ohlcv(150, seed=3)
    bands = vwap_bands(bars.high(), bars.low(), bars.close(), bars.volume(), std_mults=(1.0, 2.0))
    valid = ~np.isnan(bands.vwap)
    assert np.all(bands.upper[valid, 0] >= bands.vwap[valid] - 1e-9)
    assert np.all(bands.lower[valid, 0] <= bands.vwap[valid] + 1e-9)
    # 2-sigma band is at least as wide as the 1-sigma band everywhere.
    assert np.all(bands.upper[valid, 1] >= bands.upper[valid, 0] - 1e-9)
    assert np.all(bands.lower[valid, 1] <= bands.lower[valid, 0] + 1e-9)


def test_vwap_bands_reject_empty_mults() -> None:
    bars = synthetic_ohlcv(10)
    with pytest.raises(ValueError):
        vwap_bands(bars.high(), bars.low(), bars.close(), bars.volume(), std_mults=())


def test_session_vwap_resets_per_session() -> None:
    n = 12
    closes = np.concatenate([np.full(6, 10.0), np.full(6, 20.0)])
    bars = make_ohlcv(
        closes, closes + 0.1, closes - 0.1, closes, np.ones(n), freq=timedelta(minutes=20)
    )
    sv = session_vwap(bars, period="1h")
    assert sv[0] == pytest.approx(10.0, abs=0.2)
    # First bar of the second session re-anchors near 20, not a blend toward 10.
    assert sv[6] == pytest.approx(20.0, abs=0.2)


def test_anchored_vwap_starts_at_typical_price() -> None:
    n = 30
    closes = np.linspace(100.0, 110.0, n)
    highs = closes + 0.5
    lows = closes - 0.5
    bars = make_ohlcv(closes, highs, lows, closes, np.ones(n))
    vwap = anchored_vwap(bars, anchor_idx=5)
    assert np.isnan(vwap[:5]).all()
    expected_first = (highs[5] + lows[5] + closes[5]) / 3.0
    assert vwap[5] == pytest.approx(expected_first)
