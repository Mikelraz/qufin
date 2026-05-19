"""Fractal and ZigZag swing-point detection."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.wyckoff import find_swings, zigzag
from tests.wyckoff.conftest import make_ohlcv


def _sine_bars(n: int = 200, period: int = 20, amp: float = 5.0, base: float = 100.0):
    t = np.arange(n, dtype=np.float64)
    closes = base + amp * np.sin(2.0 * np.pi * t / period)
    opens = closes
    highs = closes + 0.05
    lows = closes - 0.05
    vols = np.full(n, 1.0)
    return make_ohlcv(opens, highs, lows, closes, vols)


def test_fractal_swings_alternate_kinds_on_sine() -> None:
    bars = _sine_bars(n=200, period=20)
    sw = find_swings(bars, left=3, right=3)
    assert len(sw) > 4
    kinds = [s.kind for s in sw]
    # Highs and lows should alternate cleanly on a clean sine.
    for a, b in zip(kinds, kinds[1:]):
        assert a != b


def test_fractal_swings_strength_equals_min_left_right() -> None:
    bars = _sine_bars(period=24)
    sw = find_swings(bars, left=4, right=2)
    assert all(s.strength == 2 for s in sw)


def test_fractal_swings_empty_when_window_too_large() -> None:
    n = 5
    closes = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    bars = make_ohlcv(closes, closes + 0.1, closes - 0.1, closes, np.full(n, 1.0))
    sw = find_swings(bars, left=10, right=10)
    assert sw == []


def test_zigzag_requires_minimum_pct_reversal() -> None:
    n = 300
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    bars = make_ohlcv(closes, closes + 0.05, closes - 0.05, closes, np.full(n, 1.0))
    # Very small reversal → many pivots; very large → almost none.
    fine = zigzag(bars, pct=0.005)
    coarse = zigzag(bars, pct=0.10)
    assert len(fine) > len(coarse)


def test_zigzag_invalid_pct() -> None:
    bars = _sine_bars(n=30)
    with pytest.raises(ValueError):
        zigzag(bars, pct=0.0)
    with pytest.raises(ValueError):
        zigzag(bars, pct=1.5)
