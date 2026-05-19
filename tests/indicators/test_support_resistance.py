"""Tests for support / resistance utilities."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import (
    cluster_levels,
    fibonacci_pivot_points,
    pivot_points,
    pivot_points_series,
)


def test_classic_pivot_known_values() -> None:
    pp = pivot_points(prev_high=110.0, prev_low=90.0, prev_close=105.0)
    # PP = (110+90+105)/3 = 101.667
    assert pp.pp == pytest.approx(305.0 / 3.0)
    assert pp.r1 == pytest.approx(2 * pp.pp - 90.0)
    assert pp.s1 == pytest.approx(2 * pp.pp - 110.0)
    assert pp.r2 == pytest.approx(pp.pp + 20.0)
    assert pp.s2 == pytest.approx(pp.pp - 20.0)


def test_fibonacci_pivot_known_values() -> None:
    pp = fibonacci_pivot_points(prev_high=110.0, prev_low=90.0, prev_close=100.0)
    assert pp.pp == pytest.approx(100.0)
    assert pp.r1 == pytest.approx(100.0 + 0.382 * 20.0)
    assert pp.s1 == pytest.approx(100.0 - 0.382 * 20.0)
    assert pp.r3 == pytest.approx(120.0)
    assert pp.s3 == pytest.approx(80.0)


def test_pivot_series_uses_prior_bar() -> None:
    highs = np.array([100.0, 105.0, 108.0])
    lows = np.array([95.0, 99.0, 101.0])
    closes = np.array([98.0, 102.0, 105.0])
    out = pivot_points_series(highs, lows, closes, method="classic")
    assert np.isnan(out["PP"][0])
    assert out["PP"][1] == pytest.approx((100.0 + 95.0 + 98.0) / 3.0)
    assert out["PP"][2] == pytest.approx((105.0 + 99.0 + 102.0) / 3.0)


def test_pivot_series_invalid_method() -> None:
    with pytest.raises(ValueError):
        pivot_points_series(np.zeros(3), np.zeros(3), np.zeros(3), method="zzz")


def test_cluster_levels_merges_near_prices() -> None:
    prices = np.array([100.0, 100.4, 100.2, 110.0, 110.1, 90.0])
    kinds = ["H", "H", "H", "L", "L", "L"]
    indices = np.array([1, 5, 9, 15, 20, 30])
    levels = cluster_levels(prices, kinds, indices, tolerance=0.01)
    # Expect 3 clusters: ~100, ~110, 90
    assert len(levels) == 3
    centres = sorted(level_.price for level_ in levels)
    assert centres[0] == pytest.approx(90.0)
    assert centres[1] == pytest.approx(100.2, abs=0.05)
    assert centres[2] == pytest.approx(110.05, abs=0.05)


def test_cluster_kinds_assigned_correctly() -> None:
    prices = np.array([100.0, 100.1, 100.05])
    kinds = ["H", "L", "H"]
    indices = np.array([1, 2, 3])
    levels = cluster_levels(prices, kinds, indices, tolerance=0.01)
    assert len(levels) == 1
    assert levels[0].kind == "SR"
    assert levels[0].touches == 3
