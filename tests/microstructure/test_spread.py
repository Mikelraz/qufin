"""Spread estimators: Roll, quoted/effective/realized, Corwin-Schultz, Abdi-Ranaldo."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import (
    abdi_ranaldo,
    corwin_schultz,
    effective_spread,
    quoted_spread,
    realized_spread,
    roll_spread,
)
from tests.microstructure.conftest import bid_ask_bounce


def test_roll_recovers_spread_from_bounce() -> None:
    prices, _ = bid_ask_bounce(60_000, spread=0.10, seed=3)
    est = roll_spread(prices)
    assert est == pytest.approx(0.10, abs=0.01)


def test_roll_nan_when_autocovariance_nonnegative() -> None:
    # A trending series has positive autocovariance → Roll model invalid.
    prices = np.cumsum(np.ones(100)) + 100.0
    assert np.isnan(roll_spread(prices))


def test_quoted_spread_absolute_and_relative() -> None:
    bid = np.array([99.95, 99.90])
    ask = np.array([100.05, 100.10])
    np.testing.assert_allclose(quoted_spread(bid, ask), np.array([0.10, 0.20]))
    rel = quoted_spread(bid, ask, relative=True)
    np.testing.assert_allclose(rel, np.array([0.10 / 100.0, 0.20 / 100.0]))


def test_effective_spread_equals_full_spread_under_bounce() -> None:
    prices, q = bid_ask_bounce(2000, mid=100.0, spread=0.10, seed=5)
    bid = np.full_like(prices, 99.95)
    ask = np.full_like(prices, 100.05)
    eff = effective_spread(prices, bid, ask, signs=q)
    np.testing.assert_allclose(eff, 0.10)


def test_realized_spread_shape_and_tail_nan() -> None:
    n = 50
    prices = 100.0 + np.zeros(n)
    mid = 100.0 + np.zeros(n)
    signs = np.ones(n)
    rs = realized_spread(prices, mid, signs=signs, tau=5)
    assert rs.shape == (n,)
    assert np.all(np.isnan(rs[-5:]))
    np.testing.assert_allclose(rs[:-5], 0.0)


def test_corwin_schultz_nonnegative_with_leading_nan() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200)))
    high = close * (1.0 + rng.uniform(0.0, 0.01, size=200))
    low = close * (1.0 - rng.uniform(0.0, 0.01, size=200))
    cs = corwin_schultz(high, low)
    assert cs.shape == (200,)
    assert np.isnan(cs[0])
    assert np.all(cs[1:] >= 0.0)


def test_corwin_schultz_wider_ranges_give_larger_spread() -> None:
    n = 100
    close = np.full(n, 100.0)
    narrow = corwin_schultz(close * 1.001, close * 0.999)
    wide = corwin_schultz(close * 1.01, close * 0.99)
    assert np.nanmean(wide[1:]) > np.nanmean(narrow[1:])


def test_abdi_ranaldo_nonnegative_with_trailing_nan() -> None:
    rng = np.random.default_rng(9)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=150)))
    high = close * (1.0 + rng.uniform(0.0, 0.01, size=150))
    low = close * (1.0 - rng.uniform(0.0, 0.01, size=150))
    ar = abdi_ranaldo(high, low, close)
    assert ar.shape == (150,)
    assert np.isnan(ar[-1])
    assert np.all(ar[:-1] >= 0.0)
