"""Trade-sign classification: tick, quote, Lee-Ready, EMO, BVC."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import bvc, emo_rule, lee_ready, quote_rule, tick_rule


def test_tick_rule_monotone_uptick_all_buys() -> None:
    signs = tick_rule(np.arange(1.0, 11.0))
    assert signs[0] == 0.0
    assert np.all(signs[1:] == 1.0)


def test_tick_rule_carries_sign_on_flat() -> None:
    # up, flat, flat, down → +1, +1 (carry), +1 (carry), -1
    signs = tick_rule(np.array([1.0, 2.0, 2.0, 2.0, 1.0]))
    np.testing.assert_array_equal(signs, np.array([0.0, 1.0, 1.0, 1.0, -1.0]))


def test_quote_rule_uses_midpoint() -> None:
    price = np.array([10.4, 9.6, 10.0])
    bid = np.array([9.5, 9.5, 9.5])
    ask = np.array([10.5, 10.5, 10.5])
    signs = quote_rule(price, bid, ask)
    np.testing.assert_array_equal(signs, np.array([1.0, -1.0, 0.0]))


def test_lee_ready_tiebreak_at_mid_uses_tick() -> None:
    # All trades exactly at the midpoint (10.0); signs follow the tick rule.
    price = np.array([10.0, 10.0, 10.0])
    bid = np.full(3, 9.5)
    ask = np.full(3, 10.5)
    # Tick rule on a constant series is all zeros, so Lee-Ready is all zeros too.
    np.testing.assert_array_equal(lee_ready(price, bid, ask), np.zeros(3))
    # Above the mid → buy regardless of tick.
    price2 = np.array([10.0, 10.4, 9.6])
    assert lee_ready(price2, bid, ask)[1] == 1.0
    assert lee_ready(price2, bid, ask)[2] == -1.0


def test_emo_classifies_at_quotes_else_tick() -> None:
    price = np.array([10.0, 10.5, 9.5, 10.1])
    bid = np.full(4, 9.5)
    ask = np.full(4, 10.5)
    signs = emo_rule(price, bid, ask)
    assert signs[1] == 1.0  # at ask → buy
    assert signs[2] == -1.0  # at bid → sell
    assert signs[3] == 1.0  # inside spread, up-tick from 9.5 → buy


def test_bvc_fraction_in_unit_interval_and_symmetric() -> None:
    rng = np.random.default_rng(1)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=500))
    frac = bvc(prices)
    assert frac.shape == prices.shape
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    # Up moves are classified as more-buy than down moves.
    up = np.diff(prices) > 0
    assert frac[1:][up].mean() > 0.5
    assert frac[1:][~up].mean() < 0.5


def test_bvc_constant_prices_returns_half() -> None:
    frac = bvc(np.full(20, 50.0))
    np.testing.assert_allclose(frac, 0.5)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        quote_rule(np.ones(3), np.ones(2), np.ones(3))
