"""Volume delta, CVD, divergence, and delta-by-price footprint."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.volume_distribution import (
    bar_delta,
    cumulative_volume_delta,
    delta_divergence,
    delta_profile,
    signed_tick_volume,
)
from tests.volume_distribution.conftest import make_ticks


def test_tick_rule_signs_monotonic_uptick_all_positive() -> None:
    prices = np.arange(1.0, 11.0)
    sizes = np.full(10, 2.0)
    ticks = make_ticks(prices, sizes)
    signed = signed_tick_volume(ticks)
    assert np.all(signed > 0.0)
    assert signed.sum() == pytest.approx(sizes.sum())


def test_cvd_is_cumsum_of_signed_volume() -> None:
    prices = np.array([1.0, 2.0, 1.5, 1.5, 3.0])
    sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ticks = make_ticks(prices, sizes)
    signed = signed_tick_volume(ticks)
    cvd = cumulative_volume_delta(ticks)
    np.testing.assert_allclose(cvd, np.cumsum(signed))


def test_lee_ready_requires_bid_ask() -> None:
    ticks = make_ticks(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        signed_tick_volume(ticks, method="lee_ready")


def test_lee_ready_uses_mid() -> None:
    # mid = 10.0 for both rows; 10.4 is above (buy), 9.6 is below (sell).
    ticks = make_ticks(np.array([10.4, 9.6]), np.array([1.0, 1.0])).with_columns(
        bid=np.array([9.5, 9.5]), ask=np.array([10.5, 10.5])
    )
    signed = signed_tick_volume(ticks, method="lee_ready")
    assert signed[0] > 0.0
    assert signed[1] < 0.0


def test_bar_delta_sign_follows_close_position() -> None:
    # Close at the high → all buyers → positive delta == volume.
    delta = bar_delta(
        np.array([1.0]), np.array([2.0]), np.array([1.0]), np.array([2.0]), np.array([10.0])
    )
    assert delta[0] == pytest.approx(10.0)
    # Close at the low → all sellers → negative delta.
    delta = bar_delta(
        np.array([2.0]), np.array([2.0]), np.array([1.0]), np.array([1.0]), np.array([10.0])
    )
    assert delta[0] == pytest.approx(-10.0)


def test_delta_divergence_fires_on_price_up_cvd_down() -> None:
    n = 30
    price = np.linspace(100.0, 110.0, n)  # rising
    cvd = np.linspace(0.0, -50.0, n)  # falling
    div = delta_divergence(price, cvd, window=5)
    assert np.all(div[:5] == 0.0)
    assert div[-1] == -1.0


def test_delta_profile_buckets_by_price() -> None:
    prices = np.array([10.0, 11.0, 10.0, 9.0])
    sizes = np.array([1.0, 1.0, 1.0, 1.0])
    ticks = make_ticks(prices, sizes)
    prof = delta_profile(ticks, n_bins=10)
    assert prof.buy_volume.shape == prof.sell_volume.shape
    np.testing.assert_allclose(prof.delta, prof.buy_volume - prof.sell_volume)
    total = prof.buy_volume.sum() + prof.sell_volume.sum()
    assert total == pytest.approx(sizes.sum())
