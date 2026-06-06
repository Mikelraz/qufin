"""Order-flow: signed volume, rolling imbalance, Cont-Kukanov-Stoikov OFI."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import order_flow_imbalance, signed_volume, trade_imbalance


def test_signed_volume_is_sign_times_size() -> None:
    signs = np.array([1.0, -1.0, 1.0])
    size = np.array([10.0, 5.0, 2.0])
    np.testing.assert_allclose(signed_volume(signs, size), np.array([10.0, -5.0, 2.0]))


def test_trade_imbalance_all_buys_is_plus_one() -> None:
    signs = np.ones(100)
    ti = trade_imbalance(signs, window=10)
    assert np.all(np.isnan(ti[:9]))
    np.testing.assert_allclose(ti[9:], 1.0)


def test_trade_imbalance_volume_weighted() -> None:
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    volume = np.array([3.0, 1.0, 3.0, 1.0])
    ti = trade_imbalance(signs, volume, window=4)
    # (3 - 1 + 3 - 1) / (3 + 1 + 3 + 1) = 4 / 8 = 0.5
    assert ti[-1] == pytest.approx(0.5)


def test_ofi_equal_quotes_uses_size_change() -> None:
    bid = np.array([10.0, 10.0])
    ask = np.array([11.0, 11.0])
    bid_size = np.array([5.0, 8.0])
    ask_size = np.array([5.0, 5.0])
    e = order_flow_imbalance(bid, ask, bid_size, ask_size)
    assert e[0] == 0.0
    assert e[1] == pytest.approx(3.0)  # Δbid_size − Δask_size = 3 − 0


def test_ofi_bid_uptick_adds_full_bid_size() -> None:
    bid = np.array([10.0, 10.5])
    ask = np.array([11.0, 11.0])
    bid_size = np.array([5.0, 8.0])
    ask_size = np.array([5.0, 5.0])
    e = order_flow_imbalance(bid, ask, bid_size, ask_size)
    assert e[1] == pytest.approx(8.0)


def test_ofi_rolling_window_sums_events() -> None:
    bid = np.array([10.0, 10.0, 10.0])
    ask = np.array([11.0, 11.0, 11.0])
    bid_size = np.array([5.0, 8.0, 10.0])
    ask_size = np.array([5.0, 5.0, 5.0])
    rolled = order_flow_imbalance(bid, ask, bid_size, ask_size, window=2)
    # events: [0, 3, 2]; rolling-2 sums: [nan, 3, 5]
    assert np.isnan(rolled[0])
    assert rolled[1] == pytest.approx(3.0)
    assert rolled[2] == pytest.approx(5.0)
