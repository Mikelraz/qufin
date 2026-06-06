"""The new volume profile must match what wyckoff re-exports, bit-for-bit."""

from __future__ import annotations

import numpy as np

from qufin import wyckoff
from qufin.volume_distribution import VolumeProfile, value_area, volume_profile
from tests.volume_distribution.conftest import synthetic_ohlcv


def test_wyckoff_reexports_same_objects() -> None:
    assert wyckoff.volume_profile is volume_profile
    assert wyckoff.value_area is value_area
    assert wyckoff.VolumeProfile is VolumeProfile


def test_profile_identical_to_wyckoff_path() -> None:
    bars = synthetic_ohlcv(200, seed=7)
    a = volume_profile(bars, n_bins=50)
    b = wyckoff.volume_profile(bars, n_bins=50)
    np.testing.assert_array_equal(a.price_bins, b.price_bins)
    np.testing.assert_array_equal(a.volume, b.volume)
    assert a.poc == b.poc
    assert a.vah == b.vah
    assert a.val == b.val
    np.testing.assert_array_equal(a.hvn_idx, b.hvn_idx)
    np.testing.assert_array_equal(a.lvn_idx, b.lvn_idx)


def test_anchored_vwap_matches_wyckoff() -> None:
    bars = synthetic_ohlcv(60, seed=2)
    a = wyckoff.anchored_vwap(bars, anchor_idx=10)
    from qufin.volume_distribution import anchored_vwap

    b = anchored_vwap(bars, anchor_idx=10)
    np.testing.assert_array_equal(a, b)
