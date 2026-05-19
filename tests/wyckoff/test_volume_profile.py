"""Volume-by-price profile, POC, value area, anchored VWAP."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.wyckoff import anchored_vwap, value_area, volume_profile
from tests.wyckoff.conftest import make_ohlcv


def test_volume_concentrates_at_poc() -> None:
    # 200 bars: 80% of them sit tightly around 100, 20% wander widely.
    rng = np.random.default_rng(0)
    n = 200
    closes = np.concatenate(
        [
            rng.normal(100.0, 0.1, size=int(0.8 * n)),
            rng.normal(120.0, 5.0, size=int(0.2 * n)),
        ]
    )
    opens = closes
    highs = closes + 0.05
    lows = closes - 0.05
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    profile = volume_profile(bars, n_bins=40)
    assert abs(profile.poc - 100.0) < 5.0


def test_value_area_contains_target_fraction() -> None:
    rng = np.random.default_rng(1)
    n = 300
    closes = rng.normal(50.0, 1.0, size=n)
    opens = closes
    highs = closes + 0.05
    lows = closes - 0.05
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    profile = volume_profile(bars, n_bins=50, value_area_frac=0.70)
    edges = profile.price_bins
    hist = profile.volume
    centres = 0.5 * (edges[:-1] + edges[1:])
    in_va = (centres >= profile.val) & (centres <= profile.vah)
    captured = hist[in_va].sum() / hist.sum()
    assert captured >= 0.69 - 1e-9
    # And re-computing with value_area() gives equivalent bounds.
    val2, vah2 = value_area(profile, frac=0.70)
    assert val2 == pytest.approx(profile.val)
    assert vah2 == pytest.approx(profile.vah)


def test_anchored_vwap_starts_at_typical_price() -> None:
    n = 30
    closes = np.linspace(100.0, 110.0, n)
    opens = closes
    highs = closes + 0.5
    lows = closes - 0.5
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    vwap = anchored_vwap(bars, anchor_idx=5)
    assert np.isnan(vwap[:5]).all()
    expected_first = (highs[5] + lows[5] + closes[5]) / 3.0
    assert vwap[5] == pytest.approx(expected_first)
    # VWAP is bounded by the price range it accumulates over.
    assert vwap[-1] >= closes[5] - 1.0
    assert vwap[-1] <= closes[-1] + 1.0
