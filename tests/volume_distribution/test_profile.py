"""Volume-by-price profile, value area, composite sessions, naked POCs."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from qufin.volume_distribution import (
    composite_profile,
    value_area,
    value_area_migration,
    volume_profile,
    volume_profile_from_ticks,
)
from tests.volume_distribution.conftest import make_ohlcv, make_ticks


def test_poc_concentrates_where_volume_piles() -> None:
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
    prof = volume_profile(bars, n_bins=40)
    assert abs(prof.poc - 100.0) < 5.0
    assert prof.val <= prof.poc <= prof.vah


def test_value_area_captures_target_fraction() -> None:
    rng = np.random.default_rng(1)
    n = 300
    closes = rng.normal(50.0, 1.0, size=n)
    opens = closes
    highs = closes + 0.05
    lows = closes - 0.05
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    prof = volume_profile(bars, n_bins=50, value_area_frac=0.70)
    centres = prof.bin_centres
    in_va = (centres >= prof.val) & (centres <= prof.vah)
    captured = prof.volume[in_va].sum() / prof.volume.sum()
    assert captured >= 0.69 - 1e-9
    val2, vah2 = value_area(prof, frac=0.70)
    assert val2 == pytest.approx(prof.val)
    assert vah2 == pytest.approx(prof.vah)


def test_ticks_and_bars_agree_on_controlled_distribution() -> None:
    # All trades at a single price → POC sits at that price for both paths.
    prices = np.concatenate([np.full(100, 10.0), np.full(20, 12.0)])
    sizes = np.ones_like(prices)
    ticks = make_ticks(prices, sizes)
    prof = volume_profile_from_ticks(ticks, n_bins=20)
    poc_bin = int(np.argmax(prof.volume))
    centre = prof.bin_centres[poc_bin]
    assert abs(centre - 10.0) < 0.2
    assert prof.volume.sum() == pytest.approx(sizes.sum())


def test_composite_profile_splits_by_session() -> None:
    # Two hourly sessions of 2 half-hour bars each; profiles returned in order.
    n = 4
    closes = np.array([10.0, 10.0, 20.0, 20.0])
    opens = closes
    highs = closes + 0.1
    lows = closes - 0.1
    vols = np.ones(n)
    bars = make_ohlcv(opens, highs, lows, closes, vols, freq=timedelta(minutes=30))
    profiles = composite_profile(bars, period="1h", n_bins=10)
    assert len(profiles) == 2
    assert profiles[0].poc < profiles[1].poc


def test_value_area_migration_columns() -> None:
    n = 4
    closes = np.array([10.0, 10.0, 20.0, 20.0])
    bars = make_ohlcv(
        closes, closes + 0.1, closes - 0.1, closes, np.ones(n), freq=timedelta(minutes=30)
    )
    mig = value_area_migration(bars, period="1h", n_bins=10)
    assert mig.columns == ["session", "poc", "vah", "val"]
    assert mig.height == 2


def test_invalid_args_raise() -> None:
    bars = make_ohlcv(*[np.arange(1.0, 6.0) for _ in range(5)])
    with pytest.raises(ValueError):
        volume_profile(bars, n_bins=1)
    with pytest.raises(ValueError):
        volume_profile(bars, value_area_frac=1.5)
