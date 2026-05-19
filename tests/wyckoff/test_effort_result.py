"""Effort-vs-result rolling absorption flags."""

from __future__ import annotations

import numpy as np

from qufin.wyckoff import effort_vs_result
from tests.wyckoff.conftest import make_ohlcv


def test_planted_absorption_bar_is_flagged() -> None:
    rng = np.random.default_rng(0)
    n = 200
    # Baseline: tight body, moderate volume.
    closes = 100.0 + np.cumsum(rng.normal(0, 0.2, n))
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + 0.4
    lows = np.minimum(opens, closes) - 0.4
    vols = rng.lognormal(mean=10.0, sigma=0.1, size=n)

    # Plant a wide-range, high-volume bar where close ≈ open (low result).
    k = 150
    highs[k] = closes[k] + 5.0
    lows[k] = closes[k] - 5.0
    opens[k] = closes[k] - 0.01  # body almost zero relative to true range
    vols[k] *= 50.0

    bars = make_ohlcv(opens, highs, lows, closes, vols)
    er = effort_vs_result(bars, window=50)
    assert bool(er.flag_absorption[k])


def test_normal_bars_do_not_trigger() -> None:
    rng = np.random.default_rng(1)
    n = 200
    closes = 100.0 + np.cumsum(rng.normal(0, 0.2, n))
    opens = closes + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + 0.2
    lows = np.minimum(opens, closes) - 0.2
    vols = rng.lognormal(mean=10.0, sigma=0.1, size=n)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    er = effort_vs_result(bars, window=50)
    # On uniform random data, fewer than 5% of bars should be flagged.
    assert er.flag_absorption.sum() < 0.05 * n
