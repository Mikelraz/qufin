"""Market Profile / TPO construction."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from qufin.volume_distribution import bracket_letters, tpo_profile
from tests.volume_distribution.conftest import make_ohlcv


def _intraday_bars(highs: np.ndarray, lows: np.ndarray):
    n = highs.shape[0]
    closes = 0.5 * (highs + lows)
    return make_ohlcv(closes, highs, lows, closes, np.ones(n), freq=timedelta(minutes=30))


def test_bracket_letters_overflow() -> None:
    labels = bracket_letters(54)
    assert labels[0] == "A"
    assert labels[25] == "Z"
    assert labels[26] == "a"
    assert labels[51] == "z"
    assert labels[52] == "AA"


def test_initial_balance_is_first_n_brackets() -> None:
    # 4 half-hour bars → 4 brackets; IB = first 2.
    highs = np.array([11.0, 12.0, 15.0, 9.0])
    lows = np.array([10.0, 11.0, 13.0, 8.0])
    bars = _intraday_bars(highs, lows)
    prof = tpo_profile(bars, n_bins=20, period="30m", n_initial=2)
    # IB spans the low/high of the first two brackets: low=min(10,11), high=max(11,12).
    assert prof.initial_balance == (10.0, 12.0)
    assert prof.range_extension_up  # bracket 3 trades up to 15 > IB high
    assert prof.range_extension_down  # bracket 4 trades down to 8 < IB low


def test_single_prints_on_one_touch_level() -> None:
    # A level only the spike bracket reaches should be a single print.
    highs = np.array([11.0, 11.0, 20.0, 11.0])
    lows = np.array([10.0, 10.0, 19.0, 10.0])
    bars = _intraday_bars(highs, lows)
    prof = tpo_profile(bars, n_bins=40, period="30m")
    sp_centres = prof.bin_centres[prof.single_prints]
    assert np.any(sp_centres > 15.0)


def test_poc_in_value_area() -> None:
    highs = np.array([11.0, 11.5, 12.0, 11.0])
    lows = np.array([10.0, 10.5, 11.0, 10.0])
    bars = _intraday_bars(highs, lows)
    prof = tpo_profile(bars, n_bins=20, period="30m")
    assert prof.val <= prof.poc <= prof.vah


def test_tpo_rejects_zero_bars() -> None:
    bars = _intraday_bars(np.array([11.0]), np.array([10.0])).slice_bars(0, 0)
    with pytest.raises(ValueError):
        tpo_profile(bars)
