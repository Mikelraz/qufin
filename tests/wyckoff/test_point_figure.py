"""Point-and-Figure chart construction and cause-and-effect counts."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.wyckoff import horizontal_count, pnf_from_bars, vertical_count
from tests.wyckoff.conftest import make_ohlcv


def _monotone_bars(closes: np.ndarray):
    opens = closes
    highs = closes + 0.01
    lows = closes - 0.01
    vols = np.full(closes.shape[0], 1.0)
    return make_ohlcv(opens, highs, lows, closes, vols)


def test_monotone_uptrend_produces_single_x_column() -> None:
    closes = np.linspace(100.0, 130.0, 60)
    bars = _monotone_bars(closes)
    chart = pnf_from_bars(bars, box_size=1.0, reversal=3)
    assert chart.n_columns == 1
    col = chart.columns[0]
    assert col.direction == "X"
    assert col.boxes_low == pytest.approx(100.0)
    assert col.boxes_high == pytest.approx(130.0)
    assert col.n_boxes == 31


def test_up_then_down_yields_two_columns_with_reversal_offset() -> None:
    up = np.linspace(100.0, 120.0, 40)
    down = np.linspace(120.0, 100.0, 40)
    closes = np.concatenate([up, down])
    bars = _monotone_bars(closes)
    chart = pnf_from_bars(bars, box_size=1.0, reversal=3)
    assert chart.n_columns >= 2
    first, second = chart.columns[0], chart.columns[1]
    assert first.direction == "X"
    assert second.direction == "O"
    # New O column starts one box below prior X top.
    assert second.boxes_high == pytest.approx(first.boxes_high - 1.0)


def test_vertical_count_projects_from_column_origin() -> None:
    closes = np.linspace(100.0, 120.0, 40)
    bars = _monotone_bars(closes)
    chart = pnf_from_bars(bars, box_size=1.0, reversal=3)
    target = vertical_count(chart, column_idx=0)
    assert target.method == "vertical"
    assert target.direction == "up"
    # n_boxes = 21, reversal = 3, box = 1 ⇒ move = 63 from low = 100 ⇒ 163.
    assert target.projected_price == pytest.approx(100.0 + 21 * 1.0 * 3)


def test_horizontal_count_requires_a_base_column() -> None:
    closes = np.linspace(100.0, 120.0, 40)
    bars = _monotone_bars(closes)
    chart = pnf_from_bars(bars, box_size=1.0, reversal=3)
    with pytest.raises(ValueError):
        horizontal_count(chart, column_idx=0)


def test_horizontal_count_projects_from_breakout() -> None:
    # Oscillation must span at least ``reversal`` boxes (3) to trigger
    # reversal columns; we use ~10 boxes (95↔105) for several swings, then a
    # clean upside breakout.
    base_lo, base_hi = 95.0, 105.0
    oscillation = np.concatenate(
        [
            np.linspace(base_lo, base_hi, 20),
            np.linspace(base_hi, base_lo, 20),
            np.linspace(base_lo, base_hi, 20),
            np.linspace(base_hi, base_lo, 20),
        ]
    )
    breakout = np.linspace(base_lo, 120.0, 40)
    closes = np.concatenate([oscillation, breakout])
    bars = _monotone_bars(closes)
    chart = pnf_from_bars(bars, box_size=1.0, reversal=3)
    assert chart.n_columns > 1
    target = horizontal_count(chart, column_idx=chart.n_columns - 1)
    assert target.direction == "up"
    assert target.projected_price > target.breakout_price
