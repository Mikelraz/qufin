"""Alternative-bar construction tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qufin.data._types import TICK_SCHEMA
from qufin.data.bars import (
    dollar_bars,
    dollar_imbalance_bars,
    tick_imbalance_bars,
    tick_runs_bars,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
)

from .conftest import make_ohlcv


def _make_ticks(
    prices: list[float],
    sizes: list[float],
    *,
    start: datetime | None = None,
    step: timedelta = timedelta(seconds=1),
) -> pl.DataFrame:
    if start is None:
        start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [start + step * i for i in range(len(prices))],
            "price": prices,
            "size": sizes,
        },
        schema=TICK_SCHEMA,
    )


def test_time_bars_aggregate_open_high_low_close_volume() -> None:
    bars = make_ohlcv(60, step=timedelta(minutes=1), symbol="AAPL", seed=0)
    out = time_bars(bars, every="5m")
    assert out.n_bars == 12
    # First 5m bar should match agg of first 5 minute bars
    first_open = bars.data["open"][0]
    first_close = bars.data["close"][4]
    first_volume = bars.data["volume"][:5].sum()
    assert out.data["open"][0] == first_open
    assert out.data["close"][0] == first_close
    np.testing.assert_allclose(out.data["volume"][0], first_volume)


def test_volume_bars_close_when_threshold_reached() -> None:
    # 6 ticks of 40 each = 240 total; threshold 100 → 2 closed bars (partial dropped).
    ticks = _make_ticks(
        prices=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        sizes=[40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
    )
    out = volume_bars(ticks, threshold=100.0, symbol="X")
    assert out.n_bars == 2


def test_volume_bars_emit_ohlc_correctly() -> None:
    ticks = _make_ticks(
        prices=[100.0, 102.0, 99.0, 101.0, 105.0, 103.0],
        sizes=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    )
    out = volume_bars(ticks, threshold=100.0)
    first = out.data.row(0, named=True)
    assert first["open"] == 100.0
    assert first["high"] == 102.0
    assert first["low"] == 100.0
    assert first["close"] == 102.0
    assert first["volume"] == 100.0


def test_dollar_bars_use_price_times_size() -> None:
    ticks = _make_ticks(
        prices=[100.0, 200.0, 50.0, 100.0],
        sizes=[10.0, 10.0, 10.0, 10.0],
    )
    out = dollar_bars(ticks, threshold=2000.0)
    # cum dollars: 1000, 3000 (bar end), then 500, 1500 (no bar — dropped).
    assert out.n_bars == 1
    assert out.data["close"][0] == 200.0


def test_volume_bars_reject_non_positive_threshold() -> None:
    ticks = _make_ticks([100.0, 101.0], [10.0, 10.0])
    with pytest.raises(ValueError):
        volume_bars(ticks, threshold=0.0)


def test_tick_imbalance_bars_emit_on_signed_imbalance() -> None:
    # Sharp uptrend: every tick has positive sign after the first → imbalance grows
    prices = [100.0 + i * 0.1 for i in range(20)]
    sizes = [10.0] * 20
    ticks = _make_ticks(prices, sizes)
    out = tick_imbalance_bars(ticks, initial_threshold=5.0, ema_alpha=0.5)
    assert out.n_bars >= 2
    # All bars should be entirely up-bars: close > open
    closes = out.data["close"].to_numpy()
    opens = out.data["open"].to_numpy()
    assert (closes >= opens).all()


def test_volume_imbalance_uses_signed_size() -> None:
    prices = [100.0] + [99.0] * 9  # tick-rule: all down after first
    sizes = [10.0] * 10
    ticks = _make_ticks(prices, sizes)
    out = volume_imbalance_bars(ticks, initial_threshold=30.0, ema_alpha=0.3)
    assert out.n_bars >= 1


def test_dollar_imbalance_uses_signed_dollars() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    sizes = [10.0] * 6
    ticks = _make_ticks(prices, sizes)
    out = dollar_imbalance_bars(ticks, initial_threshold=2000.0, ema_alpha=0.2)
    assert out.n_bars >= 1


def test_tick_runs_bars_close_on_long_uptick_run() -> None:
    prices = [100.0] + [100.0 + i * 0.1 for i in range(1, 11)]
    sizes = [1.0] * len(prices)
    ticks = _make_ticks(prices, sizes)
    out = tick_runs_bars(ticks, initial_threshold=4.0, ema_alpha=0.5)
    assert out.n_bars >= 1


def test_information_bars_reject_invalid_alpha() -> None:
    ticks = _make_ticks([100.0, 101.0], [10.0, 10.0])
    with pytest.raises(ValueError):
        tick_imbalance_bars(ticks, initial_threshold=1.0, ema_alpha=0.0)
    with pytest.raises(ValueError):
        tick_runs_bars(ticks, initial_threshold=1.0, ema_alpha=1.5)


def test_bars_validate_tick_schema() -> None:
    bad = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "price": [1.0]})
    with pytest.raises(ValueError):
        volume_bars(bad, threshold=1.0)


def test_bars_validate_sort_order() -> None:
    ticks = _make_ticks([100.0, 101.0, 99.0], [10.0, 10.0, 10.0])
    shuffled = ticks.sort("timestamp", descending=True)
    with pytest.raises(ValueError):
        volume_bars(shuffled, threshold=1.0)


def test_time_bars_preserve_symbol() -> None:
    bars = make_ohlcv(60, step=timedelta(minutes=1), symbol="MSFT", seed=1)
    out = time_bars(bars, every="15m")
    assert out.symbol == "MSFT"


def test_volume_bars_no_bars_when_threshold_too_high() -> None:
    ticks = _make_ticks([100.0, 101.0, 102.0], [10.0, 10.0, 10.0])
    out = volume_bars(ticks, threshold=10_000.0)
    assert out.n_bars == 0


def test_tick_imbalance_bars_min_bar_size_floor() -> None:
    prices = [100.0 + i * 0.1 for i in range(50)]
    sizes = [10.0] * 50
    ticks = _make_ticks(prices, sizes)
    out = tick_imbalance_bars(
        ticks, initial_threshold=1.0, ema_alpha=0.5, min_bar_size=10
    )
    if out.n_bars >= 2:
        starts = [0] + [
            int(out.data["volume"][:k].sum() / 10) for k in range(1, out.n_bars)
        ]
        diffs = np.diff(starts)
        assert (diffs >= 10).all() or out.n_bars <= 1
