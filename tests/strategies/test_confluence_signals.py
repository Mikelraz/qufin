"""Unit tests for ConfluenceSignalEngine and the regime classifier."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qufin.strategies.confluence import (
    ConfluenceParams,
    ConfluenceSignalEngine,
    ExitReason,
    RegimeClassifier,
)
from qufin.wyckoff import OHLCV


def _bars_from_close(
    close: np.ndarray,
    *,
    symbol: str = "TEST",
    start: datetime | None = None,
) -> OHLCV:
    start = start or datetime(2018, 1, 2, tzinfo=UTC)
    n = close.shape[0]
    rng = np.random.default_rng(7)
    spread = np.abs(rng.normal(scale=0.5, size=n)) + 0.1
    high = close + spread
    low = close - spread
    open_ = close + rng.normal(scale=0.1, size=n)
    volume = 1_000_000 + rng.integers(0, 200_000, size=n).astype(np.float64)
    ts = [start + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")))
    return OHLCV.from_records(df, symbol=symbol)


@pytest.fixture
def rising_bars() -> OHLCV:
    rng = np.random.default_rng(0)
    close = np.linspace(100.0, 200.0, 600) + rng.normal(scale=0.5, size=600)
    return _bars_from_close(close)


@pytest.fixture
def falling_bars() -> OHLCV:
    rng = np.random.default_rng(1)
    close = np.linspace(200.0, 100.0, 600) + rng.normal(scale=0.5, size=600)
    return _bars_from_close(close)


@pytest.fixture
def choppy_bars() -> OHLCV:
    rng = np.random.default_rng(2)
    base = 150.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, 600))
    close = base + rng.normal(scale=0.3, size=600)
    return _bars_from_close(close)


# ----- Signal engine ------------------------------------------------------


def test_signal_engine_outputs_aligned(rising_bars: OHLCV) -> None:
    engine = ConfluenceSignalEngine(ConfluenceParams())
    sig = engine.evaluate(rising_bars)
    n = rising_bars.n_bars
    assert sig.long_entry.shape == (n,)
    assert sig.confluence_count.shape == (n,)
    assert sig.exit_reason.shape == (n,)
    assert sig.long_entry.dtype == bool
    assert all(
        name in sig.flags
        for name in ("w1_wyckoff_event", "w2_effort_result", "h_hull", "m_momentum", "v_volume")
    )


def test_signal_engine_warmup_is_silent(rising_bars: OHLCV) -> None:
    engine = ConfluenceSignalEngine(ConfluenceParams())
    sig = engine.evaluate(rising_bars)
    warmup = max(ConfluenceParams().hull_slow_length * 2, ConfluenceParams().range_min_bars)
    assert not sig.long_entry[:warmup].any()


def test_signal_engine_uptrend_fires_long(rising_bars: OHLCV) -> None:
    engine = ConfluenceSignalEngine(ConfluenceParams(min_confluences=2))
    sig = engine.evaluate(rising_bars)
    assert sig.long_entry.any(), "a clean uptrend should produce at least one long entry"


def test_signal_engine_downtrend_exits(falling_bars: OHLCV) -> None:
    engine = ConfluenceSignalEngine(ConfluenceParams())
    sig = engine.evaluate(falling_bars)
    exit_values = {
        ExitReason.HULL_FLIP.value,
        ExitReason.CHANDELIER.value,
        ExitReason.SWING_STOP.value,
        ExitReason.WYCKOFF_BEARISH.value,
    }
    fired = {v for v in sig.exit_reason if v in exit_values}
    assert fired, "a clean downtrend must fire at least one exit reason"


def test_signal_engine_causal(rising_bars: OHLCV) -> None:
    """Truncating the input must not change earlier values of the signal."""
    engine = ConfluenceSignalEngine(ConfluenceParams())
    full = engine.evaluate(rising_bars)
    cut = 450
    sub = OHLCV.from_records(rising_bars.data.head(cut), symbol=rising_bars.symbol)
    partial = engine.evaluate(sub)
    assert np.array_equal(partial.long_entry, full.long_entry[:cut])
    assert np.array_equal(partial.exit_reason, full.exit_reason[:cut])


def test_signal_count_is_bounded(choppy_bars: OHLCV) -> None:
    engine = ConfluenceSignalEngine(ConfluenceParams())
    sig = engine.evaluate(choppy_bars)
    assert sig.confluence_count.min() >= 0
    assert sig.confluence_count.max() <= 5


# ----- Regime classifier --------------------------------------------------


def test_regime_warmup_returns_no_defense() -> None:
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(scale=0.5, size=100))
    bars = _bars_from_close(close)
    rc = RegimeClassifier(warmup_bars=250)
    res = rc.fit_predict(bars)
    assert res.cash_defense.sum() == 0
    assert res.p_bear.sum() == 0.0


def test_regime_persistence_rule() -> None:
    rc = RegimeClassifier(bear_threshold=0.5, persistence_bars=3)
    values = np.array([0.0, 0.6, 0.7, 0.8, 0.4, 0.9, 0.9, 0.9, 0.0])
    flag = rc._persistent_above(values, threshold=0.5, k=3)
    expected = np.array([False, False, False, True, False, False, False, True, False])
    assert np.array_equal(flag, expected)
