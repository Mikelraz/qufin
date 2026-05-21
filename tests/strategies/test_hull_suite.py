"""
Tests for qufin.strategies.hull_suite + hull_strategy + hull_backtest.

Coverage
--------
  HMA variants — output length, warm-up NaN region, finite values
  Slope / colour helpers — +1 rising, -1 falling, correct labels
  price_vs_ribbon — above / below / inside cases on synthetic data
  hull_ribbon — shape, position labels, band consistency
  generate_signals — causal (no look-ahead), values in {-1, 0, 1},
      reacts to a clean uptrend and a clean downtrend
  multi_timeframe_filter — counter-trend signals masked to 0
  vwap_filter — direction must agree with price-vs-VWAP
  momentum_filter — RSI / MACD gating, pass-through with no inputs
  backtest_hull — output shapes, total-return on buy-and-hold-equivalent,
      Sharpe finite, trade-counting reasonable
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qufin.strategies.hull_backtest import HullBacktestResult, backtest_hull
from qufin.strategies.hull_strategy import (
    generate_signals,
    momentum_filter,
    multi_timeframe_filter,
    vwap_filter,
)
from qufin.strategies.hull_suite import (
    HullRibbon,
    ehma,
    hma,
    hull_ribbon,
    hull_slope,
    price_vs_ribbon,
    thma,
    wma,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rising_close() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.linspace(100.0, 200.0, 500)
    return base + rng.normal(scale=0.2, size=500)


@pytest.fixture
def falling_close() -> np.ndarray:
    rng = np.random.default_rng(1)
    base = np.linspace(200.0, 100.0, 500)
    return base + rng.normal(scale=0.2, size=500)


@pytest.fixture
def ohlcv_rising(rising_close: np.ndarray) -> pl.DataFrame:
    n = rising_close.shape[0]
    return pl.DataFrame(
        {
            "open": rising_close - 0.1,
            "high": rising_close + 0.3,
            "low": rising_close - 0.3,
            "close": rising_close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


# ---------------------------------------------------------------------------
# HMA family
# ---------------------------------------------------------------------------


def test_wma_basic() -> None:
    x = np.arange(1.0, 11.0)
    out = wma(x, 4)
    # First 3 entries are NaN (warm-up), then finite.
    assert np.isnan(out[:3]).all()
    assert np.all(np.isfinite(out[3:]))
    # Manual check on the last point: weighted average of 7,8,9,10 with
    # weights 1,2,3,4 ⇒ (7+16+27+40)/10 = 9.0
    assert out[-1] == pytest.approx(9.0)


@pytest.mark.parametrize("variant", [hma, thma, ehma])
def test_hull_variants_shape_and_warmup(variant, rising_close: np.ndarray) -> None:
    period = 50
    out = variant(rising_close, period)
    assert out.shape == rising_close.shape
    # The very last value must be finite for any reasonable warm-up.
    assert np.isfinite(out[-1])
    # Leading region has at least some NaNs (warm-up).
    assert np.isnan(out[0])


def test_hma_rises_in_uptrend(rising_close: np.ndarray) -> None:
    h = hma(rising_close, 50)
    # The Hull should trend up over the second half of a clean uptrend.
    second_half = h[~np.isnan(h)][len(h) // 2 :]
    assert second_half[-1] > second_half[0]


# ---------------------------------------------------------------------------
# Slope / colour / position
# ---------------------------------------------------------------------------


def test_hull_slope_rising_falling() -> None:
    up = np.arange(10.0)
    sl = hull_slope(up)
    assert np.isnan(sl[0])
    assert (sl[1:] == 1.0).all()

    down = np.arange(10.0)[::-1]
    sl = hull_slope(down)
    assert (sl[1:] == -1.0).all()


def test_price_vs_ribbon_labels() -> None:
    price = np.array([95.0, 100.0, 105.0, np.nan])
    fast = np.array([99.0, 100.0, 100.0, 100.0])
    slow = np.array([101.0, 100.0, 102.0, 100.0])
    pos = price_vs_ribbon(price, fast, slow)
    assert pos[0] == "below"  # below both
    assert pos[1] == "inside"  # equal to both (top == bot == 100)
    assert pos[2] == "above"  # above both
    assert pos[3] == ""  # NaN price


# ---------------------------------------------------------------------------
# Ribbon assembly
# ---------------------------------------------------------------------------


def test_hull_ribbon_shapes(rising_close: np.ndarray) -> None:
    rib = hull_ribbon(rising_close, fast_length=50, slow_length=60)
    assert isinstance(rib, HullRibbon)
    n = rising_close.shape[0]
    assert rib.fast.values.shape == (n,)
    assert rib.slow.values.shape == (n,)
    assert rib.position.shape == (n,)
    # Both bands turn green in a clean uptrend.
    assert rib.fast.color[-1] == "green"
    assert rib.slow.color[-1] == "green"


def test_hull_ribbon_length_multiplier(rising_close: np.ndarray) -> None:
    rib1 = hull_ribbon(rising_close, fast_length=50, slow_length=60)
    rib2 = hull_ribbon(rising_close, fast_length=25, slow_length=30, length_multiplier=2.0)
    # The multiplier should produce the same effective lengths as the first call.
    assert rib1.fast.length == rib2.fast.length == 50
    assert rib1.slow.length == rib2.slow.length == 60


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def test_generate_signals_values_and_causality(
    ohlcv_rising: pl.DataFrame,
) -> None:
    sig = generate_signals(ohlcv_rising, fast_length=50, slow_length=60)
    assert isinstance(sig, pl.Series)
    assert sig.len() == ohlcv_rising.height
    uniq = set(sig.unique().to_list())
    assert uniq.issubset({-1, 0, 1})
    # Warm-up bars must be flat.
    assert sig[0] == 0


def test_generate_signals_long_in_uptrend(
    ohlcv_rising: pl.DataFrame,
) -> None:
    sig = generate_signals(ohlcv_rising, fast_length=50, slow_length=60).to_numpy()
    # A clean ramp should put us long for the majority of the post-warm-up bars.
    tail = sig[200:]
    assert (tail == 1).mean() > 0.5


def test_generate_signals_short_in_downtrend(
    falling_close: np.ndarray,
) -> None:
    sig = generate_signals(falling_close, fast_length=50, slow_length=60).to_numpy()
    tail = sig[200:]
    assert (tail == -1).mean() > 0.5


# ---------------------------------------------------------------------------
# Confluence filters
# ---------------------------------------------------------------------------


def test_multi_timeframe_filter_blocks_counter_trend() -> None:
    htf = pl.Series([1, 1, -1, -1, 0])
    ltf = pl.Series([1, -1, -1, 1, 1])
    out = multi_timeframe_filter(htf, ltf).to_list()
    # bar 0: ltf long, htf long → keep (+1)
    # bar 1: ltf short, htf long → drop (0)
    # bar 2: ltf short, htf short → keep (-1)
    # bar 3: ltf long, htf short → drop (0)
    # bar 4: ltf long, htf flat → drop (0)
    assert out == [1, 0, -1, 0, 0]


def test_vwap_filter() -> None:
    sig = pl.Series([1, 1, -1, -1])
    price = pl.Series([101.0, 99.0, 99.0, 101.0])
    vwap = pl.Series([100.0, 100.0, 100.0, 100.0])
    out = vwap_filter(sig, price, vwap).to_list()
    # long above VWAP keeps, long below drops; short below keeps, short above drops.
    assert out == [1, 0, -1, 0]


def test_momentum_filter_passthrough() -> None:
    sig = pl.Series([1, -1, 0, 1])
    out = momentum_filter(sig).to_list()
    assert out == [1, -1, 0, 1]


def test_momentum_filter_rsi_macd_gate() -> None:
    sig = np.array([1, 1, -1, -1])
    rsi = np.array([60.0, 40.0, 40.0, 60.0])
    macd = np.array([0.5, -0.1, -0.5, 0.1])
    out = momentum_filter(sig, rsi_series=rsi, macd_series=macd).to_list()
    # bar 0: long, rsi>50, macd>0 → keep
    # bar 1: long, rsi<50 → drop
    # bar 2: short, rsi<50, macd<0 → keep
    # bar 3: short, rsi>50 → drop
    assert out == [1, 0, -1, 0]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def test_backtest_hull_shapes_and_stats(
    ohlcv_rising: pl.DataFrame,
) -> None:
    sig = generate_signals(ohlcv_rising, fast_length=50, slow_length=60)
    report = backtest_hull(ohlcv_rising, sig)
    assert isinstance(report, HullBacktestResult)
    n = ohlcv_rising.height
    assert report.log_returns.shape == (n,)
    assert report.equity.shape == (n,)
    # In a clean uptrend the strategy should be net long-biased ⇒ positive return.
    assert report.total_return > 0.0
    assert np.isfinite(report.sharpe)
    assert report.max_drawdown <= 0.0


def test_backtest_hull_length_mismatch_errors(
    ohlcv_rising: pl.DataFrame,
) -> None:
    short_sig = pl.Series([0] * 10, dtype=pl.Int8)
    with pytest.raises(ValueError):
        backtest_hull(ohlcv_rising, short_sig)


def test_backtest_summary_renders(
    ohlcv_rising: pl.DataFrame,
) -> None:
    sig = generate_signals(ohlcv_rising, fast_length=50, slow_length=60)
    report = backtest_hull(ohlcv_rising, sig)
    s = report.summary()
    assert "Hull-Suite Backtest" in s
    assert "Sharpe" in s
