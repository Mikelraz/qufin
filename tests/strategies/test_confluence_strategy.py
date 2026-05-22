"""Integration test: drive ConfluenceStrategy through the BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from qufin.strategies.confluence import (
    ConfluenceParams,
    ConfluenceStrategy,
    make_strategy,
)
from qufin.trading import BacktestEngine
from qufin.trading.engine import Clock, EngineConfig
from qufin.wyckoff._types import BAR_SCHEMA


def _synthetic_frame(
    *,
    symbol: str,
    n_days: int = 350,
    drift: float = 0.0005,
    vol: float = 0.012,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = drift + vol * rng.standard_normal(n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread = close * 0.002
    high = close + spread
    low = close - spread
    open_ = close + rng.normal(scale=close * 0.0005)
    volume = 1_000_000 + rng.integers(0, 100_000, size=n_days).astype(np.float64)
    start = datetime(2018, 1, 2, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume,
        }
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")),
        *(pl.col(c).cast(d) for c, d in BAR_SCHEMA.items() if c != "timestamp"),
    )


def test_strategy_runs_to_completion_on_synthetic_data() -> None:
    symbols = ["SPY", "QQQ", "IWM"]
    bars = {sym: _synthetic_frame(symbol=sym, n_days=320, seed=i) for i, sym in enumerate(symbols)}
    clock = Clock(bars=bars)
    params = ConfluenceParams(
        regime_warmup_bars=120,
        regime_refit_period=40,
        min_confluences=2,
        use_gex_overlay=False,
    )
    strategy = ConfluenceStrategy(params=params, symbols=symbols)
    engine = BacktestEngine(
        strategy=strategy, clock=clock, config=EngineConfig(starting_cash=100_000.0)
    )
    report = engine.run()
    assert report.equity_curve.height > 0
    assert "equity" in report.equity_curve.columns
    finite = np.isfinite(report.equity_curve["equity"].to_numpy())
    assert finite.all(), "equity curve must contain only finite values"


def test_make_strategy_factory() -> None:
    strat = make_strategy(symbols=["SPY", "QQQ"])
    assert isinstance(strat, ConfluenceStrategy)
    assert strat.symbols == ["SPY", "QQQ"]
    assert strat.params is not None
