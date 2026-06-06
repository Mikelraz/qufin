"""Time-series momentum strategy: run, fit grid search, parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.strategies import TimeSeriesMomentumStrategy, TSMOMParams


def _trending_prices(seed: int = 0, n: int = 1500, drift: float = 0.0008) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.006, size=n)))


def test_run_produces_result() -> None:
    res = TimeSeriesMomentumStrategy().run(_trending_prices())
    assert res.position.shape == res.prices.shape
    assert res.log_returns.shape[0] == res.prices.shape[0] - 1
    assert np.isfinite(res.sharpe)
    assert "Time-Series Momentum" in res.summary()


def test_run_dataframe_columns() -> None:
    df = TimeSeriesMomentumStrategy().run(_trending_prices()).to_dataframe()
    assert set(df.columns) == {"price", "signal", "position", "strat_ret"}
    assert df.height == 1500


def test_fit_selects_best_and_updates_params() -> None:
    strat = TimeSeriesMomentumStrategy()
    train = strat.fit(
        _trending_prices(),
        lookbacks=(63, 126, 252),
        vol_windows=(20, 60),
        target_vols=(0.15,),
    )
    assert train.grid  # non-empty
    best = max(train.grid, key=lambda g: g[3])
    assert train.sharpe == pytest.approx(best[3])
    assert strat.params.lookback == train.params.lookback


def test_fit_raises_when_series_too_short() -> None:
    short = _trending_prices(n=120)
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy().fit(short, lookbacks=(252,), vol_windows=(60,))


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        TSMOMParams(lookback=1)
    with pytest.raises(ValueError):
        TSMOMParams(target_vol=0.0)
    with pytest.raises(ValueError):
        TSMOMParams(vol_window=1)
