"""Smoke test for the cointegration pairs strategy (guards the shared-util refactor)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.strategies import CointegrationPairsStrategy, PairsParams


def _cointegrated_prices(seed: int = 0, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0, 0.01, size=n))
    log_x = 4.6 + common + rng.normal(0.0, 0.01, size=n)
    log_y = 4.6 + 1.2 * common + rng.normal(0.0, 0.01, size=n)
    return np.exp(log_y), np.exp(log_x)


def test_run_produces_consistent_result() -> None:
    y, x = _cointegrated_prices()
    res = CointegrationPairsStrategy().run(y, x)
    assert res.pos_y.shape == y.shape
    assert res.pos_x.shape == y.shape
    assert res.z_score.shape == y.shape
    assert res.log_returns.shape[0] == y.shape[0] - 1
    assert np.isfinite(res.sharpe)
    assert 0.0 <= res.active_fraction <= 1.0
    assert "Cointegration Pairs" in res.summary()


def test_run_dataframe_columns() -> None:
    y, x = _cointegrated_prices()
    df = CointegrationPairsStrategy().run(y, x).to_dataframe()
    assert {"y", "x", "beta", "spread", "z_score", "pos_y", "pos_x"} <= set(df.columns)


def test_pairs_params_validation() -> None:
    with pytest.raises(ValueError):
        PairsParams(exit_z=2.0, entry_z=1.0)  # exit must be < entry
