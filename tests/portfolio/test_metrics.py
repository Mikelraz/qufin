from __future__ import annotations

import math

import numpy as np
import pytest

from qufin.portfolio.metrics import (
    annualized_volatility,
    calmar_ratio,
    conditional_var,
    historical_var,
    max_drawdown,
    portfolio_metrics,
    sharpe_ratio,
    sortino_ratio,
)


@pytest.fixture
def flat_returns() -> np.ndarray:
    """Deterministic returns: +1 % every period for easy manual checks."""
    return np.full(252, 0.01)


@pytest.fixture
def noisy_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.015, 1000)


def test_max_drawdown_no_loss(flat_returns: np.ndarray) -> None:
    assert max_drawdown(flat_returns) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_single_drop() -> None:
    # wealth: 1 → 2 → 1  → drawdown = 0.5
    r = np.array([1.0, -0.5])
    assert max_drawdown(r) == pytest.approx(0.5, rel=1e-10)


def test_max_drawdown_positive(noisy_returns: np.ndarray) -> None:
    mdd = max_drawdown(noisy_returns)
    assert 0.0 <= mdd < 1.0


def test_annualized_vol_known() -> None:
    # constant returns: std = 0 → vol = 0
    assert annualized_volatility(np.full(100, 0.001)) == pytest.approx(0.0, abs=1e-12)


def test_annualized_vol_scales(noisy_returns: np.ndarray) -> None:
    daily_std = float(np.std(noisy_returns, ddof=1))
    assert annualized_volatility(noisy_returns, 252) == pytest.approx(
        daily_std * math.sqrt(252), rel=1e-10
    )


def test_sharpe_zero_excess() -> None:
    # returns equal the risk-free rate → Sharpe = 0
    rfr = 0.04
    r = np.full(252, rfr / 252)
    assert sharpe_ratio(r, rfr, 252) == pytest.approx(0.0, abs=1e-6)


def test_sharpe_positive_for_positive_mean(noisy_returns: np.ndarray) -> None:
    # shift so mean is clearly positive
    r = noisy_returns + 0.005
    assert sharpe_ratio(r, 0.0) > 0.0


def test_sortino_infinite_when_no_downside(flat_returns: np.ndarray) -> None:
    assert sortino_ratio(flat_returns, 0.0) == math.inf


def test_sortino_ge_sharpe_for_skewed_upside(noisy_returns: np.ndarray) -> None:
    r = noisy_returns + 0.002
    assert sortino_ratio(r, 0.0) >= sharpe_ratio(r, 0.0)


def test_calmar_infinite_when_no_drawdown(flat_returns: np.ndarray) -> None:
    assert calmar_ratio(flat_returns) == math.inf


def test_calmar_positive(noisy_returns: np.ndarray) -> None:
    r = noisy_returns + 0.002
    assert calmar_ratio(r) > 0.0


def test_historical_var_ordering(noisy_returns: np.ndarray) -> None:
    var_95 = historical_var(noisy_returns, 0.95)
    var_99 = historical_var(noisy_returns, 0.99)
    assert var_99 >= var_95


def test_conditional_var_ge_var(noisy_returns: np.ndarray) -> None:
    var = historical_var(noisy_returns, 0.95)
    cvar = conditional_var(noisy_returns, 0.95)
    assert cvar >= var


def test_portfolio_metrics_keys(noisy_returns: np.ndarray) -> None:
    mat = np.column_stack([noisy_returns, noisy_returns * 0.8])
    w = np.array([0.6, 0.4])
    m = portfolio_metrics(w, mat)
    expected_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "var_95",
        "cvar_95",
    }
    assert set(m.keys()) == expected_keys
