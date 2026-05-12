from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.optimize import (
    efficient_frontier,
    efficient_return,
    max_sharpe,
    min_variance,
    risk_parity,
)


@pytest.fixture
def inputs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    mu = np.array([0.08, 0.12, 0.06, 0.10])
    cov = np.array(
        [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.01, 0.02],
            [0.02, 0.01, 0.06, 0.01],
            [0.01, 0.02, 0.01, 0.08],
        ]
    )
    return mu, cov, ["A", "B", "C", "D"]


# ── min_variance ──────────────────────────────────────────────────────────────


def test_min_variance_weights_sum_to_one(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = min_variance(mu, cov, names)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_min_variance_long_only(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = min_variance(mu, cov, names)
    assert np.all(result.weights >= -1e-8)


def test_min_variance_success(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = min_variance(mu, cov, names)
    assert result.success


def test_min_variance_is_actually_minimum(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    mv = min_variance(mu, cov, names)
    # Equal-weight portfolio should have higher or equal volatility
    w_eq = np.full(4, 0.25)
    eq_vol = float(np.sqrt(w_eq @ cov @ w_eq))
    assert mv.expected_volatility <= eq_vol + 1e-8


# ── max_sharpe ────────────────────────────────────────────────────────────────


def test_max_sharpe_weights_sum_to_one(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = max_sharpe(mu, cov, names, risk_free_rate=0.04)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_dominates_min_variance_sharpe(
    inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    mu, cov, names = inputs
    ms = max_sharpe(mu, cov, names, risk_free_rate=0.04)
    mv = min_variance(mu, cov, names, risk_free_rate=0.04)
    assert ms.sharpe_ratio >= mv.sharpe_ratio - 1e-6


def test_max_sharpe_long_only(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = max_sharpe(mu, cov, names)
    assert np.all(result.weights >= -1e-8)


# ── efficient_return ──────────────────────────────────────────────────────────


def test_efficient_return_achieves_target(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    target = 0.09
    result = efficient_return(mu, cov, target, names)
    assert result.expected_return == pytest.approx(target, abs=1e-5)


def test_efficient_return_weights_sum_to_one(
    inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    mu, cov, names = inputs
    result = efficient_return(mu, cov, 0.09, names)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


# ── risk_parity ───────────────────────────────────────────────────────────────


def test_risk_parity_weights_sum_to_one(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = risk_parity(cov, names)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_parity_equal_risk_contributions(
    inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    mu, cov, names = inputs
    result = risk_parity(cov, names)
    w = result.weights
    port_var = float(w @ cov @ w)
    rc = w * (cov @ w) / port_var
    # Each asset's risk contribution should be ~1/n
    np.testing.assert_allclose(rc, np.full(4, 0.25), atol=1e-4)


def test_risk_parity_all_positive(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    result = risk_parity(cov, names)
    assert np.all(result.weights > 0)


# ── efficient_frontier ────────────────────────────────────────────────────────


def test_frontier_monotone_vol(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    ef = efficient_frontier(mu, cov, names, n_points=20)
    assert len(ef.returns) > 1
    # Higher return should not correspond to lower volatility
    assert np.all(np.diff(ef.volatilities) >= -1e-6)


def test_frontier_weights_sum_to_one(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    ef = efficient_frontier(mu, cov, names, n_points=20)
    sums = ef.weights.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-5)


def test_frontier_n_points(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    ef = efficient_frontier(mu, cov, names, n_points=30)
    # All points should succeed for this well-conditioned problem
    assert len(ef.returns) == 30


def test_frontier_asset_names(inputs: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    mu, cov, names = inputs
    ef = efficient_frontier(mu, cov, names, n_points=10)
    assert ef.asset_names == names
