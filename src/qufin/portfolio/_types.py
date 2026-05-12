"""Shared result containers for portfolio analysis and optimization.

Both dataclasses use ``slots=True`` for memory efficiency; many instances
are created during frontier tracing and back-testing loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class OptimizationResult:
    """Outcome of a single portfolio optimization run.

    All return and volatility figures are in the same units as the inputs
    passed to the optimizer (typically annualized).

    Attributes:
        weights: Portfolio weight vector of shape (n,), summing to 1.
        asset_names: Asset identifiers in the same order as ``weights``.
        expected_return: Weighted sum of asset expected returns (w @ mu).
        expected_volatility: Portfolio standard deviation sqrt(w' Sigma w).
        sharpe_ratio: (expected_return - risk_free_rate) / expected_volatility.
            Zero when volatility is effectively zero.
        success: True if the SLSQP solver converged to a feasible solution.
        message: Solver status string; useful for diagnosing convergence failures.
    """

    weights: NDArray[np.float64]
    asset_names: list[str]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str

    def as_dict(self) -> dict[str, float]:
        """Return ``{asset_name: weight}`` mapping for quick inspection."""
        return dict(zip(self.asset_names, self.weights.tolist(), strict=True))


@dataclass(slots=True)
class EfficientFrontier:
    """Collection of portfolios that trace the mean-variance efficient frontier.

    Points are ordered from minimum-variance (lowest return / lowest vol)
    to maximum-return (highest expected return in the asset universe).

    Attributes:
        returns: Annualized expected return for each frontier point, shape (n_points,).
        volatilities: Annualized portfolio volatility for each point, shape (n_points,).
        sharpe_ratios: Sharpe ratio for each point, shape (n_points,).
        weights: Weight matrix of shape (n_points, n_assets).
        asset_names: Asset identifiers matching the columns of ``weights``.
        risk_free_rate: Annual risk-free rate used when computing Sharpe ratios.
    """

    returns: NDArray[np.float64]  # (n_points,)
    volatilities: NDArray[np.float64]  # (n_points,)
    sharpe_ratios: NDArray[np.float64]  # (n_points,)
    weights: NDArray[np.float64]  # (n_points, n_assets)
    asset_names: list[str]
    risk_free_rate: float
