from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class OptimizationResult:
    weights: NDArray[np.float64]
    asset_names: list[str]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.asset_names, self.weights.tolist(), strict=True))


@dataclass(slots=True)
class EfficientFrontier:
    returns: NDArray[np.float64]        # (n_points,)
    volatilities: NDArray[np.float64]   # (n_points,)
    sharpe_ratios: NDArray[np.float64]  # (n_points,)
    weights: NDArray[np.float64]        # (n_points, n_assets)
    asset_names: list[str]
    risk_free_rate: float
