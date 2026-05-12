from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ._types import EfficientFrontier, OptimizationResult

# ── internal helpers ──────────────────────────────────────────────────────────


def _port_return(w: NDArray[np.float64], mu: NDArray[np.float64]) -> float:
    return float(w @ mu)


def _port_variance(w: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    return float(w @ cov @ w)


def _port_vol(w: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    return math.sqrt(max(0.0, _port_variance(w, cov)))


def _make_result(
    opt_result: Any,
    weights: NDArray[np.float64],
    asset_names: list[str],
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    risk_free_rate: float,
) -> OptimizationResult:
    er = _port_return(weights, mu)
    vol = _port_vol(weights, cov)
    sr = (er - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return OptimizationResult(
        weights=weights,
        asset_names=asset_names,
        expected_return=er,
        expected_volatility=vol,
        sharpe_ratio=sr,
        success=bool(opt_result.success),
        message=str(opt_result.message),
    )


def _weight_bounds(n: int, long_only: bool) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n


_SUM_TO_ONE: dict[str, Any] = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

# ── public optimizers ─────────────────────────────────────────────────────────


def min_variance(
    expected_returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    asset_names: list[str],
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> OptimizationResult:
    """Minimum global variance portfolio."""
    n = len(expected_returns)
    w0 = np.full(n, 1.0 / n)
    res = minimize(
        fun=lambda w: _port_variance(w, cov),
        x0=w0,
        method="SLSQP",
        bounds=_weight_bounds(n, long_only),
        constraints=[_SUM_TO_ONE],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return _make_result(res, res.x, asset_names, expected_returns, cov, risk_free_rate)


def max_sharpe(
    expected_returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    asset_names: list[str],
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> OptimizationResult:
    """Maximum Sharpe ratio portfolio."""
    n = len(expected_returns)
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w: NDArray[np.float64]) -> float:
        er = _port_return(w, expected_returns)
        vol = _port_vol(w, cov)
        if vol < 1e-12:
            return 0.0
        return -(er - risk_free_rate) / vol

    res = minimize(
        fun=neg_sharpe,
        x0=w0,
        method="SLSQP",
        bounds=_weight_bounds(n, long_only),
        constraints=[_SUM_TO_ONE],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return _make_result(res, res.x, asset_names, expected_returns, cov, risk_free_rate)


def efficient_return(
    expected_returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    target_return: float,
    asset_names: list[str],
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> OptimizationResult:
    """Minimum variance portfolio achieving at least target_return."""
    n = len(expected_returns)
    w0 = np.full(n, 1.0 / n)
    constraints: list[dict[str, Any]] = [
        _SUM_TO_ONE,
        {"type": "eq", "fun": lambda w, r=target_return: _port_return(w, expected_returns) - r},
    ]
    res = minimize(
        fun=lambda w: _port_variance(w, cov),
        x0=w0,
        method="SLSQP",
        bounds=_weight_bounds(n, long_only),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return _make_result(res, res.x, asset_names, expected_returns, cov, risk_free_rate)


def risk_parity(
    cov: NDArray[np.float64],
    asset_names: list[str],
    expected_returns: NDArray[np.float64] | None = None,
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    """Equal risk contribution (risk parity) portfolio."""
    n = cov.shape[0]
    mu = expected_returns if expected_returns is not None else np.zeros(n)
    w0 = np.full(n, 1.0 / n)
    target_rc = 1.0 / n

    def objective(w: NDArray[np.float64]) -> float:
        port_var = _port_variance(w, cov)
        if port_var < 1e-20:
            return 0.0
        rc = w * (cov @ w) / port_var
        return float(np.sum((rc - target_rc) ** 2))

    res = minimize(
        fun=objective,
        x0=w0,
        method="SLSQP",
        # small positive lower bound for numerical stability of rc formula
        bounds=[(1e-6, 1.0)] * n,
        constraints=[_SUM_TO_ONE],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return _make_result(res, res.x, asset_names, mu, cov, risk_free_rate)


def efficient_frontier(
    expected_returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    asset_names: list[str],
    n_points: int = 50,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> EfficientFrontier:
    """Trace the efficient frontier from minimum-variance to maximum-return."""
    mv = min_variance(expected_returns, cov, asset_names, risk_free_rate, long_only)
    max_ret = float(np.max(expected_returns))
    target_rets = np.linspace(mv.expected_return, max_ret, n_points)

    pts_ret: list[float] = []
    pts_vol: list[float] = []
    pts_sr: list[float] = []
    pts_w: list[NDArray[np.float64]] = []

    for r in target_rets:
        result = efficient_return(expected_returns, cov, r, asset_names, risk_free_rate, long_only)
        if result.success:
            pts_ret.append(result.expected_return)
            pts_vol.append(result.expected_volatility)
            pts_sr.append(result.sharpe_ratio)
            pts_w.append(result.weights)

    weights_arr = np.array(pts_w) if pts_w else np.empty((0, len(asset_names)))
    return EfficientFrontier(
        returns=np.array(pts_ret),
        volatilities=np.array(pts_vol),
        sharpe_ratios=np.array(pts_sr),
        weights=weights_arr,
        asset_names=asset_names,
        risk_free_rate=risk_free_rate,
    )
