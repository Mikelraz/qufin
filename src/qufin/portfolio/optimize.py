"""Portfolio optimization via mean-variance analysis.

All optimizers use SLSQP (Sequential Least SQuares Programming) from
``scipy.optimize.minimize``.  Inputs are expected to be in *annualized* units:
pass annualized expected returns (``mu``) and an annualized covariance matrix
(``sigma = daily_cov * periods_per_year``).

Available optimizers
--------------------
- ``min_variance``:    lowest achievable portfolio variance.
- ``max_sharpe``:      highest risk-adjusted return (non-convex, sensitivity
                       to ``expected_returns`` estimate).
- ``efficient_return``: lowest variance portfolio subject to a return target.
- ``efficient_frontier``: traces the full frontier from min-variance to
                       max-return by solving ``efficient_return`` at each point.
- ``risk_parity``:     equal risk contribution from every asset.

Long-only constraint
--------------------
When ``long_only=True`` (default) weights are bounded to [0, 1].  Set
``long_only=False`` to allow short positions with bounds [-1, 1].
"""

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


# Reused across all solvers — weights must sum to exactly 1.
_SUM_TO_ONE: dict[str, Any] = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

# ── public optimizers ─────────────────────────────────────────────────────────


def min_variance(
    expected_returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    asset_names: list[str],
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> OptimizationResult:
    """Find the portfolio with the lowest attainable variance.

    Solves::

        min  w' Sigma w
        s.t. sum(w) = 1
             w_i >= 0   (if long_only)

    The minimum-variance portfolio lies at the left tip of the efficient
    frontier and represents the most diversified allocation in a
    variance-minimisation sense, regardless of expected returns.

    Args:
        expected_returns: Annualized expected return vector of shape (n,).
            Used only to populate result metrics, not in the optimization.
        cov: Annualized covariance matrix of shape (n, n).
        asset_names: Asset identifiers in the same order as ``expected_returns``.
        risk_free_rate: Annual risk-free rate for Sharpe ratio computation.
        long_only: If True, restrict weights to [0, 1]; otherwise [-1, 1].

    Returns:
        OptimizationResult with the minimum-variance weights.
    """
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
    """Find the portfolio with the highest Sharpe ratio (tangency portfolio).

    Solves::

        max  (w' mu - r_f) / sqrt(w' Sigma w)
        s.t. sum(w) = 1,  w_i >= 0  (if long_only)

    This is a non-convex problem; SLSQP finds a local optimum.  The result is
    highly sensitive to the expected-return estimates: small errors in ``mu``
    can lead to extreme weight concentrations.  Consider using ``min_variance``
    or ``risk_parity`` when returns are difficult to estimate reliably.

    Args:
        expected_returns: Annualized expected return vector of shape (n,).
        cov: Annualized covariance matrix of shape (n, n).
        asset_names: Asset identifiers.
        risk_free_rate: Annual risk-free rate (subtracted from portfolio return
            before dividing by volatility).
        long_only: If True, restrict weights to [0, 1]; otherwise [-1, 1].

    Returns:
        OptimizationResult with the maximum-Sharpe weights.
    """
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
    """Find the minimum-variance portfolio that achieves a given return target.

    Solves::

        min  w' Sigma w
        s.t. sum(w) = 1
             w' mu = target_return
             w_i >= 0  (if long_only)

    This is the building block for ``efficient_frontier``: by sweeping
    ``target_return`` from the minimum-variance return to the maximum asset
    return, the full efficient frontier is traced.

    Args:
        expected_returns: Annualized expected return vector of shape (n,).
        cov: Annualized covariance matrix of shape (n, n).
        target_return: Annualized return target.  Must lie between the
            minimum-variance portfolio return and ``max(expected_returns)``.
        asset_names: Asset identifiers.
        risk_free_rate: Annual risk-free rate for Sharpe ratio computation.
        long_only: If True, restrict weights to [0, 1]; otherwise [-1, 1].

    Returns:
        OptimizationResult; check ``.success`` if the target is infeasible.
    """
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
    """Find the equal risk contribution (risk parity) portfolio.

    Each asset contributes an equal fraction (1/n) of total portfolio variance.
    The risk contribution of asset *i* is::

        RC_i = w_i * (Sigma w)_i / (w' Sigma w)

    The objective minimises the sum of squared deviations from ``1/n``::

        min  sum_i (RC_i - 1/n)^2

    Risk parity ignores expected returns entirely, making it more robust than
    mean-variance optimisation when return forecasts are unreliable.  It
    typically produces well-diversified portfolios that are less concentrated
    than minimum-variance solutions.

    A small lower bound of 1e-6 on weights is enforced for numerical stability
    of the risk-contribution formula (avoids division by near-zero variance).

    Args:
        cov: Annualized covariance matrix of shape (n, n).
        asset_names: Asset identifiers.
        expected_returns: Optional annualized return vector.  Used only to
            compute Sharpe ratio in the result; ignored during optimization.
        risk_free_rate: Annual risk-free rate for Sharpe ratio computation.

    Returns:
        OptimizationResult with equal risk contribution weights.
    """
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
    """Trace the efficient frontier from minimum-variance to maximum-return.

    Sweeps ``n_points`` evenly spaced return targets between the return of the
    minimum-variance portfolio and ``max(expected_returns)``, solving
    ``efficient_return`` at each target.  Points that fail to converge are
    silently dropped.

    The resulting curve is the set of all portfolios that maximise expected
    return for a given level of risk (or equivalently minimise risk for a given
    level of return).  Portfolios below the frontier are suboptimal.

    Args:
        expected_returns: Annualized expected return vector of shape (n,).
        cov: Annualized covariance matrix of shape (n, n).
        asset_names: Asset identifiers.
        n_points: Number of frontier points to compute.
        risk_free_rate: Annual risk-free rate for Sharpe ratio computation.
        long_only: If True, restrict weights to [0, 1]; otherwise [-1, 1].

    Returns:
        EfficientFrontier dataclass containing arrays of returns, volatilities,
        Sharpe ratios, and weights for all successfully converged points.
    """
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
