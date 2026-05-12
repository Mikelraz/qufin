from __future__ import annotations

import math

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True, parallel=True)
def _lw_beta_sq(x: NDArray[np.float64], s_mat: NDArray[np.float64]) -> float:
    """
    LW (2004) sampling-error estimate: (1/T²) Σ_t ||x_t x_t' - S||²_F.

    Using the identity ||x_t x_t' - S||²_F = ||x_t||⁴ - 2 x_t'Sx_t + tr(S²).
    """
    t, n = x.shape

    tr_s2 = 0.0
    for i in range(n):
        for j in range(n):
            tr_s2 += s_mat[i, j] * s_mat[i, j]

    partial = np.zeros(t)
    for k in numba.prange(t):
        norm2 = 0.0
        for i in range(n):
            norm2 += x[k, i] * x[k, i]

        x_s_x = 0.0
        for i in range(n):
            for j in range(n):
                x_s_x += x[k, i] * s_mat[i, j] * x[k, j]

        partial[k] = norm2 * norm2 - 2.0 * x_s_x + tr_s2

    return partial.sum() / (t * t)


def sample_cov(returns: NDArray[np.float64], ddof: int = 1) -> NDArray[np.float64]:
    """Unbiased sample covariance matrix (T × n input)."""
    return np.cov(returns.T, ddof=ddof)


def ledoit_wolf_cov(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Analytical Ledoit-Wolf shrinkage toward a scaled identity target.

    Ledoit & Wolf (2004): optimal shrinkage minimizes expected squared
    Frobenius loss relative to the true covariance.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    s = x.T @ x / t  # biased sample covariance (consistent with LW derivation)

    mu = float(np.trace(s)) / n

    # ||S - μI||²_F = tr(S²) - n·μ²
    delta_sq = float(np.sum(s**2) - n * mu**2)
    if delta_sq < 1e-20:
        return s.copy()

    beta_sq = float(_lw_beta_sq(np.ascontiguousarray(x), np.ascontiguousarray(s)))
    alpha = min(1.0, beta_sq / delta_sq)

    return (1.0 - alpha) * s + alpha * mu * np.eye(n)


def ewm_cov(returns: NDArray[np.float64], halflife: float) -> NDArray[np.float64]:
    """Exponentially weighted covariance with the given halflife (in periods)."""
    t, n = returns.shape
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    # weights: w_i ∝ (1-α)^(T-1-i), newest observation has weight (1-α)^0 = 1
    raw = np.array([(1.0 - alpha) ** (t - 1 - i) for i in range(t)])
    w = raw / raw.sum()
    mu = (w[:, None] * returns).sum(axis=0)
    x = returns - mu
    return x.T @ (w[:, None] * x)


def annualize_cov(cov: NDArray[np.float64], periods_per_year: int) -> NDArray[np.float64]:
    """Scale a per-period covariance matrix to annualized units."""
    return cov * periods_per_year


def cov_to_corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)
