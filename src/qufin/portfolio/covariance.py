"""Covariance matrix estimation methods for portfolio construction.

Three estimators are provided, ordered by increasing statistical sophistication:

1. ``sample_cov`` — classical unbiased estimator; noisy for n/T > 0.1.
2. ``ledoit_wolf_cov`` — analytical shrinkage toward a scaled identity target;
   optimal under a squared Frobenius loss criterion (Ledoit & Wolf 2004).
3. ``ewm_cov`` — exponentially weighted estimator; adapts to recent volatility
   regimes, useful in non-stationary markets.

All estimators accept a (T × n) returns matrix and return an (n × n)
*per-period* covariance matrix.  Multiply by ``periods_per_year`` via
``annualize_cov`` before passing to the optimizers.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True, parallel=True)
def _lw_beta_sq(x: NDArray[np.float64], s_mat: NDArray[np.float64]) -> float:
    """Estimate the LW (2004) sampling-error term beta^2.

    Computes ``(1/T^2) * sum_t || x_t x_t' - S ||_F^2`` efficiently by
    expanding the Frobenius norm::

        || x_t x_t' - S ||_F^2 = ||x_t||^4 - 2 * x_t' S x_t + tr(S^2)

    This avoids forming T outer-product matrices explicitly.  The outer loop
    over t is parallelised with ``numba.prange``.

    Args:
        x: Demeaned returns matrix of shape (T, n).
        s_mat: Biased sample covariance (1/T) of shape (n, n).

    Returns:
        Scalar estimate of beta^2.
    """
    t, n = x.shape

    # tr(S^2) = sum_ij S_ij^2  (S is symmetric)
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
    """Unbiased sample covariance matrix.

    Equivalent to ``numpy.cov`` with ``rowvar=False``.  For n assets and T
    observations the estimator is consistent but noisy when T is small relative
    to n; consider ``ledoit_wolf_cov`` in that regime.

    Args:
        returns: Returns matrix of shape (T, n).
        ddof: Degrees-of-freedom correction (1 = unbiased, 0 = MLE).

    Returns:
        Symmetric positive-definite (n × n) covariance matrix.
    """
    return np.cov(returns.T, ddof=ddof)


def ledoit_wolf_cov(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytical Ledoit-Wolf shrinkage toward a scaled identity target.

    The estimator is a convex combination of the biased sample covariance *S*
    and a scaled identity matrix ``mu * I``::

        Sigma_hat = (1 - alpha) * S  +  alpha * mu * I

    where ``mu = tr(S) / n`` is the average eigenvalue (average variance) and
    the optimal shrinkage intensity ``alpha`` minimises the expected squared
    Frobenius loss to the true covariance.

    The analytic formula for ``alpha`` is::

        alpha* = min(1, beta^2 / delta^2)
        delta^2 = || S - mu*I ||_F^2       # distance of S from target
        beta^2  = (1/T^2) sum_t || x_t x_t' - S ||_F^2  # sampling noise

    A larger ``alpha`` means more shrinkage toward the identity (diagonal,
    equal-variance structure).  Shrinkage reduces estimation error at the cost
    of introducing bias, which is optimal when T/n is small.

    Reference: Ledoit, O. and Wolf, M. (2004). "A well-conditioned estimator
    for large-dimensional covariance matrices." *Journal of Multivariate
    Analysis*, 88(2), 365-411.

    Args:
        returns: Returns matrix of shape (T, n).  Demeaned internally.

    Returns:
        Regularised (n × n) covariance matrix.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    # Use biased (1/T) sample covariance — required for the LW beta formula.
    s = x.T @ x / t

    mu = float(np.trace(s)) / n

    # ||S - mu*I||^2_F = tr(S^2) - n*mu^2
    delta_sq = float(np.sum(s**2) - n * mu**2)
    if delta_sq < 1e-20:
        # S is already proportional to the identity; no shrinkage needed.
        return s.copy()

    beta_sq = float(_lw_beta_sq(np.ascontiguousarray(x), np.ascontiguousarray(s)))
    alpha = min(1.0, beta_sq / delta_sq)

    return (1.0 - alpha) * s + alpha * mu * np.eye(n)


def ewm_cov(returns: NDArray[np.float64], halflife: float) -> NDArray[np.float64]:
    """Exponentially weighted covariance matrix.

    Assigns exponentially decaying weights to observations so that recent data
    has more influence.  The weight for observation ``i`` periods ago is::

        w_i  ∝  (1 - alpha)^i,   alpha = 1 - exp(-ln(2) / halflife)

    This means observations ``halflife`` periods in the past carry half the
    weight of the most recent observation.  A shorter halflife produces a
    covariance matrix that reacts faster to changing market conditions but is
    noisier; a longer halflife is smoother but slower to adapt.

    Args:
        returns: Returns matrix of shape (T, n).
        halflife: Decay halflife in number of periods (same unit as rows).
            Common choices: 21 (1-month), 63 (1-quarter), 126 (6-month).

    Returns:
        Positive-semidefinite (n × n) covariance matrix.
    """
    t, n = returns.shape
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    # Most-recent observation gets weight 1; oldest gets (1-alpha)^(T-1).
    raw = np.array([(1.0 - alpha) ** (t - 1 - i) for i in range(t)])
    w = raw / raw.sum()
    mu = (w[:, None] * returns).sum(axis=0)
    x = returns - mu
    return x.T @ (w[:, None] * x)


def annualize_cov(cov: NDArray[np.float64], periods_per_year: int) -> NDArray[np.float64]:
    """Scale a per-period covariance matrix to annualized units.

    Under i.i.d. returns the variance of a T-period sum scales linearly with T,
    so multiplying the per-period covariance by ``periods_per_year`` gives the
    annualized covariance used in mean-variance optimization.

    Args:
        cov: Per-period (n × n) covariance matrix.
        periods_per_year: 252 for daily, 52 for weekly, 12 for monthly.

    Returns:
        Annualized (n × n) covariance matrix.
    """
    return cov * periods_per_year


def cov_to_corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a covariance matrix to a correlation matrix.

    Normalizes each element by the product of the corresponding marginal
    standard deviations::

        corr[i, j] = cov[i, j] / (sigma_i * sigma_j)

    Args:
        cov: Symmetric (n × n) covariance matrix with positive diagonal.

    Returns:
        Correlation matrix with ones on the diagonal and all entries in [-1, 1].
    """
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)
