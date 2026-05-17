"""
Shared numba-jitted numerical kernels for the timeseries subpackage.

Convention
----------
* All kernels accept only primitive types: ``np.ndarray`` (float64 contiguous),
  scalars, and booleans.  Never dataclasses, polars objects, or Python objects.
* Every kernel is decorated with ``@njit(cache=True)`` so compilation is
  amortised across test runs.
* ``parallel=True`` is used only when iterations are independent (outer-lag
  loops in ``sample_acf``); recursive loops (Durbin-Levinson) are plain njit.

The kernels are deliberately small and verb-named.  Result containers and
public API live in the model modules that consume them.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def sample_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Sample autocorrelation function at lags 1 … nlags.

    Uses the biased estimator (divisor n, not n - k) — standard for ACF plots
    and consistent with the Ljung-Box test definition.

    Parameters
    ----------
    x      Demeaning is done internally.
    nlags  Number of positive lags to compute (>= 1).

    Returns
    -------
    np.ndarray, shape (nlags,)
    """
    n = x.shape[0]
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    var = 0.0
    for i in range(n):
        d = x[i] - mean
        var += d * d
    var /= n

    out = np.zeros(nlags)
    if var == 0.0:
        return out

    for k in prange(1, nlags + 1):
        s = 0.0
        for i in range(n - k):
            s += (x[i] - mean) * (x[i + k] - mean)
        out[k - 1] = s / (n * var)
    return out


@njit(cache=True, parallel=True)
def sample_acovf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Sample autocovariances γ_0, γ_1, …, γ_nlags (length nlags + 1).

    Biased estimator (divisor n).
    """
    n = x.shape[0]
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    out = np.zeros(nlags + 1)
    for k in prange(nlags + 1):
        s = 0.0
        for i in range(n - k):
            s += (x[i] - mean) * (x[i + k] - mean)
        out[k] = s / n
    return out


@njit(cache=True)
def durbin_levinson(acovs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Durbin-Levinson recursion.

    Given autocovariances γ_0, …, γ_p, return:
      * AR(p) Yule-Walker coefficients φ_1, …, φ_p
      * Partial autocorrelations at lags 1 … p (PACF)
      * Innovation variances σ²_0, σ²_1, …, σ²_p

    The recursion is O(p²) and numerically stable for typical p ≤ 40.
    """
    p = acovs.shape[0] - 1
    if p < 1:
        return np.zeros(0), np.zeros(0), np.array([acovs[0]])

    phi = np.zeros((p, p))
    pacf_out = np.zeros(p)
    sigma2 = np.zeros(p + 1)
    sigma2[0] = acovs[0]

    if sigma2[0] <= 0.0:
        return np.zeros(p), np.zeros(p), sigma2

    phi[0, 0] = acovs[1] / sigma2[0]
    pacf_out[0] = phi[0, 0]
    sigma2[1] = sigma2[0] * (1.0 - phi[0, 0] * phi[0, 0])

    for m in range(1, p):
        if sigma2[m] <= 0.0:
            break
        s = 0.0
        for j in range(m):
            s += phi[m - 1, j] * acovs[m - j]
        phi_mm = (acovs[m + 1] - s) / sigma2[m]
        for j in range(m):
            phi[m, j] = phi[m - 1, j] - phi_mm * phi[m - 1, m - 1 - j]
        phi[m, m] = phi_mm
        pacf_out[m] = phi_mm
        sigma2[m + 1] = sigma2[m] * (1.0 - phi_mm * phi_mm)

    ar = phi[p - 1, :].copy()
    return ar, pacf_out, sigma2


@njit(cache=True)
def lag_matrix(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the lag-design matrix and aligned target for an AR(p) regression.

    The regression form is

        y_t = φ_1 x_{t-1} + φ_2 x_{t-2} + … + φ_p x_{t-p} + ε_t

    Returns
    -------
    X : shape (T - p, p)
        Row t has columns ``[x_{t+p-1}, x_{t+p-2}, …, x_t]`` — most recent lag first.
    y : shape (T - p,)
        ``y[t] = x[t + p]``.
    """
    t_total = x.shape[0]
    if t_total <= p:
        return np.empty((0, p)), np.empty(0)
    n = t_total - p
    out_x = np.empty((n, p))
    out_y = np.empty(n)
    for t in range(n):
        for i in range(p):
            out_x[t, i] = x[t + p - 1 - i]
        out_y[t] = x[t + p]
    return out_x, out_y


@njit(cache=True)
def yule_walker_solve(acovs: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve the Yule-Walker equations for AR(p) coefficients.

    Convenience wrapper around ``durbin_levinson``: given autocovariances
    γ_0 … γ_p, return the AR coefficients and the innovation variance σ²_p.
    """
    ar, _, sigma2 = durbin_levinson(acovs)
    return ar, float(sigma2[-1])
