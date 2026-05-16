"""
General utilities for the timeseries subpackage.

* ``difference`` / ``inverse_difference`` — d-fold integer differencing and
  its exact inverse, given the first d values of the original series.
* ``seasonal_difference`` — multiplicative seasonal differencing of period s.
* ``info_criteria`` — AIC, BIC, HQIC from a log-likelihood + parameter count.
"""

from __future__ import annotations

import math

import numpy as np

from ._io import to_numpy_1d


def difference(x: np.ndarray, d: int = 1) -> np.ndarray:
    """
    Apply integer differencing of order ``d``.

    Equivalent to ``np.diff(x, n=d)`` — included here for symmetry with
    ``inverse_difference``.  Returns an array of length ``len(x) - d``.
    """
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}.")
    arr = to_numpy_1d(x)
    if d == 0:
        return arr.copy()
    if arr.shape[0] <= d:
        raise ValueError(
            f"difference: cannot apply order-{d} differencing to a series of length {arr.shape[0]}."
        )
    return np.diff(arr, n=d)


def seasonal_difference(x: np.ndarray, s: int, d_seasonal: int = 1) -> np.ndarray:
    """
    Apply seasonal differencing of period ``s``, repeated ``d_seasonal`` times.

    ``Δ_s x_t = x_t - x_{t-s}``.  Returns an array of length
    ``len(x) - s * d_seasonal``.  The order parameter is named ``d_seasonal``
    (rather than the SARIMA-standard ``D``) for ruff lower-case compliance.
    """
    if s < 1:
        raise ValueError(f"s must be >= 1, got {s}.")
    if d_seasonal < 0:
        raise ValueError(f"d_seasonal must be >= 0, got {d_seasonal}.")
    arr = to_numpy_1d(x)
    if d_seasonal == 0:
        return arr.copy()
    if arr.shape[0] <= s * d_seasonal:
        raise ValueError(
            f"seasonal_difference: cannot apply order-{d_seasonal} period-{s} "
            f"differencing to a series of length {arr.shape[0]}."
        )
    out = arr
    for _ in range(d_seasonal):
        out = out[s:] - out[:-s]
    return out


def inverse_difference(diffs: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Exact inverse of ``difference``.

    Given the ``d``-fold differenced series and the first ``d`` values of the
    original (``seeds = x[:d]``), recover the full original series of length
    ``len(diffs) + d``.

    Parameters
    ----------
    diffs   The d-fold-differenced array.
    seeds   The first d values of the undifferenced series, in order.

    Returns
    -------
    np.ndarray, shape (len(diffs) + d,)
    """
    diffs_arr = to_numpy_1d(diffs)
    seeds_arr = to_numpy_1d(seeds)
    d = seeds_arr.shape[0]
    if d == 0:
        return diffs_arr.copy()
    if diffs_arr.shape[0] == 0:
        return seeds_arr.copy()

    out = diffs_arr.copy()
    for k in range(d - 1, -1, -1):
        # At level k, integrate once.  The first value of the k-fold-difference
        # series is the k-th forward difference of seeds at index 0.
        start = float(np.diff(seeds_arr, n=k)[0]) if k > 0 else float(seeds_arr[0])
        out = np.concatenate(([start], start + np.cumsum(out)))
    return out


def info_criteria(log_lik: float, n_obs: int, n_params: int) -> tuple[float, float, float]:
    """
    Compute (AIC, BIC, HQIC) from a log-likelihood and parameter count.

    All three penalise model complexity differently:

        AIC  = -2 ℓ + 2 k
        BIC  = -2 ℓ + k ln n
        HQIC = -2 ℓ + 2 k ln ln n

    Smaller is better.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}.")
    if n_params < 0:
        raise ValueError(f"n_params must be >= 0, got {n_params}.")
    aic = -2.0 * log_lik + 2.0 * n_params
    bic = -2.0 * log_lik + n_params * math.log(n_obs)
    hq_factor = math.log(math.log(n_obs)) if n_obs > math.e else 0.0
    hqic = -2.0 * log_lik + 2.0 * n_params * hq_factor
    return aic, bic, hqic
