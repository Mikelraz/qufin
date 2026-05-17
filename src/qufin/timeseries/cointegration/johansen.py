"""
Johansen (1991) cointegration test.

VECM representation
-------------------
A VAR(k) on a k-dimensional series y_t can be rewritten as the vector
error-correction model

    Δy_t = Π y_{t-1} + Σ_{i=1}^{k_ar_diff} Γ_i Δy_{t-i} + μ + ε_t,

where Π = αβ' has rank r equal to the number of cointegrating relations.
The Johansen procedure tests sequentially  H₀: rank(Π) = r  vs.
H₁: rank(Π) > r  using either the *trace* or *maximum-eigenvalue* statistic
constructed from a canonical-correlation analysis of Δy_t against y_{t-1}
after partialling out the lagged-difference regressors.

Algorithm
---------
1. Form Δy_t and the regressor matrix Z_t of lagged Δy plus deterministic
   terms (constant / trend) according to ``det_order``.
2. OLS-regress Δy_t and y_{t-1} on Z_t; collect residuals R₀_t and R₁_t.
3. Compute the moment matrices S_{00}, S_{01}, S_{11}.
4. Solve the generalised eigenproblem
   |λ S_{11} − S_{10} S_{00}^{-1} S_{01}| = 0.
5. Trace stat   = −T Σ_{i=r+1}^k log(1 − λ_i).
   Max-eig stat = −T log(1 − λ_{r+1}).

Critical values
---------------
Hardcoded asymptotic critical values from Osterwald-Lenum (1992) Table 1
for the constant-in-cointegrating-vector (``det_order = 0``) specification,
and Table 2 for the unrestricted-constant (``det_order = 1``) specification.
Available for k ≤ 12.  P-values are obtained by piecewise log-linear
interpolation through the 1 %, 5 %, 10 % critical values plus a soft
right-side anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._io import to_numpy_2d, validate_finite, validate_min_length

# ruff: noqa: N803, N806  — econometric matrices use standard uppercase (S, Pi, R, Z, X, Y)


# ---------------------------------------------------------------------------
# Osterwald-Lenum (1992) critical values
# ---------------------------------------------------------------------------
#
# Index layout: _OL_CRITS[det_order][stat_type][n_cointegrated_residuals (= k − r)][level]
# det_order:   0 = constant in cointegrating relation
#              1 = unrestricted constant (drift in levels)
# stat_type:   'trace' or 'max_eig'
# Tables 1 and 2 from Osterwald-Lenum (1992).  Indices m = k − r in {1, …, 12}.

_LEVELS = (0.10, 0.05, 0.01)

# fmt: off
_OL_TRACE_DET0: dict[int, tuple[float, float, float]] = {
    1:  ( 2.71,  3.84,  6.65),
    2:  (13.31, 15.41, 20.04),
    3:  (27.16, 29.68, 35.65),
    4:  (44.49, 47.21, 54.46),
    5:  (65.82, 68.52, 76.07),
    6:  (90.39, 94.15, 103.18),
    7:  (118.99, 124.24, 133.57),
    8:  (151.38, 156.00, 168.36),
    9:  (186.54, 192.89, 204.95),
    10: (226.34, 233.13, 247.18),
    11: (269.53, 277.71, 293.44),
    12: (316.55, 326.06, 343.20),
}

_OL_MAX_DET0: dict[int, tuple[float, float, float]] = {
    1:  ( 2.71,  3.84,  6.65),
    2:  (12.07, 14.07, 18.63),
    3:  (18.60, 20.97, 25.52),
    4:  (24.73, 27.07, 32.24),
    5:  (30.90, 33.46, 38.77),
    6:  (36.76, 39.37, 45.10),
    7:  (42.32, 45.28, 51.57),
    8:  (48.33, 51.42, 57.69),
    9:  (53.98, 57.12, 63.71),
    10: (59.62, 62.81, 69.94),
    11: (65.38, 68.83, 76.63),
    12: (70.60, 74.36, 82.45),
}

_OL_TRACE_DET1: dict[int, tuple[float, float, float]] = {
    1:  ( 2.69,  3.76,  6.65),
    2:  (13.33, 15.41, 20.04),
    3:  (26.79, 29.68, 35.65),
    4:  (43.95, 47.21, 54.46),
    5:  (64.84, 68.52, 76.07),
    6:  (89.48, 94.15, 103.18),
    7:  (118.50, 124.24, 133.57),
    8:  (150.53, 156.00, 168.36),
    9:  (186.39, 192.89, 204.95),
    10: (225.85, 233.13, 247.18),
    11: (269.96, 277.71, 293.44),
    12: (317.95, 326.06, 343.20),
}

_OL_MAX_DET1: dict[int, tuple[float, float, float]] = {
    1:  ( 2.69,  3.76,  6.65),
    2:  (12.07, 14.07, 18.63),
    3:  (18.60, 20.97, 25.52),
    4:  (24.73, 27.07, 32.24),
    5:  (30.90, 33.46, 38.77),
    6:  (36.76, 39.37, 45.10),
    7:  (42.32, 45.28, 51.57),
    8:  (48.33, 51.42, 57.69),
    9:  (53.98, 57.12, 63.71),
    10: (59.62, 62.81, 69.94),
    11: (65.38, 68.83, 76.63),
    12: (70.60, 74.36, 82.45),
}
# fmt: on


def _crit_table(det_order: int, stat: str) -> dict[int, tuple[float, float, float]]:
    if det_order == 0 and stat == "trace":
        return _OL_TRACE_DET0
    if det_order == 0 and stat == "max_eig":
        return _OL_MAX_DET0
    if det_order == 1 and stat == "trace":
        return _OL_TRACE_DET1
    if det_order == 1 and stat == "max_eig":
        return _OL_MAX_DET1
    raise ValueError(f"No critical-value table for det_order={det_order}, stat={stat!r}.")


def _pvalue_from_crits(stat: float, crits_at_10_5_1: tuple[float, float, float]) -> float:
    """Piecewise log-linear p-value through (10%, 5%, 1%) critical values."""
    c10, c5, c1 = crits_at_10_5_1
    if stat <= 0.0:
        return 1.0 - 1e-10
    if stat <= c10:
        # Smooth ramp 0.10 → 0.99 between 0 and c10.
        return float(min(1.0 - 1e-10, 0.10 + (c10 - stat) / c10 * 0.89))
    nodes = np.array([c10, c5, c1])
    log_ps = np.log(np.array([0.10, 0.05, 0.01]))
    if stat >= c1:
        slope = (log_ps[2] - log_ps[1]) / (nodes[2] - nodes[1])
        log_p = log_ps[2] + slope * (stat - c1)
        return float(max(1e-10, math.exp(log_p)))
    log_p = float(np.interp(stat, nodes, log_ps))
    return float(min(1.0 - 1e-10, max(1e-10, math.exp(log_p))))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JohansenResult:
    """Johansen test result.

    Attributes
    ----------
    eigenvalues        Squared canonical correlations λ_1 ≥ … ≥ λ_k, shape (k,)
    eigenvectors       Cointegrating vectors β, shape (k, k) — column i is β_i
    loadings           Adjustment coefficients α = S_{01} β,  shape (k, k)
    trace_stats        Trace statistics for r = 0, …, k − 1, shape (k,)
    max_eig_stats      Max-eigenvalue statistics, shape (k,)
    trace_p_values     p-values for the trace statistics
    max_eig_p_values   p-values for the max-eigenvalue statistics
    trace_crits        Trace critical values, shape (k, 3)  [10 %, 5 %, 1 %]
    max_eig_crits      Max-eigenvalue critical values, shape (k, 3)
    rank_trace         Estimated cointegration rank at the 5 % level (trace)
    rank_max_eig       Estimated rank at the 5 % level (max-eigenvalue)
    n_obs              Effective sample size T − k_ar_diff − 1
    k                  Number of variables
    k_ar_diff          Number of lagged-difference terms
    det_order          Deterministic specification used
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    loadings: np.ndarray
    trace_stats: np.ndarray
    max_eig_stats: np.ndarray
    trace_p_values: np.ndarray
    max_eig_p_values: np.ndarray
    trace_crits: np.ndarray
    max_eig_crits: np.ndarray
    rank_trace: int
    rank_max_eig: int
    n_obs: int
    k: int
    k_ar_diff: int
    det_order: int


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def johansen(
    y: np.ndarray,
    *,
    k_ar_diff: int = 1,
    det_order: int = 0,
    alpha: float = 0.05,
) -> JohansenResult:
    """Johansen cointegration test (canonical-correlation method).

    Parameters
    ----------
    y          T × k observation matrix (each column a variable).
    k_ar_diff  Number of lagged-difference terms Σ Γ_i Δy_{t-i}.  Note that
               this counts *differences*; a VAR(k=2) corresponds to
               ``k_ar_diff = 1``.  Must be ≥ 0.
    det_order  ``0``: constant restricted to the cointegration relation
               (no drift in levels).
               ``1``: unrestricted constant (drift permitted in levels).
    alpha      Significance level for the convenience rank estimates.

    Returns
    -------
    JohansenResult
    """
    if det_order not in (0, 1):
        raise ValueError(f"det_order must be 0 or 1, got {det_order}.")
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be ≥ 0, got {k_ar_diff}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    arr = to_numpy_2d(y)
    validate_finite(arr, "y")
    T, k = arr.shape
    if k > 12:
        raise ValueError(f"Johansen critical values are tabulated for k ≤ 12; got k = {k}.")
    validate_min_length(arr, k_ar_diff + 3, "y")

    # Differences
    dy = np.diff(arr, axis=0)  # (T − 1, k)
    lev = arr[:-1, :]  # y_{t-1}, length T − 1

    # Effective sample after dropping k_ar_diff initial differences.
    T_eff = dy.shape[0] - k_ar_diff
    if T_eff <= k:
        raise ValueError(f"Insufficient observations after differencing: T_eff={T_eff}, k={k}.")

    # Z: lagged differences and deterministic terms
    z_cols: list[np.ndarray] = []
    for i in range(1, k_ar_diff + 1):
        z_cols.append(dy[k_ar_diff - i : -i, :] if i > 0 else dy[k_ar_diff:, :])

    R_dy = dy[k_ar_diff:, :]  # response for Δy
    R_lev = lev[k_ar_diff:, :]  # response for y_{t-1}

    # Deterministic
    # det_order = 0 → constant restricted: append a column of 1s to lev (R_1).
    # det_order = 1 → unrestricted constant: append a column of 1s to Z.
    if det_order == 0:
        R_lev = np.column_stack([R_lev, np.ones(T_eff)])
    if det_order == 1:
        z_cols.append(np.ones((T_eff, 1)))

    if z_cols:
        Z = np.column_stack(z_cols)
        # Partial out Z from R_dy and R_lev via OLS residuals
        R0 = _residualise(R_dy, Z)
        R1 = _residualise(R_lev, Z)
    else:
        R0 = R_dy
        R1 = R_lev

    # Moment matrices (divide by T_eff)
    S00 = R0.T @ R0 / T_eff
    S01 = R0.T @ R1 / T_eff
    S11 = R1.T @ R1 / T_eff
    S10 = S01.T

    # Solve the generalised eigenproblem  S_{11}^{-1} S_{10} S_{00}^{-1} S_{01} v = λ v
    try:
        S00_inv_S01 = np.linalg.solve(S00, S01)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "S_00 is singular — reduce k_ar_diff or check for collinear columns in y."
        ) from exc
    try:
        M = np.linalg.solve(S11, S10 @ S00_inv_S01)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "S_11 is singular — likely caused by perfectly collinear data."
        ) from exc

    eigvals, eigvecs = np.linalg.eig(M)
    # Force real (small imaginary parts can appear due to numerical noise).
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    # Sort descending, clip into (0, 1) for log-stability.
    order = np.argsort(-eigvals)
    eigvals = np.clip(eigvals[order], 1e-15, 1.0 - 1e-15)
    eigvecs = eigvecs[:, order]

    # Only the first k eigenvalues correspond to the y_{t-1} block when
    # det_order == 0 (the (k+1)-th relates to the restricted constant).
    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:k, :k]

    # Trace and max-eigenvalue statistics for r = 0, …, k − 1.
    trace_stats = np.zeros(k)
    max_eig_stats = np.zeros(k)
    for r in range(k):
        trace_stats[r] = -T_eff * float(np.sum(np.log(1.0 - eigvals_k[r:])))
        max_eig_stats[r] = -T_eff * float(np.log(1.0 - eigvals_k[r]))

    # Loadings α = S_{01} β  (using the un-normalised eigenvectors).
    loadings = S01[:k, :k] @ eigvecs_k

    # Critical values and p-values
    trace_table = _crit_table(det_order, "trace")
    max_table = _crit_table(det_order, "max_eig")
    trace_crits = np.zeros((k, 3))
    max_crits = np.zeros((k, 3))
    trace_pvals = np.zeros(k)
    max_pvals = np.zeros(k)
    for r in range(k):
        m = k - r
        trace_crits[r, :] = trace_table[m]
        max_crits[r, :] = max_table[m]
        trace_pvals[r] = _pvalue_from_crits(trace_stats[r], trace_table[m])
        max_pvals[r] = _pvalue_from_crits(max_eig_stats[r], max_table[m])

    # Estimate rank: smallest r such that we cannot reject H₀: rank ≤ r at level α.
    rank_trace = _estimate_rank(trace_stats, trace_crits, alpha)
    rank_max = _estimate_rank(max_eig_stats, max_crits, alpha)

    return JohansenResult(
        eigenvalues=eigvals_k,
        eigenvectors=eigvecs_k,
        loadings=loadings,
        trace_stats=trace_stats,
        max_eig_stats=max_eig_stats,
        trace_p_values=trace_pvals,
        max_eig_p_values=max_pvals,
        trace_crits=trace_crits,
        max_eig_crits=max_crits,
        rank_trace=rank_trace,
        rank_max_eig=rank_max,
        n_obs=T_eff,
        k=k,
        k_ar_diff=k_ar_diff,
        det_order=det_order,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _residualise(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return ``A − Z (Z'Z)^{-1} Z'A`` (column-wise OLS residuals)."""
    ZtZ = Z.T @ Z
    coef = np.linalg.solve(ZtZ, Z.T @ A)
    return A - Z @ coef


def _estimate_rank(stats: np.ndarray, crits: np.ndarray, alpha: float) -> int:
    """Sequential test: smallest r such that stats[r] does not exceed crit_α."""
    col_idx = {0.10: 0, 0.05: 1, 0.01: 2}
    if alpha not in col_idx:
        raise ValueError(f"alpha must be one of {sorted(col_idx)}, got {alpha}.")
    j = col_idx[alpha]
    k = stats.shape[0]
    for r in range(k):
        if stats[r] <= crits[r, j]:
            return r
    return k
