"""
Vector Error Correction Model (VECM) estimation by reduced-rank regression.

Model
-----
    Δy_t = Π y_{t-1} + Σ_{i=1}^{k_ar_diff} Γ_i Δy_{t-i} + μ + ε_t,

with the long-run impact matrix factorised as Π = α β'.  Here

    β  (k × r)  cointegrating vectors,
    α  (k × r)  loading (adjustment) coefficients,
    r          cointegration rank (1 ≤ r ≤ k − 1 for a proper VECM).

Estimation
----------
Johansen's reduced-rank maximum-likelihood procedure:

1. Concentrate out Z_t = [ΔY_{lags}, μ] by OLS, giving residuals R₀, R₁.
2. Solve the generalised eigenproblem on the moment matrices and take the
   r leading eigenvectors as β.
3. Recover α = S_{01} β (S_{11} normalisation gives β' S_{11} β = I).
4. Estimate Γ_i from the OLS of R₀ − α β' R₁ on the original Z_t.

The implementation delegates the eigendecomposition to ``johansen`` and
performs the final reduced-rank fit explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .._io import to_numpy_2d, validate_finite, validate_min_length
from .johansen import johansen

# ruff: noqa: N803, N806  — econometric matrices use standard uppercase (alpha=α, beta=β, Gamma=Γ)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VECMResult:
    """Reduced-rank VECM fit.

    Attributes
    ----------
    rank             Cointegration rank r used.
    k                Number of variables.
    k_ar_diff        Number of lagged-difference Γ matrices.
    det_order        Deterministic specification (0 or 1).
    alpha            Loading matrix α, shape (k, r).
    beta             Cointegrating vectors β, shape (k, r) — first row normalised to 1.
    pi               Π = α β', shape (k, k).
    gamma            Stacked Γ matrices, shape (k_ar_diff, k, k).
    const            Drift μ (zero if det_order == 0).
    sigma_u          Innovation covariance Σ_u, shape (k, k).
    residuals        Fitted-equation residuals, shape (T_eff, k).
    fitted_values    Fitted Δy_t values, shape (T_eff, k).
    log_lik          Gaussian log-likelihood at the rank-r solution.
    n_obs            Effective sample size T_eff.
    """

    rank: int
    k: int
    k_ar_diff: int
    det_order: int
    alpha: np.ndarray
    beta: np.ndarray
    pi: np.ndarray
    gamma: np.ndarray
    const: np.ndarray
    sigma_u: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    log_lik: float
    n_obs: int

    def __str__(self) -> str:
        return (
            f"VECM(rank={self.rank}, k={self.k}, k_ar_diff={self.k_ar_diff})\n"
            f"  log_lik={self.log_lik:.4f}  n_obs={self.n_obs}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Long-format DataFrame of α, β coefficients.

        Columns: ``component`` ('alpha' | 'beta'), ``i``, ``r``, ``value``.
        """
        records: list[dict[str, object]] = []
        for i in range(self.k):
            for r in range(self.rank):
                records.append(
                    {
                        "component": "alpha",
                        "i": i,
                        "r": r,
                        "value": float(self.alpha[i, r]),
                    }
                )
                records.append(
                    {
                        "component": "beta",
                        "i": i,
                        "r": r,
                        "value": float(self.beta[i, r]),
                    }
                )
        return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def vecm(
    y: np.ndarray,
    *,
    coint_rank: int,
    k_ar_diff: int = 1,
    det_order: int = 0,
) -> VECMResult:
    """Estimate a rank-restricted VECM.

    Parameters
    ----------
    y           T × k observation matrix.
    coint_rank  Cointegration rank r.  Must satisfy 1 ≤ r ≤ k − 1.
    k_ar_diff   Number of lagged-difference terms.
    det_order   ``0`` (constant restricted to cointegration relation)
                or ``1`` (unrestricted constant).

    Returns
    -------
    VECMResult
    """
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be ≥ 0, got {k_ar_diff}.")
    if det_order not in (0, 1):
        raise ValueError(f"det_order must be 0 or 1, got {det_order}.")
    arr = to_numpy_2d(y)
    validate_finite(arr, "y")
    T, k = arr.shape
    if not (1 <= coint_rank <= k - 1):
        raise ValueError(f"coint_rank must satisfy 1 ≤ r ≤ k-1; got r={coint_rank}, k={k}.")
    validate_min_length(arr, k_ar_diff + 3, "y")

    # Step 1: Johansen eigendecomposition for β
    jres = johansen(arr, k_ar_diff=k_ar_diff, det_order=det_order)
    beta = jres.eigenvectors[:, :coint_rank].copy()  # (k, r)
    # Normalise so the first row is the identity block (Phillips-style).
    # Use Moore-Penrose to handle near-singular first block.
    first_block = beta[:coint_rank, :]  # (r, r)
    if np.linalg.matrix_rank(first_block) == coint_rank:
        beta = beta @ np.linalg.solve(first_block, np.eye(coint_rank))

    # Step 2: rebuild the partialled-out matrices to recover α and Γ_i.
    dy = np.diff(arr, axis=0)
    lev = arr[:-1, :]
    T_eff = dy.shape[0] - k_ar_diff

    z_cols: list[np.ndarray] = []
    for i in range(1, k_ar_diff + 1):
        z_cols.append(dy[k_ar_diff - i : -i, :])
    R_dy = dy[k_ar_diff:, :]
    R_lev = lev[k_ar_diff:, :]
    if det_order == 1:
        z_cols.append(np.ones((T_eff, 1)))

    if z_cols:
        Z = np.column_stack(z_cols)
        ZtZ = Z.T @ Z
        Z_proj = Z @ np.linalg.solve(ZtZ, Z.T)
        R0 = R_dy - Z_proj @ R_dy
        R1 = R_lev - Z_proj @ R_lev
    else:
        R0 = R_dy
        R1 = R_lev

    # α = S_{01} β  (with S_{11} normalisation already implicit in eigvecs)
    S01 = R0.T @ R1 / T_eff
    alpha_mat = S01 @ beta  # (k, r)
    pi_mat = alpha_mat @ beta.T  # (k, k)

    # Step 3: estimate Γ_i (and μ) from the full OLS  Δy_t = α β' y_{t-1} + Γ Z_t + ε_t
    # Build full regressor matrix [β' y_{t-1}, Z]
    ec_term = R_lev @ beta  # (T_eff, r)
    if z_cols:
        full_X = np.column_stack([ec_term, Z])
    else:
        full_X = ec_term
    full_Y = R_dy
    coef, *_ = np.linalg.lstsq(full_X, full_Y, rcond=None)
    # coef shape (r + k*k_ar_diff [+1], k); transpose to read row-wise per equation.
    coef = coef.T  # (k, ncols)
    # First r columns are α (overwrite the moment-based estimate for tighter fit)
    alpha_mat = coef[:, :coint_rank]
    pi_mat = alpha_mat @ beta.T

    # Γ matrices
    gamma = np.zeros((k_ar_diff, k, k))
    for i in range(k_ar_diff):
        gamma[i] = coef[:, coint_rank + i * k : coint_rank + (i + 1) * k]

    const = coef[:, coint_rank + k_ar_diff * k].copy() if det_order == 1 else np.zeros(k)

    # Fitted values and residuals on the original Δy scale
    fitted = full_X @ coef.T
    resid = full_Y - fitted
    sigma_u = (resid.T @ resid) / T_eff

    sign, logdet = np.linalg.slogdet(sigma_u)
    if sign <= 0:
        log_lik = -np.inf
    else:
        log_lik = float(-0.5 * T_eff * (k * np.log(2.0 * np.pi) + logdet + k))

    return VECMResult(
        rank=coint_rank,
        k=k,
        k_ar_diff=k_ar_diff,
        det_order=det_order,
        alpha=alpha_mat,
        beta=beta,
        pi=pi_mat,
        gamma=gamma,
        const=const,
        sigma_u=sigma_u,
        residuals=resid,
        fitted_values=fitted,
        log_lik=log_lik,
        n_obs=T_eff,
    )
