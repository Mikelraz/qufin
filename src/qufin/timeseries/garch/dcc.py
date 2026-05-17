"""
Engle (2002) Dynamic Conditional Correlation (DCC) GARCH.

Two-stage estimation
--------------------
Stage 1.  Fit a univariate GARCH model to each return series independently
          to obtain conditional variances h_{i,t} and standardised residuals
          z_{i,t} = ε_{i,t} / √h_{i,t}.

Stage 2.  Estimate the DCC parameters (a, b) from the standardised
          residual matrix Z ∈ ℝ^{T × k} via the recursion

              Q_t = (1 − a − b) Q̄ + a (z_{t-1} z_{t-1}') + b Q_{t-1}
              R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

          where Q̄ is the unconditional correlation of Z (i.e. its empirical
          covariance after standardisation).  The Gaussian log-likelihood
          over R_t is maximised by L-BFGS-B on the logit/softplus
          reparameterisation that keeps a, b > 0 and a + b < 1.

The combined conditional covariance is then H_t = D_t R_t D_t with
D_t = diag(√h_{1,t}, …, √h_{k,t}).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.optimize
from numba import njit

from .._io import to_numpy_2d, validate_finite, validate_min_length
from ..utils import info_criteria
from .garch import GARCH, GARCHFitResult

# ruff: noqa: N803, N806  — matrix variables use econometric uppercase (D, H, Q, R, Z)


# ---------------------------------------------------------------------------
# Numba-jitted DCC recursion
# ---------------------------------------------------------------------------


@njit(cache=True)
def _dcc_recursion(
    Z: np.ndarray, a: float, b: float, Q_bar: np.ndarray
) -> tuple[np.ndarray, float]:
    """Run the DCC Q-recursion and Gaussian log-likelihood over R_t.

    Parameters
    ----------
    Z       Standardised residuals, shape (T, k).
    a       DCC ARCH-type coefficient (≥ 0).
    b       DCC GARCH-type coefficient (≥ 0).  Stationarity ⇒ a + b < 1.
    Q_bar   Unconditional correlation of Z, shape (k, k).

    Returns
    -------
    R_paths : shape (T, k, k)  — sequence of correlation matrices.
    log_lik : float
        Σ_t −½ [log|R_t| + z_t' R_t^{-1} z_t − z_t' z_t].
        (Standard DCC stage-2 log-likelihood; the z_t' z_t subtraction
        cancels the diagonal contribution already accounted for in the
        univariate likelihoods.)
    """
    t_total, k = Z.shape
    R_paths = np.empty((t_total, k, k))
    Q = Q_bar.copy()
    log_lik = 0.0
    one_minus_ab = 1.0 - a - b
    for t in range(t_total):
        if t == 0:
            Q = Q_bar.copy()
        else:
            z_lag = Z[t - 1]
            outer = np.outer(z_lag, z_lag)
            Q = one_minus_ab * Q_bar + a * outer + b * Q
        # Build R_t = diag(Q)^{-1/2} Q diag(Q)^{-1/2}
        d = np.empty(k)
        for i in range(k):
            d[i] = math.sqrt(Q[i, i]) if Q[i, i] > 0.0 else 1e-12
        R = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = Q[i, j] / (d[i] * d[j])
        R_paths[t] = R
        # Log-likelihood contribution
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0.0:
            return R_paths, -1e10
        z_t = Z[t]
        # Solve R x = z_t
        try:
            sol = np.linalg.solve(R, z_t)
        except Exception:
            return R_paths, -1e10
        quad = 0.0
        ztz = 0.0
        for i in range(k):
            quad += z_t[i] * sol[i]
            ztz += z_t[i] * z_t[i]
        log_lik += -0.5 * (logdet + quad - ztz)
    return R_paths, log_lik


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DCCFitResult:
    """Fitted DCC-GARCH model output."""

    k: int
    a: float
    b: float
    univariate_results: list[GARCHFitResult]
    sigma2: np.ndarray  # univariate conditional variances, shape (T, k)
    std_residuals: np.ndarray  # standardised residuals z_t = ε_t / σ_t, shape (T, k)
    R: np.ndarray  # correlation paths, shape (T, k, k)
    H: np.ndarray  # full conditional covariance, shape (T, k, k)
    Q_bar: np.ndarray  # unconditional correlation
    log_lik: float  # combined two-stage log-likelihood
    dcc_log_lik: float  # stage-2 log-likelihood
    aic: float
    bic: float
    hqic: float
    n_obs: int
    persistence: float  # a + b

    def __str__(self) -> str:
        return (
            f"DCC-GARCH  k={self.k}  n_obs={self.n_obs}\n"
            f"  a={self.a:.4f}  b={self.b:.4f}  a+b={self.persistence:.4f}\n"
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        """One row per (t, i, j) entry of R_t."""
        t_total, k, _ = self.R.shape
        ts = np.repeat(np.arange(t_total, dtype=np.int64), k * k)
        is_ = np.tile(np.repeat(np.arange(k, dtype=np.int64), k), t_total)
        js = np.tile(np.tile(np.arange(k, dtype=np.int64), k), t_total)
        return pl.DataFrame(
            {
                "t": ts,
                "i": is_,
                "j": js,
                "R": self.R.ravel(),
            }
        )


# ---------------------------------------------------------------------------
# DCC class
# ---------------------------------------------------------------------------


class DCC:
    """Two-stage Engle (2002) DCC-GARCH.

    Parameters
    ----------
    garch_specs : list[GARCH] | None
        Per-series univariate GARCH specifications.  Must have length k
        (matching the number of columns of the input).  If None, defaults
        to GARCH(1, 1, mean='constant') for every column.
    """

    def __init__(self, garch_specs: list[GARCH] | None = None) -> None:
        self.garch_specs = garch_specs
        self._result: DCCFitResult | None = None

    @property
    def result(self) -> DCCFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    def fit(self, returns: np.ndarray) -> DCCFitResult:
        """Fit the two-stage DCC-GARCH model.

        Parameters
        ----------
        returns : array_like, shape (T, k)
            Multivariate return matrix.

        Returns
        -------
        DCCFitResult
        """
        arr = to_numpy_2d(returns)
        validate_finite(arr)
        validate_min_length(arr, 30, "returns")
        T, k = arr.shape
        if k < 2:
            raise ValueError(f"DCC requires k ≥ 2 series; got k = {k}.")

        # Stage 1: univariate GARCH per column.
        if self.garch_specs is None:
            specs = [GARCH(p=1, q=1, mean="constant") for _ in range(k)]
        else:
            if len(self.garch_specs) != k:
                raise ValueError(
                    f"garch_specs has length {len(self.garch_specs)} but returns has k = {k}."
                )
            specs = self.garch_specs

        univ_results: list[GARCHFitResult] = []
        sigma2 = np.empty((T, k))
        std_resid = np.empty((T, k))
        univ_ll = 0.0
        for i, spec in enumerate(specs):
            r = spec.fit(arr[:, i])
            univ_results.append(r)
            sigma2[:, i] = r.sigma2
            std_resid[:, i] = r.std_residuals
            univ_ll += r.log_lik

        # Stage 2: estimate (a, b) by L-BFGS-B on standardised residuals.
        Z = np.ascontiguousarray(std_resid)
        Q_bar = (Z.T @ Z) / T
        # Normalise Q_bar diagonal to 1 (it's a covariance of standardised
        # residuals — diagonal is the sample variance of z_i,t which should
        # be ≈ 1 but we enforce exactly 1 for numerical stability).
        d = np.sqrt(np.diag(Q_bar))
        d = np.where(d <= 0, 1.0, d)
        Q_bar = Q_bar / np.outer(d, d)

        def _unpack(params: np.ndarray) -> tuple[float, float]:
            # softplus → positive, then project a + b into (0, 1).
            sa = math.log1p(math.exp(min(float(params[0]), 30.0)))
            sb = math.log1p(math.exp(min(float(params[1]), 30.0)))
            tot = sa + sb
            if tot <= 0:
                return 0.0, 0.0
            eff = tot / (1.0 + tot)
            scale = eff / tot
            return sa * scale, sb * scale

        def neg_ll(params: np.ndarray) -> float:
            a, b = _unpack(params)
            try:
                _, ll = _dcc_recursion(Z, a, b, Q_bar)
            except Exception:
                return 1e10
            if not math.isfinite(ll):
                return 1e10
            return -ll

        # Warm-start: a ≈ 0.05, b ≈ 0.9 → softplus pre-images ~ −3, 2.25.
        x0 = np.array([-3.0, 2.25])
        opt = scipy.optimize.minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        a, b = _unpack(opt.x)
        R_paths, dcc_ll = _dcc_recursion(Z, a, b, Q_bar)

        # Build H_t = D_t R_t D_t
        H = np.empty((T, k, k))
        for t in range(T):
            d_t = np.sqrt(sigma2[t])
            H[t] = R_paths[t] * np.outer(d_t, d_t)

        total_ll = univ_ll + dcc_ll
        n_params_total = (
            sum(1 + r.q + r.p + (1 if r.mu != 0.0 else 0) for r in univ_results) + 2
        )  # +2 for DCC (a, b)
        aic_v, bic_v, hqic_v = info_criteria(total_ll, T, n_params_total)

        persistence = a + b
        if persistence >= 1.0 - 1e-8:
            warnings.warn(
                f"Fitted DCC has a + b = {persistence:.4f} ≥ 1 — correlation "
                "process is non-stationary.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._result = DCCFitResult(
            k=k,
            a=float(a),
            b=float(b),
            univariate_results=univ_results,
            sigma2=sigma2,
            std_residuals=std_resid,
            R=R_paths,
            H=H,
            Q_bar=Q_bar,
            log_lik=float(total_ll),
            dcc_log_lik=float(dcc_ll),
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=T,
            persistence=float(persistence),
        )
        return self._result
