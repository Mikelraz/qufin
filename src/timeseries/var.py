"""
Vector Autoregression (VAR) models for multivariate time series.

Model
-----
A VAR(p) for a k-dimensional series y_t is

    y_t = c + A_1 y_{t-1} + … + A_p y_{t-p} + u_t,    u_t ~ N(0, Σ_u)

where each A_i is k × k.  Estimation is by equation-by-equation OLS (which
coincides with the Gaussian MLE for the full system because the regressors
are identical across equations).

API
---
    VAR(p).fit(y, include_const=True) → VARFitResult
    granger_causality(result, caused, causing, lags) → GrangerResult
    impulse_response(result, h, orthogonalized=True) → np.ndarray  (h, k, k)

Companion-matrix stationarity check warns if any eigenvalue of the
stacked transition matrix lies on or outside the unit circle.

Notation conventions
--------------------
* ``coef``  shape (p, k, k) — ``coef[ell, i, j] = A_{ell+1}[i, j]``.
* IRF output shape (h, k, k) — entry [t, i, j] is response of variable *i*
  at horizon *t* to a unit shock in variable *j*.  Orthogonalised IRFs use
  a lower-triangular Cholesky factor of Σ_u (Sims 1980 ordering by column
  index of the input ``y``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.stats

from ._io import to_numpy_2d, validate_finite, validate_min_length
from .utils import info_criteria

# ruff: noqa: N803, N806  — matrix variables use standard econometric uppercase (A, B, Y, X, Z, P)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class VARFitResult:
    """Fitted VAR(p) model output.

    Attributes
    ----------
    order            : p
    k                : number of variables
    coef             : (p, k, k)  AR coefficient matrices A_1, …, A_p
    const            : (k,)       intercepts (zeros if include_const is False)
    sigma_u          : (k, k)     residual covariance Σ_u (MLE divisor: T_eff)
    residuals        : (T_eff, k) one-step prediction errors
    fitted_values    : (T_eff, k) one-step ahead fitted values
    log_lik          : Gaussian log-likelihood
    aic, bic, hqic   : information criteria
    n_obs            : T_eff (= T - p)
    is_stationary    : all companion-matrix eigenvalues strictly inside unit circle
    include_const    : whether the intercept was estimated
    """

    order: int
    k: int
    coef: np.ndarray
    const: np.ndarray
    sigma_u: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    is_stationary: bool
    include_const: bool

    def __str__(self) -> str:
        lines = [
            f"VAR({self.order})  k={self.k}  n_obs={self.n_obs}",
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  stationary={self.is_stationary}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Long-format DataFrame: one row per coefficient (lag, i, j, value)."""
        records: list[dict[str, float | int]] = []
        for ell in range(self.order):
            for i in range(self.k):
                for j in range(self.k):
                    records.append(
                        {
                            "lag": ell + 1,
                            "i": i,
                            "j": j,
                            "value": float(self.coef[ell, i, j]),
                        }
                    )
        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Companion matrix and eigenvalue diagnostics
    # ------------------------------------------------------------------

    def companion_matrix(self) -> np.ndarray:
        """Build the (k p) × (k p) companion matrix of the VAR(p)."""
        k, p = self.k, self.order
        kp = k * p
        comp = np.zeros((kp, kp))
        for ell in range(p):
            comp[:k, ell * k : (ell + 1) * k] = self.coef[ell]
        if p > 1:
            comp[k:, : (p - 1) * k] = np.eye((p - 1) * k)
        return comp

    @property
    def companion_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the companion matrix (modulus < 1 ⇔ stationary)."""
        return np.linalg.eigvals(self.companion_matrix())


@dataclass
class GrangerResult:
    """Result of a Granger-causality test.

    Tests H₀: ``causing`` does **not** Granger-cause ``caused`` (all
    coefficients on the ``causing`` series in the ``caused`` equation are zero).
    F-statistic and p-value from a standard restricted-vs-unrestricted SSR
    comparison.
    """

    f_stat: float
    p_value: float
    df_num: int
    df_den: int
    caused: int
    causing: int
    lags: int


# ---------------------------------------------------------------------------
# VAR class
# ---------------------------------------------------------------------------


class VAR:
    """Vector autoregression VAR(p).

    Parameters
    ----------
    p : int
        Lag order (≥ 1).
    """

    def __init__(self, p: int) -> None:
        if p < 1:
            raise ValueError(f"VAR order p must be ≥ 1, got {p}.")
        self.p = p
        self._result: VARFitResult | None = None

    @property
    def result(self) -> VARFitResult:
        """Fitted result.  Raises ``RuntimeError`` before ``fit()``."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray, *, include_const: bool = True) -> VARFitResult:
        """Fit the VAR(p) model by equation-by-equation OLS.

        Parameters
        ----------
        y : array_like, shape (T, k)
            Multivariate observation matrix.  Each column is a variable.
        include_const : bool
            If True, include a per-equation intercept.
        """
        arr = to_numpy_2d(y)
        validate_finite(arr)
        validate_min_length(arr, self.p + 2, "y")
        T, k = arr.shape
        if k < 1:
            raise ValueError("y must have at least one column.")
        T_eff = T - self.p

        # Build design Z: shape (T_eff, k*p [+ 1])
        # Row t corresponds to time index t + p in arr.
        # Columns: [y_{t+p-1}, y_{t+p-2}, …, y_t]   (most-recent lag first)
        lag_blocks: list[np.ndarray] = []
        for ell in range(1, self.p + 1):
            lag_blocks.append(arr[self.p - ell : T - ell, :])
        Z = np.column_stack(lag_blocks)
        if include_const:
            Z = np.column_stack([Z, np.ones(T_eff)])
        Y = arr[self.p :, :]

        # OLS: B = (Z'Z)^-1 Z'Y   shape (kp [+1], k)
        ZtZ = Z.T @ Z
        try:
            B = np.linalg.solve(ZtZ, Z.T @ Y)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Design matrix Z'Z is singular; reduce lag order or check for collinearity in y."
            ) from exc

        # Unpack coefficients
        coef = np.empty((self.p, k, k))
        for ell in range(self.p):
            # Block ell occupies rows ell*k : (ell+1)*k, transposed to (k, k)
            coef[ell] = B[ell * k : (ell + 1) * k, :].T
        const = B[-1, :].copy() if include_const else np.zeros(k)

        fitted = Z @ B
        resid = Y - fitted
        # MLE (divisor T_eff) for likelihood consistency with statsmodels
        sigma_u = (resid.T @ resid) / T_eff

        # Log-likelihood: -T_eff/2 (k ln 2π + ln|Σ_u| + k)
        sign, logdet = np.linalg.slogdet(sigma_u)
        if sign <= 0:
            log_lik = -math.inf
        else:
            log_lik = -0.5 * T_eff * (k * math.log(2.0 * math.pi) + logdet + k)

        n_params = self.p * k * k + (k if include_const else 0)
        aic_v, bic_v, hqic_v = info_criteria(log_lik, T_eff, n_params)

        is_stat = self._check_stationary(coef)

        self._result = VARFitResult(
            order=self.p,
            k=k,
            coef=coef,
            const=const,
            sigma_u=sigma_u,
            residuals=resid,
            fitted_values=fitted,
            log_lik=float(log_lik),
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=T_eff,
            is_stationary=is_stat,
            include_const=include_const,
        )
        if not is_stat:
            warnings.warn(
                f"Fitted VAR({self.p}) is not stationary (a companion-matrix "
                "eigenvalue has modulus ≥ 1).",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._result

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, h: int) -> np.ndarray:
        """Deterministic h-step-ahead point forecasts, shape (h, k).

        Uses the recursion  ŷ_{T+s+1} = c + Σ_{ell=1}^p A_ell ŷ_{T+s+1-ell},
        with actual observations substituted in place of forecasts whenever
        s + 1 - ell ≤ 0.
        """
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")
        res = self.result
        k, p = res.k, res.order
        # Reconstruct last p actual observations.
        Y_actual = res.fitted_values + res.residuals  # shape (T_eff, k)
        # Rolling buffer: row -1 is the most recent observation, row 0 the oldest.
        buf = Y_actual[-p:, :].copy()
        out = np.empty((h, k))
        for s in range(h):
            y_hat = res.const.copy()
            for ell in range(p):
                # ell = 0 → lag 1 (buf[-1]); ell = p-1 → lag p (buf[0]).
                y_hat = y_hat + res.coef[ell] @ buf[p - 1 - ell, :]
            out[s] = y_hat
            buf = np.vstack([buf[1:], y_hat[None, :]])
        return out

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
        y0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate ``t_total`` observations from the fitted VAR(p)."""
        if t_total < 1:
            raise ValueError(f"t_total must be ≥ 1, got {t_total}.")
        res = self.result
        rng = np.random.default_rng(seed)
        k, p = res.k, res.order
        n = t_total + burnin
        L = np.linalg.cholesky(res.sigma_u + 1e-12 * np.eye(k))
        noise = rng.standard_normal((n, k)) @ L.T
        y = np.zeros((n + p, k))
        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float)
            if y0_arr.shape != (p, k):
                raise ValueError(f"y0 must have shape ({p}, {k}), got {y0_arr.shape}.")
            y[:p, :] = y0_arr
        for t in range(n):
            ti = p + t
            y_hat = res.const.copy()
            for ell in range(p):
                y_hat = y_hat + res.coef[ell] @ y[ti - 1 - ell, :]
            y[ti, :] = y_hat + noise[t]
        return y[p + burnin :, :]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_stationary(coef: np.ndarray) -> bool:
        """All companion-matrix eigenvalues strictly inside the unit disk."""
        p, k, _ = coef.shape
        kp = k * p
        comp = np.zeros((kp, kp))
        for ell in range(p):
            comp[:k, ell * k : (ell + 1) * k] = coef[ell]
        if p > 1:
            comp[k:, : (p - 1) * k] = np.eye((p - 1) * k)
        eigs = np.linalg.eigvals(comp)
        return bool(np.all(np.abs(eigs) < 1.0 - 1e-10))


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------


def granger_causality(
    y: np.ndarray,
    caused: int,
    causing: int,
    lags: int,
    *,
    include_const: bool = True,
) -> GrangerResult:
    """Granger-causality F-test.

    Tests whether past values of variable ``causing`` help predict variable
    ``caused`` over and above what past values of ``caused`` itself
    explain.  The unrestricted equation regresses y_{caused, t} on lags
    of both variables; the restricted equation excludes lags of
    ``causing``.

    Parameters
    ----------
    y             : array_like, shape (T, k)
    caused        : column index of the dependent variable
    causing       : column index of the candidate causal variable
    lags          : number of lags to include in each equation (≥ 1)
    include_const : include an intercept in both equations

    Returns
    -------
    GrangerResult
    """
    if lags < 1:
        raise ValueError(f"lags must be ≥ 1, got {lags}.")
    if caused == causing:
        raise ValueError("caused and causing must be different column indices.")
    arr = to_numpy_2d(y)
    validate_finite(arr)
    T, k = arr.shape
    if caused < 0 or caused >= k or causing < 0 or causing >= k:
        raise ValueError(f"caused/causing must be in [0, {k}).")
    validate_min_length(arr, lags + 2, "y")
    T_eff = T - lags

    y_dep = arr[lags:, caused]

    # Lag blocks for caused and causing variables
    lag_caused = np.column_stack([arr[lags - ell : T - ell, caused] for ell in range(1, lags + 1)])
    lag_causing = np.column_stack(
        [arr[lags - ell : T - ell, causing] for ell in range(1, lags + 1)]
    )

    if include_const:
        X_r = np.column_stack([lag_caused, np.ones(T_eff)])
        X_u = np.column_stack([lag_caused, lag_causing, np.ones(T_eff)])
    else:
        X_r = lag_caused
        X_u = np.column_stack([lag_caused, lag_causing])

    beta_r, *_ = np.linalg.lstsq(X_r, y_dep, rcond=None)
    beta_u, *_ = np.linalg.lstsq(X_u, y_dep, rcond=None)
    rss_r = float(np.sum((y_dep - X_r @ beta_r) ** 2))
    rss_u = float(np.sum((y_dep - X_u @ beta_u) ** 2))

    df_num = lags
    df_den = T_eff - X_u.shape[1]
    if df_den <= 0 or rss_u <= 0.0:
        raise ValueError("Granger test: insufficient degrees of freedom or zero unrestricted RSS.")
    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p_value = float(scipy.stats.f.sf(f_stat, df_num, df_den))

    return GrangerResult(
        f_stat=float(f_stat),
        p_value=p_value,
        df_num=df_num,
        df_den=df_den,
        caused=caused,
        causing=causing,
        lags=lags,
    )


# ---------------------------------------------------------------------------
# Impulse response functions
# ---------------------------------------------------------------------------


def impulse_response(
    result: VARFitResult,
    h: int,
    *,
    orthogonalized: bool = True,
) -> np.ndarray:
    """Impulse response functions over ``h`` horizons.

    The MA(∞) representation y_t = Σ_j Ψ_j u_{t-j} satisfies the recursion

        Ψ_0 = I,
        Ψ_j = Σ_{ell=1}^{min(j, p)} Ψ_{j-ell} A_ell        (j ≥ 1).

    When ``orthogonalized=True`` the shocks are first decorrelated via the
    Cholesky factor P with PP' = Σ_u (Sims 1980 ordering by column index of
    the input ``y``); the returned IRF[j] = Ψ_j @ P.

    Parameters
    ----------
    result         Fitted ``VARFitResult``.
    h              Number of horizons (≥ 1).  IRF has shape (h, k, k).
    orthogonalized If True, Cholesky-orthogonalised IRF.  Otherwise the raw
                   Ψ_j matrices.

    Returns
    -------
    np.ndarray, shape (h, k, k)
        ``out[t, i, j]`` = response of variable *i* at horizon *t* to a unit
        (or unit-stdev for orthogonalised) shock in variable *j*.
    """
    if h <= 0:
        raise ValueError(f"h must be ≥ 1, got {h}.")
    k, p = result.k, result.order

    psi = np.zeros((h, k, k))
    psi[0] = np.eye(k)
    for j in range(1, h):
        acc = np.zeros((k, k))
        for ell in range(1, min(j, p) + 1):
            acc += psi[j - ell] @ result.coef[ell - 1]
        psi[j] = acc

    if not orthogonalized:
        return psi

    # Cholesky of Σ_u (lower triangular).  Add tiny nugget for PSD safety.
    P = np.linalg.cholesky(result.sigma_u + 1e-12 * np.eye(k))
    out = np.empty_like(psi)
    for j in range(h):
        out[j] = psi[j] @ P
    return out
