"""
Univariate ARMA-family models: AR, MA, ARMA, ARIMA, SARIMA.

State-space representation (Hamilton 1994)
------------------------------------------
All ARMA(p, q) models are fitted via the same linear Kalman filter
with the companion-matrix state-space form:

    x_t = F x_{t-1} + g ε_t,    ε_t ~ N(0, σ²)
    y_t = H x_t                  (no measurement noise)

where  r = max(p, q+1),  the state dimension, and

    F[0, :p] = [φ_1, …, φ_p],   F[i, i-1] = 1  (companion)
    H = [1, 0, …, 0]                              (1 × r)
    g = [1, θ_1, …, θ_q, 0, …, 0]               (r × 1)
    Q = σ² g gᵀ

Initial state covariance P₀ is the solution of the discrete Lyapunov
equation P = F P Fᵀ + Q via ``scipy.linalg.solve_discrete_lyapunov``.

Fitting methods
---------------
AR  : yule_walker  — Durbin-Levinson recursion on sample autocovariances
      ols          — OLS on the lag-design matrix
      mle          — state-space log-likelihood + L-BFGS-B
MA  : css          — conditional sum of squares (pre-sample ε = 0)
      mle          — state-space log-likelihood + L-BFGS-B
ARMA: css          — conditional sum of squares
      mle          — state-space log-likelihood + L-BFGS-B

ARIMA wraps ARMA with ``utils.difference``.
SARIMA expands seasonal polynomials then wraps ARMA.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.linalg
import scipy.optimize
import scipy.stats

from ._io import to_numpy_1d, validate_finite, validate_min_length
from ._kernels import durbin_levinson, lag_matrix, sample_acovf
from ._types import ForecastResult
from .utils import difference, info_criteria, seasonal_difference

# ---------------------------------------------------------------------------
# Private helpers — state-space construction
# ---------------------------------------------------------------------------


def _build_companion(phi: np.ndarray, r: int) -> np.ndarray:
    """Build r×r Hamilton (1994) state-space transition matrix.

    AR coefficients φ_1, …, φ_p go in the **first column** (rows 0 to p−1).
    Ones go on the **superdiagonal** (position [i, i+1] for i = 0, …, r−2).

    This gives the standard Hamilton (1994, pp. 374-375) companion form:

        F = [[φ₁, 1, 0, …, 0],
             [φ₂, 0, 1, …, 0],
             [⋮            ⋮ ],
             [φ_p, 0, 0, …, 1],
             [0,   0, 0, …, 0]]

    Under this form the first state ξ_{t,0} = y_t (the observation), and
    the MA contributions enter via the g-vector in the Q = σ² g gᵀ covariance.
    """
    f = np.zeros((r, r))
    p = len(phi)
    if p:
        f[:p, 0] = phi  # AR coefficients in first column
    for i in range(r - 1):
        f[i, i + 1] = 1.0  # ones on superdiagonal
    return f


def _arma_state_space(
    phi: np.ndarray, theta: np.ndarray, sigma2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build Hamilton state-space matrices for ARMA(p, q).

    Returns
    -------
    F   : (r, r) state transition (companion)
    H   : (1, r) observation
    Q   : (r, r) process noise covariance  =  σ² g gᵀ
    R   : (1, 1) measurement noise (tiny nugget for numerical stability)
    r   : state dimension
    """
    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    f_mat = _build_companion(phi, r)

    h_mat = np.zeros((1, r))
    h_mat[0, 0] = 1.0

    g = np.zeros(r)
    g[0] = 1.0
    if q:
        g[1 : q + 1] = theta

    q_mat = sigma2 * np.outer(g, g)
    r_mat = np.array([[max(sigma2, 1.0) * 1e-10]])  # tiny positive nugget

    return f_mat, h_mat, q_mat, r_mat, r


def _p0_lyapunov(f_mat: np.ndarray, q_mat: np.ndarray, r: int) -> np.ndarray:
    """Stationary initial covariance from discrete Lyapunov equation."""
    try:
        p0 = scipy.linalg.solve_discrete_lyapunov(f_mat, q_mat)
        p0 = 0.5 * (p0 + p0.T)
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(p0)
        if np.any(eigvals < 0):
            p0 += (-eigvals.min() + 1e-8) * np.eye(r)
    except Exception:
        p0 = np.eye(r) * 1e6  # diffuse fallback
    return p0


def _arma_log_likelihood(phi: np.ndarray, theta: np.ndarray, sigma2: float, y: np.ndarray) -> float:
    """Exact log-likelihood for ARMA(p, q) via KalmanFilter."""
    from ..filters.kalman import KalmanFilter  # lazy import — avoids circular dep

    f_mat, h_mat, q_mat, r_mat, r = _arma_state_space(phi, theta, sigma2)
    p0 = _p0_lyapunov(f_mat, q_mat, r)
    x0 = np.zeros(r)

    kf = KalmanFilter(F=f_mat, H=h_mat, Q=q_mat, R=r_mat, x0=x0, P0=p0)
    return kf.log_likelihood(y.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Private helpers — polynomial tools
# ---------------------------------------------------------------------------


def _pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
    """
    Convert partial autocorrelations to AR coefficients via forward
    Levinson-Durbin recursion.  All |pacf[k]| < 1 guarantees stationarity.
    """
    p = len(pacf)
    if p == 0:
        return np.zeros(0)
    phi = np.zeros((p, p))
    phi[0, 0] = pacf[0]
    for m in range(1, p):
        phi_mm = pacf[m]
        for j in range(m):
            phi[m, j] = phi[m - 1, j] - phi_mm * phi[m - 1, m - 1 - j]
        phi[m, m] = phi_mm
    return phi[p - 1, :].copy()


def _is_stationary(phi: np.ndarray) -> bool:
    """True iff the AR(p) companion-matrix eigenvalues are inside the unit disk.

    Equivalently, all roots of the lag polynomial
    Φ(z) = 1 − φ₁z − … − φ_p zᵖ lie outside the unit circle.
    ``np.roots([1, −φ_1, …, −φ_p])`` returns the companion eigenvalues
    (reciprocals of the lag-polynomial roots); stationarity requires
    all eigenvalues to have modulus strictly less than 1.
    """
    if len(phi) == 0:
        return True
    coeffs = np.concatenate([[1.0], -phi])
    return bool(np.all(np.abs(np.roots(coeffs)) < 1.0))


def _is_invertible(theta: np.ndarray) -> bool:
    """True iff the MA polynomial roots lie outside the unit circle.

    ``np.roots([1, θ_1, …, θ_q])`` returns the reciprocals of the MA
    polynomial roots; invertibility requires all of those to have modulus
    strictly less than 1.
    """
    if len(theta) == 0:
        return True
    coeffs = np.concatenate([[1.0], theta])
    return bool(np.all(np.abs(np.roots(coeffs)) < 1.0))


def _poly_mult_ar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two AR lag-polynomial coefficient arrays.

    Given AR(p) coefs *a* = [φ₁,…,φ_p] and AR(P) coefs *b* = [Φ₁,…,Φ_P],
    both expressed in the ``1 − a₁L − …`` convention, return the combined
    AR coefficient array of length p + len(b).

    Uses ``np.polymul`` on the sign-flipped representations.
    """
    pa = np.concatenate([[1.0], -np.asarray(a, dtype=float)])
    pb = np.concatenate([[1.0], -np.asarray(b, dtype=float)])
    prod = np.polymul(pa, pb)
    return -prod[1:]  # strip the leading 1, negate


def _expand_seasonal(coef: np.ndarray, s: int) -> np.ndarray:
    """
    Expand seasonal AR/MA coefficients [Φ₁,…,Φ_P] at period s into a
    standard lag-polynomial array of length P*s.

    Φ(Lˢ) = 1 − Φ₁Lˢ − Φ₂L^{2s} − … → coef array indexed by lag.
    """
    p_s = len(coef)
    result = np.zeros(p_s * s)
    for k in range(p_s):
        result[(k + 1) * s - 1] = coef[k]
    return result


# ---------------------------------------------------------------------------
# Private helpers — residuals and CSS
# ---------------------------------------------------------------------------


def _css_residuals(phi: np.ndarray, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute ARMA(p, q) residuals via the conditional sum-of-squares recursion.

    Pre-sample observations and innovations are treated as zero.

    Parameters
    ----------
    phi   : AR coefficients φ_1, …, φ_p
    theta : MA coefficients θ_1, …, θ_q
    y     : demeaned observation vector

    Returns
    -------
    eps : residuals, same length as y
    """
    p = len(phi)
    q = len(theta)
    t_total = len(y)
    eps = np.zeros(t_total)

    for t in range(t_total):
        e = y[t]
        for j in range(min(t, p)):
            e -= phi[j] * y[t - 1 - j]
        for j in range(min(t, q)):
            e -= theta[j] * eps[t - 1 - j]
        eps[t] = e

    return eps


# ---------------------------------------------------------------------------
# Private helpers — impulse-response and forecasting
# ---------------------------------------------------------------------------


def _arma_impulse_response(phi: np.ndarray, theta: np.ndarray, h: int) -> np.ndarray:
    """
    MA(∞) representation coefficients ψ_0, …, ψ_{h-1} of the ARMA model.

    ψ_j = φ_1 ψ_{j-1} + … + φ_p ψ_{j-p} + θ_j   (θ_j = 0 for j > q, ψ_j = 0 for j < 0)
    ψ_0 = 1
    """
    p = len(phi)
    q = len(theta)
    psi = np.zeros(h)
    if h == 0:
        return psi
    psi[0] = 1.0
    for j in range(1, h):
        for k in range(1, min(j, p) + 1):
            psi[j] += phi[k - 1] * psi[j - k]
        if 1 <= j <= q:
            psi[j] += theta[j - 1]
    return psi


def _arma_forecast_mean(
    phi: np.ndarray,
    theta: np.ndarray,
    last_y: np.ndarray,
    last_eps: np.ndarray,
    h: int,
    const: float,
) -> np.ndarray:
    """
    Compute h-step-ahead point forecasts for an ARMA model.

    Parameters
    ----------
    phi      : AR coefficients
    theta    : MA coefficients
    last_y   : recent observations (length ≥ p), most-recent last
    last_eps : recent residuals (length ≥ q), most-recent last
    h        : forecast horizon
    const    : unconditional mean (added back to each forecast)
    """
    p = len(phi)
    q = len(theta)
    # Extend observed history with forecasts
    extended_y = np.concatenate([last_y, np.zeros(h)])
    extended_eps = np.concatenate([last_eps, np.zeros(h)])  # future ε_t → 0

    t_start = len(last_y)
    for i in range(h):
        t = t_start + i
        yhat = 0.0
        for k in range(min(t, p)):
            yhat += phi[k] * extended_y[t - 1 - k]
        for k in range(min(t, q)):
            yhat += theta[k] * extended_eps[t - 1 - k]
        extended_y[t] = yhat

    return extended_y[t_start:] + const


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ARFitResult:
    """Fitted AR(p) model output."""

    method: str  # 'yule_walker' | 'ols' | 'mle'
    order: int  # p
    coef: np.ndarray  # AR coefficients φ_1, …, φ_p  shape (p,)
    const: float  # unconditional mean μ
    sigma2: float  # innovation variance
    residuals: np.ndarray  # shape (n_obs,)
    fitted_values: np.ndarray  # shape (n_obs,)
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    is_stationary: bool

    def __str__(self) -> str:
        lines = [
            f"AR({self.order})  method={self.method}  n_obs={self.n_obs}",
            f"  const   = {self.const:.6g}",
        ]
        for i, c in enumerate(self.coef, 1):
            lines.append(f"  phi[{i}]  = {c:.6g}")
        lines += [
            f"  sigma²  = {self.sigma2:.6g}",
            f"  log_lik = {self.log_lik:.4f}",
            f"  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  stationary={self.is_stationary}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Return a DataFrame with parameter names and values."""
        names = ["const"] + [f"phi_{i}" for i in range(1, self.order + 1)] + ["sigma2"]
        values = [self.const, *self.coef.tolist(), self.sigma2]
        return pl.DataFrame({"parameter": names, "value": values})


@dataclass
class MAFitResult:
    """Fitted MA(q) model output."""

    method: str  # 'css' | 'mle'
    order: int  # q
    coef: np.ndarray  # MA coefficients θ_1, …, θ_q  shape (q,)
    const: float  # unconditional mean μ
    sigma2: float  # innovation variance
    residuals: np.ndarray  # shape (n_obs,)
    fitted_values: np.ndarray  # shape (n_obs,)
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    is_invertible: bool

    def __str__(self) -> str:
        lines = [
            f"MA({self.order})  method={self.method}  n_obs={self.n_obs}",
            f"  const   = {self.const:.6g}",
        ]
        for i, c in enumerate(self.coef, 1):
            lines.append(f"  theta[{i}] = {c:.6g}")
        lines += [
            f"  sigma²  = {self.sigma2:.6g}",
            f"  log_lik = {self.log_lik:.4f}",
            f"  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  invertible={self.is_invertible}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        names = ["const"] + [f"theta_{i}" for i in range(1, self.order + 1)] + ["sigma2"]
        values = [self.const, *self.coef.tolist(), self.sigma2]
        return pl.DataFrame({"parameter": names, "value": values})


@dataclass
class ARMAFitResult:
    """Fitted ARMA(p, q) model output."""

    method: str  # 'css' | 'mle'
    ar_order: int  # p
    ma_order: int  # q
    ar_coef: np.ndarray  # φ_1, …, φ_p  shape (p,)
    ma_coef: np.ndarray  # θ_1, …, θ_q  shape (q,)
    const: float  # unconditional mean μ
    sigma2: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    is_stationary: bool
    is_invertible: bool

    def __str__(self) -> str:
        lines = [
            f"ARMA({self.ar_order},{self.ma_order})  method={self.method}  n_obs={self.n_obs}",
            f"  const   = {self.const:.6g}",
        ]
        for i, c in enumerate(self.ar_coef, 1):
            lines.append(f"  phi[{i}]   = {c:.6g}")
        for i, c in enumerate(self.ma_coef, 1):
            lines.append(f"  theta[{i}]  = {c:.6g}")
        lines += [
            f"  sigma²   = {self.sigma2:.6g}",
            f"  log_lik  = {self.log_lik:.4f}",
            f"  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  stationary={self.is_stationary}  invertible={self.is_invertible}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        names = (
            ["const"]
            + [f"phi_{i}" for i in range(1, self.ar_order + 1)]
            + [f"theta_{i}" for i in range(1, self.ma_order + 1)]
            + ["sigma2"]
        )
        values = [self.const, *self.ar_coef.tolist(), *self.ma_coef.tolist(), self.sigma2]
        return pl.DataFrame({"parameter": names, "value": values})


@dataclass
class ARIMAFitResult:
    """Fitted ARIMA(p, d, q) model output."""

    method: str
    ar_order: int
    diff_order: int  # d
    ma_order: int
    ar_coef: np.ndarray
    ma_coef: np.ndarray
    const: float  # drift in the differenced series
    sigma2: float
    residuals: np.ndarray  # on the differenced series
    fitted_values: np.ndarray  # on the differenced series
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    is_stationary: bool
    is_invertible: bool

    def __str__(self) -> str:
        lines = [
            f"ARIMA({self.ar_order},{self.diff_order},{self.ma_order})"
            f"  method={self.method}  n_obs={self.n_obs}",
            f"  drift   = {self.const:.6g}",
        ]
        for i, c in enumerate(self.ar_coef, 1):
            lines.append(f"  phi[{i}]   = {c:.6g}")
        for i, c in enumerate(self.ma_coef, 1):
            lines.append(f"  theta[{i}]  = {c:.6g}")
        lines += [
            f"  sigma²   = {self.sigma2:.6g}",
            f"  log_lik  = {self.log_lik:.4f}",
            f"  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        names = (
            ["drift"]
            + [f"phi_{i}" for i in range(1, self.ar_order + 1)]
            + [f"theta_{i}" for i in range(1, self.ma_order + 1)]
            + ["sigma2"]
        )
        values = [self.const, *self.ar_coef.tolist(), *self.ma_coef.tolist(), self.sigma2]
        return pl.DataFrame({"parameter": names, "value": values})


@dataclass
class SARIMAFitResult:
    """Fitted SARIMA(p,d,q)(P,D,Q,s) model output."""

    method: str
    ar_order: int
    diff_order: int
    ma_order: int
    seasonal_ar_order: int  # P
    seasonal_diff_order: int  # D
    seasonal_ma_order: int  # Q
    period: int  # s
    ar_coef: np.ndarray  # structural: φ_1,…,φ_p
    ma_coef: np.ndarray  # structural: θ_1,…,θ_q
    seasonal_ar_coef: np.ndarray  # structural: Φ_1,…,Φ_P
    seasonal_ma_coef: np.ndarray  # structural: Θ_1,…,Θ_Q
    ar_coef_expanded: np.ndarray  # product polynomial
    ma_coef_expanded: np.ndarray  # product polynomial
    const: float
    sigma2: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int

    def __str__(self) -> str:
        return (
            f"SARIMA({self.ar_order},{self.diff_order},{self.ma_order})"
            f"({self.seasonal_ar_order},{self.seasonal_diff_order},{self.seasonal_ma_order})"
            f"[{self.period}]  method={self.method}  n_obs={self.n_obs}\n"
            f"  drift={self.const:.6g}  sigma²={self.sigma2:.6g}\n"
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        names = (
            ["drift"]
            + [f"phi_{i}" for i in range(1, self.ar_order + 1)]
            + [f"Phi_{i}" for i in range(1, self.seasonal_ar_order + 1)]
            + [f"theta_{i}" for i in range(1, self.ma_order + 1)]
            + [f"Theta_{i}" for i in range(1, self.seasonal_ma_order + 1)]
            + ["sigma2"]
        )
        values = (
            [self.const]
            + self.ar_coef.tolist()
            + self.seasonal_ar_coef.tolist()
            + self.ma_coef.tolist()
            + self.seasonal_ma_coef.tolist()
            + [self.sigma2]
        )
        return pl.DataFrame({"parameter": names, "value": values})


# ---------------------------------------------------------------------------
# AR model
# ---------------------------------------------------------------------------


class AR:
    """
    Autoregressive model AR(p).

    Parameters
    ----------
    p : int
        Lag order (≥ 1).
    """

    def __init__(self, p: int) -> None:
        if p < 1:
            raise ValueError(f"AR order p must be ≥ 1, got {p}.")
        self.p = p
        self._result: ARFitResult | None = None

    @property
    def result(self) -> ARFitResult:
        """Fitted model result.  Raises ``RuntimeError`` before ``fit()``."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        method: str = "yule_walker",
        *,
        include_const: bool = True,
    ) -> ARFitResult:
        """
        Fit the AR(p) model.

        Parameters
        ----------
        x : array_like, shape (T,)
            Observation sequence.  Accepts ``np.ndarray`` or ``pl.Series``.
        method : {'yule_walker', 'ols', 'mle'}
            Estimation method.
        include_const : bool
            If True, the unconditional mean is estimated from the sample and
            subtracted before fitting.

        Returns
        -------
        ARFitResult
        """
        arr = to_numpy_1d(x)
        validate_finite(arr)
        validate_min_length(arr, self.p + 2, "x")

        const = float(np.mean(arr)) if include_const else 0.0
        y = arr - const

        match method:
            case "yule_walker":
                phi, sigma2 = self._fit_yw(y)
            case "ols":
                phi, sigma2 = self._fit_ols(y)
            case "mle":
                phi, sigma2 = self._fit_mle(y)
            case _:
                raise ValueError(
                    f"Unknown method {method!r}; choose 'yule_walker', 'ols', or 'mle'."
                )

        # Residuals and fitted values on the demeaned series
        x_design, y_target = lag_matrix(y, self.p)
        phi_np = np.asarray(phi)
        fitted_dm = x_design @ phi_np  # (T-p,)
        residuals = y_target - fitted_dm

        # Overwrite sigma2 from residuals for YW/OLS (more accurate)
        if method in ("yule_walker", "ols"):
            sigma2 = float(np.var(residuals, ddof=self.p + 1))

        # Log-likelihood (use ARMA SS for consistency)
        n_eff = len(y_target)
        ll = _arma_log_likelihood(phi_np, np.zeros(0), sigma2, y_target)
        aic_v, bic_v, hqic_v = info_criteria(ll, n_eff, self.p + 1)

        stat = _is_stationary(phi_np)
        if not stat:
            warnings.warn(f"Fitted AR({self.p}) model is not stationary.", stacklevel=2)

        self._result = ARFitResult(
            method=method,
            order=self.p,
            coef=phi_np,
            const=const,
            sigma2=sigma2,
            residuals=residuals,
            fitted_values=fitted_dm + const,
            log_lik=ll,
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=n_eff,
            is_stationary=stat,
        )
        return self._result

    def _fit_yw(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Yule-Walker via Durbin-Levinson recursion."""
        acovs = sample_acovf(y, self.p)
        ar, _, sigma2_arr = durbin_levinson(acovs)
        sigma2 = float(sigma2_arr[-1])
        return ar, sigma2

    def _fit_ols(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """OLS on the lag-design matrix."""
        x_design, y_target = lag_matrix(y, self.p)
        phi, *_ = np.linalg.lstsq(x_design, y_target, rcond=None)
        resid = y_target - x_design @ phi
        sigma2 = float(np.var(resid, ddof=self.p))
        return phi, sigma2

    def _fit_mle(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """State-space MLE via L-BFGS-B."""
        # Warm-start from OLS
        phi0, s2_0 = self._fit_ols(y)
        s2_0 = max(s2_0, 1e-8)
        params0 = np.concatenate([phi0, [np.log(s2_0)]])

        def neg_ll(params: np.ndarray) -> float:
            phi = params[: self.p]
            sigma2 = np.exp(params[self.p])
            return -_arma_log_likelihood(phi, np.zeros(0), sigma2, y)

        res = scipy.optimize.minimize(
            neg_ll,
            params0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        phi = res.x[: self.p]
        sigma2 = float(np.exp(res.x[self.p]))
        return phi, sigma2

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        *,
        alpha: float | None = 0.05,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        """
        h-step-ahead forecasts with optional Gaussian prediction intervals.

        Parameters
        ----------
        h     : forecast horizon
        alpha : two-sided coverage level (None → no intervals)
        n_paths : if given, also return Monte Carlo sample paths
        seed  : RNG seed for Monte Carlo
        """
        res = self.result
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")

        # Reconstruct last_p demeaned observations
        last_p = res.fitted_values[-self.p :] - res.const + res.residuals[-self.p :]
        last_eps = res.residuals[-max(1, self.p) :]

        means = _arma_forecast_mean(res.coef, np.zeros(0), last_p, last_eps, h, res.const)

        lower = upper = paths = None
        if alpha is not None:
            psi = _arma_impulse_response(res.coef, np.zeros(0), h)
            var_h = res.sigma2 * np.cumsum(psi**2)
            z = float(scipy.stats.norm.ppf(1.0 - alpha / 2))
            half = z * np.sqrt(var_h)
            lower = means - half
            upper = means + half

        if n_paths is not None:
            paths = self._simulate_paths(h, n_paths, seed, res)

        return ForecastResult(
            mean=means,
            horizon=h,
            alpha=alpha,
            lower=lower,
            upper=upper,
            paths=paths,
        )

    def _simulate_paths(
        self, h: int, n_paths: int, seed: int | None, res: ARFitResult
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        # Last p demeaned values as initial conditions
        init = res.fitted_values[-self.p :] - res.const + res.residuals[-self.p :]

        all_paths = np.empty((n_paths, h))
        for path_idx in range(n_paths):
            y_buf = np.concatenate([init, np.zeros(h)])
            eps = rng.standard_normal(h) * sigma
            for t in range(h):
                ti = self.p + t
                y_buf[ti] = res.const + eps[t]
                for k in range(self.p):
                    y_buf[ti] += res.coef[k] * (y_buf[ti - 1 - k] - res.const)
            all_paths[path_idx] = y_buf[self.p :]
        return all_paths

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> np.ndarray:
        """
        Simulate ``t_total`` observations from the fitted AR(p) model.

        Parameters
        ----------
        t_total : length of the returned series
        seed    : RNG seed
        burnin  : warm-up steps discarded from the beginning
        """
        res = self.result
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        n = t_total + burnin
        eps = rng.standard_normal(n) * sigma
        y = np.zeros(n)
        for t in range(n):
            y[t] = res.const + eps[t]
            for k in range(min(t, self.p)):
                y[t] += res.coef[k] * (y[t - 1 - k] - res.const)
        return y[burnin:]


# ---------------------------------------------------------------------------
# MA model
# ---------------------------------------------------------------------------


class MA:
    """
    Moving-average model MA(q).

    Parameters
    ----------
    q : int
        Lag order (≥ 1).
    """

    def __init__(self, q: int) -> None:
        if q < 1:
            raise ValueError(f"MA order q must be ≥ 1, got {q}.")
        self.q = q
        self._result: MAFitResult | None = None

    @property
    def result(self) -> MAFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        method: str = "css",
        *,
        include_const: bool = True,
    ) -> MAFitResult:
        """
        Fit the MA(q) model.

        Parameters
        ----------
        x : array_like, shape (T,)
        method : {'css', 'mle'}
            'css'  — conditional sum of squares (pre-sample ε = 0, fast)
            'mle'  — exact MLE via Kalman filter
        include_const : bool
        """
        arr = to_numpy_1d(x)
        validate_finite(arr)
        validate_min_length(arr, self.q + 2, "x")

        const = float(np.mean(arr)) if include_const else 0.0
        y = arr - const

        match method:
            case "css":
                theta, sigma2 = self._fit_css(y)
            case "mle":
                theta, sigma2 = self._fit_mle(y)
            case _:
                raise ValueError(f"Unknown method {method!r}; choose 'css' or 'mle'.")

        theta_np = np.asarray(theta)
        residuals = _css_residuals(np.zeros(0), theta_np, y)
        if method == "css":
            sigma2 = float(np.var(residuals, ddof=self.q))

        fitted_dm = y - residuals  # ŷ = y - ε
        n_eff = len(y)
        ll = _arma_log_likelihood(np.zeros(0), theta_np, sigma2, y)
        aic_v, bic_v, hqic_v = info_criteria(ll, n_eff, self.q + 1)

        inv = _is_invertible(theta_np)
        if not inv:
            warnings.warn(f"Fitted MA({self.q}) model is not invertible.", stacklevel=2)

        self._result = MAFitResult(
            method=method,
            order=self.q,
            coef=theta_np,
            const=const,
            sigma2=sigma2,
            residuals=residuals,
            fitted_values=fitted_dm + const,
            log_lik=ll,
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=n_eff,
            is_invertible=inv,
        )
        return self._result

    def _fit_css(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """CSS objective minimised over theta."""
        theta0 = np.zeros(self.q)

        def obj(theta: np.ndarray) -> float:
            eps = _css_residuals(np.zeros(0), theta, y)
            return float(np.sum(eps**2))

        res = scipy.optimize.minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-12},
        )
        theta = res.x
        sigma2 = float(np.var(_css_residuals(np.zeros(0), theta, y), ddof=self.q))
        return theta, sigma2

    def _fit_mle(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """State-space MLE."""
        # Warm-start from CSS
        theta0, s2_0 = self._fit_css(y)
        s2_0 = max(s2_0, 1e-8)
        params0 = np.concatenate([theta0, [np.log(s2_0)]])

        def neg_ll(params: np.ndarray) -> float:
            theta = params[: self.q]
            sigma2 = np.exp(params[self.q])
            return -_arma_log_likelihood(np.zeros(0), theta, sigma2, y)

        res = scipy.optimize.minimize(
            neg_ll,
            params0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        theta = res.x[: self.q]
        sigma2 = float(np.exp(res.x[self.q]))
        return theta, sigma2

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        *,
        alpha: float | None = 0.05,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        """h-step-ahead forecasts.  For MA(q), ŷ_{T+h} = 0 for h > q."""
        res = self.result
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")

        last_eps = np.zeros(self.q)
        recent = res.residuals[-self.q :]
        last_eps[: len(recent)] = recent

        means = _arma_forecast_mean(np.zeros(0), res.coef, np.zeros(0), last_eps, h, res.const)

        lower = upper = paths = None
        if alpha is not None:
            psi = _arma_impulse_response(np.zeros(0), res.coef, h)
            var_h = res.sigma2 * np.cumsum(psi**2)
            z = float(scipy.stats.norm.ppf(1.0 - alpha / 2))
            half = z * np.sqrt(var_h)
            lower = means - half
            upper = means + half

        if n_paths is not None:
            rng = np.random.default_rng(seed)
            sigma = np.sqrt(res.sigma2)
            paths_arr = np.empty((n_paths, h))
            for path_idx in range(n_paths):
                eps_buf = np.concatenate([last_eps, rng.standard_normal(h) * sigma])
                y_path = np.zeros(h)
                for t in range(h):
                    yt = res.const + eps_buf[self.q + t]
                    for k in range(self.q):
                        yt += res.coef[k] * eps_buf[self.q + t - 1 - k]
                    y_path[t] = yt
                paths_arr[path_idx] = y_path
            paths = paths_arr

        return ForecastResult(
            mean=means, horizon=h, alpha=alpha, lower=lower, upper=upper, paths=paths
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> np.ndarray:
        """Simulate ``t_total`` observations from the fitted MA(q) model."""
        res = self.result
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        n = t_total + burnin
        eps = rng.standard_normal(n) * sigma
        y = np.zeros(n)
        for t in range(n):
            y[t] = res.const + eps[t]
            for k in range(min(t, self.q)):
                y[t] += res.coef[k] * eps[t - 1 - k]
        return y[burnin:]


# ---------------------------------------------------------------------------
# ARMA model
# ---------------------------------------------------------------------------


class ARMA:
    """
    Autoregressive moving-average model ARMA(p, q).

    Parameters
    ----------
    p : int   AR order (≥ 0)
    q : int   MA order (≥ 0)
    """

    def __init__(self, p: int, q: int) -> None:
        if p < 0:
            raise ValueError(f"AR order p must be ≥ 0, got {p}.")
        if q < 0:
            raise ValueError(f"MA order q must be ≥ 0, got {q}.")
        if p + q == 0:
            raise ValueError("At least one of p or q must be > 0.")
        self.p = p
        self.q = q
        self._result: ARMAFitResult | None = None

    @property
    def result(self) -> ARMAFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        method: str = "mle",
        *,
        include_const: bool = True,
    ) -> ARMAFitResult:
        """
        Fit the ARMA(p, q) model.

        Parameters
        ----------
        x : array_like, shape (T,)
        method : {'css', 'mle'}
        include_const : bool
        """
        arr = to_numpy_1d(x)
        validate_finite(arr)
        min_len = max(self.p, self.q) + 2
        validate_min_length(arr, min_len, "x")

        const = float(np.mean(arr)) if include_const else 0.0
        y = arr - const

        match method:
            case "css":
                phi, theta, sigma2 = self._fit_css(y)
            case "mle":
                phi, theta, sigma2 = self._fit_mle(y)
            case _:
                raise ValueError(f"Unknown method {method!r}; choose 'css' or 'mle'.")

        phi_np = np.asarray(phi)
        theta_np = np.asarray(theta)
        residuals = _css_residuals(phi_np, theta_np, y)
        if method == "css":
            sigma2 = float(np.var(residuals, ddof=self.p + self.q))

        fitted_dm = y - residuals
        n_eff = len(y)
        ll = _arma_log_likelihood(phi_np, theta_np, sigma2, y)
        n_params = self.p + self.q + 1 + (1 if include_const else 0)
        aic_v, bic_v, hqic_v = info_criteria(ll, n_eff, n_params)

        stat = _is_stationary(phi_np)
        inv = _is_invertible(theta_np)
        if not stat:
            warnings.warn(
                f"Fitted ARMA({self.p},{self.q}) AR part is not stationary.", stacklevel=2
            )
        if not inv:
            warnings.warn(
                f"Fitted ARMA({self.p},{self.q}) MA part is not invertible.", stacklevel=2
            )

        self._result = ARMAFitResult(
            method=method,
            ar_order=self.p,
            ma_order=self.q,
            ar_coef=phi_np,
            ma_coef=theta_np,
            const=const,
            sigma2=sigma2,
            residuals=residuals,
            fitted_values=fitted_dm + const,
            log_lik=ll,
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=n_eff,
            is_stationary=stat,
            is_invertible=inv,
        )
        return self._result

    def _fit_css(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """CSS: minimise sum of squared residuals."""
        params0 = np.zeros(self.p + self.q)

        def obj(params: np.ndarray) -> float:
            phi = params[: self.p]
            theta = params[self.p :]
            eps = _css_residuals(phi, theta, y)
            return float(np.sum(eps**2))

        res = scipy.optimize.minimize(
            obj,
            params0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        phi = res.x[: self.p]
        theta = res.x[self.p :]
        eps = _css_residuals(phi, theta, y)
        sigma2 = float(np.var(eps, ddof=self.p + self.q))
        return phi, theta, sigma2

    def _fit_mle(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """State-space MLE via L-BFGS-B, warm-started from CSS."""
        phi0, theta0, s2_0 = self._fit_css(y)
        s2_0 = max(s2_0, 1e-8)
        params0 = np.concatenate([phi0, theta0, [np.log(s2_0)]])

        def neg_ll(params: np.ndarray) -> float:
            phi = params[: self.p]
            theta = params[self.p : self.p + self.q]
            sigma2 = np.exp(params[-1])
            try:
                return -_arma_log_likelihood(phi, theta, sigma2, y)
            except Exception:
                return 1e10

        res = scipy.optimize.minimize(
            neg_ll,
            params0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        phi = res.x[: self.p]
        theta = res.x[self.p : self.p + self.q]
        sigma2 = float(np.exp(res.x[-1]))
        return phi, theta, sigma2

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        *,
        alpha: float | None = 0.05,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        """h-step-ahead forecasts with analytic Gaussian prediction intervals."""
        res = self.result
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")

        # Last p demeaned observations and last q residuals
        y_dm = res.fitted_values - res.const + res.residuals  # approximate demeaned history
        last_p = y_dm[-max(self.p, 1) :]
        last_eps = res.residuals[-max(self.q, 1) :]

        means = _arma_forecast_mean(res.ar_coef, res.ma_coef, last_p, last_eps, h, res.const)

        lower = upper = paths = None
        if alpha is not None:
            psi = _arma_impulse_response(res.ar_coef, res.ma_coef, h)
            var_h = res.sigma2 * np.cumsum(psi**2)
            z = float(scipy.stats.norm.ppf(1.0 - alpha / 2))
            half = z * np.sqrt(var_h)
            lower = means - half
            upper = means + half

        if n_paths is not None:
            paths = self._simulate_paths(h, n_paths, seed, res, last_p, last_eps)

        return ForecastResult(
            mean=means, horizon=h, alpha=alpha, lower=lower, upper=upper, paths=paths
        )

    def _simulate_paths(
        self,
        h: int,
        n_paths: int,
        seed: int | None,
        res: ARMAFitResult,
        last_p: np.ndarray,
        last_eps: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        all_paths = np.empty((n_paths, h))
        for path_idx in range(n_paths):
            y_buf = np.concatenate([last_p, np.zeros(h)])
            eps_buf = np.concatenate([last_eps, rng.standard_normal(h) * sigma])
            for t in range(h):
                ti = len(last_p) + t
                ei = len(last_eps) + t
                yhat = res.const
                for k in range(self.p):
                    yhat += res.ar_coef[k] * (y_buf[ti - 1 - k] - res.const)
                for k in range(self.q):
                    yhat += res.ma_coef[k] * eps_buf[ei - 1 - k]
                yhat += eps_buf[ei]
                y_buf[ti] = yhat
            all_paths[path_idx] = y_buf[len(last_p) :]
        return all_paths

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> np.ndarray:
        """Simulate ``t_total`` observations from the fitted ARMA model."""
        res = self.result
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        n = t_total + burnin
        eps = rng.standard_normal(n) * sigma
        y = np.zeros(n)
        for t in range(n):
            y[t] = res.const + eps[t]
            for k in range(min(t, self.p)):
                y[t] += res.ar_coef[k] * (y[t - 1 - k] - res.const)
            for k in range(min(t, self.q)):
                y[t] += res.ma_coef[k] * eps[t - 1 - k]
        return y[burnin:]


# ---------------------------------------------------------------------------
# ARIMA model
# ---------------------------------------------------------------------------


class ARIMA:
    """
    Autoregressive integrated moving-average model ARIMA(p, d, q).

    Internally differences the series ``d`` times, then delegates to
    ``ARMA(p, q)`` for estimation.

    Parameters
    ----------
    p : int   AR order
    d : int   Integration order (differences)
    q : int   MA order
    """

    def __init__(self, p: int, d: int, q: int) -> None:
        if p < 0 or d < 0 or q < 0:
            raise ValueError("p, d, q must all be ≥ 0.")
        if p + q == 0:
            raise ValueError("At least one of p or q must be > 0.")
        self.p = p
        self.d = d
        self.q = q
        self._arma = ARMA(p, q)
        self._result: ARIMAFitResult | None = None
        self._seed_obs: np.ndarray | None = None  # needed for forecasting

    @property
    def result(self) -> ARIMAFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    def fit(
        self,
        x: np.ndarray,
        method: str = "mle",
        *,
        include_const: bool = True,
    ) -> ARIMAFitResult:
        """
        Fit the ARIMA(p, d, q) model.

        Parameters
        ----------
        x : array_like, shape (T,)
        method : {'css', 'mle'}
        include_const : bool
            If True, drift is estimated (mean of the differenced series).
        """
        arr = to_numpy_1d(x)
        validate_finite(arr)
        validate_min_length(arr, self.d + self.p + self.q + 2, "x")

        # Store the last observation at each differencing level so that
        # forecasts can be integrated back to the original scale.
        # self._diff_seeds[k] = last observation of the k-th differenced series
        # (k=0 → original, k=1 → first-diff, …, k=d → d-th diff = fitted series)
        diff_seeds: list[float] = []
        y_temp = arr.copy()
        diff_seeds.append(float(y_temp[-1]))
        for _ in range(self.d):
            y_temp = difference(y_temp, 1)
            diff_seeds.append(float(y_temp[-1]))
        self._diff_seeds: list[float] = diff_seeds  # length d+1

        # Difference d times (y_temp is now the d-th differenced series)
        y_diff = y_temp

        # Fit ARMA on differenced series
        arma_res = self._arma.fit(y_diff, method=method, include_const=include_const)

        self._result = ARIMAFitResult(
            method=arma_res.method,
            ar_order=self.p,
            diff_order=self.d,
            ma_order=self.q,
            ar_coef=arma_res.ar_coef,
            ma_coef=arma_res.ma_coef,
            const=arma_res.const,
            sigma2=arma_res.sigma2,
            residuals=arma_res.residuals,
            fitted_values=arma_res.fitted_values,
            log_lik=arma_res.log_lik,
            aic=arma_res.aic,
            bic=arma_res.bic,
            hqic=arma_res.hqic,
            n_obs=arma_res.n_obs,
            is_stationary=arma_res.is_stationary,
            is_invertible=arma_res.is_invertible,
        )
        return self._result

    def forecast(
        self,
        h: int,
        *,
        alpha: float | None = 0.05,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        """
        h-step-ahead forecasts in the original (undifferenced) space.

        Prediction intervals are computed on the differenced scale and
        then integrated back using the observed last value at each
        differencing level stored during ``fit()``.
        """
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")

        # Forecast on the d-th differenced scale
        fcast_diff = self._arma.forecast(h, alpha=alpha, n_paths=n_paths, seed=seed)

        if self.d == 0:
            return fcast_diff

        def _integrate(series: np.ndarray) -> np.ndarray:
            """Integrate series from d-th differences back to original scale."""
            out = series.copy()
            for i in range(self.d):
                # Seed at the (d-1-i)-th differencing level: integrate inward→out
                seed_val = self._diff_seeds[self.d - 1 - i]
                out = np.concatenate([[seed_val], out])
                out = np.cumsum(out)[1:]
            return out

        mean_orig = _integrate(fcast_diff.mean)

        lower = upper = paths = None
        if fcast_diff.lower is not None and fcast_diff.upper is not None:
            lower = _integrate(fcast_diff.lower)
            upper = _integrate(fcast_diff.upper)

        if fcast_diff.paths is not None:
            paths = np.array([_integrate(p) for p in fcast_diff.paths])

        return ForecastResult(
            mean=mean_orig, horizon=h, alpha=alpha, lower=lower, upper=upper, paths=paths
        )

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> np.ndarray:
        """Simulate on the d-th differenced scale and integrate back."""
        res = self.result
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        n = t_total + burnin
        eps = rng.standard_normal(n) * sigma
        y = np.zeros(n)
        for t in range(n):
            y[t] = res.const + eps[t]
            for k in range(min(t, self.p)):
                y[t] += res.ar_coef[k] * (y[t - 1 - k] - res.const)
            for k in range(min(t, self.q)):
                y[t] += res.ma_coef[k] * eps[t - 1 - k]
        diff_sim = y[burnin:]

        if self.d == 0:
            return diff_sim

        # Integrate d times using stored diff seeds
        out = diff_sim
        for i in range(self.d):
            seed_val = self._diff_seeds[self.d - 1 - i]
            out = np.concatenate([[seed_val], out])
            out = np.cumsum(out)[1:]
        return out


# ---------------------------------------------------------------------------
# SARIMA model
# ---------------------------------------------------------------------------


class SARIMA:
    """
    Seasonal ARIMA model: SARIMA(p, d, q)(P, D, Q)[s].

    Internally:
    1. Applies D seasonal differences (lag s) then d regular differences.
    2. Expands the seasonal AR/MA polynomials and fits via the product-
       polynomial parameterisation: the optimiser sees the structural
       parameters (φ₁,…,φ_p, Φ₁,…,Φ_P, θ₁,…,θ_q, Θ₁,…,Θ_Q, log σ²)
       and the expanded coefficient arrays are computed internally.

    Parameters
    ----------
    p, d, q : int   Non-seasonal AR, diff, MA orders
    P, D, Q : int   Seasonal AR, diff, MA orders
    s       : int   Seasonal period (e.g. 12 for monthly, 4 for quarterly)
    """

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        P: int,  # noqa: N803
        D: int,  # noqa: N803
        Q: int,  # noqa: N803
        s: int,
    ) -> None:
        for name, val in [("p", p), ("d", d), ("q", q), ("P", P), ("D", D), ("Q", Q)]:
            if val < 0:
                raise ValueError(f"{name} must be ≥ 0, got {val}.")
        if s < 2:
            raise ValueError(f"Seasonal period s must be ≥ 2, got {s}.")
        if p + q + P + Q == 0:
            raise ValueError("At least one of p, q, P, Q must be > 0.")
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self._result: SARIMAFitResult | None = None

    @property
    def result(self) -> SARIMAFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        method: str = "mle",
        *,
        include_const: bool = True,
    ) -> SARIMAFitResult:
        """
        Fit the SARIMA model.

        Parameters
        ----------
        x      : array_like, shape (T,)
        method : {'css', 'mle'}
        """
        arr = to_numpy_1d(x)
        validate_finite(arr)
        min_len = self.D * self.s + self.d + (self.p + self.P * self.s) + 2
        validate_min_length(arr, min_len, "x")

        # 1. Seasonal differences
        y_diff = arr.copy()
        for _ in range(self.D):
            y_diff = seasonal_difference(y_diff, self.s)
        # 2. Regular differences
        for _ in range(self.d):
            y_diff = difference(y_diff, 1)

        const = float(np.mean(y_diff)) if include_const else 0.0
        y = y_diff - const

        # 3. Optimise over structural parameters
        match method:
            case "css" | "mle":
                phi, phi_s, theta, theta_s, sigma2 = self._fit_structural(y, method)
            case _:
                raise ValueError(f"Unknown method {method!r}; choose 'css' or 'mle'.")

        # 4. Expand polynomials
        phi_exp = _poly_mult_ar(phi, _expand_seasonal(phi_s, self.s)) if self.P else phi.copy()
        theta_exp = (
            _poly_mult_ar(theta, _expand_seasonal(theta_s, self.s)) if self.Q else theta.copy()
        )

        residuals = _css_residuals(phi_exp, theta_exp, y)
        fitted_dm = y - residuals
        n_eff = len(y)
        ll = _arma_log_likelihood(phi_exp, theta_exp, sigma2, y)
        n_params = self.p + self.P + self.q + self.Q + 1 + (1 if include_const else 0)
        aic_v, bic_v, hqic_v = info_criteria(ll, n_eff, n_params)

        self._result = SARIMAFitResult(
            method=method,
            ar_order=self.p,
            diff_order=self.d,
            ma_order=self.q,
            seasonal_ar_order=self.P,
            seasonal_diff_order=self.D,
            seasonal_ma_order=self.Q,
            period=self.s,
            ar_coef=phi,
            ma_coef=theta,
            seasonal_ar_coef=phi_s,
            seasonal_ma_coef=theta_s,
            ar_coef_expanded=phi_exp,
            ma_coef_expanded=theta_exp,
            const=const,
            sigma2=sigma2,
            residuals=residuals,
            fitted_values=fitted_dm + const,
            log_lik=ll,
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=n_eff,
        )
        return self._result

    def _fit_structural(
        self, y: np.ndarray, method: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Optimise over structural SARIMA parameters.

        Returns: phi (p,), phi_s (P,), theta (q,), theta_s (Q,), sigma2
        """
        n_params = self.p + self.P + self.q + self.Q
        params0 = np.zeros(n_params + 1)  # +1 for log_sigma2

        def _unpack(
            params: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            i = 0
            phi = params[i : i + self.p]
            i += self.p
            phi_s = params[i : i + self.P]
            i += self.P
            theta = params[i : i + self.q]
            i += self.q
            theta_s = params[i : i + self.Q]
            return phi, phi_s, theta, theta_s, float(params[-1])

        def objective(params: np.ndarray) -> float:
            phi, phi_s, theta, theta_s, log_s2 = _unpack(params)
            phi_exp = _poly_mult_ar(phi, _expand_seasonal(phi_s, self.s)) if self.P else phi
            theta_exp = _poly_mult_ar(theta, _expand_seasonal(theta_s, self.s)) if self.Q else theta
            sigma2 = np.exp(log_s2)
            if method == "css":
                eps = _css_residuals(phi_exp, theta_exp, y)
                return float(np.sum(eps**2))
            try:
                return -_arma_log_likelihood(phi_exp, theta_exp, sigma2, y)
            except Exception:
                return 1e10

        # Warm-start log_sigma2 from sample variance
        params0[-1] = np.log(max(float(np.var(y)), 1e-8))

        res = scipy.optimize.minimize(
            objective,
            params0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )

        phi, phi_s, theta, theta_s, log_s2 = _unpack(res.x)
        sigma2 = float(np.exp(log_s2))
        if method == "css":
            phi_exp = _poly_mult_ar(phi, _expand_seasonal(phi_s, self.s)) if self.P else phi
            theta_exp = _poly_mult_ar(theta, _expand_seasonal(theta_s, self.s)) if self.Q else theta
            eps = _css_residuals(phi_exp, theta_exp, y)
            sigma2 = float(np.var(eps, ddof=n_params))
        return phi, phi_s, theta, theta_s, sigma2

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        *,
        alpha: float | None = 0.05,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> ForecastResult:
        """h-step-ahead forecasts on the differenced scale (integration not applied)."""
        res = self.result
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")

        phi_exp = res.ar_coef_expanded
        theta_exp = res.ma_coef_expanded
        r = len(phi_exp)
        last_y = res.fitted_values[-max(r, 1) :] - res.const + res.residuals[-max(r, 1) :]
        q_exp = len(theta_exp)
        last_eps = res.residuals[-max(q_exp, 1) :]

        means = _arma_forecast_mean(phi_exp, theta_exp, last_y, last_eps, h, res.const)

        lower = upper = paths = None
        if alpha is not None:
            psi = _arma_impulse_response(phi_exp, theta_exp, h)
            var_h = res.sigma2 * np.cumsum(psi**2)
            z = float(scipy.stats.norm.ppf(1.0 - alpha / 2))
            half = z * np.sqrt(var_h)
            lower = means - half
            upper = means + half

        if n_paths is not None:
            rng = np.random.default_rng(seed)
            sigma = np.sqrt(res.sigma2)
            paths_arr = np.empty((n_paths, h))
            for pi in range(n_paths):
                y_buf = np.concatenate([last_y, np.zeros(h)])
                eps_buf = np.concatenate([last_eps, rng.standard_normal(h) * sigma])
                for t in range(h):
                    ti = len(last_y) + t
                    ei = len(last_eps) + t
                    yhat = res.const
                    for k in range(len(phi_exp)):
                        yhat += phi_exp[k] * (y_buf[ti - 1 - k] - res.const)
                    for k in range(len(theta_exp)):
                        yhat += theta_exp[k] * eps_buf[ei - 1 - k]
                    yhat += eps_buf[ei]
                    y_buf[ti] = yhat
                paths_arr[pi] = y_buf[len(last_y) :]
            paths = paths_arr

        return ForecastResult(
            mean=means, horizon=h, alpha=alpha, lower=lower, upper=upper, paths=paths
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> np.ndarray:
        """Simulate on the fully-differenced scale."""
        res = self.result
        phi_exp = res.ar_coef_expanded
        theta_exp = res.ma_coef_expanded
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(res.sigma2)
        n = t_total + burnin
        eps = rng.standard_normal(n) * sigma
        y = np.zeros(n)
        for t in range(n):
            y[t] = res.const + eps[t]
            for k in range(min(t, len(phi_exp))):
                y[t] += phi_exp[k] * (y[t - 1 - k] - res.const)
            for k in range(min(t, len(theta_exp))):
                y[t] += theta_exp[k] * eps[t - 1 - k]
        return y[burnin:]
