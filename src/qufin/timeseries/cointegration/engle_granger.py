"""
Engle-Granger (1987) two-step cointegration test.

Procedure
---------
1. Run the static OLS regression  y_t = α + β x_t + u_t  (or with multiple
   regressors  y_t = α + β' X_t + u_t).
2. Apply the Augmented Dickey-Fuller test to the residuals û_t with the
   *no-intercept, no-trend* regression.

A rejection of the unit-root null in step 2 is evidence that (y, X) is
cointegrated with cointegrating vector (1, −β).

Critical values
---------------
The residual-based ADF test does **not** use the standard Dickey-Fuller
critical values because the residuals are themselves estimated.  We use the
Engle-Granger / MacKinnon (1990, 2010) response-surface coefficients indexed
by the number of regressors in step 1 (the *cointegration rank* test, with
1, 2, or 3 stochastic regressors plus the dependent variable).

For more than three regressors (rare in practice) we fall back to the
two-variable critical values with a warning.  The p-value is approximated
via piecewise log-linear interpolation through the 1 %, 5 %, 10 % response
surfaces.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._io import to_numpy_1d, to_numpy_2d, validate_finite, validate_min_length
from ..stationarity import _adf_design

# MacKinnon (2010) cointegration critical-value response surfaces.
# Coefficients (β_∞, β_1, β_2) for τ_α(T) = β_∞ + β_1/T + β_2/T².
# Indexed by (n_regressors, regression_type, level).
# Source: MacKinnon (2010) "Critical Values for Cointegration Tests", QED WP No. 1227,
# constant-only specification ("c"), for residual ADF.
_EG_COEFS: dict[int, dict[float, tuple[float, float, float]]] = {
    # n_regressors = 1 (i.e. y on x, single x)
    1: {
        0.01: (-3.9001, -10.534, -30.03),
        0.05: (-3.3377, -5.967, -8.98),
        0.10: (-3.0462, -4.069, -5.73),
    },
    # n_regressors = 2
    2: {
        0.01: (-4.2981, -13.790, -46.37),
        0.05: (-3.7429, -8.352, -13.41),
        0.10: (-3.4518, -6.241, -2.79),
    },
    # n_regressors = 3
    3: {
        0.01: (-4.6493, -17.188, -59.20),
        0.05: (-4.1000, -10.745, -21.57),
        0.10: (-3.8110, -8.317, -5.19),
    },
}


def _eg_critical_value(level: float, n_reg: int, n: int) -> float:
    """Engle-Granger critical value with finite-sample correction."""
    if n_reg < 1:
        n_reg = 1
    if n_reg > 3:
        warnings.warn(
            f"Engle-Granger critical values for {n_reg} regressors are not "
            "tabulated; using the 3-regressor approximation.",
            RuntimeWarning,
            stacklevel=3,
        )
        n_reg = 3
    coefs = _EG_COEFS[n_reg][level]
    return coefs[0] + coefs[1] / n + coefs[2] / (n * n)


def _eg_pvalue(tau: float, n_reg: int, n: int) -> float:
    """Piecewise log-linear p-value interpolation."""
    c1 = _eg_critical_value(0.01, n_reg, n)
    c5 = _eg_critical_value(0.05, n_reg, n)
    c10 = _eg_critical_value(0.10, n_reg, n)
    taus = np.array([c1, c5, c10, 0.0])
    log_ps = np.log(np.array([0.01, 0.05, 0.10, 0.50]))

    if tau >= 0.0:
        return float(min(1.0 - 1e-10, 0.5 + 0.5 * math.tanh(tau)))
    if tau <= taus[0]:
        slope = (log_ps[1] - log_ps[0]) / (taus[1] - taus[0])
        log_p = log_ps[0] + slope * (tau - taus[0])
        return float(max(1e-10, math.exp(log_p)))
    log_p_interp = float(np.interp(tau, taus, log_ps))
    return float(min(1.0 - 1e-10, max(1e-10, math.exp(log_p_interp))))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EngleGrangerResult:
    """Engle-Granger two-step test result.

    Attributes
    ----------
    beta              Cointegrating slope coefficient(s)  shape (k,)
    alpha             Intercept of the static regression
    residuals         OLS residuals û_t                   shape (T,)
    adf_stat          ADF τ-statistic on the residuals (no-intercept regression)
    p_value           Approximate Engle-Granger p-value
    used_lag          Number of lagged-difference terms in the residual ADF
    n_obs             Length of the cointegration regression
    n_regressors      Number of x columns (k)
    cointegrated      Convenience boolean at the 5 % level
    critical_values   {0.01, 0.05, 0.10} → Engle-Granger τ critical values
    """

    beta: np.ndarray
    alpha: float
    residuals: np.ndarray
    adf_stat: float
    p_value: float
    used_lag: int
    n_obs: int
    n_regressors: int
    cointegrated: bool
    critical_values: dict[float, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def engle_granger(
    y: Any,
    x: Any,
    *,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
    alpha: float = 0.05,
) -> EngleGrangerResult:
    """Engle-Granger two-step cointegration test.

    Parameters
    ----------
    y            1-D dependent series, length T.
    x            Independent series — 1-D (single regressor) or 2-D (T × k).
    trend        Trend specification of the static regression:
                 ``'n'`` (no constant), ``'c'`` (constant), ``'ct'``
                 (constant + linear trend).  The residual ADF always uses
                 the ``'n'`` (no-intercept) regression as required by
                 Engle-Granger.
    maxlag       Maximum lagged-difference order in the residual ADF.
                 Defaults to ⌈12 (T/100)^(1/4)⌉ (Schwert 1989).
    autolag      Lag selection for the residual ADF: ``'aic'``, ``'bic'``,
                 or ``None`` (use ``maxlag`` directly).
    alpha        Significance level for the convenience ``cointegrated``
                 boolean.

    Returns
    -------
    EngleGrangerResult
    """
    if trend not in ("n", "c", "ct"):
        raise ValueError(f"trend must be one of 'n', 'c', 'ct', got {trend!r}.")
    if autolag is not None and autolag not in ("aic", "bic"):
        raise ValueError(f"autolag must be None, 'aic', or 'bic', got {autolag!r}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    y_arr = to_numpy_1d(y)
    validate_finite(y_arr, "y")

    try:
        x_arr = to_numpy_2d(x)
    except ValueError:
        x_arr = to_numpy_1d(x).reshape(-1, 1)
    validate_finite(x_arr, "x")

    n = y_arr.shape[0]
    if x_arr.shape[0] != n:
        raise ValueError(f"y and x must have the same length; got {n} and {x_arr.shape[0]}.")
    validate_min_length(y_arr, 10, "y")
    k = x_arr.shape[1]

    # Step 1: static OLS regression
    cols = [x_arr]
    if trend in ("c", "ct"):
        cols.append(np.ones((n, 1)))
    if trend == "ct":
        cols.append(np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    resid = y_arr - design @ coef

    beta = coef[:k].copy()
    intercept = float(coef[k]) if trend in ("c", "ct") else 0.0

    # Step 2: residual ADF with no constant
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    if maxlag < 0:
        raise ValueError(f"maxlag must be ≥ 0, got {maxlag}.")

    if autolag is None:
        tau, used_lag = _residual_adf_tstat(resid, maxlag), maxlag
    else:
        best_score = math.inf
        used_lag = 0
        tau = math.nan
        for p in range(maxlag + 1):
            try:
                t_p, n_eff, rss, n_params = _residual_adf_details(resid, p)
            except np.linalg.LinAlgError:
                continue
            if rss <= 0.0 or n_eff <= n_params:
                continue
            log_lik = -0.5 * n_eff * (math.log(2.0 * math.pi * rss / n_eff) + 1.0)
            penalty = 2.0 * n_params if autolag == "aic" else n_params * math.log(n_eff)
            score = -2.0 * log_lik + penalty
            if score < best_score:
                best_score = score
                used_lag = p
                tau = t_p

    p_value = _eg_pvalue(tau, k, n)
    crit = {lvl: _eg_critical_value(lvl, k, n) for lvl in (0.01, 0.05, 0.10)}
    cointegrated = bool(tau < crit[alpha]) if alpha in crit else bool(p_value < alpha)

    return EngleGrangerResult(
        beta=beta,
        alpha=intercept,
        residuals=resid,
        adf_stat=float(tau),
        p_value=p_value,
        used_lag=int(used_lag),
        n_obs=n,
        n_regressors=k,
        cointegrated=cointegrated,
        critical_values=crit,
    )


def _residual_adf_details(resid: np.ndarray, p: int) -> tuple[float, int, float, int]:
    """ADF on residuals with ``regression='n'``.  Returns (τ, n_eff, RSS, k)."""
    design, y = _adf_design(resid, p, regression="n")
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    eps = y - design @ coef
    n_eff = y.shape[0]
    k = design.shape[1]
    rss = float(eps @ eps)
    if n_eff <= k:
        raise np.linalg.LinAlgError("Insufficient degrees of freedom in residual ADF.")
    sigma2 = rss / (n_eff - k)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_gamma = math.sqrt(sigma2 * xtx_inv[0, 0])
    tau = float(coef[0] / se_gamma) if se_gamma > 0.0 else math.nan
    return tau, n_eff, rss, k


def _residual_adf_tstat(resid: np.ndarray, p: int) -> float:
    return _residual_adf_details(resid, p)[0]
