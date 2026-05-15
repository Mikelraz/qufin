"""
Residual / serial-correlation diagnostic tests.

Functions
---------
* ``acf``         — sample autocorrelation with Bartlett confidence bands
* ``pacf``        — sample partial autocorrelation via Durbin-Levinson
* ``ljung_box``   — Q statistic for joint autocorrelation up to lag h
* ``jarque_bera`` — joint test of skewness and excess kurtosis vs. normality
* ``arch_lm``     — Engle (1982) Lagrange-multiplier test for ARCH effects

All return either a ``(statistic, p_value)`` tuple (matching the existing
``OrnsteinUhlenbeck.ljung_box`` signature) or an ``ACFResult`` dataclass when
multi-lag arrays are involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ._io import to_numpy_1d, validate_finite, validate_min_length
from ._kernels import durbin_levinson, sample_acf, sample_acovf


@dataclass
class ACFResult:
    """
    Sample (P)ACF together with white-noise confidence bands.

    The bands are symmetric around zero: under H0 (white noise) each
    sample (P)ACF coefficient has approximate standard error 1 / √n.

    Attributes
    ----------
    values     The sample (P)ACF values at lags 1 … nlags.
    lower_ci   Lower band (constant ``-z / √n``).
    upper_ci   Upper band (constant ``+z / √n``).
    confidence Two-sided coverage level (e.g. 0.95).
    n_obs      Sample size used.
    """

    values: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    confidence: float
    n_obs: int

    def to_dataframe(self) -> pl.DataFrame:
        """Return a polars DataFrame with columns (lag, value, lower, upper)."""
        return pl.DataFrame(
            {
                "lag": np.arange(1, self.values.shape[0] + 1, dtype=np.int64),
                "value": self.values,
                "lower": self.lower_ci,
                "upper": self.upper_ci,
            }
        )


def _bartlett_bands(n: int, nlags: int, ci: float) -> tuple[np.ndarray, np.ndarray]:
    z = float(stats.norm.ppf(1.0 - (1.0 - ci) / 2.0))
    se = z / np.sqrt(n)
    return np.full(nlags, -se), np.full(nlags, +se)


def acf(x: Any, nlags: int = 20, ci: float = 0.95) -> ACFResult:
    """
    Sample autocorrelation function with Bartlett white-noise CI bands.

    Parameters
    ----------
    x      Series (np.ndarray, pl.Series, pl.DataFrame, …).
    nlags  Number of positive lags to compute (>= 1).
    ci     Two-sided coverage level for the white-noise bands.

    Returns
    -------
    ACFResult
    """
    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}.")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, nlags + 2, "x")

    values = sample_acf(arr, nlags)
    lower, upper = _bartlett_bands(arr.shape[0], nlags, ci)
    return ACFResult(
        values=values,
        lower_ci=lower,
        upper_ci=upper,
        confidence=ci,
        n_obs=arr.shape[0],
    )


def pacf(x: Any, nlags: int = 20, ci: float = 0.95) -> ACFResult:
    """
    Sample partial autocorrelation function via the Durbin-Levinson recursion.

    Under H0 (white noise) each sample PACF coefficient has approximate
    standard error 1 / √n — same Bartlett bands as the ACF.

    Parameters
    ----------
    x      Series (np.ndarray, pl.Series, pl.DataFrame, …).
    nlags  Number of positive lags to compute (>= 1).
    ci     Two-sided coverage level.

    Returns
    -------
    ACFResult
    """
    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}.")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, nlags + 2, "x")

    acovs = sample_acovf(arr, nlags)
    _, pacf_vals, _ = durbin_levinson(acovs)
    lower, upper = _bartlett_bands(arr.shape[0], nlags, ci)
    return ACFResult(
        values=pacf_vals,
        lower_ci=lower,
        upper_ci=upper,
        confidence=ci,
        n_obs=arr.shape[0],
    )


def ljung_box(x: Any, lags: int = 10, dof_adjust: int = 0) -> tuple[float, float]:
    """
    Ljung-Box portmanteau test for joint autocorrelation.

        Q = n (n + 2) Σ_{k=1}^h ρ²_k / (n - k)

    Under H0 (i.i.d.), Q ~ χ²(h - dof_adjust) asymptotically.  Set
    ``dof_adjust`` to the number of fitted ARMA parameters when applying
    to model residuals (Box-Pierce correction).

    Parameters
    ----------
    x          Series (typically model residuals).
    lags       Number of lags h in the test statistic (>= 1).
    dof_adjust Subtract from the chi-squared degrees of freedom.

    Returns
    -------
    statistic : float
    p_value : float
    """
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}.")
    if dof_adjust < 0:
        raise ValueError(f"dof_adjust must be >= 0, got {dof_adjust}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, lags + 2, "x")

    n = arr.shape[0]
    rho = sample_acf(arr, lags)
    k = np.arange(1, lags + 1)
    q = float(n * (n + 2) * np.sum(rho * rho / (n - k)))
    df = lags - dof_adjust
    if df <= 0:
        raise ValueError(f"degrees of freedom (lags - dof_adjust) must be > 0, got {df}.")
    p = float(stats.chi2.sf(q, df=df))
    return q, p


def jarque_bera(x: Any) -> tuple[float, float]:
    """
    Jarque-Bera joint test of skewness and excess kurtosis.

        JB = (n / 6) [ S² + (1/4) K² ]

    where S is sample skewness and K is sample excess kurtosis.  Under H0
    (normality), JB ~ χ²(2).

    Returns
    -------
    statistic : float
    p_value : float
    """
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, 4, "x")

    n = arr.shape[0]
    d = arr - arr.mean()
    m2 = float(np.mean(d * d))
    if m2 == 0.0:
        return 0.0, 1.0
    m3 = float(np.mean(d * d * d))
    m4 = float(np.mean(d * d * d * d))
    skew = m3 / m2**1.5
    excess_kurt = m4 / (m2 * m2) - 3.0
    jb = n / 6.0 * (skew * skew + 0.25 * excess_kurt * excess_kurt)
    p = float(stats.chi2.sf(jb, df=2))
    return float(jb), p


def arch_lm(x: Any, lags: int = 12) -> tuple[float, float]:
    """
    Engle (1982) Lagrange-multiplier test for ARCH effects.

    Regresses x²_t on a constant and ``lags`` lags of x²_{t-i} (OLS).
    Under H0 (no ARCH), the test statistic LM = (n - lags) · R² is
    asymptotically χ²(lags).

    Parameters
    ----------
    x      Returns or model residuals.
    lags   Number of squared lags in the auxiliary regression.

    Returns
    -------
    statistic : float
    p_value : float
    """
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, lags + 2, "x")

    sq = arr * arr
    n = sq.shape[0]
    y = sq[lags:]
    cols = [np.ones(n - lags)]
    for k in range(1, lags + 1):
        cols.append(sq[lags - k : n - k])
    design = np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    lm = float((n - lags) * r_squared)
    p = float(stats.chi2.sf(lm, df=lags))
    return lm, p
