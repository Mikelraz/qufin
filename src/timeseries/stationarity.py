"""
Stationarity / unit-root tests.

Tests
-----
* ``adf``             — Augmented Dickey-Fuller (null: unit root)
* ``kpss``            — Kwiatkowski et al. 1992 (null: stationary)
* ``phillips_perron`` — Phillips-Perron (null: unit root, non-parametric correction)
* ``variance_ratio``  — Lo-MacKinlay 1988 random-walk test

Critical values for ADF and Phillips-Perron come from MacKinnon (1996)
response surfaces:

    τ_α(T) = β_∞ + β_1 / T + β_2 / T² + β_3 / T³

evaluated at the realised sample size.  KPSS critical values are the
small-sample tabulated values from Kwiatkowski et al. (1992).

P-values for ADF and Phillips-Perron are computed by **piecewise log-linear
interpolation** through the 1 %, 5 %, 10 % critical values plus a τ = 0 /
p = 0.5 anchor.  This is an approximation suitable for diagnostic use — for
precise inference at non-standard significance levels, compare the test
statistic to the published MacKinnon (1996) table directly.  The variance-
ratio test uses the exact heteroskedasticity-robust Z statistic from
Lo-MacKinlay (1988), so its p-value is normal-distribution exact.

All test functions accept ``np.ndarray | pl.Series | pl.DataFrame`` input
and return a result dataclass with a ``(statistic, p_value)`` tuple available
as the ``.stat`` and ``.p_value`` attributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from ._io import to_numpy_1d, validate_finite, validate_min_length

# ---------------------------------------------------------------------------
# MacKinnon (1996) response surfaces
# ---------------------------------------------------------------------------

# τ_α(T) = β_∞ + β_1/T + β_2/T² + β_3/T³
_ADF_CRIT_COEFS: dict[str, dict[float, tuple[float, float, float, float]]] = {
    "n": {
        0.01: (-2.56574, -2.2358, -3.627, 0.0),
        0.05: (-1.94100, -0.2686, -3.365, 31.223),
        0.10: (-1.61682, 0.2656, -2.714, 25.364),
    },
    "c": {
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.05: (-2.86154, -2.8903, -4.234, -40.040),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {
        0.01: (-3.95877, -9.0531, -28.428, -134.155),
        0.05: (-3.41049, -4.3904, -9.036, -45.374),
        0.10: (-3.12705, -2.5856, -3.925, -22.380),
    },
}

# KPSS (1992) asymptotic critical values
_KPSS_CRITS: dict[str, dict[float, float]] = {
    "c": {0.01: 0.739, 0.05: 0.463, 0.10: 0.347},
    "ct": {0.01: 0.216, 0.05: 0.146, 0.10: 0.119},
}


def _adf_critical_value(level: float, regression: str, n: int) -> float:
    """Asymptotic + small-sample-corrected critical value for ADF / Phillips-Perron."""
    coefs = _ADF_CRIT_COEFS[regression][level]
    return coefs[0] + coefs[1] / n + coefs[2] / (n * n) + coefs[3] / (n * n * n)


def _adf_pvalue(tau: float, regression: str, n: int) -> float:
    """
    Approximate p-value via piecewise log-linear interpolation through the
    1 %, 5 %, 10 % critical values plus a (τ = 0, p = 0.5) right-side anchor.

    For τ to the left of the 1 % critical value, log-linear extrapolation
    from the 1 % - 5 % slope.  For τ ≥ 0, return values approaching 1.0.
    Output is always clamped to (1e-10, 1 - 1e-10).
    """
    c1 = _adf_critical_value(0.01, regression, n)
    c5 = _adf_critical_value(0.05, regression, n)
    c10 = _adf_critical_value(0.10, regression, n)

    # Nodes in ascending τ, with paired p-values.  τ = 0 → p ≈ 0.5 is a soft
    # right-side anchor (the true asymptotic value is regression-dependent
    # but always near 0.5 for finite samples — this is the diagnostic-use
    # approximation noted in the module docstring).
    taus = np.array([c1, c5, c10, 0.0])
    log_ps = np.log(np.array([0.01, 0.05, 0.10, 0.50]))

    if tau >= 0.0:
        # Smooth tail towards 1 as τ becomes positive.
        return float(min(1.0 - 1e-10, 0.5 + 0.5 * math.tanh(tau)))

    if tau <= taus[0]:
        # Linear extrapolation in (τ, log p) using the 1 % – 5 % slope.
        slope = (log_ps[1] - log_ps[0]) / (taus[1] - taus[0])
        log_p = log_ps[0] + slope * (tau - taus[0])
        return float(max(1e-10, math.exp(log_p)))

    log_p_interp = float(np.interp(tau, taus, log_ps))
    return float(min(1.0 - 1e-10, max(1e-10, math.exp(log_p_interp))))


def _kpss_pvalue(stat: float, regression: str) -> float:
    """Piecewise interpolation through the KPSS critical values."""
    crits = _KPSS_CRITS[regression]
    # Critical values are increasing in significance (small p → large stat).
    stats_arr = np.array([crits[0.10], crits[0.05], crits[0.01]])
    ps = np.array([0.10, 0.05, 0.01])
    if stat <= stats_arr[0]:
        return float(min(1.0 - 1e-10, 0.10 + (stats_arr[0] - stat) / stats_arr[0] * 0.40))
    if stat >= stats_arr[-1]:
        # Extrapolate log-linearly past the 1 % level.
        log_ps = np.log(ps)
        slope = (log_ps[-1] - log_ps[-2]) / (stats_arr[-1] - stats_arr[-2])
        log_p = log_ps[-1] + slope * (stat - stats_arr[-1])
        return float(max(1e-10, math.exp(log_p)))
    log_p = float(np.interp(stat, stats_arr, np.log(ps)))
    return float(math.exp(log_p))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ADFResult:
    """Augmented Dickey-Fuller test outcome."""

    stat: float
    p_value: float
    used_lag: int
    n_obs: int
    regression: str
    critical_values: dict[float, float] = field(default_factory=dict)

    def __str__(self) -> str:
        crits = "  ".join(
            f"{int(100 * lvl)}%: {cv:.4f}" for lvl, cv in sorted(self.critical_values.items())
        )
        return (
            f"ADF (regression={self.regression!r}, lag={self.used_lag}, n={self.n_obs})\n"
            f"  τ-statistic : {self.stat:.6f}\n"
            f"  p-value     : {self.p_value:.6f}\n"
            f"  crit. values: {crits}"
        )


@dataclass
class KPSSResult:
    """KPSS test outcome (null: trend stationary)."""

    stat: float
    p_value: float
    used_lag: int
    n_obs: int
    regression: str
    critical_values: dict[float, float] = field(default_factory=dict)


@dataclass
class PPResult:
    """Phillips-Perron test outcome."""

    stat: float
    p_value: float
    used_lag: int
    n_obs: int
    regression: str
    critical_values: dict[float, float] = field(default_factory=dict)


@dataclass
class VRResult:
    """Lo-MacKinlay (1988) variance-ratio test outcome."""

    stat: float
    p_value: float
    z_stat: float
    q: int
    n_obs: int


# ---------------------------------------------------------------------------
# ADF
# ---------------------------------------------------------------------------


def _default_maxlag(n: int) -> int:
    """Schwert (1989) upper bound on the number of lagged differences."""
    return int(np.ceil(12.0 * (n / 100.0) ** 0.25))


def adf(
    x: Any,
    regression: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> ADFResult:
    """
    Augmented Dickey-Fuller test for a unit root.

    The auxiliary regression is

        Δy_t = α + β t + γ y_{t-1} + Σ_{i=1}^p δ_i Δy_{t-i} + ε_t

    where the constant α and trend β are included based on ``regression``.
    The test statistic is the t-statistic on γ; the null hypothesis is γ = 0
    (unit root present).

    Parameters
    ----------
    x          Series under test (``np.ndarray`` / ``pl.Series`` / ``pl.DataFrame``).
    regression One of ``'n'`` (no constant), ``'c'`` (constant), ``'ct'``
               (constant + trend).
    maxlag     Maximum number of lagged differences to consider.  If ``None``,
               uses Schwert's (1989) upper bound ⌈12 (n/100)^(1/4)⌉.
    autolag    Lag-selection rule applied for 0 … maxlag lags:
               ``'aic'``, ``'bic'``, or ``None`` (use ``maxlag`` directly).

    Returns
    -------
    ADFResult
    """
    if regression not in ("n", "c", "ct"):
        raise ValueError(f"regression must be one of 'n', 'c', 'ct', got {regression!r}.")
    if autolag is not None and autolag not in ("aic", "bic"):
        raise ValueError(f"autolag must be one of None, 'aic', 'bic', got {autolag!r}.")

    arr = to_numpy_1d(x)
    validate_finite(arr)
    n = arr.shape[0]
    if maxlag is None:
        maxlag = _default_maxlag(n)
    if maxlag < 0:
        raise ValueError(f"maxlag must be >= 0, got {maxlag}.")
    validate_min_length(arr, maxlag + 4, "x")

    if autolag is None:
        used_lag = maxlag
        stat = _adf_regression_tstat(arr, used_lag, regression)
    else:
        best_score = math.inf
        used_lag = 0
        stat = math.nan
        for p in range(maxlag + 1):
            try:
                tau, n_eff, rss, k = _adf_regression_details(arr, p, regression)
            except np.linalg.LinAlgError:
                continue
            # AIC / BIC on the auxiliary regression (Gaussian innovations).
            log_lik = -0.5 * n_eff * (math.log(2.0 * math.pi * rss / n_eff) + 1.0)
            penalty = 2.0 * k if autolag == "aic" else k * math.log(n_eff)
            score = -2.0 * log_lik + penalty
            if score < best_score:
                best_score = score
                used_lag = p
                stat = tau

    crit_values = {lvl: _adf_critical_value(lvl, regression, n) for lvl in (0.01, 0.05, 0.10)}
    p_value = _adf_pvalue(stat, regression, n)
    return ADFResult(
        stat=float(stat),
        p_value=p_value,
        used_lag=int(used_lag),
        n_obs=n,
        regression=regression,
        critical_values=crit_values,
    )


def _adf_design(arr: np.ndarray, p: int, regression: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct the ADF auxiliary-regression design matrix and target."""
    dy = np.diff(arr)
    n_eff = dy.shape[0] - p
    if n_eff < 2:
        raise np.linalg.LinAlgError("Series too short for the requested lag.")
    y = dy[p:]
    cols = [arr[p : p + n_eff]]  # y_{t-1}
    for i in range(1, p + 1):
        cols.append(dy[p - i : p - i + n_eff])
    if regression in ("c", "ct"):
        cols.append(np.ones(n_eff))
    if regression == "ct":
        cols.append(np.arange(1, n_eff + 1, dtype=np.float64))
    design = np.column_stack(cols)
    return design, y


def _adf_regression_details(
    arr: np.ndarray, p: int, regression: str
) -> tuple[float, int, float, int]:
    """Return (τ, n_eff, RSS, k) for the ADF regression at lag p."""
    design, y = _adf_design(arr, p, regression)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    n_eff = y.shape[0]
    k = design.shape[1]
    rss = float(resid @ resid)
    sigma2 = rss / (n_eff - k)
    # Variance of γ_hat is sigma² · (X'X)^{-1}[0, 0].
    xtx_inv = np.linalg.inv(design.T @ design)
    se_gamma = math.sqrt(sigma2 * xtx_inv[0, 0])
    tau = float(coef[0] / se_gamma) if se_gamma > 0.0 else math.nan
    return tau, n_eff, rss, k


def _adf_regression_tstat(arr: np.ndarray, p: int, regression: str) -> float:
    tau, _, _, _ = _adf_regression_details(arr, p, regression)
    return tau


# ---------------------------------------------------------------------------
# KPSS
# ---------------------------------------------------------------------------


def kpss(x: Any, regression: str = "c", lags: int | None = None) -> KPSSResult:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for level / trend stationarity.

    Null hypothesis: ``x`` is (trend-)stationary.  A small p-value rejects
    stationarity, i.e. evidence for a unit root.

    Parameters
    ----------
    x          Series under test.
    regression ``'c'`` (level-stationary) or ``'ct'`` (trend-stationary).
    lags       Number of Bartlett-kernel lags for the long-run variance
               estimator.  Defaults to ⌊4 (n/100)^(1/4)⌋ (Schwert 1989).

    Returns
    -------
    KPSSResult
    """
    if regression not in ("c", "ct"):
        raise ValueError(f"regression must be one of 'c', 'ct', got {regression!r}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    n = arr.shape[0]
    validate_min_length(arr, 8, "x")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}.")

    if regression == "c":
        resid = arr - arr.mean()
    else:
        t = np.arange(1, n + 1, dtype=np.float64)
        design = np.column_stack([np.ones(n), t])
        coef, *_ = np.linalg.lstsq(design, arr, rcond=None)
        resid = arr - design @ coef

    s = np.cumsum(resid)
    eta = float(np.sum(s * s) / (n * n))
    lr_var = _bartlett_long_run_var(resid, lags)
    if lr_var <= 0.0:
        lr_var = float(np.var(resid))
    stat = eta / lr_var
    p_value = _kpss_pvalue(stat, regression)
    return KPSSResult(
        stat=stat,
        p_value=p_value,
        used_lag=lags,
        n_obs=n,
        regression=regression,
        critical_values=dict(_KPSS_CRITS[regression]),
    )


def _bartlett_long_run_var(resid: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance estimator with Bartlett kernel."""
    n = resid.shape[0]
    gamma_0 = float(np.dot(resid, resid) / n)
    total = gamma_0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(np.dot(resid[:-k], resid[k:]) / n)
        total += 2.0 * w * gamma_k
    return total


# ---------------------------------------------------------------------------
# Phillips-Perron
# ---------------------------------------------------------------------------


def phillips_perron(
    x: Any,
    regression: str = "c",
    lags: int | None = None,
) -> PPResult:
    """
    Phillips-Perron unit-root test.

    Runs the un-augmented Dickey-Fuller regression (no lagged differences)
    and applies a non-parametric correction to the t-statistic to account
    for serial correlation in the residuals.  The long-run variance is
    estimated with the Bartlett kernel à la Newey-West.

    Parameters
    ----------
    x          Series under test.
    regression ``'n'``, ``'c'``, or ``'ct'``.
    lags       Bartlett-kernel lag truncation (default: Schwert's rule).

    Returns
    -------
    PPResult
    """
    if regression not in ("n", "c", "ct"):
        raise ValueError(f"regression must be one of 'n', 'c', 'ct', got {regression!r}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    n = arr.shape[0]
    validate_min_length(arr, 8, "x")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}.")

    design, y = _adf_design(arr, p=0, regression=regression)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    n_eff = y.shape[0]
    k = design.shape[1]
    sigma2 = float(resid @ resid) / (n_eff - k)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_gamma = math.sqrt(sigma2 * xtx_inv[0, 0])
    tau = float(coef[0] / se_gamma)

    sigma2_short = sigma2
    sigma2_long = _bartlett_long_run_var(resid, lags)
    if sigma2_short <= 0.0 or sigma2_long <= 0.0:
        z_tau = tau
    else:
        # PP Z_τ correction (Phillips-Perron 1988, eq. 9).
        denom = math.sqrt(sigma2_long) * math.sqrt(sigma2)
        correction = 0.5 * (sigma2_long - sigma2_short) * se_gamma / denom
        z_tau = tau * math.sqrt(sigma2_short / sigma2_long) - correction

    crit_values = {lvl: _adf_critical_value(lvl, regression, n) for lvl in (0.01, 0.05, 0.10)}
    p_value = _adf_pvalue(z_tau, regression, n)
    return PPResult(
        stat=z_tau,
        p_value=p_value,
        used_lag=lags,
        n_obs=n,
        regression=regression,
        critical_values=crit_values,
    )


# ---------------------------------------------------------------------------
# Variance ratio (Lo-MacKinlay 1988)
# ---------------------------------------------------------------------------


def variance_ratio(x: Any, q: int, robust: bool = True) -> VRResult:
    """
    Lo-MacKinlay (1988) variance-ratio test of the random-walk null.

    Tests whether ``x`` follows a random walk by comparing the variance of
    q-period returns to q times the variance of one-period returns.  Under
    the random-walk null,

        VR(q) = Var(r_t + r_{t-1} + … + r_{t-q+1}) / (q · Var(r_t)) = 1.

    The Z statistic is asymptotically standard normal.  If ``robust=True``,
    uses the heteroskedasticity-consistent estimator (Lo-MacKinlay 1988,
    eq. 4).

    Parameters
    ----------
    x       Series of *returns* (or first differences).  Pass log returns
            for the classical random-walk-in-log-prices interpretation.
    q       Aggregation horizon (>= 2).
    robust  If True, use the heteroskedasticity-robust standard error.

    Returns
    -------
    VRResult
    """
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}.")
    arr = to_numpy_1d(x)
    validate_finite(arr)
    n = arr.shape[0]
    validate_min_length(arr, max(q + 2, 8), "x")

    mu = float(arr.mean())
    centred = arr - mu
    var_1 = float(np.sum(centred * centred) / (n - 1))

    # Overlapping q-period returns.
    cum = np.concatenate(([0.0], np.cumsum(arr)))
    agg = cum[q:] - cum[:-q] - q * mu
    var_q = float(np.sum(agg * agg) / (q * (n - q + 1) * (1 - q / n)))
    vr = var_q / var_1

    if robust:
        # Heteroskedasticity-consistent variance of VR(q) - 1 from Lo-MacKinlay
        # (1988) eq. 4:  δ̂(j) = Σ ε² ε_{t-j}² / (Σ ε²)²   (no n multiplier;
        # under i.i.d. δ̂(j) → 1/n asymptotically, which recovers the homo-
        # skedastic 2(2q-1)(q-1)/(3qn) limit).
        sum_sq = float(np.dot(centred, centred))
        delta = 0.0
        for k in range(1, q):
            gamma_k = float(np.dot(centred[k:] * centred[:-k], centred[k:] * centred[:-k]))
            delta += (2.0 * (q - k) / q) ** 2 * gamma_k / (sum_sq * sum_sq)
        homoskedastic_var = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
        se = math.sqrt(delta) if delta > 0.0 else math.sqrt(homoskedastic_var)
    else:
        # Homoskedastic Lo-MacKinlay (1988) eq. 11.
        se = math.sqrt(2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n))

    z = (vr - 1.0) / se if se > 0.0 else math.nan
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    return VRResult(
        stat=float(vr),
        p_value=p_value,
        z_stat=float(z),
        q=q,
        n_obs=n,
    )
