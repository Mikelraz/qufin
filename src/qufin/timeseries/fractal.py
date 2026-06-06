"""
Long-memory and fractal analysis.

Estimators of the **Hurst exponent** ``H`` and the **fractal dimension** ``D``
of a time series — diagnostics for persistence (trendiness), anti-persistence
(mean-reversion), and self-similarity.

Hurst exponent
--------------
* ``rs_analysis``         — classical rescaled-range (Hurst 1951; Mandelkbrot-Wallis).
* ``dfa``                 — detrended fluctuation analysis (Peng et al. 1994),
  robust to non-stationary trends.
* ``aggregated_variance`` — variance of block means vs block size.
* ``hurst``               — dispatcher over the three methods above.

Interpretation of ``H``:

* ``H = 0.5``   — no memory (random walk increments / Brownian motion).
* ``H > 0.5``   — persistent / trending (positively autocorrelated increments).
* ``H < 0.5``   — anti-persistent / mean-reverting.

The R/S and aggregated-variance estimators target ``H`` for a stationary
*increment* series.  DFA reports the scaling exponent ``α``, which equals ``H``
for stationary noise and ``H + 1`` for an integrated (random-walk) series — so
run DFA on returns, not prices, to read ``α`` as a Hurst exponent.

Fractal dimension
-----------------
* ``fractal_dimension`` — Higuchi's (1988) method, or ``D = 2 − H`` from R/S.
  ``D`` lies in ``[1, 2]``: ``D → 1`` is a smooth/trending curve, ``D → 2`` a
  rough/space-filling one.

Multifractal analysis
---------------------
* ``mfdfa`` — multifractal DFA (Kantelhardt et al. 2002): a *spectrum* of
  generalised Hurst exponents ``h(q)``.  A flat ``h(q)`` is monofractal; a
  steeply decreasing ``h(q)`` (wide singularity spectrum ``f(α)``) signals
  multifractality, common in volatility and volume series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._io import to_numpy_1d, validate_finite, validate_min_length
from ._kernels import (
    detrended_seg_vars,
    dfa_fluctuations,
    higuchi_lengths,
    rescaled_range,
)

HurstMethod = Literal["rs", "dfa", "agg_var"]


@dataclass(slots=True, frozen=True)
class HurstResult:
    """
    Hurst-exponent estimate from a log-log scaling regression.

    Attributes
    ----------
    exponent      Estimated Hurst exponent ``H`` (or DFA scaling exponent ``α``).
    intercept     Intercept of the log-log fit.
    r_squared     Coefficient of determination of the scaling regression — close
                  to 1 indicates clean power-law scaling and a trustworthy ``H``.
    method        ``"rs"``, ``"dfa"`` or ``"agg_var"``.
    scales        Window sizes used, shape ``(k,)``.
    fluctuations  Per-scale statistic regressed against ``scales`` (R/S, F(n) or
                  block-mean variance), shape ``(k,)``.
    n_obs         Length of the input series.
    """

    exponent: float
    intercept: float
    r_squared: float
    method: str
    scales: np.ndarray
    fluctuations: np.ndarray
    n_obs: int

    def __str__(self) -> str:
        return (
            f"Hurst(method={self.method!r}, H={self.exponent:.4f}, "
            f"R²={self.r_squared:.4f}, n={self.n_obs}, scales={self.scales.shape[0]})"
        )


def _log_scales(n: int, min_scale: int, max_scale: int | None, n_scales: int) -> np.ndarray:
    """Unique log-spaced integer window sizes in ``[min_scale, max_scale]``."""
    if min_scale < 4:
        raise ValueError(f"min_scale must be >= 4, got {min_scale}.")
    upper = n // 2 if max_scale is None else min(max_scale, n // 2)
    if upper <= min_scale:
        raise ValueError(f"series too short: need max_scale ({upper}) > min_scale ({min_scale}).")
    raw = np.logspace(math.log10(min_scale), math.log10(upper), n_scales)
    scales = np.unique(np.floor(raw).astype(np.int64))
    return scales[scales >= min_scale]


def _loglog_fit(scales: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """OLS of ``log(values)`` on ``log(scales)``; returns (slope, intercept, R²)."""
    mask = np.isfinite(values) & (values > 0.0)
    if int(mask.sum()) < 2:
        raise ValueError("not enough valid scales for the scaling regression.")
    log_x = np.log(scales[mask].astype(np.float64))
    log_y = np.log(values[mask])
    design = np.column_stack([log_x, np.ones_like(log_x)])
    coef, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = log_y - design @ coef
    rss = float(resid @ resid)
    tss = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
    return slope, intercept, r2


def rs_analysis(
    x: Any,
    *,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> HurstResult:
    """Rescaled-range (R/S) Hurst estimate.  ``H`` is the log-log slope of R/S vs n."""
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, 4 * min_scale, "x")
    scales = _log_scales(arr.shape[0], min_scale, max_scale, n_scales)
    rs = rescaled_range(arr, scales)
    slope, intercept, r2 = _loglog_fit(scales, rs)
    return HurstResult(
        exponent=slope,
        intercept=intercept,
        r_squared=r2,
        method="rs",
        scales=scales,
        fluctuations=rs,
        n_obs=arr.shape[0],
    )


def dfa(
    x: Any,
    *,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> HurstResult:
    """
    Detrended fluctuation analysis.  ``exponent`` is the DFA scaling exponent α
    (= H for a stationary-noise input, H + 1 for an integrated series).
    """
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, 4 * min_scale, "x")
    profile = np.cumsum(arr - arr.mean())
    scales = _log_scales(arr.shape[0], min_scale, max_scale, n_scales)
    fluct = dfa_fluctuations(profile, scales)
    slope, intercept, r2 = _loglog_fit(scales, fluct)
    return HurstResult(
        exponent=slope,
        intercept=intercept,
        r_squared=r2,
        method="dfa",
        scales=scales,
        fluctuations=fluct,
        n_obs=arr.shape[0],
    )


def aggregated_variance(
    x: Any,
    *,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> HurstResult:
    """
    Aggregated-variance Hurst estimate.

    The variance of non-overlapping block means scales as ``m^{2H−2}``, so
    ``H = 1 + slope / 2`` where ``slope`` is the log-log fit of block-mean
    variance against block size ``m``.
    """
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, 4 * min_scale, "x")
    n = arr.shape[0]
    scales = _log_scales(n, min_scale, max_scale, n_scales)
    variances = np.empty(scales.shape[0], dtype=np.float64)
    for i, m in enumerate(scales):
        n_blocks = n // int(m)
        trimmed = arr[: n_blocks * int(m)].reshape(n_blocks, int(m))
        block_means = trimmed.mean(axis=1)
        variances[i] = float(np.var(block_means)) if n_blocks > 1 else np.nan
    slope, intercept, r2 = _loglog_fit(scales, variances)
    return HurstResult(
        exponent=1.0 + 0.5 * slope,
        intercept=intercept,
        r_squared=r2,
        method="agg_var",
        scales=scales,
        fluctuations=variances,
        n_obs=n,
    )


def hurst(
    x: Any,
    method: HurstMethod = "rs",
    *,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> HurstResult:
    """
    Estimate the Hurst exponent of ``x``.

    Parameters
    ----------
    x          Input series (``np.ndarray`` / ``pl.Series`` / ``pl.DataFrame``).
               Use the *increment* series (returns) for ``"rs"`` / ``"agg_var"``.
    method     ``"rs"`` (rescaled range), ``"dfa"`` (detrended fluctuation), or
               ``"agg_var"`` (aggregated variance).
    min_scale  Smallest window size.
    max_scale  Largest window size (default ``len(x) // 2``).
    n_scales   Number of log-spaced scales.

    Returns
    -------
    HurstResult
    """
    match method:
        case "rs":
            return rs_analysis(x, min_scale=min_scale, max_scale=max_scale, n_scales=n_scales)
        case "dfa":
            return dfa(x, min_scale=min_scale, max_scale=max_scale, n_scales=n_scales)
        case "agg_var":
            return aggregated_variance(
                x, min_scale=min_scale, max_scale=max_scale, n_scales=n_scales
            )
        case _:
            raise ValueError(f"method must be 'rs', 'dfa' or 'agg_var', got {method!r}.")


def fractal_dimension(
    x: Any,
    method: Literal["higuchi", "hurst"] = "higuchi",
    *,
    k_max: int | None = None,
) -> float:
    """
    Fractal dimension ``D`` of a curve, in ``[1, 2]``.

    ``method="higuchi"`` fits Higuchi's curve length ``L(k) ∝ k^{−D}`` directly;
    ``method="hurst"`` returns ``D = 2 − H`` from R/S analysis.  Higuchi is the
    more reliable estimator for short series.

    Parameters
    ----------
    x       Input series.
    method  ``"higuchi"`` or ``"hurst"``.
    k_max   Maximum lag for Higuchi (default ``min(len(x) // 4, 20)``).
    """
    arr = to_numpy_1d(x)
    validate_finite(arr)
    match method:
        case "higuchi":
            validate_min_length(arr, 16, "x")
            kk: int = min(arr.shape[0] // 4, 20) if k_max is None else k_max
            if kk < 3:
                raise ValueError(f"k_max must be >= 3, got {kk}.")
            lengths = higuchi_lengths(arr, kk)
            ks = np.arange(1, kk + 1, dtype=np.int64)
            slope, _, _ = _loglog_fit(ks, lengths)
            return float(-slope)
        case "hurst":
            return 2.0 - rs_analysis(arr).exponent
        case _:
            raise ValueError(f"method must be 'higuchi' or 'hurst', got {method!r}.")


@dataclass(slots=True, frozen=True)
class MFDFAResult:
    """
    Multifractal DFA spectrum (Kantelhardt et al. 2002).

    Attributes
    ----------
    q             Moment orders, shape ``(nq,)``.
    hq            Generalised Hurst exponent ``h(q)``, shape ``(nq,)``.  ``h(2)``
                  is the ordinary DFA exponent.  A flat ``h(q)`` ⇒ monofractal.
    tau           Mass / Rényi exponent ``τ(q) = q·h(q) − 1``.
    alpha         Singularity strength ``α = dτ/dq``.
    f_alpha       Singularity spectrum ``f(α) = q·α − τ(q)``.
    scales        Window sizes used, shape ``(ns,)``.
    fluctuations  ``F_q(s)`` matrix, shape ``(nq, ns)``.
    n_obs         Length of the input series.
    """

    q: np.ndarray
    hq: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray
    f_alpha: np.ndarray
    scales: np.ndarray
    fluctuations: np.ndarray
    n_obs: int

    @property
    def width(self) -> float:
        """Singularity-spectrum width ``max α − min α`` — the degree of multifractality."""
        return float(self.alpha.max() - self.alpha.min())

    def __str__(self) -> str:
        return (
            f"MFDFA(nq={self.q.shape[0]}, h(2)≈{float(np.interp(2.0, self.q, self.hq)):.3f}, "
            f"width={self.width:.3f}, n={self.n_obs})"
        )


def mfdfa(
    x: Any,
    *,
    q_values: np.ndarray | None = None,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> MFDFAResult:
    """
    Multifractal detrended fluctuation analysis.

    Computes the q-dependent fluctuation function

        F_q(s) = { (1/N_s) Σ_v [F²(v, s)]^{q/2} }^{1/q}   (q ≠ 0)

    over a range of scales ``s`` (order-1 detrending), with the usual
    logarithmic average for ``q = 0``.  The generalised Hurst exponent
    ``h(q)`` is the slope of ``log F_q(s)`` against ``log s``; the Legendre
    transform yields the singularity spectrum ``f(α)``.

    Parameters
    ----------
    x          Input series.  As with DFA, run on the increment series.
    q_values   Moment orders (default ``linspace(-5, 5, 17)``, which includes
               ``q = 2``).
    min_scale  Smallest window size.
    max_scale  Largest window size (default ``len(x) // 4``).
    n_scales   Number of log-spaced scales.

    Returns
    -------
    MFDFAResult
    """
    arr = to_numpy_1d(x)
    validate_finite(arr)
    validate_min_length(arr, 4 * min_scale, "x")
    q = np.linspace(-5.0, 5.0, 17) if q_values is None else np.asarray(q_values, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] < 2:
        raise ValueError("q_values must be a 1-D array with at least 2 entries.")

    upper = arr.shape[0] // 4 if max_scale is None else max_scale
    scales = _log_scales(arr.shape[0], min_scale, upper, n_scales)
    profile = np.cumsum(arr - arr.mean())

    fluct = np.empty((q.shape[0], scales.shape[0]), dtype=np.float64)
    for si, s in enumerate(scales):
        seg_var = np.maximum(detrended_seg_vars(profile, int(s)), 1e-300)
        for qi, qq in enumerate(q):
            if abs(qq) < 1e-6:
                fluct[qi, si] = np.exp(0.25 * np.mean(np.log(seg_var)))
            else:
                fluct[qi, si] = np.mean(seg_var ** (qq / 2.0)) ** (1.0 / qq)

    hq = np.array([_loglog_fit(scales, fluct[qi])[0] for qi in range(q.shape[0])])
    tau = q * hq - 1.0
    alpha = np.gradient(tau, q)
    f_alpha = q * alpha - tau
    return MFDFAResult(
        q=q,
        hq=hq,
        tau=tau,
        alpha=alpha,
        f_alpha=f_alpha,
        scales=scales,
        fluctuations=fluct,
        n_obs=arr.shape[0],
    )
