"""
Spread-construction and mean-reversion utilities.

Shared building blocks for pairs / statistical-arbitrage strategies:

* ``hedge_ratio``   — estimate the cointegrating coefficient β (OLS, total
  least squares, or a Kalman final estimate).
* ``spread``        — form the spread ``y − β x − α``.
* ``half_life``     — Ornstein-Uhlenbeck mean-reversion half-life from an
  AR(1) fit of the spread.
* ``zscore``        — full-sample standardisation.
* ``rolling_zscore``— causal trailing-window standardisation.

These centralise logic that previously lived inside individual strategies, so
:mod:`qufin.strategies.mean_reversion` and
:mod:`qufin.strategies.cointegration_pairs` share one implementation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from ..timeseries.models import HedgeRatioFilter
from ..utils import to_numpy_1d

HedgeMethod = Literal["ols", "tls", "kalman"]


def hedge_ratio(
    y: np.ndarray,
    x: np.ndarray,
    *,
    method: HedgeMethod = "ols",
    delta: float = 1e-4,
) -> float:
    """
    Estimate the hedge ratio β in ``y ≈ β x + α``.

    Parameters
    ----------
    y, x    Aligned 1-D series, length T.
    method  ``"ols"`` (regress y on x), ``"tls"`` (total least squares /
            orthogonal regression, symmetric in y and x), or ``"kalman"``
            (final filtered β from a random-walk :class:`HedgeRatioFilter`).
    delta   State-noise variance for the Kalman method.

    Returns
    -------
    float
        The (final) hedge ratio β.
    """
    yy, xx = to_numpy_1d(y), to_numpy_1d(x)
    if yy.shape != xx.shape:
        raise ValueError(f"y and x must align; got {yy.shape} vs {xx.shape}.")
    if yy.shape[0] < 3:
        raise ValueError("need at least 3 observations.")

    match method:
        case "ols":
            x_mean = float(xx.mean())
            sxx = float(np.sum((xx - x_mean) ** 2))
            if sxx <= 0.0:
                raise ValueError("x has zero variance.")
            return float(np.sum((xx - x_mean) * (yy - yy.mean())) / sxx)
        case "tls":
            # Orthogonal regression: smallest-eigenvalue direction of the
            # centred covariance matrix.
            yc = yy - yy.mean()
            xc = xx - xx.mean()
            cov = np.cov(np.vstack([xc, yc]))
            _, eigvecs = np.linalg.eigh(cov)
            vx, vy = eigvecs[:, -1]  # largest-eigenvalue (principal) direction
            if abs(vx) < 1e-12:
                raise ValueError("total least squares is degenerate (vertical fit).")
            return float(vy / vx)
        case "kalman":
            kf = HedgeRatioFilter(delta=delta, obs_var=float(np.var(yy)) or 1.0)
            res = kf.filter(yy, xx)
            return float(res["beta"][-1])
        case _:
            raise ValueError(f"method must be 'ols', 'tls' or 'kalman', got {method!r}.")


def spread(y: np.ndarray, x: np.ndarray, beta: float, alpha: float = 0.0) -> np.ndarray:
    """Form the spread ``y − β x − α``."""
    yy, xx = to_numpy_1d(y), to_numpy_1d(x)
    if yy.shape != xx.shape:
        raise ValueError(f"y and x must align; got {yy.shape} vs {xx.shape}.")
    return yy - beta * xx - alpha


def half_life(series: np.ndarray) -> float:
    """
    Ornstein-Uhlenbeck mean-reversion half-life of a (spread) series.

    Fits the AR(1) ``Δs_t = a + b · s_{t−1} + ε_t`` by OLS; the half-life is
    ``−ln(2) / b``.  Returns ``+inf`` when ``b ≥ 0`` (no mean reversion).
    """
    s = to_numpy_1d(series)
    if s.shape[0] < 3:
        raise ValueError("need at least 3 observations.")
    lagged = s[:-1]
    delta = np.diff(s)
    lag_mean = float(lagged.mean())
    var = float(np.sum((lagged - lag_mean) ** 2))
    if var <= 0.0:
        return math.inf
    b = float(np.sum((lagged - lag_mean) * (delta - delta.mean())) / var)
    if b >= 0.0:
        return math.inf
    return -math.log(2.0) / b


def zscore(x: np.ndarray) -> np.ndarray:
    """Full-sample z-score ``(x − mean) / std`` (zeros if std is 0)."""
    arr = to_numpy_1d(x)
    std = float(np.std(arr))
    if std <= 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def rolling_zscore(x: np.ndarray, window: int, *, min_periods: int = 5) -> np.ndarray:
    """
    Causal trailing-window z-score.

    ``out[t] = (x[t] − mean(x[t−window+1 : t+1])) / std(...)`` using only data
    up to and including ``t``.  NaNs in the input window are ignored; a window
    with fewer than ``min_periods`` finite points (or zero std) yields ``NaN``.
    """
    arr = to_numpy_1d(x)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        win = arr[lo : t + 1]
        valid = win[~np.isnan(win)]
        if valid.shape[0] < min_periods:
            continue
        std = float(np.std(valid))
        if std > 1e-12:
            out[t] = (arr[t] - float(np.mean(valid))) / std
    return out
