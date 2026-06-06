"""
Realized volatility from high-frequency returns, and the HAR-RV model.

* ``realized_variance`` / ``realized_volatility`` — the sum of squared intraday
  returns, the model-free estimator of integrated variance.
* ``bipower_variation`` — Barndorff-Nielsen & Shephard's (2004) jump-robust
  variation; the gap ``RV − BV`` estimates the jump contribution.
* ``HARRV`` — Corsi's (2009) Heterogeneous AutoRegressive model: regress daily
  realized variance on its own daily, weekly and monthly averages.  Despite its
  simplicity it forecasts volatility competitively with GARCH and is the
  standard benchmark for realized-volatility forecasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ._io import to_numpy_1d, validate_finite, validate_min_length
from ._types import ForecastResult

_MU1 = math.sqrt(2.0 / math.pi)  # E|Z| for standard normal


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cum = np.cumsum(x)
    out[window - 1] = cum[window - 1]
    out[window:] = cum[window:] - cum[:-window]
    return out


def realized_variance(returns: Any, *, window: int | None = None) -> float | np.ndarray:
    """
    Realized variance — the sum of squared returns.

    With ``window=None`` returns the whole-sample scalar ``Σ r²``; otherwise a
    trailing rolling sum over ``window`` observations (leading NaNs).
    """
    r = to_numpy_1d(returns)
    validate_finite(r)
    sq = r * r
    if window is None:
        return float(np.sum(sq))
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    return _rolling_sum(sq, window)


def realized_volatility(
    returns: Any,
    *,
    window: int | None = None,
    annualize: bool = False,
    periods: float = 252.0,
) -> float | np.ndarray:
    """
    Realized volatility — the square root of :func:`realized_variance`.

    When ``annualize`` is True the *per-observation* volatility is scaled by
    ``√periods`` (use ``window`` = bars-per-day and ``periods`` = trading days
    for an annual figure).
    """
    rv = realized_variance(returns, window=window)
    scale = math.sqrt(periods) if annualize else 1.0
    if isinstance(rv, float):
        if window is None:
            # Whole-sample: report the per-observation vol so annualisation is meaningful.
            r = to_numpy_1d(returns)
            return float(math.sqrt(rv / r.shape[0]) * scale)
        return rv
    return np.sqrt(rv) * scale


def bipower_variation(returns: Any, *, window: int | None = None) -> float | np.ndarray:
    """
    Realized bipower variation ``(π/2) · Σ |r_t| |r_{t−1}|`` (jump-robust).

    The difference ``realized_variance − bipower_variation`` estimates the
    contribution of jumps to total variation.
    """
    r = to_numpy_1d(returns)
    validate_finite(r)
    if r.shape[0] < 2:
        raise ValueError("need at least 2 returns.")
    prod = np.abs(r[1:]) * np.abs(r[:-1]) / (_MU1 * _MU1)
    if window is None:
        return float(np.sum(prod))
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    out = np.full(r.shape[0], np.nan, dtype=np.float64)
    out[1:] = _rolling_sum(prod, window)
    return out


@dataclass(slots=True)
class HARRVResult:
    """Fitted HAR-RV model with multi-step forecasting."""

    beta: np.ndarray  # [intercept, daily, weekly, monthly]
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    log_lik: float
    aic: float
    bic: float
    n_obs: int
    _history: np.ndarray  # trailing realized-variance series for forecasting
    _lags: tuple[int, int, int]

    def forecast(self, h: int = 1, *, alpha: float | None = None) -> ForecastResult:
        """
        Iterated multi-step variance forecast.

        Each step predicts the next daily RV from the current daily/weekly/
        monthly averages, then rolls the prediction into the history.  Forecasts
        are floored at 0.  When ``alpha`` is given, a (approximate) normal
        prediction interval is built from the one-step residual standard
        deviation.
        """
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}.")
        ld, lw, lm = self._lags
        buf = list(self._history)
        mean = np.empty(h, dtype=np.float64)
        for i in range(h):
            rv_d = buf[-ld] if ld == 1 else float(np.mean(buf[-ld:]))
            rv_w = float(np.mean(buf[-lw:]))
            rv_m = float(np.mean(buf[-lm:]))
            pred = float(
                self.beta[0] + self.beta[1] * rv_d + self.beta[2] * rv_w + self.beta[3] * rv_m
            )
            pred = max(pred, 0.0)
            mean[i] = pred
            buf.append(pred)

        if alpha is None:
            return ForecastResult(mean=mean, horizon=h)
        sigma = float(np.std(self.residuals))
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        band = z * sigma
        return ForecastResult(
            mean=mean,
            horizon=h,
            alpha=alpha,
            lower=np.maximum(mean - band, 0.0),
            upper=mean + band,
        )


class HARRV:
    """
    Corsi (2009) Heterogeneous AutoRegressive realized-volatility model.

        RV_t = β₀ + β_d RV_{t−1} + β_w \\overline{RV}_{t−1}^{(w)}
                  + β_m \\overline{RV}_{t−1}^{(m)} + ε_t

    where the weekly / monthly regressors are trailing averages of past daily
    realized variances.

    Parameters
    ----------
    daily, weekly, monthly
        Averaging horizons in bars (defaults 1 / 5 / 22, the standard
        daily / weekly / monthly cascade).
    """

    def __init__(self, daily: int = 1, weekly: int = 5, monthly: int = 22) -> None:
        if not (0 < daily < weekly < monthly):
            raise ValueError("require 0 < daily < weekly < monthly.")
        self.daily = daily
        self.weekly = weekly
        self.monthly = monthly

    def fit(self, rv: Any) -> HARRVResult:
        """Fit by OLS to a series of (daily) realized variances ``rv``."""
        series = to_numpy_1d(rv)
        validate_finite(series)
        validate_min_length(series, self.monthly + 2, "rv")
        n = series.shape[0]

        rows: list[list[float]] = []
        target: list[float] = []
        for t in range(self.monthly, n):
            rv_d = (
                series[t - self.daily]
                if self.daily == 1
                else float(series[t - self.daily : t].mean())
            )
            rv_w = float(series[t - self.weekly : t].mean())
            rv_m = float(series[t - self.monthly : t].mean())
            rows.append([1.0, rv_d, rv_w, rv_m])
            target.append(float(series[t]))

        design = np.asarray(rows, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        resid = y - fitted
        m = y.shape[0]
        rss = float(resid @ resid)
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
        log_lik = -0.5 * m * (math.log(2.0 * math.pi * rss / m) + 1.0) if rss > 0.0 else math.inf
        k = design.shape[1]
        aic = -2.0 * log_lik + 2.0 * k
        bic = -2.0 * log_lik + k * math.log(m)
        return HARRVResult(
            beta=beta,
            fitted_values=fitted,
            residuals=resid,
            r_squared=r2,
            log_lik=log_lik,
            aic=aic,
            bic=bic,
            n_obs=m,
            _history=series.copy(),
            _lags=(self.daily, self.weekly, self.monthly),
        )
