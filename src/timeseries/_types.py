"""
Shared types and result containers for the timeseries subpackage.

Protocols
---------
``HasInfoCriteria`` and ``HasResiduals`` describe cross-cutting fit-result
accessors without forcing a class hierarchy.  Fit-result dataclasses (defined
per model) satisfy these structurally — no inheritance required.

Shared dataclasses
------------------
``ForecastResult`` is the uniform return type of every model's
``forecast(h, *, alpha, n_paths, seed)`` method.

``BacktestEvalResult`` is returned by the rolling backtest utilities in
``forecast_eval`` (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl


class HasInfoCriteria(Protocol):
    """Any fit result that exposes information criteria for model comparison."""

    log_lik: float
    aic: float
    bic: float
    n_obs: int


class HasResiduals(Protocol):
    """Any fit result that exposes residuals and fitted values."""

    residuals: np.ndarray
    fitted_values: np.ndarray


@dataclass
class ForecastResult:
    """
    Uniform forecast output.

    ``mean`` is always populated.  ``lower`` / ``upper`` are populated when
    a prediction interval was requested (``alpha is not None``).  ``paths``
    is populated when Monte Carlo sample paths were requested
    (``n_paths is not None``).

    Attributes
    ----------
    mean        Point forecasts, shape (h,)
    horizon     Forecast horizon h
    alpha       Two-sided coverage level (None if no interval was requested)
    lower       Lower interval bound, shape (h,) or None
    upper       Upper interval bound, shape (h,) or None
    paths       Simulated sample paths, shape (n_paths, h) or None
    """

    mean: np.ndarray
    horizon: int
    alpha: float | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    paths: np.ndarray | None = None

    def to_dataframe(self) -> pl.DataFrame:
        """Return a polars DataFrame with one row per forecast step."""
        cols: dict[str, np.ndarray] = {
            "h": np.arange(1, self.horizon + 1, dtype=np.int64),
            "mean": self.mean,
        }
        if self.lower is not None:
            cols["lower"] = self.lower
        if self.upper is not None:
            cols["upper"] = self.upper
        return pl.DataFrame(cols)


@dataclass
class BacktestEvalResult:
    """
    Rolling / expanding backtest evaluation summary.

    Attributes
    ----------
    forecasts   Stacked point forecasts, shape (n_windows, h)
    actuals     Realised values aligned with each forecast, shape (n_windows, h)
    errors      ``actuals - forecasts``, shape (n_windows, h)
    rmse        Root-mean-square error across all (window, horizon) pairs
    mape        Mean absolute percentage error
    mase        Mean absolute scaled error (None if naive scale is undefined)
    n_windows   Number of evaluation windows
    horizon     Forecast horizon h
    """

    forecasts: np.ndarray
    actuals: np.ndarray
    errors: np.ndarray
    rmse: float
    mape: float
    mase: float | None
    n_windows: int
    horizon: int

    def to_dataframe(self) -> pl.DataFrame:
        """Return a long-format DataFrame with columns (window, h, forecast, actual, error)."""
        n, h = self.forecasts.shape
        win = np.repeat(np.arange(n, dtype=np.int64), h)
        step = np.tile(np.arange(1, h + 1, dtype=np.int64), n)
        return pl.DataFrame(
            {
                "window": win,
                "h": step,
                "forecast": self.forecasts.ravel(),
                "actual": self.actuals.ravel(),
                "error": self.errors.ravel(),
            }
        )
