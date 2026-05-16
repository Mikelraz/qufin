# ruff: noqa: N806  — econometric sample-size variable T
"""
Forecast-evaluation framework.

Components
----------
* ``RollingBacktest``  — generic walk-forward / expanding-window evaluator.
* ``diebold_mariano``  — DM test comparing two forecast-error series.
* ``rmse`` / ``mae`` / ``mape`` / ``mase`` — standard accuracy metrics.
* ``crps``  — continuous ranked probability score (empirical-CDF form).

Model protocol expected by ``RollingBacktest``:

    class Model:
        def fit(self, y: np.ndarray) -> Any: ...
        def forecast(self, h: int) -> Any  # numpy array, ForecastResult, or scalar

The evaluator calls ``model.fit(train_y)`` then ``model.forecast(h)`` and
takes the first ``h`` values from the returned object.  If the result has a
``.mean`` attribute (e.g. the ``ForecastResult`` dataclass) that is used;
otherwise it is converted to a numpy array directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl
import scipy.stats
from numba import njit

from ._io import to_numpy_1d, validate_finite, validate_min_length
from ._types import BacktestEvalResult

# ---------------------------------------------------------------------------
# Model protocol
# ---------------------------------------------------------------------------


class ForecastModel(Protocol):
    """Minimal contract for models used by ``RollingBacktest``."""

    def fit(self, y: np.ndarray, /) -> Any: ...
    def forecast(self, h: int, /) -> Any: ...


def _extract_forecast(out: Any, h: int) -> np.ndarray:
    """Coerce a heterogeneous forecast output into a 1-D numpy array of length ``h``."""
    if hasattr(out, "mean") and not isinstance(out, np.ndarray):
        # ForecastResult or similar
        arr = np.asarray(out.mean, dtype=float).ravel()
    elif np.isscalar(out):
        arr = np.array([float(out)] * h)
    else:
        arr = np.asarray(out, dtype=float).ravel()
    if arr.shape[0] < h:
        raise ValueError(f"Model forecast returned {arr.shape[0]} values, expected ≥ {h}.")
    return arr[:h]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean square error."""
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, forecast {forecast.shape}.")
    err = actual - forecast
    return float(math.sqrt(np.mean(err * err)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, forecast {forecast.shape}.")
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error.  Skips entries with |actual| < 1e-12."""
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, forecast {forecast.shape}.")
    mask = np.abs(actual) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])))


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    training: np.ndarray,
    *,
    seasonality: int = 1,
) -> float | None:
    """Mean absolute scaled error (Hyndman & Koehler 2006).

    Scaled by the mean absolute error of an in-sample naive forecast
    (seasonal naive when ``seasonality > 1``).  Returns ``None`` if the
    scaling denominator is zero (constant training series).
    """
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    training = np.asarray(training, dtype=float).ravel()
    if seasonality < 1:
        raise ValueError(f"seasonality must be ≥ 1, got {seasonality}.")
    if training.shape[0] <= seasonality:
        raise ValueError(
            f"training must have length > seasonality; got {training.shape[0]} ≤ {seasonality}."
        )
    naive_diff = training[seasonality:] - training[:-seasonality]
    scale = float(np.mean(np.abs(naive_diff)))
    if scale == 0.0:
        return None
    return float(np.mean(np.abs(actual - forecast)) / scale)


@njit(cache=True)
def _crps_ecdf_kernel(obs: float, sample: np.ndarray) -> float:
    """Empirical-CDF CRPS for a single observation against ``sample``."""
    n = sample.shape[0]
    if n == 0:
        return 0.0
    # CRPS = (1/n) Σ |s_i − obs| − (1/(2 n²)) ΣΣ |s_i − s_j|
    abs_sum = 0.0
    for i in range(n):
        abs_sum += abs(sample[i] - obs)
    pairwise = 0.0
    for i in range(n):
        for j in range(n):
            pairwise += abs(sample[i] - sample[j])
    return abs_sum / n - pairwise / (2.0 * n * n)


def crps(observation: float | np.ndarray, sample: np.ndarray) -> float:
    """Continuous ranked probability score via the empirical-CDF formula.

    Parameters
    ----------
    observation  Scalar observed value, or 1-D array of multiple observations
                 (the average CRPS is returned).
    sample       Forecast ensemble, shape (n_paths,) — corresponds to the
                 single observation.  For multiple observations, pass a 2-D
                 array of shape ``(n_observations, n_paths)``.
    """
    if np.isscalar(observation):
        s = np.ascontiguousarray(sample, dtype=np.float64).ravel()
        return float(_crps_ecdf_kernel(float(observation), s))
    obs_arr = np.asarray(observation, dtype=float).ravel()
    sample_arr = np.asarray(sample, dtype=float)
    if sample_arr.ndim == 1:
        # Broadcast: same sample for every observation
        s = np.ascontiguousarray(sample_arr)
        total = 0.0
        for o in obs_arr:
            total += _crps_ecdf_kernel(float(o), s)
        return float(total / obs_arr.shape[0])
    if sample_arr.shape[0] != obs_arr.shape[0]:
        raise ValueError(
            f"sample first dim {sample_arr.shape[0]} ≠ observations {obs_arr.shape[0]}."
        )
    total = 0.0
    for i in range(obs_arr.shape[0]):
        s = np.ascontiguousarray(sample_arr[i])
        total += _crps_ecdf_kernel(float(obs_arr[i]), s)
    return float(total / obs_arr.shape[0])


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    """Diebold-Mariano (1995) test result.

    Null hypothesis: the two competing forecasts have equal predictive
    accuracy under the chosen loss function.  ``stat > 0`` means model A
    has higher loss (worse) than model B; ``stat < 0`` means B is worse.

    Attributes
    ----------
    stat       DM test statistic (Newey-West heteroskedasticity-robust).
    p_value    Two-sided p-value under the asymptotic standard-normal
               approximation.
    n_obs      Number of paired forecast errors.
    h          Forecast horizon (used to set the lag truncation).
    loss       Loss function applied to each error (e.g. ``"squared"``).
    """

    stat: float
    p_value: float
    n_obs: int
    h: int
    loss: str


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    h: int = 1,
    loss: str = "squared",
) -> DMResult:
    """Diebold-Mariano (1995) test for equal forecast accuracy.

    Parameters
    ----------
    errors_a, errors_b   Forecast-error series of equal length.
    h                    Forecast horizon (controls the Newey-West lag).
    loss                 One of ``"squared"`` or ``"absolute"``.
    """
    if loss not in ("squared", "absolute"):
        raise ValueError(f"loss must be 'squared' or 'absolute'; got {loss!r}.")
    a = to_numpy_1d(errors_a)
    b = to_numpy_1d(errors_b)
    validate_finite(a, "errors_a")
    validate_finite(b, "errors_b")
    if a.shape != b.shape:
        raise ValueError(f"errors_a {a.shape} ≠ errors_b {b.shape}.")
    n = a.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 observations; got {n}.")

    if loss == "squared":
        loss_a = a * a
        loss_b = b * b
    else:
        loss_a = np.abs(a)
        loss_b = np.abs(b)
    d = loss_a - loss_b
    d_mean = float(d.mean())

    # Newey-West long-run variance with Bartlett kernel, lag = h − 1.
    d_centred = d - d_mean
    gamma0 = float(np.dot(d_centred, d_centred) / n)
    lr_var = gamma0
    for k in range(1, h):
        w = 1.0 - k / h
        gamma_k = float(np.dot(d_centred[:-k], d_centred[k:]) / n)
        lr_var += 2.0 * w * gamma_k
    if lr_var <= 0.0:
        lr_var = gamma0
    se = math.sqrt(lr_var / n) if lr_var > 0 else math.nan
    stat = d_mean / se if se > 0 else math.nan
    p_value = float(2.0 * scipy.stats.norm.sf(abs(stat))) if math.isfinite(stat) else math.nan
    return DMResult(stat=float(stat), p_value=p_value, n_obs=n, h=h, loss=loss)


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------


class RollingBacktest:
    """Walk-forward / expanding-window backtest.

    Parameters
    ----------
    model_factory : Callable[[], ForecastModel]
        Zero-argument callable returning a *fresh* model instance.  A new
        instance is created for every refit so model state cannot leak
        between windows.
    window : int
        Length of the training window.  Use ``window = None`` for an
        expanding window (the training set grows from the start).
    refit_every : int
        Number of forecast steps between refits.  ``1`` means refit every
        period; larger values keep parameters frozen across multiple
        forecasts to amortise the fit cost.
    h : int
        Forecast horizon at each step.

    Notes
    -----
    The expected backtest length is the number of complete forecast windows:
    ``n_windows = (len(y) − window − h + 1) // refit_every + 1`` (or similar
    for expanding-window mode).
    """

    def __init__(
        self,
        model_factory: Any,
        *,
        window: int | None = 250,
        refit_every: int = 1,
        h: int = 1,
    ) -> None:
        if window is not None and window < 5:
            raise ValueError(f"window must be ≥ 5 or None; got {window}.")
        if refit_every < 1:
            raise ValueError(f"refit_every must be ≥ 1; got {refit_every}.")
        if h < 1:
            raise ValueError(f"h must be ≥ 1; got {h}.")
        self.model_factory = model_factory
        self.window = window
        self.refit_every = refit_every
        self.h = h

    def run(self, y: np.ndarray) -> BacktestEvalResult:
        """Execute the backtest and return a ``BacktestEvalResult``."""
        arr = to_numpy_1d(y)
        validate_finite(arr)
        min_required = (self.window or 10) + self.h
        validate_min_length(arr, min_required, "y")
        T = arr.shape[0]

        # Build the set of forecast-origin indices.
        if self.window is None:
            # Expanding window: start at index 10, step by refit_every.
            start_origins = list(range(10, T - self.h + 1, self.refit_every))
        else:
            start_origins = list(range(self.window, T - self.h + 1, self.refit_every))

        forecasts: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        cached_model: ForecastModel | None = None
        cached_train_end: int = -1
        for step_idx, origin in enumerate(start_origins):
            # Refit if needed
            if step_idx % 1 == 0:  # refit_every already controlled by stride
                if self.window is None:
                    train = arr[:origin]
                else:
                    train = arr[origin - self.window : origin]
                cached_model = self.model_factory()
                cached_model.fit(train)
                cached_train_end = origin
            assert cached_model is not None  # for type-checker

            # Forecast and record
            out = cached_model.forecast(self.h)
            f = _extract_forecast(out, self.h)
            actual = arr[origin : origin + self.h]
            forecasts.append(f)
            actuals.append(actual)
            _ = cached_train_end  # silence linter; reserved for future caching reuse

        f_mat = np.asarray(forecasts)
        a_mat = np.asarray(actuals)
        errs = a_mat - f_mat
        rmse_val = float(math.sqrt(np.mean(errs * errs)))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_val = float(np.mean(np.abs(errs[a_mat != 0] / a_mat[a_mat != 0])))
        mase_val: float | None = None
        # MASE: compare against in-sample naive
        if self.window is not None and self.window > 1:
            scale_train = arr[: self.window]
            naive_diff = scale_train[1:] - scale_train[:-1]
            naive_scale = float(np.mean(np.abs(naive_diff)))
            if naive_scale > 0:
                mase_val = float(np.mean(np.abs(errs)) / naive_scale)

        return BacktestEvalResult(
            forecasts=f_mat,
            actuals=a_mat,
            errors=errs,
            rmse=rmse_val,
            mape=mape_val if math.isfinite(mape_val) else float("nan"),
            mase=mase_val,
            n_windows=f_mat.shape[0],
            horizon=self.h,
        )

    @staticmethod
    def to_dataframe(result: BacktestEvalResult) -> pl.DataFrame:
        """Long-format DataFrame of (window, h, forecast, actual, error)."""
        return result.to_dataframe()
