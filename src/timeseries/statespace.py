"""
State-space wrapper exposing ARMA / ARIMA via the existing KalmanFilter.

This module provides a thin convenience layer:

    ARMAStateSpace(p, q).fit(y) → SmootherResult

The ``fit`` method builds the Hamilton state-space form for ARMA(p, q),
runs the Kalman filter forward pass, applies the RTS smoother, and returns
the smoothed state means and covariances together with the innovation
sequence.  It is useful when the caller wants filtered / smoothed hidden
states (e.g., reconstructing unobserved MA innovations), not just point
forecasts.

Relation to ``arima.ARMA``
--------------------------
``arima.ARMA`` focuses on parameter estimation (MLE / CSS) and forecasting.
``ARMAStateSpace`` assumes the model parameters are *given* and provides
direct access to the Kalman filter internals (smoothed states, Kalman gains,
innovations).  The two classes are complementary: a typical workflow is
to estimate parameters with ``ARMA.fit(method='mle')`` and then call
``ARMAStateSpace.from_result(fit_result).smooth(y)`` to recover the latent
states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..filters.kalman import FilterResult, KalmanFilter, SmootherResult
from ._io import to_numpy_1d, validate_min_length
from .arima import ARMAFitResult, _arma_state_space, _p0_lyapunov

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StateSpaceResult:
    """
    Combined output of a Kalman filter + RTS smoother pass.

    Attributes
    ----------
    filter_result   : Full forward-pass output (states, innovations, gains, …)
    smoother_result : Full backward-pass output (smoothed states, gains, …)
    ar_order        : p
    ma_order        : q
    state_dim       : r = max(p, q+1)
    log_likelihood  : Total log-likelihood from the forward pass
    """

    filter_result: FilterResult
    smoother_result: SmootherResult
    ar_order: int
    ma_order: int
    state_dim: int
    log_likelihood: float

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def filtered_states(self) -> np.ndarray:
        """Shape (T, r) — filtered state means."""
        return self.filter_result.states

    @property
    def smoothed_states(self) -> np.ndarray:
        """Shape (T, r) — smoothed state means."""
        return self.smoother_result.states

    @property
    def innovations(self) -> np.ndarray:
        """Shape (T, 1) — one-step prediction errors (NaN for missing obs)."""
        return self.filter_result.innovations

    @property
    def filtered_obs(self) -> np.ndarray:
        """Shape (T,) — filtered observation means (H x_{t|t})."""
        return self.filter_result.states[:, 0]

    @property
    def smoothed_obs(self) -> np.ndarray:
        """Shape (T,) — smoothed observation means (H x_{t|T})."""
        return self.smoother_result.states[:, 0]

    def to_dataframe(self) -> pl.DataFrame:
        """
        Long-format DataFrame with columns:
        ``t``, ``filtered``, ``smoothed``, ``innovation``.
        """
        t_total = self.filter_result.states.shape[0]
        return pl.DataFrame(
            {
                "t": np.arange(t_total, dtype=np.int64),
                "filtered": self.filtered_obs,
                "smoothed": self.smoothed_obs,
                "innovation": self.filter_result.innovations[:, 0],
            }
        )


# ---------------------------------------------------------------------------
# ARMAStateSpace class
# ---------------------------------------------------------------------------


class ARMAStateSpace:
    """
    State-space representation of an ARMA(p, q) model.

    Given *fixed* model parameters (AR coefficients, MA coefficients,
    innovation variance), this class runs the Kalman filter and RTS smoother
    over an observation sequence.

    Parameters
    ----------
    p       : AR order
    q       : MA order
    ar_coef : AR coefficients φ_1, …, φ_p  (shape (p,))
    ma_coef : MA coefficients θ_1, …, θ_q  (shape (q,))
    sigma2  : Innovation variance
    const   : Unconditional mean (subtracted before filtering, added back after)
    """

    def __init__(
        self,
        p: int,
        q: int,
        ar_coef: np.ndarray,
        ma_coef: np.ndarray,
        sigma2: float,
        const: float = 0.0,
    ) -> None:
        if p < 0 or q < 0:
            raise ValueError(f"p and q must be ≥ 0, got p={p}, q={q}.")
        if p + q == 0:
            raise ValueError("At least one of p or q must be > 0.")
        if sigma2 <= 0:
            raise ValueError(f"sigma2 must be > 0, got {sigma2}.")

        self.p = p
        self.q = q
        self.ar_coef = np.asarray(ar_coef, dtype=float)
        self.ma_coef = np.asarray(ma_coef, dtype=float)
        self.sigma2 = float(sigma2)
        self.const = float(const)

        # Build state-space matrices once
        self._f, self._h, self._q_mat, self._r, self._r_dim = _arma_state_space(
            self.ar_coef, self.ma_coef, self.sigma2
        )
        self._p0 = _p0_lyapunov(self._f, self._q_mat, self._r_dim)
        self._x0 = np.zeros(self._r_dim)

        self._kf = KalmanFilter(
            F=self._f,
            H=self._h,
            Q=self._q_mat,
            R=self._r,
            x0=self._x0,
            P0=self._p0,
        )

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_result(cls, result: ARMAFitResult) -> ARMAStateSpace:
        """
        Construct an ``ARMAStateSpace`` from a fitted ``ARMAFitResult``.

        This is the typical pattern: estimate parameters with
        ``ARMA.fit(method='mle')`` then call ``ARMAStateSpace.from_result(r)``
        to get smoothed states.
        """
        return cls(
            p=result.ar_order,
            q=result.ma_order,
            ar_coef=result.ar_coef,
            ma_coef=result.ma_coef,
            sigma2=result.sigma2,
            const=result.const,
        )

    # ------------------------------------------------------------------
    # Filter and smooth
    # ------------------------------------------------------------------

    def filter(
        self,
        x: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> FilterResult:
        """
        Run the Kalman filter forward pass.

        Parameters
        ----------
        x   : observation sequence, shape (T,) or (T, 1).
              NaN values are treated as missing (predict step runs, update skipped).
        x0  : initial state vector, shape (r,).  Defaults to zeros.
        p0  : initial state covariance, shape (r, r).  Defaults to stationary P₀.

        Returns
        -------
        FilterResult
        """
        arr = to_numpy_1d(x)
        validate_min_length(arr, 2, "x")
        y = arr - self.const
        return self._kf.filter(y.reshape(-1, 1), x0=x0, P0=p0)

    def smooth(
        self,
        x: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> StateSpaceResult:
        """
        Run the Kalman filter + RTS smoother.

        Parameters
        ----------
        x   : observation sequence, shape (T,).
        x0  : optional initial state.
        p0  : optional initial covariance.

        Returns
        -------
        StateSpaceResult
        """
        filter_res = self.filter(x, x0=x0, p0=p0)
        smoother_res = self._kf.smooth(filter_res)
        return StateSpaceResult(
            filter_result=filter_res,
            smoother_result=smoother_res,
            ar_order=self.p,
            ma_order=self.q,
            state_dim=self._r_dim,
            log_likelihood=filter_res.log_likelihood,
        )

    def log_likelihood(self, x: np.ndarray) -> float:
        """Total log-likelihood of the observation sequence under the model."""
        return self.filter(x).log_likelihood

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """State dimension r = max(p, q+1)."""
        return self._r_dim

    @property
    def F(self) -> np.ndarray:  # noqa: N802
        """State transition matrix (r, r)."""
        return self._f.copy()

    @property
    def H(self) -> np.ndarray:  # noqa: N802
        """Observation matrix (1, r)."""
        return self._h.copy()

    @property
    def Q(self) -> np.ndarray:  # noqa: N802
        """Process noise covariance (r, r)."""
        return self._q_mat.copy()
