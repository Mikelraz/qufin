"""
Pre-built Kalman Filter models for common financial use cases.

HedgeRatioFilter
    Tracks a time-varying hedge ratio β and intercept α in the model
    y_t = β_t · x_t + α_t + ε_t using a random-walk state and a
    time-varying observation matrix.

TrendFilter
    Smooths price or return data with a constant-velocity (level + trend)
    state-space model.  Optionally applies the RTS backward smoother for
    offline (full-batch) optimal smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import FilterResult, KalmanFilter, SmootherResult

# ---------------------------------------------------------------------------
# HedgeRatioFilter
# ---------------------------------------------------------------------------

class HedgeRatioFilter:
    """
    Time-varying hedge ratio tracker.

    Models the relationship between two assets as:

        y_t = β_t · x_t + α_t + ε_t,    ε_t ~ N(0, obs_var)

    where the state [β_t, α_t] evolves as a *random walk*:

        [β_t, α_t]^T = [β_{t-1}, α_{t-1}]^T + w_t,   w_t ~ N(0, δ · I)

    The observation equation has a time-varying design row H_t = [x_t, 1],
    so this class manages the filter step-by-step, updating H before each
    measurement update.

    Parameters
    ----------
    delta : float
        State-noise variance (controls how fast β and α can drift).
        Smaller values impose more smoothness.  Typical range: 1e-5 – 1e-2.
    obs_var : float
        Variance of the observation noise ε_t.  Can be estimated from the
        residuals of an OLS regression as a starting point.
    x0 : array_like, shape (2,), optional
        Initial state [β_0, α_0].  Defaults to [1.0, 0.0].
    P0 : array_like, shape (2, 2), optional
        Initial state covariance.  Defaults to (1/δ) · I (diffuse / vague).
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_var: float = 1.0,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive.")
        if obs_var <= 0:
            raise ValueError("obs_var must be positive.")

        self.delta   = delta
        self.obs_var = obs_var

        F  = np.eye(2)
        H  = np.array([[1.0, 1.0]])  # placeholder; updated per observation
        Q  = delta * np.eye(2)
        R  = np.array([[obs_var]])

        _x0 = np.array([1.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float).ravel()
        _P0 = (1.0 / delta) * np.eye(2) if P0 is None else np.asarray(P0, dtype=float)

        self._kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=_x0, P0=_P0)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        """Current hedge-ratio estimate."""
        return float(self._kf.x[0])

    @property
    def alpha(self) -> float:
        """Current intercept estimate."""
        return float(self._kf.x[1])

    @property
    def beta_variance(self) -> float:
        """Posterior variance of the hedge-ratio estimate."""
        return float(self._kf.P[0, 0])

    @property
    def alpha_variance(self) -> float:
        """Posterior variance of the intercept estimate."""
        return float(self._kf.P[1, 1])

    @property
    def covariance(self) -> np.ndarray:
        return self._kf.P.copy()

    # ------------------------------------------------------------------
    # Single-step interface
    # ------------------------------------------------------------------

    def step(self, y: float, x: float) -> tuple[float, float, float]:
        """
        Process a single (y_t, x_t) observation.

        Parameters
        ----------
        y : float
            Observed level of the dependent asset.
        x : float
            Observed level of the independent (hedge) asset.

        Returns
        -------
        beta : float   Updated β estimate.
        alpha : float  Updated α estimate.
        spread : float Residual  y - β · x - α  (the innovation).
        """
        self._kf.H = np.array([[x, 1.0]])
        self._kf.predict()
        _, _, innov, _, _ = self._kf.update(np.array([y]))
        return self.beta, self.alpha, float(innov[0])

    def reset(
        self,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        """Reinitialise the filter to its initial conditions."""
        _x0 = self._kf._x0 if x0 is None else np.asarray(x0, dtype=float).ravel()
        _P0 = self._kf._P0 if P0 is None else np.asarray(P0, dtype=float)
        self._kf.x = _x0.copy()
        self._kf.P = _P0.copy()

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def filter(
        self,
        y: np.ndarray | pd.Series,
        x: np.ndarray | pd.Series,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Run the filter over full price/return series.

        Parameters
        ----------
        y, x : array_like, shape (T,)
            Dependent and independent asset series (prices or log-prices).
        x0 : optional initial state override for this run.
        P0 : optional initial covariance override for this run.

        Returns
        -------
        pd.DataFrame with columns:
            beta       – filtered hedge-ratio estimate
            alpha      – filtered intercept estimate
            spread     – residual  y - β·x - α  (the innovation)
            beta_std   – posterior standard deviation of β
            alpha_std  – posterior standard deviation of α
        """
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        if len(y_arr) != len(x_arr):
            raise ValueError("y and x must have the same length.")

        T = len(y_arr)
        self.reset(x0, P0)

        betas      = np.empty(T)
        alphas     = np.empty(T)
        spreads    = np.empty(T)
        beta_stds  = np.empty(T)
        alpha_stds = np.empty(T)

        for t in range(T):
            b, a, s    = self.step(float(y_arr[t]), float(x_arr[t]))
            betas[t]   = b
            alphas[t]  = a
            spreads[t] = s
            beta_stds[t]  = np.sqrt(max(self._kf.P[0, 0], 0.0))
            alpha_stds[t] = np.sqrt(max(self._kf.P[1, 1], 0.0))

        index = y.index if isinstance(y, pd.Series) else pd.RangeIndex(T)
        return pd.DataFrame(
            {
                "beta":       betas,
                "alpha":      alphas,
                "spread":     spreads,
                "beta_std":   beta_stds,
                "alpha_std":  alpha_stds,
            },
            index=index,
        )


# ---------------------------------------------------------------------------
# TrendFilter
# ---------------------------------------------------------------------------

class TrendFilter:
    """
    Constant-velocity Kalman smoother for price or return data.

    State vector: [level, velocity]

    Dynamics (continuous-time Wiener-process acceleration model, discretised):

        level_{t+1}    = level_t + dt · velocity_t + w1
        velocity_{t+1} = velocity_t + w2

    Observation:

        z_t = level_t + v_t,   v_t ~ N(0, obs_var)

    Parameters
    ----------
    process_var : float
        Integrated process noise variance.  Controls how quickly the
        estimated velocity can change.  Smaller → smoother output.
        Typical starting point: ratio of obs_var / expected_trend_lifetime².
    obs_var : float
        Observation noise variance.
    dt : float
        Time step between observations (default 1.0 for daily data).
    """

    def __init__(
        self,
        process_var: float = 1e-4,
        obs_var: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        if process_var <= 0:
            raise ValueError("process_var must be positive.")
        if obs_var <= 0:
            raise ValueError("obs_var must be positive.")

        self.process_var = process_var
        self.obs_var     = obs_var
        self.dt          = dt

        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])

        # Continuous white-noise acceleration model (Singer model):
        # Q = σ² · [[dt³/3, dt²/2], [dt²/2, dt]]
        Q = process_var * np.array(
            [
                [dt ** 3 / 3.0, dt ** 2 / 2.0],
                [dt ** 2 / 2.0, dt],
            ]
        )
        R  = np.array([[obs_var]])
        x0 = np.array([0.0, 0.0])
        P0 = np.diag([obs_var, process_var])

        self._kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def filter(
        self,
        prices: np.ndarray | pd.Series,
        smooth: bool = False,
    ) -> pd.DataFrame:
        """
        Filter (and optionally smooth) a price or return series.

        Parameters
        ----------
        prices : array_like, shape (T,)
            Observed prices or returns.  NaN values are treated as missing
            observations.
        smooth : bool
            If True, apply the RTS backward smoother after the forward pass.
            This yields the minimum-variance estimate given *all* data, at
            the cost of non-causality (future data affects past estimates).

        Returns
        -------
        pd.DataFrame with columns:
            level        – filtered (or smoothed) price level estimate
            velocity     – filtered (or smoothed) rate of change
            level_std    – posterior std of the level
            velocity_std – posterior std of the velocity
        """
        arr = np.asarray(prices, dtype=float)
        T   = len(arr)

        # Warm-start: initialise level from first non-NaN observation
        first_valid = arr[~np.isnan(arr)]
        init_level  = float(first_valid[0]) if len(first_valid) else 0.0
        x0 = np.array([init_level, 0.0])
        P0 = np.diag([self.obs_var, self.process_var])

        result: FilterResult | SmootherResult = self._kf.filter(
            arr.reshape(-1, 1), x0=x0, P0=P0
        )
        if smooth:
            result = self._kf.smooth(result)  # type: ignore[arg-type]

        levels   = result.states[:, 0]
        velocity = result.states[:, 1]
        lev_std  = np.sqrt(np.maximum(result.covariances[:, 0, 0], 0.0))
        vel_std  = np.sqrt(np.maximum(result.covariances[:, 1, 1], 0.0))

        index = prices.index if isinstance(prices, pd.Series) else pd.RangeIndex(T)
        return pd.DataFrame(
            {
                "level":        levels,
                "velocity":     velocity,
                "level_std":    lev_std,
                "velocity_std": vel_std,
            },
            index=index,
        )

    def log_likelihood(
        self,
        prices: np.ndarray | pd.Series,
    ) -> float:
        """Log-likelihood of the price series under the current model parameters."""
        arr = np.asarray(prices, dtype=float)
        first_valid = arr[~np.isnan(arr)]
        init_level  = float(first_valid[0]) if len(first_valid) else 0.0
        x0 = np.array([init_level, 0.0])
        P0 = np.diag([self.obs_var, self.process_var])
        return self._kf.log_likelihood(arr.reshape(-1, 1), x0=x0, P0=P0)
