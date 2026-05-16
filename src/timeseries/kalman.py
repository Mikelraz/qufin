# ruff: noqa: N803, N806  — matrix variables use standard control-theory uppercase (F, H, Q, R, P, L, A, B, M)
"""
Linear Kalman Filter for financial time series.

State-space model
-----------------
    x_k = F @ x_{k-1} + B @ u_k + w_k,   w_k ~ N(0, Q)
    z_k = H @ x_k + v_k,                   v_k ~ N(0, R)

    x  : state vector,       shape (n,)
    z  : observation vector, shape (m,)
    u  : control vector,     shape (l,)   [optional]
    F  : state transition,   shape (n, n)
    H  : observation matrix, shape (m, n)
    Q  : process noise cov,  shape (n, n) [PSD]
    R  : measurement noise,  shape (m, m) [PD]
    B  : control matrix,     shape (n, l) [optional]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outputs of a forward Kalman filter pass over T observations."""

    states: np.ndarray            # (T, n)  filtered state mean
    covariances: np.ndarray       # (T, n, n)  filtered state covariance
    pred_states: np.ndarray       # (T, n)  one-step predicted state mean
    pred_covariances: np.ndarray  # (T, n, n)  one-step predicted covariance
    innovations: np.ndarray       # (T, m)  innovation sequence (NaN for missing obs)
    innovation_covs: np.ndarray   # (T, m, m)  innovation covariance
    gains: np.ndarray             # (T, n, m)  Kalman gain matrices
    log_likelihood: float         # total log-likelihood (missing obs skipped)


@dataclass
class SmootherResult:
    """Outputs of an RTS backward smoother pass."""

    states: np.ndarray       # (T, n)  smoothed state mean
    covariances: np.ndarray  # (T, n, n)  smoothed state covariance
    gains: np.ndarray        # (T-1, n, n)  RTS smoother gains
    log_likelihood: float    # same as the forward filter log-likelihood


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Linear Kalman Filter with RTS smoother for financial time series.

    The filter maintains a *mutable* state (x, P) that is updated in-place
    by each call to ``predict`` / ``update``.  Use ``filter()`` for batch
    processing; it accepts optional ``x0`` / ``P0`` to reinitialise cleanly
    without mutating the constructor defaults.

    Parameters
    ----------
    F : array_like, shape (n, n)
        State transition matrix.
    H : array_like, shape (m, n)
        Observation matrix.
    Q : array_like, shape (n, n)
        Process noise covariance (positive semi-definite).
    R : array_like, shape (m, m)
        Measurement noise covariance (positive definite).
    x0 : array_like, shape (n,)
        Initial state estimate.
    P0 : array_like, shape (n, n)
        Initial state covariance (positive semi-definite).
    B : array_like, shape (n, l), optional
        Control-input matrix.  Required if ``u`` is passed to ``predict``.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        B: np.ndarray | None = None,
    ) -> None:
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = np.asarray(B, dtype=float) if B is not None else None

        self.n: int = self.F.shape[0]
        self.m: int = self.H.shape[0]

        self._validate()

        # Working state — mutated by predict/update
        self.x: np.ndarray = np.asarray(x0, dtype=float).ravel()
        self.P: np.ndarray = np.asarray(P0, dtype=float)
        self._symmetrize_params()

        # Cached initial conditions so the user can reset cleanly
        self._x0: np.ndarray = self.x.copy()
        self._P0: np.ndarray = self.P.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        n, m = self.n, self.m
        if self.F.shape != (n, n):
            raise ValueError(f"F must be ({n},{n}), got {self.F.shape}")
        if self.H.shape != (m, n):
            raise ValueError(f"H must be ({m},{n}), got {self.H.shape}")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must be ({n},{n}), got {self.Q.shape}")
        if self.R.shape != (m, m):
            raise ValueError(f"R must be ({m},{m}), got {self.R.shape}")

    def _symmetrize_params(self) -> None:
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.R = 0.5 * (self.R + self.R.T)
        self.P = 0.5 * (self.P + self.P.T)

    @staticmethod
    def _sym(M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.T)

    @staticmethod
    def _solve_pd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve A @ X = B for positive-definite A via Cholesky."""
        try:
            L = np.linalg.cholesky(A)
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, B, rcond=None)[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def reset(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        """Reset state to constructor defaults (or supplied values)."""
        self.x = np.asarray(x0, dtype=float).ravel() if x0 is not None else self._x0.copy()
        self.P = np.asarray(P0, dtype=float) if P0 is not None else self._P0.copy()
        self.P = self._sym(self.P)

    def set_dynamics(
        self,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        B: np.ndarray | None = None,
    ) -> None:
        """
        Update any subset of model matrices in-place.

        Useful for iterative optimisation (e.g. ARMA MLE) where F and Q are
        rebuilt at every function evaluation — avoids re-instantiating the
        filter object on every call.

        Parameters
        ----------
        F : array_like, shape (n, n), optional
        H : array_like, shape (m, n), optional
        Q : array_like, shape (n, n), optional
        R : array_like, shape (m, m), optional
        B : array_like, shape (n, l), optional
        """
        if F is not None:
            self.F = np.asarray(F, dtype=float)
        if H is not None:
            self.H = np.asarray(H, dtype=float)
        if Q is not None:
            self.Q = self._sym(np.asarray(Q, dtype=float))
        if R is not None:
            self.R = self._sym(np.asarray(R, dtype=float))
        if B is not None:
            self.B = np.asarray(B, dtype=float)
        self._validate()

    # ------------------------------------------------------------------
    # Single-step interface
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Time-update (prediction) step.

        Parameters
        ----------
        u : array_like, shape (l,), optional
            Control input at this step.

        Returns
        -------
        x_pred : np.ndarray, shape (n,)
        P_pred : np.ndarray, shape (n, n)
        """
        x_pred = self.F @ self.x
        if u is not None:
            if self.B is None:
                raise ValueError("Control input u supplied but B is None.")
            x_pred = x_pred + self.B @ np.ravel(u)

        P_pred = self._sym(self.F @ self.P @ self.F.T + self.Q)

        self.x = x_pred
        self.P = P_pred
        return x_pred.copy(), P_pred.copy()

    def update(
        self, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Measurement-update step.

        Parameters
        ----------
        z : array_like, shape (m,)
            Observation vector.

        Returns
        -------
        x : filtered state mean
        P : filtered state covariance (Joseph form — numerically stable)
        innovation : z - H @ x_pred
        S : innovation covariance
        K : Kalman gain
        """
        z = np.asarray(z, dtype=float).ravel()
        innovation = z - self.H @ self.x

        PHt = self.P @ self.H.T                         # (n, m)
        S = self._sym(self.H @ PHt + self.R)             # (m, m)

        # Gain: K = P H^T S^{-1}  — solved via Cholesky for stability
        K = self._solve_pd(S.T, PHt.T).T                # (n, m)

        self.x = self.x + K @ innovation

        # Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        IKH = np.eye(self.n) - K @ self.H
        self.P = self._sym(IKH @ self.P @ IKH.T + K @ self.R @ K.T)

        return self.x.copy(), self.P.copy(), innovation, S, K

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def filter(
        self,
        observations: np.ndarray,
        controls: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> FilterResult:
        """
        Forward Kalman filter pass over a full observation sequence.

        Parameters
        ----------
        observations : array_like, shape (T,) or (T, m)
            Observation sequence.  NaN values are treated as missing
            observations — the predict step runs but the update is skipped.
        controls : array_like, shape (T, l), optional
            Control input sequence.
        x0 : array_like, shape (n,), optional
            Override the initial state for this pass (does not change
            constructor defaults stored in ``_x0``).
        P0 : array_like, shape (n, n), optional
            Override the initial covariance for this pass.

        Returns
        -------
        FilterResult
        """
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)         # (T, 1) for scalar observations
        T = obs.shape[0]

        self.reset(x0, P0)

        states      = np.empty((T, self.n))
        covs        = np.empty((T, self.n, self.n))
        pred_states = np.empty((T, self.n))
        pred_covs   = np.empty((T, self.n, self.n))
        innovations = np.full((T, self.m), np.nan)
        innov_covs  = np.full((T, self.m, self.m), np.nan)
        gains       = np.zeros((T, self.n, self.m))
        log_lik     = 0.0

        for t in range(T):
            u = controls[t] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            pred_states[t] = x_pred
            pred_covs[t]   = P_pred

            z = obs[t]
            if np.any(np.isnan(z)):
                # Missing observation: propagate prediction, skip update
                states[t] = x_pred
                covs[t]   = P_pred
            else:
                x_f, P_f, innov, S, K = self.update(z)
                states[t]      = x_f
                covs[t]        = P_f
                innovations[t] = innov
                innov_covs[t]  = S
                gains[t]       = K

                # Log-likelihood: -½(m ln 2π + ln|S| + ν^T S^{-1} ν)
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    log_lik += -0.5 * (
                        self.m * np.log(2.0 * np.pi)
                        + logdet
                        + innov @ self._solve_pd(S, innov)
                    )

        return FilterResult(
            states=states,
            covariances=covs,
            pred_states=pred_states,
            pred_covariances=pred_covs,
            innovations=innovations,
            innovation_covs=innov_covs,
            gains=gains,
            log_likelihood=log_lik,
        )

    def smooth(self, result: FilterResult) -> SmootherResult:
        """
        Rauch-Tung-Striebel (RTS) backward smoother.

        Runs a single backward pass over a ``FilterResult`` obtained from
        ``filter()``.  The smoother minimises the mean-squared error of the
        state estimates given *all* observations, not just the past ones.

        Parameters
        ----------
        result : FilterResult
            Must have been produced by ``filter()`` with the same model.

        Returns
        -------
        SmootherResult
        """
        T = result.states.shape[0]
        x_s = result.states.copy()      # (T, n)
        P_s = result.covariances.copy() # (T, n, n)
        G   = np.empty((T - 1, self.n, self.n))

        for t in range(T - 2, -1, -1):
            P_pred = result.pred_covariances[t + 1]  # P_{t+1|t}

            # Smoother gain: G_t = P_t F^T P_{t+1|t}^{-1}
            # Solve P_{t+1|t} G_t^T = (P_t F^T)^T  → G_t = solve(P_{t+1|t}, FP_t^T)^T
            FPt = self.F @ result.covariances[t]     # (n, n)
            G[t] = self._solve_pd(P_pred.T, FPt.T).T  # (n, n)

            dx = x_s[t + 1] - result.pred_states[t + 1]
            x_s[t] = x_s[t] + G[t] @ dx

            dP = P_s[t + 1] - P_pred
            P_s[t] = self._sym(P_s[t] + G[t] @ dP @ G[t].T)

        return SmootherResult(
            states=x_s,
            covariances=P_s,
            gains=G,
            log_likelihood=result.log_likelihood,
        )

    def log_likelihood(
        self,
        observations: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> float:
        """
        Total log-likelihood of an observation sequence under the current model.

        Useful for numerical optimisation of Q, R, or other hyperparameters.
        """
        return self.filter(observations, x0=x0, P0=P0).log_likelihood
