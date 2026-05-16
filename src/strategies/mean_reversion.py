"""
Adaptive mean-reversion trading strategy.

Architecture
------------
The strategy couples an OU process model with a real-time Kalman Filter
to track the time-varying dynamics of any mean-reverting price series
(single asset level, residual spread, log price ratio, etc.).

Step 1 — Online state estimation
    A Kalman Filter treats the price series as an AR(1) process with
    *time-varying* coefficients [β_t, α_t] that evolve as a random walk:

        X_t  =  β_t · X_{t-1}  +  α_t  +  ε_t,     ε_t ~ N(0, obs_var)
        [β_t, α_t]  =  [β_{t-1}, α_{t-1}]  +  w_t,  w_t ~ N(0, δ·I₂)

    This is the HedgeRatioFilter from src.timeseries applied to the series
    against its own lag, giving the OU parameters in closed form:

        θ_t  = −ln(β_t) / Δt           (mean-reversion speed)
        μ_t  = α_t / (1 − β_t)          (long-run mean)

Step 2 — Stationary-std estimation
    The OU stationary std σ_eq,t is estimated from the rolling KF
    innovations ν_t = X_t − (β_{t-1}·X_{t-1} + α_{t-1}):

        σ_eq,t  =  std(ν_{t-W : t})  /  √(1 − β_t²)

    This identity follows from Var(ν) ≈ σ²_eq·(1−β²) for a stationary OU.

Step 3 — Signal
    The z-score z_t = (X_t − μ_t) / σ_eq,t drives threshold signals:

        z_t < −entry_z  →  long   (price below adaptive mean)
        z_t > +entry_z  →  short  (price above adaptive mean)
        |z_t| < exit_z  →  close  (mean reversion realised)
        |z_t| > stop_z  →  close  (regime break / stop-loss)

Trainable parameters
--------------------
    delta    : KF process noise — how quickly OU params can drift
    obs_var  : KF observation noise — AR(1) residual variance
    entry_z  : z-score entry threshold
    exit_z   : z-score exit threshold
    stop_z   : z-score stop-loss

All five are jointly optimised by ``fit(method='sharpe')`` to maximise
the annualised Sharpe ratio, or ``fit(method='likelihood')`` optimises
(delta, obs_var) only by maximising the exact KF log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from ..timeseries.kalman import KalmanFilter

_ANNUAL = np.sqrt(252)   # annualisation for daily Sharpe
_LOG_2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    """
    Hyperparameters of ``MeanReversionStrategy``.

    Attributes
    ----------
    delta : float
        Kalman Filter process noise for [β_t, α_t].  Controls how quickly
        the estimated OU parameters can drift.  Smaller → more stable
        estimates, slower adaptation.  (Trainable)
    obs_var : float
        KF observation noise — assumed variance of the AR(1) residual.
        Sets the Kalman gain: larger → trust observations more.  (Trainable)
    entry_z : float
        Enter a position when |z_t| ≥ entry_z.  (Trainable)
    exit_z : float
        Exit a position when the z-score crosses back within ±exit_z.
        Must satisfy 0 ≤ exit_z < entry_z.  (Trainable)
    stop_z : float
        Emergency close when |z_t| ≥ stop_z (regime-change guard).
        Must satisfy stop_z > entry_z.  (Trainable)
    vol_window : int
        Rolling window (bars) used to estimate σ_eq from innovations.
        Fixed at construction; not optimised by fit().
    dt : float
        Sampling interval in consistent time units (1.0 = one day).
        Fixed at construction; not optimised by fit().
    """

    delta: float = 1e-4
    obs_var: float = 1.0
    entry_z: float = 1.5
    exit_z: float = 0.5
    stop_z: float = 3.5
    vol_window: int = 60
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("delta must be > 0")
        if self.obs_var <= 0:
            raise ValueError("obs_var must be > 0")
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.exit_z < 0:
            raise ValueError("exit_z must be >= 0")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be > entry_z")
        if self.vol_window < 5:
            raise ValueError("vol_window must be >= 5")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "obs_var": self.obs_var,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "stop_z": self.stop_z,
        }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Output of ``MeanReversionStrategy.run()``.

    All arrays have length T (same as the input price series).
    ``log_returns`` has length T−1; index position t holds the return
    earned from bar t to bar t+1 while holding ``signal[t]``.
    """

    mu: np.ndarray           # (T,) filtered OU long-run mean
    theta: np.ndarray        # (T,) filtered mean-reversion speed θ_t
    half_life: np.ndarray    # (T,) ln(2) / θ_t
    sigma_eq: np.ndarray     # (T,) rolling estimate of stationary std
    z_score: np.ndarray      # (T,) (price − μ_t) / σ_eq,t
    signal: np.ndarray       # (T,) position ∈ {−1, 0, +1}
    log_returns: np.ndarray  # (T−1,) signal[t] × Δ log(price[t+1])
    prices: np.ndarray       # (T,) original prices
    index: object | None = None   # original pd.Index if input was a Series

    # ------------------------------------------------------------------

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio (daily data, 252 trading days)."""
        r = self.log_returns
        std = np.std(r)
        return float(np.mean(r) / max(std, 1e-10) * _ANNUAL)

    @property
    def total_return(self) -> float:
        """Sum of strategy log-returns over the period."""
        return float(np.nansum(self.log_returns))

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough log-return drawdown."""
        cum = np.nancumsum(self.log_returns)
        peak = np.maximum.accumulate(cum)
        return float(np.nanmin(cum - peak))

    @property
    def n_trades(self) -> int:
        """Number of round-trip trades (entry + exit pairs)."""
        pos = self.signal
        entries = np.sum((pos[1:] != 0) & (pos[:-1] == 0))
        return int(entries)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all series as a single aligned DataFrame."""
        T = len(self.prices)
        # strat_ret[t] = return *earned* at bar t (i.e. signal[t-1] × Δlog)
        strat_ret = np.concatenate([[np.nan], self.log_returns])
        idx = self.index if self.index is not None else pd.RangeIndex(T)
        return pd.DataFrame(
            {
                "price":     self.prices,
                "mu":        self.mu,
                "theta":     self.theta,
                "half_life": self.half_life,
                "sigma_eq":  self.sigma_eq,
                "z_score":   self.z_score,
                "signal":    self.signal,
                "strat_ret": strat_ret,
            },
            index=idx,
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "Backtest Summary",
                "=" * 35,
                f"  Sharpe (ann.) : {self.sharpe:.4f}",
                f"  Total return  : {self.total_return * 100:.2f}%",
                f"  Max drawdown  : {self.max_drawdown * 100:.2f}%",
                f"  Num trades    : {self.n_trades}",
                f"  Avg half-life : {np.nanmean(self.half_life):.2f}",
                f"  Avg |z_score| : {np.nanmean(np.abs(self.z_score)):.3f}",
            ]
        )


@dataclass
class TrainResult:
    """Output of ``MeanReversionStrategy.fit()``."""

    params: StrategyParams
    train_sharpe: float
    train_log_lik: float
    n_iter: int
    converged: bool
    method: str

    def __str__(self) -> str:
        p = self.params
        return "\n".join(
            [
                f"TrainResult  method={self.method}  converged={self.converged}  "
                f"iters={self.n_iter}",
                "=" * 55,
                f"  train Sharpe   : {self.train_sharpe:.4f}",
                f"  train log-lik  : {self.train_log_lik:.2f}",
                "  Fitted params:",
                f"    delta   = {p.delta:.2e}",
                f"    obs_var = {p.obs_var:.4f}",
                f"    entry_z = {p.entry_z:.3f}",
                f"    exit_z  = {p.exit_z:.3f}",
                f"    stop_z  = {p.stop_z:.3f}",
            ]
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """
    Adaptive OU mean-reversion strategy with a Kalman-filtered parameter update.

    See the module docstring for a full description of the architecture.

    Parameters
    ----------
    params : StrategyParams, optional
        Starting hyperparameters.  Default values are used when omitted.

    Examples
    --------
    Quick-start with default parameters::

        strategy = MeanReversionStrategy()
        result   = strategy.run(prices)
        print(result.summary())

    Fit to historical data before running::

        train_result = strategy.fit(prices, method='sharpe')
        result       = strategy.run(prices)

    Online (streaming) usage::

        strategy.reset()
        for price in live_feed:
            state = strategy.step(price)
            send_order(state['signal'])
    """

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params: StrategyParams = params if params is not None else StrategyParams()

        # Online-mode state (mutated by step())
        self._initialized: bool = False
        self._warmup_buffer: list[float] = []
        self._kf: KalmanFilter | None = None
        self._prev_price: float | None = None
        self._innovations: list[float] = []
        self._cur_pos: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ou_from_ar1(
        beta: float, alpha: float, dt: float
    ) -> tuple[float, float]:
        """Return (θ, μ) from AR(1) coefficients, with numerical guards."""
        b = float(np.clip(beta, 1e-9, 1.0 - 1e-9))
        theta = -np.log(b) / dt
        mu = alpha / (1.0 - b)
        return theta, mu

    @staticmethod
    def _sigma_eq(
        innovations: np.ndarray, beta: float
    ) -> float:
        """
        σ_eq from rolling innovations.

        Var(ν_t) ≈ σ²_eq · (1 − β²)  ⟹  σ_eq = std(ν) / √(1 − β²)
        """
        b = float(np.clip(beta, 1e-9, 1.0 - 1e-9))
        denom = np.sqrt(max(1.0 - b ** 2, 1e-9))
        valid = innovations[~np.isnan(innovations)]
        if len(valid) < 3:
            return np.nan
        return float(np.std(valid) / denom)

    @staticmethod
    def _build_kf_from_ols(
        warmup_prices: np.ndarray, params: StrategyParams
    ) -> tuple[KalmanFilter, list[float]]:
        """
        OLS-initialised KF warm-up on the first vol_window prices.

        Runs OLS to get a stable initial [β, α], then runs the KF over
        those bars to settle the covariance matrix.  Returns the warmed-up
        KF and the list of innovations collected during warm-up.
        """
        warmup = len(warmup_prices)
        y_w = warmup_prices[1:]
        X_w = np.column_stack([warmup_prices[:-1], np.ones(warmup - 1)])
        coeffs = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
        b_init = float(np.clip(coeffs[0], 1e-9, 1.0 - 1e-9))
        a_init = float(coeffs[1])
        resid = y_w - X_w @ coeffs
        obs_var_init = max(float(np.var(resid)), params.obs_var * 0.01)

        kf = KalmanFilter(
            F=np.eye(2),
            H=np.array([[1.0, 1.0]]),      # placeholder; updated per bar
            Q=params.delta * np.eye(2),
            R=np.array([[params.obs_var]]),
            x0=np.array([b_init, a_init]),
            P0=obs_var_init * np.eye(2),   # stable init, not diffuse
        )

        innovations: list[float] = []
        for t in range(1, warmup):
            kf.H = np.array([[warmup_prices[t - 1], 1.0]])
            kf.predict()
            inn = warmup_prices[t] - float((kf.H @ kf.x)[0])
            innovations.append(inn)
            kf.update(np.array([warmup_prices[t]]))

        return kf, innovations

    # ------------------------------------------------------------------
    # Core simulation loop (used by run() and fit())
    # ------------------------------------------------------------------

    def _simulate(
        self, prices: np.ndarray, params: StrategyParams
    ) -> dict[str, np.ndarray]:
        """
        Causal forward pass over a price series.

        Returns arrays of length T for state estimates and signals,
        and length T−1 for log-returns.
        """
        T = len(prices)
        warmup = params.vol_window

        # Phase 1 — OLS initialisation + KF warm-up
        kf, warmup_innov = self._build_kf_from_ols(prices[:warmup], params)

        mu_arr       = np.full(T, np.nan)
        theta_arr    = np.full(T, np.nan)
        sigma_eq_arr = np.full(T, np.nan)
        z_arr        = np.full(T, np.nan)
        innov_arr    = np.full(T, np.nan)

        # Store warm-up innovations in innov_arr (indices 1..warmup-1)
        for i, v in enumerate(warmup_innov):
            innov_arr[i + 1] = v

        # Phase 2 — live filtering from bar `warmup` onwards
        for t in range(warmup, T):
            kf.H = np.array([[prices[t - 1], 1.0]])

            kf.predict()
            inn_val = prices[t] - float((kf.H @ kf.x)[0])
            innov_arr[t] = inn_val
            kf.update(np.array([prices[t]]))

            beta, alpha = float(kf.x[0]), float(kf.x[1])

            theta_t, mu_t = self._ou_from_ar1(beta, alpha, params.dt)
            mu_arr[t]    = mu_t
            theta_arr[t] = theta_t

            # Rolling σ_eq over the last vol_window innovations
            w_start = max(1, t - params.vol_window + 1)
            sig = self._sigma_eq(innov_arr[w_start : t + 1], beta)
            sigma_eq_arr[t] = sig

            if not np.isnan(sig) and sig > 0.0:
                z_arr[t] = (prices[t] - mu_t) / sig

        # ---- Signal generation (stateful, causal, live bars only) -----
        signal = np.zeros(T, dtype=float)
        cur_pos = 0.0
        entry_z = params.entry_z
        exit_z  = params.exit_z
        stop_z  = params.stop_z

        for t in range(warmup, T):
            z = z_arr[t]
            if np.isnan(z):
                signal[t] = cur_pos
                continue

            if cur_pos != 0.0 and abs(z) > stop_z:     # stop-loss
                cur_pos = 0.0
            elif cur_pos == 0.0:                         # flat — check entry
                if z < -entry_z:
                    cur_pos = 1.0                        # long: price too low
                elif z > entry_z:
                    cur_pos = -1.0                       # short: price too high
            elif cur_pos > 0.0:                          # long — check exit
                if z >= -exit_z:
                    cur_pos = 0.0
            else:                                        # short — check exit
                if z <= exit_z:
                    cur_pos = 0.0

            signal[t] = cur_pos

        # ---- Log-returns (signal[t] is the position from t to t+1) ---
        log_ret_asset = np.diff(np.log(np.maximum(prices, 1e-10)))
        strat_ret = signal[:-1] * log_ret_asset

        # ---- half-life ------------------------------------------------
        hl = np.full(T, np.nan)
        valid = theta_arr > 0
        hl[valid] = np.log(2.0) / theta_arr[valid]

        return {
            "mu":         mu_arr,
            "theta":      theta_arr,
            "half_life":  hl,
            "sigma_eq":   sigma_eq_arr,
            "z_score":    z_arr,
            "signal":     signal,
            "log_returns": strat_ret,
            "innovations": innov_arr,
        }

    # ------------------------------------------------------------------
    # Log-likelihood (exact KF, used for 'likelihood' training)
    # ------------------------------------------------------------------

    def _log_likelihood(
        self, prices: np.ndarray, params: StrategyParams
    ) -> float:
        """
        Exact Kalman Filter log-likelihood for the time-varying AR(1) model.

            log L = −½ Σ_t [ log(2π S_t) + ν_t² / S_t ]

        where S_t = H_t P_{t|t-1} H_t' + obs_var is the innovation covariance
        and ν_t = X_t − H_t x_{t|t-1} is the one-step prediction error.
        """
        warmup = params.vol_window
        kf, _ = self._build_kf_from_ols(prices[:warmup], params)
        ll = 0.0

        for t in range(warmup, len(prices)):
            kf.H = np.array([[prices[t - 1], 1.0]])
            kf.predict()

            PHt = kf.P @ kf.H.T
            S_t = float((kf.H @ PHt + kf.R)[0, 0])
            nu_t = prices[t] - float((kf.H @ kf.x)[0])

            if S_t > 1e-14:
                ll += -0.5 * (_LOG_2PI + np.log(S_t) + nu_t ** 2 / S_t)

            kf.update(np.array([prices[t]]))

        return float(ll)

    # ------------------------------------------------------------------
    # Public API — run
    # ------------------------------------------------------------------

    def run(
        self,
        prices: np.ndarray | pd.Series,
    ) -> BacktestResult:
        """
        Batch causal backtest on a price series.

        Parameters
        ----------
        prices : array_like, shape (T,)
            Observed prices or any mean-reverting series (spreads, log-ratios).
            Must be positive if log-returns are to be interpreted as prices.

        Returns
        -------
        BacktestResult
        """
        index = prices.index if isinstance(prices, pd.Series) else None
        arr = np.asarray(prices, dtype=float)
        min_len = self.params.vol_window + 5
        if len(arr) < min_len:
            raise ValueError(
                f"Need ≥ {min_len} bars (vol_window + 5). Got {len(arr)}."
            )

        out = self._simulate(arr, self.params)
        return BacktestResult(
            mu=out["mu"],
            theta=out["theta"],
            half_life=out["half_life"],
            sigma_eq=out["sigma_eq"],
            z_score=out["z_score"],
            signal=out["signal"],
            log_returns=out["log_returns"],
            prices=arr,
            index=index,
        )

    # ------------------------------------------------------------------
    # Public API — online step
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise streaming state (call before replaying new data)."""
        self._initialized = True
        self._warmup_buffer = []
        self._kf = None
        self._prev_price = None
        self._innovations = []
        self._cur_pos = 0.0

    def step(self, price: float) -> dict:
        """
        Process a single new price in streaming (online) mode.

        Must call ``reset()`` before the first ``step()`` call (or after
        changing ``self.params``).

        Parameters
        ----------
        price : float
            Latest observed price.

        Returns
        -------
        dict with keys:
            signal    : float  — current position {−1, 0, +1}
            z_score   : float  — current z-score (nan during warm-up)
            mu        : float  — current OU mean estimate
            theta     : float  — current mean-reversion speed
            half_life : float  — ln(2)/theta
            sigma_eq  : float  — current stationary std estimate
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        warmup = self.params.vol_window
        price = float(price)
        _nan_state = {
            "signal": 0.0, "z_score": np.nan, "mu": np.nan,
            "theta": np.nan, "half_life": np.nan, "sigma_eq": np.nan,
        }

        # ---- Warm-up phase: buffer first vol_window prices ------------
        if len(self._warmup_buffer) < warmup:
            self._warmup_buffer.append(price)

            if len(self._warmup_buffer) == warmup:
                # OLS init + KF warm-up over buffered prices
                buf = np.array(self._warmup_buffer)
                self._kf, innov_list = self._build_kf_from_ols(buf, self.params)
                self._innovations = innov_list
                self._prev_price = price   # last bar of warm-up

            return _nan_state

        # ---- Live phase -----------------------------------------------
        kf = self._kf
        kf.H = np.array([[self._prev_price, 1.0]])
        kf.predict()
        inn = price - float((kf.H @ kf.x)[0])
        kf.update(np.array([price]))

        self._innovations.append(inn)
        if len(self._innovations) > warmup:
            self._innovations.pop(0)

        beta, alpha = float(kf.x[0]), float(kf.x[1])
        theta, mu = self._ou_from_ar1(beta, alpha, self.params.dt)
        sig = self._sigma_eq(np.array(self._innovations), beta)

        z = float((price - mu) / sig) if (not np.isnan(sig) and sig > 0) else np.nan

        if not np.isnan(z):
            if self._cur_pos != 0.0 and abs(z) > self.params.stop_z:
                self._cur_pos = 0.0
            elif self._cur_pos == 0.0:
                if z < -self.params.entry_z:
                    self._cur_pos = 1.0
                elif z > self.params.entry_z:
                    self._cur_pos = -1.0
            elif self._cur_pos > 0.0:
                if z >= -self.params.exit_z:
                    self._cur_pos = 0.0
            else:
                if z <= self.params.exit_z:
                    self._cur_pos = 0.0

        self._prev_price = price
        hl = np.log(2.0) / theta if theta > 0 else np.nan
        return {
            "signal":    self._cur_pos,
            "z_score":   z,
            "mu":        mu,
            "theta":     theta,
            "half_life": hl,
            "sigma_eq":  sig,
        }

    # ------------------------------------------------------------------
    # Public API — fit
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: np.ndarray | pd.Series,
        method: str = "sharpe",
        train_frac: float = 1.0,
        n_restarts: int = 5,
        verbose: bool = False,
    ) -> TrainResult:
        """
        Optimise the trainable parameters from historical price data.

        Parameters
        ----------
        prices : array_like, shape (T,)
            Historical price series used for training.
        method : {'sharpe', 'likelihood'}
            * ``'sharpe'`` — maximise the annualised Sharpe ratio.  All five
              trainable parameters (delta, obs_var, entry_z, exit_z, stop_z)
              are optimised jointly via Nelder-Mead with multiple restarts.
            * ``'likelihood'`` — maximise the exact KF log-likelihood.  Only
              (delta, obs_var) are trained; signal thresholds are unchanged.
              Uses L-BFGS-B with numerical gradients.
        train_frac : float
            Fraction of ``prices`` to use for training.  Remaining data can
            be used for out-of-sample evaluation after calling ``run()``.
        n_restarts : int
            Number of random restarts (Sharpe only; ignored for likelihood).
        verbose : bool
            Print Sharpe at each restart when True.

        Returns
        -------
        TrainResult
            Contains the optimal ``StrategyParams`` and training statistics.
            The strategy's ``self.params`` is updated in-place on success.
        """
        arr = np.asarray(prices, dtype=float)
        n_train = max(
            int(len(arr) * train_frac),
            self.params.vol_window + 10,
        )
        train = arr[:n_train]

        if method == "sharpe":
            result = self._fit_sharpe(train, n_restarts, verbose)
        elif method == "likelihood":
            result = self._fit_likelihood(train, verbose)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'sharpe' or 'likelihood'.")

        self.params = result.params
        return result

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _unpack_sharpe_x(
        self, x: np.ndarray, entry_z_ref: float
    ) -> StrategyParams:
        """
        Decode the Nelder-Mead parameter vector → StrategyParams.

        Reparameterisation keeps the optimizer unconstrained:
            x[0] = log(delta)
            x[1] = log(obs_var)
            x[2] = entry_z        (clipped to [0.3, 4])
            x[3] = raw_exit_frac  → exit_z = clip(sigmoid(x[3])·entry_z, 0, entry_z−0.05)
            x[4] = log(stop_z)    (clipped so stop_z > entry_z)
        """
        delta   = float(np.exp(np.clip(x[0], -16.0, 0.0)))
        obs_var = float(np.exp(np.clip(x[1], -4.0, 8.0)))
        entry_z = float(np.clip(x[2], 0.3, 4.0))
        # exit_z is expressed as a fraction of entry_z via a sigmoid
        exit_frac = 1.0 / (1.0 + np.exp(-x[3]))   # (0, 1)
        exit_z = float(np.clip(exit_frac * entry_z, 0.0, entry_z - 0.05))
        stop_z  = float(np.exp(np.clip(x[4], np.log(entry_z + 0.1), 2.3)))
        stop_z  = max(stop_z, entry_z + 0.05)
        return StrategyParams(
            delta=delta, obs_var=obs_var,
            entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
            vol_window=self.params.vol_window,
            dt=self.params.dt,
        )

    def _fit_sharpe(
        self, prices: np.ndarray, n_restarts: int, verbose: bool
    ) -> TrainResult:
        rng = np.random.default_rng(42)
        best_neg_sharpe = np.inf
        best_result: optimize.OptimizeResult | None = None

        # Encode current params as starting point
        p0 = self.params
        exit_frac0 = p0.exit_z / max(p0.entry_z, 1e-3)
        raw_exit0  = np.log(exit_frac0 / max(1.0 - exit_frac0, 1e-9))
        x0 = np.array([
            np.log(p0.delta),
            np.log(p0.obs_var),
            p0.entry_z,
            raw_exit0,
            np.log(p0.stop_z),
        ])

        def objective(x: np.ndarray) -> float:
            try:
                p = self._unpack_sharpe_x(x, x[2])
                out = self._simulate(prices, p)
                r = out["log_returns"]
                std = float(np.std(r))
                traded = float(np.sum(np.abs(out["signal"])))
                if std < 1e-10 or traded < 5.0:
                    return 500.0          # penalty: flat strategy
                return float(-np.mean(r) / std * _ANNUAL)
            except Exception:
                return 500.0

        for restart in range(n_restarts):
            start = x0 if restart == 0 else x0 + rng.normal(0, 0.5, size=5)
            res = optimize.minimize(
                objective,
                x0=start,
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5},
            )
            if verbose:
                print(f"  restart {restart + 1}/{n_restarts}  "
                      f"Sharpe = {-res.fun:.4f}")
            if res.fun < best_neg_sharpe:
                best_neg_sharpe = res.fun
                best_result = res

        assert best_result is not None
        best_params = self._unpack_sharpe_x(best_result.x, best_result.x[2])
        ll = self._log_likelihood(prices, best_params)

        return TrainResult(
            params=best_params,
            train_sharpe=-best_neg_sharpe,
            train_log_lik=ll,
            n_iter=best_result.nit,
            converged=best_result.success,
            method="sharpe",
        )

    def _fit_likelihood(self, prices: np.ndarray, verbose: bool) -> TrainResult:
        """
        Maximise the exact KF log-likelihood over (delta, obs_var).

        Signal thresholds (entry_z, exit_z, stop_z) are held fixed at their
        current values.  Uses L-BFGS-B on the log-parameterised space.
        """
        p0 = self.params
        x0 = np.array([np.log(p0.delta), np.log(p0.obs_var)])

        def neg_ll(x: np.ndarray) -> float:
            delta   = float(np.exp(np.clip(x[0], -16.0, 0.0)))
            obs_var = float(np.exp(np.clip(x[1], -4.0, 8.0)))
            try:
                p = StrategyParams(
                    delta=delta, obs_var=obs_var,
                    entry_z=p0.entry_z, exit_z=p0.exit_z, stop_z=p0.stop_z,
                    vol_window=p0.vol_window, dt=p0.dt,
                )
                return float(-self._log_likelihood(prices, p))
            except Exception:
                return 1e9

        res = optimize.minimize(
            neg_ll,
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
        )
        if verbose:
            print(f"  L-BFGS-B: log-lik = {-res.fun:.2f}  "
                  f"converged = {res.success}")

        delta_opt   = float(np.exp(np.clip(res.x[0], -16.0, 0.0)))
        obs_var_opt = float(np.exp(np.clip(res.x[1], -4.0, 8.0)))
        best_params = StrategyParams(
            delta=delta_opt, obs_var=obs_var_opt,
            entry_z=p0.entry_z, exit_z=p0.exit_z, stop_z=p0.stop_z,
            vol_window=p0.vol_window, dt=p0.dt,
        )

        # Compute resulting Sharpe for the report
        out = self._simulate(prices, best_params)
        r   = out["log_returns"]
        std = float(np.std(r))
        sh  = float(np.mean(r) / max(std, 1e-10) * _ANNUAL)

        return TrainResult(
            params=best_params,
            train_sharpe=sh,
            train_log_lik=-res.fun,
            n_iter=res.nit,
            converged=res.success,
            method="likelihood",
        )
