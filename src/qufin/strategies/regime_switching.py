"""
Markov-switching regime-conditional momentum strategy.

Architecture
------------
A Markov-switching AR(p) model is fitted to log-returns of a single asset.
Each of the K latent regimes has its own (μ_k, φ_k, σ²_k), giving an
expected return and risk per regime.  At each bar the strategy:

    1. Filters a regime distribution ξ_{t|t} causally.
    2. Computes a regime-weighted expected return ē_t = Σ_k ξ_{t|t,k} · μ_k.
    3. Computes a regime-weighted volatility σ̄_t = √(Σ_k ξ_{t|t,k} · σ²_k).
    4. Sets a continuous position:

           pos_t = clip( ē_t / max(σ̄²_t, ε)  ·  risk_scale,  ±leverage_cap )

       i.e. a *Kelly-style* scaling on the regime-conditional Sharpe.

The strategy is therefore a long/short trend-follower that *automatically*
flips when the model detects a high-vol, negative-drift regime — the
classic "bear" state in financial time series — and de-leverages when the
predictive content (|ē_t|/σ̄_t) is weak.

Filtering between refits
------------------------
Refitting the MS-AR model by EM is comparatively expensive.  We refit
every ``refit_every`` bars on a rolling window of size ``fit_window`` and
between refits roll the Hamilton step forward using the saved model
parameters and the new return.  This keeps the entire backtest causal
while remaining computationally tractable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..timeseries.regime import MarkovSwitchingAR

_SeriesLike = np.ndarray | pl.Series
_ANNUAL = math.sqrt(252.0)
_LOG_2PI = math.log(2.0 * math.pi)


def _to_numpy_1d(x: _SeriesLike) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64, copy=False)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Parameters and result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RegimeSwitchingParams:
    """Hyperparameters of :class:`RegimeSwitchingStrategy`.

    Attributes
    ----------
    p : int
        AR order inside each regime.  ``p = 0`` reduces to a Gaussian HMM
        on returns.
    k_regimes : int
        Number of latent regimes.  Use 2 for bull/bear, 3 to add a
        crash/jump state.
    fit_window : int
        Rolling window size (bars of returns) used by each EM refit.
    refit_every : int
        Bars between EM refits.  The Hamilton filter is rolled forward
        analytically between refits.
    warmup : int
        Bars consumed before the first signal.  Must be ≥ ``fit_window`` + 5.
    risk_scale : float
        Multiplier applied to (ē_t / σ̄²_t) before clipping.  Set < 1 to
        trade conservatively, > 1 to lever the Kelly signal.
    leverage_cap : float
        Hard cap on |position|.
    em_max_iter : int
        Maximum EM iterations per refit.
    em_tol : float
        Relative log-likelihood tolerance for EM convergence.
    seed : int
        RNG seed for the K-means initialisation inside the EM refits.
    """

    p: int = 0
    k_regimes: int = 2
    fit_window: int = 400
    refit_every: int = 50
    warmup: int = 400
    risk_scale: float = 1.0
    leverage_cap: float = 3.0
    em_max_iter: int = 100
    em_tol: float = 1e-5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.p < 0:
            raise ValueError("p must be >= 0")
        if self.k_regimes < 2:
            raise ValueError("k_regimes must be >= 2")
        if self.fit_window < 50:
            raise ValueError("fit_window must be >= 50")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.warmup < self.fit_window + 5:
            raise ValueError("warmup must be >= fit_window + 5")
        if self.risk_scale <= 0:
            raise ValueError("risk_scale must be > 0")
        if self.leverage_cap <= 0:
            raise ValueError("leverage_cap must be > 0")
        if self.em_max_iter < 5:
            raise ValueError("em_max_iter must be >= 5")
        if self.em_tol <= 0:
            raise ValueError("em_tol must be > 0")


@dataclass(slots=True)
class RegimeSwitchingResult:
    """Output of :meth:`RegimeSwitchingStrategy.run`."""

    expected_return: np.ndarray  # (T,) regime-weighted ē_t
    expected_vol: np.ndarray  # (T,) regime-weighted σ̄_t
    filtered_probs: np.ndarray  # (T, K) causal ξ_{t|t}
    most_likely: np.ndarray  # (T,) argmax regime (−1 during warm-up)
    position: np.ndarray  # (T,) continuous Kelly position
    log_returns: np.ndarray  # (T−1,) strategy PnL
    prices: np.ndarray  # (T,) input prices

    @property
    def sharpe(self) -> float:
        r = self.log_returns
        std = float(np.nanstd(r))
        return float(np.nanmean(r) / max(std, 1e-12) * _ANNUAL)

    @property
    def total_return(self) -> float:
        return float(np.nansum(self.log_returns))

    @property
    def max_drawdown(self) -> float:
        cum = np.nancumsum(self.log_returns)
        peak = np.maximum.accumulate(cum)
        return float(np.nanmin(cum - peak))

    @property
    def avg_abs_position(self) -> float:
        return float(np.nanmean(np.abs(self.position)))

    def regime_durations(self) -> np.ndarray:
        """Return the count of bars spent in each regime (most-likely)."""
        ml = self.most_likely
        valid = ml[ml >= 0]
        if valid.size == 0:
            return np.zeros(0, dtype=np.int64)
        k = int(valid.max()) + 1
        out = np.zeros(k, dtype=np.int64)
        for r in valid:
            out[r] += 1
        return out

    def to_dataframe(self) -> pl.DataFrame:
        strat_ret = np.concatenate([[np.nan], self.log_returns])
        data: dict[str, np.ndarray] = {
            "price": self.prices,
            "expected_return": self.expected_return,
            "expected_vol": self.expected_vol,
            "most_likely": self.most_likely,
            "position": self.position,
            "strat_ret": strat_ret,
        }
        for k in range(self.filtered_probs.shape[1]):
            data[f"prob_regime_{k}"] = self.filtered_probs[:, k]
        return pl.DataFrame(data)

    def summary(self) -> str:
        durations = self.regime_durations()
        dur_str = ", ".join(f"R{k}={c}" for k, c in enumerate(durations))
        return "\n".join(
            [
                "Regime-Switching Summary",
                "=" * 35,
                f"  Sharpe (ann.)  : {self.sharpe:.4f}",
                f"  Total return   : {self.total_return * 100:.2f}%",
                f"  Max drawdown   : {self.max_drawdown * 100:.2f}%",
                f"  Avg |position| : {self.avg_abs_position:.3f}",
                f"  Regime durations: {dur_str}",
            ]
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class RegimeSwitchingStrategy:
    """Markov-switching AR(p) regime-conditional Kelly strategy.

    Parameters
    ----------
    params : RegimeSwitchingParams, optional

    Examples
    --------
    >>> strat = RegimeSwitchingStrategy()
    >>> result = strat.run(prices)
    >>> print(result.summary())
    """

    def __init__(self, params: RegimeSwitchingParams | None = None) -> None:
        self.params: RegimeSwitchingParams = (
            params if params is not None else RegimeSwitchingParams()
        )

    # ------------------------------------------------------------------
    # Hamilton step (causal one-bar update without refitting)
    # ------------------------------------------------------------------

    @staticmethod
    def _emission(
        y_t: float,
        lags: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
        sigma2: np.ndarray,
        p: int,
    ) -> np.ndarray:
        """Gaussian likelihood ξ(y_t | regime=k) for K regimes."""
        K = mu.shape[0]
        out = np.empty(K)
        for k in range(K):
            mean_k = mu[k]
            if p > 0:
                mean_k = mean_k + float(phi[k] @ (lags - mu[k]))
            s2 = max(float(sigma2[k]), 1e-14)
            r = y_t - mean_k
            out[k] = math.exp(-0.5 * (r * r / s2 + math.log(s2) + _LOG_2PI))
        return out

    @staticmethod
    def _hamilton_step(
        prev_filtered: np.ndarray,
        emission: np.ndarray,
        transition: np.ndarray,
    ) -> np.ndarray:
        """One forward Hamilton update: ξ_{t|t} from ξ_{t-1|t-1}."""
        pred = transition.T @ prev_filtered
        joint = pred * emission
        marginal = joint.sum()
        if marginal <= 0.0 or not math.isfinite(marginal):
            return pred / max(pred.sum(), 1e-12)
        return joint / marginal

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run(self, prices: _SeriesLike) -> RegimeSwitchingResult:
        """Causal backtest on a single price series.

        Parameters
        ----------
        prices : array_like, shape (T,)
            Strictly positive price series.  Log-returns are modelled.
        """
        p_cfg = self.params
        arr = _to_numpy_1d(prices)
        T = arr.shape[0]
        if T < p_cfg.warmup + 5:
            raise ValueError(f"Need ≥ {p_cfg.warmup + 5} bars; got {T}.")
        if np.any(arr <= 0.0):
            raise ValueError("All prices must be strictly positive.")

        log_p = np.log(arr)
        log_ret = np.diff(log_p)  # length T-1, log_ret[i] = return bar i → i+1

        K = p_cfg.k_regimes
        p_ord = p_cfg.p

        expected_ret = np.full(T, np.nan)
        expected_vol = np.full(T, np.nan)
        filtered_probs = np.full((T, K), np.nan)
        most_likely = np.full(T, -1, dtype=np.int64)
        position = np.zeros(T)

        # Saved model parameters between refits
        mu_saved = np.zeros(K)
        phi_saved = np.zeros((K, p_ord))
        sigma2_saved = np.ones(K)
        transition_saved = np.eye(K)
        prev_filtered = np.full(K, 1.0 / K)
        have_model = False
        bars_since_refit = 0

        for t in range(p_cfg.warmup, T):
            # Latest fully-observed return prior to decision at bar t:
            #   log_ret[t-1]  (return from bar t-1 to bar t)
            y_t = float(log_ret[t - 1])

            if bars_since_refit == 0 or bars_since_refit >= p_cfg.refit_every:
                window_start = max(0, t - 1 - p_cfg.fit_window)
                window = log_ret[window_start : t]
                if window.shape[0] >= p_ord + K + 5:
                    try:
                        model = MarkovSwitchingAR(p=p_ord, k_regimes=K)
                        fit = model.fit(
                            window,
                            max_iter=p_cfg.em_max_iter,
                            tol=p_cfg.em_tol,
                            seed=p_cfg.seed,
                        )
                        mu_saved = np.asarray(fit.mu, dtype=np.float64)
                        phi_saved = np.asarray(fit.phi, dtype=np.float64).reshape(K, p_ord)
                        sigma2_saved = np.asarray(fit.sigma2, dtype=np.float64)
                        transition_saved = np.asarray(fit.transition, dtype=np.float64)
                        prev_filtered = np.asarray(
                            fit.filtered_probs[-1], dtype=np.float64
                        ).copy()
                        have_model = True
                    except Exception:
                        # Keep prior model; if no prior, signal stays neutral.
                        pass
                bars_since_refit = 0
            bars_since_refit += 1

            if not have_model:
                continue

            # Build the lag vector ending at log_ret[t-1] (most recent return)
            if p_ord > 0:
                lo = t - 1 - p_ord
                if lo < 0:
                    continue
                lags = log_ret[lo : t - 1][::-1]  # (lag1, lag2, ...) ordering
            else:
                lags = np.zeros(0)

            emission = self._emission(y_t, lags, mu_saved, phi_saved, sigma2_saved, p_ord)
            prev_filtered = self._hamilton_step(prev_filtered, emission, transition_saved)

            filtered_probs[t] = prev_filtered
            most_likely[t] = int(np.argmax(prev_filtered))

            # Regime-weighted predictive moments (one-step ahead)
            pred = transition_saved.T @ prev_filtered
            e_ret = float(pred @ mu_saved)
            e_var = float(pred @ sigma2_saved)
            expected_ret[t] = e_ret
            expected_vol[t] = math.sqrt(max(e_var, 1e-14))

            kelly = e_ret / max(e_var, 1e-12) * p_cfg.risk_scale
            position[t] = max(-p_cfg.leverage_cap, min(p_cfg.leverage_cap, kelly))

        strat_ret = position[:-1] * log_ret

        return RegimeSwitchingResult(
            expected_return=expected_ret,
            expected_vol=expected_vol,
            filtered_probs=filtered_probs,
            most_likely=most_likely,
            position=position,
            log_returns=strat_ret,
            prices=arr,
        )
