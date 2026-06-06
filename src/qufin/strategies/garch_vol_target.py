"""
GARCH-driven volatility-targeted trend strategy.

Architecture
------------
The strategy combines a momentum/trend signal with a conditional-variance
model that dynamically scales the position to a constant ex-ante
volatility target.

Step 1 — Trend signal
    A two-EMA crossover on log-prices:

        s_t  =  sign(EMA_fast(log p_t) − EMA_slow(log p_t))

    s_t ∈ {−1, 0, +1}.  A neutral band keeps the strategy flat when the
    crossover magnitude (normalised by σ̂_t) is below ``signal_band``.

Step 2 — Conditional-variance forecast
    A GARCH(1, 1) or EGARCH(1, 1) model is refit on a rolling window of
    log-returns every ``refit_every`` bars.  Between refits the conditional
    variance recursion is rolled forward analytically using the most
    recently observed return:

        σ²_{t+1}  =  ω + α ε²_t + β σ²_t                      (GARCH)
        log σ²_{t+1} = ω + α z_t + γ (|z_t| − E|z|) + β log σ²_t (EGARCH)

Step 3 — Volatility-targeted position
    Position size:

        pos_t  =  s_t  ·  min( leverage_cap,  target_vol_ann / σ̂_t,ann )

    σ̂_t,ann = σ̂_t · √(252 / dt).  This keeps realised PnL volatility close
    to ``target_vol_ann`` regardless of regime.

Trainable parameters
--------------------
    fast_span         EMA span for the fast leg
    slow_span         EMA span for the slow leg
    target_vol_ann    desired annualised volatility of strategy PnL
    leverage_cap      hard cap on |pos_t|
    signal_band       trend-strength deadband (in σ units)

The conditional-variance model parameters are estimated by maximum
likelihood inside :class:`qufin.timeseries.GARCH` / ``EGARCH`` and are
not tuned by the strategy itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..timeseries.garch import EGARCH, GARCH
from ..utils import to_numpy_1d as _to_numpy_1d

_SeriesLike = np.ndarray | pl.Series
_ANNUAL = math.sqrt(252.0)
_E_ABS_Z = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------------------
# Parameters and result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GARCHVolTargetParams:
    """Hyperparameters of :class:`GARCHVolTargetStrategy`.

    Attributes
    ----------
    fast_span : int
        EMA span of the fast trend leg.  Must be ≥ 2.
    slow_span : int
        EMA span of the slow trend leg.  Must be > ``fast_span``.
    target_vol_ann : float
        Desired annualised volatility of the strategy PnL stream.
    leverage_cap : float
        Maximum |position|.  Caps the vol-target multiplier when realised
        variance becomes very small.
    signal_band : float
        Neutral deadband in units of σ̂_t,daily applied to the EMA spread
        before signing.
    model : {'garch', 'egarch'}
        Conditional-variance specification.  EGARCH captures the leverage
        effect via the ``γ`` coefficient.
    fit_window : int
        Rolling-window size (in bars) used for each refit.
    refit_every : int
        Bars between model refits.  The variance recursion is rolled
        forward analytically between refits.
    warmup : int
        Bars consumed before the first signal is emitted.  Must be ≥
        ``fit_window``.
    dt : float
        Sampling interval (1.0 = one trading day).
    """

    fast_span: int = 20
    slow_span: int = 100
    target_vol_ann: float = 0.10
    leverage_cap: float = 4.0
    signal_band: float = 0.10
    model: str = "garch"
    fit_window: int = 250
    refit_every: int = 25
    warmup: int = 250
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.fast_span < 2:
            raise ValueError("fast_span must be >= 2")
        if self.slow_span <= self.fast_span:
            raise ValueError("slow_span must be > fast_span")
        if self.target_vol_ann <= 0:
            raise ValueError("target_vol_ann must be > 0")
        if self.leverage_cap <= 0:
            raise ValueError("leverage_cap must be > 0")
        if self.signal_band < 0:
            raise ValueError("signal_band must be >= 0")
        if self.model not in ("garch", "egarch"):
            raise ValueError("model must be 'garch' or 'egarch'")
        if self.fit_window < 50:
            raise ValueError("fit_window must be >= 50")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.warmup < self.fit_window:
            raise ValueError("warmup must be >= fit_window")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")


@dataclass(slots=True)
class GARCHVolTargetResult:
    """Output of :meth:`GARCHVolTargetStrategy.run`."""

    sigma_daily: np.ndarray  # (T,) conditional volatility forecast
    trend: np.ndarray  # (T,) EMA spread normalised by σ̂
    signal: np.ndarray  # (T,) sign of the trend after deadband, in {−1, 0, +1}
    position: np.ndarray  # (T,) vol-targeted position (continuous)
    log_returns: np.ndarray  # (T−1,) position[t] · Δlog price
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
    def realised_vol_ann(self) -> float:
        return float(np.nanstd(self.log_returns) * _ANNUAL)

    @property
    def avg_abs_position(self) -> float:
        return float(np.nanmean(np.abs(self.position)))

    def to_dataframe(self) -> pl.DataFrame:
        strat_ret = np.concatenate([[np.nan], self.log_returns])
        return pl.DataFrame(
            {
                "price": self.prices,
                "sigma_daily": self.sigma_daily,
                "trend": self.trend,
                "signal": self.signal,
                "position": self.position,
                "strat_ret": strat_ret,
            }
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "GARCH Vol-Target Summary",
                "=" * 35,
                f"  Sharpe (ann.)  : {self.sharpe:.4f}",
                f"  Total return   : {self.total_return * 100:.2f}%",
                f"  Max drawdown   : {self.max_drawdown * 100:.2f}%",
                f"  Realised vol   : {self.realised_vol_ann * 100:.2f}%",
                f"  Avg |position| : {self.avg_abs_position:.3f}",
            ]
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class GARCHVolTargetStrategy:
    """EMA trend-follower whose position is scaled to a constant
    ex-ante volatility target via a GARCH/EGARCH conditional-variance
    forecast.

    Parameters
    ----------
    params : GARCHVolTargetParams, optional
        Hyperparameters.  Default settings are used when omitted.

    Examples
    --------
    >>> strat = GARCHVolTargetStrategy()
    >>> result = strat.run(prices)
    >>> print(result.summary())
    """

    def __init__(self, params: GARCHVolTargetParams | None = None) -> None:
        self.params: GARCHVolTargetParams = params if params is not None else GARCHVolTargetParams()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(x)
        out[0] = x[0]
        for t in range(1, x.shape[0]):
            out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
        return out

    def _fit_variance_model(self, log_ret: np.ndarray) -> tuple[float, dict[str, float]]:
        """Fit GARCH/EGARCH and return (one-step σ²_{t+1} forecast, state)."""
        p = self.params
        if p.model == "garch":
            model = GARCH(1, 1, mean="zero")
            res = model.fit(log_ret)
            eps_last = float(res.residuals[-1])
            sigma2_last = float(res.sigma2[-1])
            sigma2_next = (
                res.omega + float(res.alpha[0]) * eps_last**2 + float(res.beta[0]) * sigma2_last
            )
            state = {
                "omega": float(res.omega),
                "alpha": float(res.alpha[0]),
                "beta": float(res.beta[0]),
                "sigma2": sigma2_next,
            }
            return sigma2_next, state

        model = EGARCH(1, 1, mean="zero")
        res = model.fit(log_ret)
        eps_last = float(res.residuals[-1])
        log_s_last = math.log(max(float(res.sigma2[-1]), 1e-12))
        z_last = eps_last / math.sqrt(max(float(res.sigma2[-1]), 1e-12))
        log_s_next = (
            float(res.omega)
            + float(res.alpha[0]) * z_last
            + float(res.gamma[0]) * (abs(z_last) - _E_ABS_Z)
            + float(res.beta[0]) * log_s_last
        )
        sigma2_next = math.exp(min(max(log_s_next, -50.0), 50.0))
        state = {
            "omega": float(res.omega),
            "alpha": float(res.alpha[0]),
            "gamma": float(res.gamma[0]),
            "beta": float(res.beta[0]),
            "log_sigma2": log_s_next,
        }
        return sigma2_next, state

    def _roll_variance(
        self, state: dict[str, float], eps_t: float
    ) -> tuple[float, dict[str, float]]:
        """Roll the variance recursion one bar forward without refitting."""
        if self.params.model == "garch":
            sigma2 = state["sigma2"]
            new_sigma2 = state["omega"] + state["alpha"] * eps_t**2 + state["beta"] * sigma2
            state["sigma2"] = max(new_sigma2, 1e-14)
            return state["sigma2"], state

        log_s = state["log_sigma2"]
        sigma = math.sqrt(math.exp(min(max(log_s, -50.0), 50.0)))
        z = eps_t / max(sigma, 1e-12)
        new_log_s = (
            state["omega"]
            + state["alpha"] * z
            + state["gamma"] * (abs(z) - _E_ABS_Z)
            + state["beta"] * log_s
        )
        state["log_sigma2"] = min(max(new_log_s, -50.0), 50.0)
        return math.exp(state["log_sigma2"]), state

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run(self, prices: _SeriesLike) -> GARCHVolTargetResult:
        """Causal backtest on a single price series.

        Parameters
        ----------
        prices : array_like, shape (T,)
            Positive price series.  Returns are computed as Δlog(prices).
        """
        p = self.params
        arr = _to_numpy_1d(prices)
        T = arr.shape[0]
        if T < p.warmup + 5:
            raise ValueError(f"Need ≥ {p.warmup + 5} bars; got {T}.")
        if np.any(arr <= 0.0):
            raise ValueError("All prices must be strictly positive.")

        log_p = np.log(arr)
        log_ret = np.diff(log_p)  # length T-1, indexed t → return from t to t+1

        fast = self._ema(log_p, p.fast_span)
        slow = self._ema(log_p, p.slow_span)

        sigma_daily = np.full(T, np.nan)
        trend = np.full(T, np.nan)
        signal = np.zeros(T)
        position = np.zeros(T)

        state: dict[str, float] | None = None
        bars_since_refit = 0
        sigma2_next = np.nan

        # Refit immediately at warmup, then every refit_every bars.
        for t in range(p.warmup, T):
            window_start = max(0, t - p.fit_window)
            # log_ret index: return at index i corresponds to bar i → i+1.
            # The latest fully-observed return at bar t is log_ret[t-1].
            ret_window = log_ret[window_start:t]
            if state is None or bars_since_refit >= p.refit_every:
                try:
                    sigma2_next, state = self._fit_variance_model(ret_window)
                except Exception:
                    # Fall back to sample variance on the window if the GARCH fit fails.
                    var = float(np.var(ret_window))
                    sigma2_next = max(var, 1e-12)
                    state = {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "sigma2": sigma2_next}
                bars_since_refit = 0
            else:
                eps_t = float(log_ret[t - 1])  # newest observed return
                sigma2_next, state = self._roll_variance(state, eps_t)

            bars_since_refit += 1

            sigma_t = math.sqrt(max(sigma2_next, 1e-14))
            sigma_daily[t] = sigma_t

            # Trend signal: EMA spread normalised by σ̂_t
            spread = fast[t] - slow[t]
            tnorm = spread / max(sigma_t, 1e-12)
            trend[t] = tnorm
            if tnorm > p.signal_band:
                s = 1.0
            elif tnorm < -p.signal_band:
                s = -1.0
            else:
                s = 0.0
            signal[t] = s

            sigma_ann = sigma_t * math.sqrt(252.0 / p.dt)
            target_scale = p.target_vol_ann / max(sigma_ann, 1e-12)
            scale = min(p.leverage_cap, target_scale)
            position[t] = s * scale

        # PnL: position[t] held from bar t to bar t+1 earns log_ret[t]
        strat_ret = position[:-1] * log_ret

        return GARCHVolTargetResult(
            sigma_daily=sigma_daily,
            trend=trend,
            signal=signal,
            position=position,
            log_returns=strat_ret,
            prices=arr,
        )
