"""
Cointegration pairs trading via rolling Engle-Granger.

Architecture
------------
Two assets ``y`` and ``x`` are traded as a market-neutral pair whenever
the Engle-Granger residual ADF test rejects the no-cointegration null.

Step 1 — Rolling Engle-Granger
    Every ``refit_every`` bars a window of length ``fit_window`` is used to
    re-run :func:`qufin.timeseries.engle_granger`.  The fit yields
    (β_t, α_t, p-value).  The pair is *active* whenever p-value < ``eg_alpha``;
    otherwise the strategy is forced flat (existing positions are closed).

Step 2 — Spread and z-score
    Between refits the spread is rolled forward with the most-recent
    (β, α):

        spread_t  =  y_t  −  β·x_t  −  α
        z_t       =  (spread_t  −  mean(spread)_W)  /  std(spread)_W

    where mean/std are taken over the last ``zscore_window`` observations
    of the live spread.

Step 3 — Threshold trades
    z_t  < −entry_z  →  long pair  (long y, short β units of x)
    z_t  > +entry_z  →  short pair
    |z_t| < exit_z   →  close
    |z_t| > stop_z   →  emergency close (regime break)

Position arrays
---------------
The strategy returns two position arrays — ``pos_y[t]`` and ``pos_x[t]``
— with ``pos_x[t] = −β_t · pos_y[t]``.  Strategy returns are computed in
log-space as ``pos_y[t]·Δlog(y) + pos_x[t]·Δlog(x)`` so the same series
can be reused as inputs to other tools in the toolkit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..timeseries.cointegration import engle_granger

_SeriesLike = np.ndarray | pl.Series
_ANNUAL = math.sqrt(252.0)


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
class PairsParams:
    """Hyperparameters of :class:`CointegrationPairsStrategy`.

    Attributes
    ----------
    fit_window : int
        Bars used in every Engle-Granger refit.
    refit_every : int
        Bars between refits.  Spread parameters are held constant in between.
    zscore_window : int
        Rolling window for computing the spread mean and std.
    eg_alpha : float
        Significance level for the cointegration gate.  When the residual
        ADF p-value exceeds ``eg_alpha`` the strategy is forced flat.
    use_log_prices : bool
        If True, the regression is run on log-prices.  Recommended for
        equities and FX.
    entry_z : float
        Enter a pair position when |z| ≥ entry_z.
    exit_z : float
        Close when |z| crosses back inside ±exit_z.  Must satisfy exit_z < entry_z.
    stop_z : float
        Emergency close at |z| ≥ stop_z.  Must satisfy stop_z > entry_z.
    """

    fit_window: int = 200
    refit_every: int = 25
    zscore_window: int = 60
    eg_alpha: float = 0.10
    use_log_prices: bool = True
    entry_z: float = 1.5
    exit_z: float = 0.3
    stop_z: float = 4.0

    def __post_init__(self) -> None:
        if self.fit_window < 30:
            raise ValueError("fit_window must be >= 30")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.zscore_window < 10:
            raise ValueError("zscore_window must be >= 10")
        if not (0.0 < self.eg_alpha < 1.0):
            raise ValueError("eg_alpha must be in (0, 1)")
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.exit_z < 0:
            raise ValueError("exit_z must be >= 0")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be > entry_z")


@dataclass(slots=True)
class PairsBacktestResult:
    """Output of :meth:`CointegrationPairsStrategy.run`."""

    beta: np.ndarray  # (T,) live cointegrating slope
    alpha: np.ndarray  # (T,) live intercept
    spread: np.ndarray  # (T,) live spread = y − β·x − α
    z_score: np.ndarray  # (T,) standardised spread
    p_value: np.ndarray  # (T,) most recent EG p-value
    active: np.ndarray  # (T,) bool — cointegration gate open
    pos_y: np.ndarray  # (T,) position in y in {−1, 0, +1}
    pos_x: np.ndarray  # (T,) hedge position = −β·pos_y
    log_returns: np.ndarray  # (T−1,) strategy PnL in log space
    y: np.ndarray  # (T,) input series y
    x: np.ndarray  # (T,) input series x

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
    def n_trades(self) -> int:
        entries = np.sum((self.pos_y[1:] != 0) & (self.pos_y[:-1] == 0))
        return int(entries)

    @property
    def active_fraction(self) -> float:
        return float(np.nanmean(self.active.astype(np.float64)))

    def to_dataframe(self) -> pl.DataFrame:
        strat_ret = np.concatenate([[np.nan], self.log_returns])
        return pl.DataFrame(
            {
                "y": self.y,
                "x": self.x,
                "beta": self.beta,
                "alpha": self.alpha,
                "spread": self.spread,
                "z_score": self.z_score,
                "p_value": self.p_value,
                "active": self.active,
                "pos_y": self.pos_y,
                "pos_x": self.pos_x,
                "strat_ret": strat_ret,
            }
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "Cointegration Pairs Summary",
                "=" * 35,
                f"  Sharpe (ann.)   : {self.sharpe:.4f}",
                f"  Total return    : {self.total_return * 100:.2f}%",
                f"  Max drawdown    : {self.max_drawdown * 100:.2f}%",
                f"  Num trades      : {self.n_trades}",
                f"  Active fraction : {self.active_fraction * 100:.1f}%",
                f"  Avg |z_score|   : {np.nanmean(np.abs(self.z_score)):.3f}",
            ]
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class CointegrationPairsStrategy:
    """Rolling Engle-Granger pairs trading strategy.

    Parameters
    ----------
    params : PairsParams, optional

    Examples
    --------
    >>> strat = CointegrationPairsStrategy()
    >>> result = strat.run(y, x)
    >>> print(result.summary())
    """

    def __init__(self, params: PairsParams | None = None) -> None:
        self.params: PairsParams = params if params is not None else PairsParams()

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run(self, y: _SeriesLike, x: _SeriesLike) -> PairsBacktestResult:
        """Causal backtest on two aligned price series.

        Parameters
        ----------
        y, x : array_like, shape (T,)
            Price series of the two legs.  Must be strictly positive when
            ``use_log_prices`` is True.
        """
        p = self.params
        y_raw = _to_numpy_1d(y)
        x_raw = _to_numpy_1d(x)
        if y_raw.shape != x_raw.shape:
            raise ValueError(f"y and x must have same shape; got {y_raw.shape} vs {x_raw.shape}.")
        T = y_raw.shape[0]
        if T < p.fit_window + p.zscore_window + 5:
            raise ValueError(
                f"Need ≥ {p.fit_window + p.zscore_window + 5} bars; got {T}."
            )
        if p.use_log_prices:
            if np.any(y_raw <= 0.0) or np.any(x_raw <= 0.0):
                raise ValueError("All prices must be > 0 when use_log_prices is True.")
            y_arr = np.log(y_raw)
            x_arr = np.log(x_raw)
        else:
            y_arr = y_raw.copy()
            x_arr = x_raw.copy()

        beta_arr = np.full(T, np.nan)
        alpha_arr = np.full(T, np.nan)
        spread_arr = np.full(T, np.nan)
        z_arr = np.full(T, np.nan)
        pval_arr = np.full(T, np.nan)
        active_arr = np.zeros(T, dtype=bool)
        pos_y = np.zeros(T)
        pos_x = np.zeros(T)

        cur_beta = np.nan
        cur_alpha = np.nan
        cur_pval = 1.0
        cur_active = False
        cur_pos = 0.0
        bars_since_refit = 0

        warmup = p.fit_window

        for t in range(warmup, T):
            # Refit on a rolling window ending at t-1 (causal: t-th decision
            # can only use data up to and including bar t-1 for the fit).
            if bars_since_refit == 0 or bars_since_refit >= p.refit_every:
                y_win = y_arr[t - p.fit_window : t]
                x_win = x_arr[t - p.fit_window : t]
                try:
                    eg = engle_granger(y_win, x_win, trend="c", autolag="aic")
                    cur_beta = float(eg.beta[0])
                    cur_alpha = float(eg.alpha)
                    cur_pval = float(eg.p_value)
                    cur_active = bool(cur_pval < p.eg_alpha)
                except Exception:
                    cur_active = False
                    cur_pval = 1.0
                bars_since_refit = 0
            bars_since_refit += 1

            beta_arr[t] = cur_beta
            alpha_arr[t] = cur_alpha
            pval_arr[t] = cur_pval
            active_arr[t] = cur_active

            if not math.isfinite(cur_beta) or not math.isfinite(cur_alpha):
                pos_y[t] = 0.0
                pos_x[t] = 0.0
                continue

            spread_t = y_arr[t] - cur_beta * x_arr[t] - cur_alpha
            spread_arr[t] = spread_t

            w_start = max(warmup, t - p.zscore_window + 1)
            window = spread_arr[w_start : t + 1]
            valid = window[~np.isnan(window)]
            if valid.shape[0] < 5:
                z = np.nan
            else:
                std = float(np.std(valid))
                if std > 1e-12:
                    z = float((spread_t - float(np.mean(valid))) / std)
                else:
                    z = np.nan
            z_arr[t] = z

            # Force flat when cointegration is rejected
            if not cur_active:
                cur_pos = 0.0
            elif not math.isnan(z):
                if cur_pos != 0.0 and abs(z) > p.stop_z:
                    cur_pos = 0.0
                elif cur_pos == 0.0:
                    if z < -p.entry_z:
                        cur_pos = 1.0
                    elif z > p.entry_z:
                        cur_pos = -1.0
                elif cur_pos > 0.0:
                    if z >= -p.exit_z:
                        cur_pos = 0.0
                else:
                    if z <= p.exit_z:
                        cur_pos = 0.0

            pos_y[t] = cur_pos
            pos_x[t] = -cur_beta * cur_pos

        if p.use_log_prices:
            d_y = np.diff(y_arr)
            d_x = np.diff(x_arr)
        else:
            d_y = np.diff(np.log(np.maximum(y_raw, 1e-12)))
            d_x = np.diff(np.log(np.maximum(x_raw, 1e-12)))

        strat_ret = pos_y[:-1] * d_y + pos_x[:-1] * d_x

        return PairsBacktestResult(
            beta=beta_arr,
            alpha=alpha_arr,
            spread=spread_arr,
            z_score=z_arr,
            p_value=pval_arr,
            active=active_arr,
            pos_y=pos_y,
            pos_x=pos_x,
            log_returns=strat_ret,
            y=y_raw,
            x=x_raw,
        )
