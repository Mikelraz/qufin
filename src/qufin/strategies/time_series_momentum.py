"""
Time-series momentum (TSMOM) trading strategy.

Architecture
------------
A thin, trainable backtest wrapper around
:func:`qufin.analysis.time_series_momentum` (Moskowitz, Ooi & Pedersen 2012):

    s_t    =  sign( log p_t − log p_{t−lookback} )           (trend sign)
    pos_t  =  s_t · min( leverage_cap, target_vol / σ̂_{t,ann} )

The position is held from bar ``t`` to ``t + 1`` and earns the realised log
return.  Sizing to a constant ex-ante volatility target keeps PnL volatility
stable across regimes and across assets, which is what makes TSMOM poolable
into a diversified trend portfolio.

Trainable parameters
--------------------
    lookback     trend-formation window
    vol_window   realised-volatility estimation window
    target_vol   annualised volatility target

``fit`` performs an in-sample grid search over these to maximise the
annualised Sharpe ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..analysis.momentum import time_series_momentum
from ..utils import to_numpy_1d as _to_numpy_1d

_SeriesLike = np.ndarray | pl.Series
_ANNUAL = math.sqrt(252.0)


# ---------------------------------------------------------------------------
# Parameters and result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TSMOMParams:
    """Hyperparameters of :class:`TimeSeriesMomentumStrategy`.

    Attributes
    ----------
    lookback : int
        Trend-formation window in bars.  Must be >= 2.
    vol_window : int
        Realised-volatility estimation window.  Must be >= 2.
    target_vol : float
        Desired annualised volatility of the position.
    leverage_cap : float
        Hard cap on |position|.
    ann : float
        Periods per year used to annualise volatility.
    """

    lookback: int = 252
    vol_window: int = 60
    target_vol: float = 0.15
    leverage_cap: float = 3.0
    ann: float = 252.0

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.vol_window < 2:
            raise ValueError("vol_window must be >= 2")
        if self.target_vol <= 0:
            raise ValueError("target_vol must be > 0")
        if self.leverage_cap <= 0:
            raise ValueError("leverage_cap must be > 0")
        if self.ann <= 0:
            raise ValueError("ann must be > 0")


@dataclass(slots=True)
class TSMOMResult:
    """Output of :meth:`TimeSeriesMomentumStrategy.run`."""

    signal: np.ndarray  # (T,) trend sign in {−1, 0, +1}
    position: np.ndarray  # (T,) vol-targeted position
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
                "signal": self.signal,
                "position": self.position,
                "strat_ret": strat_ret,
            }
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "Time-Series Momentum Summary",
                "=" * 35,
                f"  Sharpe (ann.)  : {self.sharpe:.4f}",
                f"  Total return   : {self.total_return * 100:.2f}%",
                f"  Max drawdown   : {self.max_drawdown * 100:.2f}%",
                f"  Realised vol   : {self.realised_vol_ann * 100:.2f}%",
                f"  Avg |position| : {self.avg_abs_position:.3f}",
            ]
        )


@dataclass(slots=True)
class TSMOMTrainResult:
    """Output of :meth:`TimeSeriesMomentumStrategy.fit`."""

    params: TSMOMParams
    sharpe: float
    grid: list[tuple[int, int, float, float]]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class TimeSeriesMomentumStrategy:
    """Volatility-targeted time-series-momentum backtest.

    Parameters
    ----------
    params : TSMOMParams, optional

    Examples
    --------
    >>> strat = TimeSeriesMomentumStrategy()
    >>> result = strat.run(prices)
    >>> print(result.summary())
    """

    def __init__(self, params: TSMOMParams | None = None) -> None:
        self.params: TSMOMParams = params if params is not None else TSMOMParams()

    def run(self, prices: _SeriesLike) -> TSMOMResult:
        """Causal backtest on a single positive price series."""
        p = self.params
        arr = _to_numpy_1d(prices)
        factor = time_series_momentum(
            arr,
            lookback=p.lookback,
            vol_window=p.vol_window,
            target_vol=p.target_vol,
            leverage_cap=p.leverage_cap,
            ann=p.ann,
        )
        return TSMOMResult(
            signal=factor.signal,
            position=factor.weights,
            log_returns=factor.factor_returns,
            prices=arr,
        )

    def fit(
        self,
        prices: _SeriesLike,
        *,
        lookbacks: tuple[int, ...] = (63, 126, 252),
        vol_windows: tuple[int, ...] = (20, 60),
        target_vols: tuple[float, ...] = (0.15,),
    ) -> TSMOMTrainResult:
        """Grid-search ``(lookback, vol_window, target_vol)`` for the best Sharpe.

        The best configuration is stored on ``self.params`` so a subsequent
        :meth:`run` uses it.
        """
        arr = _to_numpy_1d(prices)
        best_sharpe = -math.inf
        best_params = self.params
        grid: list[tuple[int, int, float, float]] = []
        for lb in lookbacks:
            for vw in vol_windows:
                for tv in target_vols:
                    if arr.shape[0] <= lb + vw + 1:
                        continue
                    candidate = TSMOMParams(
                        lookback=lb,
                        vol_window=vw,
                        target_vol=tv,
                        leverage_cap=self.params.leverage_cap,
                        ann=self.params.ann,
                    )
                    sharpe = TimeSeriesMomentumStrategy(candidate).run(arr).sharpe
                    grid.append((lb, vw, tv, sharpe))
                    if math.isfinite(sharpe) and sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = candidate
        if not grid:
            raise ValueError("no grid configuration fits the series length.")
        self.params = best_params
        return TSMOMTrainResult(params=best_params, sharpe=best_sharpe, grid=grid)
