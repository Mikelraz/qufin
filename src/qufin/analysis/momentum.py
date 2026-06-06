"""
Momentum factors.

Two complementary constructions of the momentum premium:

* ``time_series_momentum`` — Moskowitz, Ooi & Pedersen (2012): go long an asset
  that has risen over the lookback and short one that has fallen, sizing the
  position to a constant volatility target.  Each asset is judged against *its
  own* past — an absolute, market-direction-bearing signal.
* ``cross_sectional_momentum`` — Jegadeesh & Titman (1993): rank a universe by
  trailing return and go long the winners / short the losers in equal,
  dollar-neutral weights.  A *relative* signal with little net market exposure.

Helpers ``trailing_return`` and ``volatility_scaled_signal`` are exposed for
building custom variants.  Everything is array-in / array-out so the factors
compose with the rest of the toolkit; the matching backtest lives in
:class:`qufin.strategies.TimeSeriesMomentumStrategy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..data._types import to_numpy_1d

_ANNUAL = 252.0

__all__ = [
    "MomentumFactorResult",
    "cross_sectional_momentum",
    "time_series_momentum",
    "trailing_return",
    "volatility_scaled_signal",
]


@dataclass(slots=True, frozen=True)
class MomentumFactorResult:
    """
    Momentum factor output.

    Attributes
    ----------
    signal          Raw momentum signal.  Shape ``(T,)`` for time-series
                    momentum, ``(T, N)`` for cross-sectional (per-asset score).
    weights         Position / portfolio weights, same shape as ``signal``.
                    Cross-sectional weights are dollar-neutral (sum to ~0).
    factor_returns  Realised factor return stream, shape ``(T − 1,)``.
    kind            ``"time_series"`` or ``"cross_sectional"``.
    """

    signal: np.ndarray
    weights: np.ndarray
    factor_returns: np.ndarray
    kind: str

    @property
    def sharpe(self) -> float:
        r = self.factor_returns
        std = float(np.nanstd(r))
        return float(np.nanmean(r) / std * math.sqrt(_ANNUAL)) if std > 0.0 else 0.0

    @property
    def cumulative_return(self) -> float:
        return float(np.nansum(self.factor_returns))


def _to_numpy_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) panel, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def trailing_return(prices: Any, lookback: int, *, log: bool = True) -> np.ndarray:
    """
    Trailing return over ``lookback`` bars, aligned to the price index.

    Element ``t`` is the return from ``t − lookback`` to ``t``; the first
    ``lookback`` entries are ``NaN``.  Log returns by default.
    """
    p = to_numpy_1d(prices)
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")
    n = p.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= lookback:
        return out
    if log:
        if np.any(p <= 0.0):
            raise ValueError("prices must be strictly positive for log returns.")
        out[lookback:] = np.log(p[lookback:] / p[:-lookback])
    else:
        out[lookback:] = p[lookback:] / p[:-lookback] - 1.0
    return out


def _rolling_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling std; ``out[t] = std(returns[t-window+1 : t+1])`` (NaN-aware)."""
    n = returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(returns, window)
    out[window - 1 :] = np.nanstd(view, axis=1)
    return out


def volatility_scaled_signal(
    signal: Any,
    returns: Any,
    *,
    vol_window: int = 60,
    target_vol: float = 0.15,
    leverage_cap: float = 3.0,
    ann: float = _ANNUAL,
) -> np.ndarray:
    """
    Scale a directional ``signal`` to a constant ex-ante volatility target.

    ``position_t = signal_t · min(leverage_cap, target_vol / σ̂_{t,ann})`` where
    ``σ̂_{t,ann}`` is the trailing annualised volatility of ``returns``.

    Parameters
    ----------
    signal       Directional signal aligned to the price index, shape ``(T,)``.
    returns      Per-bar returns *into* each bar, shape ``(T,)`` (element 0 is
                 typically ``NaN``); the realised volatility is estimated from
                 these.
    vol_window   Lookback for the volatility estimate.
    target_vol   Desired annualised volatility of the scaled position.
    leverage_cap Hard cap on ``|position|``.
    ann          Periods per year used to annualise volatility.
    """
    s = to_numpy_1d(signal)
    r = to_numpy_1d(returns)
    if s.shape != r.shape:
        raise ValueError(f"signal and returns must align; got {s.shape} vs {r.shape}.")
    if vol_window < 2:
        raise ValueError(f"vol_window must be >= 2, got {vol_window}.")
    vol_ann = _rolling_vol(r, vol_window) * math.sqrt(ann)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.minimum(leverage_cap, target_vol / vol_ann)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return np.nan_to_num(s, nan=0.0) * scale


def time_series_momentum(
    prices: Any,
    *,
    lookback: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.15,
    leverage_cap: float = 3.0,
    ann: float = _ANNUAL,
) -> MomentumFactorResult:
    """
    Single-asset time-series momentum (Moskowitz-Ooi-Pedersen 2012).

    The signal is the sign of the trailing ``lookback`` return; the position is
    scaled to ``target_vol``.  ``factor_returns[t]`` is the position held from
    bar ``t`` to ``t + 1`` times the realised log return.
    """
    p = to_numpy_1d(prices)
    n = p.shape[0]
    if n <= lookback + vol_window:
        raise ValueError(f"need > {lookback + vol_window} prices; got {n}.")
    if np.any(p <= 0.0):
        raise ValueError("prices must be strictly positive.")

    log_p = np.log(p)
    r_in = np.full(n, np.nan, dtype=np.float64)
    r_in[1:] = np.diff(log_p)

    signal = np.sign(trailing_return(p, lookback, log=True))
    position = volatility_scaled_signal(
        signal,
        r_in,
        vol_window=vol_window,
        target_vol=target_vol,
        leverage_cap=leverage_cap,
        ann=ann,
    )
    factor_returns = position[:-1] * r_in[1:]
    return MomentumFactorResult(
        signal=signal,
        weights=position,
        factor_returns=factor_returns,
        kind="time_series",
    )


def _rank_weights(scores: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Dollar-neutral long-top / short-bottom equal weights from a score vector."""
    n = scores.shape[0]
    finite = np.isfinite(scores)
    n_valid = int(finite.sum())
    n_side = max(1, n_valid // n_quantiles)
    w = np.zeros(n, dtype=np.float64)
    if n_valid < 2 * n_side:
        return w
    order = np.argsort(np.where(finite, scores, -np.inf))
    valid_order = order[n - n_valid :]  # ascending scores among valid assets
    w[valid_order[-n_side:]] = 1.0 / n_side
    w[valid_order[:n_side]] = -1.0 / n_side
    return w


def cross_sectional_momentum(
    prices: Any,
    *,
    lookback: int = 252,
    skip: int = 21,
    holding: int = 21,
    n_quantiles: int = 5,
    returns: Literal["simple", "log"] = "simple",
) -> MomentumFactorResult:
    """
    Cross-sectional momentum on a panel of prices (Jegadeesh-Titman 1993).

    At each rebalance, assets are ranked by their trailing return measured over
    ``[t − lookback − skip, t − skip]`` (the ``skip`` gap excludes the most
    recent bars to avoid short-term reversal).  The top ``1/n_quantiles`` are
    held long and the bottom short, in equal dollar-neutral weights, rebalanced
    every ``holding`` bars.

    Parameters
    ----------
    prices       Price panel, shape ``(T, N)`` — one column per asset.
    lookback     Formation-window length in bars.
    skip         Bars skipped between the formation window and the holding period.
    holding      Rebalance interval in bars.
    n_quantiles  Number of ranking buckets (5 ⇒ quintile long-short).
    returns      Asset return convention used for both scoring and PnL.

    Returns
    -------
    MomentumFactorResult with 2-D ``signal`` / ``weights`` and 1-D
    ``factor_returns`` (the long-short portfolio return per bar).
    """
    p = _to_numpy_2d(prices)
    t_len, n_assets = p.shape
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}.")
    if holding < 1:
        raise ValueError(f"holding must be >= 1, got {holding}.")
    start = lookback + skip
    if t_len <= start + 1:
        raise ValueError(f"need > {start + 1} rows; got {t_len}.")
    if np.any(p <= 0.0):
        raise ValueError("prices must be strictly positive.")

    if returns == "simple":
        asset_ret = p[1:] / p[:-1] - 1.0
    else:
        asset_ret = np.diff(np.log(p), axis=0)

    scores = np.full((t_len, n_assets), np.nan, dtype=np.float64)
    weights = np.zeros((t_len, n_assets), dtype=np.float64)
    cur_w = np.zeros(n_assets, dtype=np.float64)
    for t in range(start, t_len):
        if (t - start) % holding == 0:
            score = np.log(p[t - skip] / p[t - start])
            scores[t] = score
            cur_w = _rank_weights(score, n_quantiles)
        else:
            scores[t] = scores[t - 1]
        weights[t] = cur_w

    factor_returns = np.sum(weights[:-1] * asset_ret, axis=1)
    return MomentumFactorResult(
        signal=scores,
        weights=weights,
        factor_returns=factor_returns,
        kind="cross_sectional",
    )
