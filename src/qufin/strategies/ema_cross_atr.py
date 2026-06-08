"""
EMA-crossover trend strategy with an ATR trailing stop.

Architecture
------------
A deliberately small long-only trend strategy designed to exercise the
qufin trading stack end-to-end (data → backtest → training → live paper
broker). It is **not** intended as a production alpha source.

Step 1 — Trend signal
    Two EMAs on the close. Go long when the fast EMA crosses above the
    slow EMA; flatten when it crosses back.

        long_t  =  EMA_fast(close)_t  >  EMA_slow(close)_t

Step 2 — ATR trailing stop
    Once long, track the highest close since entry and exit if price
    falls more than ``atr_mult × ATR(atr_window)`` below that high. This
    caps the downside on trend reversals before the slow EMA catches up.

Step 3 — Sizing
    The strategy emits ``TARGET_WEIGHT`` signals — the engine converts
    them to share deltas using the latest mark and account equity. Weight
    is either ``target_weight`` (default 1.0 = fully invested) when long
    or 0.0 when flat. Short side is intentionally omitted.

Trainable parameters
--------------------
    fast_window    EMA span for the fast leg (typically 5 – 50)
    slow_window    EMA span for the slow leg (must exceed fast_window)
    atr_window     ATR span for the trailing stop
    atr_mult       Trailing-stop distance, in ATR units

All four are jointly searchable via ``qufin.trading.training.GridSearch``;
see ``scripts/train_ibkr_demo.py`` for an end-to-end example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..indicators import atr, ema
from ..trading._types import Signal, SignalKind, SymbolOrContract
from ..trading.strategy.base import StrategyBase, StrategyContext


@dataclass(slots=True)
class EmaCrossAtrParams:
    """Hyperparameters of :class:`EmaCrossAtrStrategy`."""

    fast_window: int = 12
    slow_window: int = 26
    atr_window: int = 14
    atr_mult: float = 3.0
    target_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.fast_window < 2:
            raise ValueError("fast_window must be >= 2")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be > fast_window")
        if self.atr_window < 2:
            raise ValueError("atr_window must be >= 2")
        if self.atr_mult <= 0.0 or not math.isfinite(self.atr_mult):
            raise ValueError("atr_mult must be positive and finite")
        if not (0.0 < self.target_weight <= 1.0):
            raise ValueError("target_weight must be in (0, 1]")

    @property
    def min_bars(self) -> int:
        """Minimum bars needed before the first non-NaN signal."""
        return max(self.slow_window, self.atr_window) + 2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "atr_window": self.atr_window,
            "atr_mult": self.atr_mult,
            "target_weight": self.target_weight,
        }


@dataclass(slots=True)
class EmaCrossAtrStrategy(StrategyBase):
    """Long-only EMA-cross with ATR trailing-stop exit.

    Parameters
    ----------
    symbol  Ticker the strategy trades (single asset by design).
    params  Hyperparameters; defaults give a 12/26/14 ATR3 setup.
    """

    symbol: str = "SPY"
    params: EmaCrossAtrParams = field(default_factory=EmaCrossAtrParams)
    _in_long: bool = field(default=False, init=False)
    _entry_close: float = field(default=0.0, init=False)
    _highest_close: float = field(default=0.0, init=False)

    def on_start(self, ctx: StrategyContext) -> None:
        del ctx
        self._in_long = False
        self._entry_close = 0.0
        self._highest_close = 0.0

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        frame = ctx.history.get(self.symbol)
        if frame is None or frame.height < self.params.min_bars:
            return []

        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        high = frame["high"].to_numpy().astype(np.float64, copy=False)
        low = frame["low"].to_numpy().astype(np.float64, copy=False)

        fast = ema(close, self.params.fast_window)
        slow = ema(close, self.params.slow_window)
        atr_arr = atr(high, low, close, window=self.params.atr_window)

        fast_now = float(fast[-1])
        slow_now = float(slow[-1])
        atr_now = float(atr_arr[-1])
        close_now = float(close[-1])
        if not (math.isfinite(fast_now) and math.isfinite(slow_now) and math.isfinite(atr_now)):
            return []

        long_signal = fast_now > slow_now

        if self._in_long:
            self._highest_close = max(self._highest_close, close_now)
            trail_stop = self._highest_close - self.params.atr_mult * atr_now
            stop_hit = close_now <= trail_stop
            cross_down = not long_signal
            if stop_hit or cross_down:
                self._in_long = False
                return [self._weight_signal(0.0, tag="exit_stop" if stop_hit else "exit_cross")]
            return []

        # Currently flat — only enter on a fresh cross-up.
        if long_signal and not bool(fast[-2] > slow[-2]):
            self._in_long = True
            self._entry_close = close_now
            self._highest_close = close_now
            return [self._weight_signal(self.params.target_weight, tag="entry_cross")]
        return []

    def _weight_signal(self, weight: float, *, tag: str) -> Signal:
        asset: SymbolOrContract = self.symbol
        return Signal(asset=asset, kind=SignalKind.TARGET_WEIGHT, value=weight, tag=tag)


def make_strategy(params: dict[str, float | int], *, symbol: str = "SPY") -> EmaCrossAtrStrategy:
    """Factory used by ``GridSearch`` to build a fresh strategy per point."""
    return EmaCrossAtrStrategy(
        symbol=symbol,
        params=EmaCrossAtrParams(
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            atr_window=int(params["atr_window"]),
            atr_mult=float(params["atr_mult"]),
            target_weight=float(params.get("target_weight", 1.0)),
        ),
    )


@dataclass(slots=True)
class CurrentState:
    """Snapshot used by the live runner to decide whether to trade.

    Attributes
    ----------
    target_weight     0.0 (flat) or ``params.target_weight`` (long).
    fast_ema          Last value of the fast EMA.
    slow_ema          Last value of the slow EMA.
    atr               Last ATR value.
    last_close        Most recent bar close.
    trail_stop        Active trailing-stop price when long; ``nan`` when flat.
    highest_close     Highest close observed since current entry; ``nan`` when flat.
    """

    target_weight: float
    fast_ema: float
    slow_ema: float
    atr: float
    last_close: float
    trail_stop: float
    highest_close: float


def current_state(
    bars_close: np.ndarray, bars_high: np.ndarray, bars_low: np.ndarray, params: EmaCrossAtrParams
) -> CurrentState:
    """Replay a full bar history and return the strategy's *current* desired state.

    Pure function — no broker calls, no I/O. The live runner uses this each
    poll cycle to compute "what should the position be right now?", then
    reconciles against the broker's reported position.

    Parameters
    ----------
    bars_close, bars_high, bars_low
        Float64 arrays of equal length, oldest-first.
    params
        Strategy hyperparameters.
    """
    if not (len(bars_close) == len(bars_high) == len(bars_low)):
        raise ValueError("close/high/low arrays must have the same length")
    if len(bars_close) < params.min_bars:
        raise ValueError(f"need >= {params.min_bars} bars, got {len(bars_close)}")

    fast = ema(bars_close, params.fast_window)
    slow = ema(bars_close, params.slow_window)
    atr_arr = atr(bars_high, bars_low, bars_close, window=params.atr_window)

    in_long = False
    highest_close = math.nan
    trail_stop = math.nan

    # Iterate from the first bar where all three series are non-NaN.
    first_valid = max(params.slow_window, params.atr_window) + 1
    for t in range(first_valid, len(bars_close)):
        f_t = float(fast[t])
        s_t = float(slow[t])
        f_prev = float(fast[t - 1])
        s_prev = float(slow[t - 1])
        a_t = float(atr_arr[t])
        c_t = float(bars_close[t])
        if not (math.isfinite(f_t) and math.isfinite(s_t) and math.isfinite(a_t)):
            continue
        long_signal = f_t > s_t
        if in_long:
            highest_close = max(highest_close, c_t)
            trail_stop = highest_close - params.atr_mult * a_t
            if c_t <= trail_stop or not long_signal:
                in_long = False
                highest_close = math.nan
                trail_stop = math.nan
        else:
            if long_signal and not (f_prev > s_prev):
                in_long = True
                highest_close = c_t
                trail_stop = c_t - params.atr_mult * a_t

    return CurrentState(
        target_weight=params.target_weight if in_long else 0.0,
        fast_ema=float(fast[-1]),
        slow_ema=float(slow[-1]),
        atr=float(atr_arr[-1]),
        last_close=float(bars_close[-1]),
        trail_stop=trail_stop,
        highest_close=highest_close,
    )
