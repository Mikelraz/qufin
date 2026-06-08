"""
Long-call trend follower (options version of EMA-cross + ATR demo).

Architecture
------------
A minimal long-only options strategy. The trend signal is the same EMA
crossover used by ``ema_cross_atr`` on the underlying's close. Instead
of buying shares, on a fresh bullish cross we open one (or more) long
call contracts at a target strike-moneyness and target days-to-expiry;
on a bearish cross — or once the held option's DTE falls below an exit
threshold — we close the position.

The point is to exercise the *options* path of the trading stack
(``OptionContract`` orders, ``OptionsEngine`` marks, IBKR option-chain
plumbing) without piling on options-specific signal complexity.

Trainable parameters
--------------------
    fast_window        EMA span for the fast leg
    slow_window        EMA span for the slow leg (must exceed fast_window)
    strike_moneyness   Strike / spot ratio at entry (1.00 = ATM,
                       1.02 = 2% OTM call, 0.98 = 2% ITM call)
    target_dte         Target days-to-expiry at entry (e.g. 30, 45, 60)
    exit_dte           Close the position when its DTE drops to or below this
    contracts          Number of contracts to trade (fixed at construction)

Live use
--------
The strategy emits fully-specified ``Order`` payloads via
``SignalKind.ORDER``; in backtest mode the synthetic ``OptionContract``
has its expiry computed exactly as ``today + target_dte`` days. The live
runner replaces that synthetic contract with the closest tradable
(expiry, strike) pair pulled from the IBKR option chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ..indicators import ema
from ..options._types import CALL, OptionContract
from ..trading._types import Order, OrderType, Signal, SignalKind, TimeInForce
from ..trading.strategy.base import StrategyBase, StrategyContext


@dataclass(slots=True)
class EmaCrossLongCallParams:
    """Hyperparameters of :class:`EmaCrossLongCallStrategy`."""

    fast_window: int = 12
    slow_window: int = 26
    strike_moneyness: float = 1.00
    target_dte: int = 30
    exit_dte: int = 5
    contracts: int = 1
    strike_step: float = 1.0  # round strike to this $-grid (1.0 = nearest dollar)

    def __post_init__(self) -> None:
        if self.fast_window < 2:
            raise ValueError("fast_window must be >= 2")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be > fast_window")
        if not (0.5 <= self.strike_moneyness <= 1.5):
            raise ValueError("strike_moneyness must be in [0.5, 1.5]")
        if self.target_dte < 7:
            raise ValueError("target_dte must be >= 7")
        if self.exit_dte < 0:
            raise ValueError("exit_dte must be >= 0")
        if self.exit_dte >= self.target_dte:
            raise ValueError("exit_dte must be < target_dte")
        if self.contracts < 1:
            raise ValueError("contracts must be >= 1")
        if self.strike_step <= 0.0:
            raise ValueError("strike_step must be > 0")

    @property
    def min_bars(self) -> int:
        return self.slow_window + 2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "strike_moneyness": self.strike_moneyness,
            "target_dte": self.target_dte,
            "exit_dte": self.exit_dte,
            "contracts": self.contracts,
            "strike_step": self.strike_step,
        }


def round_to_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step`` (must be > 0)."""
    return round(value / step) * step


def select_call_contract(
    *, spot: float, as_of: date, params: EmaCrossLongCallParams, underlying: str
) -> OptionContract:
    """Build the synthetic call contract used at entry in backtest.

    Strike = ``round(spot × strike_moneyness, step)``; expiry =
    ``as_of + target_dte`` calendar days. The live runner replaces this
    with the closest tradable (expiry, strike) from the real chain.
    """
    raw_strike = spot * params.strike_moneyness
    strike = round_to_step(raw_strike, params.strike_step)
    expiry = as_of + timedelta(days=params.target_dte)
    return OptionContract(
        strike=float(strike),
        expiry=expiry,
        option_type=CALL,
        underlying=underlying,
    )


@dataclass(slots=True)
class EmaCrossLongCallStrategy(StrategyBase):
    """Long-call trend-follower used by both backtest and live demo.

    Parameters
    ----------
    symbol    Underlying ticker (single asset by design).
    params    Hyperparameters; defaults give 12/26 EMA, ATM, 30/5 DTE.
    """

    symbol: str = "AAPL"
    params: EmaCrossLongCallParams = field(default_factory=EmaCrossLongCallParams)
    _open_contract: OptionContract | None = field(default=None, init=False)

    def on_start(self, ctx: StrategyContext) -> None:
        del ctx
        self._open_contract = None

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        frame = ctx.history.get(self.symbol)
        if frame is None or frame.height < self.params.min_bars:
            return []

        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        fast = ema(close, self.params.fast_window)
        slow = ema(close, self.params.slow_window)
        if not (math.isfinite(fast[-1]) and math.isfinite(slow[-1])):
            return []

        long_signal = bool(fast[-1] > slow[-1])
        long_prev = bool(fast[-2] > slow[-2])
        spot = float(close[-1])
        as_of = ctx.bar.timestamp.date()

        # Held → check exit conditions first.
        if self._open_contract is not None:
            held = self._open_contract
            dte = (held.expiry - as_of).days
            close_for_cross = not long_signal
            close_for_dte = dte <= self.params.exit_dte
            if close_for_cross or close_for_dte:
                tag = "exit_cross" if close_for_cross else "exit_dte"
                self._open_contract = None
                return [self._market_order(held, qty=-self.params.contracts, tag=tag)]
            return []

        # Flat → only open on a fresh cross-up.
        if long_signal and not long_prev:
            contract = select_call_contract(
                spot=spot, as_of=as_of, params=self.params, underlying=self.symbol
            )
            self._open_contract = contract
            return [self._market_order(contract, qty=self.params.contracts, tag="entry_cross")]
        return []

    def on_fill(self, fill, ctx: StrategyContext) -> None:  # type: ignore[no-untyped-def]
        del ctx
        # If a sell-to-close fills, clear our internal handle even if the
        # contract object differs from the one we built (e.g. when the live
        # runner substitutes a real contract).
        if (
            isinstance(fill.asset, OptionContract)
            and self._open_contract is not None
            and fill.qty < 0
            and fill.asset.underlying == self._open_contract.underlying
        ):
            self._open_contract = None

    def _market_order(self, asset: OptionContract, *, qty: int, tag: str) -> Signal:
        order = Order(
            asset=asset,
            qty=float(qty),
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            tag=tag,
        )
        return Signal(asset=asset, kind=SignalKind.ORDER, value=0.0, order=order, tag=tag)


def make_strategy(
    params: dict[str, float | int], *, symbol: str = "AAPL"
) -> EmaCrossLongCallStrategy:
    """Factory used by ``GridSearch`` to build a fresh strategy per point."""
    return EmaCrossLongCallStrategy(
        symbol=symbol,
        params=EmaCrossLongCallParams(
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            strike_moneyness=float(params["strike_moneyness"]),
            target_dte=int(params["target_dte"]),
            exit_dte=int(params["exit_dte"]),
            contracts=int(params.get("contracts", 1)),
            strike_step=float(params.get("strike_step", 1.0)),
        ),
    )


@dataclass(slots=True)
class CurrentState:
    """Snapshot used by the live runner to decide whether to trade.

    Attributes
    ----------
    desire_long       True if the strategy wants to be long a call right now.
    on_fresh_cross    True iff *this bar* produced the cross that opened a new
                      position (i.e. the runner should buy if currently flat).
    fast_ema, slow_ema, last_close   Diagnostics for the runner to log.
    suggested_contract  The synthetic contract the strategy would open (the
                        live runner uses moneyness/dte from this to look up the
                        closest tradable strike/expiry in the chain). ``None``
                        when ``desire_long`` is False.
    """

    desire_long: bool
    on_fresh_cross: bool
    fast_ema: float
    slow_ema: float
    last_close: float
    suggested_contract: OptionContract | None


def current_state(
    bars_close: np.ndarray, as_of: date, params: EmaCrossLongCallParams, *, symbol: str
) -> CurrentState:
    """Replay a full bar history and return the strategy's *current* desire.

    Pure function — no broker calls, no I/O. The live runner uses this each
    poll cycle to compute "what should the position be right now?".
    """
    if len(bars_close) < params.min_bars:
        raise ValueError(f"need >= {params.min_bars} bars, got {len(bars_close)}")
    fast = ema(bars_close, params.fast_window)
    slow = ema(bars_close, params.slow_window)
    long_now = bool(fast[-1] > slow[-1])
    long_prev = bool(fast[-2] > slow[-2])
    on_cross = long_now and not long_prev

    suggested: OptionContract | None = None
    if long_now:
        suggested = select_call_contract(
            spot=float(bars_close[-1]), as_of=as_of, params=params, underlying=symbol
        )
    return CurrentState(
        desire_long=long_now,
        on_fresh_cross=on_cross,
        fast_ema=float(fast[-1]),
        slow_ema=float(slow[-1]),
        last_close=float(bars_close[-1]),
        suggested_contract=suggested,
    )
