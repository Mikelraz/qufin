"""
Core dataclasses, enums, and protocols for the trading subpackage.

Conventions
-----------
* Quantities are signed: positive = long, negative = short.
* Cash, prices, and PnL are ``float64``.
* Timestamps are tz-aware UTC (``zoneinfo.ZoneInfo('UTC')``); the engine's
  bar schema reuses ``qufin.wyckoff._types.BAR_SCHEMA``.
* Options reuse ``qufin.options._types.OptionContract`` directly so option
  pricing and greeks computations interoperate without re-wrapping.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Protocol, TypeAlias

import numpy as np
import polars as pl

from ..options._types import OptionContract

CASH_TOL: float = 1e-9  # tolerance used by the round-trip parity test


# ---------------------------------------------------------------------------
# Enums


class Side(Enum):
    """Trade direction for a single fill or order."""

    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        return self.value


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class AssetKind(Enum):
    EQUITY = "equity"
    OPTION = "option"


class SignalKind(Enum):
    """How a strategy expresses intent.

    ``TARGET_WEIGHT`` lets the engine convert a desired portfolio weight to a
    delta in shares using the latest mark. ``TARGET_QTY`` is an absolute
    target position size. ``ORDER`` bypasses sizing — the strategy emitted
    a fully-specified ``Order`` itself.
    """

    TARGET_WEIGHT = "target_weight"
    TARGET_QTY = "target_qty"
    ORDER = "order"


# ---------------------------------------------------------------------------
# Identifiers and IDs

OrderId: TypeAlias = str
SymbolOrContract: TypeAlias = str | OptionContract

_ORDER_ID_COUNTER = itertools.count(1)


def new_order_id(prefix: str = "ord") -> OrderId:
    """Monotonic deterministic order id used by the engine and paper broker."""
    return f"{prefix}-{next(_ORDER_ID_COUNTER):08d}"


# ---------------------------------------------------------------------------
# Bar event (engine-internal alias around the polars row)


@dataclass(slots=True, frozen=True)
class BarEvent:
    """A single bar emitted by the engine clock.

    The engine consumes polars frames that match
    ``qufin.wyckoff._types.BAR_SCHEMA``. ``BarEvent`` is the materialised view
    of one row that strategies see in ``on_bar``.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    index: int  # row position in the source frame — useful for slicing history


# ---------------------------------------------------------------------------
# Orders and fills


@dataclass(slots=True, frozen=True)
class Order:
    """A trade request emitted by a strategy and routed by a broker.

    Quantity is signed: positive = buy/long-side, negative = sell/short-side.
    For an option contract, ``qty`` is the number of contracts (not shares).
    """

    asset: SymbolOrContract
    qty: float
    order_type: OrderType = OrderType.MARKET
    tif: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    client_id: OrderId = ""  # filled in by ``with_id`` if blank
    tag: str = ""  # free-form label set by the strategy (e.g. "entry", "stop")

    def __post_init__(self) -> None:
        if self.qty == 0.0 or not np.isfinite(self.qty):
            raise ValueError(f"order qty must be non-zero and finite, got {self.qty!r}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires stop_price")

    @property
    def side(self) -> Side:
        return Side.BUY if self.qty > 0 else Side.SELL

    @property
    def asset_kind(self) -> AssetKind:
        return AssetKind.OPTION if isinstance(self.asset, OptionContract) else AssetKind.EQUITY

    def with_id(self, order_id: OrderId) -> Order:
        return Order(
            asset=self.asset,
            qty=self.qty,
            order_type=self.order_type,
            tif=self.tif,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            client_id=order_id,
            tag=self.tag,
        )


@dataclass(slots=True, frozen=True)
class Fill:
    """An executed slice of an order."""

    order_id: OrderId
    asset: SymbolOrContract
    timestamp: datetime
    qty: float  # signed; sums (over partials) to the parent order qty
    price: float
    commission: float = 0.0
    slippage: float = 0.0  # bps applied vs. reference price; informational only


# ---------------------------------------------------------------------------
# Live order state

# Broker-native status strings we treat as working / terminal / rejected.
# IBKR has no explicit "Rejected" status — rejections surface as ``Inactive``
# (with an accompanying error) or via a non-None ``reject_reason``.
_WORKING_STATUSES = frozenset({"PreSubmitted", "Submitted"})
_DONE_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


@dataclass(slots=True, frozen=True)
class OrderStatus:
    """Broker-agnostic snapshot of a live order's state.

    ``status`` carries the broker-native string (e.g. IBKR's ``Submitted`` /
    ``Filled`` / ``Inactive``). The boolean properties classify it so callers
    don't hard-code those strings.
    """

    order_id: OrderId
    status: str
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float | None = None
    reject_reason: str | None = None

    @property
    def is_working(self) -> bool:
        return self.status in _WORKING_STATUSES

    @property
    def is_filled(self) -> bool:
        return self.status == "Filled"

    @property
    def is_done(self) -> bool:
        return self.status in _DONE_STATUSES

    @property
    def is_rejected(self) -> bool:
        return self.reject_reason is not None or self.status == "Inactive"


class OrderRejectedError(RuntimeError):
    """Raised when a broker rejects or cancels an order during submission."""

    def __init__(self, status: OrderStatus) -> None:
        self.status = status
        reason = status.reject_reason or "no reason reported"
        super().__init__(f"order {status.order_id} not accepted ({status.status}): {reason}")


# ---------------------------------------------------------------------------
# Positions and account


@dataclass(slots=True)
class Position:
    """Open position in a single asset.

    Realised and unrealised PnL are split: realised crystallises on
    closes/flips; unrealised marks against the latest available price.
    Greeks are populated by the options engine for option legs.
    """

    asset: SymbolOrContract
    qty: float
    avg_price: float
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    last_mark: float = 0.0
    multiplier: float = 1.0  # 100 for US equity options, 1 for equities
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.last_mark * self.multiplier

    @property
    def is_option(self) -> bool:
        return isinstance(self.asset, OptionContract)


@dataclass(slots=True)
class AccountSnapshot:
    """Account state at one moment in time."""

    timestamp: datetime
    cash: float
    equity: float  # cash + sum(position.market_value)
    buying_power: float
    margin_used: float = 0.0
    day_pnl: float = 0.0
    total_pnl: float = 0.0

    def to_row(self) -> dict[str, float | datetime]:
        return {
            "timestamp": self.timestamp,
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "margin_used": self.margin_used,
            "day_pnl": self.day_pnl,
            "total_pnl": self.total_pnl,
        }


# ---------------------------------------------------------------------------
# Strategy intents (signals)


@dataclass(slots=True, frozen=True)
class Signal:
    """High-level strategy intent.

    Use ``TARGET_WEIGHT`` for portfolio-style allocators (the engine resolves
    the share delta), ``TARGET_QTY`` for explicit position sizing, or
    ``ORDER`` to pass through a fully-specified ``Order`` (the engine wraps
    it without sizing).
    """

    asset: SymbolOrContract
    kind: SignalKind
    value: float  # weight in [-1, 1] for TARGET_WEIGHT; share count otherwise
    order: Order | None = None  # populated when kind == ORDER
    tag: str = ""

    def __post_init__(self) -> None:
        if self.kind == SignalKind.ORDER and self.order is None:
            raise ValueError("SignalKind.ORDER requires a non-None order")
        if self.kind == SignalKind.TARGET_WEIGHT and not (-1.0 <= self.value <= 1.0):
            raise ValueError(f"TARGET_WEIGHT must be in [-1, 1], got {self.value}")


# ---------------------------------------------------------------------------
# Execution-model protocols


class SlippageModel(Protocol):
    """Pluggable price adjustment applied to fills."""

    def adjust(self, *, ref_price: float, qty: float, asset_kind: AssetKind) -> float:
        """Return the fill price after slippage; ``qty`` is signed."""
        ...


class CommissionModel(Protocol):
    """Pluggable per-fill commission."""

    def charge(self, *, qty: float, price: float, asset_kind: AssetKind) -> float:
        """Return the commission for one fill; ``qty`` is signed."""
        ...


@dataclass(slots=True, frozen=True)
class NoSlippage:
    """Slippage model that fills at the reference price."""

    def adjust(self, *, ref_price: float, qty: float, asset_kind: AssetKind) -> float:
        del qty, asset_kind
        return ref_price


@dataclass(slots=True, frozen=True)
class PercentSlippage:
    """Symmetric slippage in basis points around the reference price.

    Buys pay ``ref * (1 + bps/10000)``, sells receive ``ref * (1 - bps/10000)``.
    """

    bps: float = 1.0  # one basis point default

    def adjust(self, *, ref_price: float, qty: float, asset_kind: AssetKind) -> float:
        del asset_kind
        sign = 1.0 if qty > 0 else -1.0
        return ref_price * (1.0 + sign * self.bps * 1e-4)


@dataclass(slots=True, frozen=True)
class FixedCommission:
    """Flat per-share / per-contract commission.

    ``per_share`` applies to equities; ``per_contract`` to options. Both are
    multiplied by ``abs(qty)``.
    """

    per_share: float = 0.0
    per_contract: float = 0.65  # IBKR/Alpaca-style options default

    def charge(self, *, qty: float, price: float, asset_kind: AssetKind) -> float:
        del price
        match asset_kind:
            case AssetKind.EQUITY:
                return self.per_share * abs(qty)
            case AssetKind.OPTION:
                return self.per_contract * abs(qty)


# ---------------------------------------------------------------------------
# Backtest report


@dataclass(slots=True)
class BacktestReport:
    """Output of one backtest run.

    Attributes
    ----------
    equity_curve   polars frame with columns: timestamp, cash, equity,
                   buying_power, margin_used, day_pnl, total_pnl.
    trades         polars frame with one row per fill: timestamp, asset,
                   qty, price, commission, slippage, tag.
    positions_eod  optional per-day position snapshots (empty by default).
    summary        scalar metrics dict (Sharpe, MDD, etc.); populated by
                   ``qufin.trading.evaluation.tearsheet``.
    metadata       free-form dict (strategy name, parameters, seed, …).
    """

    equity_curve: pl.DataFrame
    trades: pl.DataFrame
    positions_eod: pl.DataFrame = field(default_factory=pl.DataFrame)
    summary: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def returns(self) -> np.ndarray:
        """Period-over-period equity returns as a float64 numpy array."""
        if self.equity_curve.height < 2:
            return np.empty(0, dtype=np.float64)
        eq = self.equity_curve["equity"].to_numpy().astype(np.float64, copy=False)
        return np.diff(eq) / eq[:-1]


# ---------------------------------------------------------------------------
# Re-export the option contract so users only import from one place

__all__ = [
    "CASH_TOL",
    "AccountSnapshot",
    "AssetKind",
    "BacktestReport",
    "BarEvent",
    "CommissionModel",
    "Fill",
    "FixedCommission",
    "NoSlippage",
    "OptionContract",
    "Order",
    "OrderId",
    "OrderRejectedError",
    "OrderStatus",
    "OrderType",
    "PercentSlippage",
    "Position",
    "Side",
    "Signal",
    "SignalKind",
    "SlippageModel",
    "SymbolOrContract",
    "TimeInForce",
    "new_order_id",
]


# Literal aliases used by other modules (kept at the bottom so the public API
# section above stays readable).
TradeColumns: TypeAlias = Literal[
    "timestamp", "asset", "qty", "price", "commission", "slippage", "tag"
]
