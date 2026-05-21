"""
Trading subpackage — training, backtesting, evaluation, paper-broker connectivity.

Provides a unified event-driven engine that any signal source (rule-based
strategies from ``qufin.strategies``, sklearn pipelines, hand-written rules)
can plug into, plus broker adapters with the same surface for paper and live
execution.

Submodules
----------
    _types        Order, Fill, Position, Signal, AccountSnapshot, BacktestReport, enums
    engine        Event-driven backtest engine (clock, portfolio, execution)
    strategy      Strategy protocol and adapters for existing qufin strategies
    data          OHLC and option-chain loaders (yfinance, alpaca) + parquet cache
    training      Hyperparameter search, walk-forward CV, ML pipelines
    evaluation    Tearsheet, attribution, multi-strategy comparison
    brokers       Broker protocol + PaperBroker, AlpacaBroker, IBKRBroker
"""

from __future__ import annotations

from ._types import (
    CASH_TOL,
    AccountSnapshot,
    AssetKind,
    BacktestReport,
    BarEvent,
    CommissionModel,
    Fill,
    FixedCommission,
    NoSlippage,
    Order,
    OrderId,
    OrderType,
    PercentSlippage,
    Position,
    Side,
    Signal,
    SignalKind,
    SlippageModel,
    TimeInForce,
    new_order_id,
)
from .brokers.base import Broker
from .brokers.paper import PaperBroker
from .engine.engine import BacktestEngine
from .strategy.base import Strategy

__all__ = [
    "CASH_TOL",
    "AccountSnapshot",
    "AssetKind",
    "BacktestEngine",
    "BacktestReport",
    "BarEvent",
    "Broker",
    "CommissionModel",
    "Fill",
    "FixedCommission",
    "NoSlippage",
    "Order",
    "OrderId",
    "OrderType",
    "PaperBroker",
    "PercentSlippage",
    "Position",
    "Side",
    "Signal",
    "SignalKind",
    "SlippageModel",
    "Strategy",
    "TimeInForce",
    "new_order_id",
]
