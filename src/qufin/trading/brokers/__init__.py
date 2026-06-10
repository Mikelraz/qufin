"""Broker abstractions and concrete adapters.

A ``Broker`` is the unified async surface every adapter implements. The
in-process ``PaperBroker`` wraps the backtest engine; ``AlpacaBroker`` and
``IBKRBroker`` are thin shims over their respective SDKs.
"""

from __future__ import annotations

from .._types import OrderRejectedError, OrderStatus
from .base import Broker
from .paper import PaperBroker
from .quotes import MarketDataType, Quote, QuoteSession, quote_option, quote_options, quote_stock

__all__ = [
    "AlpacaBroker",
    "Broker",
    "IBKRBroker",
    "MarketDataType",
    "OrderRejectedError",
    "OrderStatus",
    "PaperBroker",
    "Quote",
    "QuoteSession",
    "TradeRepublicBroker",
    "quote_option",
    "quote_options",
    "quote_stock",
]


def __getattr__(name: str) -> type:
    # Lazy import keeps optional broker SDK deps unloaded until needed.
    if name == "AlpacaBroker":
        from .alpaca import AlpacaBroker
        return AlpacaBroker
    if name == "IBKRBroker":
        from .ibkr import IBKRBroker
        return IBKRBroker
    if name == "TradeRepublicBroker":
        from .trade_republic import TradeRepublicBroker
        return TradeRepublicBroker
    raise AttributeError(name)
