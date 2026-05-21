"""Broker abstractions and concrete adapters.

A ``Broker`` is the unified async surface every adapter implements. The
in-process ``PaperBroker`` wraps the backtest engine; ``AlpacaBroker`` and
``IBKRBroker`` are thin shims over their respective SDKs.
"""

from __future__ import annotations

from .base import Broker
from .paper import PaperBroker

__all__ = ["AlpacaBroker", "Broker", "IBKRBroker", "PaperBroker"]


def __getattr__(name: str) -> type:
    # Lazy import keeps optional broker SDK deps unloaded until needed.
    if name == "AlpacaBroker":
        from .alpaca import AlpacaBroker
        return AlpacaBroker
    if name == "IBKRBroker":
        from .ibkr import IBKRBroker
        return IBKRBroker
    raise AttributeError(name)
