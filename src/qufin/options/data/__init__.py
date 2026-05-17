"""
Option-chain data loaders.

Each loader returns an ``OptionChain`` with the canonical
``qufin.options._types.CHAIN_SCHEMA`` so downstream pricing/GEX code is
source-agnostic.  Paid providers (Polygon, Tradier, ORATS, …) can be plugged
in by adding a new module here that emits the same schema.
"""

from __future__ import annotations

from .yfinance import load_chain_yfinance

__all__ = ["load_chain_yfinance"]
