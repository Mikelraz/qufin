"""Wyckoff-Hull confluence strategy with GEX defense and Kelly+LW sizing.

The package decomposes the strategy into five independently-testable
components:

* :mod:`params` — frozen ``ConfluenceParams`` dataclass.
* :mod:`regime` — rolling-HMM macro regime + cash-defense flag.
* :mod:`signals` — five-channel confluence signal engine.
* :mod:`gex_overlay` — SPY options gamma exposure macro defense.
* :mod:`sizing` — half-Kelly + Ledoit-Wolf covariance portfolio sizer.
* :mod:`strategy` — orchestrator subclass of ``qufin.trading.Strategy``.

Quick start
-----------
    >>> from qufin.strategies.confluence import ConfluenceParams, make_strategy
    >>> strat = make_strategy(symbols=["SPY", "QQQ", "IWM"])
"""

from __future__ import annotations

from .gex_overlay import GEXDefense, GEXFlags
from .params import ConfluenceParams
from .regime import RegimeClassifier, RegimeResult
from .signals import (
    ConfluenceSignalEngine,
    ExitReason,
    SignalFrame,
)
from .sizing import KellyCovarianceSizer, SymbolEdge, panel_returns
from .strategy import (
    ConfluenceStrategy,
    GEXChainProvider,
    make_strategy,
)

__all__ = [
    "ConfluenceParams",
    "ConfluenceSignalEngine",
    "ConfluenceStrategy",
    "ExitReason",
    "GEXChainProvider",
    "GEXDefense",
    "GEXFlags",
    "KellyCovarianceSizer",
    "RegimeClassifier",
    "RegimeResult",
    "SignalFrame",
    "SymbolEdge",
    "make_strategy",
    "panel_returns",
]
