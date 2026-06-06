"""
Market-microstructure subpackage — inferring liquidity, order-flow and
informed-trading from trades and quotes.

Layout
------
    _types          PriceImpactResult, VPINResult result containers + helpers
    _kernels        numba-jitted trade-sign rules and VPIN bucket packing
    classification  trade-sign inference: tick / quote / Lee-Ready / EMO / BVC
    spread          quoted / effective / realized spreads (quotes) and Roll /
                    Corwin-Schultz / Abdi-Ranaldo estimators (prices, OHLC)
    impact          Kyle's λ, Hasbrouck's λ, Amihud illiquidity
    flow            signed volume, rolling trade imbalance, Cont-Kukanov-Stoikov
                    order-flow imbalance
    vpin            volume-synchronised Probability of Informed Trading
    pin             structural Probability of Informed Trading (Easley et al. MLE)

Inputs are plain ``np.ndarray`` / ``pl.Series`` columns (trade prices and sizes,
best bid/ask and depths, or OHLC bars), so the tools work with any vendor feed.
Trades can be supplied directly from a ``TICK_SCHEMA`` frame via
:func:`as_trade_arrays`.

Quick start
-----------
    from qufin.microstructure import lee_ready, effective_spread, kyle_lambda, vpin

    q = lee_ready(price, bid, ask)                    # aggressor side
    eff = effective_spread(price, bid, ask, signs=q)  # cost paid vs midpoint
    impact = kyle_lambda(np.diff(price), q[1:] * size[1:])
    tox = vpin(close, volume, n_buckets=200, window=50).vpin
"""

from __future__ import annotations

from ._types import (
    TICK_SCHEMA,
    PriceImpactResult,
    VPINResult,
    as_trade_arrays,
)
from .classification import bvc, emo_rule, lee_ready, quote_rule, tick_rule
from .flow import order_flow_imbalance, signed_volume, trade_imbalance
from .impact import amihud_illiquidity, hasbrouck_lambda, kyle_lambda
from .pin import PINResult, pin
from .spread import (
    abdi_ranaldo,
    corwin_schultz,
    effective_spread,
    quoted_spread,
    realized_spread,
    roll_spread,
)
from .vpin import vpin

__all__ = [
    # Types / schemas
    "TICK_SCHEMA",
    "PriceImpactResult",
    "VPINResult",
    "PINResult",
    "as_trade_arrays",
    # Classification
    "tick_rule",
    "quote_rule",
    "lee_ready",
    "emo_rule",
    "bvc",
    # Spread
    "quoted_spread",
    "effective_spread",
    "realized_spread",
    "roll_spread",
    "corwin_schultz",
    "abdi_ranaldo",
    # Impact / illiquidity
    "kyle_lambda",
    "hasbrouck_lambda",
    "amihud_illiquidity",
    # Order flow
    "signed_volume",
    "trade_imbalance",
    "order_flow_imbalance",
    # Informed trading
    "vpin",
    "pin",
]
