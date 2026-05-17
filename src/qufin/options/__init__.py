"""
Options subpackage — pricing, greeks, IV, and GEX (gamma-exposure) analytics.

Layout
------
    _types        OptionContract, Greeks, OptionChain, GEXProfile, StrikeExposure
    _kernels      numba-jitted BS pricing / greeks / IV / spot-grid greeks
    pricing       black_scholes_price, price_chain
    greeks        black_scholes_greeks, greeks_for_chain
    iv            implied_vol, implied_vol_chain (Newton + bisection)
    gex           dealer exposure aggregation, zero-gamma flip, walls, profile
    data          yfinance loader (paid feeds plug in here)

GEX quick-start
---------------
    from qufin.options.data import load_chain_yfinance
    from qufin.options.gex import aggregate_exposure, zero_gamma_level, gex_profile

    chain = load_chain_yfinance("SPY")
    exposure = aggregate_exposure(chain)              # per-strike GEX/DEX/VEX/charm
    flip = zero_gamma_level(chain)                    # gamma-flip spot
    profile = gex_profile(chain, n_spot=401)          # exposures across a spot sweep
"""

from __future__ import annotations

from ._types import (
    CALL,
    CHAIN_SCHEMA,
    PUT,
    GEXProfile,
    Greeks,
    OptionChain,
    OptionContract,
    StrikeExposure,
)
from .gex import (
    DealerConvention,
    aggregate_exposure,
    call_wall,
    dealer_signs,
    gex_profile,
    max_pain,
    put_wall,
    zero_gamma_level,
)
from .greeks import black_scholes_greeks, greeks_for_chain
from .iv import implied_vol, implied_vol_chain
from .pricing import black_scholes_price, price_chain

__all__ = [
    # Types / constants
    "CALL",
    "PUT",
    "CHAIN_SCHEMA",
    "OptionContract",
    "OptionChain",
    "Greeks",
    "GEXProfile",
    "StrikeExposure",
    # Pricing
    "black_scholes_price",
    "price_chain",
    # Greeks
    "black_scholes_greeks",
    "greeks_for_chain",
    # IV
    "implied_vol",
    "implied_vol_chain",
    # GEX
    "DealerConvention",
    "aggregate_exposure",
    "dealer_signs",
    "zero_gamma_level",
    "gex_profile",
    "call_wall",
    "put_wall",
    "max_pain",
]
