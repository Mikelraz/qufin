"""
GEX profile — dealer exposure swept across a grid of hypothetical spot levels.

The profile lets you see *how* dealer hedging needs change as spot moves: how
much SPX they need to buy if the index rallies 1%, how the gamma flip moves
in response, where vanna kicks in if volatility shifts.  All four exposures
(GEX, DEX, VEX, charm) are computed on the same spot grid in one pass.
"""

from __future__ import annotations

import numpy as np

from .._kernels import greeks_at_spot
from .._types import GEXProfile, OptionChain
from .exposure import DealerConvention, dealer_signs


def gex_profile(
    chain: OptionChain,
    *,
    convention: DealerConvention = DealerConvention.CLASSIC,
    signs: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    spot_range_pct: float = 0.15,
    n_spot: int = 201,
) -> GEXProfile:
    """
    Sweep dealer exposures across a spot grid centred on the chain's spot.

    Parameters
    ----------
    chain            OptionChain snapshot.
    convention       Dealer-sign convention (default classic).
    signs            Custom per-contract signs (overrides convention).
    sigma            Override IVs (default: chain.iv).
    spot_range_pct   Half-width of the grid as a fraction of spot.
    n_spot           Grid resolution.
    """
    K = chain.strikes()
    T = chain.time_to_expiry()
    s = chain.implied_vols() if sigma is None else np.asarray(sigma, dtype=np.float64)
    r = np.full(K.shape[0], chain.r, dtype=np.float64)
    q = np.full(K.shape[0], chain.q, dtype=np.float64)
    oi = chain.open_interest()
    is_call = chain.is_call()

    if signs is None:
        sgn = dealer_signs(is_call, convention=convention)
    else:
        sgn = signs.astype(np.float64)

    lo = chain.spot * (1.0 - spot_range_pct)
    hi = chain.spot * (1.0 + spot_range_pct)
    spot_grid = np.linspace(lo, hi, n_spot)

    delta_m, gamma_m, vanna_m, charm_m = greeks_at_spot(
        spot_grid, K, T, r, q, s, is_call.astype(np.uint8)
    )

    notional = oi * chain.multiplier * sgn
    s_sq_pct = spot_grid * spot_grid * 0.01

    gex = (gamma_m @ notional) * s_sq_pct
    dex = (delta_m @ notional) * spot_grid
    vex = (vanna_m @ notional) * spot_grid
    charm = (charm_m @ notional) * spot_grid

    sign = np.sign(gex)
    changes = np.where(np.diff(sign) != 0)[0]
    flip: float | None = None
    if changes.size > 0:
        i = int(changes[0])
        g0, g1 = gex[i], gex[i + 1]
        s0, s1 = spot_grid[i], spot_grid[i + 1]
        flip = float(s0) if g1 == g0 else float(s0 - g0 * (s1 - s0) / (g1 - g0))

    return GEXProfile(
        spot_grid=spot_grid,
        gex=gex,
        dex=dex,
        vex=vex,
        charm=charm,
        flip_level=flip,
        spot=chain.spot,
    )
