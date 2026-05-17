"""
Zero-gamma flip detection.

The zero-gamma level (a.k.a. "gamma flip", "gamma neutral") is the spot at
which aggregate dealer gamma exposure changes sign.  Above it (positive GEX
regime) dealers dampen moves; below it (negative GEX regime) they amplify.
"""

from __future__ import annotations

import numpy as np

from .._kernels import gamma_at_spot
from .._types import OptionChain
from .exposure import DealerConvention, dealer_signs


def zero_gamma_level(
    chain: OptionChain,
    *,
    convention: DealerConvention = DealerConvention.CLASSIC,
    signs: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    search_pct: float = 0.20,
    n_grid: int = 401,
) -> float | None:
    """
    Numerically find the spot level at which total dealer gamma is zero.

    Sweeps spot over ``[(1 - search_pct)·S, (1 + search_pct)·S]`` on a uniform
    grid, evaluates gamma at every level, and returns the first sign change
    (linearly interpolated).  Returns ``None`` if no sign change is found.
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

    lo = chain.spot * (1.0 - search_pct)
    hi = chain.spot * (1.0 + search_pct)
    grid = np.linspace(lo, hi, n_grid)

    gamma_matrix = gamma_at_spot(grid, K, T, r, q, s)
    weights = oi * chain.multiplier * sgn
    per_spot_gamma = gamma_matrix @ weights
    total_gex = per_spot_gamma * grid * grid * 0.01

    sign = np.sign(total_gex)
    changes = np.where(np.diff(sign) != 0)[0]
    if changes.size == 0:
        return None

    i = int(changes[0])
    g0, g1 = total_gex[i], total_gex[i + 1]
    s0, s1 = grid[i], grid[i + 1]
    if g1 == g0:
        return float(s0)
    return float(s0 - g0 * (s1 - s0) / (g1 - g0))
