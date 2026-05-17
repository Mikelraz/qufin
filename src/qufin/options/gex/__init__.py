"""
Gamma-Exposure (GEX) toolkit.

What "GEX" means here
---------------------
Per-strike dealer gamma exposure, expressed in dollars per 1% spot move:

    GEX_i = γ_i · OI_i · multiplier · S² · 0.01 · dealer_sign_i

where ``dealer_sign_i`` is +1 if dealers are net-long that contract and -1
otherwise.  Aggregated across contracts, positive GEX implies dealers must
*sell* into rallies and *buy* dips (mean-reverting flow); negative GEX
implies the opposite (trend-amplifying flow).

Dealer-sign conventions
-----------------------
* ``DealerConvention.CLASSIC`` — SqueezeMetrics-style heuristic: dealers are
  short calls (sign = -1) and long puts (sign = +1).  This is the default.
* ``DealerConvention.CUSTOM`` — caller supplies a per-contract signed array,
  letting users with better OI-flow data plug in their own positioning.

Public API
----------
* ``aggregate_exposure``   per-strike GEX/DEX/VEX/charm at current spot
* ``zero_gamma_level``     spot at which dealer gamma flips sign
* ``call_wall``, ``put_wall``, ``max_pain``
* ``gex_profile``          sweep exposures across a spot grid
"""

from __future__ import annotations

from .exposure import DealerConvention, aggregate_exposure, dealer_signs
from .flip import zero_gamma_level
from .profile import gex_profile
from .walls import call_wall, max_pain, put_wall

__all__ = [
    "DealerConvention",
    "aggregate_exposure",
    "dealer_signs",
    "zero_gamma_level",
    "gex_profile",
    "call_wall",
    "max_pain",
    "put_wall",
]
