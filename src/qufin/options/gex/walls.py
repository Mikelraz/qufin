"""
Strike-level magnet detection: call wall, put wall, max-pain.

These are heuristics — not theorems — but they are widely watched and tend to
act as intraday support/resistance, especially in positive-GEX regimes when
dealer pinning is strong.

Definitions
-----------
* **Call wall**  — strike with the largest *positive* (call) dealer GEX.  Acts
  as resistance because dealers must sell into a rally toward it.
* **Put wall**   — strike with the largest *negative* (put) dealer GEX.  Acts
  as support for the symmetric reason.
* **Max-pain**   — strike at which the aggregate option-holder payoff is
  minimised at expiry.  Old-school OI metric, ignores greeks.
"""

from __future__ import annotations

import numpy as np

from .._types import OptionChain
from .exposure import signed_call_put_split


def call_wall(chain: OptionChain, *, sigma: np.ndarray | None = None) -> float:
    """Strike with the most negative dealer GEX from calls (largest |short γ|)."""
    strikes, call_gex, _ = signed_call_put_split(chain, sigma=sigma)
    if strikes.size == 0:
        raise ValueError("empty chain")
    return float(strikes[int(np.argmin(call_gex))])


def put_wall(chain: OptionChain, *, sigma: np.ndarray | None = None) -> float:
    """Strike with the most positive dealer GEX from puts (largest |long γ|)."""
    strikes, _, put_gex = signed_call_put_split(chain, sigma=sigma)
    if strikes.size == 0:
        raise ValueError("empty chain")
    return float(strikes[int(np.argmax(put_gex))])


def max_pain(chain: OptionChain) -> float:
    """
    Max-pain strike — the strike that minimises total long-option payoff at expiry.

    Computed across the union of all expiries.  For multi-expiry chains the
    result is the single strike minimising the aggregate payoff if every
    contract were held to its own expiry against the candidate spot.
    """
    K = chain.strikes()
    oi = chain.open_interest()
    is_call = chain.is_call()
    strikes = np.unique(K)
    if strikes.size == 0:
        raise ValueError("empty chain")

    pain = np.empty(strikes.shape[0], dtype=np.float64)
    for j, s in enumerate(strikes):
        call_payoff = np.maximum(s - K[is_call], 0.0) * oi[is_call]
        put_payoff = np.maximum(K[~is_call] - s, 0.0) * oi[~is_call]
        pain[j] = call_payoff.sum() + put_payoff.sum()
    return float(strikes[int(np.argmin(pain))])


def max_pain_curve(chain: OptionChain) -> tuple[np.ndarray, np.ndarray]:
    """Return (strike grid, pain values) — useful for plotting the V-shape."""
    K = chain.strikes()
    oi = chain.open_interest()
    is_call = chain.is_call()
    strikes = np.unique(K)
    pain = np.empty(strikes.shape[0], dtype=np.float64)
    K_calls = K[is_call]
    oi_calls = oi[is_call]
    K_puts = K[~is_call]
    oi_puts = oi[~is_call]
    for j, s in enumerate(strikes):
        pain[j] = (np.maximum(s - K_calls, 0.0) * oi_calls).sum() + (
            np.maximum(K_puts - s, 0.0) * oi_puts
        ).sum()
    return strikes, pain
