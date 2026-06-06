"""
Trade-sign classification.

Aggressor side ("who crossed the spread?") is not reported by most public feeds
and must be inferred from prices and — when available — quotes.

* ``tick_rule``  — sign of the price change, carrying the last sign on flat ticks.
* ``quote_rule`` — buy above the midpoint, sell below it.
* ``lee_ready``  — Lee & Ready (1991): quote rule with a tick-rule tiebreak at
  the midpoint.  The de-facto standard for TAQ data.
* ``emo_rule``   — Ellis, Michaely & O'Hara (2000): quotes classify trades *at*
  the bid/ask, the tick rule classifies trades *inside* the spread.
* ``bvc``        — Bulk Volume Classification (Easley, López de Prado & O'Hara
  2012): assigns a *fraction* of each bar's volume to the buy side from the
  standardised price change, rather than a hard ±1 label.

The ±1 classifiers return a float64 array of ``+1`` (buyer-initiated), ``-1``
(seller-initiated) and ``0`` (indeterminate).  ``bvc`` returns the buy fraction
in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

from ._kernels import emo_signs, lee_ready_signs, quote_signs, tick_signs
from ._types import check_lengths, to_numpy_1d


def tick_rule(prices: Any) -> np.ndarray:
    """Tick-rule trade signs with sign-carry on zero ticks (Lee-Ready 1991)."""
    p = to_numpy_1d(prices)
    if p.shape[0] < 2:
        raise ValueError("prices must have at least 2 observations.")
    return tick_signs(p)


def quote_rule(prices: Any, bid: Any, ask: Any) -> np.ndarray:
    """Quote-rule trade signs from contemporaneous best bid / ask."""
    p, b, a = to_numpy_1d(prices), to_numpy_1d(bid), to_numpy_1d(ask)
    check_lengths(p, b, a)
    return quote_signs(p, b, a)


def lee_ready(prices: Any, bid: Any, ask: Any) -> np.ndarray:
    """Lee & Ready (1991) trade signs: quote rule with tick-rule midpoint tiebreak."""
    p, b, a = to_numpy_1d(prices), to_numpy_1d(bid), to_numpy_1d(ask)
    check_lengths(p, b, a)
    return lee_ready_signs(p, b, a)


def emo_rule(prices: Any, bid: Any, ask: Any) -> np.ndarray:
    """Ellis-Michaely-O'Hara (2000) trade signs."""
    p, b, a = to_numpy_1d(prices), to_numpy_1d(bid), to_numpy_1d(ask)
    check_lengths(p, b, a)
    return emo_signs(p, b, a)


def bvc(
    prices: Any,
    *,
    distribution: Literal["normal", "t"] = "t",
    dof: float = 0.25,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Bulk Volume Classification (Easley, López de Prado & O'Hara 2012).

    For a sequence of bar (or bucket) closing prices, the buyer-initiated
    *fraction* of bar ``i``'s volume is

        f_i = CDF(ΔP_i / σ_ΔP)

    where ``CDF`` is the standard normal or a standardised Student-t with ``dof``
    degrees of freedom, ``ΔP_i = P_i − P_{i-1}`` and ``σ_ΔP`` is the standard
    deviation of price changes.  Heavy-tailed Student-t (small ``dof``) is the
    original BVC choice; ``distribution="normal"`` recovers the Gaussian variant.

    Parameters
    ----------
    prices        Bar / bucket closing prices, shape ``(n,)``.
    distribution  ``"t"`` (default) or ``"normal"``.
    dof           Student-t degrees of freedom (ignored for the normal).
    sigma         Override for the price-change std; estimated from the data when
                  ``None``.

    Returns
    -------
    np.ndarray, shape ``(n,)``
        Buy fraction in ``[0, 1]``.  Element 0 is ``0.5`` (no prior price).
    """
    p = to_numpy_1d(prices)
    n = p.shape[0]
    if n < 2:
        raise ValueError("prices must have at least 2 observations.")
    if distribution not in ("normal", "t"):
        raise ValueError(f"distribution must be 'normal' or 't', got {distribution!r}.")
    if distribution == "t" and dof <= 0.0:
        raise ValueError(f"dof must be > 0, got {dof}.")

    dp = np.diff(p)
    if sigma is None:
        sigma = float(np.std(dp))
    if sigma <= 0.0:
        return np.full(n, 0.5, dtype=np.float64)

    z = dp / sigma
    if distribution == "normal":
        frac = stats.norm.cdf(z)
    else:
        # Standardise the Student-t so its variance is 1 before evaluating, so
        # ``sigma`` carries the full scale (matches the BVC formulation).
        scale = np.sqrt(dof / (dof - 2.0)) if dof > 2.0 else 1.0
        frac = stats.t.cdf(z * scale, df=dof)

    out = np.empty(n, dtype=np.float64)
    out[0] = 0.5
    out[1:] = frac
    return out
