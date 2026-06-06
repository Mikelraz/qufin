"""
Bid-ask spread estimators.

Direct (need quotes)
--------------------
* ``quoted_spread``    — ask − bid.
* ``effective_spread`` — ``2 q_t (p_t − m_t)``; what the taker actually paid
  relative to the midpoint ``m_t``.
* ``realized_spread``  — ``2 q_t (p_t − m_{t+τ})``; the post-trade-reversion
  (non-adverse-selection) component of the effective spread.

Indirect (trades or OHLC only)
------------------------------
* ``roll_spread``      — Roll (1984), from the negative autocovariance of price
  changes induced by bid-ask bounce.
* ``corwin_schultz``   — Corwin & Schultz (2012), from consecutive daily high-low
  ranges.
* ``abdi_ranaldo``     — Abdi & Ranaldo (2017) CHL estimator, from close prices
  and the high-low midrange.

Per-trade / per-window estimators return arrays; ``roll_spread`` returns a single
whole-sample float.  Absolute spreads share the price units of the input; pass
``relative=True`` for proportional (spread / midpoint) spreads where supported.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._types import check_lengths, to_numpy_1d
from .classification import lee_ready

_CS_DENOM = 3.0 - 2.0 * math.sqrt(2.0)


def quoted_spread(bid: Any, ask: Any, *, relative: bool = False) -> np.ndarray:
    """Quoted spread ``ask − bid`` (or ``(ask − bid) / midpoint`` if relative)."""
    b, a = to_numpy_1d(bid), to_numpy_1d(ask)
    check_lengths(b, a)
    spread = a - b
    if relative:
        mid = 0.5 * (a + b)
        return np.where(mid > 0.0, spread / mid, np.nan)
    return spread


def effective_spread(
    prices: Any,
    bid: Any,
    ask: Any,
    *,
    signs: Any | None = None,
    relative: bool = False,
) -> np.ndarray:
    """
    Effective spread ``2 q_t (p_t − m_t)`` per trade.

    The trade sign ``q_t`` is taken from ``signs`` if provided, else inferred
    with :func:`qufin.microstructure.lee_ready`.
    """
    p, b, a = to_numpy_1d(prices), to_numpy_1d(bid), to_numpy_1d(ask)
    check_lengths(p, b, a)
    q = lee_ready(p, b, a) if signs is None else to_numpy_1d(signs)
    check_lengths(p, q)
    mid = 0.5 * (a + b)
    eff = 2.0 * q * (p - mid)
    if relative:
        return np.where(mid > 0.0, eff / mid, np.nan)
    return eff


def realized_spread(
    prices: Any,
    mid: Any,
    *,
    signs: Any,
    tau: int = 5,
    relative: bool = False,
) -> np.ndarray:
    """
    Realized spread ``2 q_t (p_t − m_{t+τ})`` per trade.

    Measures the share of the effective spread that is *not* adverse selection:
    it compares the trade price to the midpoint ``τ`` events later.  The final
    ``τ`` observations are ``NaN`` (no future midpoint).

    Parameters
    ----------
    prices  Trade prices, shape ``(n,)``.
    mid     Midpoint series aligned to the trades, shape ``(n,)``.
    signs   Trade signs ``q_t`` (e.g. from :func:`lee_ready`).
    tau     Forward horizon in observations.
    """
    p, m, q = to_numpy_1d(prices), to_numpy_1d(mid), to_numpy_1d(signs)
    check_lengths(p, m, q)
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}.")
    n = p.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= tau:
        return out
    future_mid = m[tau:]
    rs = 2.0 * q[:-tau] * (p[:-tau] - future_mid)
    if relative:
        m0 = m[:-tau]
        rs = np.where(m0 > 0.0, rs / m0, np.nan)
    out[:-tau] = rs
    return out


def roll_spread(prices: Any) -> float:
    """
    Roll (1984) implied effective spread ``2 √(−Cov(Δp_t, Δp_{t−1}))``.

    Under the Roll model, bid-ask bounce induces negative first-order
    autocovariance in price changes.  When the sample autocovariance is
    non-negative the model is mis-specified and the estimate is ``NaN``.
    """
    p = to_numpy_1d(prices)
    if p.shape[0] < 3:
        raise ValueError("prices must have at least 3 observations.")
    dp = np.diff(p)
    dp_t = dp[1:]
    dp_lag = dp[:-1]
    cov = float(np.mean((dp_t - dp_t.mean()) * (dp_lag - dp_lag.mean())))
    if cov >= 0.0:
        return float("nan")
    return 2.0 * math.sqrt(-cov)


def corwin_schultz(high: Any, low: Any) -> np.ndarray:
    """
    Corwin & Schultz (2012) high-low spread estimator (proportional).

    Uses overlapping two-day windows.  Element ``t`` is computed from days
    ``t−1`` and ``t``; element 0 is ``NaN``.  Negative point estimates are
    floored at 0 as recommended by the authors.
    """
    h, l = to_numpy_1d(high), to_numpy_1d(low)  # noqa: E741
    check_lengths(h, l)
    n = h.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    if np.any(l <= 0.0):
        raise ValueError("low prices must be strictly positive.")

    log_hl = np.log(h / l) ** 2  # single-day β components
    beta = log_hl[1:] + log_hl[:-1]
    h2 = np.maximum(h[1:], h[:-1])
    l2 = np.minimum(l[1:], l[:-1])
    gamma = np.log(h2 / l2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    out[1:] = np.maximum(spread, 0.0)
    return out


def abdi_ranaldo(high: Any, low: Any, close: Any) -> np.ndarray:
    """
    Abdi & Ranaldo (2017) CHL spread estimator (proportional), per two-day window.

    With log close ``c_t`` and log midrange ``η_t = (ln H_t + ln L_t) / 2``,

        S_t = 2 √( max( (c_t − η_t)(c_t − η_{t+1}), 0 ) ).

    Element ``t`` uses days ``t`` and ``t+1``; the final element is ``NaN``.
    """
    h, l, c = to_numpy_1d(high), to_numpy_1d(low), to_numpy_1d(close)  # noqa: E741
    check_lengths(h, l, c)
    n = h.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    if np.any(l <= 0.0) or np.any(h <= 0.0) or np.any(c <= 0.0):
        raise ValueError("prices must be strictly positive.")

    log_c = np.log(c)
    eta = 0.5 * (np.log(h) + np.log(l))
    prod = (log_c[:-1] - eta[:-1]) * (log_c[:-1] - eta[1:])
    out[:-1] = 2.0 * np.sqrt(np.maximum(prod, 0.0))
    return out
