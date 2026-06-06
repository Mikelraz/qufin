"""
Order-flow imbalance measures.

* ``signed_volume``        — per-trade signed volume ``q_t V_t``.
* ``trade_imbalance``      — rolling buy/sell imbalance in ``[-1, 1]`` from trade
  signs (optionally volume-weighted).
* ``order_flow_imbalance`` — Cont, Kukanov & Stoikov (2014) OFI from best-level
  quote updates; the single most informative L1 order-book event measure for
  short-horizon price moves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import check_lengths, to_numpy_1d


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum; the first ``window − 1`` entries are ``NaN``."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cum = np.cumsum(x)
    out[window - 1] = cum[window - 1]
    out[window:] = cum[window:] - cum[:-window]
    return out


def signed_volume(signs: Any, volume: Any) -> np.ndarray:
    """Per-trade signed volume ``q_t · V_t``."""
    q, v = to_numpy_1d(signs), to_numpy_1d(volume)
    check_lengths(q, v)
    return q * v


def trade_imbalance(signs: Any, volume: Any | None = None, *, window: int = 50) -> np.ndarray:
    """
    Rolling order imbalance in ``[-1, 1]``.

    Without ``volume`` this is the mean trade sign over the trailing ``window``;
    with ``volume`` it is volume-weighted: ``Σ q_t V_t / Σ V_t``.  Leading
    entries are ``NaN`` until the window fills.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    q = to_numpy_1d(signs)
    if volume is None:
        return _rolling_sum(q, window) / window
    v = to_numpy_1d(volume)
    check_lengths(q, v)
    num = _rolling_sum(q * v, window)
    den = _rolling_sum(v, window)
    return np.where(den > 0.0, num / den, np.nan)


def order_flow_imbalance(
    bid: Any,
    ask: Any,
    bid_size: Any,
    ask_size: Any,
    *,
    window: int | None = None,
) -> np.ndarray:
    """
    Order-flow imbalance from best-level quote events (Cont-Kukanov-Stoikov 2014).

    For each quote update the contribution is ``e_n = ΔV^b_n − ΔV^a_n`` where

        ΔV^b_n = q^b_n·1{P^b_n ≥ P^b_{n−1}} − q^b_{n−1}·1{P^b_n ≤ P^b_{n−1}}
        ΔV^a_n = q^a_n·1{P^a_n ≤ P^a_{n−1}} − q^a_{n−1}·1{P^a_n ≥ P^a_{n−1}}

    Positive ``e_n`` reflects net demand at the top of book.  Element 0 is 0 (no
    previous quote).

    Parameters
    ----------
    bid, ask              Best bid / ask prices, shape ``(n,)``.
    bid_size, ask_size    Best bid / ask depths, shape ``(n,)``.
    window                If given, return the trailing rolling sum of ``e_n``
                          over ``window`` events (the OFI over each interval);
                          otherwise return the per-event contributions.
    """
    pb, pa = to_numpy_1d(bid), to_numpy_1d(ask)
    qb, qa = to_numpy_1d(bid_size), to_numpy_1d(ask_size)
    n = check_lengths(pb, pa, qb, qa)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out

    bid_up = pb[1:] > pb[:-1]
    bid_dn = pb[1:] < pb[:-1]
    d_vb = np.where(bid_up, qb[1:], np.where(bid_dn, -qb[:-1], qb[1:] - qb[:-1]))

    ask_up = pa[1:] > pa[:-1]
    ask_dn = pa[1:] < pa[:-1]
    d_va = np.where(ask_dn, qa[1:], np.where(ask_up, -qa[:-1], qa[1:] - qa[:-1]))

    out[1:] = d_vb - d_va
    if window is not None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        return _rolling_sum(out, window)
    return out
