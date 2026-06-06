"""Shared fixtures for microstructure tests."""

from __future__ import annotations

import numpy as np


def bid_ask_bounce(
    n: int,
    *,
    mid: float = 100.0,
    spread: float = 0.10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant-fundamental price with pure bid-ask bounce (Roll's model).

    Returns ``(prices, signs)`` where ``prices = mid + (spread/2) · q`` and
    ``q`` are i.i.d. ±1 aggressor signs.  The lag-1 autocovariance of price
    changes is ``−(spread/2)²``, so Roll's estimator recovers ``spread``.
    """
    rng = np.random.default_rng(seed)
    q = rng.choice(np.array([-1.0, 1.0]), size=n)
    prices = mid + 0.5 * spread * q
    return prices, q


def quotes_around(prices: np.ndarray, *, half_spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Return constant ``(bid, ask)`` arrays straddling ``mid`` by ``half_spread``."""
    mid = np.full_like(prices, float(np.mean(prices)))
    return mid - half_spread, mid + half_spread
