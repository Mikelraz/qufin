"""
VPIN — Volume-synchronised Probability of Informed Trading.

Easley, López de Prado & O'Hara (2012) measure order-flow toxicity in *volume
time* rather than clock time:

1. Bulk-classify each bar's volume into buy / sell with :func:`bvc`.
2. Pack the classified volume into equal-volume buckets of size ``V``.
3. VPIN is the trailing average over ``window`` buckets of the per-bucket order
   imbalance ``|V_buy − V_sell| / V``.

High VPIN signals a high share of informed (toxic) flow and has been linked to
liquidity-provider withdrawal ahead of volatility events (e.g. the 2010 Flash
Crash).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._kernels import accumulate_buckets
from ._types import VPINResult, check_lengths, to_numpy_1d
from .classification import bvc


def vpin(
    prices: Any,
    volumes: Any,
    *,
    bucket_size: float | None = None,
    n_buckets: int | None = None,
    window: int = 50,
    distribution: Literal["normal", "t"] = "normal",
    dof: float = 0.25,
) -> VPINResult:
    """
    Compute VPIN over equal-volume buckets.

    Provide exactly one of ``bucket_size`` (the volume ``V`` per bucket) or
    ``n_buckets`` (target bucket count, from which ``V`` is derived as
    total-volume / ``n_buckets``).

    Parameters
    ----------
    prices        Bar closing prices, shape ``(n,)``.
    volumes       Bar volumes (non-negative), shape ``(n,)``.
    bucket_size   Fixed volume per bucket.  Mutually exclusive with ``n_buckets``.
    n_buckets     Target number of buckets.  Mutually exclusive with ``bucket_size``.
    window        Number of buckets in the trailing VPIN average.
    distribution  Bulk-classification distribution (``"normal"`` is the original
                  VPIN choice).
    dof           Student-t degrees of freedom when ``distribution="t"``.

    Returns
    -------
    VPINResult
    """
    p = to_numpy_1d(prices)
    v = to_numpy_1d(volumes)
    check_lengths(p, v)
    if np.any(v < 0.0):
        raise ValueError("volumes must be non-negative.")
    if (bucket_size is None) == (n_buckets is None):
        raise ValueError("provide exactly one of bucket_size or n_buckets.")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")

    total = float(v.sum())
    if n_buckets is not None:
        if n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {n_buckets}.")
        bucket_size = total / n_buckets
    assert bucket_size is not None  # narrowed for the type checker
    if bucket_size <= 0.0:
        raise ValueError("bucket_size must be > 0 (is total volume zero?).")

    frac = bvc(p, distribution=distribution, dof=dof)
    buy_bar = v * frac
    sell_bar = v * (1.0 - frac)
    out_buy, out_sell = accumulate_buckets(buy_bar, sell_bar, bucket_size)

    nb = out_buy.shape[0]
    vpin_series = np.full(nb, np.nan, dtype=np.float64)
    if nb >= window:
        imbalance = np.abs(out_buy - out_sell) / bucket_size
        cum = np.cumsum(imbalance)
        roll = np.empty(nb - window + 1, dtype=np.float64)
        roll[0] = cum[window - 1]
        roll[1:] = cum[window:] - cum[:-window]
        vpin_series[window - 1 :] = roll / window

    return VPINResult(
        vpin=vpin_series,
        buy_volume=out_buy,
        sell_volume=out_sell,
        bucket_size=float(bucket_size),
        window=window,
    )
