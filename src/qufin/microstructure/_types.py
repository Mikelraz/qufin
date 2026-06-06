"""
Result containers and shared helpers for the market-microstructure subpackage.

Core data primitives (``TICK_SCHEMA``, ``to_numpy_1d``) live in
:mod:`qufin.data._types` and are re-exported here so the subpackage's modules
import them from a single place.

Conventions
-----------
* Array-backed containers store contiguous ``float64`` ``numpy`` arrays.
* Single-valued estimators (Roll, Amihud) return ``float``; per-observation or
  per-window estimators (effective spread, Corwin-Schultz, OFI) return
  ``np.ndarray`` — the array-in / array-out convention shared with
  :mod:`qufin.indicators`.
* Multi-field regression / model results are ``@dataclass(slots=True, frozen=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from ..data._types import TICK_SCHEMA, to_numpy_1d

__all__ = [
    "TICK_SCHEMA",
    "ClassifierMethod",
    "PriceImpactResult",
    "VPINResult",
    "as_trade_arrays",
    "check_lengths",
    "to_numpy_1d",
]

ClassifierMethod = Literal["tick", "quote", "lee_ready", "emo", "bvc"]


def check_lengths(*arrays: np.ndarray) -> int:
    """Verify all arrays share a common length and return it."""
    if not arrays:
        raise ValueError("at least one array is required")
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError(f"length mismatch: {a.shape[0]} != {n}")
    return n


def as_trade_arrays(trades: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(price, size)`` float64 arrays from a ``TICK_SCHEMA`` frame."""
    return to_numpy_1d(trades["price"]), to_numpy_1d(trades["size"])


@dataclass(slots=True, frozen=True)
class PriceImpactResult:
    """
    Linear price-impact regression outcome (Kyle / Hasbrouck λ).

    Attributes
    ----------
    lam        Estimated price-impact coefficient λ (slope).  Units are price
               per signed-volume unit (Kyle) or price per √(signed volume)
               (Hasbrouck).
    r_squared  Coefficient of determination of the impact regression.
    t_stat     t-statistic on λ (HAC-free OLS standard error).
    n_obs      Number of observations entering the regression.
    intercept  Fitted intercept of the regression.
    """

    lam: float
    r_squared: float
    t_stat: float
    n_obs: int
    intercept: float

    def __str__(self) -> str:
        return (
            f"PriceImpact(λ={self.lam:.6g}, R²={self.r_squared:.4f}, "
            f"t={self.t_stat:.3f}, n={self.n_obs})"
        )


@dataclass(slots=True, frozen=True)
class VPINResult:
    """
    Volume-synchronised Probability of Informed Trading (Easley, López de
    Prado & O'Hara 2012).

    Attributes
    ----------
    vpin          Rolling VPIN series, one value per bucket once ``window``
                  buckets are available, shape ``(n_buckets,)`` with leading
                  NaNs for the warm-up.
    buy_volume    Aggressor-buy volume per equal-volume bucket, shape
                  ``(n_buckets,)``.
    sell_volume   Aggressor-sell volume per bucket, shape ``(n_buckets,)``.
    bucket_size   The fixed volume ``V`` contained in each bucket.
    window        Number of buckets averaged in the rolling VPIN.
    """

    vpin: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray
    bucket_size: float
    window: int

    @property
    def n_buckets(self) -> int:
        return self.vpin.shape[0]

    def to_dataframe(self) -> pl.DataFrame:
        order_imbalance = np.abs(self.buy_volume - self.sell_volume) / self.bucket_size
        return pl.DataFrame(
            {
                "bucket": np.arange(self.n_buckets, dtype=np.int64),
                "buy_volume": self.buy_volume,
                "sell_volume": self.sell_volume,
                "order_imbalance": order_imbalance,
                "vpin": self.vpin,
            }
        )
