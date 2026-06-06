"""
Result containers and shared helpers for the volume-distribution subpackage.

Core data primitives (``BAR_SCHEMA``, ``TICK_SCHEMA``, ``OHLCV``,
``to_numpy_1d``) live in :mod:`qufin.data._types` and are re-exported here so
the subpackage's modules import them from a single place.

Conventions
-----------
* Array-backed containers store contiguous ``float64`` ``numpy`` arrays.
* Multi-field results are frozen ``@dataclass(slots=True)`` containers.
* Histograms follow the ``price_bins`` (``n_bins + 1`` edges) / ``volume``
  (``n_bins`` counts) convention shared with the Wyckoff volume profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from ..data._types import BAR_SCHEMA, OHLCV, TICK_SCHEMA, to_numpy_1d

__all__ = [
    "BAR_SCHEMA",
    "OHLCV",
    "TICK_SCHEMA",
    "DeltaProfile",
    "DistributionStats",
    "ProfileShape",
    "Side",
    "SignMethod",
    "TPOProfile",
    "VWAPBands",
    "VolumeProfile",
    "as_tick_arrays",
    "check_lengths",
    "coerce_ohlcv",
    "to_numpy_1d",
    "value_area_bounds",
]

Side = Literal["buy", "sell"]
SignMethod = Literal["tick_rule", "lee_ready"]
ProfileShape = Literal["normal", "b", "p", "D"]


def check_lengths(*arrays: np.ndarray) -> int:
    """Verify all arrays share a common length and return it."""
    if not arrays:
        raise ValueError("at least one array is required")
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError(f"length mismatch: {a.shape[0]} != {n}")
    return n


@dataclass(slots=True, frozen=True)
class VolumeProfile:
    """
    Volume-by-price histogram over a window of bars (or ticks).

    Attributes
    ----------
    price_bins   Bin edges, shape ``(n_bins + 1,)``.
    volume       Volume per bin, shape ``(n_bins,)``.
    poc          Point of Control — bin centre with the most volume.
    vah          Value Area High — upper edge of the value area.
    val          Value Area Low — lower edge of the value area.
    hvn_idx      Indices of High Volume Nodes (local maxima).
    lvn_idx      Indices of Low Volume Nodes (local minima).
    """

    price_bins: np.ndarray
    volume: np.ndarray
    poc: float
    vah: float
    val: float
    hvn_idx: np.ndarray
    lvn_idx: np.ndarray

    def to_dataframe(self) -> pl.DataFrame:
        centres = 0.5 * (self.price_bins[:-1] + self.price_bins[1:])
        return pl.DataFrame({"price": centres, "volume": self.volume})

    @property
    def bin_centres(self) -> np.ndarray:
        return 0.5 * (self.price_bins[:-1] + self.price_bins[1:])


@dataclass(slots=True, frozen=True)
class TPOProfile:
    """
    Market-Profile / Time-Price-Opportunity histogram.

    Attributes
    ----------
    price_bins        Bin edges, shape ``(n_bins + 1,)``.
    tpo_counts        TPO count per price bin (number of time brackets that
                      traded the level), shape ``(n_bins,)``.
    letters           Per-bin string of bracket letters (``"A"``, ``"B"`` …).
    poc               Point of Control — bin centre with the most TPOs.
    vah / val         Value Area High / Low from the TPO distribution.
    initial_balance   (low, high) price range of the first ``n_initial``
                      brackets.
    single_prints     Bin indices touched by exactly one bracket.
    range_extension_up / _down  Whether trade occurred above / below the
                      initial balance.
    """

    price_bins: np.ndarray
    tpo_counts: np.ndarray
    letters: list[str]
    poc: float
    vah: float
    val: float
    initial_balance: tuple[float, float]
    single_prints: np.ndarray
    range_extension_up: bool
    range_extension_down: bool

    @property
    def bin_centres(self) -> np.ndarray:
        return 0.5 * (self.price_bins[:-1] + self.price_bins[1:])


@dataclass(slots=True, frozen=True)
class DeltaProfile:
    """
    Buy/sell volume split by price (footprint-style).

    Attributes
    ----------
    price_bins   Bin edges, shape ``(n_bins + 1,)``.
    buy_volume   Aggressor-buy volume per bin, shape ``(n_bins,)``.
    sell_volume  Aggressor-sell volume per bin, shape ``(n_bins,)``.
    delta        ``buy_volume - sell_volume`` per bin.
    """

    price_bins: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray
    delta: np.ndarray


@dataclass(slots=True, frozen=True)
class VWAPBands:
    """
    Cumulative VWAP with volume-weighted standard-deviation envelopes.

    Attributes
    ----------
    vwap       VWAP line, shape ``(n,)``.
    upper      Upper bands, shape ``(n, k)`` — column ``j`` is
               ``vwap + std_mults[j] * sigma``.
    lower      Lower bands, shape ``(n, k)``.
    std_mults  The standard-deviation multipliers, length ``k``.
    """

    vwap: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    std_mults: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class DistributionStats:
    """
    Shape statistics of a volume-at-price distribution.

    Attributes
    ----------
    gini      Gini coefficient of concentration in ``[0, 1]``.
    entropy   Normalised Shannon entropy in ``[0, 1]`` (1 = uniform).
    skew      Volume-weighted skewness of price.
    kurtosis  Volume-weighted excess kurtosis of price.
    shape     Coarse profile-shape label.
    """

    gini: float
    entropy: float
    skew: float
    kurtosis: float
    shape: ProfileShape


def as_tick_arrays(ticks: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(price, size)`` float64 arrays from a ``TICK_SCHEMA`` frame."""
    price = to_numpy_1d(ticks["price"])
    size = to_numpy_1d(ticks["size"])
    return price, size


def coerce_ohlcv(bars: Any) -> OHLCV:
    """Accept an ``OHLCV`` or a raw DataFrame and return an ``OHLCV``."""
    match bars:
        case OHLCV():
            return bars
        case pl.DataFrame():
            return OHLCV.from_records(bars)
        case _:
            raise TypeError(f"expected OHLCV or polars DataFrame, got {type(bars).__name__}")


def value_area_bounds(edges: np.ndarray, hist: np.ndarray, frac: float) -> tuple[float, float]:
    """
    Value-area ``(val, vah)`` for a histogram: expand outward from the POC bin
    until ``frac`` of total mass is held, taking the larger neighbour each step.
    Shared by the volume profile and the TPO profile.
    """
    n_bins = hist.shape[0]
    if n_bins == 0:
        return float(edges[0]), float(edges[-1])
    poc_idx = int(np.argmax(hist))
    total = float(hist.sum())
    target = total * frac
    lo = hi = poc_idx
    acc = hist[poc_idx]
    while acc < target and (lo > 0 or hi < n_bins - 1):
        left_v = hist[lo - 1] if lo > 0 else -1.0
        right_v = hist[hi + 1] if hi < n_bins - 1 else -1.0
        if right_v >= left_v and hi < n_bins - 1:
            hi += 1
            acc += hist[hi]
        elif lo > 0:
            lo -= 1
            acc += hist[lo]
        else:
            break
    return float(edges[lo]), float(edges[hi + 1])
