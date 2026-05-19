"""
Shared types and result containers for the indicators subpackage.

Conventions
-----------
* All inputs are coerced to contiguous ``float64`` ``numpy`` arrays — see
  :func:`to_numpy_1d`.
* Indicators that require fewer than ``window`` samples for warm-up pad the
  warm-up region with ``NaN`` so output arrays match the input length.
* Multi-line indicators (MACD, ADX, Bollinger, …) are returned as frozen
  ``@dataclass(slots=True)`` containers with named ``np.ndarray`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

PriceSource = Literal["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]
TrendDirection = Literal["up", "down"]


def to_numpy_1d(x: Any) -> np.ndarray:
    """Coerce a 1-D array-like / polars Series to a contiguous float64 array."""
    match x:
        case np.ndarray():
            arr = x
        case pl.Series():
            arr = x.to_numpy()
        case _:
            arr = np.asarray(x)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


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
class MACDResult:
    """MACD line, signal line, and histogram (line - signal)."""

    macd: np.ndarray
    signal: np.ndarray
    hist: np.ndarray


@dataclass(slots=True, frozen=True)
class BollingerBands:
    """Bollinger bands: middle (SMA), upper, lower, and bandwidth/percent-b."""

    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    bandwidth: np.ndarray
    percent_b: np.ndarray


@dataclass(slots=True, frozen=True)
class KeltnerChannels:
    """Keltner channels: EMA midline plus ATR-scaled envelope."""

    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


@dataclass(slots=True, frozen=True)
class DonchianChannels:
    """Donchian channels: rolling high, rolling low, and midline."""

    upper: np.ndarray
    lower: np.ndarray
    middle: np.ndarray


@dataclass(slots=True, frozen=True)
class StochasticResult:
    """Stochastic oscillator %K and %D."""

    k: np.ndarray
    d: np.ndarray


@dataclass(slots=True, frozen=True)
class ADXResult:
    """Average Directional Index with directional movement components."""

    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray


@dataclass(slots=True, frozen=True)
class AroonResult:
    """Aroon Up, Aroon Down, and Aroon Oscillator (up - down)."""

    up: np.ndarray
    down: np.ndarray
    oscillator: np.ndarray


@dataclass(slots=True, frozen=True)
class SupertrendResult:
    """Supertrend line and the bar-by-bar trend direction (+1 up / -1 down)."""

    line: np.ndarray
    direction: np.ndarray


@dataclass(slots=True, frozen=True)
class IchimokuResult:
    """Ichimoku Kinko Hyo components."""

    tenkan: np.ndarray  # Conversion line
    kijun: np.ndarray  # Base line
    senkou_a: np.ndarray  # Leading span A (already shifted forward)
    senkou_b: np.ndarray  # Leading span B (already shifted forward)
    chikou: np.ndarray  # Lagging span (close shifted backward)


@dataclass(slots=True, frozen=True)
class PivotPoints:
    """Classic floor pivot points and their resistance / support levels."""

    pp: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    def as_dict(self) -> dict[str, float]:
        return {
            "PP": self.pp,
            "R1": self.r1,
            "R2": self.r2,
            "R3": self.r3,
            "S1": self.s1,
            "S2": self.s2,
            "S3": self.s3,
        }


@dataclass(slots=True, frozen=True)
class SupportResistanceLevel:
    """A clustered horizontal price level.

    Attributes
    ----------
    price       Cluster centroid (mean of constituent pivot prices).
    kind        ``'S'`` (support) or ``'R'`` (resistance) by dominant swing kind,
                or ``'SR'`` if both kinds contributed.
    touches     Number of pivots merged into the cluster.
    strength    ``touches`` scaled by the mean pivot ``strength`` field.
    first_idx   Bar index of the earliest pivot in the cluster.
    last_idx    Bar index of the most recent pivot in the cluster.
    """

    price: float
    kind: Literal["S", "R", "SR"]
    touches: int
    strength: float
    first_idx: int
    last_idx: int
