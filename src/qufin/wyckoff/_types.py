"""
Shared types and result containers for the Wyckoff subpackage.

Conventions
-----------
* All array-backed containers store ``np.ndarray`` of ``float64`` to match the
  project numerical convention.
* OHLCV input frames carry a fixed polars schema (``BAR_SCHEMA``) so downstream
  code can operate without re-validating columns.
* Timestamps are timezone-aware UTC by convention (``zoneinfo.ZoneInfo('UTC')``).
* Event indices refer to bar positions in the source ``OHLCV.data`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Self

import numpy as np
import polars as pl

BAR_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
}

SwingKind = Literal["H", "L"]
ClimaxKind = Literal["SC", "BC"]
SpringKind = Literal["Spring", "UT", "UTAD"]
StructuralKind = Literal["PS", "AR", "ST", "SOS", "LPS", "SOW", "LPSY"]
Schematic = Literal["Acc", "Dist"]
PhaseLabel = Literal["A", "B", "C", "D", "E"]
PnFDirection = Literal["X", "O"]
TargetDirection = Literal["up", "down"]


def to_numpy_1d(x: Any) -> np.ndarray:
    """Coerce a 1-D polars Series / numpy array / array-like to float64 numpy."""
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


@dataclass(slots=True)
class OHLCV:
    """
    Polars-backed OHLCV bar sequence.

    Attributes
    ----------
    data       Long-format polars DataFrame with the schema in ``BAR_SCHEMA``.
    symbol     Ticker symbol (informational).

    The frame must be in monotonically non-decreasing timestamp order; the
    constructor enforces this.
    """

    data: pl.DataFrame
    symbol: str = ""

    def __post_init__(self) -> None:
        missing = set(BAR_SCHEMA) - set(self.data.columns)
        if missing:
            raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
        # Lightweight order check; full validation lives in bars.validate_ohlcv.
        if self.data.height >= 2:
            ts = self.data["timestamp"]
            if not ts.is_sorted():
                raise ValueError("OHLCV data must be sorted by timestamp ascending.")

    @classmethod
    def from_records(cls, records: pl.DataFrame, *, symbol: str = "") -> Self:
        """Coerce a DataFrame to ``BAR_SCHEMA`` and construct an ``OHLCV``."""
        missing = set(BAR_SCHEMA) - set(records.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {sorted(missing)}")
        coerced = records.select(*(pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items()))
        return cls(data=coerced, symbol=symbol)

    def __len__(self) -> int:
        return self.data.height

    @property
    def n_bars(self) -> int:
        return self.data.height

    def timestamps(self) -> np.ndarray:
        return self.data["timestamp"].to_numpy()

    def open(self) -> np.ndarray:
        return self.data["open"].to_numpy().astype(np.float64, copy=False)

    def high(self) -> np.ndarray:
        return self.data["high"].to_numpy().astype(np.float64, copy=False)

    def low(self) -> np.ndarray:
        return self.data["low"].to_numpy().astype(np.float64, copy=False)

    def close(self) -> np.ndarray:
        return self.data["close"].to_numpy().astype(np.float64, copy=False)

    def volume(self) -> np.ndarray:
        return self.data["volume"].to_numpy().astype(np.float64, copy=False)

    def slice_bars(self, start: int, end: int) -> Self:
        """Return a new OHLCV with rows [start, end) — half-open."""
        return type(self)(data=self.data.slice(start, end - start), symbol=self.symbol)


@dataclass(slots=True, frozen=True)
class SwingPoint:
    """A pivot / fractal point in an OHLCV sequence."""

    idx: int
    timestamp: datetime
    price: float
    kind: SwingKind
    strength: int  # number of bars on each side dominated by this pivot


@dataclass(slots=True, frozen=True)
class TradingRange:
    """A lateral consolidation bracketed by support and resistance."""

    start_idx: int
    end_idx: int
    support: float
    resistance: float

    def __post_init__(self) -> None:
        if self.end_idx <= self.start_idx:
            raise ValueError("end_idx must be greater than start_idx")
        if self.resistance <= self.support:
            raise ValueError("resistance must be greater than support")

    @property
    def mid(self) -> float:
        return 0.5 * (self.support + self.resistance)

    @property
    def width(self) -> float:
        return self.resistance - self.support

    @property
    def n_bars(self) -> int:
        return self.end_idx - self.start_idx

    def contains_idx(self, idx: int) -> bool:
        return self.start_idx <= idx < self.end_idx


@dataclass(slots=True, frozen=True)
class ClimaxEvent:
    """A selling (SC) or buying (BC) climax bar."""

    idx: int
    kind: ClimaxKind
    z_volume: float
    z_range: float
    price: float


@dataclass(slots=True, frozen=True)
class SpringEvent:
    """A spring, upthrust (UT), or UTAD (upthrust after distribution)."""

    idx: int
    kind: SpringKind
    penetration: float  # in price units, positive
    recovery_bars: int
    z_volume: float


@dataclass(slots=True, frozen=True)
class StructuralEvent:
    """Generic schematic milestone (PS, AR, ST, SOS, LPS, SOW, LPSY)."""

    idx: int
    kind: StructuralKind
    price: float
    z_volume: float


@dataclass(slots=True, frozen=True)
class WyckoffPhase:
    """A phase A-E inside an accumulation or distribution schematic."""

    start_idx: int
    end_idx: int
    schematic: Schematic
    phase: PhaseLabel


@dataclass(slots=True)
class VolumeProfile:
    """
    Volume-by-price histogram over a window of bars.

    Attributes
    ----------
    price_bins   Bin edges, shape (n_bins + 1,).
    volume       Volume per bin, shape (n_bins,).
    poc          Point of Control — bin centre with the most volume.
    vah          Value Area High — upper edge of value area.
    val          Value Area Low — lower edge of value area.
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


@dataclass(slots=True, frozen=True)
class PnFColumn:
    """A single column in a Point-and-Figure chart."""

    direction: PnFDirection
    start_idx: int  # bar index where the column started
    boxes_low: float
    boxes_high: float
    n_boxes: int


@dataclass(slots=True)
class PnFChart:
    """Point-and-Figure chart over a bar sequence."""

    box_size: float
    reversal: int
    columns: list[PnFColumn]

    @property
    def n_columns(self) -> int:
        return len(self.columns)


@dataclass(slots=True, frozen=True)
class CauseEffectTarget:
    """Projected target from a P&F count (cause-and-effect law)."""

    anchor_col: int
    count_boxes: int
    box_size: float
    reversal: int
    breakout_price: float
    projected_price: float
    direction: TargetDirection
    method: Literal["horizontal", "vertical"]
