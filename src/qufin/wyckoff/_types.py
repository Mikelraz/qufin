"""
Result containers and Wyckoff-specific types.

Core data primitives (``BAR_SCHEMA``, ``OHLCV``, ``to_numpy_1d``) live in
``qufin.data._types`` and are re-exported here so existing Wyckoff modules
continue to import them from a single place.

Conventions
-----------
* Array-backed containers store ``np.ndarray`` of ``float64``.
* Timestamps are timezone-aware UTC.
* Event indices refer to bar positions in the source ``OHLCV.data`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..data._types import BAR_SCHEMA, OHLCV, to_numpy_1d
from ..volume_distribution._types import VolumeProfile

__all__ = [
    "BAR_SCHEMA",
    "OHLCV",
    "to_numpy_1d",
    "SwingKind",
    "ClimaxKind",
    "SpringKind",
    "StructuralKind",
    "Schematic",
    "PhaseLabel",
    "PnFDirection",
    "TargetDirection",
    "SwingPoint",
    "TradingRange",
    "ClimaxEvent",
    "SpringEvent",
    "StructuralEvent",
    "WyckoffPhase",
    "VolumeProfile",
    "PnFColumn",
    "PnFChart",
    "CauseEffectTarget",
]

SwingKind = Literal["H", "L"]
ClimaxKind = Literal["SC", "BC"]
SpringKind = Literal["Spring", "UT", "UTAD"]
StructuralKind = Literal["PS", "AR", "ST", "SOS", "LPS", "SOW", "LPSY"]
Schematic = Literal["Acc", "Dist"]
PhaseLabel = Literal["A", "B", "C", "D", "E"]
PnFDirection = Literal["X", "O"]
TargetDirection = Literal["up", "down"]


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
