"""Rule-based Wyckoff phase classifier."""

from __future__ import annotations

import numpy as np

from qufin.wyckoff import (
    ClimaxEvent,
    SpringEvent,
    StructuralEvent,
    TradingRange,
    classify_phases,
)
from tests.wyckoff.conftest import make_ohlcv


def test_phases_order_a_through_e_accumulation() -> None:
    # Synthetic bar grid is mostly irrelevant — classify_phases works off events.
    n = 200
    closes = np.full(n, 100.0)
    closes[150:] = np.linspace(100.0, 110.0, n - 150)  # range exit upward
    opens = closes
    highs = closes + 0.1
    lows = closes - 0.1
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)

    tr = TradingRange(start_idx=10, end_idx=140, support=98.0, resistance=102.0)
    climaxes = [
        ClimaxEvent(idx=15, kind="SC", z_volume=3.0, z_range=3.0, price=97.5),
    ]
    structural = [
        StructuralEvent(idx=20, kind="AR", price=101.5, z_volume=1.0),
        StructuralEvent(idx=30, kind="ST", price=98.2, z_volume=0.2),
        StructuralEvent(idx=150, kind="SOS", price=103.0, z_volume=2.0),
        StructuralEvent(idx=160, kind="LPS", price=102.2, z_volume=0.5),
    ]
    springs = [
        SpringEvent(idx=120, kind="Spring", penetration=1.0, recovery_bars=2, z_volume=0.5),
    ]
    phases = classify_phases(bars, [tr], climaxes, structural, springs)
    labels = [p.phase for p in phases]
    assert labels == ["A", "B", "C", "D", "E"]
    schematics = {p.schematic for p in phases}
    assert schematics == {"Acc"}
    # Indices are monotonic.
    assert all(phases[i].start_idx <= phases[i + 1].start_idx for i in range(len(phases) - 1))


def test_phases_distribution_schematic() -> None:
    n = 200
    closes = np.full(n, 100.0)
    closes[150:] = np.linspace(100.0, 90.0, n - 150)  # downward range exit
    opens = closes
    highs = closes + 0.1
    lows = closes - 0.1
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)

    tr = TradingRange(start_idx=10, end_idx=140, support=98.0, resistance=102.0)
    climaxes = [
        ClimaxEvent(idx=15, kind="BC", z_volume=3.0, z_range=3.0, price=102.5),
    ]
    structural = [
        StructuralEvent(idx=20, kind="AR", price=98.5, z_volume=1.0),
        StructuralEvent(idx=30, kind="ST", price=101.8, z_volume=0.2),
        StructuralEvent(idx=150, kind="SOW", price=97.0, z_volume=2.0),
        StructuralEvent(idx=160, kind="LPSY", price=97.8, z_volume=0.5),
    ]
    upthrusts = [
        SpringEvent(idx=120, kind="UTAD", penetration=1.0, recovery_bars=2, z_volume=0.5),
    ]
    phases = classify_phases(bars, [tr], climaxes, structural, [], upthrusts)
    assert [p.phase for p in phases] == ["A", "B", "C", "D", "E"]
    assert all(p.schematic == "Dist" for p in phases)
