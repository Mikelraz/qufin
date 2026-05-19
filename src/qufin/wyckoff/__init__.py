"""
Wyckoff subpackage — quantitative tools for Wyckoff-method market analysis.

Layout
------
    _types              OHLCV, SwingPoint, TradingRange, ClimaxEvent,
                        SpringEvent, StructuralEvent, WyckoffPhase,
                        VolumeProfile, PnFChart, CauseEffectTarget
    _kernels            numba-jitted scan kernels (fractal swings, zigzag, P&F)
    bars                OHLCV validation, true_range, ATR, resample,
                        rolling z-scores and slopes
    swings              fractal & zigzag pivot detection
    ranges              trading-range detection from swings + ATR
    volume_profile      VBP, POC, value area, HVN/LVN, anchored VWAP
    effort_result       effort-vs-result rolling z-score and absorption flags
    events              SC, BC, AR, ST, Spring, UT, UTAD, SOS, LPS, SOW, LPSY
    phases              rule-based Phase A-E classifier
    phases_hmm          optional 4-state Gaussian HMM macro classifier
    point_figure        P&F chart construction with horizontal/vertical counts
    relative_strength   Wyckoff RS = asset/benchmark and cross-sectional RS rank

Quick start
-----------
    from qufin.wyckoff import (
        OHLCV, find_swings, detect_trading_ranges,
        detect_climax, detect_automatic_rally, detect_secondary_test,
        detect_spring, detect_sos_lps, classify_phases,
        volume_profile, pnf_from_bars, vertical_count,
    )

    bars = OHLCV.from_records(df)                       # validated OHLCV frame
    ranges = detect_trading_ranges(bars, min_bars=20)
    climaxes = detect_climax(bars)
    ar = detect_automatic_rally(bars, climaxes[0])
    st = detect_secondary_test(bars, climaxes[0], ar) if ar else None
    springs = detect_spring(bars, ranges[0]) if ranges else []
    sos, lps = detect_sos_lps(bars, ranges[0]) if ranges else ([], [])

    structural = [e for e in [ar, st] if e is not None] + sos + lps
    phases = classify_phases(bars, ranges, climaxes, structural, springs)

    profile = volume_profile(bars, n_bins=50)
    chart = pnf_from_bars(bars)
    target = vertical_count(chart, column_idx=-1)
"""

from __future__ import annotations

from ._types import (
    BAR_SCHEMA,
    OHLCV,
    CauseEffectTarget,
    ClimaxEvent,
    PnFChart,
    PnFColumn,
    SpringEvent,
    StructuralEvent,
    SwingPoint,
    TradingRange,
    VolumeProfile,
    WyckoffPhase,
)
from .bars import atr, normalize_volume, resample, rolling_slope, true_range, validate_ohlcv
from .effort_result import EffortResult, effort_vs_result
from .events import (
    detect_automatic_rally,
    detect_climax,
    detect_preliminary_support,
    detect_secondary_test,
    detect_sos_lps,
    detect_sow_lpsy,
    detect_spring,
    detect_upthrust,
)
from .phases import classify_phases
from .phases_hmm import HMMPhaseResult, WyckoffHMMClassifier
from .point_figure import (
    default_box_size,
    horizontal_count,
    pnf_from_bars,
    vertical_count,
)
from .ranges import detect_trading_ranges, is_in_range
from .relative_strength import relative_strength, rs_rank, rs_slope
from .swings import find_swings, swing_extremes, zigzag
from .volume_profile import anchored_vwap, value_area, volume_profile

__all__ = [
    # Types / schemas
    "BAR_SCHEMA",
    "OHLCV",
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
    # Bars
    "validate_ohlcv",
    "true_range",
    "atr",
    "resample",
    "normalize_volume",
    "rolling_slope",
    # Swings
    "find_swings",
    "zigzag",
    "swing_extremes",
    # Ranges
    "detect_trading_ranges",
    "is_in_range",
    # Volume profile
    "volume_profile",
    "value_area",
    "anchored_vwap",
    # Effort vs result
    "EffortResult",
    "effort_vs_result",
    # Events
    "detect_climax",
    "detect_automatic_rally",
    "detect_secondary_test",
    "detect_spring",
    "detect_upthrust",
    "detect_sos_lps",
    "detect_sow_lpsy",
    "detect_preliminary_support",
    # Phases
    "classify_phases",
    "WyckoffHMMClassifier",
    "HMMPhaseResult",
    # Point and Figure
    "default_box_size",
    "pnf_from_bars",
    "vertical_count",
    "horizontal_count",
    # Relative strength
    "relative_strength",
    "rs_slope",
    "rs_rank",
]
