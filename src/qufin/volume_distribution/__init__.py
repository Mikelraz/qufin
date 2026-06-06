"""
Volume-distribution subpackage — how traded volume is distributed across price
and time, with the corresponding indicators.

Layout
------
    _types          VolumeProfile, TPOProfile, DeltaProfile, VWAPBands,
                    DistributionStats result containers + shared helpers
    _kernels        numba-jitted hot loops (VBP allocation, TPO touch counts,
                    tick-rule signing, CVD scan)
    profile         volume-by-price profile, value area, composite session
                    profiles, naked POCs, value-area migration
    tpo             Market Profile / Time-Price-Opportunity construction
    delta           signed-tick / bar volume delta, CVD, delta divergence,
                    delta-by-price footprint
    vwap            cumulative VWAP with std-dev bands, session & anchored VWAP
    stats           Gini, entropy, skew/kurtosis, profile-shape classification

Inputs are the canonical ``BAR_SCHEMA`` (``OHLCV``) and ``TICK_SCHEMA``
(price, size) containers from :mod:`qufin.data`, so the indicators work with any
vendor feed (IBKR, Revolut X, …) without coupling to it.

Quick start
-----------
    from qufin.data import OHLCV
    from qufin.volume_distribution import (
        volume_profile, tpo_profile, vwap_bands,
        cumulative_volume_delta, classify_profile_shape,
    )

    bars = OHLCV.from_records(df)
    prof = volume_profile(bars, n_bins=50)        # POC / value area / HVN-LVN
    tpo = tpo_profile(bars, period="30m")         # Market Profile
    bands = vwap_bands(bars.high(), bars.low(), bars.close(), bars.volume())
    shape = classify_profile_shape(prof)          # Gini, entropy, skew, kurtosis
    cvd = cumulative_volume_delta(ticks)          # from a TICK_SCHEMA frame
"""

from __future__ import annotations

from ._types import (
    BAR_SCHEMA,
    OHLCV,
    TICK_SCHEMA,
    DeltaProfile,
    DistributionStats,
    TPOProfile,
    VolumeProfile,
    VWAPBands,
)
from .delta import (
    bar_delta,
    cumulative_volume_delta,
    delta_divergence,
    delta_profile,
    signed_tick_volume,
)
from .profile import (
    composite_profile,
    naked_pocs,
    value_area,
    value_area_migration,
    volume_profile,
    volume_profile_from_ticks,
)
from .stats import (
    classify_profile_shape,
    profile_kurtosis,
    profile_skew,
    volume_concentration,
    volume_entropy,
)
from .tpo import bracket_letters, tpo_profile
from .vwap import anchored_vwap, session_vwap, vwap_bands

__all__ = [
    # Types / schemas
    "BAR_SCHEMA",
    "OHLCV",
    "TICK_SCHEMA",
    "DeltaProfile",
    "DistributionStats",
    "TPOProfile",
    "VWAPBands",
    "VolumeProfile",
    # Profile
    "composite_profile",
    "naked_pocs",
    "value_area",
    "value_area_migration",
    "volume_profile",
    "volume_profile_from_ticks",
    # TPO / Market Profile
    "bracket_letters",
    "tpo_profile",
    # Delta / CVD
    "bar_delta",
    "cumulative_volume_delta",
    "delta_divergence",
    "delta_profile",
    "signed_tick_volume",
    # VWAP
    "anchored_vwap",
    "session_vwap",
    "vwap_bands",
    # Stats
    "classify_profile_shape",
    "profile_kurtosis",
    "profile_skew",
    "volume_concentration",
    "volume_entropy",
]
