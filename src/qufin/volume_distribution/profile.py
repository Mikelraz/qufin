"""
Volume-by-price profiles: Point of Control, value area, HVN/LVN, composite
session profiles, naked POCs, and value-area migration.

Each bar's volume is distributed uniformly across its high-low range — the
standard approximation when intrabar tick data is unavailable. When ticks are
available, :func:`volume_profile_from_ticks` builds an exact volume-at-price
histogram instead.

This module owns the canonical volume-by-price implementation for the whole
codebase; :mod:`qufin.wyckoff.volume_profile` re-exports from here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..data._types import OHLCV
from ._kernels import vbp_allocate_kernel
from ._types import VolumeProfile, as_tick_arrays, coerce_ohlcv, value_area_bounds
from .vwap import anchored_vwap  # noqa: F401  (re-exported for wyckoff compat)

__all__ = [
    "anchored_vwap",
    "composite_profile",
    "naked_pocs",
    "value_area",
    "value_area_migration",
    "volume_profile",
    "volume_profile_from_ticks",
]


def _empty_profile(p_lo: float, total: float) -> VolumeProfile:
    """Degenerate single-price profile (all bars at one level)."""
    edges = np.array([p_lo, p_lo + 1.0], dtype=np.float64)
    vol_hist = np.array([total], dtype=np.float64)
    return VolumeProfile(
        price_bins=edges,
        volume=vol_hist,
        poc=p_lo,
        vah=p_lo,
        val=p_lo,
        hvn_idx=np.zeros(0, dtype=np.int64),
        lvn_idx=np.zeros(0, dtype=np.int64),
    )


def _local_extrema(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = v.shape[0]
    if n < 3:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    hvn: list[int] = []
    lvn: list[int] = []
    for i in range(1, n - 1):
        if v[i] > v[i - 1] and v[i] > v[i + 1]:
            hvn.append(i)
        elif v[i] < v[i - 1] and v[i] < v[i + 1]:
            lvn.append(i)
    return np.array(hvn, dtype=np.int64), np.array(lvn, dtype=np.int64)


def _profile_from_hist(edges: np.ndarray, hist: np.ndarray, frac: float) -> VolumeProfile:
    poc_idx = int(np.argmax(hist))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])
    val, vah = value_area_bounds(edges, hist, frac)
    hvn_idx, lvn_idx = _local_extrema(hist)
    return VolumeProfile(
        price_bins=edges,
        volume=hist,
        poc=float(poc),
        vah=vah,
        val=val,
        hvn_idx=hvn_idx,
        lvn_idx=lvn_idx,
    )


def volume_profile(
    bars: OHLCV | pl.DataFrame,
    *,
    n_bins: int = 50,
    start: int = 0,
    end: int | None = None,
    value_area_frac: float = 0.70,
) -> VolumeProfile:
    """
    Build a volume-by-price profile over a bar range.

    Each bar's volume ``v_t`` is spread uniformly across its high-low range
    using fractional-area allocation: a price bin overlapping the bar by
    ``Δp`` receives ``v_t · Δp / (high_t - low_t)``. Zero-range bars deposit
    their full volume in the single bin containing the price.

    Parameters
    ----------
    bars             ``OHLCV`` sequence (or a ``BAR_SCHEMA`` DataFrame).
    n_bins           Number of price bins (uniform width).
    start, end       Half-open window of bar indices to profile.
    value_area_frac  Fraction of total volume contained in the value area.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not 0.0 < value_area_frac <= 1.0:
        raise ValueError(f"value_area_frac must be in (0, 1], got {value_area_frac}")
    ohlcv = coerce_ohlcv(bars)
    n = ohlcv.n_bars
    if end is None:
        end = n
    if not 0 <= start < end <= n:
        raise ValueError(f"invalid window [{start}, {end}) for {n} bars")

    high = ohlcv.high()[start:end]
    low = ohlcv.low()[start:end]
    vol = ohlcv.volume()[start:end]

    p_lo = float(low.min())
    p_hi = float(high.max())
    if p_hi <= p_lo:
        return _empty_profile(p_lo, float(vol.sum()))

    edges = np.linspace(p_lo, p_hi, n_bins + 1)
    bin_width = (p_hi - p_lo) / n_bins
    vol_hist = vbp_allocate_kernel(high, low, vol, edges, p_lo, bin_width, n_bins)
    return _profile_from_hist(edges, vol_hist, value_area_frac)


def volume_profile_from_ticks(
    ticks: pl.DataFrame,
    *,
    n_bins: int = 50,
    value_area_frac: float = 0.70,
) -> VolumeProfile:
    """
    Exact volume-at-price profile from a ``TICK_SCHEMA`` (price, size) frame.

    Each trade deposits its full size in the bin containing its price — no
    intrabar approximation is needed.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not 0.0 < value_area_frac <= 1.0:
        raise ValueError(f"value_area_frac must be in (0, 1], got {value_area_frac}")
    price, size = as_tick_arrays(ticks)
    if price.shape[0] == 0:
        raise ValueError("tick frame is empty")
    p_lo = float(price.min())
    p_hi = float(price.max())
    if p_hi <= p_lo:
        return _empty_profile(p_lo, float(size.sum()))
    edges = np.linspace(p_lo, p_hi, n_bins + 1)
    vol_hist, _ = np.histogram(price, bins=edges, weights=size)
    return _profile_from_hist(edges, vol_hist.astype(np.float64, copy=False), value_area_frac)


def value_area(profile: VolumeProfile, frac: float = 0.70) -> tuple[float, float]:
    """Re-compute ``(val, vah)`` at a different ``frac`` from an existing profile."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if profile.volume.shape[0] == 0:
        return profile.val, profile.vah
    return value_area_bounds(profile.price_bins, profile.volume, frac)


def composite_profile(
    bars: OHLCV | pl.DataFrame,
    period: str = "1d",
    *,
    n_bins: int = 50,
    value_area_frac: float = 0.70,
) -> list[VolumeProfile]:
    """
    One volume profile per session, splitting bars on a fixed ``period``.

    ``period`` is a polars duration string (``"1d"``, ``"1h"``, ``"30m"`` …).
    Sessions are formed with :meth:`polars.DataFrame.group_by_dynamic` on the
    UTC ``timestamp`` column; profiles are returned in chronological order.
    """
    ohlcv = coerce_ohlcv(bars)
    if ohlcv.n_bars == 0:
        return []
    tagged = ohlcv.data.sort("timestamp").with_columns(
        pl.col("timestamp").dt.truncate(period).alias("__session__")
    )
    profiles: list[VolumeProfile] = []
    for session in tagged.partition_by("__session__", maintain_order=True):
        frame = session.drop("__session__")
        profiles.append(
            volume_profile(
                OHLCV(data=frame, symbol=ohlcv.symbol),
                n_bins=n_bins,
                value_area_frac=value_area_frac,
            )
        )
    return profiles


def naked_pocs(profiles: list[VolumeProfile], prices: np.ndarray) -> np.ndarray:
    """
    Return the POCs (one per profile) never revisited by any later price.

    A "naked" / virgin POC is a prior session's Point of Control that price has
    not traded back through. ``prices`` is the full forward price path (e.g.
    bar closes) used to test revisits.

    Each POC at index ``i`` is naked if no price *after* that session crosses
    it; since we only have per-profile POCs (not their session end indices),
    the conservative test used here is: the POC is naked if it lies outside the
    ``[min, max]`` envelope of all *subsequent* profiles' price ranges.
    """
    p = np.asarray(prices, dtype=np.float64)
    out: list[float] = []
    for i, prof in enumerate(profiles):
        poc = prof.poc
        later = profiles[i + 1 :]
        revisited = False
        for lp in later:
            lo = float(lp.price_bins[0])
            hi = float(lp.price_bins[-1])
            if lo <= poc <= hi:
                revisited = True
                break
        if not revisited and p.shape[0] > 0:
            # also guard against the forward price path crossing the level
            revisited = bool(np.any((p[:-1] - poc) * (p[1:] - poc) <= 0.0))
        if not revisited:
            out.append(poc)
    return np.array(out, dtype=np.float64)


def value_area_migration(
    bars: OHLCV | pl.DataFrame,
    period: str = "1d",
    *,
    n_bins: int = 50,
    value_area_frac: float = 0.70,
) -> pl.DataFrame:
    """
    Per-session POC / VAH / VAL series for value-area trend analysis.

    Returns a DataFrame with columns ``session`` (0-based index), ``poc``,
    ``vah``, and ``val`` — one row per session profile.
    """
    profiles = composite_profile(bars, period, n_bins=n_bins, value_area_frac=value_area_frac)
    return pl.DataFrame(
        {
            "session": list(range(len(profiles))),
            "poc": [p.poc for p in profiles],
            "vah": [p.vah for p in profiles],
            "val": [p.val for p in profiles],
        }
    )
