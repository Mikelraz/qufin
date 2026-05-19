"""
Volume-by-price profile, Point of Control, value area, and anchored VWAP.

Each bar's volume is distributed uniformly across its high-low range, which
is the standard approximation in absence of intrabar tick data. The resulting
histogram supports:

* ``poc`` — Point of Control: bin centre with the most volume.
* ``vah`` / ``val`` — Value Area High / Low: tightest range containing
  ``frac`` (default 0.70) of total volume centred on the POC.
* ``hvn_idx`` / ``lvn_idx`` — local maxima / minima in volume.

``anchored_vwap`` returns the running volume-weighted average price from a
chosen anchor index forward.
"""

from __future__ import annotations

import numpy as np

from ._types import OHLCV, VolumeProfile


def volume_profile(
    bars: OHLCV,
    *,
    n_bins: int = 50,
    start: int = 0,
    end: int | None = None,
    value_area_frac: float = 0.70,
) -> VolumeProfile:
    """
    Build a volume-by-price profile over a bar range.

    Each bar's volume ``v_t`` is spread uniformly across its high-low range
    using a fractional-area allocation: a price bin overlapping the bar by
    ``Δp`` receives ``v_t · Δp / (high_t - low_t)``. Zero-range bars deposit
    their full volume in the single bin containing the price.

    Parameters
    ----------
    bars             OHLCV sequence.
    n_bins           Number of price bins (uniform width).
    start, end       Half-open window of bar indices to profile.
    value_area_frac  Fraction of total volume contained in the value area.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not 0.0 < value_area_frac <= 1.0:
        raise ValueError(f"value_area_frac must be in (0, 1], got {value_area_frac}")
    n = bars.n_bars
    if end is None:
        end = n
    if not 0 <= start < end <= n:
        raise ValueError(f"invalid window [{start}, {end}) for {n} bars")

    high = bars.high()[start:end]
    low = bars.low()[start:end]
    vol = bars.volume()[start:end]

    p_lo = float(low.min())
    p_hi = float(high.max())
    if p_hi <= p_lo:
        edges = np.array([p_lo, p_lo + 1.0], dtype=np.float64)
        vol_hist = np.array([float(vol.sum())], dtype=np.float64)
        return VolumeProfile(
            price_bins=edges,
            volume=vol_hist,
            poc=p_lo,
            vah=p_lo,
            val=p_lo,
            hvn_idx=np.zeros(0, dtype=np.int64),
            lvn_idx=np.zeros(0, dtype=np.int64),
        )

    edges = np.linspace(p_lo, p_hi, n_bins + 1)
    bin_width = (p_hi - p_lo) / n_bins
    vol_hist = np.zeros(n_bins, dtype=np.float64)

    for t in range(high.shape[0]):
        h = float(high[t])
        l = float(low[t])  # noqa: E741
        v = float(vol[t])
        if v <= 0.0:
            continue
        if h == l:
            k = min(int((h - p_lo) / bin_width), n_bins - 1)
            vol_hist[k] += v
            continue
        # Indices of bins overlapping [l, h]; use clamped continuous overlap.
        k_lo = max(int((l - p_lo) / bin_width), 0)
        k_hi = min(int((h - p_lo) / bin_width), n_bins - 1)
        span = h - l
        for k in range(k_lo, k_hi + 1):
            be = edges[k]
            be_next = edges[k + 1]
            overlap = min(h, be_next) - max(l, be)
            if overlap > 0.0:
                vol_hist[k] += v * overlap / span

    poc_idx = int(np.argmax(vol_hist))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])

    total = float(vol_hist.sum())
    target = total * value_area_frac
    lo = hi = poc_idx
    acc = vol_hist[poc_idx]
    while acc < target and (lo > 0 or hi < n_bins - 1):
        left_v = vol_hist[lo - 1] if lo > 0 else -1.0
        right_v = vol_hist[hi + 1] if hi < n_bins - 1 else -1.0
        if right_v >= left_v and hi < n_bins - 1:
            hi += 1
            acc += vol_hist[hi]
        elif lo > 0:
            lo -= 1
            acc += vol_hist[lo]
        else:
            break
    val = float(edges[lo])
    vah = float(edges[hi + 1])

    hvn_idx, lvn_idx = _local_extrema(vol_hist)

    return VolumeProfile(
        price_bins=edges,
        volume=vol_hist,
        poc=float(poc),
        vah=vah,
        val=val,
        hvn_idx=hvn_idx,
        lvn_idx=lvn_idx,
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


def value_area(profile: VolumeProfile, frac: float = 0.70) -> tuple[float, float]:
    """Re-compute value area at a different ``frac`` from an existing profile."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    edges = profile.price_bins
    hist = profile.volume
    n_bins = hist.shape[0]
    if n_bins == 0:
        return profile.val, profile.vah
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


def anchored_vwap(bars: OHLCV, anchor_idx: int) -> np.ndarray:
    """
    Running volume-weighted average price from ``anchor_idx`` forward.

    Uses the typical price ``(high + low + close) / 3`` as the per-bar
    contribution. Returns an array of length ``len(bars)``; entries before
    ``anchor_idx`` are NaN.
    """
    n = bars.n_bars
    if not 0 <= anchor_idx < n:
        raise ValueError(f"anchor_idx out of range: {anchor_idx} for {n} bars")
    h = bars.high()
    l = bars.low()  # noqa: E741
    c = bars.close()
    v = bars.volume()
    tp = (h + l + c) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(anchor_idx, n):
        cum_pv += tp[i] * v[i]
        cum_v += v[i]
        out[i] = cum_pv / cum_v if cum_v > 0.0 else np.nan
    return out
