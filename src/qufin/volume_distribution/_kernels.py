"""
Numba-jitted hot loops for the volume-distribution subpackage.

Only inherently-sequential or hard-to-vectorise routines live here:

* ``vbp_allocate_kernel``    — fractional-area volume-by-price allocation.
* ``tpo_touch_kernel``       — per-bracket price-level touch counting (TPO).
* ``tick_rule_sign_kernel``  — uptick/downtick trade-side classification.
* ``cvd_kernel``             — cumulative sum of signed volume.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def vbp_allocate_kernel(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    edges: np.ndarray,
    p_lo: float,
    bin_width: float,
    n_bins: int,
) -> np.ndarray:
    """
    Distribute each bar's volume across price bins by fractional overlap.

    A bin overlapping a bar's ``[low, high]`` range by ``Δp`` receives
    ``v * Δp / (high - low)``. Zero-range bars deposit their full volume in the
    single bin containing the price. ``edges`` are the bin boundaries (length
    ``n_bins + 1``); using them verbatim keeps results bit-for-bit identical to
    the reference loop the Wyckoff profile shipped with.
    """
    hist = np.zeros(n_bins, dtype=np.float64)
    for t in range(high.shape[0]):
        h = high[t]
        l = low[t]  # noqa: E741
        v = volume[t]
        if v <= 0.0:
            continue
        if h == l:
            k = int((h - p_lo) / bin_width)
            if k > n_bins - 1:
                k = n_bins - 1
            hist[k] += v
            continue
        k_lo = int((l - p_lo) / bin_width)
        if k_lo < 0:
            k_lo = 0
        k_hi = int((h - p_lo) / bin_width)
        if k_hi > n_bins - 1:
            k_hi = n_bins - 1
        span = h - l
        for k in range(k_lo, k_hi + 1):
            be = edges[k]
            be_next = edges[k + 1]
            hi = h if h < be_next else be_next
            lo = l if l > be else be
            overlap = hi - lo
            if overlap > 0.0:
                hist[k] += v * overlap / span
    return hist


@njit(cache=True)
def tpo_touch_kernel(
    high: np.ndarray,
    low: np.ndarray,
    bracket: np.ndarray,
    p_lo: float,
    bin_width: float,
    n_bins: int,
) -> np.ndarray:
    """
    Count, per price bin, the number of distinct time brackets that traded it.

    ``bracket`` holds the bracket id of each bar (non-decreasing). A bin is
    credited at most once per bracket — the classic TPO count. Returns a
    ``(n_bins,)`` float64 array of counts.
    """
    counts = np.zeros(n_bins, dtype=np.float64)
    seen = np.full(n_bins, -1, dtype=np.int64)
    for t in range(high.shape[0]):
        b = bracket[t]
        k_lo = int((low[t] - p_lo) / bin_width)
        if k_lo < 0:
            k_lo = 0
        k_hi = int((high[t] - p_lo) / bin_width)
        if k_hi > n_bins - 1:
            k_hi = n_bins - 1
        for k in range(k_lo, k_hi + 1):
            if seen[k] != b:
                seen[k] = b
                counts[k] += 1.0
    return counts


@njit(cache=True)
def tick_rule_sign_kernel(price: np.ndarray) -> np.ndarray:
    """
    Sign each tick by the tick rule: +1 on an uptick, -1 on a downtick, and
    the previous sign carried forward on a zero tick (first tick seeds +1).
    """
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    prev = 1.0
    out[0] = prev
    for i in range(1, n):
        d = price[i] - price[i - 1]
        if d > 0.0:
            prev = 1.0
        elif d < 0.0:
            prev = -1.0
        out[i] = prev
    return out


@njit(cache=True)
def cvd_kernel(signed_volume: np.ndarray) -> np.ndarray:
    """Cumulative sum of signed volume (cumulative volume delta)."""
    n = signed_volume.shape[0]
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        acc += signed_volume[i]
        out[i] = acc
    return out
