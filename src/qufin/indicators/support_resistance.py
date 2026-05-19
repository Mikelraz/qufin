"""
Support and resistance levels.

Two complementary methods are provided:

* **Pivot points** — deterministic levels derived from the prior bar's
  high/low/close. Both *classic* and *Fibonacci* variants are supported, in
  single-bar and rolling forms.
* **Swing clustering** — agglomerative 1-D clustering of pivot prices into
  horizontal levels. Inputs can be raw arrays or :class:`SwingPoint` records
  from :mod:`qufin.wyckoff`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._types import PivotPoints, SupportResistanceLevel, check_lengths, to_numpy_1d


def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> PivotPoints:
    """
    Classic floor pivot points derived from a single prior bar.

    Formulas
    --------
    ``PP = (H + L + C) / 3``
    ``R1 = 2·PP - L``       ``S1 = 2·PP - H``
    ``R2 = PP + (H - L)``   ``S2 = PP - (H - L)``
    ``R3 = H + 2·(PP - L)`` ``S3 = L - 2·(H - PP)``
    """
    h = float(prev_high)
    l = float(prev_low)  # noqa: E741
    c = float(prev_close)
    if h < l:
        raise ValueError(f"high ({h}) must be >= low ({l})")
    pp = (h + l + c) / 3.0
    rng = h - l
    return PivotPoints(
        pp=pp,
        r1=2.0 * pp - l,
        s1=2.0 * pp - h,
        r2=pp + rng,
        s2=pp - rng,
        r3=h + 2.0 * (pp - l),
        s3=l - 2.0 * (h - pp),
    )


def fibonacci_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> PivotPoints:
    """
    Fibonacci pivot points: ``PP`` as in the classic method, with R/S levels at
    0.382, 0.618, and 1.0 times the prior range.
    """
    h = float(prev_high)
    l = float(prev_low)  # noqa: E741
    c = float(prev_close)
    if h < l:
        raise ValueError(f"high ({h}) must be >= low ({l})")
    pp = (h + l + c) / 3.0
    rng = h - l
    return PivotPoints(
        pp=pp,
        r1=pp + 0.382 * rng,
        s1=pp - 0.382 * rng,
        r2=pp + 0.618 * rng,
        s2=pp - 0.618 * rng,
        r3=pp + rng,
        s3=pp - rng,
    )


def pivot_points_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    method: str = "classic",
) -> dict[str, np.ndarray]:
    """
    Per-bar pivot points using the previous bar's HLC.

    Returns a dict with arrays for ``PP``, ``R1``..``R3``, ``S1``..``S3``. The
    first element of each array is NaN (no prior bar available).
    """
    if method not in ("classic", "fibonacci"):
        raise ValueError(f"method must be 'classic' or 'fibonacci', got {method!r}")
    h = to_numpy_1d(high)
    l = to_numpy_1d(low)  # noqa: E741
    c = to_numpy_1d(close)
    check_lengths(h, l, c)
    n = h.shape[0]
    keys = ("PP", "R1", "R2", "R3", "S1", "S2", "S3")
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}
    if n < 2:
        return out
    ph = h[:-1]
    pl_ = l[:-1]
    pc = c[:-1]
    pp = (ph + pl_ + pc) / 3.0
    rng = ph - pl_
    if method == "classic":
        out["PP"][1:] = pp
        out["R1"][1:] = 2.0 * pp - pl_
        out["S1"][1:] = 2.0 * pp - ph
        out["R2"][1:] = pp + rng
        out["S2"][1:] = pp - rng
        out["R3"][1:] = ph + 2.0 * (pp - pl_)
        out["S3"][1:] = pl_ - 2.0 * (ph - pp)
    else:
        out["PP"][1:] = pp
        out["R1"][1:] = pp + 0.382 * rng
        out["S1"][1:] = pp - 0.382 * rng
        out["R2"][1:] = pp + 0.618 * rng
        out["S2"][1:] = pp - 0.618 * rng
        out["R3"][1:] = pp + rng
        out["S3"][1:] = pp - rng
    return out


def cluster_levels(
    prices: np.ndarray,
    kinds: Sequence[str] | np.ndarray,
    indices: np.ndarray,
    strengths: np.ndarray | None = None,
    *,
    tolerance: float = 0.005,
) -> list[SupportResistanceLevel]:
    """
    Cluster pivot prices into horizontal support/resistance levels.

    Single-link agglomerative clustering on the price axis: pivots whose price
    differs by less than ``tolerance`` (as a fraction of the smaller price) are
    merged. The output is sorted by descending ``strength``.

    Parameters
    ----------
    prices      Pivot prices, shape (n,).
    kinds       Per-pivot kind labels in ``{'H', 'L', 'S', 'R'}``. ``'H'``/``'R'``
                count as resistance contributions, ``'L'``/``'S'`` as support.
    indices     Bar indices of each pivot, shape (n,).
    strengths   Optional per-pivot strength weights (e.g. fractal width). When
                omitted every pivot weighs equally.
    tolerance   Maximum relative price gap (fraction) for two pivots to share a
                cluster — e.g. ``0.005`` merges pivots within 0.5 % of each other.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    p = to_numpy_1d(prices)
    idx = np.asarray(indices, dtype=np.int64)
    n = p.shape[0]
    if n == 0:
        return []
    if idx.shape[0] != n:
        raise ValueError(f"indices length {idx.shape[0]} != prices length {n}")
    kinds_arr = np.asarray(kinds, dtype=object)
    if kinds_arr.shape[0] != n:
        raise ValueError(f"kinds length {kinds_arr.shape[0]} != prices length {n}")
    if strengths is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = to_numpy_1d(strengths)
        if w.shape[0] != n:
            raise ValueError(f"strengths length {w.shape[0]} != prices length {n}")

    order = np.argsort(p)
    p_sorted = p[order]
    idx_sorted = idx[order]
    kinds_sorted = kinds_arr[order]
    w_sorted = w[order]

    levels: list[SupportResistanceLevel] = []
    i = 0
    while i < n:
        j = i + 1
        cluster_max = p_sorted[i]
        while j < n and p_sorted[j] - p_sorted[i] <= tolerance * min(p_sorted[i], p_sorted[j]):
            if p_sorted[j] > cluster_max:
                cluster_max = p_sorted[j]
            j += 1
        seg_p = p_sorted[i:j]
        seg_k = kinds_sorted[i:j]
        seg_idx = idx_sorted[i:j]
        seg_w = w_sorted[i:j]
        touches = j - i
        price_centre = float(seg_p.mean())
        n_res = int(np.sum([k in ("H", "R") for k in seg_k]))
        n_sup = int(np.sum([k in ("L", "S") for k in seg_k]))
        if n_res and not n_sup:
            kind = "R"
        elif n_sup and not n_res:
            kind = "S"
        else:
            kind = "SR"
        strength = float(touches * seg_w.mean())
        levels.append(
            SupportResistanceLevel(
                price=price_centre,
                kind=kind,  # type: ignore[arg-type]
                touches=int(touches),
                strength=strength,
                first_idx=int(seg_idx.min()),
                last_idx=int(seg_idx.max()),
            )
        )
        i = j
    levels.sort(key=lambda level_: level_.strength, reverse=True)
    return levels


def support_resistance_from_swings(
    swings: Sequence[object], *, tolerance: float = 0.005
) -> list[SupportResistanceLevel]:
    """
    Cluster :class:`qufin.wyckoff.SwingPoint` objects into S/R levels.

    Uses each swing's ``price`` for clustering, ``kind`` to label support vs
    resistance, ``idx`` for chronology, and ``strength`` as the weight.
    """
    if not swings:
        return []
    prices = np.array([float(s.price) for s in swings], dtype=np.float64)  # type: ignore[attr-defined]
    kinds = np.array([str(s.kind) for s in swings], dtype=object)  # type: ignore[attr-defined]
    indices = np.array([int(s.idx) for s in swings], dtype=np.int64)  # type: ignore[attr-defined]
    strengths = np.array([float(getattr(s, "strength", 1)) for s in swings], dtype=np.float64)
    return cluster_levels(prices, kinds, indices, strengths, tolerance=tolerance)
