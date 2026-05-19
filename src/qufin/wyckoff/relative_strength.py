"""
Wyckoff-style relative strength (RS).

Two utilities:

* :func:`relative_strength` — the canonical ``asset / benchmark`` ratio,
  optionally normalised so the first observation is 1.0. A rising RS line
  during a market down-leg is the classical Wyckoff "stronger than the
  market" signature.
* :func:`rs_rank` — cross-sectional rank of an asset's RS slope over a
  trailing window. Returns a value in ``[0, 1]`` per asset (1.0 = strongest).
"""

from __future__ import annotations

import numpy as np

from ._types import to_numpy_1d


def relative_strength(
    asset: np.ndarray,
    benchmark: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the relative-strength series of an asset against a benchmark.

    ``RS_t = asset_t / benchmark_t`` (optionally rebased so ``RS_0 = 1``).
    Both series must have the same length and be strictly positive.
    """
    a = to_numpy_1d(asset)
    b = to_numpy_1d(benchmark)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: asset={a.shape[0]}, benchmark={b.shape[0]}")
    if (b <= 0.0).any() or (a <= 0.0).any():
        raise ValueError("asset and benchmark must be strictly positive")
    rs = a / b
    if normalize:
        rs = rs / rs[0]
    return rs


def rs_slope(rs: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OLS slope of a relative-strength series over ``window`` samples.

    NaN-padded for the first ``window - 1`` positions.
    """
    from .bars import rolling_slope

    return rolling_slope(np.ascontiguousarray(rs, dtype=np.float64), window)


def rs_rank(
    series: dict[str, np.ndarray],
    benchmark: np.ndarray,
    *,
    window: int = 63,
) -> dict[str, np.ndarray]:
    """
    Cross-sectional rolling rank of each asset's RS slope vs the universe.

    Parameters
    ----------
    series     ``{symbol: price_series}`` — all aligned to the benchmark.
    benchmark  Benchmark price series.
    window     Trailing window (bars) for the RS slope; default 63 (~quarter).

    Returns
    -------
    ``{symbol: rank_series}`` with each rank in [0, 1], 1.0 being the
    strongest cross-sectionally at that bar. NaN until enough data exists.
    """
    if not series:
        return {}
    n = benchmark.shape[0]
    slopes: dict[str, np.ndarray] = {}
    for sym, prices in series.items():
        rs = relative_strength(prices, benchmark, normalize=True)
        slopes[sym] = rs_slope(rs, window)
    symbols = list(slopes.keys())
    n_assets = len(symbols)
    stacked = np.stack([slopes[s] for s in symbols], axis=0)  # (n_assets, n)
    ranks = np.full_like(stacked, np.nan)
    for t in range(n):
        col = stacked[:, t]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        order = np.argsort(np.where(valid, col, -np.inf))
        positions = np.full(n_assets, np.nan, dtype=np.float64)
        valid_idx = order[-int(valid.sum()) :]
        for k, idx in enumerate(valid_idx):
            positions[idx] = (k + 1) / max(int(valid.sum()), 1)
        ranks[:, t] = positions
    return {sym: ranks[i] for i, sym in enumerate(symbols)}
