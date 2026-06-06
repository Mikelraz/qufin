"""
Market Profile / Time-Price-Opportunity (TPO) construction.

Bars are grouped into fixed time brackets (default 30 minutes). Each bracket is
assigned a letter (``A``, ``B``, …) and every price level it trades through
earns one TPO. The resulting count-by-price histogram yields the TPO Point of
Control, value area, initial balance, single prints, and range extension.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..data._types import OHLCV
from ._kernels import tpo_touch_kernel
from ._types import TPOProfile, coerce_ohlcv, value_area_bounds

__all__ = ["bracket_letters", "tpo_profile"]

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def bracket_letters(n: int) -> list[str]:
    """Return ``n`` bracket labels: A…Z, a…z, then AA, AB, … for overflow."""
    out: list[str] = []
    for i in range(n):
        if i < len(_ALPHABET):
            out.append(_ALPHABET[i])
        else:
            first = _ALPHABET[(i // len(_ALPHABET)) - 1]
            second = _ALPHABET[i % len(_ALPHABET)]
            out.append(first + second)
    return out


def tpo_profile(
    bars: OHLCV | pl.DataFrame,
    *,
    n_bins: int = 50,
    period: str = "30m",
    n_initial: int = 2,
    value_area_frac: float = 0.70,
) -> TPOProfile:
    """
    Build a TPO / Market Profile over a session of bars.

    Parameters
    ----------
    bars             ``OHLCV`` sequence (typically one trading session).
    n_bins           Number of price bins (uniform width).
    period           Polars duration string for one time bracket.
    n_initial        Number of opening brackets forming the initial balance.
    value_area_frac  Fraction of TPOs contained in the value area.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if n_initial < 1:
        raise ValueError(f"n_initial must be >= 1, got {n_initial}")
    if not 0.0 < value_area_frac <= 1.0:
        raise ValueError(f"value_area_frac must be in (0, 1], got {value_area_frac}")
    ohlcv = coerce_ohlcv(bars)
    if ohlcv.n_bars == 0:
        raise ValueError("cannot build a TPO profile from zero bars")

    tagged = ohlcv.data.sort("timestamp").with_columns(
        pl.col("timestamp").dt.truncate(period).alias("__bracket__")
    )
    bracket_ids = tagged["__bracket__"].rle_id().to_numpy().astype(np.int64, copy=False)
    n_brackets = int(bracket_ids[-1]) + 1

    high = tagged["high"].to_numpy().astype(np.float64, copy=False)
    low = tagged["low"].to_numpy().astype(np.float64, copy=False)

    p_lo = float(low.min())
    p_hi = float(high.max())
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    edges = np.linspace(p_lo, p_hi, n_bins + 1)
    bin_width = (p_hi - p_lo) / n_bins
    counts = tpo_touch_kernel(high, low, bracket_ids, p_lo, bin_width, n_bins)

    poc_idx = int(np.argmax(counts))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])
    val, vah = value_area_bounds(edges, counts, value_area_frac)

    letters = _bin_letter_strings(bracket_ids, high, low, p_lo, bin_width, n_bins, n_brackets)

    ib_mask = bracket_ids < n_initial
    ib_low = float(low[ib_mask].min())
    ib_high = float(high[ib_mask].max())

    single_prints = np.flatnonzero(counts == 1.0).astype(np.int64)
    ext_up = bool(high.max() > ib_high)
    ext_down = bool(low.min() < ib_low)

    return TPOProfile(
        price_bins=edges,
        tpo_counts=counts,
        letters=letters,
        poc=float(poc),
        vah=vah,
        val=val,
        initial_balance=(ib_low, ib_high),
        single_prints=single_prints,
        range_extension_up=ext_up,
        range_extension_down=ext_down,
    )


def _bin_letter_strings(
    bracket_ids: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    p_lo: float,
    bin_width: float,
    n_bins: int,
    n_brackets: int,
) -> list[str]:
    """Per-bin string of the bracket letters that touched the price level."""
    labels = bracket_letters(n_brackets)
    seen: list[set[int]] = [set() for _ in range(n_bins)]
    for t in range(high.shape[0]):
        b = int(bracket_ids[t])
        k_lo = max(int((low[t] - p_lo) / bin_width), 0)
        k_hi = min(int((high[t] - p_lo) / bin_width), n_bins - 1)
        for k in range(k_lo, k_hi + 1):
            seen[k].add(b)
    return ["".join(labels[b] for b in sorted(s)) for s in seen]
