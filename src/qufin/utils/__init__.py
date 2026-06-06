"""Utility functions and helpers shared across the strategy / models layers."""

from __future__ import annotations

import numpy as np
import polars as pl

_SeriesLike = np.ndarray | pl.Series

__all__ = ["to_numpy_1d"]


def to_numpy_1d(x: _SeriesLike) -> np.ndarray:
    """
    Coerce a 1-D ``polars.Series`` or ``numpy`` array to a float64 numpy array.

    The canonical converter for the ``strategies`` and ``models`` layers.  It is
    strict — a 2-D input raises rather than being silently flattened (subpackage
    converters in :mod:`qufin.data._types` / :mod:`qufin.timeseries._io` apply
    ravel semantics for their own internal use).
    """
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64, copy=False)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {arr.shape}")
    return arr
