"""
Input/output conversions between polars/numpy for the timeseries subpackage.

All public timeseries APIs accept ``np.ndarray | pl.Series | pl.DataFrame``
and convert immediately to a contiguous float64 numpy array via these
helpers.  Polars never crosses a numerical kernel boundary; numba never
sees a polars object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def to_numpy_1d(x: Any) -> np.ndarray:
    """
    Convert a 1-D input to a contiguous float64 numpy array.

    Accepts ``np.ndarray`` (any shape; flattened if 2-D with one column or row),
    ``pl.Series``, ``pl.DataFrame`` (must have a single column), or any
    array-like coercible by ``np.asarray``.

    Raises
    ------
    ValueError
        If the input cannot be unambiguously interpreted as 1-D.
    """
    match x:
        case np.ndarray():
            arr = x
        case pl.Series():
            arr = x.to_numpy()
        case pl.DataFrame():
            if x.width != 1:
                raise ValueError(
                    f"to_numpy_1d: DataFrame must have exactly 1 column, got width={x.width}."
                )
            arr = x.to_numpy().ravel()
        case _:
            arr = np.asarray(x)

    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"to_numpy_1d: expected 1-D input, got shape {arr.shape}.")

    return np.ascontiguousarray(arr, dtype=np.float64)


def to_numpy_2d(x: Any) -> np.ndarray:
    """
    Convert a 2-D input to a contiguous float64 numpy array of shape (T, k).

    Accepts ``np.ndarray``, ``pl.DataFrame``, or any array-like.
    """
    match x:
        case np.ndarray():
            arr = x
        case pl.DataFrame():
            arr = x.to_numpy()
        case _:
            arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"to_numpy_2d: expected 2-D input, got shape {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def validate_min_length(x: np.ndarray, min_length: int, name: str = "x") -> None:
    """Raise ``ValueError`` if ``x`` has fewer than ``min_length`` observations."""
    if x.shape[0] < min_length:
        raise ValueError(f"{name} must have at least {min_length} observations, got {x.shape[0]}.")


def validate_finite(x: np.ndarray, name: str = "x") -> None:
    """Raise ``ValueError`` if ``x`` contains any NaN or inf."""
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values (NaN or inf).")
