"""
Black-Scholes pricing (vectorised public API).

The numerical work happens in ``_kernels.bs_price`` — this module is the
broadcasting / coercion layer over polars / numpy inputs.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from . import _kernels
from ._types import CALL, OptionChain


def _to_f64(x: np.ndarray | float, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    if arr.shape[0] != n:
        raise ValueError(f"length mismatch: expected {n}, got {arr.shape[0]}")
    return np.ascontiguousarray(arr)


def _is_call_array(option_type: np.ndarray | str, n: int) -> np.ndarray:
    if isinstance(option_type, str):
        return np.full(n, 1 if option_type == CALL else 0, dtype=np.uint8)
    arr = np.asarray(option_type)
    if arr.ndim == 0:
        return np.full(n, 1 if str(arr) == CALL else 0, dtype=np.uint8)
    return np.ascontiguousarray((arr == CALL).astype(np.uint8))


@overload
def black_scholes_price(
    *, S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> float: ...
@overload
def black_scholes_price(
    *,
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    q: np.ndarray | float,
    sigma: np.ndarray | float,
    option_type: np.ndarray | str,
) -> np.ndarray: ...
def black_scholes_price(
    *,
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    q: np.ndarray | float,
    sigma: np.ndarray | float,
    option_type: np.ndarray | str,
) -> np.ndarray | float:
    """
    Black-Scholes-Merton price (European).

    Scalar inputs return a scalar.  Mixed scalar/array inputs broadcast against
    the longest array — every array argument must share that length.

    Parameters
    ----------
    S          Spot.
    K          Strike.
    T          Time to expiry in years.
    r          Continuously compounded risk-free rate.
    q          Continuously compounded dividend yield.
    sigma      Volatility (annualised, decimal).
    option_type ``'C'`` or ``'P'``.
    """
    arrays = [a for a in (S, K, T, r, q, sigma) if isinstance(a, np.ndarray) and a.ndim > 0]
    is_scalar = not arrays and not isinstance(option_type, np.ndarray)
    n = 1 if is_scalar else max((a.shape[0] for a in arrays), default=1)
    if isinstance(option_type, np.ndarray) and option_type.ndim > 0:
        n = max(n, option_type.shape[0])

    out = _kernels.bs_price(
        _to_f64(S, n),
        _to_f64(K, n),
        _to_f64(T, n),
        _to_f64(r, n),
        _to_f64(q, n),
        _to_f64(sigma, n),
        _is_call_array(option_type, n),
    )
    return float(out[0]) if is_scalar else out


def price_chain(chain: OptionChain, *, sigma: np.ndarray | None = None) -> np.ndarray:
    """
    Price every contract in an option chain at the chain's snapshot spot.

    Uses the chain's stored ``iv`` column by default; pass ``sigma`` to override.
    """
    K = chain.strikes()
    T = chain.time_to_expiry()
    s = chain.implied_vols() if sigma is None else _to_f64(sigma, K.shape[0])
    return _kernels.bs_price(
        np.full(K.shape[0], chain.spot, dtype=np.float64),
        K,
        T,
        np.full(K.shape[0], chain.r, dtype=np.float64),
        np.full(K.shape[0], chain.q, dtype=np.float64),
        s,
        chain.is_call().astype(np.uint8),
    )
