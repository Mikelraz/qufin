"""
Implied volatility solvers.

``implied_vol`` is a vectorised Newton-Raphson with bisection fallback, seeded
by the Manaster-Koehler initial guess.  Returns ``np.nan`` for inputs that lie
outside the no-arbitrage price bounds or that fail to converge.
"""

from __future__ import annotations

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


def implied_vol(
    *,
    price: np.ndarray | float,
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float = 0.0,
    q: np.ndarray | float = 0.0,
    option_type: np.ndarray | str,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> np.ndarray | float:
    """
    Implied vol from option price.

    Parameters mirror ``black_scholes_price``; ``price`` replaces ``sigma``.
    Returns ``np.nan`` for any contract whose price falls outside the
    no-arbitrage band ``[max(0, S e^{-qT} - K e^{-rT}), S e^{-qT}]`` (call) or
    ``[max(0, K e^{-rT} - S e^{-qT}), K e^{-rT}]`` (put).
    """
    arrays = [a for a in (price, S, K, T, r, q) if isinstance(a, np.ndarray) and a.ndim > 0]
    is_scalar = not arrays and not isinstance(option_type, np.ndarray)
    n = 1 if is_scalar else max((a.shape[0] for a in arrays), default=1)
    if isinstance(option_type, np.ndarray) and option_type.ndim > 0:
        n = max(n, option_type.shape[0])

    out = _kernels.implied_vol(
        _to_f64(price, n),
        _to_f64(S, n),
        _to_f64(K, n),
        _to_f64(T, n),
        _to_f64(r, n),
        _to_f64(q, n),
        _is_call_array(option_type, n),
        float(tol),
        int(max_iter),
    )
    return float(out[0]) if is_scalar else out


def implied_vol_chain(chain: OptionChain, *, use_mid: bool = True) -> np.ndarray:
    """
    Solve implied vol for every contract in the chain from market prices.

    ``use_mid=True`` uses ``(bid+ask)/2`` (falling back to ``last``); otherwise
    the ``last`` column is used.  Designed for feeds where ``iv`` is missing or
    stale (yfinance often has zeros / NaNs).
    """
    prices = chain.mid() if use_mid else chain.data["last"].to_numpy().astype(np.float64)
    K = chain.strikes()
    out = implied_vol(
        price=prices,
        S=chain.spot,
        K=K,
        T=chain.time_to_expiry(),
        r=chain.r,
        q=chain.q,
        option_type=chain.data["option_type"].to_numpy(),
    )
    assert isinstance(out, np.ndarray)
    return out
