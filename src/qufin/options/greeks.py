"""
Black-Scholes greeks (vectorised public API).

Returns the ``Greeks`` dataclass with first- and selected second-order greeks.
All greeks are reported per one unit of underlying — multiply by
``OptionChain.multiplier`` and open interest to get dealer-position exposures
(see ``qufin.options.gex``).
"""

from __future__ import annotations

import numpy as np

from . import _kernels
from ._types import CALL, Greeks, OptionChain


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


def black_scholes_greeks(
    *,
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: np.ndarray | float,
    q: np.ndarray | float,
    sigma: np.ndarray | float,
    option_type: np.ndarray | str,
) -> Greeks:
    """
    Compute Δ, Γ, vega, θ, ρ, vanna, charm, vomma, speed.

    Conventions:
    * Δ, Γ, vanna, charm are per 1.0 change in spot / vol / time-to-expiry.
    * vega and vomma are per 1.0 change in vol (not per 1%).
    * θ is per year (divide by 365 for per-day).
    """
    arrays = [a for a in (S, K, T, r, q, sigma) if isinstance(a, np.ndarray) and a.ndim > 0]
    n = max((a.shape[0] for a in arrays), default=1)
    if isinstance(option_type, np.ndarray) and option_type.ndim > 0:
        n = max(n, option_type.shape[0])

    delta, gamma, vega, theta, rho, vanna, charm, vomma, speed = _kernels.bs_greeks(
        _to_f64(S, n),
        _to_f64(K, n),
        _to_f64(T, n),
        _to_f64(r, n),
        _to_f64(q, n),
        _to_f64(sigma, n),
        _is_call_array(option_type, n),
    )
    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        charm=charm,
        vomma=vomma,
        speed=speed,
    )


def greeks_for_chain(chain: OptionChain, *, sigma: np.ndarray | None = None) -> Greeks:
    """Greeks for every contract in a chain (uses chain.iv unless overridden)."""
    K = chain.strikes()
    T = chain.time_to_expiry()
    s = chain.implied_vols() if sigma is None else _to_f64(sigma, K.shape[0])
    return black_scholes_greeks(
        S=chain.spot,
        K=K,
        T=T,
        r=chain.r,
        q=chain.q,
        sigma=s,
        option_type=chain.data["option_type"].to_numpy(),
    )
