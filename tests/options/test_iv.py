"""Implied volatility solver: round-trip and edge cases."""

from __future__ import annotations

import math

import numpy as np

from qufin.options import black_scholes_price, implied_vol


def test_iv_round_trip_scalar() -> None:
    S, K, T, r, q, sigma = 100.0, 105.0, 0.6, 0.04, 0.01, 0.27
    price = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="C")
    iv = implied_vol(price=price, S=S, K=K, T=T, r=r, q=q, option_type="C")
    assert math.isclose(iv, sigma, abs_tol=1e-6)


def test_iv_round_trip_vectorised() -> None:
    rng = np.random.default_rng(11)
    n = 32
    S = rng.uniform(80, 120, n)
    K = rng.uniform(80, 120, n)
    T = rng.uniform(0.05, 1.5, n)
    sigma = rng.uniform(0.10, 0.60, n)
    r = np.full(n, 0.03)
    q = np.full(n, 0.0)
    types = np.where(rng.uniform(size=n) > 0.5, "C", "P")
    prices = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=types)
    iv = implied_vol(price=prices, S=S, K=K, T=T, r=r, q=q, option_type=types)
    # Drop trivially-zero prices: IV is indeterminate for deep OTM (vega ≈ 0).
    keep = np.isfinite(iv) & (prices > 1e-4)
    assert keep.mean() > 0.85
    assert np.allclose(iv[keep], sigma[keep], atol=1e-5)


def test_iv_returns_nan_outside_bounds() -> None:
    # Negative price is unreachable; price above S e^{-qT} is too rich.
    iv_high = implied_vol(price=1000.0, S=100.0, K=100.0, T=1.0, r=0.0, q=0.0, option_type="C")
    assert math.isnan(iv_high)
    iv_neg = implied_vol(price=-1.0, S=100.0, K=100.0, T=1.0, r=0.0, q=0.0, option_type="C")
    assert math.isnan(iv_neg)
