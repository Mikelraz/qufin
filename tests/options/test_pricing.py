"""Black-Scholes pricing correctness and put-call parity."""

from __future__ import annotations

import math

import numpy as np

from qufin.options import black_scholes_price


def _reference_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    from math import erf, exp, log, sqrt

    def N(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    v = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)


def test_call_price_matches_textbook_reference() -> None:
    p = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type="C")
    expected = _reference_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    assert math.isclose(p, expected, rel_tol=1e-12)


def test_put_call_parity_holds() -> None:
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.03, 0.01, 0.25
    c = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="C")
    p = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="P")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, abs_tol=1e-10)


def test_intrinsic_at_zero_time_and_zero_vol() -> None:
    # T = 0: pure intrinsic.
    assert (
        black_scholes_price(S=110.0, K=100.0, T=0.0, r=0.0, q=0.0, sigma=0.2, option_type="C")
        == 10.0
    )
    # sigma = 0: still intrinsic (degenerate).
    assert (
        black_scholes_price(S=90.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=0.0, option_type="P")
        == 10.0
    )


def test_vectorised_matches_scalar() -> None:
    rng = np.random.default_rng(7)
    n = 64
    S = rng.uniform(50, 150, n)
    K = rng.uniform(50, 150, n)
    T = rng.uniform(0.05, 2.0, n)
    sigma = rng.uniform(0.1, 0.6, n)
    r = np.full(n, 0.03)
    q = np.full(n, 0.0)
    types = np.where(rng.uniform(size=n) > 0.5, "C", "P")

    vec = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=types)
    for i in range(n):
        scalar = black_scholes_price(
            S=float(S[i]),
            K=float(K[i]),
            T=float(T[i]),
            r=0.03,
            q=0.0,
            sigma=float(sigma[i]),
            option_type=str(types[i]),
        )
        assert math.isclose(vec[i], scalar, rel_tol=1e-12, abs_tol=1e-12)


def test_call_price_monotonic_in_vol() -> None:
    vols = np.linspace(0.05, 1.0, 20)
    prices = np.array(
        [
            black_scholes_price(
                S=100.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=float(v), option_type="C"
            )
            for v in vols
        ]
    )
    assert np.all(np.diff(prices) > 0.0)
