"""Black-Scholes greeks: finite-difference cross-checks and put-call relations."""

from __future__ import annotations

import math

import numpy as np

from qufin.options import black_scholes_greeks, black_scholes_price


def _fd_delta(S, K, T, r, q, sigma, kind, h=1e-4):
    up = black_scholes_price(S=S + h, K=K, T=T, r=r, q=q, sigma=sigma, option_type=kind)
    dn = black_scholes_price(S=S - h, K=K, T=T, r=r, q=q, sigma=sigma, option_type=kind)
    return (up - dn) / (2 * h)


def _fd_gamma(S, K, T, r, q, sigma, kind, h=1e-4):
    up = black_scholes_price(S=S + h, K=K, T=T, r=r, q=q, sigma=sigma, option_type=kind)
    md = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=kind)
    dn = black_scholes_price(S=S - h, K=K, T=T, r=r, q=q, sigma=sigma, option_type=kind)
    return (up - 2 * md + dn) / (h * h)


def _fd_vega(S, K, T, r, q, sigma, kind, h=1e-5):
    up = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma + h, option_type=kind)
    dn = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma - h, option_type=kind)
    return (up - dn) / (2 * h)


def test_greeks_match_finite_differences_for_call() -> None:
    S, K, T, r, q, sigma = 100.0, 105.0, 0.75, 0.04, 0.01, 0.30
    g = black_scholes_greeks(
        S=np.array([S]),
        K=np.array([K]),
        T=np.array([T]),
        r=np.array([r]),
        q=np.array([q]),
        sigma=np.array([sigma]),
        option_type="C",
    )
    assert math.isclose(g.delta[0], _fd_delta(S, K, T, r, q, sigma, "C"), abs_tol=1e-6)
    assert math.isclose(g.gamma[0], _fd_gamma(S, K, T, r, q, sigma, "C"), abs_tol=1e-4)
    assert math.isclose(g.vega[0], _fd_vega(S, K, T, r, q, sigma, "C"), abs_tol=1e-6)


def test_put_call_delta_relation() -> None:
    # Δ_call - Δ_put = e^{-qT}
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.02, 0.25
    gc = black_scholes_greeks(
        S=np.array([S]),
        K=np.array([K]),
        T=np.array([T]),
        r=np.array([r]),
        q=np.array([q]),
        sigma=np.array([sigma]),
        option_type="C",
    )
    gp = black_scholes_greeks(
        S=np.array([S]),
        K=np.array([K]),
        T=np.array([T]),
        r=np.array([r]),
        q=np.array([q]),
        sigma=np.array([sigma]),
        option_type="P",
    )
    assert math.isclose(gc.delta[0] - gp.delta[0], math.exp(-q * T), abs_tol=1e-12)


def test_put_call_gamma_and_vega_equal() -> None:
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.05, 0.01, 0.40
    gc = black_scholes_greeks(
        S=np.array([S]),
        K=np.array([K]),
        T=np.array([T]),
        r=np.array([r]),
        q=np.array([q]),
        sigma=np.array([sigma]),
        option_type="C",
    )
    gp = black_scholes_greeks(
        S=np.array([S]),
        K=np.array([K]),
        T=np.array([T]),
        r=np.array([r]),
        q=np.array([q]),
        sigma=np.array([sigma]),
        option_type="P",
    )
    assert math.isclose(gc.gamma[0], gp.gamma[0], rel_tol=1e-12)
    assert math.isclose(gc.vega[0], gp.vega[0], rel_tol=1e-12)


def test_atm_gamma_peaks_near_strike() -> None:
    strikes = np.linspace(80, 120, 41)
    n = strikes.shape[0]
    g = black_scholes_greeks(
        S=np.full(n, 100.0),
        K=strikes,
        T=np.full(n, 0.25),
        r=np.full(n, 0.0),
        q=np.full(n, 0.0),
        sigma=np.full(n, 0.20),
        option_type="C",
    )
    peak = strikes[int(np.argmax(g.gamma))]
    assert abs(peak - 100.0) <= 2.0
