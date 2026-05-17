"""
Numba-jitted kernels for Black-Scholes pricing, greeks and implied vol.

Convention
----------
* All kernels take primitive types only: ``np.ndarray`` of ``float64`` /
  ``bool_``, scalars, and integers.  Polars and dataclass types stay out.
* The pricing formula assumes continuous compounding for ``r`` (risk-free)
  and ``q`` (dividend / borrow yield).
* ``is_call`` is a uint8 array (1 for call, 0 for put) so it can be
  cheaply broadcast into the kernels.
* All vectorised kernels release the GIL and run with ``parallel=True``;
  iterations across strikes/expiries are independent.

Greek conventions follow Hull (10e):

    delta_call = e^{-qT} N(d1)
    delta_put  = -e^{-qT} N(-d1)
    gamma      = e^{-qT} φ(d1) / (S σ √T)
    vega       = S e^{-qT} φ(d1) √T              (per 1.0 vol, not per 1%)
    theta      = -S e^{-qT} φ(d1) σ / (2√T)
                 ∓ r K e^{-rT} N(±d2) ± q S e^{-qT} N(±d1)  (per year)
    rho_call   =  K T e^{-rT} N(d2)
    rho_put    = -K T e^{-rT} N(-d2)
    vanna      = -e^{-qT} φ(d1) d2 / σ           (∂Δ/∂σ)
    charm_call = -e^{-qT} φ(d1)·(2(r-q)T - d2 σ √T)/(2 T σ √T)
                 - q e^{-qT} N(d1)               (∂Δ/∂T  ⇒ negated for ∂Δ/∂t)
    vomma      = vega · d1 d2 / σ
    speed      = -Γ (d1/(σ √T) + 1) / S          (∂Γ/∂S)
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_SQRT_2: float = math.sqrt(2.0)
_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


@njit(cache=True, inline="always")
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


@njit(cache=True, inline="always")
def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True, inline="always")
def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


@njit(cache=True, inline="always")
def _bs_price_scalar(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: int
) -> float:
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = (S - K) if is_call == 1 else (K - S)
        return intrinsic if intrinsic > 0.0 else 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if is_call == 1:
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


@njit(cache=True, parallel=True, nogil=True)
def bs_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    n = S.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = _bs_price_scalar(S[i], K[i], T[i], r[i], q[i], sigma[i], is_call[i])
    return out


@njit(cache=True, parallel=True, nogil=True)
def bs_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = S.shape[0]
    delta = np.zeros(n, dtype=np.float64)
    gamma = np.zeros(n, dtype=np.float64)
    vega = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    rho = np.zeros(n, dtype=np.float64)
    vanna = np.zeros(n, dtype=np.float64)
    charm = np.zeros(n, dtype=np.float64)
    vomma = np.zeros(n, dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        Si = S[i]
        Ki = K[i]
        Ti = T[i]
        ri = r[i]
        qi = q[i]
        si = sigma[i]
        call = is_call[i] == 1

        if Ti <= 0.0 or si <= 0.0 or Si <= 0.0 or Ki <= 0.0:
            continue

        sqrtT = math.sqrt(Ti)
        d1 = (math.log(Si / Ki) + (ri - qi + 0.5 * si * si) * Ti) / (si * sqrtT)
        d2 = d1 - si * sqrtT
        disc_q = math.exp(-qi * Ti)
        disc_r = math.exp(-ri * Ti)
        pdf_d1 = _norm_pdf(d1)
        N_d1 = _norm_cdf(d1)
        N_md1 = 1.0 - N_d1
        N_d2 = _norm_cdf(d2)
        N_md2 = 1.0 - N_d2

        g = disc_q * pdf_d1 / (Si * si * sqrtT)
        v = Si * disc_q * pdf_d1 * sqrtT
        gamma[i] = g
        vega[i] = v
        vomma[i] = v * d1 * d2 / si
        vanna[i] = -disc_q * pdf_d1 * d2 / si
        speed[i] = -g * (d1 / (si * sqrtT) + 1.0) / Si

        common_charm = (
            -disc_q * pdf_d1 * (2.0 * (ri - qi) * Ti - d2 * si * sqrtT) / (2.0 * Ti * si * sqrtT)
        )

        if call:
            delta[i] = disc_q * N_d1
            theta[i] = (
                -Si * disc_q * pdf_d1 * si / (2.0 * sqrtT)
                - ri * Ki * disc_r * N_d2
                + qi * Si * disc_q * N_d1
            )
            rho[i] = Ki * Ti * disc_r * N_d2
            charm[i] = common_charm - qi * disc_q * N_d1
        else:
            delta[i] = -disc_q * N_md1
            theta[i] = (
                -Si * disc_q * pdf_d1 * si / (2.0 * sqrtT)
                + ri * Ki * disc_r * N_md2
                - qi * Si * disc_q * N_md1
            )
            rho[i] = -Ki * Ti * disc_r * N_md2
            charm[i] = common_charm + qi * disc_q * N_md1

    return delta, gamma, vega, theta, rho, vanna, charm, vomma, speed


@njit(cache=True, inline="always")
def _vega_scalar(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


@njit(cache=True, inline="always")
def _iv_scalar(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: int,
    tol: float,
    max_iter: int,
) -> float:
    """
    Implied vol via Newton-Raphson with bisection fallback.

    Returns ``np.nan`` if the target price is outside the no-arbitrage bounds or
    if convergence cannot be achieved.
    """
    if T <= 0.0 or price <= 0.0:
        return np.nan

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if is_call == 1:
        lo_bound = max(0.0, S * disc_q - K * disc_r)
        hi_bound = S * disc_q
    else:
        lo_bound = max(0.0, K * disc_r - S * disc_q)
        hi_bound = K * disc_r

    if price <= lo_bound + 1e-12 or price >= hi_bound - 1e-12:
        return np.nan

    # Manaster-Koehler seed.
    seed_arg = abs(math.log(S / K) + (r - q) * T)
    sigma = math.sqrt(2.0 * seed_arg / T) if T > 0.0 else 0.2
    if sigma < 1e-4:
        sigma = 0.2

    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        if sigma <= lo or sigma >= hi:
            sigma = 0.5 * (lo + hi)
        p = _bs_price_scalar(S, K, T, r, q, sigma, is_call)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        v = _vega_scalar(S, K, T, r, q, sigma)
        if v < 1e-10:
            sigma = 0.5 * (lo + hi)
            continue
        step = diff / v
        next_sigma = sigma - step
        if next_sigma <= lo or next_sigma >= hi:
            next_sigma = 0.5 * (lo + hi)
        sigma = next_sigma
    return sigma if (hi - lo) < 1e-3 else np.nan


@njit(cache=True, parallel=True, nogil=True)
def implied_vol(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = _iv_scalar(price[i], S[i], K[i], T[i], r[i], q[i], is_call[i], tol, max_iter)
    return out


@njit(cache=True, parallel=True, nogil=True)
def gamma_at_spot(
    S_grid: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Gamma evaluated for every (spot in S_grid, contract i) pair.

    Returns
    -------
    np.ndarray, shape (n_spot, n_contracts)
    """
    n_s = S_grid.shape[0]
    n_c = K.shape[0]
    out = np.zeros((n_s, n_c), dtype=np.float64)
    for s in prange(n_s):
        S = S_grid[s]
        for i in range(n_c):
            Ti = T[i]
            si = sigma[i]
            Ki = K[i]
            qi = q[i]
            if Ti <= 0.0 or si <= 0.0 or S <= 0.0 or Ki <= 0.0:
                continue
            sqrtT = math.sqrt(Ti)
            d1 = (math.log(S / Ki) + (r[i] - qi + 0.5 * si * si) * Ti) / (si * sqrtT)
            out[s, i] = math.exp(-qi * Ti) * _norm_pdf(d1) / (S * si * sqrtT)
    return out


@njit(cache=True, parallel=True, nogil=True)
def greeks_at_spot(
    S_grid: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate (delta, gamma, vanna, charm) on a spot grid for every contract.

    Returns four ``(n_spot, n_contracts)`` arrays.  Used by the GEX profile to
    sweep dealer exposure across hypothetical spot levels without re-running
    the full greeks kernel each time.
    """
    n_s = S_grid.shape[0]
    n_c = K.shape[0]
    delta = np.zeros((n_s, n_c), dtype=np.float64)
    gamma = np.zeros((n_s, n_c), dtype=np.float64)
    vanna = np.zeros((n_s, n_c), dtype=np.float64)
    charm = np.zeros((n_s, n_c), dtype=np.float64)
    for s in prange(n_s):
        S = S_grid[s]
        for i in range(n_c):
            Ti = T[i]
            si = sigma[i]
            Ki = K[i]
            ri = r[i]
            qi = q[i]
            if Ti <= 0.0 or si <= 0.0 or S <= 0.0 or Ki <= 0.0:
                continue
            sqrtT = math.sqrt(Ti)
            d1 = (math.log(S / Ki) + (ri - qi + 0.5 * si * si) * Ti) / (si * sqrtT)
            d2 = d1 - si * sqrtT
            disc_q = math.exp(-qi * Ti)
            pdf_d1 = _norm_pdf(d1)
            g = disc_q * pdf_d1 / (S * si * sqrtT)
            gamma[s, i] = g
            vanna[s, i] = -disc_q * pdf_d1 * d2 / si
            common_charm = (
                -disc_q
                * pdf_d1
                * (2.0 * (ri - qi) * Ti - d2 * si * sqrtT)
                / (2.0 * Ti * si * sqrtT)
            )
            if is_call[i] == 1:
                delta[s, i] = disc_q * _norm_cdf(d1)
                charm[s, i] = common_charm - qi * disc_q * _norm_cdf(d1)
            else:
                N_md1 = 1.0 - _norm_cdf(d1)
                delta[s, i] = -disc_q * N_md1
                charm[s, i] = common_charm + qi * disc_q * N_md1
    return delta, gamma, vanna, charm
