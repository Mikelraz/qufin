"""
Numba-jitted variance-filter recursions and Gaussian log-likelihoods for
the GARCH family.

All kernels accept primitives only (1-D float64 arrays and scalars) and
return ``(sigma2_path, log_likelihood)``.  They are sequential by nature
(each step depends on σ²_{t-1}), so they use plain ``@njit(cache=True)``
without ``parallel=True``.

Specifications
--------------
* GARCH(p, q) — Bollerslev (1986)

      σ²_t = ω + Σ_{i=1}^q α_i r²_{t-i} + Σ_{j=1}^p β_j σ²_{t-j}

* EGARCH(p, q) — Nelson (1991)

      log σ²_t = ω + Σ_{i=1}^q [α_i z_{t-i} + γ_i (|z_{t-i}| − E|z|)]
                 + Σ_{j=1}^p β_j log σ²_{t-j},
      z_t = r_t / σ_t,  E|z| = √(2/π) under standard normal innovations.

* GJR-GARCH(p, q) — Glosten-Jagannathan-Runkle (1993)

      σ²_t = ω + Σ_{i=1}^q [α_i r²_{t-i} + γ_i 1(r_{t-i} < 0) r²_{t-i}]
              + Σ_{j=1}^p β_j σ²_{t-j}

Each filter is warm-started with the sample variance.  The Gaussian
log-likelihood at step t is  −½(ln 2π + ln σ²_t + r²_t / σ²_t).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_LOG_TWO_PI = math.log(2.0 * math.pi)
_E_ABS_Z = math.sqrt(2.0 / math.pi)


@njit(cache=True)
def garch_filter(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma2_init: float,
) -> tuple[np.ndarray, float]:
    """Bollerslev GARCH(p, q) variance filter + Gaussian log-likelihood.

    Parameters
    ----------
    returns      Demeaned return series, shape (T,).
    omega        Intercept (must be > 0 for the filter to be valid).
    alpha        ARCH coefficients α_1, …, α_q, shape (q,).
    beta         GARCH coefficients β_1, …, β_p, shape (p,).
    sigma2_init  Initial variance for the pre-sample (typically sample variance).

    Returns
    -------
    sigma2 : shape (T,)    conditional variance path
    log_lik : float        Gaussian log-likelihood Σ_t −½(ln 2π + ln σ²_t + r²_t / σ²_t)
    """
    t_total = returns.shape[0]
    q = alpha.shape[0]
    p = beta.shape[0]
    sigma2 = np.empty(t_total)
    log_lik = 0.0
    for t in range(t_total):
        s = omega
        for i in range(q):
            if t - 1 - i >= 0:
                s += alpha[i] * returns[t - 1 - i] * returns[t - 1 - i]
            else:
                s += alpha[i] * sigma2_init
        for j in range(p):
            if t - 1 - j >= 0:
                s += beta[j] * sigma2[t - 1 - j]
            else:
                s += beta[j] * sigma2_init
        if s <= 0.0:
            s = 1e-12
        sigma2[t] = s
        log_lik += -0.5 * (_LOG_TWO_PI + math.log(s) + returns[t] * returns[t] / s)
    return sigma2, log_lik


@njit(cache=True)
def egarch_filter(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    sigma2_init: float,
) -> tuple[np.ndarray, float]:
    """Nelson EGARCH(p, q) filter + Gaussian log-likelihood.

    Updates log σ²_t directly, so positivity is guaranteed without
    constraints on ω, α, γ, β.

    Parameters
    ----------
    returns      Demeaned return series, shape (T,).
    omega        Intercept of the log-variance equation.
    alpha        Asymmetry coefficients α_i (on z_{t-i}), shape (q,).
    gamma        Magnitude coefficients γ_i (on |z_{t-i}| − E|z|), shape (q,).
    beta         Persistence coefficients β_j, shape (p,).
    sigma2_init  Initial variance (positive).
    """
    t_total = returns.shape[0]
    q = alpha.shape[0]
    p = beta.shape[0]
    log_init = math.log(sigma2_init)
    sigma2 = np.empty(t_total)
    log_lik = 0.0
    for t in range(t_total):
        log_s = omega
        for i in range(q):
            if t - 1 - i >= 0:
                prev_s = sigma2[t - 1 - i]
                z = returns[t - 1 - i] / math.sqrt(prev_s)
                log_s += alpha[i] * z + gamma[i] * (abs(z) - _E_ABS_Z)
            # else: pre-sample standardised innovation is 0; contributes nothing
        for j in range(p):
            if t - 1 - j >= 0:
                log_s += beta[j] * math.log(sigma2[t - 1 - j])
            else:
                log_s += beta[j] * log_init
        # Guard against absurd values that could blow up exp().
        if log_s > 50.0:
            log_s = 50.0
        if log_s < -50.0:
            log_s = -50.0
        s = math.exp(log_s)
        sigma2[t] = s
        log_lik += -0.5 * (_LOG_TWO_PI + log_s + returns[t] * returns[t] / s)
    return sigma2, log_lik


@njit(cache=True)
def gjr_filter(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    sigma2_init: float,
) -> tuple[np.ndarray, float]:
    """GJR-GARCH(p, q) filter + Gaussian log-likelihood.

    Adds an asymmetric ``γ_i`` term that activates only when r_{t-i} < 0
    (the canonical leverage effect for equity returns).
    """
    t_total = returns.shape[0]
    q = alpha.shape[0]
    p = beta.shape[0]
    sigma2 = np.empty(t_total)
    log_lik = 0.0
    for t in range(t_total):
        s = omega
        for i in range(q):
            if t - 1 - i >= 0:
                r_lag = returns[t - 1 - i]
                r_sq = r_lag * r_lag
                s += alpha[i] * r_sq
                if r_lag < 0.0:
                    s += gamma[i] * r_sq
            else:
                # Pre-sample: assume symmetric warm-start
                s += alpha[i] * sigma2_init + 0.5 * gamma[i] * sigma2_init
        for j in range(p):
            if t - 1 - j >= 0:
                s += beta[j] * sigma2[t - 1 - j]
            else:
                s += beta[j] * sigma2_init
        if s <= 0.0:
            s = 1e-12
        sigma2[t] = s
        log_lik += -0.5 * (_LOG_TWO_PI + math.log(s) + returns[t] * returns[t] / s)
    return sigma2, log_lik


@njit(cache=True)
def ewma_filter(returns: np.ndarray, lam: float, sigma2_init: float) -> np.ndarray:
    """RiskMetrics EWMA variance recursion.

        σ²_t = λ σ²_{t-1} + (1 − λ) r²_{t-1},  σ²_0 = sigma2_init.

    No likelihood is returned because EWMA has no free parameters to estimate
    by MLE (λ is fixed by the user).
    """
    t_total = returns.shape[0]
    sigma2 = np.empty(t_total)
    prev = sigma2_init
    for t in range(t_total):
        if t == 0:
            sigma2[t] = prev
        else:
            sigma2[t] = lam * prev + (1.0 - lam) * returns[t - 1] * returns[t - 1]
        prev = sigma2[t]
    return sigma2
