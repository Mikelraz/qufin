"""
PIN — Probability of Informed Trading (Easley, Kiefer, O'Hara & Paperman 1996).

A structural model of the daily order arrival process.  On each day an
information event occurs with probability ``α``; given an event it is bad news
with probability ``δ``.  Uninformed buys and sells arrive as independent Poisson
flows with rates ``εb`` and ``εs``; on an event day informed traders add a
one-sided Poisson flow of rate ``μ`` (buys on good news, sells on bad news).

The probability that a random trade is information-based is

    PIN = α μ / (α μ + εb + εs).

Parameters are estimated by maximum likelihood over a panel of daily
(buy, sell) trade counts.  The likelihood is evaluated with the Easley-Hvidkjaer-
O'Hara factorization + log-sum-exp so it is stable for the large counts seen in
liquid names (the naive Poisson-mixture form overflows).

Unlike VPIN (a model-free, volume-clock toxicity proxy) PIN is a fully
parametric daily model; the two are complementary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize
from scipy.special import gammaln, logsumexp

from ._types import check_lengths, to_numpy_1d


@dataclass(slots=True, frozen=True)
class PINResult:
    """Fitted PIN model.

    Attributes
    ----------
    pin       Probability of informed trading ``α μ / (α μ + εb + εs)``.
    alpha     Probability of an information event on a given day.
    delta     Probability an event is bad news (conditional on an event).
    mu        Informed-trader arrival rate on event days.
    eps_b     Uninformed buy arrival rate.
    eps_s     Uninformed sell arrival rate.
    log_lik   Maximised log-likelihood.
    n_obs     Number of days used.
    converged Whether the optimiser reported success.
    """

    pin: float
    alpha: float
    delta: float
    mu: float
    eps_b: float
    eps_s: float
    log_lik: float
    n_obs: int
    converged: bool

    def __str__(self) -> str:
        return (
            f"PIN={self.pin:.4f} (α={self.alpha:.3f}, δ={self.delta:.3f}, "
            f"μ={self.mu:.2f}, εb={self.eps_b:.2f}, εs={self.eps_s:.2f})"
        )


def _neg_log_likelihood(params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
    """Negative EHO-factorized PIN log-likelihood (stable via log-sum-exp)."""
    a, d, mu, eb, es = params
    if not (0.0 < a < 1.0 and 0.0 < d < 1.0) or mu <= 0.0 or eb <= 0.0 or es <= 0.0:
        return 1e12
    # Branch log-weights (no-event, bad-news, good-news), each over all days.
    t_none = np.full(buys.shape[0], np.log(1.0 - a))
    t_bad = np.log(a * d) - mu + sells * np.log1p(mu / es)
    t_good = np.log(a * (1.0 - d)) - mu + buys * np.log1p(mu / eb)
    bracket = logsumexp(np.vstack([t_none, t_bad, t_good]), axis=0)
    const = (
        -eb
        + buys * np.log(eb)
        - gammaln(buys + 1.0)
        - es
        + sells * np.log(es)
        - gammaln(sells + 1.0)
    )
    ll = float(np.sum(const + bracket))
    return -ll if np.isfinite(ll) else 1e12


def pin(buys: Any, sells: Any) -> PINResult:
    """
    Estimate the PIN model by maximum likelihood from daily buy / sell counts.

    Parameters
    ----------
    buys, sells   Per-day buyer- and seller-initiated trade counts, shape ``(n,)``
                  (non-negative; e.g. from classifying trades with
                  :func:`lee_ready` and summing signs per day).

    Returns
    -------
    PINResult

    Notes
    -----
    The likelihood is multimodal, so the fit is started from a small grid of
    ``(α, δ)`` values and the best optimum is kept.
    """
    b = to_numpy_1d(buys)
    s = to_numpy_1d(sells)
    check_lengths(b, s)
    if b.shape[0] < 5:
        raise ValueError("need at least 5 days of counts.")
    if np.any(b < 0.0) or np.any(s < 0.0):
        raise ValueError("buy/sell counts must be non-negative.")

    mean_b = float(b.mean())
    mean_s = float(s.mean())
    mu0 = max(1.0, abs(mean_b - mean_s))
    bounds = [(1e-4, 1.0 - 1e-4), (1e-4, 1.0 - 1e-4), (1e-6, None), (1e-6, None), (1e-6, None)]

    best: optimize.OptimizeResult | None = None
    for a0 in (0.1, 0.3, 0.5, 0.7):
        for d0 in (0.3, 0.5, 0.7):
            x0 = np.array([a0, d0, mu0, max(mean_b, 1e-3), max(mean_s, 1e-3)])
            res = optimize.minimize(
                _neg_log_likelihood, x0, args=(b, s), method="L-BFGS-B", bounds=bounds
            )
            if best is None or res.fun < best.fun:
                best = res
    assert best is not None

    a, d, mu, eb, es = (float(v) for v in best.x)
    informed = a * mu
    pin_val = informed / (informed + eb + es) if (informed + eb + es) > 0.0 else 0.0
    return PINResult(
        pin=pin_val,
        alpha=a,
        delta=d,
        mu=mu,
        eps_b=eb,
        eps_s=es,
        log_lik=float(-best.fun),
        n_obs=b.shape[0],
        converged=bool(best.success),
    )
