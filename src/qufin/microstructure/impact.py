"""
Price-impact and illiquidity measures.

* ``kyle_lambda``        — Kyle's (1985) λ: the slope of price changes on signed
  order flow.  Larger λ ⇒ a given net order moves the price more ⇒ less depth.
* ``hasbrouck_lambda``   — Hasbrouck's (2009) λ: the same idea against the
  *square root* of signed volume, motivated by concave price impact.
* ``amihud_illiquidity`` — Amihud's (2002) ILLIQ: the average ratio of absolute
  return to dollar volume, a low-frequency proxy requiring only daily data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import PriceImpactResult, check_lengths, to_numpy_1d


def _ols_impact(x: np.ndarray, y: np.ndarray) -> PriceImpactResult:
    """OLS of ``y`` on ``x`` with intercept, packaged as a PriceImpactResult."""
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 observations for the impact regression.")
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0.0:
        raise ValueError("regressor has zero variance.")
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    resid = y - (intercept + slope * x)
    rss = float(resid @ resid)
    tss = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
    sigma2 = rss / (n - 2)
    se = (sigma2 / sxx) ** 0.5 if sigma2 > 0.0 else float("inf")
    t_stat = slope / se if se > 0.0 else float("nan")
    return PriceImpactResult(
        lam=slope,
        r_squared=r2,
        t_stat=t_stat,
        n_obs=n,
        intercept=intercept,
    )


def kyle_lambda(price_changes: Any, signed_volume: Any) -> PriceImpactResult:
    """
    Kyle's (1985) λ from the regression ``Δp_t = λ · (q_t V_t) + ε_t``.

    Parameters
    ----------
    price_changes  Contemporaneous price changes ``Δp_t`` (or returns), shape ``(n,)``.
    signed_volume  Net signed order flow ``q_t V_t`` aligned to ``price_changes``.
    """
    dp = to_numpy_1d(price_changes)
    sv = to_numpy_1d(signed_volume)
    check_lengths(dp, sv)
    return _ols_impact(sv, dp)


def hasbrouck_lambda(price_changes: Any, signs: Any, volume: Any) -> PriceImpactResult:
    """
    Hasbrouck's (2009) λ from ``Δp_t = λ · q_t √V_t + ε_t`` (concave impact).

    Parameters
    ----------
    price_changes  Price changes ``Δp_t``, shape ``(n,)``.
    signs          Trade signs ``q_t`` in {−1, 0, +1}.
    volume         Trade / bar volume ``V_t`` (non-negative).
    """
    dp = to_numpy_1d(price_changes)
    q = to_numpy_1d(signs)
    v = to_numpy_1d(volume)
    check_lengths(dp, q, v)
    if np.any(v < 0.0):
        raise ValueError("volume must be non-negative.")
    return _ols_impact(q * np.sqrt(v), dp)


def amihud_illiquidity(returns: Any, dollar_volume: Any, *, scale: float = 1e6) -> float:
    """
    Amihud (2002) ILLIQ: ``scale · mean(|r_t| / DollarVolume_t)``.

    Periods with non-positive dollar volume are dropped.  The conventional
    ``scale`` of ``1e6`` makes daily equity values comparable across the
    literature.
    """
    r = to_numpy_1d(returns)
    dv = to_numpy_1d(dollar_volume)
    check_lengths(r, dv)
    mask = dv > 0.0
    if not np.any(mask):
        raise ValueError("dollar_volume has no positive entries.")
    ratio = np.abs(r[mask]) / dv[mask]
    return float(scale * np.mean(ratio))
