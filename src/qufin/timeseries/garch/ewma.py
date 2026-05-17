"""
RiskMetrics-style exponentially-weighted moving-average variance estimator.

    σ²_t = λ σ²_{t-1} + (1 − λ) r²_{t-1}

with the conventional λ ∈ (0, 1) decay parameter (RiskMetrics 1996 default
λ = 0.94 for daily, λ = 0.97 for monthly).  No likelihood maximisation —
λ is fixed by the user.

This is the closed-form integrated GARCH(1, 1) limit when ω = 0, α = 1−λ,
β = λ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .._io import to_numpy_1d, validate_finite, validate_min_length
from ._likelihood import ewma_filter


@dataclass(slots=True)
class EWMAResult:
    """RiskMetrics EWMA result.

    Attributes
    ----------
    lam            Decay parameter λ.
    mu             Sample mean of the input (subtracted before filtering).
    sigma2         Conditional variance path, shape (T,).
    residuals      r_t − μ.
    std_residuals  ε_t / σ_t.
    n_obs          T.
    """

    lam: float
    mu: float
    sigma2: np.ndarray
    residuals: np.ndarray
    std_residuals: np.ndarray
    n_obs: int

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "t": np.arange(self.n_obs, dtype=np.int64),
                "sigma2": self.sigma2,
                "std_residual": self.std_residuals,
            }
        )


class EWMA:
    """RiskMetrics exponentially-weighted moving-average variance estimator.

    Parameters
    ----------
    lam : float
        Decay parameter λ ∈ (0, 1).  Default 0.94 (RiskMetrics daily).
    """

    def __init__(self, lam: float = 0.94) -> None:
        if not (0.0 < lam < 1.0):
            raise ValueError(f"lam must be in (0, 1); got {lam}.")
        self.lam = float(lam)
        self._result: EWMAResult | None = None

    @property
    def result(self) -> EWMAResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    def fit(self, returns: np.ndarray, *, demean: bool = True) -> EWMAResult:
        """Run the EWMA recursion on ``returns``.

        Parameters
        ----------
        returns : array_like, shape (T,)
        demean  : bool   If True, subtract the sample mean before filtering.
        """
        arr = to_numpy_1d(returns)
        validate_finite(arr)
        validate_min_length(arr, 2, "returns")
        mu = float(arr.mean()) if demean else 0.0
        eps = arr - mu
        sample_var = float(np.var(eps))
        if sample_var <= 0.0:
            raise ValueError("Sample variance is zero — cannot run EWMA.")
        sigma2 = ewma_filter(eps, self.lam, sample_var)
        std_resid = eps / np.sqrt(sigma2)
        self._result = EWMAResult(
            lam=self.lam,
            mu=mu,
            sigma2=sigma2,
            residuals=eps,
            std_residuals=std_resid,
            n_obs=arr.shape[0],
        )
        return self._result

    def forecast(self, h: int) -> np.ndarray:
        """h-step variance forecasts.  Constant: σ²_{T+1} for all horizons."""
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")
        res = self.result
        # Next-step variance via the recursion using the last residual.
        next_var = self.lam * res.sigma2[-1] + (1.0 - self.lam) * res.residuals[-1] ** 2
        return np.full(h, next_var)
