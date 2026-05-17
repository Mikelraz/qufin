"""
GJR-GARCH (Glosten-Jagannathan-Runkle 1993).

    σ²_t = ω + Σ_{i=1}^q [α_i ε²_{t-i} + γ_i 𝟙(ε_{t-i} < 0) ε²_{t-i}]
            + Σ_{j=1}^p β_j σ²_{t-j}

This captures the leverage effect — negative returns push variance higher
than positive returns of the same magnitude.  Stationarity requires

    Σ α_i + ½ Σ γ_i + Σ β_j  <  1

(assuming symmetric innovations, so the indicator activates with
probability ½).  Same softplus / persistence-rescaling reparameterisation
as ``GARCH``; γ is also constrained to be positive (the leverage
component).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.optimize

from .._io import to_numpy_1d, validate_finite, validate_min_length
from ..utils import info_criteria
from ._likelihood import gjr_filter


@dataclass(slots=True)
class GJRFitResult:
    """Fitted GJR-GARCH(p, q) model output."""

    p: int
    q: int
    mu: float
    omega: float
    alpha: np.ndarray
    gamma: np.ndarray  # leverage coefficients, shape (q,)
    beta: np.ndarray
    sigma2: np.ndarray
    residuals: np.ndarray
    std_residuals: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    persistence: float  # Σα + ½Σγ + Σβ
    unconditional_var: float

    def __str__(self) -> str:
        return (
            f"GJR-GARCH({self.p},{self.q})  n_obs={self.n_obs}\n"
            f"  μ={self.mu:.6g}  ω={self.omega:.6g}\n"
            f"  α={self.alpha}  γ={self.gamma}  β={self.beta}\n"
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}\n"
            f"  persistence={self.persistence:.4f}  σ²_∞={self.unconditional_var:.6g}"
        )

    def to_dataframe(self) -> pl.DataFrame:
        names = (
            ["mu", "omega"]
            + [f"alpha_{i}" for i in range(1, self.q + 1)]
            + [f"gamma_{i}" for i in range(1, self.q + 1)]
            + [f"beta_{j}" for j in range(1, self.p + 1)]
        )
        values = (
            [self.mu, self.omega] + self.alpha.tolist() + self.gamma.tolist() + self.beta.tolist()
        )
        return pl.DataFrame({"parameter": names, "value": values})


def _softplus(x: float) -> float:
    if x > 30.0:
        return x
    return math.log1p(math.exp(x))


def _inv_softplus(y: float) -> float:
    if y <= 0.0:
        return -10.0
    if y > 30.0:
        return y
    return math.log(math.expm1(y))


def _unpack(params: np.ndarray, p: int, q: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack into (ω, α, γ, β) with persistence < 1 projection.

    Persistence is computed as π_raw = Σα + ½Σγ + Σβ.  All coefficients are
    rescaled by π_eff / π_raw where π_eff = π_raw / (1 + π_raw) ∈ (0, 1).
    """
    omega = math.exp(params[0])
    alpha = np.array([_softplus(float(params[1 + i])) for i in range(q)])
    gamma = np.array([_softplus(float(params[1 + q + i])) for i in range(q)])
    beta = np.array([_softplus(float(params[1 + 2 * q + j])) for j in range(p)])
    persistence_raw = float(alpha.sum() + 0.5 * gamma.sum() + beta.sum())
    if persistence_raw > 0.0:
        persistence_eff = persistence_raw / (1.0 + persistence_raw)
        scale = persistence_eff / persistence_raw
        alpha = alpha * scale
        gamma = gamma * scale
        beta = beta * scale
    return omega, alpha, gamma, beta


def _pack(omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            [math.log(max(omega, 1e-10))],
            [_inv_softplus(float(a)) for a in alpha],
            [_inv_softplus(float(g)) for g in gamma],
            [_inv_softplus(float(b)) for b in beta],
        ]
    )


class GJR:
    """GJR-GARCH(p, q) model with positive leverage coefficients."""

    def __init__(self, p: int = 1, q: int = 1, *, mean: str = "constant") -> None:
        if p < 0 or q < 0:
            raise ValueError(f"p and q must be ≥ 0; got p={p}, q={q}.")
        if p + q == 0:
            raise ValueError("At least one of p or q must be > 0.")
        if mean not in ("constant", "zero"):
            raise ValueError(f"mean must be 'constant' or 'zero'; got {mean!r}.")
        self.p = p
        self.q = q
        self.mean = mean
        self._result: GJRFitResult | None = None

    @property
    def result(self) -> GJRFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    def fit(self, returns: np.ndarray) -> GJRFitResult:
        arr = to_numpy_1d(returns)
        validate_finite(arr)
        validate_min_length(arr, max(self.p, self.q) + 10, "returns")

        mu = float(arr.mean()) if self.mean == "constant" else 0.0
        eps = arr - mu
        sample_var = float(np.var(eps))
        if sample_var <= 0.0:
            raise ValueError("Sample variance is zero — cannot fit GJR-GARCH.")

        omega0 = 0.05 * sample_var
        alpha0 = np.full(self.q, 0.03)
        gamma0 = np.full(self.q, 0.05)
        beta0 = np.full(self.p, 0.85 / max(self.p, 1)) if self.p > 0 else np.zeros(0)
        x0 = _pack(omega0, alpha0, gamma0, beta0)

        def neg_ll(params: np.ndarray) -> float:
            omega, alpha, gamma, beta = _unpack(params, self.p, self.q)
            try:
                _, ll = gjr_filter(eps, omega, alpha, gamma, beta, sample_var)
            except Exception:
                return 1e10
            if not math.isfinite(ll):
                return 1e10
            return -ll

        res = scipy.optimize.minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-10, "gtol": 1e-7},
        )
        omega, alpha, gamma, beta = _unpack(res.x, self.p, self.q)
        sigma2, log_lik = gjr_filter(eps, omega, alpha, gamma, beta, sample_var)
        sigma = np.sqrt(sigma2)
        std_resid = eps / sigma

        n_eff = arr.shape[0]
        n_params = 1 + 2 * self.q + self.p + (1 if self.mean == "constant" else 0)
        aic_v, bic_v, hqic_v = info_criteria(log_lik, n_eff, n_params)

        persistence = float(alpha.sum() + 0.5 * gamma.sum() + beta.sum())
        if persistence >= 1.0 - 1e-8:
            warnings.warn(
                f"Fitted GJR({self.p},{self.q}) has persistence ≥ 1; forecasts may diverge.",
                RuntimeWarning,
                stacklevel=2,
            )
        unc_var = omega / max(1.0 - persistence, 1e-10)

        self._result = GJRFitResult(
            p=self.p,
            q=self.q,
            mu=mu,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma2=sigma2,
            residuals=eps,
            std_residuals=std_resid,
            log_lik=float(log_lik),
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=n_eff,
            persistence=persistence,
            unconditional_var=float(unc_var),
        )
        return self._result

    # ------------------------------------------------------------------
    # Simulation + Monte-Carlo forecast
    # ------------------------------------------------------------------

    def simulate(self, t_total: int, *, seed: int | None = None, burnin: int = 500) -> np.ndarray:
        if t_total < 1:
            raise ValueError(f"t_total must be ≥ 1, got {t_total}.")
        res = self.result
        rng = np.random.default_rng(seed)
        n = t_total + burnin
        eps = np.zeros(n)
        sigma2 = np.full(n, res.unconditional_var)
        z = rng.standard_normal(n)
        for t in range(1, n):
            v = res.omega
            for i in range(self.q):
                if t - 1 - i >= 0:
                    r_lag = eps[t - 1 - i]
                    v += res.alpha[i] * r_lag * r_lag
                    if r_lag < 0.0:
                        v += res.gamma[i] * r_lag * r_lag
            for j in range(self.p):
                if t - 1 - j >= 0:
                    v += res.beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(v, 1e-12)
            eps[t] = math.sqrt(sigma2[t]) * z[t]
        return res.mu + eps[burnin:]

    def forecast(
        self,
        h: int,
        *,
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> np.ndarray:
        """Monte-Carlo variance forecasts (closed form unavailable due to indicator)."""
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")
        if n_paths < 1:
            raise ValueError(f"n_paths must be ≥ 1, got {n_paths}.")
        res = self.result
        rng = np.random.default_rng(seed)
        eps_hist = res.residuals
        sigma2_hist = res.sigma2
        paths = np.empty((n_paths, h))
        for path_idx in range(n_paths):
            sim_eps = np.concatenate([eps_hist, np.zeros(h)])
            sim_sig = np.concatenate([sigma2_hist, np.zeros(h)])
            t0 = len(eps_hist)
            for s in range(h):
                v = res.omega
                for i in range(self.q):
                    r_lag = sim_eps[t0 + s - 1 - i]
                    v += res.alpha[i] * r_lag * r_lag
                    if r_lag < 0.0:
                        v += res.gamma[i] * r_lag * r_lag
                for j in range(self.p):
                    v += res.beta[j] * sim_sig[t0 + s - 1 - j]
                sim_sig[t0 + s] = max(v, 1e-12)
                z = rng.standard_normal()
                sim_eps[t0 + s] = math.sqrt(sim_sig[t0 + s]) * z
                paths[path_idx, s] = sim_sig[t0 + s]
        return paths.mean(axis=0)
