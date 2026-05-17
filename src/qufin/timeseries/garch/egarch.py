"""
Nelson (1991) Exponential GARCH(p, q).

    log σ²_t = ω + Σ_{i=1}^q [α_i z_{t-i} + γ_i (|z_{t-i}| − E|z|)]
              + Σ_{j=1}^p β_j log σ²_{t-j}

    z_t = ε_t / σ_t,    E|z| = √(2/π) for standard-normal innovations.

Because the equation is in log σ², positivity of σ² is automatic and the
optimiser can run unconstrained over (ω, α, γ, β).  For stationarity of
log σ² we require Σβ_j < 1 in modulus.  We do not enforce it via the
optimiser; instead we warn after the fit (mirrors how ``OrnsteinUhlenbeck``
treats explosive AR(1)).
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
from ._likelihood import egarch_filter


@dataclass(slots=True)
class EGARCHFitResult:
    """Fitted EGARCH(p, q) model output."""

    p: int
    q: int
    mu: float
    omega: float
    alpha: np.ndarray  # asymmetry, shape (q,)
    gamma: np.ndarray  # magnitude, shape (q,)
    beta: np.ndarray  # persistence, shape (p,)
    sigma2: np.ndarray
    residuals: np.ndarray
    std_residuals: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    persistence: float  # Σβ (log-variance autoregressive sum)

    def __str__(self) -> str:
        return (
            f"EGARCH({self.p},{self.q})  n_obs={self.n_obs}\n"
            f"  μ={self.mu:.6g}  ω={self.omega:.6g}\n"
            f"  α={self.alpha}  γ={self.gamma}  β={self.beta}\n"
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}\n"
            f"  persistence (Σβ)={self.persistence:.4f}"
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


class EGARCH:
    """Nelson EGARCH(p, q) model.

    Parameters
    ----------
    p : int   Persistence order (log σ² lags).
    q : int   News-impact order.
    mean : {"constant", "zero"}
    """

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
        self._result: EGARCHFitResult | None = None

    @property
    def result(self) -> EGARCHFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    def fit(self, returns: np.ndarray) -> EGARCHFitResult:
        arr = to_numpy_1d(returns)
        validate_finite(arr)
        validate_min_length(arr, max(self.p, self.q) + 10, "returns")

        mu = float(arr.mean()) if self.mean == "constant" else 0.0
        eps = arr - mu
        sample_var = float(np.var(eps))
        if sample_var <= 0.0:
            raise ValueError("Sample variance is zero — cannot fit EGARCH.")

        # Warm-start: ω ≈ (1 − Σβ) log σ²_∞, β tightened to 0.9, no asymmetry yet.
        beta0 = np.full(self.p, 0.9 / max(self.p, 1)) if self.p > 0 else np.zeros(0)
        omega0 = (1.0 - beta0.sum()) * math.log(sample_var)
        alpha0 = np.zeros(self.q)
        gamma0 = np.full(self.q, 0.1)
        x0 = np.concatenate([[omega0], alpha0, gamma0, beta0])

        def _unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            omega = float(params[0])
            alpha = params[1 : 1 + self.q]
            gamma = params[1 + self.q : 1 + 2 * self.q]
            beta = params[1 + 2 * self.q :]
            return omega, alpha, gamma, beta

        def neg_ll(params: np.ndarray) -> float:
            omega, alpha, gamma, beta = _unpack(params)
            try:
                _, ll = egarch_filter(eps, omega, alpha, gamma, beta, sample_var)
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
        omega, alpha, gamma, beta = _unpack(res.x)
        sigma2, log_lik = egarch_filter(eps, omega, alpha, gamma, beta, sample_var)
        sigma = np.sqrt(sigma2)
        std_resid = eps / sigma

        n_eff = arr.shape[0]
        n_params = 1 + 2 * self.q + self.p + (1 if self.mean == "constant" else 0)
        aic_v, bic_v, hqic_v = info_criteria(log_lik, n_eff, n_params)

        persistence = float(np.abs(beta).sum())
        if persistence >= 1.0 - 1e-6:
            warnings.warn(
                f"Fitted EGARCH({self.p},{self.q}) has Σ|β| ≥ 1 — log-variance "
                "process is non-stationary; forecasts may diverge.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._result = EGARCHFitResult(
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
        )
        return self._result

    # ------------------------------------------------------------------
    # Simulation + forecast (Monte Carlo only — no closed form for EGARCH)
    # ------------------------------------------------------------------

    def simulate(self, t_total: int, *, seed: int | None = None, burnin: int = 500) -> np.ndarray:
        if t_total < 1:
            raise ValueError(f"t_total must be ≥ 1, got {t_total}.")
        res = self.result
        rng = np.random.default_rng(seed)
        n = t_total + burnin
        eps = np.zeros(n)
        log_s = np.full(n, math.log(max(np.exp(res.omega), 1e-10)))
        z = rng.standard_normal(n)
        e_abs_z = math.sqrt(2.0 / math.pi)
        for t in range(1, n):
            v = res.omega
            for i in range(self.q):
                if t - 1 - i >= 0:
                    s_prev = math.exp(log_s[t - 1 - i])
                    zlag = eps[t - 1 - i] / math.sqrt(s_prev)
                    v += res.alpha[i] * zlag + res.gamma[i] * (abs(zlag) - e_abs_z)
            for j in range(self.p):
                if t - 1 - j >= 0:
                    v += res.beta[j] * log_s[t - 1 - j]
            if v > 50.0:
                v = 50.0
            if v < -50.0:
                v = -50.0
            log_s[t] = v
            eps[t] = math.exp(0.5 * v) * z[t]
        return res.mu + eps[burnin:]

    def forecast(
        self,
        h: int,
        *,
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> np.ndarray:
        """Monte-Carlo h-step variance forecasts (no closed form)."""
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")
        if n_paths < 1:
            raise ValueError(f"n_paths must be ≥ 1, got {n_paths}.")
        res = self.result
        rng = np.random.default_rng(seed)
        e_abs_z = math.sqrt(2.0 / math.pi)
        paths = np.empty((n_paths, h))
        log_s_hist = np.log(res.sigma2)
        eps_hist = res.residuals
        for path_idx in range(n_paths):
            sim_log_s = np.concatenate([log_s_hist, np.zeros(h)])
            sim_eps = np.concatenate([eps_hist, np.zeros(h)])
            t0 = len(eps_hist)
            for s in range(h):
                v = res.omega
                for i in range(self.q):
                    s_prev = math.exp(sim_log_s[t0 + s - 1 - i])
                    zlag = sim_eps[t0 + s - 1 - i] / math.sqrt(s_prev)
                    v += res.alpha[i] * zlag + res.gamma[i] * (abs(zlag) - e_abs_z)
                for j in range(self.p):
                    v += res.beta[j] * sim_log_s[t0 + s - 1 - j]
                if v > 50.0:
                    v = 50.0
                if v < -50.0:
                    v = -50.0
                sim_log_s[t0 + s] = v
                z = rng.standard_normal()
                sim_eps[t0 + s] = math.exp(0.5 * v) * z
                paths[path_idx, s] = math.exp(v)
        return paths.mean(axis=0)
