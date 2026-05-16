"""
Bollerslev (1986) GARCH(p, q) volatility model.

    r_t = μ + ε_t,         ε_t = σ_t z_t,         z_t ~ N(0, 1)
    σ²_t = ω + Σ_{i=1}^q α_i ε²_{t-i} + Σ_{j=1}^p β_j σ²_{t-j}

Constraints
-----------
The optimiser uses the log-reparameterisation

    θ = (log ω, log α_1, …, log α_q, logit β_1, …, logit β_p)

to enforce ω, α_i, β_j > 0 and Σ(α + β) < 1 (covariance stationarity).
The logit on β is scaled so the persistence Σα + Σβ stays in (0, 1) at any
parameter vector; see ``_unpack`` for details.  This mirrors the
reparameterisation trick used in ``src/strategies/mean_reversion``.

Forecasting
-----------
``forecast(h, n_paths=None)`` returns variance forecasts.  For GARCH(1,1)
the closed-form recursion

    σ²_{T+h} = ω̄ + (α + β)^{h−1} (σ²_{T+1} − ω̄),   ω̄ = ω / (1 − α − β)

is used when no ``n_paths`` is supplied.  Monte-Carlo simulation is used
for variance-of-variance / fat-tail diagnostics when ``n_paths`` is given.
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
from ._likelihood import garch_filter

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GARCHFitResult:
    """Fitted GARCH(p, q) model output."""

    p: int  # GARCH order
    q: int  # ARCH order
    mu: float
    omega: float
    alpha: np.ndarray  # shape (q,)
    beta: np.ndarray  # shape (p,)
    sigma2: np.ndarray  # conditional variance path, shape (T,)
    residuals: np.ndarray  # ε_t = r_t − μ
    std_residuals: np.ndarray  # z_t = ε_t / σ_t
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    persistence: float  # Σα + Σβ
    unconditional_var: float

    def __str__(self) -> str:
        lines = [
            f"GARCH({self.p},{self.q})  n_obs={self.n_obs}",
            f"  μ        = {self.mu:.6g}",
            f"  ω        = {self.omega:.6g}",
        ]
        for i, a in enumerate(self.alpha, 1):
            lines.append(f"  α_{i}      = {a:.6g}")
        for j, b in enumerate(self.beta, 1):
            lines.append(f"  β_{j}      = {b:.6g}")
        lines += [
            f"  log_lik  = {self.log_lik:.4f}",
            f"  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  persistence = {self.persistence:.4f}  σ²_∞ = {self.unconditional_var:.6g}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        names = (
            ["mu", "omega"]
            + [f"alpha_{i}" for i in range(1, self.q + 1)]
            + [f"beta_{j}" for j in range(1, self.p + 1)]
        )
        values = [self.mu, self.omega, *self.alpha.tolist(), *self.beta.tolist()]
        return pl.DataFrame({"parameter": names, "value": values})


# ---------------------------------------------------------------------------
# Parameterisation helpers
# ---------------------------------------------------------------------------


def _unpack(params: np.ndarray, p: int, q: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Map unconstrained ``params`` to (ω, α, β) with the persistence < 1 constraint.

    Layout (length 1 + q + p):
        log ω | s_α_1, …, s_α_q | s_β_1, …, s_β_p

    α_i = soft(s_α_i),  β_j = soft(s_β_j) where soft(x) = log(1 + exp(x))
    enforces positivity.  Persistence π = Σα + Σβ is then projected into
    (0, 1) via the saturating map π̃ = π / (1 + π).  Coefficients are
    rescaled by π̃ / π so the model remains stationary at every step.
    """
    omega = math.exp(params[0])

    def _softplus(x: float) -> float:
        # Numerically safe softplus.
        if x > 30.0:
            return x
        return math.log1p(math.exp(x))

    alpha = np.array([_softplus(float(params[1 + i])) for i in range(q)])
    beta = np.array([_softplus(float(params[1 + q + j])) for j in range(p)])

    persistence_raw = float(alpha.sum() + beta.sum())
    if persistence_raw > 0.0:
        persistence_eff = persistence_raw / (1.0 + persistence_raw)
        scale = persistence_eff / persistence_raw
        alpha = alpha * scale
        beta = beta * scale
    return omega, alpha, beta


def _pack(omega: float, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Inverse of ``_unpack`` (approximate — used only for warm-starting)."""

    def _inv_softplus(y: float) -> float:
        if y <= 0.0:
            return -10.0
        if y > 30.0:
            return y
        return math.log(math.expm1(y))

    return np.concatenate(
        [
            [math.log(max(omega, 1e-10))],
            [_inv_softplus(float(a)) for a in alpha],
            [_inv_softplus(float(b)) for b in beta],
        ]
    )


# ---------------------------------------------------------------------------
# GARCH class
# ---------------------------------------------------------------------------


class GARCH:
    """Bollerslev GARCH(p, q) model.

    Parameters
    ----------
    p : int   GARCH (persistence) order — number of σ²_{t-j} lags.
    q : int   ARCH order — number of ε²_{t-i} lags.
    mean : {"constant", "zero"}
        ``"constant"`` estimates μ as the sample mean; ``"zero"`` forces μ = 0.
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
        self._result: GARCHFitResult | None = None

    @property
    def result(self) -> GARCHFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> GARCHFitResult:
        """Maximum-likelihood fit by L-BFGS-B on the log/softplus reparameterisation."""
        arr = to_numpy_1d(returns)
        validate_finite(arr)
        validate_min_length(arr, max(self.p, self.q) + 10, "returns")

        mu = float(arr.mean()) if self.mean == "constant" else 0.0
        eps = arr - mu
        sample_var = float(np.var(eps))
        if sample_var <= 0.0:
            raise ValueError("Sample variance of returns is zero — cannot fit GARCH.")

        # Warm-start: small ω, modest α, large β when q ≥ 1, p ≥ 1
        omega0 = 0.05 * sample_var
        alpha0 = np.full(self.q, 0.05)
        beta0 = np.full(self.p, 0.85 / max(self.p, 1)) if self.p > 0 else np.zeros(0)
        x0 = _pack(omega0, alpha0, beta0)

        def neg_ll(params: np.ndarray) -> float:
            omega, alpha, beta = _unpack(params, self.p, self.q)
            try:
                _, ll = garch_filter(eps, omega, alpha, beta, sample_var)
            except Exception:
                return 1e10
            if not math.isfinite(ll):
                return 1e10
            return -ll

        res = scipy.optimize.minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        omega, alpha, beta = _unpack(res.x, self.p, self.q)
        sigma2, log_lik = garch_filter(eps, omega, alpha, beta, sample_var)

        sigma = np.sqrt(sigma2)
        std_resid = eps / sigma

        n_eff = arr.shape[0]
        n_params = 1 + self.q + self.p + (1 if self.mean == "constant" else 0)
        aic_v, bic_v, hqic_v = info_criteria(log_lik, n_eff, n_params)

        persistence = float(alpha.sum() + beta.sum())
        if persistence >= 1.0 - 1e-8:
            warnings.warn(
                f"Fitted GARCH({self.p},{self.q}) has persistence ≥ 1 — variance is "
                "non-stationary; forecasts will diverge.",
                RuntimeWarning,
                stacklevel=2,
            )
        unc_var = omega / max(1.0 - persistence, 1e-10)

        self._result = GARCHFitResult(
            p=self.p,
            q=self.q,
            mu=mu,
            omega=omega,
            alpha=alpha,
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
    # Forecast
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        *,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """h-step variance forecasts.

        When ``n_paths`` is None, applies the deterministic recursion

            σ²_{T+s} = ω + Σ_{i=1}^q α_i ε²_{T+s-i}_hat + Σ_{j=1}^p β_j σ²_{T+s-j}_hat

        where for s ≥ 1 the future ε²_{T+s-i} is replaced by σ²_{T+s-i} (the
        conditional expectation under z² → χ²₁ with E[z²] = 1).

        With ``n_paths``, runs Monte Carlo by drawing ε from the Gaussian
        innovation distribution; returns the average σ² across paths so the
        return shape is always ``(h,)``.

        Returns
        -------
        np.ndarray, shape (h,)  — h-step variance forecasts.
        """
        if h <= 0:
            raise ValueError(f"h must be ≥ 1, got {h}.")
        res = self.result
        eps = res.residuals
        sigma2 = res.sigma2

        if n_paths is None:
            out = np.empty(h)
            recent_eps_sq = np.concatenate([eps**2, np.zeros(h)])
            recent_sigma2 = np.concatenate([sigma2, np.zeros(h)])
            t0 = len(eps)
            for s in range(h):
                v = res.omega
                for i in range(self.q):
                    idx = t0 + s - 1 - i
                    if idx < t0:
                        v += res.alpha[i] * recent_eps_sq[idx]
                    else:
                        v += res.alpha[i] * recent_sigma2[idx]
                for j in range(self.p):
                    v += res.beta[j] * recent_sigma2[t0 + s - 1 - j]
                recent_sigma2[t0 + s] = v
                recent_eps_sq[t0 + s] = v  # E[ε²] = σ²
                out[s] = v
            return out

        # Monte Carlo
        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, h))
        for p_idx in range(n_paths):
            sim_eps_sq = np.concatenate([eps**2, np.zeros(h)])
            sim_sigma2 = np.concatenate([sigma2, np.zeros(h)])
            t0 = len(eps)
            for s in range(h):
                v = res.omega
                for i in range(self.q):
                    v += res.alpha[i] * sim_eps_sq[t0 + s - 1 - i]
                for j in range(self.p):
                    v += res.beta[j] * sim_sigma2[t0 + s - 1 - j]
                sim_sigma2[t0 + s] = max(v, 1e-12)
                z = rng.standard_normal()
                sim_eps_sq[t0 + s] = sim_sigma2[t0 + s] * z * z
                paths[p_idx, s] = sim_sigma2[t0 + s]
        return paths.mean(axis=0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 500,
    ) -> np.ndarray:
        """Simulate a return path from the fitted model."""
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
                    v += res.alpha[i] * eps[t - 1 - i] ** 2
            for j in range(self.p):
                if t - 1 - j >= 0:
                    v += res.beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(v, 1e-12)
            eps[t] = math.sqrt(sigma2[t]) * z[t]
        return res.mu + eps[burnin:]
