"""
Markov-switching autoregressive model.

Model
-----
    y_t = μ_{S_t} + Σ_{i=1}^p φ_{S_t,i} (y_{t-i} − μ_{S_t}) + σ_{S_t} ε_t,
    S_t ∈ {0, …, K−1},   ε_t ~ N(0, 1)

with regime evolution governed by an irreducible first-order Markov chain
with transition matrix P (row-stochastic, P[i, j] = Pr(S_t = j | S_{t-1} = i)).

Each regime has its own mean μ_k, AR coefficients φ_{k,1…p}, and
innovation standard deviation σ_k.  The regime sequence is latent and
inferred via the Hamilton (1989) forward-backward filter.

Estimation
----------
Expectation-Maximisation (Baum-Welch):

E-step:  Hamilton filter forward pass → filtered probabilities ξ_{t|t},
         Kim (1994) smoother backward pass → smoothed probabilities ξ_{t|T}.

M-step:  Closed-form weighted least squares for (μ_k, φ_k); weighted
         residual variance for σ²_k; smoothed-joint probabilities normalise
         row-wise to update P.

Initial conditions
------------------
* Regime parameters seeded by K-means on the response y_t alone (intercepts
  spread across the empirical range; AR coefficients all initialised to 0).
* P initialised to a sticky uniform: P[i, i] = 0.9, off-diagonal uniform.
* Initial regime distribution π_0 initialised to the stationary distribution
  of the seed P (largest left-eigenvector of P).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from ._io import to_numpy_1d, validate_finite, validate_min_length
from .utils import info_criteria

# ruff: noqa: N803, N806  — matrix variables (P, S, F, K)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MSARFitResult:
    """Fitted Markov-switching AR(p) result.

    Attributes
    ----------
    p                AR order.
    k_regimes        Number of regimes.
    mu               Per-regime mean,           shape (K,)
    phi              Per-regime AR coefs,       shape (K, p)
    sigma2           Per-regime variance,       shape (K,)
    transition       Transition matrix P,       shape (K, K)
    initial          Initial regime distribution π_0, shape (K,)
    filtered_probs   ξ_{t|t}, shape (T_eff, K)
    smoothed_probs   ξ_{t|T}, shape (T_eff, K)
    predicted_probs  ξ_{t|t-1}, shape (T_eff, K)
    log_lik          Final log-likelihood.
    aic, bic, hqic   Information criteria.
    n_obs            Effective sample size T_eff = T − p.
    n_iter           Iterations until EM convergence (or max).
    converged        Whether EM converged within tolerance.
    """

    p: int
    k_regimes: int
    mu: np.ndarray
    phi: np.ndarray
    sigma2: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    filtered_probs: np.ndarray
    smoothed_probs: np.ndarray
    predicted_probs: np.ndarray
    log_lik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    n_iter: int
    converged: bool

    def __str__(self) -> str:
        lines = [
            f"MarkovSwitchingAR(p={self.p}, K={self.k_regimes})  n_obs={self.n_obs}",
            f"  log_lik={self.log_lik:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"  converged={self.converged}  iters={self.n_iter}",
        ]
        for k in range(self.k_regimes):
            lines.append(
                f"  Regime {k}: μ={self.mu[k]:.4f}  σ²={self.sigma2[k]:.4f}  φ={self.phi[k]}"
            )
        lines.append(f"  Transition matrix P =\n{self.transition}")
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Long-format DataFrame of smoothed regime probabilities (one row per (t, k))."""
        t_total, k_regimes = self.smoothed_probs.shape
        ts = np.repeat(np.arange(t_total, dtype=np.int64), k_regimes)
        ks = np.tile(np.arange(k_regimes, dtype=np.int64), t_total)
        return pl.DataFrame(
            {
                "t": ts,
                "regime": ks,
                "smoothed_prob": self.smoothed_probs.ravel(),
                "filtered_prob": self.filtered_probs.ravel(),
            }
        )

    def most_likely_regime(self) -> np.ndarray:
        """Argmax over smoothed probabilities at each step, shape (T_eff,)."""
        return np.argmax(self.smoothed_probs, axis=1)


# ---------------------------------------------------------------------------
# Hamilton filter + Kim smoother
# ---------------------------------------------------------------------------


def _hamilton_filter(
    densities: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward Hamilton (1989) filter.

    Parameters
    ----------
    densities   shape (T, K)  — likelihood ξ_t(y_t | S_t = k, history)
    transition  shape (K, K)
    initial     shape (K,)    π_0

    Returns
    -------
    filtered    ξ_{t|t},  shape (T, K)
    predicted   ξ_{t|t-1}, shape (T, K)
    log_lik     total log-likelihood
    """
    t_total, k_regimes = densities.shape
    filtered = np.zeros((t_total, k_regimes))
    predicted = np.zeros((t_total, k_regimes))
    log_lik = 0.0
    prev_filtered = initial
    for t in range(t_total):
        # Predict
        pred = transition.T @ prev_filtered
        predicted[t] = pred
        # Joint and normalise
        joint = pred * densities[t]
        marginal = joint.sum()
        if marginal <= 0.0 or not math.isfinite(marginal):
            # Degenerate step: fall back to predicted
            filtered[t] = pred
            log_lik += -1e6
        else:
            filtered[t] = joint / marginal
            log_lik += math.log(marginal)
        prev_filtered = filtered[t]
    return filtered, predicted, log_lik


def _kim_smoother(
    filtered: np.ndarray,
    predicted: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kim (1994) backward smoother.

    Returns
    -------
    smoothed         ξ_{t|T},     shape (T, K)
    joint_smoothed   ξ_{t, t-1|T}, shape (T, K, K)  — used by the M-step
                                                     for re-estimating P.
                                                     ``joint_smoothed[0]`` is zero.
    """
    t_total, k_regimes = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    joint_smoothed = np.zeros((t_total, k_regimes, k_regimes))

    for t in range(t_total - 2, -1, -1):
        # For each (i, j): joint = filtered[t, i] * P[i, j] * smoothed[t+1, j] / predicted[t+1, j]
        next_pred = predicted[t + 1]
        next_pred_safe = np.where(next_pred > 0, next_pred, 1e-300)
        # joint[i, j] = filtered[t, i] * P[i, j] * smoothed[t+1, j] / predicted[t+1, j]
        joint = (filtered[t][:, None] * transition) * (smoothed[t + 1] / next_pred_safe)[None, :]
        joint_smoothed[t + 1] = joint
        smoothed[t] = joint.sum(axis=1)
        # Guard against negative values from numerical drift
        smoothed[t] = np.clip(smoothed[t], 0.0, None)
        s = smoothed[t].sum()
        if s > 0:
            smoothed[t] = smoothed[t] / s
    return smoothed, joint_smoothed


# ---------------------------------------------------------------------------
# Markov-switching AR model
# ---------------------------------------------------------------------------


class MarkovSwitchingAR:
    """Markov-switching autoregressive model with K regimes.

    Parameters
    ----------
    p          AR order (≥ 0).  ``p = 0`` reduces to a hidden Markov model
               with Gaussian emissions (mean + variance per regime).
    k_regimes  Number of regimes (≥ 2).
    """

    def __init__(self, p: int = 0, k_regimes: int = 2) -> None:
        if p < 0:
            raise ValueError(f"p must be ≥ 0; got {p}.")
        if k_regimes < 2:
            raise ValueError(f"k_regimes must be ≥ 2; got {k_regimes}.")
        self.p = p
        self.k_regimes = k_regimes
        self._result: MSARFitResult | None = None

    @property
    def result(self) -> MSARFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted — call fit() first.")
        return self._result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> MSARFitResult:
        """Estimate parameters by the Baum-Welch EM algorithm.

        Parameters
        ----------
        y         Observation series, shape (T,).
        max_iter  Maximum EM iterations.
        tol       Convergence tolerance on relative log-likelihood change.
        seed      RNG seed for the K-means initialisation.
        """
        arr = to_numpy_1d(y)
        validate_finite(arr)
        validate_min_length(arr, self.p + self.k_regimes + 5, "y")
        T = arr.shape[0]
        p, K = self.p, self.k_regimes

        # Build lag-design matrix and target (drop first p observations)
        if p > 0:
            X = np.column_stack([arr[p - 1 - i : T - 1 - i] for i in range(p)])
        else:
            X = np.zeros((T - p, 0))
        y_target = arr[p:]
        T_eff = y_target.shape[0]

        # ---- Initialisation ----
        rng = np.random.default_rng(seed)
        mu, phi, sigma2 = self._initialise_regimes(y_target, X, K, p, rng)
        # Sticky transition matrix: 0.9 on diagonal, uniform off-diagonal
        transition = np.full((K, K), (1.0 - 0.9) / max(K - 1, 1))
        np.fill_diagonal(transition, 0.9)
        # Stationary distribution as initial
        initial = _stationary_distribution(transition)

        # ---- EM loop ----
        prev_ll = -math.inf
        converged = False
        n_iter = 0
        filtered = predicted = smoothed = joint_smoothed = np.empty(0)
        log_lik = -math.inf
        for n_iter in range(1, max_iter + 1):
            # E-step
            densities = _emission_densities(y_target, X, mu, phi, sigma2, p)
            filtered, predicted, log_lik = _hamilton_filter(densities, transition, initial)
            smoothed, joint_smoothed = _kim_smoother(filtered, predicted, transition)

            # M-step
            mu, phi, sigma2 = _m_step_regression(y_target, X, smoothed, p)
            transition = _m_step_transition(joint_smoothed)
            initial = smoothed[0].copy()

            # Convergence
            if math.isfinite(prev_ll):
                rel_change = abs(log_lik - prev_ll) / (abs(prev_ll) + 1e-10)
                if rel_change < tol:
                    converged = True
                    break
            prev_ll = log_lik

        if not converged:
            warnings.warn(
                f"MarkovSwitchingAR EM did not converge in {max_iter} iterations.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Final E-step with the converged parameters for clean diagnostics
        densities = _emission_densities(y_target, X, mu, phi, sigma2, p)
        filtered, predicted, log_lik = _hamilton_filter(densities, transition, initial)
        smoothed, _ = _kim_smoother(filtered, predicted, transition)

        n_params = K * (1 + p + 1) + K * (K - 1) + (K - 1)
        aic_v, bic_v, hqic_v = info_criteria(log_lik, T_eff, n_params)

        self._result = MSARFitResult(
            p=p,
            k_regimes=K,
            mu=mu,
            phi=phi,
            sigma2=sigma2,
            transition=transition,
            initial=initial,
            filtered_probs=filtered,
            smoothed_probs=smoothed,
            predicted_probs=predicted,
            log_lik=float(log_lik),
            aic=aic_v,
            bic=bic_v,
            hqic=hqic_v,
            n_obs=T_eff,
            n_iter=n_iter,
            converged=converged,
        )
        return self._result

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_total: int,
        *,
        seed: int | None = None,
        burnin: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate from the fitted MS-AR model.

        Returns
        -------
        y       Observation series, shape (t_total,)
        states  Latent regime sequence, shape (t_total,)
        """
        if t_total < 1:
            raise ValueError(f"t_total must be ≥ 1, got {t_total}.")
        res = self.result
        rng = np.random.default_rng(seed)
        n = t_total + burnin
        states = np.zeros(n, dtype=np.int64)
        states[0] = int(rng.choice(res.k_regimes, p=res.initial))
        y = np.zeros(n)
        # Initialise lags with zeros (warm-up absorbs the bias)
        for t in range(1, n):
            states[t] = int(rng.choice(res.k_regimes, p=res.transition[states[t - 1]]))
        for t in range(n):
            k = states[t]
            mu_k = res.mu[k]
            yhat = mu_k
            for i in range(res.p):
                if t - 1 - i >= 0:
                    yhat += res.phi[k, i] * (y[t - 1 - i] - mu_k)
            y[t] = yhat + math.sqrt(res.sigma2[k]) * rng.standard_normal()
        return y[burnin:], states[burnin:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _initialise_regimes(
        y: np.ndarray, X: np.ndarray, K: int, p: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spread regime means across the empirical quantiles."""
        qs = np.linspace(0.1, 0.9, K)
        mu = np.quantile(y, qs)
        # Perturb slightly so equal-quantile cases differ
        mu = mu + rng.standard_normal(K) * 1e-3
        phi = np.zeros((K, p))
        # Initial sigma² = global variance
        var = float(np.var(y))
        sigma2 = np.full(K, max(var, 1e-6))
        return mu, phi, sigma2


# ---------------------------------------------------------------------------
# Helpers (module-level so they're importable for testing if needed)
# ---------------------------------------------------------------------------


def _emission_densities(
    y: np.ndarray, X: np.ndarray, mu: np.ndarray, phi: np.ndarray, sigma2: np.ndarray, p: int
) -> np.ndarray:
    """Gaussian densities of y under each regime k, shape (T, K)."""
    T_eff = y.shape[0]
    K = mu.shape[0]
    out = np.empty((T_eff, K))
    log_two_pi = math.log(2.0 * math.pi)
    for k in range(K):
        s2 = max(float(sigma2[k]), 1e-12)
        # Demeaned predictor:  μ_k + φ_k' (X − μ_k) row-wise
        if p > 0:
            pred = mu[k] + (X - mu[k]) @ phi[k]
        else:
            pred = np.full(T_eff, mu[k])
        eps = y - pred
        out[:, k] = np.exp(-0.5 * (log_two_pi + math.log(s2) + eps * eps / s2))
    # Floor to avoid total zero rows
    out = np.maximum(out, 1e-300)
    return out


def _m_step_regression(
    y: np.ndarray, X: np.ndarray, smoothed: np.ndarray, p: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least squares per regime + weighted residual variance."""
    T_eff, K = smoothed.shape
    mu = np.empty(K)
    phi = np.zeros((K, p))
    sigma2 = np.empty(K)
    for k in range(K):
        w = smoothed[:, k]
        wsum = w.sum()
        if wsum < 1e-8:
            mu[k] = float(y.mean())
            sigma2[k] = max(float(np.var(y)), 1e-6)
            continue
        # Design: [1, X] for the linear regression y_t = α_k + β_k' X_t + ε
        if p > 0:
            design = np.column_stack([np.ones(T_eff), X])
            W = np.sqrt(w)[:, None]
            try:
                coef, *_ = np.linalg.lstsq(design * W, y * W.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                coef = np.zeros(1 + p)
                coef[0] = float((w * y).sum() / wsum)
            alpha_k = float(coef[0])
            beta_k = coef[1:]
            phi[k] = beta_k
            # Reparameterise: y_t = μ_k + β_k' (X_t − μ_k) + ε
            # ⇒ α_k = μ_k (1 − Σ β_k_i) ⇒ μ_k = α_k / (1 − Σ β_k_i)
            denom = 1.0 - beta_k.sum()
            mu[k] = alpha_k / denom if abs(denom) > 1e-8 else float((w * y).sum() / wsum)
            pred = mu[k] + (X - mu[k]) @ beta_k
        else:
            mu[k] = float((w * y).sum() / wsum)
            pred = np.full(T_eff, mu[k])
        eps = y - pred
        sigma2[k] = float(np.maximum((w * eps * eps).sum() / wsum, 1e-6))
    return mu, phi, sigma2


def _m_step_transition(joint_smoothed: np.ndarray) -> np.ndarray:
    """Re-estimate P from the smoothed joint probabilities.

    ``joint_smoothed[t][i, j] = Pr(S_{t-1} = i, S_t = j | data)`` for t ≥ 1.
    """
    # Sum over t ≥ 1.
    num = joint_smoothed[1:].sum(axis=0)  # shape (K, K)
    denom = num.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    P = num / denom
    # Numerical guard: clip to (1e-12, 1) and renormalise rows.
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Largest left-eigenvector of P, normalised to a probability vector."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    s = vec.sum()
    if s <= 0:
        return np.full(P.shape[0], 1.0 / P.shape[0])
    return vec / s
