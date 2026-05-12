"""Continuous-emission Gaussian Hidden Markov Model (GHMM).

Implements the Forward-Backward, Viterbi, and Baum-Welch algorithms for continuous
multivariate observations. Emissions are modelled as Multivariate Gaussian
distributions, fully evaluated in log-space to prevent underflow.

Notation
--------
K  — number of hidden states.
D  — dimensionality of the observation vector (e.g., 2 for dP and dV).
T  — sequence length.
A  — (K, K) hidden-state transition matrix; A[i, j] = P(z_{t+1}=j | z_t=i).
μ  — (K, D) mean vectors for the Gaussian emissions.
Σ  — (K, D, D) covariance matrices for the Gaussian emissions.
π  — (K,) initial hidden-state distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

_EPS: float = 1e-12

# ---------------------------------------------------------------------------
# Types (Note: Move this to your _types.py file)
# ---------------------------------------------------------------------------


@dataclass
class GaussianHMMFit:
    transition_matrix: NDArray[np.float64]
    means: NDArray[np.float64]
    covars: NDArray[np.float64]
    initial_probs: NDArray[np.float64]
    n_states: int
    n_features: int
    log_likelihood: float
    n_iter: int
    converged: bool


# ---------------------------------------------------------------------------
# Numerically stable log-space helpers
# ---------------------------------------------------------------------------


def _logsumexp_1d(a: NDArray[np.float64]) -> float:
    """Log-sum-exp of a 1-D array, returning a Python float."""
    a_max = float(a.max())
    if math.isinf(a_max) and a_max < 0.0:
        return -math.inf
    return math.log(float(np.sum(np.exp(a - a_max)))) + a_max


def _logsumexp_2d(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Log-sum-exp along one axis of a 2-D array."""
    a_max = a.max(axis=axis, keepdims=True)
    safe_max = np.where(np.isneginf(a_max), 0.0, a_max)
    return np.log(np.sum(np.exp(a - safe_max), axis=axis)) + safe_max.squeeze(axis=axis)


def _calc_log_emit_seq(
    obs: NDArray[np.float64], means: NDArray[np.float64], covars: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pre-compute the log-emission probability of every observation for every state.

    Returns:
        log_emit_seq: Array of shape (T, K) where entry [t, k] is log N(x_t | μ_k, Σ_k).
    """
    n_obs = obs.shape[0]
    n_states = means.shape[0]
    log_emit_seq = np.empty((n_obs, n_states), dtype=np.float64)

    for k in range(n_states):
        # allow_singular=True prevents crashes if a state collapses to zero variance
        log_emit_seq[:, k] = multivariate_normal.logpdf(
            obs, mean=means[k], cov=covars[k], allow_singular=True
        )
    return log_emit_seq


# ---------------------------------------------------------------------------
# Forward and backward algorithms
# ---------------------------------------------------------------------------


def forward_log(
    log_emit_seq: NDArray[np.float64],
    log_trans: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Forward algorithm in log-space (alpha recursion)."""
    n_obs, n_hidden = log_emit_seq.shape
    log_alpha = np.empty((n_obs, n_hidden), dtype=np.float64)

    log_alpha[0] = log_pi + log_emit_seq[0]
    for t in range(1, n_obs):
        scores = log_alpha[t - 1, :, np.newaxis] + log_trans  # (K, K)
        log_alpha[t] = _logsumexp_2d(scores, axis=0) + log_emit_seq[t]

    return log_alpha, _logsumexp_1d(log_alpha[-1])


def backward_log(
    log_emit_seq: NDArray[np.float64],
    log_trans: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Backward algorithm in log-space (beta recursion)."""
    n_obs, n_hidden = log_emit_seq.shape
    log_beta = np.zeros((n_obs, n_hidden), dtype=np.float64)

    for t in range(n_obs - 2, -1, -1):
        scores = log_trans + log_emit_seq[t + 1] + log_beta[t + 1]  # (K, K)
        log_beta[t] = _logsumexp_2d(scores, axis=1)

    return log_beta


# ---------------------------------------------------------------------------
# Posterior marginals
# ---------------------------------------------------------------------------


def posteriors(
    log_emit_seq: NDArray[np.float64],
    log_trans: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute state posteriors (γ) and pairwise transition posteriors (ξ)."""
    log_alpha, ll = forward_log(log_emit_seq, log_trans, log_pi)
    log_beta = backward_log(log_emit_seq, log_trans)

    log_g = log_alpha + log_beta  # (T, K)
    lse = _logsumexp_2d(log_g, axis=1)  # (T,)
    gamma = np.exp(log_g - lse[:, np.newaxis])  # (T, K)

    log_xi = (
        log_alpha[:-1, :, np.newaxis]  # (T-1, K, 1)
        + log_trans[np.newaxis, :, :]  # (1, K, K)
        + log_emit_seq[1:, np.newaxis, :]  # (T-1, 1, K)
        + log_beta[1:, np.newaxis, :]  # (T-1, 1, K)
        - ll
    )
    xi = np.exp(log_xi)  # (T-1, K, K)

    return gamma, xi, ll


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def viterbi(
    log_emit_seq: NDArray[np.float64],
    log_trans: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.intp], float]:
    """Viterbi algorithm: most probable hidden-state sequence."""
    n_obs, n_hidden = log_emit_seq.shape
    log_delta = np.empty((n_obs, n_hidden), dtype=np.float64)
    psi = np.empty((n_obs, n_hidden), dtype=np.intp)

    log_delta[0] = log_pi + log_emit_seq[0]
    psi[0] = 0

    for t in range(1, n_obs):
        scores = log_delta[t - 1, :, np.newaxis] + log_trans  # (K, K)
        psi[t] = np.argmax(scores, axis=0)  # (K,)
        log_delta[t] = scores.max(axis=0) + log_emit_seq[t]

    path = np.empty(n_obs, dtype=np.intp)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, float(log_delta[-1, path[-1]])


# ---------------------------------------------------------------------------
# Baum-Welch EM
# ---------------------------------------------------------------------------


def _init_params(
    obs: NDArray[np.float64],
    n_states: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Initialise matrices. Means are randomly sampled from observations."""
    n_obs, n_features = obs.shape

    trans_mat: NDArray[np.float64] = rng.dirichlet(np.ones(n_states), size=n_states)
    pi: NDArray[np.float64] = rng.dirichlet(np.ones(n_states))

    # Randomly select initial means from the data
    indices = rng.choice(n_obs, size=n_states, replace=False)
    means: NDArray[np.float64] = obs[indices].copy()

    # Initialize all covariances to the global empirical covariance + ridge
    global_cov = np.cov(obs, rowvar=False)
    if n_features == 1:
        global_cov = np.array([[global_cov]])

    global_cov.flat[:: n_features + 1] += 1e-3  # Ridge
    covars = np.tile(global_cov, (n_states, 1, 1))

    return trans_mat, means, covars, pi


def _m_step(
    obs: NDArray[np.float64],
    gamma: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """M-step: re-estimate transition, mean, covariance, and initial matrices."""
    n_obs, n_features = obs.shape
    n_states = gamma.shape[1]

    pi = np.maximum(gamma[0], _EPS)
    pi /= pi.sum()

    xi_sum = xi.sum(axis=0)  # (K, K)
    gamma_sum = np.maximum(gamma.sum(axis=0), _EPS)  # (K,)

    trans_mat = np.maximum(xi_sum / gamma[:-1].sum(axis=0, keepdims=True).T, _EPS)
    trans_mat /= trans_mat.sum(axis=1, keepdims=True)

    # Re-estimate means: (K, D)
    means = (gamma.T @ obs) / gamma_sum[:, np.newaxis]

    # Re-estimate covariances: (K, D, D)
    covars = np.empty((n_states, n_features, n_features), dtype=np.float64)
    for k in range(n_states):
        diff = obs - means[k]  # (T, D)
        # Vectorized weighted covariance calculation
        covars[k] = (gamma[:, k] * diff.T) @ diff / gamma_sum[k]
        # Add ridge regularization to diagonal to prevent singular matrices
        covars[k].flat[:: n_features + 1] += 1e-4

    return trans_mat, means, covars, pi


def fit(
    obs: NDArray[np.float64],
    n_states: int,
    *,
    n_init: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
    verbose: bool = False,  # <-- Added verbose flag
) -> GaussianHMMFit:
    """Fit a Gaussian HMM via Baum-Welch expectation-maximisation."""
    if len(obs) <= 1:
        raise ValueError("Observation sequence must have length > 1")
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]

    if rng is None:
        rng = np.random.default_rng()

    best_ll = -math.inf
    best: GaussianHMMFit | None = None
    n_features = obs.shape[1]

    if verbose:
        print(f"[ghmm] Starting Baum-Welch EM ({n_init} restarts, up to {max_iter} iters each)...")

    for init_idx in range(n_init):
        trans_mat, means, covars, pi = _init_params(obs, n_states, rng)
        prev_ll = -math.inf
        converged = False

        if verbose:
            print(f"  > Init {init_idx + 1}/{n_init}: ", end="", flush=True)

        for n_iter in range(1, max_iter + 1):
            log_trans = np.log(np.maximum(trans_mat, _EPS))
            log_pi = np.log(np.maximum(pi, _EPS))

            # E-Step & M-Step
            log_emit_seq = _calc_log_emit_seq(obs, means, covars)
            gamma, xi, ll = posteriors(log_emit_seq, log_trans, log_pi)
            trans_mat, means, covars, pi = _m_step(obs, gamma, xi)

            if abs(ll - prev_ll) < tol:
                converged = True
                break
            prev_ll = ll

        if verbose:
            status = "Converged" if converged else "Max Iters"
            print(f"{status} at iter {n_iter} | LL: {ll:.2f}")

        if ll > best_ll:
            best_ll = ll
            best = GaussianHMMFit(
                transition_matrix=trans_mat,
                means=means,
                covars=covars,
                initial_probs=pi,
                n_states=n_states,
                n_features=n_features,
                log_likelihood=ll,
                n_iter=n_iter,
                converged=converged,
            )

    if verbose and best is not None:
        print(f"[ghmm] Fitting complete. Best LL: {best.log_likelihood:.2f}\n")

    assert best is not None
    return best


def decode(obs: NDArray[np.float64], model: GaussianHMMFit) -> NDArray[np.intp]:
    """Decode the most probable hidden-state sequence for a fitted GHMM."""
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]

    log_trans = np.log(np.maximum(model.transition_matrix, _EPS))
    log_pi = np.log(np.maximum(model.initial_probs, _EPS))
    log_emit_seq = _calc_log_emit_seq(obs, model.means, model.covars)

    path, _ = viterbi(log_emit_seq, log_trans, log_pi)
    return path
