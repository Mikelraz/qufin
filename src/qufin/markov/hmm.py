"""Discrete-emission Hidden Markov Model (HMM).

Implements the three fundamental HMM algorithms entirely in log-space to avoid
floating-point underflow on long observation sequences:

* **Forward–Backward** — computes posterior state marginals P(z_t | x_{0:T})
  and the sequence log-likelihood log P(x_{0:T}).
* **Viterbi** — dynamic-programming decoder for the most probable hidden-state
  path argmax_{z_{0:T}} P(z_{0:T} | x_{0:T}).
* **Baum-Welch** — expectation-maximisation algorithm for fitting A, B, and π
  from an unlabelled observation sequence.

Notation
--------
K  — number of hidden states.
V  — vocabulary size (number of distinct observation symbols).
T  — sequence length.
A  — (K, K) hidden-state transition matrix; A[i, j] = P(z_{t+1}=j | z_t=i).
B  — (K, V) emission probability matrix; B[k, v] = P(x_t=v | z_t=k).
π  — (K,) initial hidden-state distribution.

Numerical convention
--------------------
- All public algorithmic functions accept and return log-space arrays
  (``log_trans``, ``log_emit``, ``log_pi``) for composability.
- The convenience :func:`fit`, :func:`decode`, :func:`score`, and
  :func:`simulate` functions work in probability space for ease of use.
- The internal epsilon floor ``_EPS = 1e-12`` prevents log(0).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from qufin.markov._types import HMMFit

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Numerically stable log-space helpers
# ---------------------------------------------------------------------------


def _logsumexp_1d(a: NDArray[np.float64]) -> float:
    """Log-sum-exp of a 1-D array, returning a Python float.

    Uses the max-shift trick to prevent overflow or underflow::

        log Σ exp(aᵢ) = a_max + log Σ exp(aᵢ - a_max)

    Returns ``-inf`` if all elements are ``-inf``.
    """
    a_max = float(a.max())
    if math.isinf(a_max) and a_max < 0.0:
        return -math.inf
    return math.log(float(np.sum(np.exp(a - a_max)))) + a_max


def _logsumexp_2d(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Log-sum-exp along one axis of a 2-D array.

    Args:
        a: Input array of shape (M, N).
        axis: Axis to reduce.  0 → result shape (N,); 1 → result shape (M,).

    Returns:
        Reduced array after applying the max-shift trick along ``axis``.
    """
    a_max = a.max(axis=axis, keepdims=True)
    # When a_max is -inf (all entries are -inf in that slice), shift by 0 so
    # that exp(-inf) = 0, log(0) = -inf — which is the correct result.
    safe_max = np.where(np.isneginf(a_max), 0.0, a_max)
    return np.log(np.sum(np.exp(a - safe_max), axis=axis)) + safe_max.squeeze(axis=axis)


# ---------------------------------------------------------------------------
# Forward and backward algorithms
# ---------------------------------------------------------------------------


def forward_log(
    obs: NDArray[np.intp],
    log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Forward algorithm in log-space (alpha recursion).

    Computes::

        log α[t, i] = log P(x_0, …, x_t, z_t = i | θ)

    Recursion::

        log α[0, i] = log π[i] + log B[i, x_0]
        log α[t, j] = logsumexp_i(log α[t-1, i] + log A[i, j]) + log B[j, x_t]

    The inner logsumexp is vectorised as a (K, 1) + (K, K) broadcast followed
    by a reduction along axis 0, costing O(K²) per time step.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        log_trans: Log transition matrix of shape (K, K).
        log_emit: Log emission matrix of shape (K, V).
        log_pi: Log initial distribution of shape (K,).

    Returns:
        Tuple of:
        - ``log_alpha``: array of shape (T, K).
        - ``log_likelihood``: scalar log P(x_{0:T} | θ).
    """
    n_obs = len(obs)
    n_hidden = log_pi.shape[0]
    log_alpha = np.empty((n_obs, n_hidden), dtype=np.float64)
    log_alpha[0] = log_pi + log_emit[:, obs[0]]
    for t in range(1, n_obs):
        # scores[i, j] = log_alpha[t-1, i] + log_trans[i, j]
        scores = log_alpha[t - 1, :, np.newaxis] + log_trans  # (K, K)
        log_alpha[t] = _logsumexp_2d(scores, axis=0) + log_emit[:, obs[t]]
    return log_alpha, _logsumexp_1d(log_alpha[-1])


def backward_log(
    obs: NDArray[np.intp],
    log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Backward algorithm in log-space (beta recursion).

    Computes::

        log β[t, i] = log P(x_{t+1}, …, x_{T-1} | z_t = i, θ)

    Recursion (initialised to ``log β[T-1, :] = 0``)::

        log β[t, i] = logsumexp_j(log A[i, j] + log B[j, x_{t+1}] + log β[t+1, j])

    The inner logsumexp is vectorised as an (K, K) broadcast reduced along
    axis 1, costing O(K²) per time step.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        log_trans: Log transition matrix of shape (K, K).
        log_emit: Log emission matrix of shape (K, V).

    Returns:
        ``log_beta`` array of shape (T, K).  ``log_beta[T-1, :] = 0`` by definition.
    """
    n_obs = len(obs)
    n_hidden = log_trans.shape[0]
    log_beta = np.zeros((n_obs, n_hidden), dtype=np.float64)
    for t in range(n_obs - 2, -1, -1):
        # scores[i, j] = log_trans[i, j] + log_emit[j, x_{t+1}] + log_beta[t+1, j]
        scores = log_trans + log_emit[:, obs[t + 1]] + log_beta[t + 1]  # (K, K)
        log_beta[t] = _logsumexp_2d(scores, axis=1)
    return log_beta


# ---------------------------------------------------------------------------
# Posterior marginals and pairwise posteriors
# ---------------------------------------------------------------------------


def posteriors(
    obs: NDArray[np.intp],
    log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute state posteriors (γ) and pairwise transition posteriors (ξ).

    These are the sufficient statistics for the Baum-Welch M-step.

    Definitions::

        γ[t, i]    = P(z_t = i | x_{0:T}, θ)
        ξ[t, i, j] = P(z_t = i, z_{t+1} = j | x_{0:T}, θ),  t < T-1

    Both quantities are computed in log-space and converted to probability space
    before being returned.  For each t, ``γ[t, :]`` and ``ξ[t, :, :]`` each
    sum to 1.

    The ξ tensor is computed with a single fully-vectorised broadcast::

        log ξ[t, i, j] = log α[t, i] + log A[i, j]
                       + log B[j, x_{t+1}] + log β[t+1, j] − log P(x)

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        log_trans: Log transition matrix of shape (K, K).
        log_emit: Log emission matrix of shape (K, V).
        log_pi: Log initial distribution of shape (K,).

    Returns:
        Tuple of:
        - ``gamma``: array of shape (T, K) in probability space.
        - ``xi``: array of shape (T-1, K, K) in probability space.
        - ``log_likelihood``: scalar log P(x_{0:T} | θ).
    """
    log_alpha, ll = forward_log(obs, log_trans, log_emit, log_pi)
    log_beta = backward_log(obs, log_trans, log_emit)

    # γ: normalise log α + log β per time step (softmax over states).
    log_g = log_alpha + log_beta  # (T, K)
    lse = _logsumexp_2d(log_g, axis=1)  # (T,)
    gamma = np.exp(log_g - lse[:, np.newaxis])  # (T, K)

    # ξ: four-term broadcast — shape (T-1, K, K).
    # log_emit[:, obs[1:]].T  →  (T-1, K):  entry [t, j] = log_emit[j, x_{t+1}]
    log_xi = (
        log_alpha[:-1, :, np.newaxis]  # (T-1, K, 1)
        + log_trans[np.newaxis, :, :]  # (1, K, K)
        + log_emit[:, obs[1:]].T[:, np.newaxis, :]  # (T-1, 1, K)
        + log_beta[1:, np.newaxis, :]  # (T-1, 1, K)
        - ll
    )
    xi = np.exp(log_xi)  # (T-1, K, K)
    return gamma, xi, ll


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def viterbi(
    obs: NDArray[np.intp],
    log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
    log_pi: NDArray[np.float64],
) -> tuple[NDArray[np.intp], float]:
    """Viterbi algorithm: most probable hidden-state sequence.

    Solves::

        z* = argmax_{z_{0:T}} P(z_{0:T} | x_{0:T}, θ)

    using dynamic programming in log-space::

        log δ[0, i] = log π[i] + log B[i, x_0]
        log δ[t, j] = max_i(log δ[t-1, i] + log A[i, j]) + log B[j, x_t]
        ψ[t, j]     = argmax_i(log δ[t-1, i] + log A[i, j])

    Backtracking recovers the optimal path from ψ in a single O(T) pass.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        log_trans: Log transition matrix of shape (K, K).
        log_emit: Log emission matrix of shape (K, V).
        log_pi: Log initial distribution of shape (K,).

    Returns:
        Tuple of:
        - ``path``: Most probable state sequence of shape (T,).
        - ``log_prob``: Log-probability of the returned path.
    """
    n_obs = len(obs)
    n_hidden = log_pi.shape[0]
    log_delta = np.empty((n_obs, n_hidden), dtype=np.float64)
    psi = np.empty((n_obs, n_hidden), dtype=np.intp)

    log_delta[0] = log_pi + log_emit[:, obs[0]]
    psi[0] = 0  # undefined for t = 0; backtracking starts at T-1

    for t in range(1, n_obs):
        # scores[i, j] = log_delta[t-1, i] + log_trans[i, j]
        scores = log_delta[t - 1, :, np.newaxis] + log_trans  # (K, K)
        psi[t] = np.argmax(scores, axis=0)  # (K,)
        log_delta[t] = scores.max(axis=0) + log_emit[:, obs[t]]

    path = np.empty(n_obs, dtype=np.intp)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, float(log_delta[-1, path[-1]])


# ---------------------------------------------------------------------------
# Baum-Welch EM
# ---------------------------------------------------------------------------


def _init_params(
    n_states: int,
    n_obs_symbols: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Randomly initialise transition, emission, and initial matrices from Dirichlet(1) priors."""
    trans_mat: NDArray[np.float64] = rng.dirichlet(np.ones(n_states), size=n_states)
    emit_mat: NDArray[np.float64] = rng.dirichlet(np.ones(n_obs_symbols), size=n_states)
    pi: NDArray[np.float64] = rng.dirichlet(np.ones(n_states))
    return trans_mat, emit_mat, pi


def _m_step(
    obs: NDArray[np.intp],
    gamma: NDArray[np.float64],
    xi: NDArray[np.float64],
    n_obs_symbols: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """M-step: re-estimate transition, emission, and initial matrices from γ and ξ.

    Uses fully vectorised NumPy operations::

        π_new[i]    = γ[0, i]
        A_new[i, j] = Σ_t ξ[t, i, j]  /  Σ_t γ[t, i]   (t < T-1)
        B_new[k, v] = Σ_{t: x_t=v} γ[t, k]  /  Σ_t γ[t, k]

    An epsilon floor (_EPS) is applied before normalisation to ensure no
    zero entries remain in the updated matrices.

    Args:
        obs: Integer observation sequence of shape (T,).
        gamma: State posterior array of shape (T, K).
        xi: Pairwise transition posterior array of shape (T-1, K, K).
        n_obs_symbols: Observation vocabulary size V.

    Returns:
        Tuple (transition_matrix, emission_matrix, initial_probs).
    """
    n_obs, _ = gamma.shape

    pi = np.maximum(gamma[0], _EPS)
    pi /= pi.sum()

    xi_sum = xi.sum(axis=0)  # (K, K)
    gamma_denom = np.maximum(gamma[:-1].sum(axis=0), _EPS)  # (K,)
    trans_mat = np.maximum(xi_sum / gamma_denom[:, np.newaxis], _EPS)
    trans_mat /= trans_mat.sum(axis=1, keepdims=True)

    # Vectorised emission re-estimation via an indicator matrix.
    obs_indicator = np.zeros((n_obs, n_obs_symbols), dtype=np.float64)
    obs_indicator[np.arange(n_obs), obs] = 1.0
    emit_mat = np.maximum(gamma.T @ obs_indicator, _EPS)  # (K, V)
    emit_mat /= emit_mat.sum(axis=1, keepdims=True)

    return trans_mat, emit_mat, pi


def _run_em(
    obs: NDArray[np.intp],
    trans_mat: NDArray[np.float64],
    emit_mat: NDArray[np.float64],
    pi: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int, bool]:
    """Run Baum-Welch EM from a fixed initialisation until convergence or limit.

    Args:
        obs: Observation sequence of shape (T,).
        trans_mat: Initial transition matrix of shape (K, K).
        emit_mat: Initial emission matrix of shape (K, V).
        pi: Initial state distribution of shape (K,).
        max_iter: Maximum number of EM iterations.
        tol: Convergence threshold on absolute change in log-likelihood.

    Returns:
        Tuple (transition_matrix, emission_matrix, initial_probs,
        log_likelihood, n_iter, converged).
    """
    prev_ll = -math.inf
    n_obs_symbols = emit_mat.shape[1]

    for n_iter in range(1, max_iter + 1):
        log_trans = np.log(np.maximum(trans_mat, _EPS))
        log_emit = np.log(np.maximum(emit_mat, _EPS))
        log_pi = np.log(np.maximum(pi, _EPS))

        gamma, xi, ll = posteriors(obs, log_trans, log_emit, log_pi)
        trans_mat, emit_mat, pi = _m_step(obs, gamma, xi, n_obs_symbols)

        if abs(ll - prev_ll) < tol:
            return trans_mat, emit_mat, pi, ll, n_iter, True
        prev_ll = ll

    return trans_mat, emit_mat, pi, prev_ll, max_iter, False


def fit(
    obs: NDArray[np.intp],
    n_states: int,
    n_obs_symbols: int,
    *,
    n_init: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> HMMFit:
    """Fit a discrete-emission HMM via Baum-Welch expectation-maximisation.

    Runs the EM algorithm from ``n_init`` independent random initialisations
    and returns the result with the highest converged log-likelihood, guarding
    against convergence to poor local optima.

    Each initialisation draws the transition matrix, emission matrix, and
    initial distribution from symmetric Dirichlet(1) priors.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
            Requires T > 1 so that at least one transition can be estimated.
        n_states: Number of hidden states K.
        n_obs_symbols: Observation vocabulary size V.
        n_init: Number of random restarts.  More restarts reduce the risk of
            poor local optima but increase runtime linearly.
        max_iter: Maximum EM iterations per restart.
        tol: Convergence criterion on absolute change in log-likelihood.
        rng: NumPy random generator; ``np.random.default_rng()`` if None.

    Returns:
        :class:`~qufin.markov.HMMFit` from the best (highest log-likelihood) restart.

    Raises:
        ValueError: If ``len(obs) <= 1`` or ``n_states < 1``.
    """
    if len(obs) <= 1:
        raise ValueError(f"observation sequence must have length > 1, got {len(obs)}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if rng is None:
        rng = np.random.default_rng()

    best_ll = -math.inf
    best: HMMFit | None = None

    for _ in range(n_init):
        trans_mat, emit_mat, pi = _init_params(n_states, n_obs_symbols, rng)
        trans_mat, emit_mat, pi, ll, n_iter, converged = _run_em(
            obs, trans_mat, emit_mat, pi, max_iter, tol
        )
        if ll > best_ll:
            best_ll = ll
            best = HMMFit(
                transition_matrix=trans_mat,
                emission_matrix=emit_mat,
                initial_probs=pi,
                n_states=n_states,
                n_obs_symbols=n_obs_symbols,
                log_likelihood=ll,
                n_iter=n_iter,
                converged=converged,
            )

    assert best is not None  # n_init >= 1 guarantees at least one run
    return best


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def decode(
    obs: NDArray[np.intp],
    model: HMMFit,
) -> NDArray[np.intp]:
    """Decode the most probable hidden-state sequence for a fitted HMM.

    Convenience wrapper around :func:`viterbi` that accepts a :class:`HMMFit`
    object instead of raw log-probability arrays.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        model: Fitted HMM parameters as returned by :func:`fit`.

    Returns:
        Most probable state sequence of shape (T,).
    """
    log_trans = np.log(np.maximum(model.transition_matrix, _EPS))
    log_emit = np.log(np.maximum(model.emission_matrix, _EPS))
    log_pi = np.log(np.maximum(model.initial_probs, _EPS))
    path, _ = viterbi(obs, log_trans, log_emit, log_pi)
    return path


def score(
    obs: NDArray[np.intp],
    model: HMMFit,
) -> float:
    """Compute the log-likelihood of an observation sequence under a fitted HMM.

    Convenience wrapper around :func:`forward_log` that accepts a
    :class:`HMMFit` object.

    Args:
        obs: Integer observation sequence of shape (T,), values in [0, V).
        model: Fitted HMM parameters as returned by :func:`fit`.

    Returns:
        log P(obs | model) as a non-positive float.
    """
    log_trans = np.log(np.maximum(model.transition_matrix, _EPS))
    log_emit = np.log(np.maximum(model.emission_matrix, _EPS))
    log_pi = np.log(np.maximum(model.initial_probs, _EPS))
    _, ll = forward_log(obs, log_trans, log_emit, log_pi)
    return ll


def simulate(
    n_steps: int,
    transition_matrix: NDArray[np.float64],
    emission_matrix: NDArray[np.float64],
    initial_probs: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Forward-sample hidden states and observations from an HMM.

    Samples the initial hidden state from ``initial_probs``, then alternates
    between sampling a next hidden state from ``transition_matrix`` and sampling
    an observation from the corresponding row of ``emission_matrix``.

    Args:
        n_steps: Total number of time steps (sequence length).
        transition_matrix: Row-stochastic matrix of shape (K, K).
        emission_matrix: Row-stochastic matrix of shape (K, V).
        initial_probs: Initial hidden-state distribution of shape (K,).
        rng: NumPy random generator; ``np.random.default_rng()`` if None.

    Returns:
        Tuple of:
        - ``states``: Hidden-state sequence of shape (n_steps,).
        - ``observations``: Observation sequence of shape (n_steps,).
    """
    if rng is None:
        rng = np.random.default_rng()

    cdf_trans = np.cumsum(transition_matrix, axis=1)
    cdf_emit = np.cumsum(emission_matrix, axis=1)
    cdf_pi = np.cumsum(initial_probs)

    states = np.empty(n_steps, dtype=np.intp)
    observations = np.empty(n_steps, dtype=np.intp)
    us_state = rng.random(n_steps)
    us_obs = rng.random(n_steps)

    state = int(np.searchsorted(cdf_pi, us_state[0]))
    for t in range(n_steps):
        if t > 0:
            state = int(np.searchsorted(cdf_trans[state], us_state[t]))
        states[t] = state
        observations[t] = int(np.searchsorted(cdf_emit[state], us_obs[t]))

    return states, observations
