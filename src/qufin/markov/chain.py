"""First-order (standard) Markov chain estimation and simulation.

All operations are fully vectorised using NumPy. No Python-level loops appear
in any hot-path function except the trajectory simulation, where sequential
state dependence is unavoidable.

State convention
----------------
States are non-negative integers in [0, S). Sequences must be encoded as
``np.intp`` (or any integer dtype compatible with advanced indexing).

Numerical convention
--------------------
- Probability matrices are in probability space unless the parameter or return
  name carries a ``log_`` prefix.
- Rows with no observed transitions receive a uniform conditional distribution
  rather than being left as zeros — the result is always a valid stochastic matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qufin.markov._types import ChainFit

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------


def estimate_transition_matrix(
    sequence: NDArray[np.intp],
    n_states: int,
) -> NDArray[np.float64]:
    """Maximum-likelihood estimate of the transition matrix from a sequence.

    Counts all co-occurrences (s_t, s_{t+1}) and normalises each row.  Rows
    with no observed transitions are set to a uniform distribution so the
    result is always a valid row-stochastic matrix.

    Args:
        sequence: 1-D integer array of observed states of shape (T,).
            All values must be in [0, n_states).
        n_states: Number of distinct states S.

    Returns:
        Row-stochastic transition matrix of shape (S, S).
    """
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(counts, (sequence[:-1], sequence[1:]), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    # Rows with no observed transitions receive a uniform count vector so that
    # division yields 1/S rather than 0/0.
    unseen = (row_sums == 0.0).ravel()
    counts[unseen] = 1.0
    row_sums[unseen] = float(n_states)
    return counts / row_sums


def stationary_distribution(
    transition_matrix: NDArray[np.float64],
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """Stationary distribution of a row-stochastic transition matrix.

    Solves ``pi @ T = pi`` with ``sum(pi) = 1`` by finding the left eigenvector
    of T corresponding to eigenvalue 1.  For ergodic chains this eigenvector is
    unique and real-valued.  A power-iteration fallback is applied when the
    eigenvector approach yields complex or negative values (e.g. near-singular
    or periodic chains).

    Args:
        transition_matrix: Row-stochastic matrix of shape (S, S).
        tol: Convergence tolerance for the power-iteration fallback.
        max_iter: Maximum power-iteration steps.

    Returns:
        Stationary distribution of shape (S,), summing to 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = eigenvectors[:, idx].real
    if np.all(pi >= -1e-9):
        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        if total > _EPS:
            return (pi / total).astype(np.float64)

    pi = np.full(len(transition_matrix), 1.0 / len(transition_matrix), dtype=np.float64)
    for _ in range(max_iter):
        pi_new: NDArray[np.float64] = pi @ transition_matrix
        if float(np.max(np.abs(pi_new - pi))) < tol:
            return pi_new
        pi = pi_new
    return pi


def n_step_matrix(transition_matrix: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Compute the n-step transition matrix T^n via repeated squaring.

    Uses matrix exponentiation by squaring in O(S³ log n) time rather than
    the naive O(S³ n) repeated multiplication.

    Args:
        transition_matrix: Row-stochastic matrix of shape (S, S).
        n: Number of steps (non-negative integer).

    Returns:
        T^n of shape (S, S).  For n = 0 returns the identity matrix.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    s = len(transition_matrix)
    if n == 0:
        return np.eye(s, dtype=np.float64)
    result = np.eye(s, dtype=np.float64)
    base = transition_matrix.astype(np.float64, copy=True)
    while n > 0:
        if n & 1:
            result = result @ base
        base = base @ base
        n >>= 1
    return result


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    transition_matrix: NDArray[np.float64],
    initial_state: int,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Simulate a Markov chain trajectory by forward sampling.

    Generates a sequence of length ``n_steps + 1`` (initial state plus
    ``n_steps`` transitions).  Random numbers are pre-drawn as a batch and
    inverse-CDF sampling (``searchsorted`` on the cumulative row) is used to
    minimise Python overhead per step.

    Args:
        transition_matrix: Row-stochastic matrix of shape (S, S).
        initial_state: Starting state index in [0, S).
        n_steps: Number of state transitions to simulate.
        rng: NumPy random generator; ``np.random.default_rng()`` if None.

    Returns:
        Integer state sequence of shape (n_steps + 1,).
    """
    if rng is None:
        rng = np.random.default_rng()
    cdf = np.cumsum(transition_matrix, axis=1)
    us: NDArray[np.float64] = rng.random(n_steps)
    trajectory = np.empty(n_steps + 1, dtype=np.intp)
    trajectory[0] = initial_state
    state = initial_state
    for t in range(n_steps):
        state = int(np.searchsorted(cdf[state], us[t]))
        trajectory[t + 1] = state
    return trajectory


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def log_likelihood(
    sequence: NDArray[np.intp],
    transition_matrix: NDArray[np.float64],
    initial_probs: NDArray[np.float64] | None = None,
) -> float:
    """Log-likelihood of an observed sequence under a first-order chain.

    Computed as::

        log P(x_0, ..., x_T) = log π[x_0] + Σ_t log T[x_t, x_{t+1}]

    where ``π`` is the initial distribution and ``T`` is the transition matrix.

    Args:
        sequence: 1-D integer array of states of shape (T,).
        transition_matrix: Row-stochastic matrix of shape (S, S).
        initial_probs: Initial distribution of shape (S,).  Defaults to the
            stationary distribution of ``transition_matrix``.

    Returns:
        Log-likelihood as a non-positive float.
    """
    if initial_probs is None:
        initial_probs = stationary_distribution(transition_matrix)
    log_trans = np.log(np.maximum(transition_matrix, _EPS))
    log_pi = np.log(np.maximum(initial_probs, _EPS))
    ll = float(log_pi[sequence[0]])
    ll += float(np.sum(log_trans[sequence[:-1], sequence[1:]]))
    return ll


# ---------------------------------------------------------------------------
# Convenience estimator
# ---------------------------------------------------------------------------


def fit(
    sequence: NDArray[np.intp],
    n_states: int,
) -> ChainFit:
    """Fit a first-order Markov chain to an observed sequence via MLE.

    Estimates the transition matrix by maximum likelihood, derives the
    stationary distribution, and evaluates the sequence log-likelihood.

    Args:
        sequence: 1-D integer array of observed states in [0, n_states).
        n_states: Number of distinct states S.

    Returns:
        :class:`~qufin.markov.ChainFit` with estimated model and diagnostics.
    """
    trans_mat = estimate_transition_matrix(sequence, n_states)
    pi = stationary_distribution(trans_mat)
    ll = log_likelihood(sequence, trans_mat, pi)
    return ChainFit(
        transition_matrix=trans_mat,
        stationary=pi,
        n_states=n_states,
        log_likelihood=ll,
    )
