"""Nth-order Markov chain estimation and simulation.

An order-N chain conditions on the last N states simultaneously::

    P(X_t | X_{t-1}, X_{t-2}, ..., X_{t-N})

The model is stored as a conditional probability tensor of shape
``(S,) * (N + 1)`` where the first N axes index the conditioning context
(oldest to newest) and the final axis indexes the next state.

The embedding of a higher-order chain into an equivalent first-order chain
over the augmented state space S^N is provided via :func:`to_first_order`,
making the full first-order toolkit (stationary distribution, n-step matrix,
etc.) directly applicable.

State convention
----------------
States are non-negative integers in [0, S).  Context tuples are ordered
oldest-first, matching the axis order of the transition tensor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qufin.markov._types import HigherOrderFit

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------


def estimate_transition_tensor(
    sequence: NDArray[np.intp],
    n_states: int,
    order: int,
) -> NDArray[np.float64]:
    """Maximum-likelihood estimate of the Nth-order transition tensor.

    Scans the sequence with a sliding window of length ``order + 1``, counts
    each (context, next_state) occurrence, and normalises over the next-state
    axis.  Contexts that are never observed receive a uniform conditional
    distribution so the result is always a valid conditional probability tensor.

    The sliding window construction uses :func:`numpy.lib.stride_tricks.sliding_window_view`
    so the counting loop is replaced by a fully vectorised ``np.add.at`` call.

    Args:
        sequence: 1-D integer array of observed states of shape (T,).
            All values must be in [0, n_states).  T must be > order.
        n_states: Number of distinct states S.
        order: Markov order N (positive integer).

    Returns:
        Conditional probability tensor of shape ``(S,) * (N + 1)``.
        The last axis indexes the next state: ``tensor[s_{t-N}, ..., s_{t-1}, j]``
        = P(X_t = j | context).

    Raises:
        ValueError: If ``order < 1`` or ``len(sequence) <= order``.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    seq_len = len(sequence)
    if seq_len <= order:
        raise ValueError(f"sequence length {seq_len} must be greater than order {order}")

    shape = (n_states,) * (order + 1)
    counts = np.zeros(shape, dtype=np.float64)

    # All overlapping windows of length order+1, shape (T - order, order + 1).
    windows = np.lib.stride_tricks.sliding_window_view(sequence, order + 1)
    idx = tuple(windows[:, k] for k in range(order + 1))
    np.add.at(counts, idx, 1.0)

    denom = counts.sum(axis=-1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return counts / denom


def to_first_order(
    transition_tensor: NDArray[np.float64],
    n_states: int,
    order: int,
) -> NDArray[np.float64]:
    """Embed an Nth-order chain into an equivalent first-order chain.

    The augmented state space consists of all N-tuples of base states indexed
    by a flat integer using C (row-major) order::

        index(s_1, s_2, ..., s_N) = s_1 · S^{N-1} + s_2 · S^{N-2} + … + s_N

    The resulting matrix has shape ``(S^N, S^N)`` and is row-stochastic.
    The only non-zero transitions from context ``(s_1, …, s_N)`` lead to
    ``(s_2, …, s_N, j)`` with probability ``P(j | s_1, …, s_N)``, corresponding
    to shifting the context window forward by one step.

    The computation is fully vectorised: all augmented states are enumerated as
    index arrays and filled with a single advanced-indexing assignment.

    Args:
        transition_tensor: Tensor of shape ``(S,) * (N + 1)`` as returned by
            :func:`estimate_transition_tensor`.
        n_states: Number of base states S.
        order: Markov order N.

    Returns:
        Row-stochastic first-order transition matrix of shape (S^N, S^N).
    """
    aug_s = n_states**order

    # All augmented states as N-tuples, shape (aug_s, N).
    contexts: NDArray[np.intp] = np.array(
        np.unravel_index(np.arange(aug_s), (n_states,) * order),
        dtype=np.intp,
    ).T

    # Reshape tensor to (aug_s, S): row k holds P(next | unravel_index(k, ...)).
    # C-order reshape aligns exactly with ravel_multi_index in C order.
    probs = transition_tensor.reshape(aug_s, n_states)  # (aug_s, S)

    # New augmented state for each (source, next_state) pair, shape (aug_s, S, N).
    new_ctx = np.empty((aug_s, n_states, order), dtype=np.intp)
    if order > 1:
        new_ctx[:, :, :-1] = contexts[:, 1:][:, np.newaxis, :]  # drop oldest
    new_ctx[:, :, -1] = np.arange(n_states)[np.newaxis, :]  # append next state

    new_idx = np.ravel_multi_index(
        tuple(new_ctx[:, :, k].ravel() for k in range(order)),
        (n_states,) * order,
    ).reshape(aug_s, n_states)

    t_aug = np.zeros((aug_s, aug_s), dtype=np.float64)
    t_aug[np.arange(aug_s)[:, np.newaxis], new_idx] = probs
    return t_aug


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    transition_tensor: NDArray[np.float64],
    initial_context: NDArray[np.intp],
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Simulate an Nth-order Markov chain by forward sampling.

    The returned trajectory starts with the ``initial_context`` seed and
    appends ``n_steps`` new states, giving total length ``N + n_steps``.

    Args:
        transition_tensor: Tensor of shape ``(S,) * (N + 1)`` as returned by
            :func:`estimate_transition_tensor`.
        initial_context: Seed state sequence of length N (the Markov order).
            Values must be in [0, S).
        n_steps: Number of new states to generate after the seed context.
        rng: NumPy random generator; ``np.random.default_rng()`` if None.

    Returns:
        Integer state sequence of shape (N + n_steps,).
    """
    if rng is None:
        rng = np.random.default_rng()

    order = len(initial_context)
    trajectory = np.empty(order + n_steps, dtype=np.intp)
    trajectory[:order] = initial_context
    # Pre-compute CDFs for every context slice encountered during sampling.
    us: NDArray[np.float64] = rng.random(n_steps)
    context = list(map(int, initial_context))

    for t in range(n_steps):
        probs: NDArray[np.float64] = transition_tensor[tuple(context)]
        cdf = np.cumsum(probs)
        next_state = int(np.searchsorted(cdf, us[t]))
        trajectory[order + t] = next_state
        context.pop(0)
        context.append(next_state)

    return trajectory


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def log_likelihood(
    sequence: NDArray[np.intp],
    transition_tensor: NDArray[np.float64],
    order: int,
) -> float:
    """Log-likelihood of an observed sequence under an Nth-order chain.

    The first ``order`` observations serve as the initial conditioning context
    and are not included in the likelihood sum::

        log P(x_N, x_{N+1}, ..., x_T | x_0, ..., x_{N-1}, model)
            = Σ_{t=N}^{T} log P(x_t | x_{t-N}, ..., x_{t-1})

    All windows are extracted in one vectorised pass using
    :func:`numpy.lib.stride_tricks.sliding_window_view`.

    Args:
        sequence: 1-D integer array of states of shape (T,).
        transition_tensor: Tensor of shape ``(S,) * (N + 1)``.
        order: Markov order N.

    Returns:
        Log-likelihood as a non-positive float.
    """
    log_tensor = np.log(np.maximum(transition_tensor, _EPS))
    windows = np.lib.stride_tricks.sliding_window_view(sequence, order + 1)
    idx = tuple(windows[:, k] for k in range(order + 1))
    return float(np.sum(log_tensor[idx]))


# ---------------------------------------------------------------------------
# Convenience estimator
# ---------------------------------------------------------------------------


def fit(
    sequence: NDArray[np.intp],
    n_states: int,
    order: int,
) -> HigherOrderFit:
    """Fit an Nth-order Markov chain to an observed sequence via MLE.

    Args:
        sequence: 1-D integer array of observed states in [0, n_states).
        n_states: Number of distinct states S.
        order: Markov order N (positive integer).

    Returns:
        :class:`~qufin.markov.HigherOrderFit` with estimated tensor and diagnostics.
    """
    tensor = estimate_transition_tensor(sequence, n_states, order)
    ll = log_likelihood(sequence, tensor, order)
    return HigherOrderFit(
        transition_tensor=tensor,
        order=order,
        n_states=n_states,
        log_likelihood=ll,
    )
