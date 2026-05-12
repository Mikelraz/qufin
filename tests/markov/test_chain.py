"""Tests for qufin.markov.chain (first-order Markov chain)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from qufin.markov.chain import (
    estimate_transition_matrix,
    fit,
    log_likelihood,
    n_step_matrix,
    simulate,
    stationary_distribution,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# estimate_transition_matrix
# ---------------------------------------------------------------------------


def test_transition_matrix_shape() -> None:
    seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    assert T.shape == (3, 3)


def test_transition_matrix_row_stochastic() -> None:
    seq = np.array([0, 1, 0, 2, 1, 0, 2, 1, 2, 0], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(3), atol=1e-12)


def test_transition_matrix_known_sequence() -> None:
    # Sequence always cycles 0 → 1 → 2 → 0 → …
    seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    np.testing.assert_allclose(T, expected, atol=1e-12)


def test_transition_matrix_unseen_state_uniform() -> None:
    # State 2 never appears → its row should be uniform.
    seq = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    np.testing.assert_allclose(T[2], np.full(3, 1 / 3), atol=1e-12)


# ---------------------------------------------------------------------------
# stationary_distribution
# ---------------------------------------------------------------------------


def test_stationary_invariant() -> None:
    seq = RNG.integers(0, 4, size=500, dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=4)
    pi = stationary_distribution(T)
    np.testing.assert_allclose(pi @ T, pi, atol=1e-8)


def test_stationary_sums_to_one() -> None:
    T = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    pi = stationary_distribution(T)
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.all(pi >= 0.0)


def test_stationary_two_state_analytical() -> None:
    # For 2-state chain with p = P(0→1) and q = P(1→0):
    # stationary = [q/(p+q), p/(p+q)]
    p, q = 0.3, 0.5
    T = np.array([[1 - p, p], [q, 1 - q]], dtype=np.float64)
    pi = stationary_distribution(T)
    expected = np.array([q / (p + q), p / (p + q)])
    np.testing.assert_allclose(pi, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# n_step_matrix
# ---------------------------------------------------------------------------


def test_n_step_matrix_zero_is_identity() -> None:
    T = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
    result = n_step_matrix(T, 0)
    np.testing.assert_allclose(result, np.eye(2), atol=1e-12)


def test_n_step_matrix_one_is_original() -> None:
    T = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
    np.testing.assert_allclose(n_step_matrix(T, 1), T, atol=1e-12)


def test_n_step_matrix_vs_naive() -> None:
    T = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.3, 0.3, 0.4]], dtype=np.float64)
    naive = T @ T @ T @ T @ T  # T^5
    fast = n_step_matrix(T, 5)
    np.testing.assert_allclose(fast, naive, atol=1e-12)


def test_n_step_matrix_negative_raises() -> None:
    T = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        n_step_matrix(T, -1)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def test_simulate_length() -> None:
    T = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    traj = simulate(T, initial_state=0, n_steps=50, rng=np.random.default_rng(0))
    assert traj.shape == (51,)


def test_simulate_starts_at_initial_state() -> None:
    T = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    traj = simulate(T, initial_state=1, n_steps=10, rng=np.random.default_rng(0))
    assert int(traj[0]) == 1


def test_simulate_states_in_range() -> None:
    T = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=np.float64)
    traj = simulate(T, initial_state=0, n_steps=200, rng=np.random.default_rng(0))
    assert int(traj.min()) >= 0
    assert int(traj.max()) <= 2


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------


def test_log_likelihood_non_positive() -> None:
    seq = np.array([0, 1, 0, 1, 2, 0], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    ll = log_likelihood(seq, T)
    assert ll <= 0.0


def test_log_likelihood_with_initial_probs() -> None:
    seq = np.array([0, 1, 2, 0], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    pi = np.array([1.0, 0.0, 0.0])  # deterministic start at 0
    ll = log_likelihood(seq, T, initial_probs=pi)
    assert ll <= 0.0


def test_log_likelihood_perfect_chain() -> None:
    # Deterministic cycle 0→1→2→0: log-likelihood from transitions = log(1)*T = 0
    seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    T = estimate_transition_matrix(seq, n_states=3)
    pi = np.array([1.0, 0.0, 0.0])
    ll = log_likelihood(seq, T, initial_probs=pi)
    # Only transition contributions: each of the 8 transitions has prob 1
    assert abs(ll) < 1e-9


# ---------------------------------------------------------------------------
# fit (integration)
# ---------------------------------------------------------------------------


def test_fit_returns_chain_fit() -> None:
    seq = RNG.integers(0, 3, size=200, dtype=np.intp)
    result = fit(seq, n_states=3)
    assert result.n_states == 3
    assert result.transition_matrix.shape == (3, 3)
    assert result.stationary.shape == (3,)
    assert result.log_likelihood <= 0.0


def test_fit_stationary_invariant() -> None:
    seq = RNG.integers(0, 4, size=500, dtype=np.intp)
    result = fit(seq, n_states=4)
    np.testing.assert_allclose(
        result.stationary @ result.transition_matrix,
        result.stationary,
        atol=1e-8,
    )
