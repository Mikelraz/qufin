"""Tests for qufin.markov.higher_order (Nth-order Markov chain)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from qufin.markov.chain import estimate_transition_matrix
from qufin.markov.higher_order import (
    estimate_transition_tensor,
    fit,
    log_likelihood,
    simulate,
    to_first_order,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# estimate_transition_tensor
# ---------------------------------------------------------------------------


def test_tensor_shape_order1() -> None:
    seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=1)
    assert tensor.shape == (3, 3)


def test_tensor_shape_order2() -> None:
    seq = np.arange(20, dtype=np.intp) % 4
    tensor = estimate_transition_tensor(seq, n_states=4, order=2)
    assert tensor.shape == (4, 4, 4)


def test_tensor_last_axis_stochastic() -> None:
    seq = RNG.integers(0, 3, size=200, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    # Every slice along the last axis must sum to 1.
    sums = tensor.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones((3, 3)), atol=1e-12)


def test_tensor_order1_matches_chain() -> None:
    seq = RNG.integers(0, 4, size=300, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=4, order=1)
    chain_T = estimate_transition_matrix(seq, n_states=4)
    np.testing.assert_allclose(tensor, chain_T, atol=1e-12)


def test_tensor_raises_on_order_zero() -> None:
    seq = np.array([0, 1, 2, 0], dtype=np.intp)
    with pytest.raises(ValueError, match="order must be >= 1"):
        estimate_transition_tensor(seq, n_states=3, order=0)


def test_tensor_raises_when_seq_too_short() -> None:
    seq = np.array([0, 1], dtype=np.intp)
    with pytest.raises(ValueError, match="greater than order"):
        estimate_transition_tensor(seq, n_states=3, order=2)


def test_tensor_deterministic_context() -> None:
    # Perfect cycle 0→1→2→0: with order=2 the context (0,1) always leads to 2.
    seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    assert abs(float(tensor[0, 1, 2]) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# to_first_order
# ---------------------------------------------------------------------------


def test_to_first_order_shape() -> None:
    tensor = np.ones((3, 3, 3), dtype=np.float64) / 3
    T_aug = to_first_order(tensor, n_states=3, order=2)
    assert T_aug.shape == (9, 9)


def test_to_first_order_row_stochastic() -> None:
    seq = RNG.integers(0, 3, size=300, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    T_aug = to_first_order(tensor, n_states=3, order=2)
    np.testing.assert_allclose(T_aug.sum(axis=1), np.ones(9), atol=1e-12)


def test_to_first_order_order1_matches_chain() -> None:
    seq = RNG.integers(0, 4, size=300, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=4, order=1)
    T_aug = to_first_order(tensor, n_states=4, order=1)
    chain_T = estimate_transition_matrix(seq, n_states=4)
    np.testing.assert_allclose(T_aug, chain_T, atol=1e-12)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def test_simulate_length() -> None:
    seq = RNG.integers(0, 3, size=100, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    ctx = np.array([0, 1], dtype=np.intp)
    traj = simulate(tensor, ctx, n_steps=50, rng=np.random.default_rng(0))
    assert traj.shape == (52,)  # order + n_steps = 2 + 50


def test_simulate_preserves_initial_context() -> None:
    seq = RNG.integers(0, 3, size=100, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    ctx = np.array([1, 2], dtype=np.intp)
    traj = simulate(tensor, ctx, n_steps=20, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(traj[:2], ctx)


def test_simulate_states_in_range() -> None:
    seq = RNG.integers(0, 4, size=300, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=4, order=3)
    ctx = np.array([0, 1, 2], dtype=np.intp)
    traj = simulate(tensor, ctx, n_steps=200, rng=np.random.default_rng(0))
    assert int(traj.min()) >= 0
    assert int(traj.max()) <= 3


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------


def test_log_likelihood_non_positive() -> None:
    seq = RNG.integers(0, 3, size=100, dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    ll = log_likelihood(seq, tensor, order=2)
    assert ll <= 0.0


def test_log_likelihood_perfect_order2() -> None:
    # Deterministic cycle: context (0,1)→2, (1,2)→0, (2,0)→1.
    seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    tensor = estimate_transition_tensor(seq, n_states=3, order=2)
    ll = log_likelihood(seq, tensor, order=2)
    # All conditional probs are 1 → log-likelihood = 0.
    assert abs(ll) < 1e-9


# ---------------------------------------------------------------------------
# fit (integration)
# ---------------------------------------------------------------------------


def test_fit_returns_higher_order_fit() -> None:
    seq = RNG.integers(0, 3, size=200, dtype=np.intp)
    result = fit(seq, n_states=3, order=2)
    assert result.order == 2
    assert result.n_states == 3
    assert result.transition_tensor.shape == (3, 3, 3)
    assert result.log_likelihood <= 0.0


def test_fit_tensor_last_axis_stochastic() -> None:
    seq = RNG.integers(0, 3, size=300, dtype=np.intp)
    result = fit(seq, n_states=3, order=3)
    sums = result.transition_tensor.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones((3, 3, 3)), atol=1e-12)
