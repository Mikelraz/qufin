"""Tests for qufin.markov.hmm (Hidden Markov Model)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from qufin.markov.hmm import (
    _EPS,
    backward_log,
    decode,
    fit,
    forward_log,
    posteriors,
    score,
    simulate,
    viterbi,
)

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Small deterministic HMM fixture (K=2 states, V=3 symbols)
# ---------------------------------------------------------------------------

_A = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
_B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
_PI = np.array([0.6, 0.4], dtype=np.float64)
_LOG_A = np.log(_A)
_LOG_B = np.log(_B)
_LOG_PI = np.log(_PI)
_OBS = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.intp)


# ---------------------------------------------------------------------------
# forward_log
# ---------------------------------------------------------------------------


def test_forward_log_alpha_shape() -> None:
    log_alpha, _ = forward_log(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert log_alpha.shape == (len(_OBS), 2)


def test_forward_log_likelihood_finite() -> None:
    _, ll = forward_log(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert math.isfinite(ll)
    assert ll <= 0.0


def test_forward_log_single_obs() -> None:
    obs = np.array([1], dtype=np.intp)
    log_alpha, ll = forward_log(obs, _LOG_A, _LOG_B, _LOG_PI)
    assert log_alpha.shape == (1, 2)
    assert math.isfinite(ll)


def test_forward_log_likelihood_closed_form() -> None:
    # Single-observation log P(x_0 = 0) = log Σ_i π[i] * B[i, 0]
    obs = np.array([0], dtype=np.intp)
    _, ll = forward_log(obs, _LOG_A, _LOG_B, _LOG_PI)
    expected = math.log((_PI * _B[:, 0]).sum())
    assert abs(ll - expected) < 1e-10


# ---------------------------------------------------------------------------
# backward_log
# ---------------------------------------------------------------------------


def test_backward_log_shape() -> None:
    log_beta = backward_log(_OBS, _LOG_A, _LOG_B)
    assert log_beta.shape == (len(_OBS), 2)


def test_backward_log_terminal_zeros() -> None:
    log_beta = backward_log(_OBS, _LOG_A, _LOG_B)
    np.testing.assert_allclose(log_beta[-1], np.zeros(2), atol=1e-12)


def test_forward_backward_likelihood_consistent() -> None:
    # Both algorithms must agree on log P(observations).
    _, ll_fwd = forward_log(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    log_alpha, _ = forward_log(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    log_beta = backward_log(_OBS, _LOG_A, _LOG_B)
    # At t = 0: log P(O) = logsumexp(log_alpha[0] + log_beta[0] + log correction)
    # Simpler: at any t, P(O) = Σ_i alpha[t, i] * beta[t, i]
    t = 3
    ll_check = float(np.logaddexp.reduce(log_alpha[t] + log_beta[t]))
    assert abs(ll_fwd - ll_check) < 1e-8


# ---------------------------------------------------------------------------
# posteriors
# ---------------------------------------------------------------------------


def test_posteriors_gamma_shape() -> None:
    gamma, xi, _ = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert gamma.shape == (len(_OBS), 2)


def test_posteriors_xi_shape() -> None:
    gamma, xi, _ = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert xi.shape == (len(_OBS) - 1, 2, 2)


def test_posteriors_gamma_rows_sum_to_one() -> None:
    gamma, _, _ = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(len(_OBS)), atol=1e-10)


def test_posteriors_xi_rows_sum_to_one() -> None:
    _, xi, _ = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    # For each t, xi[t, :, :] should sum to 1.
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), np.ones(len(_OBS) - 1), atol=1e-10)


def test_posteriors_gamma_xi_consistency() -> None:
    # gamma[t, i] = Σ_j xi[t, i, j] for t < T-1.
    gamma, xi, _ = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    gamma_from_xi = xi.sum(axis=2)  # (T-1, K)
    np.testing.assert_allclose(gamma[:-1], gamma_from_xi, atol=1e-10)


def test_posteriors_likelihood_matches_forward() -> None:
    _, ll_fwd = forward_log(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    _, _, ll_post = posteriors(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert abs(ll_fwd - ll_post) < 1e-10


# ---------------------------------------------------------------------------
# viterbi
# ---------------------------------------------------------------------------


def test_viterbi_path_length() -> None:
    path, _ = viterbi(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert path.shape == (len(_OBS),)


def test_viterbi_states_in_range() -> None:
    path, _ = viterbi(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert int(path.min()) >= 0
    assert int(path.max()) <= 1


def test_viterbi_log_prob_finite_and_negative() -> None:
    _, log_prob = viterbi(_OBS, _LOG_A, _LOG_B, _LOG_PI)
    assert math.isfinite(log_prob)
    assert log_prob <= 0.0


def test_viterbi_single_obs() -> None:
    obs = np.array([2], dtype=np.intp)
    path, log_prob = viterbi(obs, _LOG_A, _LOG_B, _LOG_PI)
    # Most probable state for x=2 should be state 1 (B[1,2]=0.6 vs B[0,2]=0.1)
    assert int(path[0]) == 1
    assert math.isfinite(log_prob)


# ---------------------------------------------------------------------------
# fit (Baum-Welch)
# ---------------------------------------------------------------------------


def _generate_obs(n_steps: int, seed: int = 1) -> NDArray[np.intp]:
    _, obs = simulate(n_steps, _A, _B, _PI, rng=np.random.default_rng(seed))
    return obs


def test_fit_returns_hmm_fit() -> None:
    obs = _generate_obs(200)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=50)
    assert model.n_states == 2
    assert model.n_obs_symbols == 3
    assert model.transition_matrix.shape == (2, 2)
    assert model.emission_matrix.shape == (2, 3)
    assert model.initial_probs.shape == (2,)


def test_fit_matrices_are_stochastic() -> None:
    obs = _generate_obs(300)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=100)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), np.ones(2), atol=1e-10)
    np.testing.assert_allclose(model.emission_matrix.sum(axis=1), np.ones(2), atol=1e-10)
    assert abs(model.initial_probs.sum() - 1.0) < 1e-10


def test_fit_log_likelihood_non_positive() -> None:
    obs = _generate_obs(200)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=100)
    assert model.log_likelihood <= 0.0


def test_fit_raises_on_short_sequence() -> None:
    obs = np.array([0], dtype=np.intp)
    with pytest.raises(ValueError, match="length > 1"):
        fit(obs, n_states=2, n_obs_symbols=3)


def test_fit_raises_on_zero_states() -> None:
    obs = _generate_obs(50)
    with pytest.raises(ValueError, match="n_states must be"):
        fit(obs, n_states=0, n_obs_symbols=3)


# ---------------------------------------------------------------------------
# decode (Viterbi convenience wrapper)
# ---------------------------------------------------------------------------


def test_decode_length() -> None:
    obs = _generate_obs(100)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=50)
    path = decode(obs, model)
    assert path.shape == (len(obs),)


def test_decode_states_in_range() -> None:
    obs = _generate_obs(100)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=50)
    path = decode(obs, model)
    assert int(path.min()) >= 0
    assert int(path.max()) <= 1


# ---------------------------------------------------------------------------
# score (forward log-likelihood wrapper)
# ---------------------------------------------------------------------------


def test_score_matches_forward_log() -> None:
    obs = _generate_obs(50)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=50)
    log_a = np.log(np.maximum(model.transition_matrix, _EPS))
    log_b = np.log(np.maximum(model.emission_matrix, _EPS))
    log_pi = np.log(np.maximum(model.initial_probs, _EPS))
    _, ll_direct = forward_log(obs, log_a, log_b, log_pi)
    assert abs(score(obs, model) - ll_direct) < 1e-10


def test_score_non_positive() -> None:
    obs = _generate_obs(100)
    model = fit(obs, n_states=2, n_obs_symbols=3, n_init=2, max_iter=50)
    assert score(obs, model) <= 0.0


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def test_simulate_output_shapes() -> None:
    states, obs = simulate(100, _A, _B, _PI, rng=np.random.default_rng(0))
    assert states.shape == (100,)
    assert obs.shape == (100,)


def test_simulate_states_in_range() -> None:
    states, _ = simulate(500, _A, _B, _PI, rng=np.random.default_rng(0))
    assert int(states.min()) >= 0
    assert int(states.max()) <= 1


def test_simulate_obs_in_range() -> None:
    _, obs = simulate(500, _A, _B, _PI, rng=np.random.default_rng(0))
    assert int(obs.min()) >= 0
    assert int(obs.max()) <= 2
