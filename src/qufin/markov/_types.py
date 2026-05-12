"""Shared result containers for Markov chain and HMM computations.

Both dataclasses use ``slots=True`` for reduced memory footprint and faster
attribute access — relevant when many instances are created during Monte Carlo
simulations or multi-restart EM sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ChainFit:
    """Outcome of fitting a first-order Markov chain via MLE.

    Attributes:
        transition_matrix: Row-stochastic transition matrix of shape (S, S).
            Entry ``[i, j]`` = P(X_{t+1} = j | X_t = i).
        stationary: Stationary distribution of shape (S,), satisfying
            ``stationary @ transition_matrix ≈ stationary``.
        n_states: Number of distinct states S.
        log_likelihood: Log-likelihood of the observed sequence under the fit.
    """

    transition_matrix: NDArray[np.float64]  # (S, S)
    stationary: NDArray[np.float64]  # (S,)
    n_states: int
    log_likelihood: float


@dataclass(slots=True)
class HigherOrderFit:
    """Outcome of fitting an Nth-order Markov chain via MLE.

    An order-N chain conditions on the last N states jointly. The model is
    stored as a conditional probability tensor of shape ``(S,) * (N + 1)``
    where the first N axes index the conditioning context (oldest first) and
    the final axis indexes the next state.

    Attributes:
        transition_tensor: Conditional probability tensor of shape ``(S,)*(N+1)``.
            ``transition_tensor[s_{t-N}, ..., s_{t-1}, j]`` = P(X_t = j | context).
        order: Markov order N.
        n_states: Number of distinct states S.
        log_likelihood: Log-likelihood of the observed sequence under the fit.
            The first ``order`` observations are treated as initial context and
            excluded from the likelihood sum.
    """

    transition_tensor: NDArray[np.float64]  # (S,) * (N + 1)
    order: int
    n_states: int
    log_likelihood: float


@dataclass(slots=True)
class HMMFit:
    """Outcome of fitting a discrete-emission Hidden Markov Model via Baum-Welch EM.

    Attributes:
        transition_matrix: Hidden-state transition matrix of shape (K, K).
            Row-stochastic: ``transition_matrix[i, j]`` = P(z_{t+1}=j | z_t=i).
        emission_matrix: Emission probability matrix of shape (K, V).
            Row-stochastic: ``emission_matrix[k, v]`` = P(x_t=v | z_t=k).
        initial_probs: Initial hidden-state distribution of shape (K,).
        n_states: Number of hidden states K.
        n_obs_symbols: Observation vocabulary size V.
        log_likelihood: Final log P(observations | model) after convergence.
        n_iter: Number of EM iterations performed before convergence or limit.
        converged: True if the EM algorithm met the tolerance criterion.
    """

    transition_matrix: NDArray[np.float64]  # (K, K)
    emission_matrix: NDArray[np.float64]  # (K, V)
    initial_probs: NDArray[np.float64]  # (K,)
    n_states: int
    n_obs_symbols: int
    log_likelihood: float
    n_iter: int
    converged: bool
