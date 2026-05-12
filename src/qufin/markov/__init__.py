"""State-transition modelling: Markov chains and Hidden Markov Models.

This package provides a pure mathematical engine for modelling discrete
state sequences.  All algorithms are fully vectorised via NumPy and operate
in log-space where numerical stability demands it.

Submodules
----------
chain
    First-order Markov chain: MLE estimation, stationary distribution,
    n-step transition matrix, forward simulation, and log-likelihood.
higher_order
    Nth-order Markov chain: MLE estimation via sliding-window counting,
    embedding into a first-order chain, forward simulation, and log-likelihood.
hmm
    Discrete-emission Hidden Markov Model: Baum-Welch EM training, Viterbi
    decoding, forward-backward algorithm, and forward simulation.

Quick start
-----------
First-order chain::

    import numpy as np
    from qufin.markov import fit_chain, simulate_chain

    seq = np.array([0, 1, 0, 2, 1, 0, 1, 2], dtype=np.intp)
    result = fit_chain(seq, n_states=3)
    trajectory = simulate_chain(result.transition_matrix, initial_state=0, n_steps=100)

Nth-order chain::

    from qufin.markov import fit_higher_order, simulate_higher_order

    result = fit_higher_order(seq, n_states=3, order=2)
    trajectory = simulate_higher_order(
        result.transition_tensor,
        initial_context=np.array([0, 1], dtype=np.intp),
        n_steps=100,
    )

Hidden Markov Model::

    from qufin.markov import fit_hmm, decode, score

    model = fit_hmm(seq, n_states=2, n_obs_symbols=3)
    path = decode(seq, model)
    ll   = score(seq, model)
"""

from __future__ import annotations

from qufin.markov._types import ChainFit, HigherOrderFit, HMMFit
from qufin.markov.chain import (
    estimate_transition_matrix,
    n_step_matrix,
    stationary_distribution,
)
from qufin.markov.chain import (
    fit as fit_chain,
)
from qufin.markov.chain import (
    log_likelihood as chain_log_likelihood,
)
from qufin.markov.chain import (
    simulate as simulate_chain,
)
from qufin.markov.higher_order import (
    estimate_transition_tensor,
    to_first_order,
)
from qufin.markov.higher_order import (
    fit as fit_higher_order,
)
from qufin.markov.higher_order import (
    log_likelihood as higher_order_log_likelihood,
)
from qufin.markov.higher_order import (
    simulate as simulate_higher_order,
)
from qufin.markov.hmm import (
    backward_log,
    decode,
    forward_log,
    posteriors,
    score,
    viterbi,
)
from qufin.markov.hmm import (
    fit as fit_hmm,
)
from qufin.markov.hmm import (
    simulate as simulate_hmm,
)

__all__: list[str] = [
    # Result containers
    "ChainFit",
    "HigherOrderFit",
    "HMMFit",
    # First-order chain
    "estimate_transition_matrix",
    "fit_chain",
    "chain_log_likelihood",
    "n_step_matrix",
    "simulate_chain",
    "stationary_distribution",
    # Higher-order chain
    "estimate_transition_tensor",
    "fit_higher_order",
    "higher_order_log_likelihood",
    "simulate_higher_order",
    "to_first_order",
    # HMM — algorithmic primitives
    "forward_log",
    "backward_log",
    "posteriors",
    "viterbi",
    # HMM — convenience API
    "fit_hmm",
    "decode",
    "score",
    "simulate_hmm",
]
