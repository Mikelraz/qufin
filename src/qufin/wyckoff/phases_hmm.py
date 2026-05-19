"""
HMM-based Wyckoff phase classification.

A Gaussian HMM is fitted on a small feature vector summarising each bar's
*return*, *log-volume*, and *range* anomaly. After Baum-Welch the latent
states are re-labelled to the four canonical Wyckoff macro-phases
*Accumulation*, *Markup*, *Distribution*, *Markdown* by sorting state means
on returns and volume:

* Largest positive mean-return state  →  Markup.
* Largest negative mean-return state  →  Markdown.
* Of the two remaining (small-mean) states, the higher mean log-volume one
  is *Accumulation* and the lower is *Distribution*.

This is intentionally a coarse macro classifier: the rule-based
:func:`qufin.wyckoff.phases.classify_phases` resolves the within-range A-E
labels; the HMM here gives a probabilistic regime overlay across the full
series.

The implementation lazily imports :mod:`qufin.markov.ghmm`, so users that do
not need the HMM path pay no import cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._types import OHLCV
from .bars import rolling_zscore, true_range

MacroPhase = Literal["Accumulation", "Markup", "Distribution", "Markdown"]
_MACRO_ORDER: tuple[MacroPhase, ...] = (
    "Accumulation",
    "Markup",
    "Distribution",
    "Markdown",
)


@dataclass(slots=True)
class HMMPhaseResult:
    """
    Output of :class:`WyckoffHMMClassifier`.

    Attributes
    ----------
    states        Viterbi path over the bars (integer codes in [0, 3]).
    labels        Macro-phase label per bar (one of ``_MACRO_ORDER``).
    state_to_label
                  Mapping from HMM state code to macro label.
    log_likelihood
                  Fitted model log-likelihood.
    """

    states: np.ndarray
    labels: list[MacroPhase]
    state_to_label: dict[int, MacroPhase]
    log_likelihood: float


@dataclass(slots=True)
class WyckoffHMMClassifier:
    """
    Four-state Gaussian HMM classifier for Wyckoff macro-phases.

    Parameters
    ----------
    feature_window  Rolling window for the volume / range z-scores; default 50.
    n_init          Number of random EM restarts; default 5.
    max_iter        Maximum Baum-Welch iterations per restart; default 200.
    tol             Log-likelihood convergence tolerance; default 1e-6.
    seed            Optional RNG seed for reproducibility.
    """

    feature_window: int = 50
    n_init: int = 5
    max_iter: int = 200
    tol: float = 1e-6
    seed: int | None = None

    def fit_predict(self, bars: OHLCV) -> HMMPhaseResult:
        """Fit the HMM on ``bars`` and return the decoded macro-phase path."""
        # Lazy import — see module docstring.
        from qufin.markov import ghmm

        features = self._features(bars)
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        model = ghmm.fit(
            features,
            n_states=4,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
        )
        states = ghmm.decode(features, model)
        state_to_label = _relabel_states(model.means)
        labels: list[MacroPhase] = [state_to_label[int(s)] for s in states]
        return HMMPhaseResult(
            states=states,
            labels=labels,
            state_to_label=state_to_label,
            log_likelihood=model.log_likelihood,
        )

    def _features(self, bars: OHLCV) -> np.ndarray:
        c = bars.close()
        v = bars.volume()
        log_ret = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12)))
        log_ret[0] = 0.0
        log_vol = np.log(v + 1.0)
        z_vol = rolling_zscore(log_vol, self.feature_window)
        tr = true_range(bars)
        z_rng = rolling_zscore(tr, self.feature_window)
        z_vol = np.where(np.isfinite(z_vol), z_vol, 0.0)
        z_rng = np.where(np.isfinite(z_rng), z_rng, 0.0)
        return np.column_stack([log_ret, z_vol, z_rng]).astype(np.float64)


def _relabel_states(means: np.ndarray) -> dict[int, MacroPhase]:
    """
    Heuristic: relabel HMM states using their (return, log-volume) means.

    The 0-th column of ``means`` is the log-return mean; the 1-st is the
    rolling-volume z-score mean. The largest positive return → Markup; the
    largest negative → Markdown; of the remaining two, the higher-volume one
    is Accumulation, the lower is Distribution.
    """
    n_states = means.shape[0]
    if n_states != 4:
        raise ValueError(f"_relabel_states expects 4 states, got {n_states}")
    ret_mean = means[:, 0]
    vol_mean = means[:, 1] if means.shape[1] > 1 else np.zeros(n_states)
    markup = int(np.argmax(ret_mean))
    markdown = int(np.argmin(ret_mean))
    remaining = [s for s in range(n_states) if s not in (markup, markdown)]
    if len(remaining) != 2:
        # Pathological tie — fall back to argsort.
        order = np.argsort(ret_mean)
        return {
            int(order[0]): "Markdown",
            int(order[1]): "Distribution",
            int(order[2]): "Accumulation",
            int(order[3]): "Markup",
        }
    if vol_mean[remaining[0]] >= vol_mean[remaining[1]]:
        accumulation, distribution = remaining[0], remaining[1]
    else:
        accumulation, distribution = remaining[1], remaining[0]
    return {
        markup: "Markup",
        markdown: "Markdown",
        accumulation: "Accumulation",
        distribution: "Distribution",
    }
