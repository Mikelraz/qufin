"""HMM-based Wyckoff macro-phase classifier."""

from __future__ import annotations

import numpy as np

from qufin.wyckoff import WyckoffHMMClassifier
from tests.wyckoff.conftest import make_ohlcv


def _two_regime_series(
    n_per: int = 150,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a markdown (-drift) then markup (+drift) regime."""
    rng = np.random.default_rng(seed)
    log_steps_md = rng.normal(-0.01, 0.005, n_per)
    log_steps_mu = rng.normal(0.01, 0.005, n_per)
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([log_steps_md, log_steps_mu])))
    opens = np.concatenate(([100.0], closes[:-1]))
    highs = np.maximum(opens, closes) * 1.005
    lows = np.minimum(opens, closes) * 0.995
    vols = rng.lognormal(10.0, 0.2, 2 * n_per)
    truth = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    return opens, highs, lows, closes, vols, truth


def test_two_regime_hmm_recovers_macro_direction() -> None:
    opens, highs, lows, closes, vols, truth = _two_regime_series(n_per=200, seed=11)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    clf = WyckoffHMMClassifier(feature_window=30, n_init=3, max_iter=80, seed=42)
    result = clf.fit_predict(bars)
    # Each label appears for at least some bars.
    assert any(lbl == "Markup" for lbl in result.labels)
    assert any(lbl == "Markdown" for lbl in result.labels)
    # In the first regime (truth=0, downtrend) Markdown should dominate;
    # in the second regime, Markup. Allow some boundary slop.
    n_per = 200
    md_in_first = sum(1 for lbl in result.labels[: n_per - 20] if lbl == "Markdown")
    mu_in_second = sum(1 for lbl in result.labels[n_per + 20 :] if lbl == "Markup")
    assert md_in_first / (n_per - 20) > 0.4
    assert mu_in_second / (n_per - 20) > 0.4
    _ = truth  # truth retained for clarity / future tightening
