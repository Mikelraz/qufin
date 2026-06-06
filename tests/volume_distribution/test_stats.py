"""Volume-distribution shape statistics."""

from __future__ import annotations

import numpy as np

from qufin.volume_distribution import (
    VolumeProfile,
    classify_profile_shape,
    volume_concentration,
    volume_entropy,
)


def _profile(volume: np.ndarray) -> VolumeProfile:
    n = volume.shape[0]
    edges = np.linspace(0.0, float(n), n + 1)
    poc_idx = int(np.argmax(volume))
    return VolumeProfile(
        price_bins=edges,
        volume=volume.astype(np.float64),
        poc=0.5 * (edges[poc_idx] + edges[poc_idx + 1]),
        vah=float(edges[-1]),
        val=float(edges[0]),
        hvn_idx=np.zeros(0, dtype=np.int64),
        lvn_idx=np.zeros(0, dtype=np.int64),
    )


def test_gini_zero_for_uniform_and_high_for_single_bin() -> None:
    uniform = _profile(np.ones(50))
    assert abs(volume_concentration(uniform)) < 1e-9
    spike = _profile(np.concatenate([np.zeros(49), [100.0]]))
    assert volume_concentration(spike) > 0.9


def test_entropy_max_for_uniform_zero_for_single_bin() -> None:
    uniform = _profile(np.ones(64))
    assert volume_entropy(uniform) == 1.0
    spike = _profile(np.concatenate([np.zeros(63), [1.0]]))
    assert volume_entropy(spike) == 0.0


def test_classify_bundles_all_stats() -> None:
    rng = np.random.default_rng(0)
    centres = np.arange(50)
    volume = np.exp(-0.5 * ((centres - 25) / 5.0) ** 2) + rng.uniform(0, 0.01, size=50)
    stats = classify_profile_shape(_profile(volume))
    assert 0.0 <= stats.entropy <= 1.0
    assert 0.0 <= stats.gini <= 1.0
    assert stats.shape in {"normal", "b", "p", "D"}
    # A symmetric bell centred mid-profile is a balanced "D" day.
    assert stats.shape == "D"
