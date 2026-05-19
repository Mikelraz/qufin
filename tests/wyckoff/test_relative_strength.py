"""Relative-strength and RS-rank utilities."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.wyckoff import relative_strength, rs_rank


def test_two_times_benchmark_is_constant_rs() -> None:
    benchmark = np.linspace(100.0, 200.0, 50)
    asset = 2.0 * benchmark
    rs = relative_strength(asset, benchmark, normalize=False)
    assert np.allclose(rs, 2.0)
    rs_norm = relative_strength(asset, benchmark, normalize=True)
    assert np.allclose(rs_norm, 1.0)


def test_rs_rank_outputs_in_unit_interval() -> None:
    n = 200
    rng = np.random.default_rng(0)
    benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    universe: dict[str, np.ndarray] = {}
    for k in range(4):
        drift = -0.001 + 0.001 * k
        universe[f"S{k}"] = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.005, n)))
    ranks = rs_rank(universe, benchmark, window=30)
    assert set(ranks.keys()) == set(universe.keys())
    for sym, r in ranks.items():
        valid = r[np.isfinite(r)]
        assert valid.shape[0] > 0
        assert valid.min() >= 0.0 - 1e-12
        assert valid.max() <= 1.0 + 1e-12
        _ = sym


def test_relative_strength_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        relative_strength(np.ones(10), np.ones(11))


def test_relative_strength_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        relative_strength(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))
