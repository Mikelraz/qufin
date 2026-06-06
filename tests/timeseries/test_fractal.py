"""Hurst exponent (R/S, DFA, aggregated variance) and fractal dimension."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import toeplitz

from qufin.timeseries import (
    HurstResult,
    MFDFAResult,
    aggregated_variance,
    dfa,
    fractal_dimension,
    hurst,
    mfdfa,
    rs_analysis,
)


def fgn(n: int, h: float, seed: int) -> np.ndarray:
    """Exact fractional Gaussian noise via Cholesky of the fGn covariance."""
    rng = np.random.default_rng(seed)
    k = np.arange(n, dtype=np.float64)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    chol = np.linalg.cholesky(toeplitz(gamma))
    return chol @ rng.standard_normal(n)


def _mean_hurst(estimator, h: float, *, n: int = 2000, seeds: int = 3) -> float:
    return float(np.mean([estimator(fgn(n, h, s)).exponent for s in range(seeds)]))


def test_rs_white_noise_near_half() -> None:
    rng = np.random.default_rng(0)
    h = rs_analysis(rng.standard_normal(4000)).exponent
    assert h == pytest.approx(0.5, abs=0.08)


def test_rs_recovers_persistent_and_antipersistent() -> None:
    assert _mean_hurst(rs_analysis, 0.75) == pytest.approx(0.75, abs=0.10)
    assert _mean_hurst(rs_analysis, 0.30) == pytest.approx(0.30, abs=0.10)


def test_hurst_ordering_is_monotonic() -> None:
    anti = _mean_hurst(rs_analysis, 0.30)
    rw = _mean_hurst(rs_analysis, 0.50)
    pers = _mean_hurst(rs_analysis, 0.70)
    assert anti < rw < pers


def test_dfa_alpha_noise_vs_random_walk() -> None:
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(4000)
    assert dfa(noise).exponent == pytest.approx(0.5, abs=0.08)
    # Integrating the noise raises the DFA exponent by ~1 (α ≈ 1.5).
    assert dfa(np.cumsum(noise)).exponent == pytest.approx(1.5, abs=0.10)


def test_aggregated_variance_white_noise() -> None:
    rng = np.random.default_rng(2)
    h = aggregated_variance(rng.standard_normal(4000)).exponent
    assert h == pytest.approx(0.5, abs=0.1)


def test_hurst_dispatch_matches_named_functions() -> None:
    x = fgn(1500, 0.65, 7)
    assert hurst(x, method="rs").exponent == rs_analysis(x).exponent
    assert hurst(x, method="dfa").exponent == dfa(x).exponent
    assert hurst(x, method="agg_var").exponent == aggregated_variance(x).exponent


def test_hurst_result_fields() -> None:
    res = hurst(fgn(1500, 0.6, 0))
    assert isinstance(res, HurstResult)
    assert res.method == "rs"
    assert res.scales.shape == res.fluctuations.shape
    assert 0.0 < res.r_squared <= 1.0
    assert "Hurst" in str(res)


def test_fractal_dimension_linear_is_one() -> None:
    assert fractal_dimension(np.linspace(0.0, 10.0, 500)) == pytest.approx(1.0, abs=0.05)


def test_fractal_dimension_white_noise_is_rough() -> None:
    rng = np.random.default_rng(3)
    assert fractal_dimension(rng.standard_normal(2000)) > 1.7


def test_fractal_dimension_hurst_method_consistent() -> None:
    x = fgn(2000, 0.7, 1)
    # D = 2 − H, so a persistent series (H > 0.5) is smoother (D < 1.5).
    assert fractal_dimension(x, method="hurst") < 1.5


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        hurst(np.random.default_rng(0).standard_normal(500), method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fractal_dimension(np.arange(100.0), method="nope")  # type: ignore[arg-type]


def _binomial_cascade(n_levels: int, p: float, seed: int) -> np.ndarray:
    """Binomial multiplicative cascade — a canonical multifractal measure."""
    rng = np.random.default_rng(seed)
    series = np.array([1.0])
    for _ in range(n_levels):
        left = series * p
        right = series * (1.0 - p)
        series = np.empty(series.size * 2)
        series[0::2] = left
        series[1::2] = right
    # Random sign shuffle of multipliers keeps the multifractal scaling.
    return series * rng.choice(np.array([-1.0, 1.0]), size=series.size)


def test_mfdfa_monofractal_has_flat_spectrum() -> None:
    rng = np.random.default_rng(0)
    res = mfdfa(rng.standard_normal(8192))
    assert isinstance(res, MFDFAResult)
    # White noise is monofractal → h(q) nearly constant and a narrow f(α).
    assert res.hq.std() < 0.1
    assert res.width < 0.4
    # h(2) is the ordinary DFA exponent ≈ 0.5.
    assert float(np.interp(2.0, res.q, res.hq)) == pytest.approx(0.5, abs=0.1)


def test_mfdfa_detects_multifractality() -> None:
    mono = mfdfa(np.random.default_rng(1).standard_normal(4096)).width
    multi = mfdfa(_binomial_cascade(12, 0.7, seed=1)).width
    assert multi > mono
    assert multi > 0.3


def test_mfdfa_hq_decreasing_for_multifractal() -> None:
    res = mfdfa(_binomial_cascade(13, 0.75, seed=2))
    # For a multifractal, h(q) decreases with q.
    assert res.hq[0] > res.hq[-1]
    assert res.q.shape == res.hq.shape == res.tau.shape == res.alpha.shape


def test_mfdfa_custom_q_values() -> None:
    res = mfdfa(np.random.default_rng(3).standard_normal(4096), q_values=np.array([-2.0, 2.0]))
    assert res.q.shape == (2,)
    assert res.fluctuations.shape == (2, res.scales.shape[0])
