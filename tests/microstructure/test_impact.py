"""Price-impact and illiquidity: Kyle λ, Hasbrouck λ, Amihud ILLIQ."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import amihud_illiquidity, hasbrouck_lambda, kyle_lambda


def test_kyle_lambda_recovers_planted_slope() -> None:
    rng = np.random.default_rng(0)
    signed_vol = rng.normal(0.0, 1.0, size=5000)
    dp = 0.5 * signed_vol + rng.normal(0.0, 0.01, size=5000)
    res = kyle_lambda(dp, signed_vol)
    assert res.lam == pytest.approx(0.5, abs=0.01)
    assert res.r_squared > 0.99
    assert res.t_stat > 50.0
    assert res.n_obs == 5000


def test_hasbrouck_lambda_recovers_sqrt_impact() -> None:
    rng = np.random.default_rng(1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=5000)
    volume = rng.uniform(1.0, 100.0, size=5000)
    dp = 0.3 * signs * np.sqrt(volume) + rng.normal(0.0, 0.05, size=5000)
    res = hasbrouck_lambda(dp, signs, volume)
    assert res.lam == pytest.approx(0.3, abs=0.01)


def test_amihud_matches_manual_mean() -> None:
    returns = np.array([0.02, -0.01, 0.03])
    dollar_volume = np.array([1e6, 2e6, 1e6])
    expected = 1e6 * np.mean(np.abs(returns) / dollar_volume)
    assert amihud_illiquidity(returns, dollar_volume) == pytest.approx(expected)


def test_amihud_drops_zero_volume_periods() -> None:
    returns = np.array([0.02, -0.01, 0.03])
    dollar_volume = np.array([1e6, 0.0, 1e6])
    expected = 1e6 * np.mean(np.abs(returns[[0, 2]]) / dollar_volume[[0, 2]])
    assert amihud_illiquidity(returns, dollar_volume) == pytest.approx(expected)


def test_hasbrouck_rejects_negative_volume() -> None:
    with pytest.raises(ValueError):
        hasbrouck_lambda(np.zeros(5), np.ones(5), -np.ones(5))
