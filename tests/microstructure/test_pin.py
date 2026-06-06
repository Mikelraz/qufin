"""PIN — structural Probability of Informed Trading (Easley et al.)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import pin


def simulate_pin(
    n_days: int,
    *,
    alpha: float,
    delta: float,
    mu: float,
    eps_b: float,
    eps_s: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw daily (buys, sells) counts from the PIN data-generating process."""
    rng = np.random.default_rng(seed)
    event = rng.random(n_days) < alpha
    bad = event & (rng.random(n_days) < delta)
    good = event & ~bad
    buys = rng.poisson(eps_b + good * mu)
    sells = rng.poisson(eps_s + bad * mu)
    return buys.astype(np.float64), sells.astype(np.float64)


def test_pin_recovers_known_parameters() -> None:
    true_alpha, true_delta, true_mu, eb, es = 0.4, 0.5, 80.0, 40.0, 40.0
    buys, sells = simulate_pin(
        2500, alpha=true_alpha, delta=true_delta, mu=true_mu, eps_b=eb, eps_s=es, seed=1
    )
    res = pin(buys, sells)
    true_pin = true_alpha * true_mu / (true_alpha * true_mu + eb + es)
    assert res.pin == pytest.approx(true_pin, abs=0.08)
    assert res.alpha == pytest.approx(true_alpha, abs=0.15)
    assert res.mu == pytest.approx(true_mu, rel=0.25)
    assert 0.0 <= res.pin <= 1.0


def test_pin_low_when_flow_is_balanced_noise() -> None:
    # No information events → PIN should be near zero.
    buys, sells = simulate_pin(2000, alpha=0.02, delta=0.5, mu=5.0, eps_b=50.0, eps_s=50.0, seed=2)
    res = pin(buys, sells)
    assert res.pin < 0.1


def test_pin_str_and_fields() -> None:
    buys, sells = simulate_pin(500, alpha=0.3, delta=0.5, mu=60.0, eps_b=30.0, eps_s=30.0, seed=3)
    res = pin(buys, sells)
    assert "PIN=" in str(res)
    assert res.n_obs == 500


def test_pin_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        pin(np.array([-1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
