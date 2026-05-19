"""Shared fixtures for indicators tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_close() -> np.ndarray:
    """Deterministic 200-bar log-random-walk close series."""
    rng = np.random.default_rng(0)
    log_rets = rng.normal(0.0005, 0.01, size=200)
    return 100.0 * np.exp(np.cumsum(log_rets))


@pytest.fixture
def synthetic_ohlc(synthetic_close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OHLC arrays derived from ``synthetic_close``."""
    rng = np.random.default_rng(1)
    closes = synthetic_close
    opens = np.concatenate(([closes[0]], closes[:-1]))
    noise = rng.uniform(0.0, 0.005, size=closes.shape[0]) * closes
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    return highs, lows, closes


@pytest.fixture
def synthetic_ohlcv(
    synthetic_ohlc: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OHLC + volume arrays."""
    rng = np.random.default_rng(2)
    highs, lows, closes = synthetic_ohlc
    volumes = rng.lognormal(mean=10.0, sigma=0.4, size=closes.shape[0])
    return highs, lows, closes, volumes
