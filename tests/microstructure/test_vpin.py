"""VPIN — volume-synchronised probability of informed trading."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.microstructure import vpin


def test_vpin_bounded_unit_interval() -> None:
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=10_000))
    volumes = rng.uniform(1.0, 10.0, size=10_000)
    res = vpin(prices, volumes, n_buckets=200, window=20)
    valid = res.vpin[~np.isnan(res.vpin)]
    assert valid.size > 0
    assert np.all((valid >= 0.0) & (valid <= 1.0))


def test_vpin_high_under_one_sided_flow() -> None:
    # Strong positive drift relative to noise → almost every price change is a
    # large positive z-score → BVC labels nearly all volume as buys →
    # near-maximal order imbalance → VPIN close to 1.
    rng = np.random.default_rng(11)
    prices = 100.0 + np.cumsum(rng.normal(0.5, 0.05, size=10_000))
    volumes = np.full(10_000, 5.0)
    res = vpin(prices, volumes, n_buckets=100, window=10)
    assert np.nanmean(res.vpin) > 0.9


def test_vpin_buckets_conserve_volume() -> None:
    rng = np.random.default_rng(2)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=5000))
    volumes = rng.uniform(1.0, 5.0, size=5000)
    res = vpin(prices, volumes, bucket_size=50.0, window=10)
    per_bucket = res.buy_volume + res.sell_volume
    np.testing.assert_allclose(per_bucket, 50.0, rtol=1e-6)


def test_vpin_requires_exactly_one_sizing_arg() -> None:
    prices = np.linspace(100.0, 101.0, 100)
    volumes = np.ones(100)
    with pytest.raises(ValueError):
        vpin(prices, volumes)
    with pytest.raises(ValueError):
        vpin(prices, volumes, bucket_size=10.0, n_buckets=10)


def test_vpin_dataframe_roundtrip() -> None:
    rng = np.random.default_rng(3)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=3000))
    volumes = rng.uniform(1.0, 5.0, size=3000)
    res = vpin(prices, volumes, n_buckets=60, window=5)
    df = res.to_dataframe()
    assert df.height == res.n_buckets
    assert set(df.columns) == {"bucket", "buy_volume", "sell_volume", "order_imbalance", "vpin"}
