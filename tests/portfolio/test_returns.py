from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qufin.portfolio.returns import (
    annualize_return,
    annualized_returns,
    cumulative_returns,
    log_returns,
    simple_returns,
    to_returns_matrix,
)


@pytest.fixture
def prices() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 252
    start = date(2023, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "A": 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))),
            "B": 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        }
    )


def test_simple_returns_shape(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    assert ret.shape == (prices.shape[0] - 1, prices.shape[1])


def test_simple_returns_values(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    p = prices["A"].to_numpy()
    expected = (p[1:] / p[:-1]) - 1.0
    np.testing.assert_allclose(ret["A"].to_numpy(), expected, rtol=1e-10)


def test_simple_returns_preserves_date_col(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    assert "date" in ret.columns


def test_log_returns_values(prices: pl.DataFrame) -> None:
    ret = log_returns(prices)
    p = prices["A"].to_numpy()
    expected = np.log(p[1:] / p[:-1])
    np.testing.assert_allclose(ret["A"].to_numpy(), expected, rtol=1e-10)


def test_log_vs_simple_small_returns(prices: pl.DataFrame) -> None:
    sr = simple_returns(prices)["A"].to_numpy()
    lr = log_returns(prices)["A"].to_numpy()
    # for small r: log(1+r) ≈ r
    np.testing.assert_allclose(lr, np.log1p(sr), rtol=1e-10)


def test_cumulative_returns_terminal(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    cum = cumulative_returns(ret)
    p = prices["A"].to_numpy()
    expected_total = p[-1] / p[0] - 1.0
    assert abs(cum["A"].to_numpy()[-1] - expected_total) < 1e-10


def test_annualize_return_one_period() -> None:
    # 10 % over 252 periods, annualized: same
    assert annualize_return(0.10, 252, 252) == pytest.approx(0.10, rel=1e-10)


def test_annualize_return_compound() -> None:
    # 5 % over 126 trading days → ~10.25 % annualized
    ann = annualize_return(0.05, 126, 252)
    assert ann == pytest.approx(1.05**2 - 1, rel=1e-10)


def test_annualized_returns_type(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    result = annualized_returns(ret, 252)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"A", "B"}


def test_to_returns_matrix_shape(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    mat, names = to_returns_matrix(ret)
    assert mat.shape == (ret.shape[0], 2)
    assert names == ["A", "B"]


def test_to_returns_matrix_dtype(prices: pl.DataFrame) -> None:
    ret = simple_returns(prices)
    mat, _ = to_returns_matrix(ret)
    assert mat.dtype == np.float64
