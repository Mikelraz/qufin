from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.covariance import (
    annualize_cov,
    cov_to_corr,
    ewm_cov,
    ledoit_wolf_cov,
    sample_cov,
)


@pytest.fixture
def returns_mat() -> np.ndarray:
    true_cov = np.array(
        [
            [4e-4, 1e-4, 2e-4],
            [1e-4, 5e-4, 1e-4],
            [2e-4, 1e-4, 3e-4],
        ]
    )
    rng = np.random.default_rng(7)
    return rng.multivariate_normal(mean=[5e-4, 6e-4, 4e-4], cov=true_cov, size=500)


def test_sample_cov_shape(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    n = returns_mat.shape[1]
    assert cov.shape == (n, n)


def test_sample_cov_matches_numpy(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    expected = np.cov(returns_mat.T, ddof=1)
    np.testing.assert_allclose(cov, expected, rtol=1e-12)


def test_sample_cov_symmetric(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)


def test_sample_cov_positive_definite(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


def test_ledoit_wolf_cov_shape(returns_mat: np.ndarray) -> None:
    cov = ledoit_wolf_cov(returns_mat)
    n = returns_mat.shape[1]
    assert cov.shape == (n, n)


def test_ledoit_wolf_cov_symmetric(returns_mat: np.ndarray) -> None:
    cov = ledoit_wolf_cov(returns_mat)
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)


def test_ledoit_wolf_cov_positive_definite(returns_mat: np.ndarray) -> None:
    cov = ledoit_wolf_cov(returns_mat)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


def test_ledoit_wolf_shrinks_toward_identity(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    lw = ledoit_wolf_cov(returns_mat)
    # Off-diagonal elements should be shrunk toward zero compared to sample cov
    off_diag_s = np.abs(cov - np.diag(np.diag(cov))).mean()
    off_diag_lw = np.abs(lw - np.diag(np.diag(lw))).mean()
    assert off_diag_lw <= off_diag_s


def test_ewm_cov_shape(returns_mat: np.ndarray) -> None:
    cov = ewm_cov(returns_mat, halflife=63)
    n = returns_mat.shape[1]
    assert cov.shape == (n, n)


def test_ewm_cov_positive_definite(returns_mat: np.ndarray) -> None:
    cov = ewm_cov(returns_mat, halflife=63)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


def test_annualize_cov_scales_correctly(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    ann = annualize_cov(cov, 252)
    np.testing.assert_allclose(ann, cov * 252, rtol=1e-12)


def test_cov_to_corr_diagonal_ones(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    corr = cov_to_corr(cov)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-12)


def test_cov_to_corr_values_in_range(returns_mat: np.ndarray) -> None:
    cov = sample_cov(returns_mat)
    corr = cov_to_corr(cov)
    assert np.all(corr >= -1.0 - 1e-10)
    assert np.all(corr <= 1.0 + 1e-10)
