"""Tests for src.timeseries._io — numpy / polars conversion helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries._io import (
    to_numpy_1d,
    to_numpy_2d,
    validate_finite,
    validate_min_length,
)


class TestToNumpy1D:
    def test_numpy_passthrough(self):
        arr = np.arange(10, dtype=np.float64)
        out = to_numpy_1d(arr)
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, arr)

    def test_numpy_int_cast(self):
        arr = np.arange(5, dtype=np.int32)
        out = to_numpy_1d(arr)
        assert out.dtype == np.float64

    def test_numpy_2d_single_column_ravels(self):
        arr = np.arange(6).reshape(-1, 1)
        out = to_numpy_1d(arr)
        assert out.ndim == 1
        assert out.shape == (6,)

    def test_numpy_2d_multicolumn_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            to_numpy_1d(np.zeros((3, 2)))

    def test_polars_series(self):
        s = pl.Series("x", [1.0, 2.0, 3.0])
        out = to_numpy_1d(s)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_polars_single_column_df(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = to_numpy_1d(df)
        assert out.shape == (3,)

    def test_polars_multicolumn_df_raises(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="1 column"):
            to_numpy_1d(df)

    def test_python_list_coerces(self):
        out = to_numpy_1d([1.0, 2.0, 3.0])
        assert out.shape == (3,)
        assert out.dtype == np.float64


class TestToNumpy2D:
    def test_numpy_passthrough(self):
        arr = np.arange(12, dtype=np.float64).reshape(4, 3)
        out = to_numpy_2d(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == np.float64

    def test_polars_dataframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        out = to_numpy_2d(df)
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out[:, 0], [1.0, 2.0, 3.0])

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            to_numpy_2d(np.arange(5))


class TestValidators:
    def test_min_length_pass(self):
        validate_min_length(np.arange(10), 5)

    def test_min_length_fail(self):
        with pytest.raises(ValueError, match="at least"):
            validate_min_length(np.arange(3), 5)

    def test_finite_pass(self):
        validate_finite(np.arange(5, dtype=np.float64))

    def test_finite_nan(self):
        x = np.array([1.0, 2.0, np.nan])
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(x)

    def test_finite_inf(self):
        x = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(x)
