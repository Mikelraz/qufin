"""Cointegration pair screening over a universe."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qufin.analysis import PairScreenResult, screen_pairs


def _panel(seed: int = 0) -> pl.DataFrame:
    """X, Y cointegrated (share a common trend); Z an independent walk."""
    rng = np.random.default_rng(seed)
    n = 800
    common = np.cumsum(rng.normal(0.0, 1.0, size=n))
    x = common + rng.normal(0.0, 0.5, size=n)
    y = 2.0 * common + rng.normal(0.0, 0.5, size=n)
    z = np.cumsum(rng.normal(0.0, 1.0, size=n))
    return pl.DataFrame({"X": x, "Y": y, "Z": z})


def test_screen_finds_cointegrated_pair() -> None:
    res = screen_pairs(_panel(), method="engle_granger", alpha=0.05, use_log=False)
    assert res
    assert all(isinstance(r, PairScreenResult) for r in res)
    top = res[0]
    assert {top.y, top.x} == {"X", "Y"}
    assert top.p_value < 0.05
    # Hedge ratio is ~2 (Y on X) or ~0.5 (X on Y) depending on orientation.
    assert (1.5 < top.beta < 2.5) or (0.35 < top.beta < 0.65)


def test_screen_excludes_independent_series() -> None:
    res = screen_pairs(_panel(), method="engle_granger", alpha=0.05, use_log=False)
    for r in res:
        assert "Z" not in {r.y, r.x}


def test_screen_accepts_array_with_names() -> None:
    df = _panel()
    res = screen_pairs(df.to_numpy(), names=list(df.columns), method="engle_granger", use_log=False)
    assert res and {res[0].y, res[0].x} == {"X", "Y"}


def test_screen_johansen_method() -> None:
    res = screen_pairs(_panel(), method="johansen", use_log=False)
    assert res
    assert {res[0].y, res[0].x} == {"X", "Y"}
    assert res[0].method == "johansen"


def test_screen_requires_two_assets() -> None:
    with pytest.raises(ValueError):
        screen_pairs(pl.DataFrame({"only": [1.0, 2.0, 3.0]}), use_log=False)
