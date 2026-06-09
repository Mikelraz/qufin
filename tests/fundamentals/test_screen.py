"""Tests for qufin.fundamentals.screen."""

from __future__ import annotations

import math

import polars as pl
import pytest

from qufin.fundamentals import (
    composite_score,
    percentile_rank,
    rank_universe,
    zscore,
)


@pytest.fixture
def universe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "roe": [0.20, 0.15, 0.10, 0.05],
            "pe": [10.0, 20.0, 15.0, 30.0],
        }
    )


def test_zscore_is_standardised(universe: pl.DataFrame) -> None:
    out = zscore(universe, ["roe", "pe"])
    assert "roe_z" in out.columns
    # z-scores sum to ~0 across the cross-section
    assert out.get_column("roe_z").sum() == pytest.approx(0.0, abs=1e-9)
    # highest ROE (ticker A) has the largest z
    assert out.filter(pl.col("ticker") == "A").get_column("roe_z").item() == pytest.approx(
        out.get_column("roe_z").max()
    )


def test_percentile_rank_bounds(universe: pl.DataFrame) -> None:
    out = percentile_rank(universe, ["roe"])
    pct = out.get_column("roe_pct")
    assert pct.min() >= 0.0
    assert pct.max() == pytest.approx(1.0)
    # best ROE -> top percentile
    assert out.filter(pl.col("ticker") == "A").get_column("roe_pct").item() == pytest.approx(1.0)


def test_composite_respects_direction(universe: pl.DataFrame) -> None:
    out = composite_score(
        universe,
        {"roe": 1.0, "pe": 1.0},
        higher_is_better={"roe": True, "pe": False},
    )
    scores = dict(zip(out.get_column("ticker"), out.get_column("composite"), strict=True))
    # A has the best ROE and the cheapest P/E -> highest composite
    assert scores["A"] == max(scores.values())
    assert scores["D"] == min(scores.values())


def test_rank_universe_orders_best_first(universe: pl.DataFrame) -> None:
    ranked = rank_universe(
        universe,
        {"roe": 1.0, "pe": 1.0},
        higher_is_better={"roe": True, "pe": False},
    )
    assert ranked.get_column("rank").to_list() == [1, 2, 3, 4]
    assert ranked.get_column("ticker").to_list()[0] == "A"
    assert ranked.get_column("ticker").to_list()[-1] == "D"


def test_missing_factor_is_neutral_not_disqualifying() -> None:
    df = pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "roe": [0.20, None, 0.05],
        }
    )
    out = composite_score(df, {"roe": 1.0})
    composite = dict(zip(out.get_column("ticker"), out.get_column("composite"), strict=True))
    assert composite["B"] == pytest.approx(0.0)  # null factor -> neutral
    assert all(not math.isnan(v) for v in composite.values())


def test_composite_requires_weights(universe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        composite_score(universe, {})


def test_winsorize_runs_and_clips_outlier() -> None:
    df = pl.DataFrame({"ticker": ["A", "B", "C", "D", "E"], "x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    out = zscore(df, ["x"], winsorize=0.2)
    # the extreme value is clipped, so its z-score is far smaller than unclipped
    raw = zscore(df, ["x"])
    assert out.get_column("x_z").max() < raw.get_column("x_z").max()
