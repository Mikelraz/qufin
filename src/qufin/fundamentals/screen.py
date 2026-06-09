"""
Cross-sectional screening and ranking over a universe of issuers.

Input is a "wide" polars frame with one row per ticker and one column per
factor (ratios, multiples, scores — produced however the caller likes).  These
helpers standardise factors across the cross-section (z-score or percentile
rank), combine them into a weighted composite, and rank the universe.

Standardisation ignores nulls when computing the cross-sectional mean/std; a
ticker missing a factor receives a *neutral* (zero) z-score for that factor in
the composite rather than being dropped.  Use ``winsorize`` to clip factor
outliers to symmetric quantiles before standardising.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def _z_expr(col: str, winsorize: float | None) -> pl.Expr:
    series = pl.col(col)
    if winsorize is not None:
        series = series.clip(
            lower_bound=pl.col(col).quantile(winsorize),
            upper_bound=pl.col(col).quantile(1.0 - winsorize),
        )
    return (series - series.mean()) / series.std()


def zscore(
    df: pl.DataFrame, cols: Sequence[str], *, winsorize: float | None = None
) -> pl.DataFrame:
    """Append cross-sectional z-scores (``{col}_z``) for each column in ``cols``.

    Args:
        df: Universe frame (one row per ticker).
        cols: Factor columns to standardise.
        winsorize: If set (e.g. ``0.05``), clip each factor to its
            ``[winsorize, 1 - winsorize]`` quantiles before z-scoring.

    Returns:
        ``df`` with one added ``{col}_z`` column per input column.
    """
    return df.with_columns([_z_expr(c, winsorize).alias(f"{c}_z") for c in cols])


def percentile_rank(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Append cross-sectional percentile ranks (``{col}_pct``) in ``[0, 1]``.

    Args:
        df: Universe frame (one row per ticker).
        cols: Factor columns to rank.

    Returns:
        ``df`` with one added ``{col}_pct`` column per input column, where the
        highest value maps to ``1.0``.
    """
    return df.with_columns(
        [(pl.col(c).rank(method="average") / pl.col(c).count()).alias(f"{c}_pct") for c in cols]
    )


def _resolve_direction(
    weights: dict[str, float], higher_is_better: bool | dict[str, bool]
) -> dict[str, bool]:
    if isinstance(higher_is_better, bool):
        return {c: higher_is_better for c in weights}
    return {c: higher_is_better.get(c, True) for c in weights}


def composite_score(
    df: pl.DataFrame,
    weights: dict[str, float],
    *,
    higher_is_better: bool | dict[str, bool] = True,
    winsorize: float | None = None,
) -> pl.DataFrame:
    """Append a weighted z-score ``composite`` column.

    Each factor is z-scored across the cross-section, its sign flipped when
    lower values are preferable (e.g. P/E, leverage), scaled by its weight, and
    summed.  Missing factor values contribute zero (neutral).

    Args:
        df: Universe frame (one row per ticker).
        weights: ``{factor_column: weight}``. Weights need not sum to one.
        higher_is_better: Either a single bool applied to every factor, or a
            ``{factor_column: bool}`` mapping. ``False`` inverts the factor so
            that smaller raw values rank higher.
        winsorize: Optional symmetric quantile clip applied before z-scoring.

    Returns:
        ``df`` with an added ``composite`` column.

    Raises:
        ValueError: If ``weights`` is empty.
    """
    if not weights:
        raise ValueError("weights must contain at least one factor")

    direction = _resolve_direction(weights, higher_is_better)
    terms = [
        _z_expr(c, winsorize).fill_null(0.0) * ((1.0 if direction[c] else -1.0) * w)
        for c, w in weights.items()
    ]
    composite = terms[0]
    for term in terms[1:]:
        composite = composite + term
    return df.with_columns(composite.alias("composite"))


def rank_universe(
    df: pl.DataFrame,
    weights: dict[str, float],
    *,
    higher_is_better: bool | dict[str, bool] = True,
    winsorize: float | None = None,
    descending: bool = True,
) -> pl.DataFrame:
    """Score and rank a universe by the weighted-composite factor model.

    Args:
        df: Universe frame (one row per ticker).
        weights: ``{factor_column: weight}`` for :func:`composite_score`.
        higher_is_better: Per-factor or global direction (see :func:`composite_score`).
        winsorize: Optional symmetric quantile clip before z-scoring.
        descending: If ``True`` (default), the highest composite ranks first.

    Returns:
        ``df`` sorted by ``composite`` with an integer ``rank`` column (1 = best),
        and the intermediate ``composite`` column retained.
    """
    scored = composite_score(df, weights, higher_is_better=higher_is_better, winsorize=winsorize)
    ordered = scored.sort("composite", descending=descending, nulls_last=True)
    return ordered.with_row_index(name="rank", offset=1)
