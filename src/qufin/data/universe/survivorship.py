"""
Detect potential survivorship bias in a feature frame.

A feature frame is *survivorship-biased* when it only contains symbols that
exist today (or at the end of the dataset). This module returns a report on
how the symbol set in ``features`` relates to the full historical
membership in an index — symbols ever-in but missing from features are
suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .pit import MEMBERSHIP_SCHEMA


@dataclass(slots=True, frozen=True)
class SurvivorshipReport:
    """Outcome of :func:`detect_potential_survivorship_bias`."""

    index: str
    ever_members: int
    features_symbols: int
    ever_but_missing: tuple[str, ...]
    bias_ratio: float  # fraction of ever-members that are missing from features


def detect_potential_survivorship_bias(
    features: pl.DataFrame,
    membership: pl.DataFrame,
    *,
    index: str,
    on_symbol: str = "symbol",
) -> SurvivorshipReport:
    """Flag symbols that were ever in ``index`` but never appear in ``features``."""
    missing_cols = set(MEMBERSHIP_SCHEMA) - set(membership.columns)
    if missing_cols:
        raise ValueError(f"membership frame is missing columns: {sorted(missing_cols)}")
    if on_symbol not in features.columns:
        raise ValueError(f"features is missing the '{on_symbol}' column")
    ever = set(
        membership.filter(pl.col("index") == index)["symbol"].to_list()
    )
    present = set(features[on_symbol].to_list())
    missing = tuple(sorted(ever - present))
    n_ever = len(ever)
    return SurvivorshipReport(
        index=index,
        ever_members=n_ever,
        features_symbols=len(present),
        ever_but_missing=missing,
        bias_ratio=len(missing) / n_ever if n_ever else 0.0,
    )
