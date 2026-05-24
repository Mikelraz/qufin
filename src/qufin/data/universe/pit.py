"""
Point-in-time (PIT) index membership primitives.

A *membership frame* is a long table where each row records one symbol's
period of inclusion in one index: ``(symbol, index, start, end)``. ``end``
should be a far-future sentinel (e.g. ``2200-01-01 UTC``) for currently
active members — note that nanosecond polars timestamps cap out at
2262-04-11.

These primitives let downstream code answer two questions without
look-ahead:

1. *"Which symbols were in index X at time T?"* → :func:`asof_universe`.
2. *"For each row of a feature frame, was the symbol in the index then?"*
   → :func:`pit_join`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

UTC = ZoneInfo("UTC")

MEMBERSHIP_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8(),
    "index": pl.Utf8(),
    "start": pl.Datetime("ns", time_zone="UTC"),
    "end": pl.Datetime("ns", time_zone="UTC"),
}


@dataclass(slots=True, frozen=True)
class UniverseSnapshot:
    """The set of symbols belonging to one index at one timestamp."""

    asof: datetime
    index: str
    symbols: tuple[str, ...]


def membership_from_records(records: list[dict[str, object]]) -> pl.DataFrame:
    """Build a ``MEMBERSHIP_SCHEMA``-conforming frame from a list of dicts."""
    if not records:
        return pl.DataFrame(schema=MEMBERSHIP_SCHEMA)
    df = pl.DataFrame(records)
    missing = set(MEMBERSHIP_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"records are missing keys: {sorted(missing)}")
    return df.select(
        *(pl.col(name).cast(dtype) for name, dtype in MEMBERSHIP_SCHEMA.items())
    )


def asof_universe(
    membership: pl.DataFrame, *, index: str, asof: datetime
) -> UniverseSnapshot:
    """Return the symbols in ``index`` whose membership window contains ``asof``."""
    _validate_schema(membership)
    if asof.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    asof_utc = asof.astimezone(UTC)
    rows = membership.filter(
        (pl.col("index") == index)
        & (pl.col("start") <= asof_utc)
        & (pl.col("end") > asof_utc)
    )
    symbols = tuple(sorted(rows["symbol"].to_list()))
    return UniverseSnapshot(asof=asof, index=index, symbols=symbols)


def pit_join(
    features: pl.DataFrame,
    membership: pl.DataFrame,
    *,
    index: str,
    on_timestamp: str = "timestamp",
    on_symbol: str = "symbol",
) -> pl.DataFrame:
    """Return rows of ``features`` whose (symbol, timestamp) was a member of ``index``.

    ``features`` must carry ``on_timestamp`` (tz-aware UTC) and ``on_symbol``
    columns. The output frame preserves the input schema and order.
    """
    _validate_schema(membership)
    if on_timestamp not in features.columns:
        raise ValueError(f"features is missing the '{on_timestamp}' column")
    if on_symbol not in features.columns:
        raise ValueError(f"features is missing the '{on_symbol}' column")

    mem = membership.filter(pl.col("index") == index).select(
        pl.col("symbol").alias(on_symbol),
        pl.col("start"),
        pl.col("end"),
    )
    if mem.is_empty():
        return features.head(0)

    out = features.join(mem, on=on_symbol, how="inner").filter(
        (pl.col(on_timestamp) >= pl.col("start"))
        & (pl.col(on_timestamp) < pl.col("end"))
    )
    return out.drop(["start", "end"])


def _validate_schema(membership: pl.DataFrame) -> None:
    missing = set(MEMBERSHIP_SCHEMA) - set(membership.columns)
    if missing:
        raise ValueError(f"membership frame is missing columns: {sorted(missing)}")
