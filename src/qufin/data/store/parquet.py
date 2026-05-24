"""
Partitioned Parquet store for OHLCV bars.

Layout: ``<root>/<symbol>/<interval>/year=<YYYY>/bars.parquet`` (Hive-style).
Polars can ``scan_parquet`` the symbol/interval subtree and predicate-pushdown
on ``timestamp``; ``get`` materialises a slice into an ``OHLCV``.

Writes are upserts at the (timestamp) grain: any existing bars with timestamps
inside the new frame's range are replaced. Coverage is recorded in
``Manifest`` so the pipeline can delta-fetch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import polars as pl

from .._types import BAR_SCHEMA, OHLCV
from .manifest import Interval, Manifest
from .partition import partition_path, symbol_root, years_in_range


@dataclass(slots=True)
class Store:
    """Partitioned Parquet store for OHLCV data."""

    root: Path
    manifest: Manifest = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest.load(self.root / "_manifest.json")

    @classmethod
    def open(cls, root: Path | str) -> Self:
        return cls(root=Path(root))

    def get(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> OHLCV | None:
        """Return bars for ``[start, end)``. ``None`` if no rows are present."""
        lf = self._scan_lazy(symbol, interval)
        if lf is None:
            return None
        frame = (
            lf.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            .sort("timestamp")
            .collect()
        )
        if frame.is_empty():
            return None
        return OHLCV.from_records(frame, symbol=symbol)

    def scan(self, symbol: str, interval: str) -> pl.LazyFrame | None:
        """Return a lazy frame over every partition for (symbol, interval), or None."""
        return self._scan_lazy(symbol, interval)

    def put(self, ohlcv: OHLCV, interval: str) -> None:
        """Write ``ohlcv`` into the store, upserting by timestamp per year partition."""
        if not ohlcv.symbol:
            raise ValueError("OHLCV.symbol is required to write to the store")
        if ohlcv.data.is_empty():
            return

        symbol = ohlcv.symbol
        ts_min: datetime = ohlcv.data["timestamp"].min()  # type: ignore[assignment]
        ts_max: datetime = ohlcv.data["timestamp"].max()  # type: ignore[assignment]

        frame_with_year = ohlcv.data.with_columns(
            pl.col("timestamp").dt.year().alias("_year")
        )
        for year in years_in_range(ts_min, ts_max):
            chunk = frame_with_year.filter(pl.col("_year") == year).drop("_year")
            if chunk.is_empty():
                continue
            self._upsert_partition(symbol, interval, year, chunk)

        # microsecond precision is the floor of stdlib datetime; bumping by one
        # μs is enough to make the recorded interval half-open without losing
        # information at the manifest granularity.
        end_exclusive = ts_max + timedelta(microseconds=1)
        self.manifest.record(symbol, interval, ts_min, end_exclusive)
        self.manifest.save()

    def missing_ranges(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Interval]:
        """Return the sub-ranges of ``[start, end)`` not yet stored."""
        return self.manifest.missing(symbol, interval, start, end)

    def invalidate(self, symbol: str, interval: str) -> None:
        """Delete all partitions for (symbol, interval) and forget coverage."""
        directory = symbol_root(self.root, symbol, interval)
        if directory.exists():
            for path in directory.rglob("*.parquet"):
                path.unlink()
            for sub in sorted(directory.glob("year=*"), reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        self.manifest.drop(symbol, interval)
        self.manifest.save()

    def _scan_lazy(self, symbol: str, interval: str) -> pl.LazyFrame | None:
        directory = symbol_root(self.root, symbol, interval)
        if not directory.exists():
            return None
        pattern = str(directory / "year=*" / "bars.parquet")
        try:
            return pl.scan_parquet(pattern)
        except FileNotFoundError:
            return None

    def _upsert_partition(
        self,
        symbol: str,
        interval: str,
        year: int,
        chunk: pl.DataFrame,
    ) -> None:
        path = partition_path(self.root, symbol, interval, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        coerced = chunk.select(
            *(pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items())
        )
        if path.exists():
            existing = pl.read_parquet(path)
            new_ts = coerced["timestamp"].implode()
            existing = existing.filter(~pl.col("timestamp").is_in(new_ts))
            merged = pl.concat([existing, coerced]).sort("timestamp")
        else:
            merged = coerced.sort("timestamp")
        merged.write_parquet(path)
