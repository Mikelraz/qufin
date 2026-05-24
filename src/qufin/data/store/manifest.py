"""
Coverage manifest for the parquet store.

Tracks which (symbol, interval) ranges are present on disk so the data pipeline
can compute delta-fetch ranges without scanning every partition.

Coverage is stored as one or more half-open ``[start, end)`` intervals per
(symbol, interval) pair. Adjacent or overlapping intervals are merged on write.
Persisted as JSON at ``<root>/_manifest.json``; absence implies an empty store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


@dataclass(slots=True, frozen=True)
class Interval:
    """Half-open ``[start, end)`` interval in UTC."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Interval end must be strictly greater than start")

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def adjacent_or_overlaps(self, other: Interval) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(slots=True)
class Coverage:
    """Sorted, non-overlapping list of covered intervals for one (symbol, interval)."""

    intervals: list[Interval] = field(default_factory=list)

    def add(self, new: Interval) -> None:
        """Insert ``new``, merging with any adjacent or overlapping entries."""
        merged: list[Interval] = []
        cur = new
        for iv in self.intervals:
            if cur.adjacent_or_overlaps(iv):
                cur = Interval(start=min(cur.start, iv.start), end=max(cur.end, iv.end))
            else:
                merged.append(iv)
        merged.append(cur)
        merged.sort(key=lambda i: i.start)
        self.intervals = merged

    def missing(self, start: datetime, end: datetime) -> list[Interval]:
        """Return the sub-intervals of ``[start, end)`` not covered by this set."""
        if end <= start:
            return []
        gaps: list[Interval] = []
        cursor = start
        for iv in self.intervals:
            if iv.end <= cursor:
                continue
            if iv.start >= end:
                break
            if iv.start > cursor:
                gaps.append(Interval(start=cursor, end=min(iv.start, end)))
            cursor = max(cursor, iv.end)
            if cursor >= end:
                break
        if cursor < end:
            gaps.append(Interval(start=cursor, end=end))
        return gaps


def _key(symbol: str, interval: str) -> str:
    return f"{symbol}\t{interval}"


def _parse_key(key: str) -> tuple[str, str]:
    symbol, interval = key.split("\t", 1)
    return symbol, interval


@dataclass(slots=True)
class Manifest:
    """On-disk coverage index. Use ``Manifest.load`` / ``Manifest.save``."""

    path: Path
    coverage: dict[tuple[str, str], Coverage] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Self:
        if not path.exists():
            return cls(path=path)
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        coverage: dict[tuple[str, str], Coverage] = {}
        for key, intervals in raw.items():
            cov = Coverage(
                intervals=[
                    Interval(
                        start=datetime.fromisoformat(s).astimezone(UTC),
                        end=datetime.fromisoformat(e).astimezone(UTC),
                    )
                    for s, e in intervals
                ]
            )
            coverage[_parse_key(key)] = cov
        return cls(path=path, coverage=coverage)

    def save(self) -> None:
        payload = {
            _key(sym, ivl): [(iv.start.isoformat(), iv.end.isoformat()) for iv in cov.intervals]
            for (sym, ivl), cov in self.coverage.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def coverage_for(self, symbol: str, interval: str) -> Coverage:
        return self.coverage.get((symbol, interval), Coverage())

    def record(self, symbol: str, interval: str, start: datetime, end: datetime) -> None:
        cov = self.coverage.setdefault((symbol, interval), Coverage())
        cov.add(Interval(start=start, end=end))

    def missing(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Interval]:
        return self.coverage_for(symbol, interval).missing(start, end)

    def drop(self, symbol: str, interval: str) -> None:
        self.coverage.pop((symbol, interval), None)
