"""Coverage / Manifest unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qufin.data.store.manifest import Coverage, Interval, Manifest


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_coverage_merges_adjacent_intervals() -> None:
    cov = Coverage()
    cov.add(Interval(_dt(2020), _dt(2021)))
    cov.add(Interval(_dt(2021), _dt(2022)))
    assert len(cov.intervals) == 1
    assert cov.intervals[0] == Interval(_dt(2020), _dt(2022))


def test_coverage_merges_overlapping_intervals() -> None:
    cov = Coverage()
    cov.add(Interval(_dt(2020), _dt(2021, 6)))
    cov.add(Interval(_dt(2021, 3), _dt(2022)))
    assert len(cov.intervals) == 1
    assert cov.intervals[0] == Interval(_dt(2020), _dt(2022))


def test_coverage_keeps_disjoint_intervals_sorted() -> None:
    cov = Coverage()
    cov.add(Interval(_dt(2022), _dt(2023)))
    cov.add(Interval(_dt(2020), _dt(2021)))
    assert [iv.start.year for iv in cov.intervals] == [2020, 2022]


def test_missing_returns_full_range_for_empty_coverage() -> None:
    cov = Coverage()
    gaps = cov.missing(_dt(2020), _dt(2022))
    assert gaps == [Interval(_dt(2020), _dt(2022))]


def test_missing_returns_empty_when_fully_covered() -> None:
    cov = Coverage(intervals=[Interval(_dt(2019), _dt(2025))])
    assert cov.missing(_dt(2020), _dt(2024)) == []


def test_missing_returns_gap_between_two_intervals() -> None:
    cov = Coverage(
        intervals=[
            Interval(_dt(2020), _dt(2021)),
            Interval(_dt(2022), _dt(2023)),
        ]
    )
    gaps = cov.missing(_dt(2020), _dt(2023))
    assert gaps == [Interval(_dt(2021), _dt(2022))]


def test_missing_handles_prefix_and_suffix_gaps() -> None:
    cov = Coverage(intervals=[Interval(_dt(2021), _dt(2022))])
    gaps = cov.missing(_dt(2020), _dt(2023))
    assert gaps == [
        Interval(_dt(2020), _dt(2021)),
        Interval(_dt(2022), _dt(2023)),
    ]


def test_manifest_round_trips_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "_manifest.json"
    m = Manifest.load(path)
    m.record("AAPL", "1d", _dt(2020), _dt(2021))
    m.record("AAPL", "1d", _dt(2021), _dt(2022))
    m.record("MSFT", "1h", _dt(2023, 6), _dt(2023, 7))
    m.save()

    reloaded = Manifest.load(path)
    assert reloaded.coverage_for("AAPL", "1d").intervals == [
        Interval(_dt(2020), _dt(2022))
    ]
    assert reloaded.coverage_for("MSFT", "1h").intervals == [
        Interval(_dt(2023, 6), _dt(2023, 7))
    ]
    assert reloaded.coverage_for("MISSING", "1d").intervals == []
