"""End-to-end DataPipeline integration test (offline; CsvOHLC vendor)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qufin.data import OHLCV, DataPipeline
from qufin.data._types import BAR_SCHEMA
from qufin.data.adjustments import ACTIONS_SCHEMA
from qufin.data.adjustments.actions import CorporateAction, actions_frame
from qufin.data.calendars import NYSECalendar
from qufin.data.store import Store
from qufin.data.vendors import CsvOHLC


def _write_csv(path: Path, n: int = 60, start: datetime | None = None) -> None:
    if start is None:
        start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.5 + i * 0.1 for i in range(n)],
            "low": [99.5 + i * 0.1 for i in range(n)],
            "close": [100.2 + i * 0.1 for i in range(n)],
            "volume": [1_000_000.0 + i for i in range(n)],
        },
        schema=BAR_SCHEMA,
    )
    df.write_parquet(path)


class _FakeActions:
    """In-memory ``ActionsSource`` for the integration test."""

    def __init__(self, actions: list[CorporateAction]) -> None:
        self._frame = actions_frame(actions)

    def fetch(self, symbols: Sequence[str]) -> pl.DataFrame:
        return self._frame.filter(pl.col("symbol").is_in(list(symbols)))


def test_pipeline_first_call_fetches_then_caches(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    _write_csv(data_dir / "AAPL.parquet", n=60)
    store = Store.open(tmp_path / "store")
    vendor = CsvOHLC(root=data_dir, extension=".parquet")
    pipe = DataPipeline(vendor=vendor, store=store, calendar=None)

    bars = pipe.load(
        "AAPL",
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 3, 2, tzinfo=UTC),
        interval="1d",
    )
    assert bars.n_bars == 60
    # Coverage now in manifest
    cov = store.manifest.coverage_for("AAPL", "1d")
    assert len(cov.intervals) == 1


def test_pipeline_second_call_hits_cache_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    _write_csv(data_dir / "AAPL.parquet", n=60)
    store = Store.open(tmp_path / "store")

    calls: list[tuple[str, datetime, datetime]] = []

    class _CountingVendor(CsvOHLC):
        def fetch(  # type: ignore[override]
            self, symbol: str, *, start: datetime, end: datetime, interval: str
        ) -> OHLCV:
            calls.append((symbol, start, end))
            return super().fetch(symbol, start=start, end=end, interval=interval)

    vendor = _CountingVendor(root=data_dir, extension=".parquet")
    pipe = DataPipeline(vendor=vendor, store=store)

    win = dict(
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 3, 2, tzinfo=UTC),
        interval="1d",
    )
    pipe.load("AAPL", **win)
    n_after_first = len(calls)
    pipe.load("AAPL", **win)
    assert len(calls) == n_after_first  # second call short-circuited


def test_pipeline_delta_fetches_only_the_gap(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    _write_csv(data_dir / "AAPL.parquet", n=180)
    store = Store.open(tmp_path / "store")

    seen_windows: list[tuple[datetime, datetime]] = []

    class _CountingVendor(CsvOHLC):
        def fetch(  # type: ignore[override]
            self, symbol: str, *, start: datetime, end: datetime, interval: str
        ) -> OHLCV:
            seen_windows.append((start, end))
            return super().fetch(symbol, start=start, end=end, interval=interval)

    vendor = _CountingVendor(root=data_dir, extension=".parquet")
    pipe = DataPipeline(vendor=vendor, store=store)

    pipe.load(
        "AAPL",
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        interval="1d",
    )
    pipe.load(
        "AAPL",
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
        interval="1d",
    )
    # Second call should have fetched only the gap (Feb–Apr), not the whole range.
    assert len(seen_windows) == 2
    second_start, _second_end = seen_windows[1]
    assert second_start >= datetime(2024, 1, 30, tzinfo=UTC)


def test_pipeline_applies_calendar_alignment(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    # Daily bars from a Sat (Jan 6, 2024) — calendar should drop weekend.
    _write_csv(
        data_dir / "AAPL.parquet",
        n=10,
        start=datetime(2024, 1, 6, 14, 30, tzinfo=UTC),
    )
    store = Store.open(tmp_path / "store")
    vendor = CsvOHLC(root=data_dir, extension=".parquet")
    cal = NYSECalendar()
    pipe = DataPipeline(vendor=vendor, store=store, calendar=cal)

    bars = pipe.load(
        "AAPL",
        start=datetime(2024, 1, 6, tzinfo=UTC),
        end=datetime(2024, 1, 20, tzinfo=UTC),
        interval="1d",
        align_calendar=True,
    )
    # Originally 10 bars (Jan 6–15); calendar drops Sat Jan 6, Sun Jan 7, Sat Jan 13,
    # Sun Jan 14, and MLK day Mon Jan 15 — 5 dropped, 5 kept.
    assert bars.n_bars == 5


def test_pipeline_applies_corporate_action_adjustments(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    _write_csv(data_dir / "AAPL.parquet", n=30)
    store = Store.open(tmp_path / "store")
    vendor = CsvOHLC(root=data_dir, extension=".parquet")
    actions = _FakeActions(
        [
            CorporateAction(
                timestamp=datetime(2024, 1, 20, 14, 30, tzinfo=UTC),
                symbol="AAPL",
                kind="split",
                ratio=2.0,
            )
        ]
    )
    pipe = DataPipeline(vendor=vendor, store=store, actions=actions)

    bars = pipe.load(
        "AAPL",
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        interval="1d",
        adjust=True,
        align_calendar=False,
    )
    # Bar on Jan 2 (raw close 100.2) should be halved by the Jan 20 2-for-1 split.
    first_close = bars.data["close"][0]
    assert first_close < 60.0
    # Bar on Jan 20 itself (raw close = 100.2 + 18*0.1 = 102.0) is unchanged.
    on_split = bars.data.filter(
        pl.col("timestamp") == datetime(2024, 1, 20, 14, 30, tzinfo=UTC)
    )
    assert on_split.height == 1
    assert abs(float(on_split["close"][0]) - 102.0) < 1e-9


def test_pipeline_load_many_returns_dict(tmp_path: Path) -> None:
    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    for sym in ("AAPL", "MSFT", "GOOG"):
        _write_csv(data_dir / f"{sym}.parquet", n=10)
    store = Store.open(tmp_path / "store")
    vendor = CsvOHLC(root=data_dir, extension=".parquet")
    pipe = DataPipeline(vendor=vendor, store=store)

    out = pipe.load_many(
        ["AAPL", "MSFT", "GOOG"],
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        interval="1d",
    )
    assert set(out.keys()) == {"AAPL", "MSFT", "GOOG"}
    for ohlcv in out.values():
        assert ohlcv.n_bars == 10


def test_pipeline_rejects_inverted_range(tmp_path: Path) -> None:
    import pytest

    data_dir = tmp_path / "vendor"
    data_dir.mkdir()
    store = Store.open(tmp_path / "store")
    vendor = CsvOHLC(root=data_dir, extension=".parquet")
    pipe = DataPipeline(vendor=vendor, store=store)
    with pytest.raises(ValueError):
        pipe.load(
            "AAPL",
            start=datetime(2024, 6, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
            interval="1d",
        )


def test_actions_schema_is_accessible() -> None:
    assert "timestamp" in ACTIONS_SCHEMA
    assert "symbol" in ACTIONS_SCHEMA
