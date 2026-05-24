"""
Canonical corporate-action schema and dataclass.

A ``CorporateAction`` row carries the *ex-date* timestamp: the first bar on
which the price reflects the action. Splits are encoded as
``ratio = new_shares / old_shares`` (2-for-1 = 2.0; 1-for-3 reverse = 1/3).
Cash dividends carry the per-share cash amount in ``cash`` and ``ratio = 1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl

ActionKind = Literal["split", "cash_div", "spinoff", "merger"]

ACTIONS_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "symbol": pl.Utf8(),
    "kind": pl.Utf8(),
    "ratio": pl.Float64(),
    "cash": pl.Float64(),
}


@dataclass(slots=True, frozen=True)
class CorporateAction:
    """One corporate event applied to one symbol on its ex-date."""

    timestamp: datetime
    symbol: str
    kind: ActionKind
    ratio: float = 1.0
    cash: float = 0.0

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if self.cash < 0:
            raise ValueError("cash must be non-negative")


def actions_frame(actions: list[CorporateAction]) -> pl.DataFrame:
    """Build an ``ACTIONS_SCHEMA``-conforming frame from a list of actions."""
    if not actions:
        return pl.DataFrame(schema=ACTIONS_SCHEMA)
    return pl.DataFrame(
        {
            "timestamp": [a.timestamp for a in actions],
            "symbol": [a.symbol for a in actions],
            "kind": [a.kind for a in actions],
            "ratio": [a.ratio for a in actions],
            "cash": [a.cash for a in actions],
        },
        schema=ACTIONS_SCHEMA,
    )
