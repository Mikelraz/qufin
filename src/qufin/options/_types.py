"""
Shared types and result containers for the options subpackage.

All array-backed containers store ``np.ndarray`` of ``float64`` to match the
project numerical convention.  ``OptionChain`` wraps a polars DataFrame with a
fixed schema so downstream code (pricing, IV, GEX) can operate without
repeatedly re-validating columns.

Conventions
-----------
* ``option_type`` is the single ASCII character ``'C'`` or ``'P'``.
* ``expiry`` is stored as a polars ``Date`` (timezone-naive calendar date);
  computations that need a year-fraction call ``OptionChain.time_to_expiry``
  with an explicit ``as_of`` date and day-count convention.
* Risk-free ``r`` and dividend yield ``q`` are continuously compounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Self

import numpy as np
import polars as pl

CALL: str = "C"
PUT: str = "P"

CHAIN_SCHEMA: dict[str, pl.DataType] = {
    "expiry": pl.Date(),
    "strike": pl.Float64(),
    "option_type": pl.Utf8(),
    "bid": pl.Float64(),
    "ask": pl.Float64(),
    "last": pl.Float64(),
    "volume": pl.Int64(),
    "open_interest": pl.Int64(),
    "iv": pl.Float64(),
}


@dataclass(slots=True, frozen=True)
class OptionContract:
    """A single European-style option contract."""

    strike: float
    expiry: date
    option_type: str
    underlying: str = ""

    def __post_init__(self) -> None:
        if self.option_type not in (CALL, PUT):
            raise ValueError(f"option_type must be 'C' or 'P', got {self.option_type!r}")


@dataclass(slots=True)
class Greeks:
    """
    Vectorised greeks for an aligned set of contracts.

    All arrays share the same length and are stored as ``float64``.  Greeks
    follow the standard Black-Scholes convention and are reported per one
    unit of underlying (i.e. *not* scaled by the contract multiplier).
    """

    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    vanna: np.ndarray
    charm: np.ndarray
    vomma: np.ndarray
    speed: np.ndarray

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "delta": self.delta,
                "gamma": self.gamma,
                "vega": self.vega,
                "theta": self.theta,
                "rho": self.rho,
                "vanna": self.vanna,
                "charm": self.charm,
                "vomma": self.vomma,
                "speed": self.speed,
            }
        )


@dataclass(slots=True)
class OptionChain:
    """
    Polars-backed option chain snapshot.

    Attributes
    ----------
    data       Long-format polars DataFrame with the schema in ``CHAIN_SCHEMA``.
    spot       Underlying spot at snapshot time.
    as_of      Snapshot date (used to derive time-to-expiry).
    underlying Ticker symbol (informational).
    r          Continuously compounded risk-free rate.
    q          Continuously compounded dividend yield.
    multiplier Contract multiplier (default 100 for US equity options).
    """

    data: pl.DataFrame
    spot: float
    as_of: date
    underlying: str = ""
    r: float = 0.0
    q: float = 0.0
    multiplier: float = 100.0

    def __post_init__(self) -> None:
        missing = set(CHAIN_SCHEMA) - set(self.data.columns)
        if missing:
            raise ValueError(f"OptionChain data is missing columns: {sorted(missing)}")

    @classmethod
    def from_records(
        cls,
        records: pl.DataFrame,
        *,
        spot: float,
        as_of: date,
        underlying: str = "",
        r: float = 0.0,
        q: float = 0.0,
        multiplier: float = 100.0,
    ) -> Self:
        """Coerce a DataFrame to the chain schema and construct an OptionChain."""
        missing = set(CHAIN_SCHEMA) - set(records.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {sorted(missing)}")
        coerced = records.select(
            *(pl.col(name).cast(dtype) for name, dtype in CHAIN_SCHEMA.items())
        )
        return cls(
            data=coerced,
            spot=float(spot),
            as_of=as_of,
            underlying=underlying,
            r=float(r),
            q=float(q),
            multiplier=float(multiplier),
        )

    def time_to_expiry(self, *, basis: float = 365.25) -> np.ndarray:
        """Year-fraction to expiry per row, ACT/365.25 by default."""
        days = (
            self.data["expiry"].to_numpy().astype("datetime64[D]") - np.datetime64(self.as_of, "D")
        ).astype(np.int64)
        return days.astype(np.float64) / basis

    def is_call(self) -> np.ndarray:
        return self.data["option_type"].to_numpy() == CALL

    def strikes(self) -> np.ndarray:
        return self.data["strike"].to_numpy().astype(np.float64)

    def open_interest(self) -> np.ndarray:
        return self.data["open_interest"].to_numpy().astype(np.float64)

    def implied_vols(self) -> np.ndarray:
        return self.data["iv"].to_numpy().astype(np.float64)

    def mid(self) -> np.ndarray:
        """Mid price (bid+ask)/2, falling back to ``last`` when either side is missing."""
        bid = self.data["bid"].to_numpy().astype(np.float64)
        ask = self.data["ask"].to_numpy().astype(np.float64)
        last = self.data["last"].to_numpy().astype(np.float64)
        mid = 0.5 * (bid + ask)
        bad = ~np.isfinite(mid) | (bid <= 0.0) | (ask <= 0.0)
        mid = np.where(bad, last, mid)
        return mid


@dataclass(slots=True)
class GEXProfile:
    """
    Gamma-exposure profile across a grid of hypothetical spot levels.

    Attributes
    ----------
    spot_grid     Spot levels at which exposures were evaluated, shape (n_spot,).
    gex           Total dealer GEX at each spot, shape (n_spot,).
    dex           Total dealer DEX at each spot, shape (n_spot,).
    vex           Total dealer vanna exposure at each spot, shape (n_spot,).
    charm         Total dealer charm exposure at each spot, shape (n_spot,).
    flip_level    Zero-gamma flip spot (None if no sign change in range).
    spot          Reference (current) spot at snapshot.
    """

    spot_grid: np.ndarray
    gex: np.ndarray
    dex: np.ndarray
    vex: np.ndarray
    charm: np.ndarray
    flip_level: float | None
    spot: float

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "spot": self.spot_grid,
                "gex": self.gex,
                "dex": self.dex,
                "vex": self.vex,
                "charm": self.charm,
            }
        )


@dataclass(slots=True)
class StrikeExposure:
    """
    Per-strike aggregated dealer exposures at the snapshot spot.

    Each array is indexed by ``strikes`` and aggregates across expiries.
    """

    strikes: np.ndarray
    gex: np.ndarray
    dex: np.ndarray
    vex: np.ndarray
    charm: np.ndarray
    call_oi: np.ndarray
    put_oi: np.ndarray
    notes: dict[str, float] = field(default_factory=lambda: dict[str, float]())

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "strike": self.strikes,
                "gex": self.gex,
                "dex": self.dex,
                "vex": self.vex,
                "charm": self.charm,
                "call_oi": self.call_oi,
                "put_oi": self.put_oi,
            }
        )
