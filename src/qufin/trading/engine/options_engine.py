"""
Option-position accounting layered on top of the equity engine.

The base ``Portfolio`` knows how to value any position given a mark price.
This module supplies marks for option positions using Black-Scholes pricing
from ``qufin.options.pricing``: given the underlying spot at the current
bar, plus per-contract IV (from the last-seen chain or a fallback constant),
each open option position is priced and its greeks are populated.

A pluggable ``MarkProvider`` lets callers override the default behaviour —
e.g. read live IV from an option-chain snapshot service, or use a
volatility surface model instead of a flat IV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from ...options._types import CALL, OptionContract
from ...options.greeks import black_scholes_greeks
from ...options.pricing import black_scholes_price
from .._types import Position, SymbolOrContract


class MarkProvider(Protocol):
    """Source of (mark_price, iv) for a single option contract at a moment in time."""

    def mark(
        self, *, contract: OptionContract, spot: float, timestamp: datetime
    ) -> tuple[float, float]: ...


@dataclass(slots=True)
class FlatIVMarkProvider:
    """Mark each contract via Black-Scholes using a single flat implied vol.

    Fast, deterministic, and adequate for backtests when only a strategy's
    PnL on the underlying matters and the option leg is held briefly.
    """

    iv: float = 0.25
    r: float = 0.0
    q: float = 0.0
    day_count: float = 365.25

    def mark(
        self, *, contract: OptionContract, spot: float, timestamp: datetime
    ) -> tuple[float, float]:
        days = (contract.expiry - timestamp.date()).days
        tau = max(days, 0) / self.day_count
        if tau <= 0.0:
            intrinsic = max(spot - contract.strike, 0.0) if contract.option_type == CALL \
                else max(contract.strike - spot, 0.0)
            return intrinsic, self.iv
        price = float(
            black_scholes_price(
                S=spot,
                K=contract.strike,
                T=tau,
                r=self.r,
                q=self.q,
                sigma=self.iv,
                option_type=contract.option_type,
            )
        )
        return price, self.iv


@dataclass(slots=True)
class OptionsEngine:
    """Marks option positions and updates their greeks each step."""

    mark_provider: MarkProvider = field(default_factory=FlatIVMarkProvider)
    r: float = 0.0
    q: float = 0.0
    day_count: float = 365.25

    def mark_options(
        self,
        *,
        timestamp: datetime,
        positions: dict[SymbolOrContract, Position],
        equity_marks: dict[SymbolOrContract, float],
    ) -> dict[SymbolOrContract, float]:
        """Return mark price for every open option position.

        The strategy must hold the underlying's symbol in ``equity_marks``
        (typically the latest close from the bar stream). Contracts whose
        underlying is missing are skipped.
        """
        out: dict[SymbolOrContract, float] = {}
        for asset, pos in positions.items():
            if not isinstance(asset, OptionContract) or pos.qty == 0.0:
                continue
            spot = equity_marks.get(asset.underlying)
            if spot is None or not np.isfinite(spot):
                continue
            mark, iv = self.mark_provider.mark(
                contract=asset, spot=spot, timestamp=timestamp
            )
            out[asset] = mark
            self._populate_greeks(pos, contract=asset, spot=spot, iv=iv, timestamp=timestamp)
        return out

    def _populate_greeks(
        self,
        pos: Position,
        *,
        contract: OptionContract,
        spot: float,
        iv: float,
        timestamp: datetime,
    ) -> None:
        days = (contract.expiry - timestamp.date()).days
        tau = max(days, 0) / self.day_count
        if tau <= 0.0:
            pos.delta = pos.gamma = pos.vega = pos.theta = 0.0
            return
        greeks = black_scholes_greeks(
            S=np.array([spot]),
            K=np.array([contract.strike]),
            T=np.array([tau]),
            r=self.r,
            q=self.q,
            sigma=np.array([iv]),
            option_type=contract.option_type,
        )
        pos.delta = float(greeks.delta[0])
        pos.gamma = float(greeks.gamma[0])
        pos.vega = float(greeks.vega[0])
        pos.theta = float(greeks.theta[0])
