"""GEX-based macro defense overlay.

The defense reads SPY's option chain at refresh time and exposes two flags:

* ``expansion`` — SPY close is below the zero-gamma flip (dealers are short
  gamma → moves amplify). The strategy halves new entry sizing.
* ``near_put_wall`` — SPY is within ``put_wall_buffer_pct`` of the largest
  long-put strike and falling. New entries are blocked and trails tightened.

Sector ETFs without a liquid chain inherit the SPY flags (macro proxy).
The overlay is intentionally optional — when chain data is missing or
``use_gex_overlay`` is False the flags default to no-defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from qufin.options import (
    OptionChain,
    aggregate_exposure,
    put_wall,
    zero_gamma_level,
)


@dataclass(slots=True, frozen=True)
class GEXFlags:
    expansion: bool = False
    near_put_wall: bool = False
    zero_gamma: float | None = None
    put_wall_strike: float | None = None

    @property
    def block_new_entries(self) -> bool:
        return self.near_put_wall

    @property
    def size_multiplier(self) -> float:
        return 0.5 if self.expansion else 1.0


@dataclass(slots=True)
class GEXDefense:
    """Refresh-on-demand GEX flags. Idempotent within one trading day."""

    put_wall_buffer_pct: float = 0.01
    _last_refresh: datetime | None = field(default=None, init=False)
    _flags: GEXFlags = field(default_factory=GEXFlags, init=False)

    def refresh(
        self,
        *,
        chain: OptionChain | None,
        spot: float,
        prev_close: float | None = None,
        timestamp: datetime | None = None,
    ) -> GEXFlags:
        """Recompute flags from a fresh option chain + the latest SPY spot.

        ``prev_close`` is used to gate the "near put wall and falling" rule.
        If ``chain`` is None the flags clear (no defense rather than wrong
        defense).
        """
        if chain is None:
            self._flags = GEXFlags()
            self._last_refresh = timestamp
            return self._flags

        try:
            _ = aggregate_exposure(chain)  # forces input validation
            zg = zero_gamma_level(chain)
            pw = put_wall(chain)
        except Exception:
            self._flags = GEXFlags()
            self._last_refresh = timestamp
            return self._flags

        expansion = zg is not None and spot < zg
        falling = True if prev_close is None else spot < prev_close
        near_pw = (
            falling
            and abs(spot - pw) / max(spot, 1e-9) <= self.put_wall_buffer_pct
        )

        self._flags = GEXFlags(
            expansion=expansion,
            near_put_wall=near_pw,
            zero_gamma=zg,
            put_wall_strike=pw,
        )
        self._last_refresh = timestamp
        return self._flags

    @property
    def flags(self) -> GEXFlags:
        return self._flags
