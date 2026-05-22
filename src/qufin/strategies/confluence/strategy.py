"""ConfluenceStrategy — orchestrates regime, signals, GEX, and sizing.

Drop into ``qufin.trading.BacktestEngine`` (paper or historical) without
modification. The strategy keeps per-symbol state across bars so the
HMM regime only refits every ``regime_refit_period`` bars (it dominates
compute cost).

GEX chain refresh is pluggable: pass a ``GEXChainProvider`` callable to
the constructor for paper-trading, or leave it None during a backtest
where chain history is not available.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from qufin.options import OptionChain
from qufin.trading import Fill, Order, Signal, SignalKind
from qufin.trading.strategy.base import StrategyBase, StrategyContext
from qufin.wyckoff import OHLCV

from .gex_overlay import GEXDefense, GEXFlags
from .params import ConfluenceParams
from .regime import RegimeClassifier, RegimeResult
from .signals import ConfluenceSignalEngine, ExitReason, SignalFrame
from .sizing import KellyCovarianceSizer, SymbolEdge

_OHLCV_COLS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


class GEXChainProvider(Protocol):
    def __call__(self, *, symbol: str, asof: object) -> OptionChain | None: ...


@dataclass(slots=True)
class _SymbolState:
    last_regime_at: int = -1
    regime: RegimeResult | None = None
    last_signal_at: int = -1
    signals: SignalFrame | None = None


@dataclass(slots=True)
class ConfluenceStrategy(StrategyBase):
    """Wyckoff-Hull confluence with GEX defense and Kelly+LW sizing."""

    params: ConfluenceParams
    symbols: list[str]
    gex_chain_provider: GEXChainProvider | None = None
    signal_engine: ConfluenceSignalEngine = field(init=False)
    regime_classifier: RegimeClassifier = field(init=False)
    sizer: KellyCovarianceSizer = field(init=False)
    gex_defense: GEXDefense = field(init=False)
    _state: dict[str, _SymbolState] = field(default_factory=lambda: defaultdict(_SymbolState))
    _trade_returns: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _entry_marks: dict[str, float] = field(default_factory=lambda: {})
    _last_gex_flags: GEXFlags = field(default_factory=GEXFlags)

    def __post_init__(self) -> None:
        self.signal_engine = ConfluenceSignalEngine(self.params)
        self.regime_classifier = RegimeClassifier(
            warmup_bars=self.params.regime_warmup_bars,
            refit_period=self.params.regime_refit_period,
            bear_threshold=self.params.regime_distribution_threshold,
            persistence_bars=self.params.regime_persistence_bars,
        )
        self.sizer = KellyCovarianceSizer(self.params)
        self.gex_defense = GEXDefense(put_wall_buffer_pct=self.params.gex_put_wall_buffer_pct)

    # ----- Strategy lifecycle -------------------------------------------------

    def on_start(self, ctx: StrategyContext) -> None:
        del ctx

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        p = self.params
        gex_flags = self._maybe_refresh_gex(ctx)

        candidates: list[str] = []
        exits: list[str] = []
        latest_close: dict[str, float] = {}

        for sym in self.symbols:
            hist = ctx.history.get(sym)
            if hist is None or hist.height < p.regime_warmup_bars:
                continue
            bars = self._to_ohlcv(hist, sym)
            sig = self.signal_engine.evaluate(bars)
            self._state[sym].signals = sig
            self._state[sym].last_signal_at = bars.n_bars

            state = self._state[sym]
            if state.regime is None or bars.n_bars - state.last_regime_at >= p.regime_refit_period:
                state.regime = self.regime_classifier.fit_predict(bars)
                state.last_regime_at = bars.n_bars

            i = bars.n_bars - 1
            close_i = float(bars.close()[i])
            latest_close[sym] = close_i
            in_position = self._has_position(ctx, sym)

            assert state.regime is not None  # ensured by the refit gate above
            ri = min(i, state.regime.cash_defense.shape[0] - 1)
            cash_defense = bool(state.regime.cash_defense[ri])
            gex_block = gex_flags.block_new_entries

            if in_position:
                if sig.exit_reason[i] != ExitReason.NONE.value or cash_defense or gex_block:
                    exits.append(sym)
            else:
                if bool(sig.long_entry[i]) and not cash_defense and not gex_block:
                    candidates.append(sym)

        intents: list[Order | Signal] = []
        for sym in exits:
            intents.append(Signal(asset=sym, kind=SignalKind.TARGET_WEIGHT, value=0.0, tag="exit"))

        if candidates:
            ret_panel = self._returns_panel(ctx)
            edges = self._compute_edges()
            target = self.sizer.allocate(
                candidates=candidates,
                edges=edges,
                returns_panel=ret_panel,
                gex_size_multiplier=gex_flags.size_multiplier,
            )
            for sym, w in target.items():
                if w > 1e-6:
                    intents.append(
                        Signal(asset=sym, kind=SignalKind.TARGET_WEIGHT, value=w, tag="entry")
                    )
        return intents

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        del ctx
        sym = fill.asset if isinstance(fill.asset, str) else None
        if sym is None:
            return
        if fill.qty > 0 and sym not in self._entry_marks:
            self._entry_marks[sym] = fill.price
        elif fill.qty < 0 and sym in self._entry_marks:
            entry = self._entry_marks.pop(sym)
            if entry > 0.0:
                ret = (fill.price - entry) / entry
                self._trade_returns[sym].append(float(ret))

    # ----- Helpers ------------------------------------------------------------

    @staticmethod
    def _to_ohlcv(history: pl.DataFrame, symbol: str) -> OHLCV:
        cols = [c for c in _OHLCV_COLS if c in history.columns]
        return OHLCV.from_records(history.select(cols), symbol=symbol)

    @staticmethod
    def _has_position(ctx: StrategyContext, sym: str) -> bool:
        pos = ctx.positions.get(sym)
        return pos is not None and not pos.is_flat

    def _returns_panel(self, ctx: StrategyContext) -> pl.DataFrame:
        rows: list[pl.DataFrame] = []
        for sym in self.symbols:
            hist = ctx.history.get(sym)
            if hist is None or hist.height < 2:
                continue
            tail = hist.tail(self.params.cov_lookback_days + 1)
            close = tail["close"].to_numpy().astype(np.float64, copy=False)
            ret = np.diff(close) / close[:-1]
            ts = tail["timestamp"][1:]
            rows.append(pl.DataFrame({"timestamp": ts, "symbol": [sym] * ret.size, "ret": ret}))
        if not rows:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("ns", time_zone="UTC"),
                    "symbol": pl.Utf8,
                    "ret": pl.Float64,
                }
            )
        return pl.concat(rows, how="vertical_relaxed")

    def _compute_edges(self) -> dict[str, SymbolEdge]:
        edges: dict[str, SymbolEdge] = {}
        for sym in self.symbols:
            arr = np.array(self._trade_returns.get(sym, []), dtype=np.float64)
            edges[sym] = self.sizer.edge(arr, sym)
        return edges

    def _maybe_refresh_gex(self, ctx: StrategyContext) -> GEXFlags:
        if not self.params.use_gex_overlay:
            return GEXFlags()
        macro = self.params.gex_macro_symbol
        hist = ctx.history.get(macro)
        if hist is None or hist.height < 2:
            return GEXFlags()
        spot = float(hist["close"].to_numpy()[-1])
        prev = float(hist["close"].to_numpy()[-2])
        if self.gex_chain_provider is None:
            return GEXFlags()
        chain = self.gex_chain_provider(symbol=macro, asof=ctx.bar.timestamp)
        flags = self.gex_defense.refresh(
            chain=chain, spot=spot, prev_close=prev, timestamp=ctx.bar.timestamp
        )
        self._last_gex_flags = flags
        return flags


def make_strategy(
    *,
    symbols: list[str],
    params: ConfluenceParams | None = None,
    gex_chain_provider: GEXChainProvider | None = None,
) -> ConfluenceStrategy:
    """Factory wrapper so CLI scripts don't need to import the dataclass directly."""
    return ConfluenceStrategy(
        params=params or ConfluenceParams(),
        symbols=symbols,
        gex_chain_provider=gex_chain_provider,
    )
