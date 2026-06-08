"""
IBKR option-chain loader.

Pulls every (expiry, strike, side) listed for a symbol within configurable
DTE / strike-band windows, fetches delayed bid/ask/IV/OI/volume per contract
via concurrent ``reqMktData`` subscriptions, and assembles an ``OptionChain``
matching ``CHAIN_SCHEMA`` so the existing GEX/walls/flip toolkit works
unchanged.

Why concurrent reqMktData and not snapshots?
--------------------------------------------
IBKR's ``snapshot=True`` mode (a) requires a real-time data subscription on
many symbols and (b) is rate-limited to ~one-per-second. For full chains
(often 200-600 contracts) that's prohibitively slow. The streaming approach
subscribes to many contracts at once, waits for the delayed feed to land
ticks, reads them, then unsubscribes. Paper Gateway allows ~100 concurrent
market-data lines; we batch under that ceiling.

Data quality notes
------------------
* Delayed (15 min) bid/ask/IV/OI is normal for paper accounts. The GEX math
  is unchanged — only the freshness of OI matters, and OI is reported daily
  by OCC anyway, so end-of-prior-day OI is what every retail GEX tool uses.
* IBKR's ``modelGreeks.impliedVol`` is computed off the mark price using the
  exchange's risk-free curve. We use it directly. If you'd rather solve IV
  from mid prices yourself, pass ``solve_iv=True`` and the existing
  ``qufin.options.iv.implied_vol_chain`` will be called.
* Open interest is filled from ``ticker.callOpenInterest`` / ``putOpenInterest``
  (depending on side). If both are missing we fall back to 0 — those contracts
  contribute nothing to GEX, which is the right behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from ...data.vendors._ib_common import (
    IBKRErrorListener,
    MarketDataType,
    connect_ib,
    safe_float,
    snapshot,
    snapshot_many,
    ymd_to_date,
)
from .._types import CALL, PUT, OptionChain


@dataclass(slots=True)
class IBKRChainLoader:
    """Streaming IBKR option-chain loader.

    Parameters
    ----------
    host, port, client_id     Gateway/TWS connection.
    use_delayed               If True, request DELAYED_FROZEN market data
                              (works on paper without a live subscription).
    settle_seconds            Seconds to wait between subscribing a batch
                              and reading the tickers. 8-12s is typical for
                              delayed feeds.
    batch_size                Max concurrent ``reqMktData`` subscriptions
                              per batch. Paper Gateway allows ~100; keep
                              some headroom for the broker connection.
    pacing_seconds            Pause between batches.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 97
    use_delayed: bool = True
    settle_seconds: float = 10.0
    batch_size: int = 60
    pacing_seconds: float = 1.0

    async def load(
        self,
        symbol: str,
        *,
        min_dte: int = 0,
        max_dte: int = 90,
        max_expiries: int | None = 6,
        strike_pct_band: float = 0.30,
        include_puts: bool = True,
        r: float = 0.045,
        q: float = 0.0,
        solve_iv: bool = False,
    ) -> OptionChain:
        """Fetch a chain into an ``OptionChain``.

        Parameters
        ----------
        symbol            Underlying ticker.
        min_dte/max_dte   DTE window for expiries to include.
        max_expiries      Hard cap on number of expiries (closest-first).
        strike_pct_band   Strikes within ``[spot*(1-band), spot*(1+band)]``.
        include_puts      If False, calls only (faster but breaks dealer-sign math).
        r, q              Rates passed through to ``OptionChain``.
        solve_iv          If True, re-solve IV from mid prices via
                          ``qufin.options.iv.implied_vol_chain``. Otherwise
                          use IBKR's model IV directly.
        """
        try:
            from ib_async import Option, Stock
        except ImportError as e:
            raise ImportError(
                "ib_async is required for IBKR chain loading. Run: uv sync --group trading-live"
            ) from e

        # DELAYED_FROZEN returns last-known delayed values even when the feed is
        # idle (e.g. lunchtime). Pass use_delayed=False for strictly fresh ticks.
        listener = IBKRErrorListener()
        ib = await connect_ib(
            self.host,
            self.port,
            self.client_id,
            market_data_type=MarketDataType.DELAYED_FROZEN if self.use_delayed else None,
            listener=listener,
        )
        try:
            # 1. Spot.
            stock = Stock(symbol, "SMART", "USD")
            await ib.qualifyContractsAsync(stock)
            spot_ticker = await snapshot(ib, stock, settle_seconds=self.settle_seconds)
            spot = (
                safe_float(spot_ticker.last)
                or safe_float(spot_ticker.close)
                or safe_float(spot_ticker.bid)
            )
            if spot is None or spot <= 0:
                raise RuntimeError(
                    f"could not fetch spot for {symbol!r} "
                    "(competing live session? options permissions?)"
                )

            # 2. Chain parameters.
            params_list = await ib.reqSecDefOptParamsAsync(symbol, "", "STK", stock.conId)
            if not params_list:
                raise RuntimeError(f"no option chain returned for {symbol!r}")
            pref = next((p for p in params_list if p.exchange == "SMART"), params_list[0])

            today = datetime.now(tz=UTC).date()
            all_expiries = sorted(ymd_to_date(e) for e in pref.expirations)
            window_exps = [e for e in all_expiries if min_dte <= (e - today).days <= max_dte]
            if max_expiries is not None and len(window_exps) > max_expiries:
                window_exps = window_exps[:max_expiries]
            if not window_exps:
                raise RuntimeError(f"no expiries in [{min_dte}, {max_dte}] DTE for {symbol!r}")

            lo = spot * (1.0 - strike_pct_band)
            hi = spot * (1.0 + strike_pct_band)
            strikes = sorted(k for k in pref.strikes if lo <= k <= hi)
            if not strikes:
                raise RuntimeError(f"no strikes in ±{strike_pct_band:.0%} of spot ${spot:.2f}")

            sides = [CALL, PUT] if include_puts else [CALL]
            print(
                f"  {symbol}: spot=${spot:.2f}  "
                f"{len(window_exps)} expiries  {len(strikes)} strikes  "
                f"{len(sides)} side(s)  "
                f"-> {len(window_exps) * len(strikes) * len(sides)} contracts"
            )

            # 3. Build + qualify all contracts in one batch.
            candidate_contracts: list[Any] = []
            for exp in window_exps:
                for k in strikes:
                    for side in sides:
                        candidate_contracts.append(
                            Option(symbol, exp.strftime("%Y%m%d"), float(k), side, "SMART")
                        )
            qualified = await ib.qualifyContractsAsync(*candidate_contracts)
            valid = [c for c in qualified if getattr(c, "conId", 0)]
            if not valid:
                raise RuntimeError("no contracts qualified")
            print(f"  qualified {len(valid)} / {len(candidate_contracts)} contracts")

            # 4. Stream market data in batches. Tick types: 100=OptionVolume,
            # 101=OptionOpenInterest, 106=ImpliedVol, 165=MiscStats.
            print(f"  streaming quotes for {len(valid)} contracts...", flush=True)
            tickers = await snapshot_many(
                ib,
                valid,
                generic_ticks="100,101,106,165",
                settle_seconds=self.settle_seconds,
                batch_size=self.batch_size,
                pacing_seconds=self.pacing_seconds,
            )
            rows = [self._ticker_to_row(c, t) for c, t in zip(valid, tickers, strict=True)]

        finally:
            listener.detach()
            ib.disconnect()

        # 5. Coerce into OptionChain.
        if not rows:
            raise RuntimeError("no rows returned")
        df = pl.DataFrame(rows)

        # Drop contracts with no usable IV — they contribute NaNs to Greeks
        # and corrupt aggregate exposure. (Typically deep OTM with no quotes.)
        before = len(df)
        df = df.filter(pl.col("iv").is_not_null() & (pl.col("iv") > 0.0))
        if len(df) < before:
            print(f"  dropped {before - len(df)} contracts with missing IV")

        chain = OptionChain.from_records(
            df,
            spot=spot,
            as_of=today,
            underlying=symbol,
            r=r,
            q=q,
        )

        if solve_iv:
            from ..iv import implied_vol_chain

            iv = implied_vol_chain(chain, use_mid=True)
            iv = np.where(np.isnan(iv) | (iv <= 0.0), chain.implied_vols(), iv)
            chain = OptionChain(
                data=chain.data.with_columns(pl.Series("iv", iv)),
                spot=chain.spot,
                as_of=chain.as_of,
                underlying=chain.underlying,
                r=chain.r,
                q=chain.q,
                multiplier=chain.multiplier,
            )
        return chain

    @staticmethod
    def _ticker_to_row(contract: Any, ticker: Any) -> dict[str, Any]:
        is_call = contract.right == "C"
        # OI lives on a side-specific tick.
        if is_call:
            oi = safe_float(getattr(ticker, "callOpenInterest", None))
        else:
            oi = safe_float(getattr(ticker, "putOpenInterest", None))
        if oi is None:
            oi = 0

        greeks = (
            ticker.modelGreeks
            or getattr(ticker, "lastGreeks", None)
            or getattr(ticker, "bidGreeks", None)
            or getattr(ticker, "askGreeks", None)
        )
        iv = safe_float(greeks.impliedVol) if greeks is not None else None

        bid = safe_float(ticker.bid) or 0.0
        ask = safe_float(ticker.ask) or 0.0
        last = safe_float(ticker.last) or safe_float(ticker.close) or 0.0
        vol = safe_float(ticker.volume)
        vol = 0 if vol is None else int(vol)

        return {
            "expiry": ymd_to_date(contract.lastTradeDateOrContractMonth),
            "strike": float(contract.strike),
            "option_type": CALL if is_call else PUT,
            "bid": float(bid),
            "ask": float(ask),
            "last": float(last),
            "volume": vol,
            "open_interest": int(oi),
            "iv": iv,
        }


async def load_chain_ibkr(
    symbol: str,
    *,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 97,
    **kwargs: Any,
) -> OptionChain:
    """Convenience async wrapper. Use ``IBKRChainLoader`` directly for tuning."""
    loader = IBKRChainLoader(host=host, port=port, client_id=client_id)
    return await loader.load(symbol, **kwargs)
