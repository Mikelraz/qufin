"""
yfinance option-chain loader.

yfinance returns pandas DataFrames; we convert immediately to polars and
coerce to the canonical chain schema.

Limitations
-----------
* yfinance OI is end-of-day and routinely stale by a session or two.
* IV reported by yfinance is unreliable — pass ``solve_iv=True`` to recompute
  via ``qufin.options.iv.implied_vol_chain`` from the chain's mid prices.
* No dividend yield is fetched; pass ``q=...`` if you need it (e.g. ~0.013 for
  SPY at the time of writing).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .._types import CALL, PUT, OptionChain

if TYPE_CHECKING:
    import pandas as pd


def _coerce_side(df: pd.DataFrame, option_type: str) -> pl.DataFrame:
    pdf = df.copy()
    pdf["option_type"] = option_type
    pl_df = pl.from_pandas(pdf)
    rename_map = {
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "lastPrice": "last",
        "volume": "volume",
        "openInterest": "open_interest",
        "impliedVolatility": "iv",
    }
    out = pl_df.select(
        *(
            pl.col(src).alias(dst) if src in pl_df.columns else pl.lit(0.0).alias(dst)
            for src, dst in rename_map.items()
        ),
        pl.col("option_type"),
    )
    return out.with_columns(
        pl.col("volume").fill_null(0).cast(pl.Int64),
        pl.col("open_interest").fill_null(0).cast(pl.Int64),
        pl.col("bid").fill_null(0.0).cast(pl.Float64),
        pl.col("ask").fill_null(0.0).cast(pl.Float64),
        pl.col("last").fill_null(0.0).cast(pl.Float64),
        pl.col("iv").fill_null(0.0).cast(pl.Float64),
        pl.col("strike").cast(pl.Float64),
    )


def load_chain_yfinance(
    ticker: str,
    *,
    expiries: list[str] | None = None,
    r: float = 0.0,
    q: float = 0.0,
    multiplier: float = 100.0,
    solve_iv: bool = True,
    as_of: date | None = None,
) -> OptionChain:
    """
    Fetch a multi-expiry option chain for ``ticker`` via yfinance.

    Parameters
    ----------
    ticker      Underlying symbol (e.g. ``"SPY"``).
    expiries    Specific expiry strings (``"YYYY-MM-DD"``); ``None`` loads all.
    r, q        Continuously compounded rates.  yfinance does not provide
                these — supply them explicitly if your analysis needs them.
    multiplier  Contract multiplier (100 for US equity options).
    solve_iv    Recompute IV from mid prices.  Recommended — yfinance IV is
                often stale or zero.
    as_of       Snapshot date.  Defaults to today (local).

    Returns
    -------
    OptionChain with all calls and puts across all requested expiries.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is required: `uv add yfinance`") from e

    tk = yf.Ticker(ticker)
    available = list(tk.options or ())
    if not available:
        raise RuntimeError(f"yfinance returned no expiries for {ticker!r}")

    targets = expiries if expiries is not None else available
    missing = [e for e in targets if e not in available]
    if missing:
        raise ValueError(f"Unknown expiries for {ticker!r}: {missing}")

    info = tk.fast_info
    spot = float(info["last_price"])

    frames: list[pl.DataFrame] = []
    for exp_str in targets:
        chain = tk.option_chain(exp_str)
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        for side, opt_type in ((chain.calls, CALL), (chain.puts, PUT)):
            if side is None or side.empty:
                continue
            side_pl = _coerce_side(side, opt_type)
            side_pl = side_pl.with_columns(pl.lit(exp_date).alias("expiry"))
            frames.append(side_pl)

    if not frames:
        raise RuntimeError(f"No option rows returned for {ticker!r}")

    raw = pl.concat(frames, how="vertical_relaxed")
    snapshot = as_of if as_of is not None else date.today()

    out = OptionChain.from_records(
        raw,
        spot=spot,
        as_of=snapshot,
        underlying=ticker,
        r=r,
        q=q,
        multiplier=multiplier,
    )

    if solve_iv:
        from ..iv import implied_vol_chain

        iv = implied_vol_chain(out, use_mid=True)
        iv = np.where(np.isnan(iv) | (iv <= 0.0), out.implied_vols(), iv)
        out = OptionChain(
            data=out.data.with_columns(pl.Series("iv", iv)),
            spot=out.spot,
            as_of=out.as_of,
            underlying=out.underlying,
            r=out.r,
            q=out.q,
            multiplier=out.multiplier,
        )
    return out
