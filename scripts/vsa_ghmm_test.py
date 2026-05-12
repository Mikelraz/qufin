"""Continuous GHMM test for VSA Regimes.

This script feeds continuous, normalized features (Z-scores of returns and volume)
into our Gaussian HMM engine. Instead of forcing data into discrete boxes, it allows
the Baum-Welch algorithm to discover regimes based on continuous clustering.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl

# Import from the continuous engine we just built
from qufin.markov.ghmm import decode, fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "META"]
YEARS = 5
N_HIDDEN_STATES = 4  # K: The macro regimes
ROLLING_WINDOW = 20  # 20-day rolling window for Z-score normalization

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def fetch_pooled_data(tickers: list[str], years: int) -> pl.DataFrame:
    """Download OHLCV data for multiple tickers and stack into a long format."""
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=years * 365 + 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=str(start), end=str(end), progress=False)

    if raw.empty:
        raise RuntimeError("yfinance returned an empty DataFrame")

    raw = raw.stack(level=1, future_stack=True).reset_index()
    raw.columns = [str(c).lower() for c in raw.columns]

    df = pl.from_pandas(raw[["date", "ticker", "close", "volume"]])
    print(f"[data] Fetched {len(df):,} rows for {len(tickers)} tickers.")
    return df


def engineer_continuous_features(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Calculate rolling Z-scores for returns and volume, grouped by ticker."""
    df = df.sort(["ticker", "date"])

    # 1. Calculate raw changes
    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("dp"),
            (pl.col("volume") / pl.col("volume").shift(1).over("ticker") - 1.0).alias("dv"),
        ]
    )

    # 2. Calculate rolling Z-scores strictly per ticker
    df = df.with_columns(
        [
            (
                (pl.col("dp") - pl.col("dp").rolling_mean(window).over("ticker"))
                / pl.col("dp").rolling_std(window).over("ticker")
            ).alias("z_ret"),
            (
                (pl.col("dv") - pl.col("dv").rolling_mean(window).over("ticker"))
                / pl.col("dv").rolling_std(window).over("ticker")
            ).alias("z_vol"),
        ]
    )

    # Drop rows that have nulls (the first `window` days of each ticker)
    df = df.drop_nulls(subset=["z_ret", "z_vol"])
    print(
        f"[features] {len(df):,} tradeable bars after dropping initial "
        f"{window}-day rolling windows."
    )
    return df


def print_matrix(mat: np.ndarray, title: str) -> None:
    """Helper to format and print stochastic matrices."""
    print(f"\n--- {title} ---")
    for i, row in enumerate(mat):
        formatted_row = "  ".join(f"{val:7.4f}" for val in row)
        print(f" State {i} | {formatted_row}")


def analyze_hidden_states(means: np.ndarray) -> None:
    """Print the learned means to interpret the discovered regimes."""
    print("\n--- DISCOVERED REGIMES (EMISSION MEANS) ---")
    print("Interpret these to label your hidden states!")
    print("          |  Z-Return (μ)  |  Z-Volume (μ) ")
    for k in range(means.shape[0]):
        print(f" State {k:d}  | {means[k, 0]:14.4f} | {means[k, 1]:13.4f}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def main() -> None:
    print("==========================================================")
    print(" VSA CONTINUOUS GHMM TEST (POOLED MULTI-TICKER) ")
    print("==========================================================")

    # 1. Prep Data
    df = fetch_pooled_data(TICKERS, YEARS)
    df = engineer_continuous_features(df, ROLLING_WINDOW)

    # 2. Extract 2D continuous observation array (T, 2)
    obs_seq = df.select(["z_ret", "z_vol"]).to_numpy().astype(np.float64)

    print(f"\n[model] Fitting Gaussian Baum-Welch EM on {len(obs_seq):,} observations...")
    print(f"[model] K={N_HIDDEN_STATES} hidden states, D=2 continuous features")

    # 3. Fit the GHMM
    model_fit = fit(
        obs=obs_seq,
        n_states=N_HIDDEN_STATES,
        n_init=10,
        max_iter=300,
        verbose=True,  # <-- Trigger the progress logging
    )

    print(f"\n[result] Converged in {model_fit.n_iter} iterations.")
    print(f"[result] Final Log-Likelihood: {model_fit.log_likelihood:.2f}")

    # 4. Analyze the learned parameters
    print_matrix(model_fit.transition_matrix, "TRANSITION MATRIX (A)")

    # The means are the most important part of the continuous HMM output.
    # They tell you exactly what VSA phase the state represents.
    analyze_hidden_states(model_fit.means)

    # 5. Decode a single asset to see the sequence
    sample_ticker = TICKERS[0]
    sample_df = df.filter(pl.col("ticker") == sample_ticker)
    sample_obs = sample_df.select(["z_ret", "z_vol"]).to_numpy().astype(np.float64)

    hidden_path = decode(sample_obs, model_fit)

    print(f"\n[viterbi] Decoded hidden path for {sample_ticker} (last 5 days):")
    for i in range(1, 6):
        idx = -i
        ret = sample_obs[idx, 0]
        vol = sample_obs[idx, 1]
        state = hidden_path[idx]
        dt = sample_df["date"][len(sample_df) + idx]
        print(f" {dt} | State {state} (z_ret: {ret:6.2f}, z_vol: {vol:6.2f})")


if __name__ == "__main__":
    main()
