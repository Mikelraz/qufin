"""VSA Markov test: do divergence states predict market reversals?

Hypothesis: Volume Spread Analysis (VSA) divergence states -- where price and
volume move in opposite directions -- carry predictive information about the
direction of the next bar.  A pure random walk would produce uniform transition
rows (~25 % each for 4 states); a real predictive signal would show divergence
rows skewed toward the opposing regime.

States (0-indexed internally, displayed as 1-4)
------------------------------------------------
0  Bullish Validation  (dP > 0, dV > 0)
1  Bullish Divergence  (dP > 0, dV < 0)   <- exhaustion candidate
2  Bearish Validation  (dP < 0, dV > 0)
3  Bearish Divergence  (dP < 0, dV < 0)   <- exhaustion candidate

Flat days (dP = 0 or dV = 0) are dropped before encoding.

Usage
-----
    uv run python scripts/vsa_markov_test.py
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qufin.markov import ChainFit, HigherOrderFit, fit_chain, fit_higher_order

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKER = "SPY"
YEARS = 5
N_STATES = 4
HO_ORDER = 2

STATE_LABELS: dict[int, str] = {
    0: "Bullish Validation  (dP>0, dV>0)",
    1: "Bullish Divergence  (dP>0, dV<0)",
    2: "Bearish Validation  (dP<0, dV>0)",
    3: "Bearish Divergence  (dP<0, dV<0)",
}

DIVERGENCE_STATES: dict[int, str] = {
    1: "Bullish Divergence",
    3: "Bearish Divergence",
}

SECTION = "=" * 70


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def fetch_data(ticker: str = TICKER, years: int = YEARS) -> pl.DataFrame:
    """Download *years* of daily OHLCV data for *ticker* via yfinance.

    Falls back to a reproducible synthetic dataset when the network is
    unavailable or the import fails.

    Returns:
        Polars DataFrame with columns: date, close, volume.
    """
    try:
        import yfinance as yf  # local import -- optional dependency path

        end = date.today()
        start = end - timedelta(days=years * 365 + 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)

        if raw.empty:
            raise RuntimeError("yfinance returned an empty DataFrame")

        # yfinance returns a pandas MultiIndex; flatten and convert immediately.
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        df = pl.from_pandas(raw[["date", "close", "volume"]])
        print(f"[data] fetched {len(df):,} rows for {ticker} via yfinance")
        return df

    except Exception as exc:  # noqa: BLE001
        print(f"[data] yfinance unavailable ({exc}); using synthetic mock dataset")
        return _mock_data(years)


def _mock_data(years: int) -> pl.DataFrame:
    """Generate synthetic GBM price + log-normal volume series."""
    rng = np.random.default_rng(42)
    n = years * 252
    log_returns = rng.normal(0.0003, 0.012, size=n)
    close = 400.0 * np.exp(np.cumsum(log_returns))
    volume = rng.lognormal(mean=19.0, sigma=0.4, size=n).astype(np.float64)
    dates = pl.date_range(
        start=date(2020, 1, 2),
        end=date(2020, 1, 2) + timedelta(days=n - 1),
        interval="1bd",
        eager=True,
    )
    # date_range with business days can produce more rows; trim to n.
    return pl.DataFrame({"date": dates[:n], "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily dP (close pct change) and dV (volume pct change).

    Rows where either change is exactly zero are dropped; they cannot be
    assigned to a VSA quadrant without an arbitrary tie-breaking rule.

    Returns:
        DataFrame with original columns plus delta_p and delta_v, sorted by
        date, with the first row (no lag) and flat days removed.
    """
    df = df.sort("date")
    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("delta_p"),
            (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("delta_v"),
        ]
    )
    # Drop the first row (null lag) and any flat day.
    df = df.drop_nulls(subset=["delta_p", "delta_v"])
    df = df.filter((pl.col("delta_p") != 0.0) & (pl.col("delta_v") != 0.0))
    print(f"[features] {len(df):,} tradeable bars after dropping flat days")
    return df


# ---------------------------------------------------------------------------
# VSA state mapping
# ---------------------------------------------------------------------------


def map_vsa_states(df: pl.DataFrame) -> NDArray[np.intp]:
    """Discretize each bar into a VSA micro-state (0-3).

    Conditions use the sign of delta_p and delta_v:
        0  dP > 0, dV > 0  -> Bullish Validation
        1  dP > 0, dV < 0  -> Bullish Divergence
        2  dP < 0, dV > 0  -> Bearish Validation
        3  dP < 0, dV < 0  -> Bearish Divergence

    Returns:
        Integer array of shape (T,) with values in {0, 1, 2, 3}.
    """
    dp = df["delta_p"].to_numpy()
    dv = df["delta_v"].to_numpy()

    states = np.empty(len(dp), dtype=np.intp)
    states[(dp > 0) & (dv > 0)] = 0
    states[(dp > 0) & (dv < 0)] = 1
    states[(dp < 0) & (dv > 0)] = 2
    states[(dp < 0) & (dv < 0)] = 3

    counts = np.bincount(states, minlength=N_STATES)
    print("[states] distribution across VSA micro-states:")
    for s, label in STATE_LABELS.items():
        pct = 100.0 * counts[s] / len(states)
        print(f"         State {s + 1}  {label}  n={counts[s]:,}  ({pct:.1f} %)")

    return states


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_models(states: NDArray[np.intp]) -> tuple[ChainFit, HigherOrderFit]:
    """Fit a first-order and order-2 Markov chain to *states*.

    Returns:
        Tuple of (ChainFit, HigherOrderFit).
    """
    chain = fit_chain(states, n_states=N_STATES)
    ho = fit_higher_order(states, n_states=N_STATES, order=HO_ORDER)
    print(f"\n[models] first-order log-likelihood : {chain.log_likelihood:.2f}")
    print(f"[models] order-{HO_ORDER} log-likelihood      : {ho.log_likelihood:.2f}")
    return chain, ho


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _fmt_matrix_row(row: NDArray[np.float64], width: int = 8) -> str:
    return "  ".join(f"{v:{width}.4f}" for v in row)


def print_transition_matrix(mat: NDArray[np.float64], title: str) -> None:
    """Print a square transition matrix with state labels."""
    print(f"\n{title}")
    print("-" * len(title))
    header = "From \\ To  " + "  ".join(f" State {j + 1} " for j in range(N_STATES))
    print(header)
    for i, row in enumerate(mat):
        print(f"  State {i + 1}  | {_fmt_matrix_row(row)}")


def print_divergence_analysis(chain_fit: ChainFit, ho_fit: HigherOrderFit) -> None:
    """Isolate and interpret divergence-state outgoing probabilities."""
    print(f"\n{SECTION}")
    print("DIVERGENCE STATE ANALYSIS")
    print(SECTION)
    print(
        "Null hypothesis (random walk): each outgoing probability ~= 0.25 for all states.\n"
        "A predictive signal shows divergence rows skewed toward the opposing regime.\n"
    )

    # First-order: rows for states 1 and 3 (0-indexed).
    print("--- First-order chain ---")
    for s, name in DIVERGENCE_STATES.items():
        row = chain_fit.transition_matrix[s]
        print(f"\n  {name} (State {s + 1}) transition probabilities:")
        for j, label in STATE_LABELS.items():
            bar = "#" * int(row[j] * 40)
            print(f"    -> State {j + 1}  {row[j]:.4f}  {bar}  {label}")

        # Simple reversal check: does the opposing regime (states 2+3 for bull
        # divergence, states 0+1 for bear divergence) have combined p > 0.50?
        if s == 1:  # Bullish Divergence -> expect bearish follow-through
            reversal_p = row[2] + row[3]
            label_rev = "bearish (States 3+4)"
        else:  # Bearish Divergence -> expect bullish follow-through
            reversal_p = row[0] + row[1]
            label_rev = "bullish (States 1+2)"
        verdict = "SIGNAL" if reversal_p > 0.50 else "no edge"
        print(f"    Combined {label_rev}: {reversal_p:.4f}  [{verdict}]")

    # Higher-order: marginalise over the first context state for each
    # divergence state as the *most recent* context element.
    print(f"\n--- Order-{HO_ORDER} chain (marginalised over oldest context state) ---")
    tensor = ho_fit.transition_tensor  # shape (S, S, S) for order=2
    for s, name in DIVERGENCE_STATES.items():
        # Average P(next | *, s) across the first axis (oldest context state).
        marginal_row: NDArray[np.float64] = tensor[:, s, :].mean(axis=0)
        print(f"\n  {name} (State {s + 1}), most-recent context - marginal next-state probs:")
        for j, label in STATE_LABELS.items():
            bar = "#" * int(marginal_row[j] * 40)
            print(f"    -> State {j + 1}  {marginal_row[j]:.4f}  {bar}  {label}")

        if s == 1:
            reversal_p = float(marginal_row[2] + marginal_row[3])
            label_rev = "bearish (States 3+4)"
        else:
            reversal_p = float(marginal_row[0] + marginal_row[1])
            label_rev = "bullish (States 1+2)"
        verdict = "SIGNAL" if reversal_p > 0.50 else "no edge"
        print(f"    Combined {label_rev}: {reversal_p:.4f}  [{verdict}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(SECTION)
    print(f"VSA MARKOV TEST  --  ticker={TICKER}  years={YEARS}  HO order={HO_ORDER}")
    print(SECTION)

    # Step 1: data
    df = fetch_data(TICKER, YEARS)

    # Step 2: features
    df = engineer_features(df)

    # Step 3: state mapping
    states = map_vsa_states(df)

    # Step 4: model fitting
    chain_fit, ho_fit = fit_models(states)

    # Step 5: analysis output
    print(f"\n{SECTION}")
    print("TRANSITION MATRICES")
    print(SECTION)

    print_transition_matrix(chain_fit.transition_matrix, "First-order Markov chain  (S x S)")

    # For order-2: show the 4x4 slice for each conditioning context -- only the
    # divergence contexts to keep output concise.  Full tensor is (4, 4, 4).
    for ctx in range(N_STATES):
        slice_mat = ho_fit.transition_tensor[ctx]  # shape (S, S), context = ctx
        print_transition_matrix(
            slice_mat,
            f"Order-{HO_ORDER} chain  context[t-2] = State {ctx + 1}: {STATE_LABELS[ctx]}",
        )

    print_divergence_analysis(chain_fit, ho_fit)

    print(f"\n{SECTION}")
    print("DONE")
    print(SECTION)


if __name__ == "__main__":
    main()
