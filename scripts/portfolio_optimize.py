#!/usr/bin/env python3
"""Portfolio optimization CLI.

Downloads historical prices via yfinance, computes returns, estimates the
covariance matrix, runs multiple optimizers, and (optionally) plots the
efficient frontier.

Usage examples
--------------
    python scripts/portfolio_optimize.py AAPL MSFT GOOG AMZN META
    python scripts/portfolio_optimize.py SPY QQQ TLT GLD --start 2019-01-01 --rfr 0.05 --plot
    python scripts/portfolio_optimize.py NVDA TSLA MSFT --cov ledoit-wolf --method max-sharpe
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean-variance portfolio optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tickers", nargs="+", help="Asset tickers (e.g. AAPL MSFT GOOG)")
    p.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD")
    p.add_argument(
        "--method",
        choices=["min-variance", "max-sharpe", "risk-parity", "all"],
        default="all",
        help="Optimization method",
    )
    p.add_argument(
        "--cov",
        choices=["sample", "ledoit-wolf", "ewm"],
        default="ledoit-wolf",
        help="Covariance estimation method",
    )
    p.add_argument("--rfr", type=float, default=0.04, help="Annual risk-free rate")
    p.add_argument(
        "--ppy", type=int, default=252, help="Trading periods per year"
    )
    p.add_argument("--ewm-halflife", type=float, default=63.0, help="EWM halflife in days")
    p.add_argument("--plot", action="store_true", help="Plot the efficient frontier")
    return p.parse_args()


def _download_prices(tickers: list[str], start: str, end: str) -> "pl.DataFrame":
    import polars as pl
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if len(tickers) > 1 else raw[["Close"]]
    close.columns = tickers if len(tickers) > 1 else tickers  # type: ignore[assignment]
    close = close.dropna()
    df = pl.from_pandas(close.reset_index()).rename({"Date": "date", "index": "date"})
    # coerce the date column to pl.Date
    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df


def _build_cov(
    mat: "np.ndarray",
    method: str,
    halflife: float,
) -> "np.ndarray":
    from qufin.portfolio.covariance import ewm_cov, ledoit_wolf_cov, sample_cov

    match method:
        case "sample":
            return sample_cov(mat)
        case "ledoit-wolf":
            return ledoit_wolf_cov(mat)
        case "ewm":
            return ewm_cov(mat, halflife)
        case _:
            raise ValueError(f"Unknown covariance method: {method}")


def _print_result(label: str, result: "OptimizationResult", ppy: int) -> None:
    from qufin.portfolio._types import OptimizationResult  # noqa: F401

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Expected return (ann.):  {result.expected_return * 100:>7.2f} %")
    print(f"  Expected vol    (ann.):  {result.expected_volatility * 100:>7.2f} %")
    print(f"  Sharpe ratio:            {result.sharpe_ratio:>7.3f}")
    print(f"  Converged:               {result.success}")
    print()
    print(f"  {'Asset':<10} {'Weight':>8}")
    print(f"  {'─' * 20}")
    for name, w in zip(result.asset_names, result.weights, strict=True):
        print(f"  {name:<10} {w * 100:>7.2f} %")


def _plot_frontier(
    ef: "EfficientFrontier",
    *extra_results: "OptimizationResult",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        ef.volatilities * 100,
        ef.returns * 100,
        lw=2,
        color="steelblue",
        label="Efficient frontier",
    )

    colors = ["firebrick", "seagreen", "darkorange"]
    markers = ["*", "D", "^"]
    for res, color, marker in zip(extra_results, colors, markers, strict=False):
        ax.scatter(
            res.expected_volatility * 100,
            res.expected_return * 100,
            s=120,
            color=color,
            marker=marker,
            zorder=5,
            label=res.message if not res.success else _result_label(res, extra_results),
        )

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Expected Return (%)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _result_label(
    res: "OptimizationResult", all_results: "tuple[OptimizationResult, ...]"
) -> str:
    idx = all_results.index(res)
    labels = ["Min Variance", "Max Sharpe", "Risk Parity"]
    return labels[idx] if idx < len(labels) else f"Portfolio {idx}"


def main() -> None:
    import numpy as np

    from qufin.portfolio._types import EfficientFrontier, OptimizationResult  # noqa: F401
    from qufin.portfolio.covariance import annualize_cov
    from qufin.portfolio.metrics import portfolio_metrics
    from qufin.portfolio.optimize import (
        efficient_frontier,
        max_sharpe,
        min_variance,
        risk_parity,
    )
    from qufin.portfolio.returns import (
        annualized_returns,
        simple_returns,
        to_returns_matrix,
    )

    args = _parse_args()

    print(f"Downloading {args.tickers} from {args.start} to {args.end} …")
    prices = _download_prices(args.tickers, args.start, args.end)
    print(f"  {len(prices)} trading days loaded.")

    ret_df = simple_returns(prices)
    mat, names = to_returns_matrix(ret_df)

    ann_rets_map = annualized_returns(ret_df, args.ppy)
    mu = np.array([ann_rets_map[n] for n in names])

    print(f"\nBuilding covariance ({args.cov}) …")
    daily_cov = _build_cov(mat, args.cov, args.ewm_halflife)
    ann_cov = annualize_cov(daily_cov, args.ppy)

    print("\nAnnualized returns and volatilities:")
    print(f"  {'Asset':<10} {'Return':>8}  {'Vol':>8}")
    for i, name in enumerate(names):
        vol = float(np.sqrt(ann_cov[i, i]))
        print(f"  {name:<10} {mu[i] * 100:>7.2f} %  {vol * 100:>7.2f} %")

    methods = (
        ["min-variance", "max-sharpe", "risk-parity"]
        if args.method == "all"
        else [args.method]
    )
    results: list[OptimizationResult] = []

    if "min-variance" in methods:
        res = min_variance(mu, ann_cov, names, risk_free_rate=args.rfr)
        _print_result("Minimum Variance Portfolio", res, args.ppy)
        m = portfolio_metrics(res.weights, mat * np.sqrt(args.ppy), risk_free_rate=args.rfr)
        print(f"  Max drawdown (in-sample): {m['max_drawdown'] * 100:.2f} %")
        results.append(res)

    if "max-sharpe" in methods:
        res = max_sharpe(mu, ann_cov, names, risk_free_rate=args.rfr)
        _print_result("Maximum Sharpe Ratio Portfolio", res, args.ppy)
        m = portfolio_metrics(res.weights, mat * np.sqrt(args.ppy), risk_free_rate=args.rfr)
        print(f"  Max drawdown (in-sample): {m['max_drawdown'] * 100:.2f} %")
        results.append(res)

    if "risk-parity" in methods:
        res = risk_parity(ann_cov, names, expected_returns=mu, risk_free_rate=args.rfr)
        _print_result("Risk Parity Portfolio", res, args.ppy)
        m = portfolio_metrics(res.weights, mat * np.sqrt(args.ppy), risk_free_rate=args.rfr)
        print(f"  Max drawdown (in-sample): {m['max_drawdown'] * 100:.2f} %")
        results.append(res)

    if args.plot:
        print("\nTracing efficient frontier …")
        ef = efficient_frontier(mu, ann_cov, names, n_points=60, risk_free_rate=args.rfr)
        _plot_frontier(ef, *results)


if __name__ == "__main__":
    main()
