"""Portfolio analysis and optimization tools.

Submodules
----------
returns
    Price-to-return transformations (simple, log, cumulative, annualized).
metrics
    Risk and performance metrics: Sharpe, Sortino, Calmar, drawdown, VaR, CVaR.
covariance
    Covariance estimators: sample, Ledoit-Wolf shrinkage, exponentially weighted.
optimize
    Portfolio optimizers: min-variance, max-Sharpe, risk parity, efficient frontier.

Typical workflow::

    from qufin.portfolio import (
        simple_returns, to_returns_matrix,
        annualized_returns,
        ledoit_wolf_cov, annualize_cov,
        max_sharpe, efficient_frontier,
    )

    ret_df = simple_returns(prices_df)
    mat, names = to_returns_matrix(ret_df)
    mu_map = annualized_returns(ret_df, periods_per_year=252)
    mu = np.array([mu_map[n] for n in names])
    cov = annualize_cov(ledoit_wolf_cov(mat), 252)

    result = max_sharpe(mu, cov, names, risk_free_rate=0.04)
    ef     = efficient_frontier(mu, cov, names, n_points=60)
"""

from qufin.portfolio._types import EfficientFrontier, OptimizationResult
from qufin.portfolio.covariance import (
    annualize_cov,
    cov_to_corr,
    ewm_cov,
    ledoit_wolf_cov,
    sample_cov,
)
from qufin.portfolio.metrics import (
    annualized_volatility,
    calmar_ratio,
    conditional_var,
    historical_var,
    max_drawdown,
    portfolio_metrics,
    sharpe_ratio,
    sortino_ratio,
)
from qufin.portfolio.optimize import (
    efficient_frontier,
    efficient_return,
    max_sharpe,
    min_variance,
    risk_parity,
)
from qufin.portfolio.returns import (
    annualize_return,
    annualized_returns,
    cumulative_returns,
    log_returns,
    simple_returns,
    to_returns_matrix,
)

__all__ = [
    # types
    "OptimizationResult",
    "EfficientFrontier",
    # returns
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "annualize_return",
    "annualized_returns",
    "to_returns_matrix",
    # metrics
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "historical_var",
    "conditional_var",
    "portfolio_metrics",
    # covariance
    "sample_cov",
    "ledoit_wolf_cov",
    "ewm_cov",
    "annualize_cov",
    "cov_to_corr",
    # optimization
    "min_variance",
    "max_sharpe",
    "efficient_return",
    "efficient_frontier",
    "risk_parity",
]
