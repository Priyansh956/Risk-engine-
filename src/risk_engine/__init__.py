"""Risk engine package for quantitative portfolio risk, CAPM, and Monte Carlo analytics."""

from risk_engine.capm import (
    calculate_beta,
    calculate_capm_expected_return,
    calculate_covariance_with_market,
    calculate_daily_risk_free_rate,
    calculate_excess_returns,
    calculate_market_returns,
    calculate_market_variance,
    compare_betas,
    compare_returns,
    run_capm_regression,
    summarize_residuals,
)
from risk_engine.data_loader import (
    fetch_market_data,
    fetch_risk_free_rate,
    fetch_stock_data,
)
from risk_engine.markowitz import (
    efficient_frontier,
    portfolio_expected_return,
    portfolio_metrics,
    portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
    simulate_random_portfolios,
)
from risk_engine.monte_carlo import (
    analyze_scenarios,
    calculate_daily_drift,
    simulate_gbm_prices,
    simulate_gbm_returns,
    simulate_portfolio_paths,
    summarize_terminal_prices,
)
from risk_engine.portfolio import (
    PortfolioAnalysisResult,
    analyze_portfolio,
)
from risk_engine.returns import (
    calculate_annualized_return,
    calculate_log_returns,
)
from risk_engine.risk import (
    calculate_annual_correlation_of_stocks,
    calculate_annual_covariance_of_stocks,
    calculate_annual_std_deviation_of_stocks,
    calculate_annual_variance_of_stocks,
    calculate_annual_variance_of_the_portfolio,
    calculate_sharpe_ratio_of_portfolio,
    calculate_the_annual_portfolio_return,
)

__all__ = [
    "PortfolioAnalysisResult",
    "analyze_portfolio",
    "analyze_scenarios",
    "calculate_annual_correlation_of_stocks",
    "calculate_annual_covariance_of_stocks",
    "calculate_annual_std_deviation_of_stocks",
    "calculate_annual_variance_of_stocks",
    "calculate_annual_variance_of_the_portfolio",
    "calculate_annualized_return",
    "calculate_beta",
    "calculate_capm_expected_return",
    "calculate_covariance_with_market",
    "calculate_daily_drift",
    "calculate_daily_risk_free_rate",
    "calculate_excess_returns",
    "calculate_log_returns",
    "calculate_market_returns",
    "calculate_market_variance",
    "calculate_sharpe_ratio_of_portfolio",
    "calculate_the_annual_portfolio_return",
    "compare_betas",
    "compare_returns",
    "efficient_frontier",
    "fetch_market_data",
    "fetch_risk_free_rate",
    "fetch_stock_data",
    "portfolio_expected_return",
    "portfolio_metrics",
    "portfolio_variance",
    "portfolio_volatility",
    "run_capm_regression",
    "sharpe_ratio",
    "simulate_gbm_prices",
    "simulate_gbm_returns",
    "simulate_portfolio_paths",
    "simulate_random_portfolios",
    "summarize_residuals",
    "summarize_terminal_prices",
]
