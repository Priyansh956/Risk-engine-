"""Risk engine package for portfolio return, risk, CAPM, and Monte Carlo analytics."""

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
    "calculate_annual_correlation_of_stocks",
    "calculate_annual_covariance_of_stocks",
    "calculate_annual_std_deviation_of_stocks",
    "calculate_annual_variance_of_stocks",
    "calculate_annual_variance_of_the_portfolio",
    "calculate_annualized_return",
    "calculate_log_returns",
    "calculate_sharpe_ratio_of_portfolio",
    "calculate_the_annual_portfolio_return",
]
