"""Risk engine parameter calculations.

Provides functions for calculating variance, standard deviation, covariance,
correlation, portfolio return, portfolio variance, and Sharpe ratio.
"""

import numpy as np
import pandas as pd


def calculate_annual_variance_of_stocks(
    annualized_log_returns: pd.DataFrame | pd.Series,
    trading_days: int = 252,
) -> pd.DataFrame | pd.Series:
    """Calculate annualized variance of stock daily log returns."""
    return annualized_log_returns.var() * trading_days


def calculate_annual_std_deviation_of_stocks(
    annualized_log_returns: pd.DataFrame | pd.Series,
    trading_days: int = 252,
) -> pd.DataFrame | pd.Series:
    """Calculate annualized standard deviation of stock daily log returns."""
    return annualized_log_returns.std() * np.sqrt(trading_days)


def calculate_annual_covariance_of_stocks(
    annualized_log_returns: pd.DataFrame,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate annualized covariance matrix of stock daily log returns."""
    return annualized_log_returns.cov() * trading_days


def calculate_annual_correlation_of_stocks(
    annualized_log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate correlation matrix of stock daily log returns."""
    return annualized_log_returns.corr()


def calculate_annual_variance_of_the_portfolio(
    weights_array: np.ndarray | pd.Series,
    annualized_covariance_of_stocks: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate annualized variance of the portfolio."""
    return float(np.dot(weights_array.T, np.dot(annualized_covariance_of_stocks, weights_array)))


def calculate_the_annual_portfolio_return(
    weights_array: np.ndarray | pd.Series,
    individual_returns: np.ndarray | pd.Series,
) -> float:
    """Calculate the expected annual portfolio return."""
    return float(np.dot(weights_array.T, individual_returns))


def calculate_sharpe_ratio_of_portfolio(
    risk_free_rate: float,
    portfolio_return: float | pd.Series,
    portfolio_volatility: float | pd.Series,
) -> float | pd.Series:
    """Calculate the Sharpe ratio of the portfolio."""
    return (portfolio_return - risk_free_rate) / portfolio_volatility
