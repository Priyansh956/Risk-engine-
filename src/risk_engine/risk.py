import numpy as np
import pandas as pd


def calculate_annual_variance_of_stocks(annualized_log_returns: pd.Series):
    return annualized_log_returns.var() * 252


def calculate_annual_std_deviation_of_stocks(annualized_log_returns: pd.Series):
    return annualized_log_returns.std() * np.sqrt(252)


def calculate_annual_covariance_of_stocks(annualized_log_returns: pd.Series):
    return annualized_log_returns.cov() * 252


def calculate_annual_correlation_of_stocks(annualized_log_returns: pd.Series):
    return annualized_log_returns.corr()


def calculate_annual_variance_of_the_portfolio(weights_array: pd.Series, annualized_covariance_of_stocks: pd.Series):
    return np.dot(weights_array.T, np.dot(annualized_covariance_of_stocks, weights_array))


def calculate_the_annual_portfolio_return(weights_array: pd.Series, individual_returns: pd.Series):
    return np.dot(weights_array.T, individual_returns)


def calculate_sharpe_ratio_of_portfolio(risk_free_rate: float, portfolio_return: pd.Series, portfolio_volatility: pd.Series):
    return (risk_free_rate - portfolio_return) / portfolio_volatility
