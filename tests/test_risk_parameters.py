import numpy as np
import pandas as pd
import pytest

# Assuming your functions are saved in risk.py
from risk_engine.risk import (
    calculate_annual_variance_of_stocks,
    calculate_annual_std_deviation_of_stocks,
    calculate_annual_covariance_of_stocks,
    calculate_annual_correlation_of_stocks,
    calculate_annual_variance_of_the_portfolio,
    calculate_the_annual_portfolio_return,
    calculate_sharpe_ratio_of_portfolio,
)


@pytest.fixture
def mock_log_returns():
    """Provides a deterministic DataFrame of log returns for 2 stocks."""
    return pd.DataFrame({"AAPL": [0.01, -0.02, 0.015, -0.005, 0.01], "LULU": [0.02, 0.01, -0.01, 0.005, 0.015]})


@pytest.fixture
def mock_weights():
    """Provides a standard weight array for a 2-stock portfolio."""
    return np.array([0.6, 0.4])


def test_calculate_annual_variance_of_stocks(mock_log_returns):
    result = calculate_annual_variance_of_stocks(mock_log_returns)
    expected_daily_var = mock_log_returns.var()
    pd.testing.assert_series_equal(result, expected_daily_var * 252)


def test_calculate_annual_std_deviation_of_stocks(mock_log_returns):
    result = calculate_annual_std_deviation_of_stocks(mock_log_returns)
    expected_daily_std = mock_log_returns.std()
    pd.testing.assert_series_equal(result, expected_daily_std * np.sqrt(252))


def test_calculate_annual_covariance_of_stocks(mock_log_returns):
    result = calculate_annual_covariance_of_stocks(mock_log_returns)
    expected_daily_cov = mock_log_returns.cov()
    pd.testing.assert_frame_equal(result, expected_daily_cov * 252)


def test_calculate_annual_correlation_of_stocks(mock_log_returns):
    result = calculate_annual_correlation_of_stocks(mock_log_returns)
    expected_corr = mock_log_returns.corr()
    pd.testing.assert_frame_equal(result, expected_corr)


def test_calculate_annual_variance_of_the_portfolio(mock_log_returns, mock_weights):
    cov_matrix = calculate_annual_covariance_of_stocks(mock_log_returns)
    result = calculate_annual_variance_of_the_portfolio(mock_weights, cov_matrix)

    # Manual calculation of matrix dot product for validation
    expected_variance = np.dot(mock_weights.T, np.dot(cov_matrix, mock_weights))
    assert np.isclose(result, expected_variance)


def test_calculate_the_annual_portfolio_return(mock_weights):
    individual_returns = pd.Series({"AAPL": 0.15, "LULU": 0.10})
    result = calculate_the_annual_portfolio_return(mock_weights, individual_returns)

    # (0.6 * 0.15) + (0.4 * 0.10) = 0.09 + 0.04 = 0.13
    assert np.isclose(result, 0.13)


def test_calculate_sharpe_ratio_of_portfolio():
    risk_free_rate = 0.05
    portfolio_return = 0.15
    portfolio_volatility = 0.20

    result = calculate_sharpe_ratio_of_portfolio(risk_free_rate, portfolio_return, portfolio_volatility)

    # Expected: (0.15 - 0.05) / 0.20 = 0.5
    assert np.isclose(result, 0.5)
