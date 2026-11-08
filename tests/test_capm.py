import numpy as np
import pandas as pd
import pytest

from capm.capm import (
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


@pytest.fixture
def mock_market_prices():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.Series([100.0, 101.0, 102.5, 101.5, 103.0, 104.5, 103.8, 105.0, 106.2, 107.0], index=dates, name="SP500")


@pytest.fixture
def mock_returns_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    market = pd.Series([0.01, -0.005, 0.015, -0.01, 0.02, 0.005, -0.002, 0.012, -0.008, 0.014], index=dates, name="MARKET")
    stocks = pd.DataFrame(
        {
            "AAPL": [0.012, -0.006, 0.018, -0.011, 0.022, 0.006, -0.003, 0.015, -0.009, 0.016],
            "MSFT": [0.008, -0.004, 0.012, -0.008, 0.015, 0.004, -0.001, 0.010, -0.006, 0.011],
        },
        index=dates,
    )
    return stocks, market


def test_calculate_market_returns(mock_market_prices):
    result = calculate_market_returns(mock_market_prices)
    assert len(result) == len(mock_market_prices) - 1
    expected_first = np.log(101.0 / 100.0)
    assert np.isclose(result.iloc[0], expected_first)


def test_calculate_covariance_with_market(mock_returns_data):
    stocks, market = mock_returns_data
    cov = calculate_covariance_with_market(stocks, market)
    assert isinstance(cov, pd.Series)
    assert "AAPL" in cov.index
    assert "MSFT" in cov.index

    cov_single = calculate_covariance_with_market(stocks["AAPL"], market)
    assert isinstance(cov_single, float)
    assert np.isclose(cov_single, cov["AAPL"])


def test_calculate_market_variance(mock_returns_data):
    _, market = mock_returns_data
    var = calculate_market_variance(market)
    assert var > 0
    assert np.isclose(var, market.var())

    flat_series = pd.Series([0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="variance must be greater than zero"):
        calculate_market_variance(flat_series)


def test_calculate_beta(mock_returns_data):
    stocks, market = mock_returns_data
    betas = calculate_beta(stocks, market)
    assert isinstance(betas, pd.Series)
    assert len(betas) == 2

    # AAPL returns are ~1.1 to 1.2x market, beta should be around 1.1-1.2
    assert 1.0 < betas["AAPL"] < 1.3

    beta_single = calculate_beta(stocks["MSFT"], market)
    assert isinstance(beta_single, float)
    assert np.isclose(beta_single, betas["MSFT"])


def test_calculate_daily_risk_free_rate():
    annual_rf = 0.05
    daily_rf = calculate_daily_risk_free_rate(annual_rf, trading_days=252)
    expected = (1 + annual_rf) ** (1 / 252) - 1
    assert np.isclose(daily_rf, expected)


def test_calculate_excess_returns(mock_returns_data):
    stocks, _ = mock_returns_data
    excess = calculate_excess_returns(stocks, risk_free_rate=0.05, is_annual_rate=True)
    daily_rf = calculate_daily_risk_free_rate(0.05)
    pd.testing.assert_frame_equal(excess, stocks - daily_rf)


def test_calculate_capm_expected_return():
    beta = 1.2
    market_return = 0.10
    risk_free_rate = 0.04
    expected = calculate_capm_expected_return(beta, market_return, risk_free_rate)
    # 0.04 + 1.2 * (0.10 - 0.04) = 0.04 + 0.072 = 0.112
    assert np.isclose(expected, 0.112)

    beta_series = pd.Series({"AAPL": 1.2, "MSFT": 0.8})
    expected_series = calculate_capm_expected_return(beta_series, market_return, risk_free_rate)
    assert np.isclose(expected_series["AAPL"], 0.112)
    assert np.isclose(expected_series["MSFT"], 0.04 + 0.8 * 0.06)


def test_run_capm_regression(mock_returns_data):
    stocks, market = mock_returns_data
    results = run_capm_regression(stocks, market, risk_free_rate=0.04)

    assert "alpha" in results
    assert "beta" in results
    assert "r_squared" in results
    assert "residuals" in results
    assert "models" in results

    assert isinstance(results["alpha"], pd.Series)
    assert isinstance(results["beta"], pd.Series)
    assert isinstance(results["r_squared"], pd.Series)
    assert isinstance(results["residuals"], pd.DataFrame)

    for stock in stocks.columns:
        assert 0 <= results["r_squared"][stock] <= 1.0

    # Test single Series
    single_res = run_capm_regression(stocks["AAPL"], market, risk_free_rate=0.04)
    assert isinstance(single_res["alpha"], float)
    assert isinstance(single_res["beta"], float)
    assert isinstance(single_res["r_squared"], float)


def test_compare_betas():
    cov_beta = pd.Series({"AAPL": 1.15, "MSFT": 0.85})
    reg_beta = pd.Series({"AAPL": 1.14, "MSFT": 0.86})
    comp = compare_betas(cov_beta, reg_beta)

    assert "Covariance Beta" in comp.columns
    assert "Regression Beta" in comp.columns
    assert "Difference" in comp.columns
    assert np.isclose(comp.loc["AAPL", "Difference"], 0.01)


def test_compare_returns():
    hist_ret = pd.Series({"AAPL": 0.15, "MSFT": 0.12})
    capm_ret = pd.Series({"AAPL": 0.14, "MSFT": 0.10})
    comp = compare_returns(hist_ret, capm_ret)

    assert "Historical Return" in comp.columns
    assert "CAPM Expected Return" in comp.columns
    assert "Difference" in comp.columns
    assert np.isclose(comp.loc["AAPL", "Difference"], 0.01)


def test_summarize_residuals(mock_returns_data):
    stocks, market = mock_returns_data
    results = run_capm_regression(stocks, market)
    summary = summarize_residuals(results["residuals"])

    assert "Mean Residual" in summary.columns
    assert "Residual Std Dev" in summary.columns
    assert "Min Residual" in summary.columns
    assert "Max Residual" in summary.columns
    assert len(summary) == 2
