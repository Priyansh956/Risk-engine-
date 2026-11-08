import numpy as np
import pandas as pd
import pytest

from risk_engine.portfolio import (
    PortfolioAnalysisResult,
    analyze_portfolio,
)


@pytest.fixture
def mock_portfolio_prices():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    # Generate prices with 60 days of history
    ret_aapl = rng.normal(0.001, 0.015, 60)
    ret_msft = rng.normal(0.0008, 0.012, 60)
    price_aapl = 150.0 * np.exp(np.cumsum(ret_aapl))
    price_msft = 250.0 * np.exp(np.cumsum(ret_msft))

    return pd.DataFrame({"AAPL": price_aapl, "MSFT": price_msft}, index=dates)


@pytest.fixture
def mock_market_prices():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    rng = np.random.default_rng(99)
    ret_mkt = rng.normal(0.0005, 0.01, 60)
    price_mkt = 400.0 * np.exp(np.cumsum(ret_mkt))
    return pd.Series(price_mkt, index=dates, name="SP500")


def test_analyze_portfolio_success(mock_portfolio_prices, mock_market_prices):
    tickers = ["AAPL", "MSFT"]
    weights = [0.6, 0.4]

    result = analyze_portfolio(
        tickers=tickers,
        weights=weights,
        prices=mock_portfolio_prices,
        market_prices=mock_market_prices,
        risk_free_rate=0.04,
        min_history=30,
        mc_time_horizon=50,
        mc_simulations=200,
        random_state=42,
    )

    assert isinstance(result, PortfolioAnalysisResult)
    assert isinstance(result.portfolio_return, float)
    assert isinstance(result.portfolio_volatility, float)
    assert result.portfolio_volatility > 0
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.portfolio_beta, float)
    assert isinstance(result.capm_expected_return, float)

    # Individual metrics
    assert len(result.individual_returns) == 2
    assert len(result.individual_betas) == 2
    assert len(result.weights) == 2

    # Monte Carlo summary
    mc = result.monte_carlo_summary
    assert "terminal_mean" in mc
    assert "terminal_median" in mc
    assert "terminal_5th_percentile" in mc
    assert "terminal_95th_percentile" in mc
    assert "terminal_std" in mc
    assert "scenarios" in mc

    scenarios = mc["scenarios"]
    assert scenarios["total_iterations"] == 200
    assert scenarios["bull_count"] + scenarios["base_count"] + scenarios["bear_count"] == 200

    # Serialization
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert "portfolio_return" in res_dict
    assert "monte_carlo_summary" in res_dict


def test_analyze_portfolio_validation_weights_sum(mock_portfolio_prices):
    tickers = ["AAPL", "MSFT"]
    weights = [0.5, 0.3]  # Sum = 0.8 != 1.0

    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        analyze_portfolio(
            tickers=tickers,
            weights=weights,
            prices=mock_portfolio_prices,
        )


def test_analyze_portfolio_validation_weights_length(mock_portfolio_prices):
    tickers = ["AAPL", "MSFT"]
    weights = [1.0]

    with pytest.raises(ValueError, match="Number of weights .* does not match"):
        analyze_portfolio(
            tickers=tickers,
            weights=weights,
            prices=mock_portfolio_prices,
        )


def test_analyze_portfolio_validation_negative_weights(mock_portfolio_prices):
    tickers = ["AAPL", "MSFT"]
    weights = [1.2, -0.2]

    with pytest.raises(ValueError, match="non-negative"):
        analyze_portfolio(
            tickers=tickers,
            weights=weights,
            prices=mock_portfolio_prices,
        )


def test_analyze_portfolio_validation_missing_tickers(mock_portfolio_prices):
    tickers = ["AAPL", "UNKNOWN_TICKER"]
    weights = [0.5, 0.5]

    with pytest.raises(ValueError, match="not found in price data"):
        analyze_portfolio(
            tickers=tickers,
            weights=weights,
            prices=mock_portfolio_prices,
        )


def test_analyze_portfolio_validation_empty_tickers():
    with pytest.raises(ValueError, match="At least one ticker"):
        analyze_portfolio(
            tickers=[],
            weights=[],
        )


def test_analyze_portfolio_validation_insufficient_history(mock_portfolio_prices):
    tickers = ["AAPL", "MSFT"]
    weights = [0.5, 0.5]
    # Request min_history of 100 on 60 rows
    with pytest.raises(ValueError, match="Insufficient price history"):
        analyze_portfolio(
            tickers=tickers,
            weights=weights,
            prices=mock_portfolio_prices,
            min_history=100,
        )
