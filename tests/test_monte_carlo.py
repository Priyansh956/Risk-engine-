import numpy as np
import pandas as pd
import pytest

from monte_carlo.monte_carlo import (
    analyze_scenarios,
    calculate_daily_drift,
    simulate_gbm_prices,
    simulate_gbm_returns,
    simulate_portfolio_paths,
    summarize_terminal_prices,
)


@pytest.fixture
def mock_daily_returns():
    return pd.DataFrame(
        {
            "AAPL": [0.01, -0.015, 0.02, -0.005, 0.012],
            "NKE": [0.005, -0.008, 0.012, 0.003, -0.002],
        }
    )


def test_calculate_daily_drift(mock_daily_returns):
    drift = calculate_daily_drift(mock_daily_returns)
    assert isinstance(drift, pd.Series)
    for col in mock_daily_returns.columns:
        expected = mock_daily_returns[col].mean() - 0.5 * mock_daily_returns[col].var()
        assert np.isclose(drift[col], expected)


def test_simulate_gbm_returns_shape_and_reproducibility():
    drift = np.array([0.0005, 0.0003])
    std = np.array([0.015, 0.012])
    horizon = 10
    sims = 50

    returns1 = simulate_gbm_returns(drift, std, time_horizon=horizon, num_simulations=sims, random_state=42)
    returns2 = simulate_gbm_returns(drift, std, time_horizon=horizon, num_simulations=sims, random_state=42)

    assert returns1.shape == (horizon, sims, 2)
    np.testing.assert_allclose(returns1, returns2)


def test_simulate_gbm_returns_correlated_cholesky():
    drift = np.array([0.0004, 0.0002])
    # Create covariance matrix with high positive correlation (0.8)
    cov = np.array(
        [
            [0.0004, 0.00024],
            [0.00024, 0.000225],
        ]
    )
    horizon = 500
    sims = 2000

    returns = simulate_gbm_returns(
        drift=drift,
        covariance_matrix=cov,
        time_horizon=horizon,
        num_simulations=sims,
        random_state=123,
    )
    assert returns.shape == (horizon, sims, 2)

    # Log returns should empirically match the input covariance matrix
    log_rets = np.log(returns)
    # Reshape to (horizon * sims, 2)
    flat_rets = log_rets.reshape(-1, 2)
    empirical_cov = np.cov(flat_rets, rowvar=False)

    np.testing.assert_allclose(empirical_cov, cov, atol=2e-4)


def test_simulate_gbm_prices():
    s0 = np.array([150.0, 80.0])
    drift = np.array([0.0005, 0.0003])
    std = np.array([0.015, 0.012])
    horizon = 20
    sims = 100

    prices = simulate_gbm_prices(s0, drift, std, horizon, sims, random_state=123)

    assert prices.shape == (horizon, sims, 2)
    # Check that day 0 prices match initial prices for all simulations
    for i in range(sims):
        np.testing.assert_allclose(prices[0, i, :], s0)
    # Prices should stay strictly positive
    assert np.all(prices > 0)


def test_summarize_terminal_prices():
    s0 = np.array([100.0, 50.0])
    drift = np.array([0.001, 0.0005])
    std = np.array([0.02, 0.015])
    prices = simulate_gbm_prices(s0, drift, std, time_horizon=50, num_simulations=500, random_state=99)

    summary = summarize_terminal_prices(prices, asset_names=["StockA", "StockB"])

    assert "Mean Price" in summary.columns
    assert "Median Price" in summary.columns
    assert "5th Percentile (Worst Case)" in summary.columns
    assert "95th Percentile (Best Case)" in summary.columns
    assert "Standard Deviation" in summary.columns
    assert len(summary) == 2

    for asset in ["StockA", "StockB"]:
        assert summary.loc[asset, "5th Percentile (Worst Case)"] <= summary.loc[asset, "Median Price"]
        assert summary.loc[asset, "Median Price"] <= summary.loc[asset, "95th Percentile (Best Case)"]


def test_simulate_portfolio_paths_normalization():
    # Test that portfolio paths properly normalize by starting price:
    # Stock A starts at 3000, Stock B starts at 100.
    # Allocation: 50% A, 50% B.
    # At t=1: Stock A stays flat (3000), Stock B goes up 10% (110).
    # Correct capital return = 0.5 * 0% + 0.5 * 10% = +5% (portfolio value goes from 1.0 to 1.05).
    # If weights were unnormalized raw share counts, 3000 would dominate and yield only +0.32%.
    time_horizon = 2
    num_simulations = 1
    num_assets = 2

    prices = np.zeros((time_horizon, num_simulations, num_assets))
    prices[0, 0, :] = [3000.0, 100.0]
    prices[1, 0, :] = [3000.0, 110.0]

    weights = [0.5, 0.5]
    portfolio_paths = simulate_portfolio_paths(prices, weights, initial_portfolio_value=1.0)

    assert portfolio_paths.shape == (2, 1)
    # Day 0: portfolio value starts at 1.0
    assert np.isclose(portfolio_paths[0, 0], 1.0)
    # Day 1: portfolio value is 1.05 (5% gain)
    assert np.isclose(portfolio_paths[1, 0], 1.05)


def test_analyze_scenarios():
    # Construct 100 paths with predictable final values
    time_horizon = 10
    num_simulations = 100
    paths = np.zeros((time_horizon, num_simulations))
    for i in range(num_simulations):
        paths[:, i] = np.linspace(100, 100 + i, time_horizon)

    scenarios = analyze_scenarios(paths, bull_percentile=90.0, bear_percentile=10.0)

    assert scenarios["total_iterations"] == 100
    assert scenarios["bull_count"] > 0
    assert scenarios["bear_count"] > 0
    assert scenarios["base_count"] > 0
    assert scenarios["bull_count"] + scenarios["base_count"] + scenarios["bear_count"] == 100

    assert scenarios["average_terminal_value_bull"] > scenarios["average_terminal_value_base"]
    assert scenarios["average_terminal_value_base"] > scenarios["average_terminal_value_bear"]
    assert scenarios["bull_threshold"] > scenarios["bear_threshold"]


def test_invalid_monte_carlo_inputs():
    with pytest.raises(ValueError):
        simulate_gbm_returns(drift=0.01, daily_std=0.02, time_horizon=0, num_simulations=10)

    with pytest.raises(ValueError):
        simulate_gbm_returns(drift=0.01, daily_std=0.02, time_horizon=10, num_simulations=0)

    with pytest.raises(ValueError):
        analyze_scenarios(np.ones((10, 10)), bull_percentile=10.0, bear_percentile=90.0)
