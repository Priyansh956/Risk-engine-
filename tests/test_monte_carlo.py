import numpy as np
import pandas as pd
import pytest

from monte_carlo.monte_carlo import (
    analyze_scenarios,
    calculate_daily_drift,
    simulate_gbm_prices,
    simulate_gbm_returns,
    simulate_gross_profit,
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

    returns1 = simulate_gbm_returns(drift, std, horizon, sims, random_state=42)
    returns2 = simulate_gbm_returns(drift, std, horizon, sims, random_state=42)

    assert returns1.shape == (horizon, sims, 2)
    np.testing.assert_allclose(returns1, returns2)


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


def test_simulate_portfolio_paths():
    prices = np.ones((10, 5, 2)) * 100.0  # (time, sims, assets)
    weights = [0.6, 0.4]

    portfolio_paths = simulate_portfolio_paths(prices, weights)
    assert portfolio_paths.shape == (10, 5)
    # 0.6 * 100 + 0.4 * 100 = 100
    np.testing.assert_allclose(portfolio_paths, 100.0)


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


def test_simulate_gross_profit():
    res = simulate_gross_profit(
        revenue_mean=1000.0,
        revenue_std=50.0,
        cogs_percentage_mean=40.0,
        cogs_percentage_std=2.0,
        num_simulations=500,
        random_state=42,
    )

    assert "revenue" in res
    assert "cost_of_goods_sold" in res
    assert "gross_profit" in res

    # gross_profit = revenue + cogs (where cogs is negative)
    np.testing.assert_allclose(res["gross_profit"], res["revenue"] + res["cost_of_goods_sold"])
    assert np.all(res["cost_of_goods_sold"] < 0)


def test_invalid_monte_carlo_inputs():
    with pytest.raises(ValueError):
        simulate_gbm_returns(drift=0.01, daily_std=0.02, time_horizon=0, num_simulations=10)

    with pytest.raises(ValueError):
        simulate_gbm_returns(drift=0.01, daily_std=0.02, time_horizon=10, num_simulations=0)

    with pytest.raises(ValueError):
        analyze_scenarios(np.ones((10, 10)), bull_percentile=10.0, bear_percentile=90.0)
