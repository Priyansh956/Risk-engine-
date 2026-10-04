import numpy as np
import pytest

from markowitz.markowitz import (
    efficient_frontier,
    portfolio_expected_return,
    portfolio_metrics,
    portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
    simulate_random_portfolios,
)


def test_portfolio_expected_return():
    weights = np.array([0.7, 0.15, 0.15])
    annualized_returns = np.array([0.10, 0.05, 0.20])

    result = portfolio_expected_return(weights, annualized_returns)

    expected = 0.7 * 0.10 + 0.15 * 0.05 + 0.15 * 0.20

    assert result == pytest.approx(expected)


def test_portfolio_variance():
    weights = np.array([0.5, 0.5])
    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )

    result = portfolio_variance(weights, covariance)

    expected = 0.5**2 * 0.04 + 0.5**2 * 0.09 + 2 * 0.5 * 0.5 * 0.01

    assert result == pytest.approx(expected)


def test_portfolio_volatility():
    weights = np.array([0.5, 0.5])
    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )

    result = portfolio_volatility(weights, covariance)

    assert result == pytest.approx(np.sqrt(0.0375))


def test_sharpe_ratio():
    result = sharpe_ratio(
        portfolio_return=0.12,
        portfolio_volatility_value=0.20,
        risk_free_rate=0.04,
    )

    assert result == pytest.approx(0.4)


def test_portfolio_metrics():
    weights = np.array([0.5, 0.5])
    returns = np.array([0.10, 0.14])
    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )

    portfolio_return, volatility, sharpe = portfolio_metrics(
        weights,
        returns,
        covariance,
        risk_free_rate=0.04,
    )

    assert portfolio_return == pytest.approx(0.12)
    assert volatility == pytest.approx(np.sqrt(0.0375))
    assert sharpe == pytest.approx((0.12 - 0.04) / np.sqrt(0.0375))


def test_random_portfolio_simulation():
    returns = np.array([0.10, 0.12, 0.15])
    covariance = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.05, 0.008],
            [0.005, 0.008, 0.06],
        ]
    )

    result = simulate_random_portfolios(
        returns,
        covariance,
        risk_free_rate=0.04,
        num_simulations=100,
        random_state=42,
    )

    assert len(result["portfolio_returns"]) == 100
    assert len(result["portfolio_volatilities"]) == 100
    assert len(result["portfolio_sharpes"]) == 100

    assert result["max_sharpe"] == pytest.approx(np.max(result["portfolio_sharpes"]))

    assert result["min_variance"] == pytest.approx(np.min(result["portfolio_volatilities"] ** 2))

    assert np.isclose(result["max_sharpe_weights"].sum(), 1.0)
    assert np.isclose(result["min_variance_weights"].sum(), 1.0)

    assert np.all(result["max_sharpe_weights"] >= 0)
    assert np.all(result["min_variance_weights"] >= 0)


def test_efficient_frontier():
    volatilities = np.array([0.30, 0.25, 0.28, 0.35, 0.32])
    returns = np.array([0.10, 0.08, 0.12, 0.11, 0.15])

    efficient_volatilities, efficient_returns = efficient_frontier(
        returns,
        volatilities,
    )

    expected_volatilities = np.array([0.25, 0.28, 0.32])
    expected_returns = np.array([0.08, 0.12, 0.15])

    assert np.allclose(efficient_volatilities, expected_volatilities)
    assert np.allclose(efficient_returns, expected_returns)


def test_invalid_simulation_count():
    returns = np.array([0.10, 0.12])
    covariance = np.eye(2)

    with pytest.raises(ValueError):
        simulate_random_portfolios(
            returns,
            covariance,
            risk_free_rate=0.04,
            num_simulations=0,
        )
