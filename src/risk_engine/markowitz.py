"""Markowitz portfolio analysis utilities.

The functions in this module operate on annualized expected returns and
an annualized covariance matrix.
"""

import numpy as np


def portfolio_expected_return(
    weights: np.ndarray,
    annualized_returns: np.ndarray,
) -> float:
    """Calculate the expected annualized return of a portfolio."""
    weights = np.asarray(weights, dtype=float)
    annualized_returns = np.asarray(annualized_returns, dtype=float)

    if weights.ndim != 1 or annualized_returns.ndim != 1:
        raise ValueError("weights and annualized_returns must be 1-D arrays.")

    if len(weights) != len(annualized_returns):
        raise ValueError("weights and annualized_returns must have the same length.")

    return float(np.dot(weights, annualized_returns))


def portfolio_variance(
    weights: np.ndarray,
    annualized_covariance: np.ndarray,
) -> float:
    """Calculate the annualized variance of a portfolio."""
    weights = np.asarray(weights, dtype=float)
    annualized_covariance = np.asarray(annualized_covariance, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array.")

    if annualized_covariance.ndim != 2:
        raise ValueError("annualized_covariance must be a 2-D matrix.")

    if annualized_covariance.shape != (len(weights), len(weights)):
        raise ValueError("Covariance matrix dimensions must match the number of weights.")

    return float(weights.T @ annualized_covariance @ weights)


def portfolio_volatility(
    weights: np.ndarray,
    annualized_covariance: np.ndarray,
) -> float:
    """Calculate the annualized portfolio volatility."""
    return float(np.sqrt(portfolio_variance(weights, annualized_covariance)))


def sharpe_ratio(
    portfolio_return: float,
    portfolio_volatility_value: float,
    risk_free_rate: float,
) -> float:
    """Calculate the Sharpe ratio."""
    if portfolio_volatility_value <= 0:
        raise ValueError("Portfolio volatility must be greater than zero.")

    return float((portfolio_return - risk_free_rate) / portfolio_volatility_value)


def portfolio_metrics(
    weights: np.ndarray,
    annualized_returns: np.ndarray,
    annualized_covariance: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Return portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = portfolio_expected_return(weights, annualized_returns)
    volatility = portfolio_volatility(weights, annualized_covariance)
    sharpe = sharpe_ratio(portfolio_return, volatility, risk_free_rate)

    return portfolio_return, volatility, sharpe


def simulate_random_portfolios(
    annualized_returns: np.ndarray,
    annualized_covariance: np.ndarray,
    risk_free_rate: float,
    num_simulations: int = 10_000,
    random_state: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Simulate random long-only portfolios and track optimal portfolios."""
    annualized_returns = np.asarray(annualized_returns, dtype=float)
    annualized_covariance = np.asarray(annualized_covariance, dtype=float)

    if annualized_returns.ndim != 1:
        raise ValueError("annualized_returns must be a 1-D array.")

    if annualized_covariance.shape != (
        len(annualized_returns),
        len(annualized_returns),
    ):
        raise ValueError("Covariance matrix dimensions must match the number of assets.")

    if num_simulations <= 0:
        raise ValueError("num_simulations must be greater than zero.")

    rng = np.random.default_rng(random_state)
    num_assets = len(annualized_returns)

    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpes = []

    max_sharpe = -np.inf
    max_sharpe_weights = None

    min_variance = np.inf
    min_variance_weights = None

    for _ in range(num_simulations):
        # Generate random long-only weights and normalize them to sum to 1.
        weights = rng.random(num_assets)
        weights /= weights.sum()

        variance = portfolio_variance(weights, annualized_covariance)
        volatility = np.sqrt(variance)
        portfolio_return = portfolio_expected_return(
            weights,
            annualized_returns,
        )
        sharpe = sharpe_ratio(
            portfolio_return,
            volatility,
            risk_free_rate,
        )

        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(volatility)
        portfolio_sharpes.append(sharpe)

        if variance < min_variance:
            min_variance = variance
            min_variance_weights = weights.copy()

        if sharpe > max_sharpe:
            max_sharpe = sharpe
            max_sharpe_weights = weights.copy()

    return {
        "portfolio_returns": np.asarray(portfolio_returns),
        "portfolio_volatilities": np.asarray(portfolio_volatilities),
        "portfolio_sharpes": np.asarray(portfolio_sharpes),
        "max_sharpe": float(max_sharpe),
        "max_sharpe_weights": max_sharpe_weights,
        "min_variance": float(min_variance),
        "min_variance_weights": min_variance_weights,
    }


def efficient_frontier(
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the approximate efficient frontier from simulations."""
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    portfolio_volatilities = np.asarray(portfolio_volatilities, dtype=float)

    if portfolio_returns.ndim != 1 or portfolio_volatilities.ndim != 1:
        raise ValueError("Returns and volatilities must be 1-D arrays.")

    if len(portfolio_returns) != len(portfolio_volatilities):
        raise ValueError("Returns and volatilities must have the same length.")

    sorted_indices = np.argsort(portfolio_volatilities)

    sorted_volatilities = portfolio_volatilities[sorted_indices]
    sorted_returns = portfolio_returns[sorted_indices]

    max_return_so_far = np.maximum.accumulate(sorted_returns)
    efficient_mask = sorted_returns >= max_return_so_far

    efficient_volatilities = sorted_volatilities[efficient_mask]
    efficient_returns = sorted_returns[efficient_mask]

    return efficient_volatilities, efficient_returns
