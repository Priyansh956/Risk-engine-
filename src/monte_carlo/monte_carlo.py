"""Monte Carlo simulation for financial forecasting, asset prices, and portfolio scenarios."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_daily_drift(
    daily_log_returns: pd.DataFrame | pd.Series | np.ndarray,
) -> pd.Series | float | np.ndarray:
    """
    Calculate daily drift using daily mean return and variance.

    Drift formula:
        drift = mu - 0.5 * sigma^2

    Parameters
    ----------
    daily_log_returns : pd.DataFrame, pd.Series, or np.ndarray
        Historical daily logarithmic returns.

    Returns
    -------
    pd.Series, float, or np.ndarray
        Calculated daily drift for each asset.
    """
    if isinstance(daily_log_returns, (pd.DataFrame, pd.Series)):
        mu = daily_log_returns.mean()
        var = daily_log_returns.var()
        return mu - (0.5 * var)

    arr = np.asarray(daily_log_returns, dtype=float)
    mu = np.mean(arr, axis=0)
    var = np.var(arr, axis=0, ddof=1) if arr.ndim > 1 or len(arr) > 1 else 0.0
    return mu - (0.5 * var)


def simulate_gbm_returns(
    drift: np.ndarray | pd.Series | float,
    daily_std: np.ndarray | pd.Series | float,
    time_horizon: int,
    num_simulations: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate daily returns using Geometric Brownian Motion (GBM).

    Parameters
    ----------
    drift : np.ndarray, pd.Series, or float
        Daily drift per asset.
    daily_std : np.ndarray, pd.Series, or float
        Daily standard deviation per asset.
    time_horizon : int
        Number of forward time steps (days) to simulate.
    num_simulations : int
        Number of simulation paths / iterations.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated returns array with shape (time_horizon, num_simulations, num_assets)
        or (time_horizon, num_simulations) if single asset.
    """
    if time_horizon <= 0:
        raise ValueError("time_horizon must be greater than zero.")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be greater than zero.")

    drift_arr = np.asarray(drift if not isinstance(drift, pd.Series) else drift.values, dtype=float)
    std_arr = np.asarray(daily_std if not isinstance(daily_std, pd.Series) else daily_std.values, dtype=float)

    is_1d_asset = drift_arr.ndim == 0 or (drift_arr.ndim == 1 and drift_arr.size == 1)
    rng = np.random.default_rng(random_state)

    if is_1d_asset:
        d = float(drift_arr)
        s = float(std_arr)
        z = rng.normal(0.0, 1.0, (time_horizon, num_simulations))
        return np.exp(d + s * z)

    num_assets = drift_arr.shape[0]
    z = rng.normal(0.0, 1.0, (time_horizon, num_simulations, num_assets))
    return np.exp(drift_arr + std_arr * z)


def simulate_gbm_prices(
    initial_prices: np.ndarray | pd.Series | float,
    drift: np.ndarray | pd.Series | float,
    daily_std: np.ndarray | pd.Series | float,
    time_horizon: int,
    num_simulations: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Simulate future stock price paths using Geometric Brownian Motion.

    Parameters
    ----------
    initial_prices : np.ndarray, pd.Series, or float
        Current/starting price for each asset.
    drift : np.ndarray, pd.Series, or float
        Daily drift.
    daily_std : np.ndarray, pd.Series, or float
        Daily standard deviation.
    time_horizon : int
        Number of forward days to forecast.
    num_simulations : int
        Number of simulation paths.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated price paths. Shape is (time_horizon, num_simulations, num_assets)
        or (time_horizon, num_simulations) if 1 asset.
    """
    daily_returns = simulate_gbm_returns(
        drift=drift,
        daily_std=daily_std,
        time_horizon=time_horizon,
        num_simulations=num_simulations,
        random_state=random_state,
    )

    s0 = np.asarray(initial_prices if not isinstance(initial_prices, pd.Series) else initial_prices.values, dtype=float)

    if daily_returns.ndim == 2:
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = float(s0)
        for t in range(1, time_horizon):
            price_paths[t] = price_paths[t - 1] * daily_returns[t]
        return price_paths

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = s0
    for t in range(1, time_horizon):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]
    return price_paths


def summarize_terminal_prices(
    price_paths: np.ndarray,
    asset_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute key statistical summary metrics across simulation terminal prices.

    Parameters
    ----------
    price_paths : np.ndarray
        Price simulation array.
    asset_names : list[str] or None, default=None
        Column labels for assets.

    Returns
    -------
    pd.DataFrame
        DataFrame with Mean, Median, 5th Percentile, 95th Percentile, and Std Dev.
    """
    if price_paths.ndim == 3:
        terminal_prices = price_paths[-1]
    elif price_paths.ndim == 2:
        terminal_prices = price_paths
    else:
        raise ValueError("price_paths must be 2D or 3D.")

    if terminal_prices.ndim == 1 or (terminal_prices.ndim == 2 and terminal_prices.shape[1] == 1):
        terminal_df = pd.DataFrame(terminal_prices.flatten(), columns=asset_names or ["Asset"])
    else:
        terminal_df = pd.DataFrame(terminal_prices, columns=asset_names)

    return pd.DataFrame(
        {
            "Mean Price": terminal_df.mean(),
            "Median Price": terminal_df.median(),
            "5th Percentile (Worst Case)": terminal_df.quantile(0.05),
            "95th Percentile (Best Case)": terminal_df.quantile(0.95),
            "Standard Deviation": terminal_df.std(),
        }
    )


def simulate_portfolio_paths(
    price_paths: np.ndarray,
    weights: np.ndarray | list[float],
) -> np.ndarray:
    """
    Calculate the total portfolio value across all time steps and iterations.

    Parameters
    ----------
    price_paths : np.ndarray
        3D simulated price paths of shape (time_horizon, num_simulations, num_assets).
    weights : np.ndarray or list[float]
        Portfolio weights of shape (num_assets,).

    Returns
    -------
    np.ndarray
        Simulated portfolio paths of shape (time_horizon, num_simulations).
    """
    weights_arr = np.asarray(weights, dtype=float)
    if price_paths.ndim != 3:
        raise ValueError("price_paths must have shape (time_horizon, num_simulations, num_assets).")

    num_assets = price_paths.shape[2]
    if len(weights_arr) != num_assets:
        raise ValueError(f"weights length ({len(weights_arr)}) does not match asset count ({num_assets}).")

    if not np.isclose(np.sum(weights_arr), 1.0, atol=1e-4):
        weights_arr = weights_arr / np.sum(weights_arr)

    return np.dot(price_paths, weights_arr)


def analyze_scenarios(
    portfolio_paths: np.ndarray,
    bull_percentile: float = 90.0,
    bear_percentile: float = 10.0,
) -> dict[str, Any]:
    """
    Filter portfolio paths into Bull, Base, and Bear market scenarios based on terminal values.

    Parameters
    ----------
    portfolio_paths : np.ndarray
        2D portfolio simulation paths of shape (time_horizon, num_simulations).
    bull_percentile : float, default=90.0
        Threshold percentile for bull market scenario (e.g. top 10%).
    bear_percentile : float, default=10.0
        Threshold percentile for bear market scenario (e.g. bottom 10%).

    Returns
    -------
    dict[str, Any]
        Scenario analysis summary containing paths, thresholds, and average terminal values.
    """
    if portfolio_paths.ndim != 2:
        raise ValueError("portfolio_paths must be a 2D array of shape (time_horizon, num_simulations).")
    if bear_percentile >= bull_percentile:
        raise ValueError("bear_percentile must be strictly less than bull_percentile.")

    terminal_portfolio_values = portfolio_paths[-1]

    bull_threshold = float(np.percentile(terminal_portfolio_values, bull_percentile))
    bear_threshold = float(np.percentile(terminal_portfolio_values, bear_percentile))

    bull_mask = terminal_portfolio_values >= bull_threshold
    bear_mask = terminal_portfolio_values <= bear_threshold
    base_mask = (terminal_portfolio_values > bear_threshold) & (terminal_portfolio_values < bull_threshold)

    bull_paths = portfolio_paths[:, bull_mask]
    bear_paths = portfolio_paths[:, bear_mask]
    base_paths = portfolio_paths[:, base_mask]

    return {
        "bull_paths": bull_paths,
        "base_paths": base_paths,
        "bear_paths": bear_paths,
        "bull_threshold": bull_threshold,
        "bear_threshold": bear_threshold,
        "average_terminal_value_bull": float(bull_paths[-1].mean()) if bull_paths.size > 0 else 0.0,
        "average_terminal_value_base": float(base_paths[-1].mean()) if base_paths.size > 0 else 0.0,
        "average_terminal_value_bear": float(bear_paths[-1].mean()) if bear_paths.size > 0 else 0.0,
        "total_iterations": int(portfolio_paths.shape[1]),
        "bull_count": int(bull_paths.shape[1]),
        "base_count": int(base_paths.shape[1]),
        "bear_count": int(bear_paths.shape[1]),
    }


def simulate_gross_profit(
    revenue_mean: float | np.ndarray | pd.Series,
    revenue_std: float | np.ndarray | pd.Series,
    cogs_percentage_mean: float,
    cogs_percentage_std: float = 0.0,
    num_simulations: int = 1000,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run Monte Carlo simulation for revenue, COGS, and gross profit.

    Parameters
    ----------
    revenue_mean : float, np.ndarray, or pd.Series
        Expected revenue.
    revenue_std : float, np.ndarray, or pd.Series
        Standard deviation of revenue.
    cogs_percentage_mean : float
        Mean percentage of revenue that COGS represents.
    cogs_percentage_std : float, default=0.0
        Standard deviation of COGS percentage.
    num_simulations : int, default=1000
        Number of simulation iterations.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Simulated arrays for "revenue", "cost_of_goods_sold", and "gross_profit".
    """
    rng = np.random.default_rng(random_state)
    rev_mean = np.asarray(revenue_mean if not isinstance(revenue_mean, pd.Series) else revenue_mean.values, dtype=float)
    rev_std = np.asarray(revenue_std if not isinstance(revenue_std, pd.Series) else revenue_std.values, dtype=float)

    cogs_mean = cogs_percentage_mean / 100.0 if cogs_percentage_mean > 1.0 else cogs_percentage_mean
    cogs_std = cogs_percentage_std / 100.0 if cogs_percentage_std > 1.0 else cogs_percentage_std

    if rev_mean.ndim > 0 and rev_mean.size > 1:
        size = (num_simulations, len(rev_mean))
    else:
        size = (num_simulations, 1)

    revenue = rng.normal(loc=rev_mean, scale=rev_std, size=size)
    cogs_factors = rng.normal(loc=cogs_mean, scale=cogs_std, size=size)
    cogs = -(revenue * cogs_factors)
    gross_profit = revenue + cogs

    return {
        "revenue": revenue,
        "cost_of_goods_sold": cogs,
        "gross_profit": gross_profit,
    }
