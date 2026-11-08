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
    daily_std: np.ndarray | pd.Series | float | None = None,
    time_horizon: int = 252,
    num_simulations: int = 1000,
    random_state: int | None = None,
    covariance_matrix: np.ndarray | pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Generate daily returns using Geometric Brownian Motion (GBM).

    Supports correlated multi-asset shocks via Cholesky decomposition of the covariance matrix.

    Parameters
    ----------
    drift : np.ndarray, pd.Series, or float
        Daily drift per asset.
    daily_std : np.ndarray, pd.Series, or float, optional
        Daily standard deviation per asset (used when covariance_matrix is not supplied).
    covariance_matrix : np.ndarray or pd.DataFrame, optional
        Daily covariance matrix of asset returns. When provided, correlated shocks
        are drawn using Cholesky decomposition (Sigma = L @ L.T).
    time_horizon : int, default=252
        Number of forward time steps (days) to simulate.
    num_simulations : int, default=1000
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
    rng = np.random.default_rng(random_state)

    if covariance_matrix is not None:
        cov_arr = np.asarray(covariance_matrix if not isinstance(covariance_matrix, pd.DataFrame) else covariance_matrix.values, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("covariance_matrix must be a square 2D matrix.")

        num_assets = cov_arr.shape[0]
        if drift_arr.ndim == 0 or drift_arr.size != num_assets:
            raise ValueError(f"drift length ({drift_arr.size}) does not match covariance dimension ({num_assets}).")

        try:
            chol = np.linalg.cholesky(cov_arr)
        except np.linalg.LinAlgError:
            jitter = 1e-8 * np.eye(num_assets)
            chol = np.linalg.cholesky(cov_arr + jitter)

        z = rng.normal(0.0, 1.0, (time_horizon, num_simulations, num_assets))
        correlated_shocks = z @ chol.T
        return np.exp(drift_arr + correlated_shocks)

    if daily_std is None:
        raise ValueError("Either daily_std or covariance_matrix must be provided.")

    std_arr = np.asarray(daily_std if not isinstance(daily_std, pd.Series) else daily_std.values, dtype=float)
    is_1d_asset = drift_arr.ndim == 0 or (drift_arr.ndim == 1 and drift_arr.size == 1)

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
    daily_std: np.ndarray | pd.Series | float | None = None,
    time_horizon: int = 252,
    num_simulations: int = 1000,
    random_state: int | None = None,
    covariance_matrix: np.ndarray | pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Simulate future stock price paths using Geometric Brownian Motion.

    Supports correlated asset paths via covariance_matrix Cholesky decomposition.

    Parameters
    ----------
    initial_prices : np.ndarray, pd.Series, or float
        Current/starting price for each asset.
    drift : np.ndarray, pd.Series, or float
        Daily drift.
    daily_std : np.ndarray, pd.Series, or float, optional
        Daily standard deviation per asset.
    time_horizon : int, default=252
        Number of forward days to forecast.
    num_simulations : int, default=1000
        Number of simulation paths.
    random_state : int or None, default=None
        Random seed for reproducibility.
    covariance_matrix : np.ndarray or pd.DataFrame, optional
        Daily covariance matrix for correlated asset simulations.

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
        covariance_matrix=covariance_matrix,
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
    weights: np.ndarray | list[float] | pd.Series,
    initial_portfolio_value: float = 1.0,
) -> np.ndarray:
    """
    Calculate portfolio value paths over time, weighting by capital allocation.

    Weights are applied to price paths normalized by each asset's starting price,
    correctly modeling holding dollar allocations rather than raw share counts.

    Parameters
    ----------
    price_paths : np.ndarray
        3D simulated price paths of shape (time_horizon, num_simulations, num_assets).
    weights : np.ndarray, list[float], or pd.Series
        Portfolio capital weights of shape (num_assets,).
    initial_portfolio_value : float, default=1.0
        Starting portfolio capital value.

    Returns
    -------
    np.ndarray
        Simulated portfolio paths of shape (time_horizon, num_simulations).
    """
    weights_arr = np.asarray(weights if not isinstance(weights, pd.Series) else weights.values, dtype=float)
    if price_paths.ndim != 3:
        raise ValueError("price_paths must have shape (time_horizon, num_simulations, num_assets).")

    num_assets = price_paths.shape[2]
    if len(weights_arr) != num_assets:
        raise ValueError(f"weights length ({len(weights_arr)}) does not match asset count ({num_assets}).")

    if not np.isclose(np.sum(weights_arr), 1.0, atol=1e-3):
        weights_arr = weights_arr / np.sum(weights_arr)

    # Normalize price paths by initial price S_0 for each asset so weights represent dollar allocation
    s0 = price_paths[0:1, :, :]  # Shape (1, num_simulations, num_assets)
    if np.any(s0 <= 0):
        raise ValueError("Initial prices must be strictly positive.")

    normalized_price_paths = price_paths / s0

    # Total portfolio value = initial_value * sum(w_i * S_{t, i} / S_{0, i})
    portfolio_paths = initial_portfolio_value * np.dot(normalized_price_paths, weights_arr)
    return portfolio_paths


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
