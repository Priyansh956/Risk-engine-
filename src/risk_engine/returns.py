import numpy as np
import pandas as pd


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.

    Returns
    -------
    pd.Series
        Logarithmic returns.
    """
    return np.log(prices / prices.shift(1))


def calculate_annualized_return(
    log_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Calculate the annualized return from daily log returns.

    Parameters
    ----------
    log_returns : pd.Series
        Daily logarithmic returns.
    trading_days : int, default=252
        Number of trading days used for annualization.

    Returns
    -------
    float
        Annualized compounded return.
    """
    mean_daily_log_return = log_returns.mean()

    return np.exp(mean_daily_log_return * trading_days) - 1