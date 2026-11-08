import numpy as np
import pandas as pd


def calculate_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate logarithmic returns from a price series or DataFrame.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Historical price series or price DataFrame.

    Returns
    -------
    pd.Series or pd.DataFrame
        Logarithmic returns.
    """
    return np.log(prices / prices.shift(1))


def calculate_annualized_return(
    log_returns: pd.Series | pd.DataFrame,
    trading_days: int = 252,
) -> float | pd.Series:
    """
    Calculate the annualized return from daily log returns.

    Parameters
    ----------
    log_returns : pd.Series or pd.DataFrame
        Daily logarithmic returns.
    trading_days : int, default=252
        Number of trading days used for annualization.

    Returns
    -------
    float or pd.Series
        Annualized compounded return.
    """
    mean_daily_log_return = log_returns.mean()

    return np.exp(mean_daily_log_return * trading_days) - 1
