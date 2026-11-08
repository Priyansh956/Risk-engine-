"""Historical financial data loader utilities using yfinance."""

import pandas as pd
import yfinance as yf


def fetch_stock_data(
    tickers: list[str] | str,
    start: str = "2015-01-01",
    end: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for one or more stock tickers.

    Parameters
    ----------
    tickers : list[str] or str
        Stock ticker symbol or list of symbols.
    start : str, default="2015-01-01"
        Start date in YYYY-MM-DD format.
    end : str or None, default=None
        End date in YYYY-MM-DD format.
    auto_adjust : bool, default=False
        Whether to automatically adjust prices in yfinance.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices with tickers as column names.
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    data = pd.DataFrame()

    for ticker in ticker_list:
        downloaded = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
        )
        if "Adj Close" in downloaded:
            data[ticker] = downloaded["Adj Close"]
        elif "Close" in downloaded:
            data[ticker] = downloaded["Close"]
        else:
            data[ticker] = downloaded.iloc[:, 0]

    return data


def fetch_market_data(
    ticker: str = "^GSPC",
    start: str = "2015-01-01",
    end: str | None = None,
    auto_adjust: bool = False,
) -> pd.Series:
    """
    Fetch historical market benchmark prices.

    Parameters
    ----------
    ticker : str, default="^GSPC"
        Market benchmark ticker symbol (default: S&P 500).
    start : str, default="2015-01-01"
        Start date.
    end : str or None, default=None
        End date.
    auto_adjust : bool, default=False
        Whether to automatically adjust prices.

    Returns
    -------
    pd.Series
        Series of adjusted close prices for the market benchmark.
    """
    downloaded = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if "Adj Close" in downloaded:
        series = downloaded["Adj Close"].squeeze()
    elif "Close" in downloaded:
        series = downloaded["Close"].squeeze()
    else:
        series = downloaded.squeeze()

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series.name = "MARKET"
    return series


def fetch_risk_free_rate(
    ticker: str = "^TNX",
    period: str = "5d",
) -> float:
    """
    Fetch the latest risk-free rate from Treasury yield (e.g., ^TNX 10-year yield).

    Parameters
    ----------
    ticker : str, default="^TNX"
        Ticker symbol for Treasury yield.
    period : str, default="5d"
        Lookback period to ensure fetching recent trading day.

    Returns
    -------
    float
        Latest annual risk-free rate as a decimal (e.g., 0.042 for 4.2%).
    """
    downloaded = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if "Adj Close" in downloaded:
        col = downloaded["Adj Close"]
    elif "Close" in downloaded:
        col = downloaded["Close"]
    else:
        col = downloaded.iloc[:, 0]

    latest_val = float(col.dropna().iloc[-1])
    return latest_val / 100.0
