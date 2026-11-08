from unittest.mock import patch

import pandas as pd
import pytest

from risk_engine.data_loader import (
    fetch_market_data,
    fetch_risk_free_rate,
    fetch_stock_data,
)


@pytest.fixture
def mock_yf_stock_download():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Adj Close": [101.5, 102.5, 103.5, 104.5, 105.5],
        },
        index=dates,
    )


@pytest.fixture
def mock_yf_treasury_download():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Adj Close": [4.15, 4.20, 4.22, 4.18, 4.25],
        },
        index=dates,
    )


def test_fetch_stock_data(mock_yf_stock_download):
    with patch("yfinance.download", return_value=mock_yf_stock_download) as mock_dl:
        df = fetch_stock_data(["AAPL", "MSFT"])
        assert mock_dl.call_count == 2
        assert "AAPL" in df.columns
        assert "MSFT" in df.columns
        assert len(df) == 5
        pd.testing.assert_series_equal(df["AAPL"], mock_yf_stock_download["Adj Close"], check_names=False)


def test_fetch_market_data(mock_yf_stock_download):
    with patch("yfinance.download", return_value=mock_yf_stock_download):
        market_series = fetch_market_data("^GSPC")
        assert isinstance(market_series, pd.Series)
        assert market_series.name == "MARKET"
        assert len(market_series) == 5
        assert market_series.iloc[-1] == 105.5


def test_fetch_risk_free_rate(mock_yf_treasury_download):
    with patch("yfinance.download", return_value=mock_yf_treasury_download):
        rate = fetch_risk_free_rate("^TNX")
        # 4.25 / 100 = 0.0425
        assert rate == pytest.approx(0.0425)
