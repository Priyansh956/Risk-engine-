"""Financial data loading package."""

from data_loader.data_loader import (
    fetch_market_data,
    fetch_risk_free_rate,
    fetch_stock_data,
)

__all__ = [
    "fetch_market_data",
    "fetch_risk_free_rate",
    "fetch_stock_data",
]
