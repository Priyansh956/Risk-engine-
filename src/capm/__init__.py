"""Capital Asset Pricing Model (CAPM) package."""

from capm.capm import (
    calculate_beta,
    calculate_capm_expected_return,
    calculate_covariance_with_market,
    calculate_daily_risk_free_rate,
    calculate_excess_returns,
    calculate_market_returns,
    calculate_market_variance,
    compare_betas,
    compare_returns,
    run_capm_regression,
    summarize_residuals,
)

__all__ = [
    "calculate_beta",
    "calculate_capm_expected_return",
    "calculate_covariance_with_market",
    "calculate_daily_risk_free_rate",
    "calculate_excess_returns",
    "calculate_market_returns",
    "calculate_market_variance",
    "compare_betas",
    "compare_returns",
    "run_capm_regression",
    "summarize_residuals",
]
