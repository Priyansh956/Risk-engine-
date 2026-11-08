"""Capital Asset Pricing Model (CAPM) analysis and regression utilities."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_market_returns(market_prices: pd.Series) -> pd.Series:
    """
    Calculate daily logarithmic returns for a market benchmark price series.

    Parameters
    ----------
    market_prices : pd.Series
        Historical prices for the market benchmark.

    Returns
    -------
    pd.Series
        Logarithmic returns of the market benchmark.
    """
    market_returns = np.log(market_prices / market_prices.shift(1)).dropna()
    if market_returns.name is None:
        market_returns.name = "MARKET"
    return market_returns


def calculate_covariance_with_market(
    stock_returns: pd.DataFrame | pd.Series,
    market_returns: pd.Series,
) -> pd.Series | float:
    """
    Calculate the covariance of stock returns with the market return series.

    Parameters
    ----------
    stock_returns : pd.DataFrame or pd.Series
        Daily returns of one or more stocks.
    market_returns : pd.Series
        Daily returns of the market benchmark.

    Returns
    -------
    pd.Series or float
        Covariance with market returns.
    """
    if isinstance(stock_returns, pd.Series):
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned) == 0:
            raise ValueError("No overlapping return observations found.")
        return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))

    m_name = market_returns.name or "MARKET"
    market_series = market_returns.rename(m_name)
    combined = pd.concat([stock_returns, market_series], axis=1).dropna()
    if len(combined) == 0:
        raise ValueError("No overlapping return observations found.")
    cov_matrix = combined.cov()
    return cov_matrix[m_name].drop(m_name)


def calculate_market_variance(market_returns: pd.Series) -> float:
    """
    Calculate variance of the market return series.

    Parameters
    ----------
    market_returns : pd.Series
        Daily returns of the market benchmark.

    Returns
    -------
    float
        Variance of market returns.
    """
    cleaned = market_returns.dropna()
    if len(cleaned) < 2:
        raise ValueError("At least 2 observations required to calculate variance.")
    var = float(cleaned.var())
    if var <= 0:
        raise ValueError("Market variance must be greater than zero.")
    return var


def calculate_beta(
    stock_returns: pd.DataFrame | pd.Series,
    market_returns: pd.Series,
) -> pd.Series | float:
    """
    Calculate CAPM Beta from covariance with market divided by market variance.

    Parameters
    ----------
    stock_returns : pd.DataFrame or pd.Series
        Daily returns of stocks.
    market_returns : pd.Series
        Daily returns of the market benchmark.

    Returns
    -------
    pd.Series or float
        Calculated beta(s).
    """
    cov = calculate_covariance_with_market(stock_returns, market_returns)
    m_var = calculate_market_variance(market_returns)
    return cov / m_var


def calculate_daily_risk_free_rate(
    annual_risk_free_rate: float,
    trading_days: int = 252,
) -> float:
    """
    Convert an annual risk-free rate to an approximate daily compounding rate.

    Parameters
    ----------
    annual_risk_free_rate : float
        Annualized risk-free rate (e.g. 0.045 for 4.5%).
    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Equivalent daily risk-free rate.
    """
    return float((1.0 + annual_risk_free_rate) ** (1.0 / trading_days) - 1.0)


def calculate_excess_returns(
    returns: pd.DataFrame | pd.Series,
    risk_free_rate: float,
    is_annual_rate: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame | pd.Series:
    """
    Subtract risk-free rate from asset returns to compute excess returns.

    Parameters
    ----------
    returns : pd.DataFrame or pd.Series
        Asset returns.
    risk_free_rate : float
        Risk-free rate.
    is_annual_rate : bool, default=True
        Whether risk_free_rate is annualized (if True, converts to daily).
    trading_days : int, default=252
        Trading days for annual-to-daily conversion.

    Returns
    -------
    pd.DataFrame or pd.Series
        Excess returns.
    """
    rf = calculate_daily_risk_free_rate(risk_free_rate, trading_days) if is_annual_rate else risk_free_rate
    return returns - rf


def calculate_capm_expected_return(
    beta: pd.Series | float,
    market_return: float,
    risk_free_rate: float,
) -> pd.Series | float:
    """
    Calculate CAPM expected return given beta, market return, and risk-free rate.

    Parameters
    ----------
    beta : pd.Series or float
        Asset beta(s).
    market_return : float
        Annualized market return.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    pd.Series or float
        Expected return under CAPM.
    """
    market_risk_premium = market_return - risk_free_rate
    return risk_free_rate + (beta * market_risk_premium)


def run_capm_regression(
    stock_returns: pd.DataFrame | pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    is_annual_rate: bool = True,
    trading_days: int = 252,
) -> dict[str, Any]:
    """
    Run OLS CAPM regression of stock excess returns against market excess returns.

    Model:
        R_{i,t} - R_{f,t} = alpha_i + beta_i * (R_{m,t} - R_{f,t}) + epsilon_{i,t}

    Parameters
    ----------
    stock_returns : pd.DataFrame or pd.Series
        Historical returns of stock(s).
    market_returns : pd.Series
        Historical returns of market benchmark.
    risk_free_rate : float, default=0.0
        Risk-free rate.
    is_annual_rate : bool, default=True
        Whether risk_free_rate is annualized.
    trading_days : int, default=252
        Trading days for annual-to-daily conversion.

    Returns
    -------
    dict[str, Any]
        Dictionary with alpha, beta, r_squared, residuals, models.
    """
    is_single_series = isinstance(stock_returns, pd.Series)
    stocks_df = stock_returns.to_frame() if is_single_series else stock_returns.copy()

    m_name = market_returns.name or "MARKET"
    combined = pd.concat([stocks_df, market_returns.rename(m_name)], axis=1).dropna()
    if len(combined) < 2:
        raise ValueError("Insufficient overlapping data points for regression.")

    aligned_market = combined[m_name]
    aligned_stocks = combined.drop(columns=[m_name])

    rf_daily = calculate_daily_risk_free_rate(risk_free_rate, trading_days) if is_annual_rate else risk_free_rate

    stock_excess = aligned_stocks - rf_daily
    market_excess = aligned_market - rf_daily

    alphas = {}
    betas = {}
    r_squareds = {}
    residuals = pd.DataFrame(index=combined.index)
    models = {}

    try:
        import statsmodels.api as sm

        has_statsmodels = True
    except ImportError:
        has_statsmodels = False

    x_vals = market_excess.to_numpy(dtype=float)
    x_mean = np.mean(x_vals)
    ss_x = np.sum((x_vals - x_mean) ** 2)

    for col in stock_excess.columns:
        y_vals = stock_excess[col].to_numpy(dtype=float)

        if has_statsmodels:
            x_matrix = sm.add_constant(market_excess)
            model = sm.OLS(stock_excess[col], x_matrix).fit()
            alphas[col] = float(model.params.get("const", model.params.iloc[0]))
            betas[col] = float(model.params.get(m_name, model.params.iloc[1]))
            r_squareds[col] = float(model.rsquared)
            residuals[col] = model.resid
            models[col] = model
        else:
            y_mean = np.mean(y_vals)
            cov_xy = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            beta_val = float(cov_xy / ss_x) if ss_x > 0 else 0.0
            alpha_val = float(y_mean - beta_val * x_mean)
            predicted = alpha_val + beta_val * x_vals
            resid = y_vals - predicted
            ss_tot = np.sum((y_vals - y_mean) ** 2)
            ss_res = np.sum(resid**2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            alphas[col] = alpha_val
            betas[col] = beta_val
            r_squareds[col] = r2
            residuals[col] = resid
            models[col] = {
                "params": {"const": alpha_val, m_name: beta_val},
                "rsquared": r2,
                "resid": resid,
            }

    alpha_series = pd.Series(alphas, name="Alpha")
    beta_series = pd.Series(betas, name="Beta")
    r2_series = pd.Series(r_squareds, name="R2")

    return {
        "alpha": alpha_series.iloc[0] if is_single_series else alpha_series,
        "beta": beta_series.iloc[0] if is_single_series else beta_series,
        "r_squared": r2_series.iloc[0] if is_single_series else r2_series,
        "residuals": residuals.iloc[:, 0] if is_single_series else residuals,
        "models": models[stock_returns.name or stock_excess.columns[0]] if is_single_series else models,
    }


def compare_betas(
    covariance_beta: pd.Series | float,
    regression_beta: pd.Series | float,
) -> pd.DataFrame:
    """Compare covariance beta and regression beta."""
    cov_s = pd.Series(covariance_beta) if not isinstance(covariance_beta, pd.Series) else covariance_beta
    reg_s = pd.Series(regression_beta) if not isinstance(regression_beta, pd.Series) else regression_beta

    df = pd.DataFrame(
        {
            "Covariance Beta": cov_s,
            "Regression Beta": reg_s,
        }
    )
    df["Difference"] = df["Covariance Beta"] - df["Regression Beta"]
    return df


def compare_returns(
    historical_returns: pd.Series | float,
    capm_expected_returns: pd.Series | float,
) -> pd.DataFrame:
    """Compare annualized historical returns against CAPM expected returns."""
    hist_s = pd.Series(historical_returns) if not isinstance(historical_returns, pd.Series) else historical_returns
    capm_s = pd.Series(capm_expected_returns) if not isinstance(capm_expected_returns, pd.Series) else capm_expected_returns

    df = pd.DataFrame(
        {
            "Historical Return": hist_s,
            "CAPM Expected Return": capm_s,
        }
    )
    df["Difference"] = df["Historical Return"] - df["CAPM Expected Return"]
    return df


def summarize_residuals(residuals: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Summarize regression residual behavior."""
    res_df = residuals.to_frame() if isinstance(residuals, pd.Series) else residuals
    return pd.DataFrame(
        {
            "Mean Residual": res_df.mean(),
            "Residual Std Dev": res_df.std(),
            "Min Residual": res_df.min(),
            "Max Residual": res_df.max(),
        }
    )
