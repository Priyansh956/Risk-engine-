"""Value at Risk (VaR), Conditional Value at Risk (CVaR), and backtesting analytics."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def calculate_historical_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Historical Value at Risk (VaR).

    Expressed as a positive loss percentage (e.g., 0.025 for 2.5% loss).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    confidence : float, default=0.95
        Confidence level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    float
        Historical VaR loss threshold as a positive number.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")

    clean_returns = np.asarray(returns[~np.isnan(returns)] if isinstance(returns, np.ndarray) else returns.dropna(), dtype=float)
    if len(clean_returns) < 2:
        raise ValueError("At least 2 observations required to compute VaR.")

    alpha_percentile = (1.0 - confidence) * 100.0
    cutoff = float(np.percentile(clean_returns, alpha_percentile))
    return float(max(0.0, -cutoff))


def calculate_historical_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Historical Conditional Value at Risk (CVaR / Expected Shortfall).

    Average loss given that the loss exceeds Historical VaR.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        Expected loss conditional on exceeding VaR.
    """
    var = calculate_historical_var(returns, confidence=confidence)
    clean_returns = np.asarray(returns[~np.isnan(returns)] if isinstance(returns, np.ndarray) else returns.dropna(), dtype=float)

    tail_losses = clean_returns[clean_returns <= -var]
    if len(tail_losses) == 0:
        return float(var)

    return float(max(var, -np.mean(tail_losses)))


def calculate_parametric_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    modified: bool = False,
) -> float:
    """
    Calculate Parametric (Variance-Covariance) Value at Risk.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    confidence : float, default=0.95
        Confidence level.
    modified : bool, default=False
        If True, applies Cornish-Fisher expansion for skewness and excess kurtosis.

    Returns
    -------
    float
        Parametric VaR loss threshold as a positive number.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")

    clean_returns = np.asarray(returns[~np.isnan(returns)] if isinstance(returns, np.ndarray) else returns.dropna(), dtype=float)
    if len(clean_returns) < 2:
        raise ValueError("At least 2 observations required to compute parametric VaR.")

    mu = float(np.mean(clean_returns))
    sigma = float(np.std(clean_returns, ddof=1))
    if sigma <= 0:
        return 0.0

    z = float(stats.norm.ppf(1.0 - confidence))

    if modified and len(clean_returns) >= 4:
        s = float(stats.skew(clean_returns))
        k = float(stats.kurtosis(clean_returns))  # Fisher excess kurtosis
        z_mod = z + (s / 6.0) * (z**2 - 1.0) + (k / 24.0) * (z**3 - 3.0 * z) - (s**2 / 36.0) * (2.0 * z**3 - 5.0 * z)
        var = -(mu + z_mod * sigma)
    else:
        var = -(mu + z * sigma)

    return float(max(0.0, var))


def calculate_parametric_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Gaussian Parametric Conditional Value at Risk (Expected Shortfall).

    Formula:
        CVaR = -(mu - sigma * phi(z) / (1 - confidence))

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        Parametric Gaussian CVaR.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")

    clean_returns = np.asarray(returns[~np.isnan(returns)] if isinstance(returns, np.ndarray) else returns.dropna(), dtype=float)
    if len(clean_returns) < 2:
        raise ValueError("At least 2 observations required to compute parametric CVaR.")

    mu = float(np.mean(clean_returns))
    sigma = float(np.std(clean_returns, ddof=1))
    if sigma <= 0:
        return 0.0

    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(alpha))
    pdf_z = float(stats.norm.pdf(z))

    cvar = -(mu - sigma * (pdf_z / alpha))
    return float(max(0.0, cvar))


def calculate_portfolio_var_cvar(
    returns: pd.Series | np.ndarray,
    confidence_levels: list[float] | None = None,
) -> dict[str, float]:
    """
    Calculate comprehensive suite of VaR and CVaR metrics across standard confidence levels.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    confidence_levels : list[float], optional
        Confidence levels to evaluate. Defaults to [0.95, 0.99].

    Returns
    -------
    dict[str, float]
        Dictionary of historical and parametric VaR and CVaR metrics.
    """
    conf_list = confidence_levels or [0.95, 0.99]
    metrics = {}

    for conf in conf_list:
        pct_tag = int(round(conf * 100))
        metrics[f"historical_var_{pct_tag}"] = calculate_historical_var(returns, confidence=conf)
        metrics[f"historical_cvar_{pct_tag}"] = calculate_historical_cvar(returns, confidence=conf)
        metrics[f"parametric_var_{pct_tag}"] = calculate_parametric_var(returns, confidence=conf, modified=False)
        metrics[f"parametric_cvar_{pct_tag}"] = calculate_parametric_cvar(returns, confidence=conf)
        metrics[f"modified_var_{pct_tag}"] = calculate_parametric_var(returns, confidence=conf, modified=True)

    return metrics


def backtest_var(
    returns: pd.Series | np.ndarray,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> dict[str, Any]:
    """
    Perform rolling out-of-sample backtest of Value at Risk.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily return series.
    window : int, default=252
        Lookback window in trading days for rolling VaR estimation.
    confidence : float, default=0.95
        VaR confidence level (e.g. 0.95).
    method : {"historical", "parametric", "modified"}, default="historical"
        VaR calculation method.

    Returns
    -------
    dict[str, Any]
        Backtest summary report including breach counts, breach rate, expected breach rate,
        Kupiec POF likelihood ratio test, rolling VaR series, and breach indicators.
    """
    if isinstance(returns, np.ndarray):
        series = pd.Series(returns).dropna()
    else:
        series = returns.dropna()

    n = len(series)
    if n <= window:
        raise ValueError(f"Total observations ({n}) must be strictly greater than rolling window ({window}).")

    rolling_var = []
    test_indices = []
    realized_rets = []

    for t in range(window, n):
        hist_window = series.iloc[t - window : t]
        if method == "parametric":
            var_t = calculate_parametric_var(hist_window, confidence=confidence, modified=False)
        elif method == "modified":
            var_t = calculate_parametric_var(hist_window, confidence=confidence, modified=True)
        else:
            var_t = calculate_historical_var(hist_window, confidence=confidence)

        rolling_var.append(var_t)
        test_indices.append(series.index[t])
        realized_rets.append(float(series.iloc[t]))

    var_series = pd.Series(rolling_var, index=test_indices, name="Rolling_VaR")
    ret_series = pd.Series(realized_rets, index=test_indices, name="Realized_Return")

    # A breach occurs when realized return is below -VaR (i.e. realized loss > VaR threshold)
    breach_series = ret_series < -var_series

    total_obs = len(breach_series)
    num_breaches = int(breach_series.sum())
    breach_rate = float(num_breaches / total_obs) if total_obs > 0 else 0.0
    expected_rate = float(1.0 - confidence)

    # Kupiec Proportion of Failures (POF) Likelihood Ratio test
    # LR_POF = -2 * ln( (1-p)^(N-x) * p^x / ((1 - x/N)^(N-x) * (x/N)^x) )
    p = expected_rate
    x = num_breaches
    N = total_obs

    if x == 0:
        lr_stat = float(-2.0 * np.log((1.0 - p) ** N))
    elif x == N:
        lr_stat = float(-2.0 * np.log(p**N))
    else:
        log_num = (N - x) * np.log(1.0 - p) + x * np.log(p)
        log_den = (N - x) * np.log(1.0 - (x / N)) + x * np.log(x / N)
        lr_stat = float(-2.0 * (log_num - log_den))

    kupiec_p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return {
        "total_observations": total_obs,
        "breaches": num_breaches,
        "breach_rate": breach_rate,
        "expected_breach_rate": expected_rate,
        "breach_ratio": float(breach_rate / expected_rate) if expected_rate > 0 else 0.0,
        "kupiec_lr_stat": lr_stat,
        "kupiec_p_value": kupiec_p_value,
        "method": method,
        "confidence": confidence,
        "rolling_var": var_series,
        "realized_returns": ret_series,
        "breach_series": breach_series,
    }
