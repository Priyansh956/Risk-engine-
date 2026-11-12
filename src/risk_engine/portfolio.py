"""Portfolio analysis orchestrator combining returns, risk, CAPM, and Monte Carlo simulation."""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_engine.capm import (
    calculate_beta,
    calculate_capm_expected_return,
    calculate_market_returns,
)
from risk_engine.data_loader import (
    fetch_market_data,
    fetch_risk_free_rate,
    fetch_stock_data,
)
from risk_engine.markowitz import (
    portfolio_expected_return,
    portfolio_volatility,
    sharpe_ratio,
)
from risk_engine.monte_carlo import (
    analyze_scenarios,
    calculate_daily_drift,
    simulate_gbm_prices,
    simulate_portfolio_paths,
)
from risk_engine.returns import (
    calculate_annualized_return,
    calculate_log_returns,
)
from risk_engine.risk import (
    calculate_annual_covariance_of_stocks,
)
from risk_engine.var import (
    calculate_portfolio_var_cvar,
)


@dataclass
class PortfolioAnalysisResult:
    """Consolidated portfolio analysis results."""

    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    portfolio_beta: float
    capm_expected_return: float
    monte_carlo_summary: dict[str, Any]
    individual_returns: pd.Series
    individual_betas: pd.Series
    weights: pd.Series
    risk_free_rate: float
    var_metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, include_paths: bool = False) -> dict[str, Any]:
        """
        Convert analysis result to a JSON-serializable dictionary.

        Parameters
        ----------
        include_paths : bool, default=False
            Whether to include full Monte Carlo simulation path arrays.
            If False, raw simulation paths are excluded to keep the dict lightweight
            and strictly JSON-serializable.

        Returns
        -------
        dict[str, Any]
            JSON-serializable representation of portfolio analysis results.
        """
        mc_clean = {}
        for k, v in self.monte_carlo_summary.items():
            if k == "scenarios":
                sc_dict = {}
                for sk, sv in v.items():
                    if sk in ("bull_paths", "base_paths", "bear_paths"):
                        if include_paths and hasattr(sv, "tolist"):
                            sc_dict[sk] = sv.tolist()
                    elif isinstance(sv, (np.floating, float)):
                        sc_dict[sk] = float(sv)
                    elif isinstance(sv, (np.integer, int)):
                        sc_dict[sk] = int(sv)
                    else:
                        sc_dict[sk] = sv
                mc_clean["scenarios"] = sc_dict
            elif isinstance(v, (np.floating, float)):
                mc_clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                mc_clean[k] = int(v)
            else:
                mc_clean[k] = v

        return {
            "portfolio_return": float(self.portfolio_return),
            "portfolio_volatility": float(self.portfolio_volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "portfolio_beta": float(self.portfolio_beta),
            "capm_expected_return": float(self.capm_expected_return),
            "monte_carlo_summary": mc_clean,
            "individual_returns": {str(k): float(v) for k, v in self.individual_returns.items()},
            "individual_betas": {str(k): float(v) for k, v in self.individual_betas.items()},
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "risk_free_rate": float(self.risk_free_rate),
            "var_metrics": {str(k): float(v) for k, v in self.var_metrics.items()},
            "warnings": list(self.warnings),
        }


def analyze_portfolio(
    tickers: list[str],
    weights: list[float] | np.ndarray | pd.Series,
    prices: pd.DataFrame | None = None,
    market_prices: pd.Series | None = None,
    risk_free_rate: float | None = None,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    min_history: int = 252,
    trading_days: int = 252,
    mc_time_horizon: int = 252,
    mc_simulations: int = 1000,
    random_state: int | None = None,
    raise_on_fallback: bool = False,
) -> PortfolioAnalysisResult:
    """
    Validate portfolio inputs and analyze return, volatility, Sharpe, beta, CAPM, and Monte Carlo paths.

    Return Convention
    -----------------
    All annualized returns (asset returns, portfolio return, and market return) are computed
    using the compounded effective annual return convention:
        R_annual = exp(mean_daily_log_return * trading_days) - 1
    The risk-free rate is treated as an effective annual yield (e.g., 0.04 for 4.0%).
    """
    warnings_list: list[str] = []

    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    if len(tickers) != len(set(tickers)):
        duplicates = [t for t in set(tickers) if tickers.count(t) > 1]
        raise ValueError(f"Duplicate tickers detected in portfolio: {duplicates}. Tickers must be unique.")

    weights_arr = np.asarray(weights if not isinstance(weights, pd.Series) else weights.values, dtype=float)

    if len(weights_arr) != len(tickers):
        raise ValueError(f"Number of weights ({len(weights_arr)}) does not match number of tickers ({len(tickers)}).")

    if not np.isclose(np.sum(weights_arr), 1.0, atol=1e-3):
        raise ValueError(f"Portfolio weights must sum to 1.0 (got {np.sum(weights_arr):.4f}).")

    if np.any(weights_arr < 0):
        raise ValueError("Portfolio weights must be non-negative for a long-only portfolio.")

    # 1. Fetch or validate stock prices
    if prices is None:
        prices = fetch_stock_data(tickers, start=start_date, end=end_date)

    missing_tickers = [t for t in tickers if t not in prices.columns]
    if missing_tickers:
        raise ValueError(f"The following tickers were not found in price data: {missing_tickers}")

    stock_prices = prices[tickers].dropna()
    if len(stock_prices) < min_history:
        raise ValueError(f"Insufficient price history: found {len(stock_prices)} observations, required at least {min_history}.")

    if len(stock_prices) < 252:
        msg = f"Historical sample has {len(stock_prices)} trading days (<252); covariance, volatility, and beta estimates may be noisy."
        warnings_list.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # 2. Daily log returns & risk engine calculations
    log_returns = calculate_log_returns(stock_prices).dropna()
    if len(log_returns) < 2:
        raise ValueError("Insufficient returns data after calculating log returns.")

    annual_returns = calculate_annualized_return(log_returns, trading_days=trading_days)
    if isinstance(annual_returns, float):
        annual_returns = pd.Series([annual_returns], index=tickers)

    annual_cov = calculate_annual_covariance_of_stocks(log_returns, trading_days=trading_days)

    port_return = portfolio_expected_return(weights_arr, annual_returns.values)
    port_vol = portfolio_volatility(weights_arr, annual_cov.values)

    # Portfolio daily returns & VaR / CVaR calculations
    port_daily_returns = np.dot(log_returns.values, weights_arr)
    var_metrics = calculate_portfolio_var_cvar(port_daily_returns, [0.95, 0.99])

    # 3. Risk-free rate
    if risk_free_rate is None:
        try:
            risk_free_rate = fetch_risk_free_rate()
        except Exception as err:
            if raise_on_fallback:
                raise RuntimeError(f"Failed to fetch risk-free rate from Treasury yield: {err}") from err
            risk_free_rate = 0.04
            msg = "Risk-free rate could not be retrieved from Treasury yield; defaulted risk_free_rate to 0.04 (4.0%)."
            warnings_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    port_sharpe = sharpe_ratio(port_return, port_vol, risk_free_rate)

    # 4. Market data & CAPM analysis
    if market_prices is None:
        try:
            market_prices = fetch_market_data(start=start_date, end=end_date)
        except Exception as err:
            if raise_on_fallback:
                raise RuntimeError(f"Failed to fetch market benchmark data: {err}") from err
            market_prices = None
            msg = "Market benchmark data could not be retrieved; defaulted portfolio beta to 1.0 and CAPM expected return to portfolio return."
            warnings_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    if market_prices is not None:
        market_returns = calculate_market_returns(market_prices).rename("MARKET")
        aligned = pd.concat([log_returns, market_returns], axis=1).dropna()
        if len(aligned) >= 2:
            ind_betas = calculate_beta(aligned[tickers], aligned["MARKET"])
            port_beta = float(np.dot(weights_arr, ind_betas.values))
            annual_market_ret = float(calculate_annualized_return(aligned["MARKET"], trading_days=trading_days))
            capm_exp_ret = float(calculate_capm_expected_return(port_beta, annual_market_ret, risk_free_rate))
        else:
            ind_betas = pd.Series(1.0, index=tickers)
            port_beta = 1.0
            capm_exp_ret = port_return
            msg = "Fewer than 2 overlapping observations between portfolio and market; defaulted beta to 1.0."
            warnings_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
    else:
        ind_betas = pd.Series(1.0, index=tickers)
        port_beta = 1.0
        capm_exp_ret = port_return

    # 5. Monte Carlo simulation with correlated shocks (Cholesky decomposition)
    drift = calculate_daily_drift(log_returns)
    daily_cov = log_returns.cov()
    s0 = stock_prices.iloc[-1].values

    simulated_price_paths = simulate_gbm_prices(
        initial_prices=s0,
        drift=drift,
        time_horizon=mc_time_horizon,
        num_simulations=mc_simulations,
        random_state=random_state,
        covariance_matrix=daily_cov,
    )

    # Normalize by starting prices to weight dollar allocations accurately
    port_paths = simulate_portfolio_paths(
        price_paths=simulated_price_paths,
        weights=weights_arr,
        initial_portfolio_value=1.0,
    )

    scenarios = analyze_scenarios(port_paths)
    terminal_values = port_paths[-1]

    mc_summary = {
        "terminal_mean": float(terminal_values.mean()),
        "terminal_median": float(np.median(terminal_values)),
        "terminal_5th_percentile": float(np.percentile(terminal_values, 5)),
        "terminal_95th_percentile": float(np.percentile(terminal_values, 95)),
        "terminal_std": float(np.std(terminal_values)),
        "scenarios": scenarios,
    }

    return PortfolioAnalysisResult(
        portfolio_return=port_return,
        portfolio_volatility=port_vol,
        sharpe_ratio=port_sharpe,
        portfolio_beta=port_beta,
        capm_expected_return=capm_exp_ret,
        monte_carlo_summary=mc_summary,
        individual_returns=annual_returns,
        individual_betas=ind_betas,
        weights=pd.Series(weights_arr, index=tickers, name="Weights"),
        risk_free_rate=risk_free_rate,
        var_metrics=var_metrics,
        warnings=warnings_list,
    )
