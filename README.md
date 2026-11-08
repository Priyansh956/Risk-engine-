# Quant Risk Engine

[![CI](https://github.com/Priyansh956/Risk-engine-/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyansh956/Risk-engine-/actions/workflows/ci.yml)

A Python library for quantitative portfolio analysis. Give it historical prices and portfolio weights, and it computes returns, risk, Sharpe ratio, Markowitz risk-return analysis, CAPM beta, and Monte Carlo price simulations.

> **Status: early development (v0.1.0).** The core calculations are implemented and tested, but the project is still being built out. See [Known limitations](#known-limitations) before relying on any results, and the [Roadmap](#roadmap) for what is coming next.

## Features

| Module | What it does |
|---|---|
| `risk_engine` | Log returns, annualized return, annualized variance / standard deviation / covariance / correlation, portfolio variance and return, Sharpe ratio |
| `markowitz` | Portfolio expected return, variance, volatility and Sharpe; random long-only portfolio simulation (min-variance and max-Sharpe portfolios); approximate efficient frontier |
| `capm` | Beta (covariance method), OLS regression of excess returns (alpha, beta, R², residuals), CAPM expected return, beta and return comparison helpers |
| `monte_carlo` | Geometric Brownian Motion price simulation, terminal price summaries (mean, median, percentiles), bull / base / bear scenario analysis |
| `data_loader` | Downloads historical prices, a market benchmark, and a risk-free rate using [yfinance](https://github.com/ranaroussi/yfinance) |

## Getting started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```sh
git clone https://github.com/Priyansh956/Risk-engine-.git
cd Risk-engine-

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .
pip install -r requirements-dev.txt  # only needed for running tests and linting
```

The notebook (`notebooks/main.ipynb`) also uses `matplotlib` and Jupyter, which are not installed by the commands above:

```sh
pip install matplotlib jupyter
```

## Quick start

The example below analyzes a three-stock portfolio. It needs an internet connection to download data.

```python
import numpy as np

from capm import calculate_market_returns, run_capm_regression
from data_loader import fetch_market_data, fetch_risk_free_rate, fetch_stock_data
from markowitz import portfolio_metrics, simulate_random_portfolios
from risk_engine import (
    calculate_annual_covariance_of_stocks,
    calculate_annualized_return,
    calculate_log_returns,
)

tickers = ["AAPL", "MSFT", "NKE"]
weights = np.array([0.5, 0.3, 0.2])  # must sum to 1

# 1. Load data
prices = fetch_stock_data(tickers, start="2020-01-01")
market_prices = fetch_market_data("^GSPC", start="2020-01-01")
risk_free_rate = fetch_risk_free_rate()  # latest US 10-year Treasury yield, e.g. 0.042

# 2. Returns and risk
log_returns = calculate_log_returns(prices).dropna()
annual_returns = calculate_annualized_return(log_returns)
annual_cov = calculate_annual_covariance_of_stocks(log_returns)

portfolio_return, volatility, sharpe = portfolio_metrics(
    weights,
    annual_returns.to_numpy(),
    annual_cov.to_numpy(),
    risk_free_rate,
)
print(f"Return: {portfolio_return:.2%}  Volatility: {volatility:.2%}  Sharpe: {sharpe:.2f}")

# 3. Markowitz: simulate random portfolios and find the best ones
simulation = simulate_random_portfolios(
    annual_returns.to_numpy(),
    annual_cov.to_numpy(),
    risk_free_rate,
    num_simulations=10_000,
    random_state=42,
)
print("Max-Sharpe weights:", dict(zip(tickers, simulation["max_sharpe_weights"].round(3))))

# 4. CAPM: beta, alpha and R² against the market
market_returns = calculate_market_returns(market_prices)
capm = run_capm_regression(log_returns, market_returns, risk_free_rate=risk_free_rate)
print(capm["beta"])
```

Monte Carlo price simulation for each stock:

```python
from monte_carlo import calculate_daily_drift, simulate_gbm_prices, summarize_terminal_prices

paths = simulate_gbm_prices(
    initial_prices=prices.iloc[-1],
    drift=calculate_daily_drift(log_returns),
    daily_std=log_returns.std(),
    time_horizon=252,       # trading days (about one year)
    num_simulations=1_000,
    random_state=1,
)
print(summarize_terminal_prices(paths, asset_names=tickers))
```

> Use `simulate_gbm_prices` for per-stock simulations only for now. `simulate_portfolio_paths` has a known bug (see below).

## Project layout

```
src/
├── risk_engine/   # returns and core risk metrics
├── markowitz/     # portfolio metrics, random-portfolio simulation, efficient frontier
├── capm/          # beta, regression, CAPM expected return
├── monte_carlo/   # GBM simulation and scenario analysis
└── data_loader/   # yfinance data access
tests/             # pytest suite for every module
notebooks/         # main.ipynb: step-by-step walkthrough of the analysis
```

## Development

```sh
pytest             # run the test suite
ruff check .       # lint
ruff format .      # format
```

CI runs the same lint, format check, and tests on every push to `main` / `develop` and on pull requests to `main`.

## Known limitations

These are known issues in the current version. They are being tracked and will be fixed.

### Monte Carlo

- **`simulate_portfolio_paths` weights raw prices instead of capital (bug).** It computes `Σ wᵢ × Pᵢ(t)` on raw simulated prices, which models holding `wᵢ` *shares* of each stock rather than investing a fraction `wᵢ` of your *money* in it. A stock with a high share price therefore dominates the result regardless of the weights. For example, with 50/50 weights on two stocks priced near 100 and 3,000, about 98% of the simulated portfolio's starting value comes from the more expensive stock. **Portfolio-level Monte Carlo results from this function should not be trusted yet.** The fix is to apply weights to price paths normalized by each stock's starting price (or to returns).
- **Simulated stocks are uncorrelated.** `simulate_gbm_returns` draws independent random shocks for each stock, ignoring the correlations between them. This understates portfolio risk. The fix is to generate correlated shocks from the covariance matrix (Cholesky decomposition).
- **Returns are assumed normally distributed** (standard Geometric Brownian Motion), so extreme market moves are underrepresented.
- **`simulate_gross_profit` is out of scope.** It simulates company revenue and cost of goods sold, which is corporate-finance logic rather than portfolio analysis, and it is planned for removal.

### Optimization

- `simulate_random_portfolios` approximates the efficient frontier by random sampling. It does not solve the optimization exactly and cannot handle constraints such as weight limits.

### Data and defaults

- Data comes from yfinance, an unofficial Yahoo Finance wrapper, so availability and data quality are outside this project's control.
- Defaults are US-centric: the S&P 500 (`^GSPC`) as the market benchmark and the US 10-year Treasury yield (`^TNX`) as the risk-free rate. Indian markets are not yet supported out of the box.
- Invalid tickers or missing data are not yet reported with clear error messages.

### Missing features

- There is no Value at Risk (VaR), Conditional VaR, or drawdown analysis yet.
- There is no web interface or API. The project is currently used as a Python library or through the notebook.

## Roadmap

- [ ] Fix the Monte Carlo issues listed above
- [ ] Single `analyze_portfolio()` entry point that validates inputs and returns a full report
- [ ] Clear error handling for invalid tickers, weights, and missing data
- [ ] Value at Risk (historical, parametric, Monte Carlo), Conditional VaR, maximum drawdown
- [ ] Exact constrained optimization (min-variance, max-Sharpe) with `scipy`
- [ ] Web app for trying your own portfolio
- [ ] Support for Indian markets (NSE tickers, Nifty 50 benchmark, Indian risk-free rate)
- [ ] Publish to PyPI under a single package namespace

## Disclaimer

This project is for educational and informational purposes only and is not financial advice. Results are based on historical data, which does not predict future performance. Do not make investment decisions based solely on this tool.

## Contributing

Contributions and bug reports are welcome. Please open an issue to discuss larger changes first. Before submitting a pull request, make sure `pytest`, `ruff check .`, and `ruff format --check .` all pass.

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
