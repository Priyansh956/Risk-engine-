# Quant Risk Engine (`quant-risk-engine`)

[![CI](https://github.com/Priyansh956/Risk-engine-/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyansh956/Risk-engine-/actions/workflows/ci.yml)

A quantitative portfolio risk and analytics library in Python. Give it tickers and weights, and it returns return, volatility, Sharpe ratio, CAPM beta, and a correlated Monte Carlo forecast in one call.

Consolidates portfolio return analytics, Markowitz Modern Portfolio Theory, Capital Asset Pricing Model (CAPM) regression, and multivariate Monte Carlo simulation into a clean `src` layout.

> **Status: early development (v0.1.0).** The core calculations are implemented and tested, but the project is still being built out. Read [Known limitations](#known-limitations) before relying on any results.

---

## Features

- **Return & Risk Engine (`risk_engine.returns`, `risk_engine.risk`)**:
  - Logarithmic return calculation for Series and DataFrames.
  - Annualized compounded effective return calculation ($\text{trading\_days} = 252$).
  - Annualized variance, volatility, covariance, and correlation matrices.
  - Portfolio variance ($\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$) and Sharpe ratio calculation.

- **Markowitz Modern Portfolio Theory (`risk_engine.markowitz`)**:
  - Long-only random-portfolio simulation across random weightings (sampling-based).
  - Tracking of Maximum Sharpe Ratio and Minimum Variance portfolios among the simulated portfolios.
  - Approximate efficient frontier extracted from the simulated portfolios.

- **Capital Asset Pricing Model (`risk_engine.capm`)**:
  - Market benchmark returns and covariance-based Beta calculation ($\beta = \operatorname{Cov}(R_i, R_m) / \operatorname{Var}(R_m)$).
  - Effective annual risk-free rate conversion to daily compounding rate: $(1 + R_f)^{1/252} - 1$.
  - OLS CAPM Regression:
    $$R_{i, t} - R_{f, t} = \alpha_i + \beta_i (R_{m, t} - R_{f, t}) + \epsilon_{i, t}$$
    Extracts regression $\alpha$, regression $\beta$, coefficient of determination $R^2$, and residuals $\epsilon$.
  - Beta and Return comparison tables (Covariance Beta vs. Regression Beta, Historical vs. CAPM Expected).
  - Residual summary statistics (mean, standard deviation, min, max).

- **Correlated Monte Carlo Simulation (`risk_engine.monte_carlo`)**:
  - Daily drift calculation using daily metrics: $\text{drift} = \mu_{\text{daily}} - \frac{1}{2} \sigma^2_{\text{daily}}$.
  - **Cholesky Decomposition for Correlated Shocks**: Decomposes covariance matrix $\mathbf{\Sigma} = L L^T$ so simulated assets reflect empirical correlations rather than independent shocks.
  - **Capital-Weighted Portfolio Paths**: Normalizes asset paths by initial prices $S_0$ so portfolio weights model capital allocation rather than raw share counts:
    $$V_t = V_0 \sum_{i=1}^N w_i \frac{S_{t, i}}{S_{0, i}}$$
  - Scenario analysis partitioning paths into **Bull Market** (top 10%), **Base Case** (middle 80%), and **Bear Market** (bottom 10%) with scenario metrics and terminal statistics.

- **Unified Portfolio Orchestrator (`risk_engine.portfolio`)**:
  - `analyze_portfolio()`: Single entry point validating inputs, evaluating returns, volatility, Sharpe, beta, CAPM expected return, and running correlated Monte Carlo simulation.
  - `PortfolioAnalysisResult`: Clean result object with JSON-serializable `.to_dict()`.

- **Data Fetching (`risk_engine.data_loader`)**:
  - Download historical adjusted close prices for tickers, market benchmarks (`^GSPC`), and US Treasury yield (`^TNX`) using [yfinance](https://github.com/ranaroussi/yfinance).

---

## Installation

Requires **Python 3.11 or higher**.

```bash
git clone https://github.com/Priyansh956/Risk-engine-.git
cd Risk-engine-
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Quickstart: `analyze_portfolio()`

This example downloads data, so it needs an internet connection.

```python
from risk_engine import analyze_portfolio

# Analyze a two-stock portfolio (60% AAPL, 40% MSFT)
result = analyze_portfolio(
    tickers=["AAPL", "MSFT"],
    weights=[0.6, 0.4],
    start_date="2020-01-01",
    min_history=252,  # Requires at least 1 trading year
    mc_time_horizon=252,  # 1-year forward forecast
    mc_simulations=1000,  # 1,000 simulation paths
    random_state=42,
)

print(f"Annualized Historical Return: {result.portfolio_return:.2%}")
print(f"Annual Volatility:            {result.portfolio_volatility:.2%}")
print(f"Sharpe Ratio:                 {result.sharpe_ratio:.2f}")
print(f"Portfolio Beta:               {result.portfolio_beta:.2f}")
print(f"CAPM Expected Return:         {result.capm_expected_return:.2%}")

# Monte Carlo terminal summary: multiples of the starting value (1.00 = break-even)
mc = result.monte_carlo_summary
print(f"MC Terminal Median:           {mc['terminal_median']:.2f}x")
print(f"MC Worst Case (5th %):        {mc['terminal_5th_percentile']:.2f}x")
print(f"MC Best Case (95th %):        {mc['terminal_95th_percentile']:.2f}x")

# Check for any fallback values that were used (see Validation & Warnings)
print(result.warnings)

# JSON-ready dictionary output (safe for APIs/web apps)
data_dict = result.to_dict(include_paths=False)
```

The Monte Carlo figures are multiples of your starting portfolio value, not currency amounts. A median of `1.08x` means the typical simulated outcome ends 8% above where it started.

---

## Return Conventions

All returns throughout the engine are standardized on the **compounded effective annual return convention**:

$$R_{\text{annual}} = \exp(\bar{r}_{\text{daily, log}} \times \text{trading\_days}) - 1$$

- **Stock / Portfolio Return**: Compounded annualized return from daily log returns.
- **Market Return**: Compounded annualized return from benchmark log returns.
- **Risk-Free Rate**: Effective annual yield $R_f$ (e.g., 0.04 for 4.0%).
- **Daily Risk-Free Rate**: $(1 + R_f)^{1/252} - 1$.
- **Market Risk Premium**: $R_m - R_f$.
- **CAPM Expected Return**: $R_f + \beta (R_m - R_f)$.

---

## Validation & Warnings

`analyze_portfolio()` strictly validates inputs before computing metrics:

1. **Weights Sum**: Raises `ValueError` if $\sum w_i \neq 1.0$.
2. **Weight Dimension**: Raises `ValueError` if weight count does not equal ticker count.
3. **Long-Only**: Raises `ValueError` if any weight is negative.
4. **Unique Tickers**: Raises `ValueError` if duplicate tickers are supplied (e.g. `["AAPL", "AAPL"]`).
5. **Ticker Existence**: Raises `ValueError` if any ticker is missing from price data.
6. **Sufficient History**: Raises `ValueError` if historical observations are fewer than `min_history` (default 252 days). If you lower `min_history` below 252, a warning is issued because covariance and beta estimates become noisy.
7. **Transparent Fallbacks & Warnings**:
   - If market data or the risk-free yield cannot be fetched, a default fallback is used, recorded in `result.warnings`, and reported via `warnings.warn`. **Always check `result.warnings`**, because CAPM results based on a fallback are not real market estimates.
   - Set `raise_on_fallback=True` to raise an error instead of using fallbacks.

---

## Notebook

`notebooks/main.ipynb` is a step-by-step walkthrough of the analysis: returns, risk, Markowitz, CAPM, and Monte Carlo. It is a standalone learning notebook and does not call the `risk_engine` package. It also uses libraries that are not installed with the package:

```bash
pip install matplotlib jupyter
```

---

## Known limitations

- **Optimization is sampling-based.** `simulate_random_portfolios` tries random weights in a Python loop. The max-Sharpe and min-variance results and the efficient frontier are approximations, the loop is slow for very large simulation counts, and weight constraints are not supported.
- **Monte Carlo assumes Geometric Brownian Motion.** Returns are modeled as normally distributed with parameters estimated from past data, so extreme market moves are underrepresented and results depend on the history window you choose.
- **Results describe the past, not the future.** "Return" is the annualized historical return. The CAPM expected return uses the historical market return as its estimate of the market premium.
- **Monte Carlo values are normalized.** Portfolio paths start at 1.0, so results are multiples of the starting value.
- **Data and defaults are US-centric.** Data comes from yfinance (an unofficial Yahoo Finance wrapper, so availability is outside this project's control), with the S&P 500 (`^GSPC`) as the benchmark and the US 10-year Treasury yield (`^TNX`) as the risk-free rate. For other markets, pass your own `market_prices` and `risk_free_rate` to `analyze_portfolio()`.
- **Not implemented yet:** Value at Risk (VaR), Conditional VaR, drawdown analysis, and any web interface or API.

## Roadmap

- [ ] Value at Risk (historical, parametric, Monte Carlo), Conditional VaR, and maximum drawdown
- [ ] Exact constrained optimization (min-variance, max-Sharpe) using a solver instead of random sampling
- [ ] Vectorize the random-portfolio simulation
- [ ] Web app for trying your own portfolio
- [ ] Indian market support (NSE tickers, Nifty 50 benchmark, Indian risk-free rate)
- [ ] Publish to PyPI

---

## Project Structure

```
├── .github/workflows/ci.yml           # CI: lint, format check, tests
├── pyproject.toml                     # Package metadata, dependencies, & tool configs
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Testing & linting tools
├── notebooks/
│   └── main.ipynb                     # Step-by-step analysis walkthrough
├── src/
│   └── risk_engine/                   # Primary package
│       ├── __init__.py                # Package exports
│       ├── returns.py                 # Return calculations
│       ├── risk.py                    # Risk & volatility metrics
│       ├── markowitz.py               # Modern Portfolio Theory & Efficient Frontier
│       ├── capm.py                    # CAPM regression & beta analysis
│       ├── monte_carlo.py             # Correlated GBM simulation & scenario filtering
│       ├── data_loader.py             # Historical data loading (yfinance)
│       └── portfolio.py               # analyze_portfolio orchestrator & PortfolioAnalysisResult
└── tests/
    ├── test_returns.py
    ├── test_risk_parameters.py
    ├── test_markowitz.py
    ├── test_capm.py
    ├── test_monte_carlo.py
    ├── test_data_loader.py
    └── test_portfolio.py
```

---

## Testing & Quality

Run the test suite:

```bash
pytest
```

Run code formatting and linting:

```bash
ruff check .
ruff format --check .
```

---

## Contributing

Contributions and bug reports are welcome. Please open an issue to discuss larger changes first. Before submitting a pull request, make sure `pytest`, `ruff check .`, and `ruff format --check .` all pass.

## Disclaimer

This project is for educational and informational purposes only and is not financial advice. Results are based on historical data, which does not predict future performance. Do not make investment decisions based solely on this tool.

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
