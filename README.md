# Quant Risk Engine (`quant-risk-engine`)

A quantitative portfolio risk, optimization, and analytics engine implemented in Python.

Consolidates portfolio return analytics, Markowitz Modern Portfolio Theory, Capital Asset Pricing Model (CAPM) regression, and multivariate Monte Carlo simulation into a clean `src` layout.

---

## Features

- **Return & Risk Engine (`risk_engine.returns`, `risk_engine.risk`)**:
  - Logarithmic return calculation for Series and DataFrames.
  - Annualized compounded effective return calculation ($\text{trading\_days} = 252$).
  - Annualized variance, volatility, covariance, and correlation matrices.
  - Portfolio variance ($\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$) and Sharpe ratio calculation.

- **Markowitz Modern Portfolio Theory (`risk_engine.markowitz`)**:
  - Vectorized long-only Monte Carlo portfolio simulation across random weightings.
  - Tracking of Maximum Sharpe Ratio and Minimum Variance portfolios.
  - Exact Pareto-optimal Efficient Frontier extraction.

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
  - Download historical adjusted close prices for tickers, market benchmarks (`^GSPC`), and US Treasury yield (`^TNX`).

---

## Installation

```bash
git clone https://github.com/Priyansh956/Risk-engine-.git
cd Risk-engine-
pip install -e .
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Quickstart: `analyze_portfolio()`

```python
from risk_engine import analyze_portfolio

# Analyze a two-stock portfolio (60% AAPL, 40% MSFT)
result = analyze_portfolio(
    tickers=["AAPL", "MSFT"],
    weights=[0.6, 0.4],
    start_date="2020-01-01",
    min_history=252,          # Requires at least 1 trading year
    mc_time_horizon=252,      # 1-year forward forecast
    mc_simulations=1000,      # 1,000 simulation paths
    random_state=42,
)

print(f"Annual Expected Return: {result.portfolio_return:.2%}")
print(f"Annual Volatility:      {result.portfolio_volatility:.2%}")
print(f"Sharpe Ratio:           {result.sharpe_ratio:.2f}")
print(f"Portfolio Beta:         {result.portfolio_beta:.2f}")
print(f"CAPM Expected Return:   {result.capm_expected_return:.2%}")

# Monte Carlo Terminal Summary
mc = result.monte_carlo_summary
print(f"MC Terminal Median:     ${mc['terminal_median']:.2f}")
print(f"MC Worst Case (5th %):  ${mc['terminal_5th_percentile']:.2f}")
print(f"MC Best Case (95th %):  ${mc['terminal_95th_percentile']:.2f}")

# JSON-ready dictionary output (safe for APIs/web apps)
data_dict = result.to_dict(include_paths=False)
```

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
6. **Sufficient History**: Raises `ValueError` if historical observations are fewer than `min_history` (default 252 days).
7. **Transparent Fallbacks & Warnings**:
   - If market data or risk-free yield cannot be fetched, default fallbacks are recorded in `result.warnings` and logged via `warnings.warn`.
   - Set `raise_on_fallback=True` to raise an error instead of using fallbacks.

---

## Project Structure

```
├── pyproject.toml                     # Package metadata, dependencies, & tool configs
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Testing & linting tools
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

