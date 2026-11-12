"""Interactive Streamlit Web Application for Quant Risk Engine."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from risk_engine.data_loader import fetch_market_data, fetch_stock_data
from risk_engine.markowitz import efficient_frontier, simulate_random_portfolios
from risk_engine.monte_carlo import simulate_gbm_prices, simulate_portfolio_paths
from risk_engine.portfolio import analyze_portfolio
from risk_engine.returns import calculate_annualized_return, calculate_log_returns
from risk_engine.risk import calculate_annual_covariance_of_stocks
from risk_engine.var import backtest_var

st.set_page_config(
    page_title="Quant Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Quant Risk Engine")
st.markdown("Quantitative Portfolio Risk Analytics, Markowitz Optimization, CAPM, and Correlated Monte Carlo Simulation.")

# Sidebar Configuration
st.sidebar.header("Portfolio Configuration")

default_tickers = "AAPL, MSFT, GOOGL, NVDA"
tickers_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

default_weights = ", ".join([f"{1.0 / max(1, len(tickers)):.2f}"] * len(tickers))
weights_input = st.sidebar.text_input("Capital Allocation Weights (comma-separated)", value=default_weights)

try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
except ValueError:
    weights = []

if st.sidebar.button("Equalize Weights"):
    if tickers:
        eq_w = 1.0 / len(tickers)
        weights = [eq_w] * len(tickers)
        st.sidebar.info(f"Set {len(tickers)} equal weights of {eq_w:.2%}")

benchmark_ticker = st.sidebar.text_input("Market Benchmark Ticker", value="^GSPC")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

st.sidebar.subheader("Simulation Settings")
mc_horizon = st.sidebar.slider("Forecast Horizon (Trading Days)", min_value=30, max_value=756, value=252, step=30)
mc_sims = st.sidebar.slider("Monte Carlo Iterations", min_value=100, max_value=5000, value=1000, step=100)
var_confidence = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99], index=0)

run_button = st.sidebar.button("🚀 Analyze Portfolio", type="primary")

if run_button:
    if not tickers:
        st.error("Please enter at least one valid ticker.")
        st.stop()

    if len(tickers) != len(weights):
        st.error(f"Number of tickers ({len(tickers)}) does not match number of weights ({len(weights)}).")
        st.stop()

    if not np.isclose(sum(weights), 1.0, atol=1e-3):
        st.warning(f"Weights sum to {sum(weights):.4f}. Normalizing automatically to 1.0.")
        weights = [w / sum(weights) for w in weights]

    with st.spinner("Downloading historical data and computing analytics..."):
        try:
            prices = fetch_stock_data(tickers, start=str(start_date), end=str(end_date))
            market_prices = fetch_market_data(ticker=benchmark_ticker, start=str(start_date), end=str(end_date))

            result = analyze_portfolio(
                tickers=tickers,
                weights=weights,
                prices=prices,
                market_prices=market_prices,
                min_history=100,
                mc_time_horizon=mc_horizon,
                mc_simulations=mc_sims,
                random_state=42,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    # Top KPI Metrics Cards
    st.subheader("Key Portfolio Risk & Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Expected Return (Ann.)", f"{result.portfolio_return:.2%}")
    col2.metric("Volatility (Ann.)", f"{result.portfolio_volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    col4.metric("Portfolio Beta", f"{result.portfolio_beta:.2f}")
    col5.metric("CAPM Expected Return", f"{result.capm_expected_return:.2%}")

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    var_key = f"historical_var_{int(var_confidence * 100)}"
    cvar_key = f"historical_cvar_{int(var_confidence * 100)}"
    col_v1.metric(f"Historical VaR ({int(var_confidence * 100)}%)", f"{result.var_metrics.get(var_key, 0.0):.2%}")
    col_v2.metric(f"Historical CVaR ({int(var_confidence * 100)}%)", f"{result.var_metrics.get(cvar_key, 0.0):.2%}")
    p_var_key = f"parametric_var_{int(var_confidence * 100)}"
    p_cvar_key = f"parametric_cvar_{int(var_confidence * 100)}"
    col_v3.metric(f"Parametric VaR ({int(var_confidence * 100)}%)", f"{result.var_metrics.get(p_var_key, 0.0):.2%}")
    col_v4.metric(f"Parametric CVaR ({int(var_confidence * 100)}%)", f"{result.var_metrics.get(p_cvar_key, 0.0):.2%}")

    # Tabs for Visuals
    tab_mc, tab_var, tab_markowitz, tab_holdings = st.tabs(
        ["Monte Carlo Forecast", "Value at Risk & Backtest", "Markowitz Efficient Frontier", "Holdings Breakdown"]
    )

    with tab_mc:
        st.subheader("Correlated Monte Carlo Fan Chart")
        st.markdown(
            "Forecast simulated with **Cholesky decomposition** of historical covariance, normalized by initial prices so weights model capital."
        )

        mc = result.monte_carlo_summary
        scenarios = mc["scenarios"]

        # Fan Chart
        log_rets = calculate_log_returns(prices[tickers].dropna()).dropna()
        s0 = prices[tickers].dropna().iloc[-1].values
        daily_cov = log_rets.cov().values
        drift = log_rets.mean().values - 0.5 * log_rets.var().values

        sim_prices = simulate_gbm_prices(
            initial_prices=s0,
            drift=drift,
            time_horizon=mc_horizon,
            num_simulations=mc_sims,
            covariance_matrix=daily_cov,
            random_state=42,
        )
        port_paths = simulate_portfolio_paths(sim_prices, weights=weights, initial_portfolio_value=10000.0)

        days = np.arange(port_paths.shape[0])
        p5 = np.percentile(port_paths, 5, axis=1)
        p25 = np.percentile(port_paths, 25, axis=1)
        p50 = np.percentile(port_paths, 50, axis=1)
        p75 = np.percentile(port_paths, 75, axis=1)
        p95 = np.percentile(port_paths, 95, axis=1)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.fill_between(days, p5, p95, color="#3498db", alpha=0.2, label="90% Confidence Interval (5th - 95th %)")
        ax.fill_between(days, p25, p75, color="#2980b9", alpha=0.35, label="50% Confidence Interval (25th - 75th %)")
        ax.plot(days, p50, color="#1a5276", linewidth=2.5, label="Median Path")
        ax.axhline(10000.0, color="gray", linestyle=":", label="Initial Capital ($10,000)")

        ax.set_title(f"{mc_horizon}-Day Portfolio Growth Forecast ($10k starting capital)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trading Days Ahead")
        ax.set_ylabel("Portfolio Capital ($)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Scenario Terminal Values")
        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Bull Market (Top 10%)", f"${scenarios['average_terminal_value_bull'] * 10000:,.2f}")
        s_col2.metric("Base Case (Middle 80%)", f"${scenarios['average_terminal_value_base'] * 10000:,.2f}")
        s_col3.metric("Bear Market (Bottom 10%)", f"${scenarios['average_terminal_value_bear'] * 10000:,.2f}")

    with tab_var:
        st.subheader("Value at Risk (VaR) Analysis & Empirical Backtest")
        daily_portfolio_returns = pd.Series(np.dot(log_rets.values, weights), index=log_rets.index)

        fig_var, ax_var = plt.subplots(figsize=(9, 4), dpi=120)
        ax_var.hist(daily_portfolio_returns, bins=60, color="#2c3e50", alpha=0.7, edgecolor="black")
        h_var_val = result.var_metrics.get(var_key, 0.0)
        h_cvar_val = result.var_metrics.get(cvar_key, 0.0)
        ax_var.axvline(
            -h_var_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Historical VaR ({int(var_confidence * 100)}%): {h_var_val:.2%}",
        )
        ax_var.axvline(
            -h_cvar_val,
            color="red",
            linestyle="-.",
            linewidth=2,
            label=f"Historical CVaR ({int(var_confidence * 100)}%): {h_cvar_val:.2%}",
        )

        ax_var.set_title("Historical Daily Return Distribution & Tail Risk", fontsize=11, fontweight="bold")
        ax_var.set_xlabel("Daily Log Return")
        ax_var.set_ylabel("Frequency")
        ax_var.legend()
        ax_var.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_var)
        plt.close(fig_var)

        # Backtest Report
        if len(daily_portfolio_returns) > 252:
            st.markdown("#### Out-of-Sample VaR Backtest (1-Year Rolling Window)")
            bt = backtest_var(daily_portfolio_returns, window=252, confidence=var_confidence, method="historical")
            bt_param = backtest_var(daily_portfolio_returns, window=252, confidence=var_confidence, method="parametric")

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Tested Days", bt["total_observations"])
            b_col2.metric("Historical Breaches", f"{bt['breaches']} ({bt['breach_rate']:.2%})")
            b_col3.metric("Parametric Breaches", f"{bt_param['breaches']} ({bt_param['breach_rate']:.2%})")
            b_col4.metric("Kupiec Test p-value", f"{bt['kupiec_p_value']:.4f}")

            st.caption(
                f"Expected breach rate at {int(var_confidence * 100)}% confidence is {1.0 - var_confidence:.1%}. "
                "Parametric Gaussian VaR typically understates tail risk due to leptokurtosis (fat tails)."
            )

    with tab_markowitz:
        st.subheader("Markowitz Efficient Frontier")
        annual_rets = calculate_annualized_return(log_rets).values
        annual_c = calculate_annual_covariance_of_stocks(log_rets).values

        sim = simulate_random_portfolios(
            annual_rets,
            annual_c,
            risk_free_rate=result.risk_free_rate,
            num_simulations=3000,
            random_state=42,
        )
        eff_v, eff_r = efficient_frontier(sim["portfolio_returns"], sim["portfolio_volatilities"])

        fig_m, ax_m = plt.subplots(figsize=(9, 5), dpi=120)
        sc = ax_m.scatter(
            sim["portfolio_volatilities"],
            sim["portfolio_returns"],
            c=sim["portfolio_sharpes"],
            cmap="viridis",
            s=10,
            alpha=0.5,
        )
        plt.colorbar(sc, ax=ax_m, label="Sharpe Ratio")
        ax_m.plot(eff_v, eff_r, color="crimson", linewidth=2.5, label="Efficient Frontier")

        # Plot current portfolio
        ax_m.scatter(
            result.portfolio_volatility,
            result.portfolio_return,
            color="red",
            s=150,
            marker="*",
            edgecolors="black",
            zorder=6,
            label="Current Portfolio",
        )

        ax_m.set_title("Markowitz Risk-Return Profile", fontsize=12, fontweight="bold")
        ax_m.set_xlabel("Annualized Volatility")
        ax_m.set_ylabel("Annualized Expected Return")
        ax_m.legend(loc="upper left")
        ax_m.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_m)
        plt.close(fig_m)

    with tab_holdings:
        st.subheader("Individual Asset Breakdown")
        df_holdings = pd.DataFrame(
            {
                "Weight": [f"{w:.2%}" for w in weights],
                "Annualized Return": [f"{result.individual_returns[t]:.2%}" for t in tickers],
                "CAPM Beta": [f"{result.individual_betas[t]:.2f}" for t in tickers],
            },
            index=tickers,
        )
        st.dataframe(df_holdings, use_container_width=True)

else:
    st.info("👈 Enter your tickers and allocation in the sidebar and click **Analyze Portfolio** to run.")
