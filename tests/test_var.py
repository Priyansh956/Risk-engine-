import numpy as np
import pandas as pd
import pytest

from risk_engine.var import (
    backtest_var,
    calculate_historical_cvar,
    calculate_historical_var,
    calculate_parametric_cvar,
    calculate_parametric_var,
    calculate_portfolio_var_cvar,
)


@pytest.fixture
def mock_normal_returns():
    rng = np.random.default_rng(42)
    # 1000 days of normal returns: mean 0.05%, std 1.5%
    return pd.Series(rng.normal(0.0005, 0.015, 1000))


def test_calculate_historical_var_and_cvar(mock_normal_returns):
    var_95 = calculate_historical_var(mock_normal_returns, confidence=0.95)
    cvar_95 = calculate_historical_cvar(mock_normal_returns, confidence=0.95)

    assert var_95 > 0
    # Expected shortfall / CVaR must be strictly greater than or equal to VaR
    assert cvar_95 >= var_95

    # Check 99% confidence is strictly higher risk than 95%
    var_99 = calculate_historical_var(mock_normal_returns, confidence=0.99)
    cvar_99 = calculate_historical_cvar(mock_normal_returns, confidence=0.99)
    assert var_99 > var_95
    assert cvar_99 > cvar_95


def test_calculate_parametric_var_and_cvar(mock_normal_returns):
    p_var_95 = calculate_parametric_var(mock_normal_returns, confidence=0.95)
    p_cvar_95 = calculate_parametric_cvar(mock_normal_returns, confidence=0.95)

    assert p_var_95 > 0
    assert p_cvar_95 > p_var_95

    # Theoretical Gaussian 95% VaR: -(0.0005 - 1.64485 * 0.015) ≈ 0.02417
    # Empirical should be very close
    assert 0.02 < p_var_95 < 0.03

    # Modified VaR with Cornish-Fisher
    mod_var_95 = calculate_parametric_var(mock_normal_returns, confidence=0.95, modified=True)
    assert mod_var_95 > 0


def test_calculate_portfolio_var_cvar(mock_normal_returns):
    metrics = calculate_portfolio_var_cvar(mock_normal_returns, [0.95, 0.99])
    assert "historical_var_95" in metrics
    assert "historical_cvar_95" in metrics
    assert "parametric_var_95" in metrics
    assert "parametric_cvar_95" in metrics
    assert "modified_var_95" in metrics
    assert "historical_var_99" in metrics
    assert "historical_cvar_99" in metrics


def test_backtest_var(mock_normal_returns):
    backtest = backtest_var(mock_normal_returns, window=252, confidence=0.95, method="historical")

    assert backtest["total_observations"] == 1000 - 252
    assert backtest["breaches"] >= 0
    assert 0.0 <= backtest["breach_rate"] <= 1.0
    assert backtest["expected_breach_rate"] == pytest.approx(0.05)
    # For 95% VaR on normal data, breach rate should be around 3% - 7%
    assert 0.02 <= backtest["breach_rate"] <= 0.08
    assert "kupiec_p_value" in backtest
    assert 0.0 <= backtest["kupiec_p_value"] <= 1.0


def test_invalid_var_inputs():
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        calculate_historical_var(pd.Series([0.01, 0.02]), confidence=1.5)

    with pytest.raises(ValueError, match="At least 2 observations"):
        calculate_historical_var(pd.Series([0.01]))

    with pytest.raises(ValueError, match="strictly greater than rolling window"):
        backtest_var(pd.Series(np.random.normal(0, 1, 100)), window=150)
