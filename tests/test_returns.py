import numpy as np
import pandas as pd

from risk_engine.returns import (
    log_returns,
    annualized_return,
)


def test_log_returns():

    prices = pd.Series([
        100,
        110,
        121,
    ])

    result = log_returns(prices)

    expected = pd.Series([
        np.nan,
        np.log(110 / 100),
        np.log(121 / 110),
    ])

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_annualized_return():

    daily_log_returns = pd.Series(
        [0.001] * 252
    )

    result = annualized_return(
        daily_log_returns
    )

    expected = np.exp(
        0.001 * 252
    ) - 1

    assert np.isclose(
        result,
        expected,
    )