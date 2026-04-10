import pandas as pd
import numpy as np

from src.backtest.engine import backtest_long_only


def _make_inputs(n_periods: int = 12, tickers=("A", "B")):
    idx = pd.date_range("2020-01-31", periods=n_periods, freq="ME")
    prices = pd.DataFrame(
        {t: np.linspace(100, 120, n_periods) for t in tickers},
        index=idx,
    )
    weights = pd.DataFrame(
        {t: 1.0 / len(tickers) for t in tickers},
        index=idx,
    )
    return prices, weights


def test_output_columns():
    prices, weights = _make_inputs()
    result = backtest_long_only(prices, weights)
    expected = {"portfolio_returns", "costs", "portfolio_returns_net", "cumulative_net", "turnover"}
    assert set(result.columns) == expected


def test_output_length():
    prices, weights = _make_inputs(n_periods=12)
    result = backtest_long_only(prices, weights)
    assert len(result) == 12


def test_lagged_weights_no_lookahead():
    """
    Core research integrity test: portfolio returns must use weights lagged by
    one period (weights.shift(1)), not contemporaneous weights.

    We verify this by constructing a case where contemporaneous and lagged weights
    produce different results, then asserting the output matches the lagged version.
    """
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    prices = pd.DataFrame({"A": [100.0, 110.0, 100.0, 110.0]}, index=idx)
    # Weight flips each period — so contemporaneous vs lagged will differ
    weights = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0]}, index=idx)

    result = backtest_long_only(prices, weights, commission_bps=0, slippage_bps=0)

    asset_returns = prices.pct_change()
    # Lagged: position set at end of T earns return at T+1
    expected_gross = (asset_returns * weights.shift(1)).sum(axis=1)

    assert np.allclose(result["portfolio_returns"], expected_gross, equal_nan=True)


def test_net_returns_lte_gross():
    prices, weights = _make_inputs()
    result = backtest_long_only(prices, weights, commission_bps=5, slippage_bps=5)
    # Net must be <= gross everywhere (costs are non-negative for positive turnover)
    assert (result["portfolio_returns_net"] <= result["portfolio_returns"] + 1e-12).all()


def test_zero_costs_gross_equals_net():
    prices, weights = _make_inputs()
    result = backtest_long_only(prices, weights, commission_bps=0, slippage_bps=0)
    assert np.allclose(result["portfolio_returns"], result["portfolio_returns_net"])


def test_cumulative_starts_near_one():
    prices, weights = _make_inputs()
    result = backtest_long_only(prices, weights)
    # First non-NaN cumulative value should be close to 1 + first net return
    first_valid = result["cumulative_net"].dropna().iloc[0]
    assert 0.5 < first_valid < 2.0
