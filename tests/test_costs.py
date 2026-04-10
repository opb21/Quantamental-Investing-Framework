import pandas as pd
import numpy as np

from src.backtest.costs import turnover, apply_costs


def _weights(data: list[list[float]], tickers=("A", "B")) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=len(data), freq="ME")
    return pd.DataFrame(data, index=idx, columns=list(tickers))


def test_turnover_zero_when_weights_unchanged():
    w = _weights([[0.5, 0.5]] * 4)
    to = turnover(w)
    # First period is NaN (diff of first row), rest should be 0
    assert to.iloc[1:].eq(0.0).all()


def test_turnover_full_rotation():
    # Flip entirely from A to B each period
    w = _weights([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    to = turnover(w)
    # Each flip: |1-0| + |0-1| = 2, one-way = 1.0
    assert np.allclose(to.iloc[1:], 1.0)


def test_apply_costs_zero_when_no_bps():
    returns = pd.Series([0.01, 0.02, -0.01])
    to = pd.Series([0.1, 0.2, 0.0])
    net = apply_costs(returns, to, commission_bps=0, slippage_bps=0)
    assert np.allclose(net, returns)


def test_apply_costs_reduces_returns():
    returns = pd.Series([0.05, 0.05])
    to = pd.Series([1.0, 1.0])
    net = apply_costs(returns, to, commission_bps=10, slippage_bps=10)
    assert (net < returns).all()


def test_apply_costs_known_value():
    # return=0.01, turnover=1.0, total_bps=20 => cost=0.002, net=0.008
    returns = pd.Series([0.01])
    to = pd.Series([1.0])
    net = apply_costs(returns, to, commission_bps=10, slippage_bps=10)
    assert abs(net.iloc[0] - 0.008) < 1e-9
