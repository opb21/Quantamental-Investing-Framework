import pandas as pd
import numpy as np

from src.analytics.performance import calculate_performance, calculate_relative_performance, rolling_sharpe


def _flat_returns(r: float, n: int, freq: str = "ME") -> pd.Series:
    idx = pd.date_range("2015-01-31", periods=n, freq=freq)
    return pd.Series([r] * n, index=idx)


def test_cagr_known_value():
    # 1% monthly return for 12 months => CAGR ≈ 12.68%
    returns = _flat_returns(0.01, 12)
    metrics = calculate_performance(returns, freq="M")
    expected_cagr = (1.01 ** 12) - 1
    assert abs(metrics["CAGR"] - round(expected_cagr, 4)) < 1e-3


def test_sharpe_positive_for_positive_returns():
    returns = _flat_returns(0.01, 36)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Sharpe"] > 0


def test_max_drawdown_is_negative_or_zero():
    returns = _flat_returns(0.01, 24)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["MaxDrawdown"] <= 0.0


def test_max_drawdown_known_loss():
    # Single -50% month after gains => drawdown >= 0.5
    returns = pd.Series([0.1, 0.1, -0.5, 0.1])
    metrics = calculate_performance(returns, freq="M")
    assert metrics["MaxDrawdown"] <= -0.4


def test_output_keys():
    returns = _flat_returns(0.005, 12)
    metrics = calculate_performance(returns, freq="M")
    assert set(metrics.keys()) == {
        "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar",
        "MaxDrawdown", "HitRate", "AvgWin", "AvgLoss",
    }


def test_zero_std_returns_zero_sharpe():
    # Constant returns have zero std => Sharpe should be 0, not error
    returns = _flat_returns(0.0, 12)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Sharpe"] == 0


def test_volatility_positive():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.standard_normal(36) * 0.03)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Volatility"] > 0


def test_sortino_gte_sharpe_for_mixed_returns():
    # Sortino only penalises downside — for a series with some losses,
    # Sortino >= Sharpe when mean return is positive
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(60) * 0.04 + 0.01)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Sortino"] >= metrics["Sharpe"]


def test_calmar_positive_for_positive_cagr():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.standard_normal(60) * 0.03 + 0.015)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Calmar"] > 0


def test_hit_rate_bounds():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_normal(60) * 0.03)
    metrics = calculate_performance(returns, freq="M")
    assert 0.0 <= metrics["HitRate"] <= 1.0


def test_avg_win_positive_avg_loss_negative():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.standard_normal(60) * 0.03)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["AvgWin"] > 0
    assert metrics["AvgLoss"] < 0


def test_sortino_zero_when_no_downside():
    # All positive returns => no downside std => Sortino returns 0 (not error)
    returns = _flat_returns(0.01, 24)
    metrics = calculate_performance(returns, freq="M")
    assert metrics["Sortino"] == 0.0


# --- rolling_sharpe ---

def test_rolling_sharpe_returns_series():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.standard_normal(36) * 0.03 + 0.01)
    result = rolling_sharpe(returns, window=12, freq="M")
    assert isinstance(result, pd.Series)
    assert len(result) == len(returns)


def test_rolling_sharpe_first_window_minus_one_is_nan():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.standard_normal(36) * 0.03 + 0.01)
    result = rolling_sharpe(returns, window=12, freq="M")
    # First 11 values (window-1) must be NaN — insufficient history
    assert result.iloc[:11].isna().all()
    assert result.iloc[11:].notna().all()


# --- calculate_relative_performance ---

def test_relative_performance_output_keys():
    port = _flat_returns(0.01, 36)
    bench = _flat_returns(0.007, 36)
    result = calculate_relative_performance(port, bench, freq="M")
    assert set(result.keys()) == {"BenchmarkCAGR", "ExcessCAGR", "Beta", "TrackingError", "InformationRatio"}


def test_relative_excess_cagr_positive_when_outperforming():
    port = _flat_returns(0.01, 36)
    bench = _flat_returns(0.005, 36)
    result = calculate_relative_performance(port, bench, freq="M")
    assert result["ExcessCAGR"] > 0


def test_relative_excess_cagr_negative_when_underperforming():
    port = _flat_returns(0.005, 36)
    bench = _flat_returns(0.01, 36)
    result = calculate_relative_performance(port, bench, freq="M")
    assert result["ExcessCAGR"] < 0


def test_relative_beta_one_for_identical_series():
    returns = _flat_returns(0.01, 36)
    result = calculate_relative_performance(returns, returns, freq="M")
    assert abs(result["Beta"] - 1.0) < 1e-6


def test_relative_tracking_error_zero_for_identical_series():
    returns = _flat_returns(0.01, 36)
    result = calculate_relative_performance(returns, returns, freq="M")
    assert result["TrackingError"] == 0.0


def test_relative_handles_misaligned_index():
    # Portfolio starts one month later — should align on overlap without error
    idx_port = pd.date_range("2020-02-29", periods=24, freq="ME")
    idx_bench = pd.date_range("2020-01-31", periods=24, freq="ME")
    port = pd.Series([0.01] * 24, index=idx_port)
    bench = pd.Series([0.007] * 24, index=idx_bench)
    result = calculate_relative_performance(port, bench, freq="M")
    assert "ExcessCAGR" in result
