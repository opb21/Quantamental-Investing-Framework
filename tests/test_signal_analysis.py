import numpy as np
import pandas as pd
import pytest

from src.analytics.signal_analysis import (
    information_coefficient,
    ic_decay,
    ic_summary,
)


def _monthly(n: int = 36, tickers=("A", "B", "C", "D", "E")) -> tuple:
    """Synthetic monthly prices and derived returns for tests."""
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        100 * (1 + rng.standard_normal((n, len(tickers))) * 0.05).cumprod(axis=0),
        index=idx,
        columns=list(tickers),
    )
    returns = prices.pct_change()
    return prices, returns


def test_ic_shape():
    """IC series should have at most len(scores)-1 valid values (last date has no forward return)."""
    _, returns = _monthly()
    scores = returns.shift(1)  # lagged returns as a simple signal proxy
    ic = information_coefficient(scores, returns)
    assert len(ic) <= len(scores) - 1
    assert ic.name == "ic"


def test_ic_perfect_signal():
    """When scores equal forward returns, IC should be 1.0 at every date."""
    _, returns = _monthly()
    forward = returns.shift(-1)
    # Scores = forward returns (perfect predictor)
    ic = information_coefficient(forward.shift(1), returns)
    # Drop the first few NaN-heavy dates; stable dates should be near 1.0
    stable = ic.dropna().iloc[2:]
    assert (stable > 0.99).all(), f"Expected near-perfect IC, got min={stable.min():.3f}"


def test_ic_uncorrelated_signal():
    """A random signal uncorrelated with returns should produce mean IC near zero."""
    _, returns = _monthly(n=120)
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.standard_normal(returns.shape),
        index=returns.index,
        columns=returns.columns,
    )
    ic = information_coefficient(scores, returns)
    # With 120 periods, mean IC of a noise signal should be well within ±0.1
    assert abs(ic.mean()) < 0.10, f"Expected near-zero mean IC, got {ic.mean():.4f}"


def test_ic_decay_length():
    """ic_decay should return exactly max_horizon entries."""
    _, returns = _monthly()
    scores = returns.shift(1)
    decay = ic_decay(scores, returns, max_horizon=6)
    assert len(decay) == 6
    assert list(decay.index) == list(range(1, 7))
    assert decay.name == "ic_decay"


def test_ic_decay_h1_consistent_with_ic():
    """Mean IC at horizon=1 from ic_decay should match mean of information_coefficient."""
    _, returns = _monthly()
    scores = returns.shift(1)
    decay = ic_decay(scores, returns, max_horizon=1)
    ic = information_coefficient(scores, returns)
    assert abs(decay[1] - ic.mean()) < 1e-6


def test_ic_summary_keys():
    """ic_summary should return all four expected keys."""
    ic_series = pd.Series([0.05, 0.10, -0.02, 0.08, 0.03])
    result = ic_summary(ic_series)
    assert set(result.keys()) == {"MeanIC", "ICIR", "ICHitRate", "ICTStat"}


def test_ic_summary_values():
    """ic_summary values should be arithmetically consistent."""
    ic_series = pd.Series([0.10, 0.20, 0.15, 0.05])
    result = ic_summary(ic_series)
    assert result["MeanIC"] == round(ic_series.mean(), 4)
    assert result["ICHitRate"] == 1.0  # all positive
    assert result["ICIR"] > 0


def test_ic_summary_empty_graceful():
    """ic_summary should not raise on an empty or all-NaN series."""
    result = ic_summary(pd.Series([], dtype=float))
    assert result["MeanIC"] == 0.0 or np.isnan(result["MeanIC"])
