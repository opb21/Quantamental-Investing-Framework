import pandas as pd
import numpy as np

from src.signals.momentum import momentum, blend_momentum


def _monthly_prices(n_months: int = 24, tickers=("A", "B")) -> pd.DataFrame:
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    data = {t: np.linspace(100, 130, n_months) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_momentum_output_shape():
    prices = _monthly_prices(n_months=24)
    scores = momentum(prices)
    assert scores.shape == prices.shape


def test_momentum_known_value():
    # With shift(1)/shift(12) at the last row (index 12):
    #   numerator  = prices.iloc[11] = 120
    #   denominator = prices.iloc[0]  = 100
    #   => score = 120/100 - 1 = 0.20
    idx = pd.date_range("2018-01-31", periods=13, freq="ME")
    prices = pd.DataFrame({"A": [100] + [105] * 10 + [120, 999]}, index=idx)
    scores = momentum(prices, lookback_months=12, skip_recent_months=1)
    last_score = scores["A"].iloc[-1]
    assert abs(last_score - 0.20) < 1e-9


def test_momentum_no_lookahead():
    # Score at row i must only depend on prices at row i-skip and earlier.
    # Concretely: shift(1)/shift(12) means row 0..11 should be NaN.
    prices = _monthly_prices(n_months=24)
    scores = momentum(prices, lookback_months=12, skip_recent_months=1)
    # First 12 rows cannot have a valid score (need 12 periods of history)
    assert scores.iloc[:12].isna().all().all()


def test_momentum_custom_lookback():
    prices = _monthly_prices(n_months=12)
    scores = momentum(prices, lookback_months=6, skip_recent_months=0)
    # With 0 skip and 6-month lookback, first 6 rows should be NaN
    assert scores.iloc[:6].isna().all().all()
    assert scores.iloc[6:].notna().all().all()


# ── blend_momentum tests ────────────────────────────────────────────────────

def _prices(n_months: int = 24, n_tickers: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(7)
    data = 100 * (1 + rng.standard_normal((n_months, n_tickers)) * 0.05).cumprod(axis=0)
    return pd.DataFrame(data, index=idx, columns=[f"T{i}" for i in range(n_tickers)])


def test_blend_momentum_shape():
    prices = _prices()
    scores = blend_momentum(prices, lookbacks=[3, 6, 12])
    assert scores.shape == prices.shape


def test_blend_momentum_none_weights_equals_equal_weights():
    """Passing None and explicit equal weights should produce identical output."""
    prices = _prices()
    a = blend_momentum(prices, lookbacks=[3, 6, 12])
    b = blend_momentum(prices, lookbacks=[3, 6, 12], blend_weights=[1, 1, 1])
    pd.testing.assert_frame_equal(a, b)


def test_blend_momentum_single_lookback_is_zscore():
    """Single-lookback blend should equal the cross-sectionally z-scored raw signal."""
    prices = _prices()
    raw = momentum(prices, lookback_months=6, skip_recent_months=1)
    expected = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    result = blend_momentum(prices, lookbacks=[6])
    pd.testing.assert_frame_equal(result, expected)


def test_blend_momentum_weights_normalised():
    """blend_weights that don't sum to 1 should be normalised; result equals equal weights."""
    prices = _prices()
    a = blend_momentum(prices, lookbacks=[3, 12], blend_weights=[2, 2])
    b = blend_momentum(prices, lookbacks=[3, 12])
    pd.testing.assert_frame_equal(a, b)
