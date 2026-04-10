import pandas as pd
import numpy as np

from src.data.pricing import resample_prices, clip_extreme_returns


def _daily_prices(n_days: int = 60, tickers=("A", "B")) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = {t: np.linspace(100, 110, n_days) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_resample_to_month_end():
    prices = _daily_prices(n_days=90)
    monthly = resample_prices(prices, frequency="M")
    # Each row should be the last trading day of the month
    assert len(monthly) < len(prices)
    assert monthly.index.freqstr in ("ME", "M")


def test_resample_preserves_tickers():
    prices = _daily_prices(tickers=("X", "Y", "Z"))
    monthly = resample_prices(prices, frequency="M")
    assert list(monthly.columns) == ["X", "Y", "Z"]


def test_resample_last_value():
    # Last day of Jan 2020 is 2020-01-31
    idx = pd.date_range("2020-01-01", "2020-01-31", freq="D")
    prices = pd.DataFrame({"A": range(len(idx))}, index=idx)
    monthly = resample_prices(prices, frequency="M")
    assert monthly["A"].iloc[0] == len(idx) - 1


def test_resample_quarterly():
    prices = _daily_prices(n_days=365)
    quarterly = resample_prices(prices, frequency="Q")
    assert len(quarterly) <= 4


def test_clip_extreme_returns_removes_spike():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    # Simulate HOME.L-style alternating price error: 100 → 1 → 100 → 1
    prices = pd.DataFrame({"A": [100.0, 100.0, 1.0, 1.0, 100.0, 1.0]}, index=idx)
    cleaned = clip_extreme_returns(prices, threshold=5.0)
    returns = cleaned["A"].pct_change().dropna()
    assert (returns.abs() <= 5.0).all(), f"Extreme return not clipped: {returns.tolist()}"


def test_clip_extreme_returns_preserves_normal_moves():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    prices = pd.DataFrame({"A": [100.0, 105.0, 98.0, 110.0, 107.0, 115.0]}, index=idx)
    cleaned = clip_extreme_returns(prices, threshold=5.0)
    pd.testing.assert_frame_equal(cleaned, prices)


def test_clip_extreme_returns_no_change_when_clean():
    prices = _daily_prices(n_days=90)
    monthly = resample_prices(prices)
    assert clip_extreme_returns(monthly, threshold=5.0).equals(monthly)
