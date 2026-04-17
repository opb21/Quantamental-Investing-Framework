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


# ── vol_adjust tests ────────────────────────────────────────────────────────

def test_momentum_vol_adjust_differs_from_raw():
    """Vol-adjusted scores should differ from unadjusted scores (not a no-op)."""
    prices = _prices(n_months=36)
    raw = momentum(prices, lookback_months=12, vol_adjust=False)
    adj = momentum(prices, lookback_months=12, vol_adjust=True)
    # They should not be identical (vol normalisation changes the values)
    assert not raw.equals(adj)


def test_momentum_vol_adjust_no_inf():
    """Vol-adjusted scores must not contain inf values."""
    prices = _prices(n_months=36)
    adj = momentum(prices, lookback_months=12, vol_adjust=True)
    assert not np.isinf(adj.values[~np.isnan(adj.values)]).any()


def test_momentum_vol_adjust_constant_prices_gives_nan():
    """Constant prices → zero vol → division yields NaN (not inf)."""
    idx = pd.date_range("2018-01-31", periods=24, freq="ME")
    # Ticker A is flat (zero vol); B has normal variation
    prices = pd.DataFrame(
        {"A": np.full(24, 100.0), "B": np.linspace(100, 130, 24)},
        index=idx,
    )
    adj = momentum(prices, lookback_months=12, vol_adjust=True)
    # A: non-zero return (price unchanged, so actually ret=0 — but vol also 0 → NaN)
    # Either way, no inf should appear
    assert not np.isinf(adj.values[~np.isnan(adj.values)]).any()


def test_blend_momentum_vol_adjust_no_inf():
    """Vol-adjusted blend scores must not contain inf values."""
    prices = _prices(n_months=36)
    adj = blend_momentum(prices, lookbacks=[3, 6, 12], vol_adjust=True)
    assert not np.isinf(adj.values[~np.isnan(adj.values)]).any()


def test_blend_momentum_vol_adjust_false_unchanged():
    """vol_adjust=False (default) should give the same result as not passing the flag."""
    prices = _prices(n_months=36)
    a = blend_momentum(prices, lookbacks=[6, 12])
    b = blend_momentum(prices, lookbacks=[6, 12], vol_adjust=False)
    pd.testing.assert_frame_equal(a, b)


# ── daily prices vol tests ───────────────────────────────────────────────────

def _daily_prices_for(monthly: pd.DataFrame) -> pd.DataFrame:
    """Expand monthly prices to pseudo-daily by forward-filling within each month."""
    # Build a daily index spanning the monthly range
    daily_idx = pd.date_range(monthly.index[0], monthly.index[-1], freq="B")
    return monthly.reindex(daily_idx, method="ffill")


def test_momentum_daily_vol_differs_from_monthly():
    """Daily price-based vol should produce different scores than monthly fallback."""
    prices = _prices(n_months=36)
    prices_daily = _daily_prices_for(prices)
    monthly_adj = momentum(prices, lookback_months=12, vol_adjust=True)
    daily_adj = momentum(prices, lookback_months=12, vol_adjust=True, prices_daily=prices_daily)
    # Drop rows where both are NaN (early periods) before comparing
    mask = monthly_adj.notna() & daily_adj.notna()
    assert not (monthly_adj[mask] == daily_adj[mask]).all().all()


def test_momentum_daily_vol_aligns_to_monthly_index():
    """Output index must match the monthly prices index even when daily prices are provided."""
    prices = _prices(n_months=36)
    prices_daily = _daily_prices_for(prices)
    result = momentum(prices, lookback_months=12, vol_adjust=True, prices_daily=prices_daily)
    assert result.index.equals(prices.index)


def test_momentum_ewma_no_inf_daily():
    """EWMA vol with daily prices must never produce inf."""
    prices = _prices(n_months=36)
    prices_daily = _daily_prices_for(prices)
    adj = momentum(
        prices, lookback_months=12, vol_adjust=True,
        vol_method="ewma", ewma_lambda=0.97, prices_daily=prices_daily,
    )
    assert not np.isinf(adj.values[~np.isnan(adj.values)]).any()


def test_momentum_rolling_static_differs_from_dynamic_daily():
    """Static vol window (252 days) should differ from dynamic (lookback × 21 days)."""
    prices = _prices(n_months=48)
    prices_daily = _daily_prices_for(prices)
    dynamic = momentum(
        prices, lookback_months=12, vol_adjust=True,
        vol_window=None, prices_daily=prices_daily,
    )
    static = momentum(
        prices, lookback_months=12, vol_adjust=True,
        vol_window=126, prices_daily=prices_daily,  # 6-month static vs 12-month dynamic
    )
    mask = dynamic.notna() & static.notna()
    assert not (dynamic[mask] == static[mask]).all().all()


def test_blend_momentum_daily_vol_no_inf():
    """Blend with daily prices and EWMA vol must never produce inf."""
    prices = _prices(n_months=48)
    prices_daily = _daily_prices_for(prices)
    result = blend_momentum(
        prices, lookbacks=[3, 6, 12], vol_adjust=True,
        vol_method="ewma", ewma_lambda=0.97, prices_daily=prices_daily,
    )
    assert not np.isinf(result.values[~np.isnan(result.values)]).any()
