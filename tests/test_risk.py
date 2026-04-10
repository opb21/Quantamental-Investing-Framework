import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import drawdown_series, rolling_vol, avg_pairwise_correlation, contribution_to_vol


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_returns(n: int = 36, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(0.01, 0.04, size=n), index=idx, name="portfolio")


def _make_asset_returns(n: int = 36, tickers=("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    data = rng.normal(0.01, 0.04, size=(n, len(tickers)))
    return pd.DataFrame(data, index=idx, columns=list(tickers))


def _make_equal_weights(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight portfolio holding all assets throughout."""
    n = len(asset_returns.columns)
    w = pd.DataFrame(1.0 / n, index=asset_returns.index, columns=asset_returns.columns)
    return w


# ── drawdown_series ───────────────────────────────────────────────────────────

def test_drawdown_series_output_name():
    s = drawdown_series(_make_returns())
    assert s.name == "drawdown"


def test_drawdown_series_always_non_positive():
    s = drawdown_series(_make_returns())
    assert (s.dropna() <= 0).all()


def test_drawdown_series_zero_at_new_high():
    """First period is always at a high-water mark → drawdown = 0."""
    s = drawdown_series(_make_returns())
    assert s.iloc[0] == pytest.approx(0.0)


def test_drawdown_series_known_value():
    """After a -20% drop from peak, drawdown should be -0.20."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    # period 0: +0% (base), period 1: +0%, period 2: -20%
    rets = pd.Series([0.0, 0.0, -0.20], index=idx)
    s = drawdown_series(rets)
    assert s.iloc[2] == pytest.approx(-0.20)


def test_drawdown_series_recovers_to_zero():
    """After a drawdown, recovery to a new high resets drawdown to 0."""
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    rets = pd.Series([0.0, -0.10, 0.20, 0.0], index=idx)
    s = drawdown_series(rets)
    # cumulative after period 3: (1)(0.9)(1.08)(1.08) — check period 2 is new high
    assert s.iloc[2] == pytest.approx(0.0, abs=1e-10)


def test_drawdown_series_preserves_index():
    returns = _make_returns(24)
    s = drawdown_series(returns)
    assert s.index.equals(returns.index)


# ── rolling_vol ───────────────────────────────────────────────────────────────

def test_rolling_vol_output_name():
    s = rolling_vol(_make_returns(), window=12, freq="M")
    assert s.name == "rolling_vol"


def test_rolling_vol_nan_before_window():
    s = rolling_vol(_make_returns(36), window=12, freq="M")
    assert s.iloc[:11].isna().all()
    assert s.iloc[11:].notna().all()


def test_rolling_vol_annualised():
    """A constant-return series has zero vol; a volatile series should be > 0."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    flat = pd.Series([0.01] * 24, index=idx)
    s = rolling_vol(flat, window=12, freq="M")
    assert (s.dropna() == 0.0).all()

    volatile = _make_returns(24)
    s2 = rolling_vol(volatile, window=12, freq="M")
    assert (s2.dropna() > 0).all()


def test_rolling_vol_scaling():
    """Daily vol scaled by sqrt(252) should differ from monthly scaled by sqrt(12)."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    rets = pd.Series(rng.normal(0.01, 0.04, 36), index=idx)
    s_monthly = rolling_vol(rets, window=12, freq="M")
    s_daily = rolling_vol(rets, window=12, freq="D")
    # sqrt(252) > sqrt(12), so daily-scaled vol should be larger
    assert (s_daily.dropna() > s_monthly.dropna()).all()


# ── avg_pairwise_correlation ──────────────────────────────────────────────────

def test_avg_pairwise_corr_output_name():
    ar = _make_asset_returns()
    w = _make_equal_weights(ar)
    s = avg_pairwise_correlation(w, ar, window=12)
    assert s.name == "avg_pairwise_corr"


def test_avg_pairwise_corr_nan_before_window():
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    s = avg_pairwise_correlation(w, ar, window=12)
    assert s.iloc[:11].isna().all()
    assert s.iloc[11:].notna().all()


def test_avg_pairwise_corr_range():
    """Correlation values must lie in [-1, 1]."""
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    s = avg_pairwise_correlation(w, ar, window=12).dropna()
    assert (s >= -1.0).all() and (s <= 1.0).all()


def test_avg_pairwise_corr_single_asset_returns_nan():
    """When only one asset is held, correlation is undefined → NaN."""
    ar = _make_asset_returns(24, tickers=("A", "B"))
    # hold only A
    w = pd.DataFrame(0.0, index=ar.index, columns=ar.columns)
    w["A"] = 1.0
    s = avg_pairwise_correlation(w, ar, window=12)
    assert s.dropna().empty


def test_avg_pairwise_corr_perfectly_correlated():
    """Two identical return series → avg pairwise correlation = 1.0."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rets = np.random.default_rng(7).normal(0.01, 0.04, 24)
    ar = pd.DataFrame({"A": rets, "B": rets}, index=idx)
    w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    s = avg_pairwise_correlation(w, ar, window=12).dropna()
    assert np.allclose(s.values, 1.0, atol=1e-10)


# ── contribution_to_vol ───────────────────────────────────────────────────────

def test_contrib_to_vol_shape():
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    df = contribution_to_vol(w, ar, window=12, freq="M")
    assert df.shape == w.shape


def test_contrib_to_vol_sums_to_one():
    """Active contributions must sum to 1.0 at each period (after window)."""
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    df = contribution_to_vol(w, ar, window=12, freq="M")
    row_sums = df.iloc[11:].sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-10)


def test_contrib_to_vol_non_negative_for_positive_cov():
    """With positive weights and non-degenerate cov, all contributions >= 0."""
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    df = contribution_to_vol(w, ar, window=12, freq="M").iloc[11:]
    assert (df.values >= -1e-12).all()


def test_contrib_to_vol_nan_before_window():
    ar = _make_asset_returns(36)
    w = _make_equal_weights(ar)
    df = contribution_to_vol(w, ar, window=12, freq="M")
    assert df.iloc[:11].isna().all(axis=None)
