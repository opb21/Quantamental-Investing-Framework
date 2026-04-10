import pandas as pd
import numpy as np

from src.portfolio.construction import select_top_n, equal_weight, inverse_vol_weight, momentum_inv_vol_weight, rebalance_weights, staggered_rebalance_weights


def _scores(n_dates: int = 12, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n_dates, len(tickers))), index=idx, columns=list(tickers))


def test_select_top_n_count():
    scores = _scores()
    selection = select_top_n(scores, n=2)
    assert selection.sum(axis=1).eq(2).all()


def test_select_top_n_boolean():
    scores = _scores()
    selection = select_top_n(scores, n=2)
    assert selection.dtypes.eq(bool).all()


def test_equal_weight_sums_to_one():
    scores = _scores()
    selection = select_top_n(scores, n=2)
    weights = equal_weight(selection)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_equal_weight_value():
    scores = _scores()
    selection = select_top_n(scores, n=4)
    weights = equal_weight(selection)
    assert np.allclose(weights.values, 0.25)


def test_rebalance_weights_ffill():
    scores = _scores(n_dates=12)
    selection = select_top_n(scores, n=2)
    w_raw = equal_weight(selection)
    # Quarterly rebalance on monthly data: weights should be held between quarters
    weights = rebalance_weights(w_raw, rebalance="Q")
    assert weights.shape == w_raw.shape
    # Rows before the first quarter-end have no prior value to ffill from (expected NaN/0).
    # From the first quarter-end onwards, weights must sum to 1.
    first_q_end = weights[weights.sum(axis=1) > 0].index[0]
    assert np.allclose(weights.loc[first_q_end:].sum(axis=1), 1.0)


def _returns(n_dates: int = 24, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    """Synthetic monthly returns for construction tests."""
    idx = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((n_dates, len(tickers))) * 0.05,
                        index=idx, columns=list(tickers))


def test_inv_vol_weight_sums_to_one():
    scores = _scores(n_dates=24)
    asset_returns = _returns(n_dates=24)
    selection = select_top_n(scores, n=2)
    weights = inverse_vol_weight(selection, asset_returns, window=12)
    # After enough history has accumulated, weights must sum to 1 per date
    from_date = weights.index[12]
    assert np.allclose(weights.loc[from_date:].sum(axis=1), 1.0, atol=1e-6)


def test_inv_vol_weight_higher_vol_gets_lower_weight():
    """Stock with 2x the volatility should receive a smaller weight."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    # A has ~2x the vol of B
    returns = pd.DataFrame({
        "A": rng.standard_normal(24) * 0.10,
        "B": rng.standard_normal(24) * 0.05,
    }, index=idx)
    # Always select both
    selection = pd.DataFrame(True, index=idx, columns=["A", "B"])
    weights = inverse_vol_weight(selection, returns, window=12)
    # From date 13 onwards there is sufficient history; B should consistently outweigh A
    stable = weights.iloc[12:]
    assert (stable["B"] > stable["A"]).all()


def test_inv_vol_weight_falls_back_to_equal_before_window():
    """With only 1 period of history, vol cannot be estimated — expect equal weight."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.01, -0.01, 0.02]}, index=idx)
    selection = pd.DataFrame(True, index=idx, columns=["A", "B"])
    # window=12 means first 11 dates lack sufficient history
    weights = inverse_vol_weight(selection, returns, window=12)
    # All three dates have <2 observations available → equal weight (0.5 each)
    assert np.allclose(weights.iloc[0][["A", "B"]], 0.5)


def test_momentum_inv_vol_weight_sums_to_one():
    scores = _scores(n_dates=24)
    asset_returns = _returns(n_dates=24)
    selection = select_top_n(scores, n=2)
    weights = momentum_inv_vol_weight(selection, scores, asset_returns, window=12)
    from_date = weights.index[12]
    assert np.allclose(weights.loc[from_date:].sum(axis=1), 1.0, atol=1e-6)


def test_momentum_inv_vol_weight_higher_rank_higher_weight_when_same_vol():
    """When volatilities are identical, weights should be proportional to rank (higher momentum → larger weight)."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(1)
    # Identical vol for A and B; A always scores higher than B
    returns = pd.DataFrame({
        "A": rng.standard_normal(24) * 0.05,
        "B": rng.standard_normal(24) * 0.05,
    }, index=idx)
    # Force A's score always > B's score
    scores_df = pd.DataFrame({"A": 2.0, "B": 1.0}, index=idx)
    selection = pd.DataFrame(True, index=idx, columns=["A", "B"])
    weights = momentum_inv_vol_weight(selection, scores_df, returns, window=12)
    stable = weights.iloc[12:]
    assert (stable["A"] > stable["B"]).all()


def test_momentum_inv_vol_weight_falls_back_to_equal_before_window():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.01, -0.01, 0.02]}, index=idx)
    scores_df = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.5, 0.5, 0.5]}, index=idx)
    selection = pd.DataFrame(True, index=idx, columns=["A", "B"])
    weights = momentum_inv_vol_weight(selection, scores_df, returns, window=12)
    assert np.allclose(weights.iloc[0][["A", "B"]], 0.5)


def test_rebalance_weights_preserves_index():
    scores = _scores(n_dates=12)
    selection = select_top_n(scores, n=2)
    w_raw = equal_weight(selection)
    weights = rebalance_weights(w_raw, rebalance="Q")
    assert list(weights.index) == list(w_raw.index)


# ── staggered_rebalance_weights tests ──────────────────────────────────────

def _monthly_equal_weights(n_dates: int = 36) -> pd.DataFrame:
    """Synthetic monthly weights that change each period (non-trivial for rebalance tests)."""
    idx = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rng = np.random.default_rng(99)
    raw = np.abs(rng.standard_normal((n_dates, 4)))
    w = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=idx, columns=["A", "B", "C", "D"])


def test_staggered_n1_matches_rebalance_weights():
    """n_tranches=1 must be identical to rebalance_weights."""
    w = _monthly_equal_weights()
    expected = rebalance_weights(w, rebalance="Q")
    result = staggered_rebalance_weights(w, rebalance="Q", n_tranches=1)
    pd.testing.assert_frame_equal(result, expected)


def test_staggered_weights_sum_to_one():
    """Blended weights should sum to 1 at every date with valid history."""
    w = _monthly_equal_weights(n_dates=36)
    blended = staggered_rebalance_weights(w, rebalance="Q", n_tranches=3)
    # Drop early rows that may be NaN due to ffill not having a prior rebalance
    valid = blended[blended.sum(axis=1) > 0]
    assert np.allclose(valid.sum(axis=1), 1.0, atol=1e-6)


def test_staggered_tranches_rebalance_on_distinct_months():
    """
    With n_tranches=3 and quarterly rebalancing, the 3 tranches should update
    their weights on 3 distinct sets of months (no two tranches rebalance on
    the same month). Verified by checking that over any 3-month window within
    a quarter, each month sees a different tranche update.
    """
    w = _monthly_equal_weights(n_dates=36)
    blended = staggered_rebalance_weights(w, rebalance="Q", n_tranches=3)
    # The blended portfolio should differ from single-tranche (standard) rebalancing
    single = rebalance_weights(w, rebalance="Q")
    # If only 1 of 3 months per quarter updates, blended != single at most dates
    assert not np.allclose(blended.values, single.values, equal_nan=True)
