import numpy as np
import pandas as pd

_FREQ_MAP = {"M": "ME", "Q": "QE", "SA": "6ME", "A": "YE", "Y": "YE"}


def select_top_n(scores: pd.DataFrame, n: int) -> pd.DataFrame:
    """Boolean selection matrix True for selected tickers at each date."""
    return scores.rank(axis=1, ascending=False, method="first") <= n


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """Convert selection matrix to weights that sum to 1 per date (or 0 if none)."""
    counts = selection.sum(axis=1)
    return selection.div(counts, axis=0).fillna(0.0)


def inverse_vol_weight(
    selection: pd.DataFrame,
    asset_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Convert selection matrix to inverse-volatility weights.

    Each selected ticker's weight is proportional to 1/σ, where σ is its
    trailing standard deviation over the last `window` observations of
    `asset_returns`. High-vol stocks receive smaller allocations.

    `asset_returns` may be at any frequency (daily or monthly). `window`
    should be expressed in the same frequency (e.g. 252 for one year of
    daily returns, 12 for one year of monthly returns). Using daily returns
    produces more stable vol estimates due to the larger sample size.

    Slicing uses date-based lookup (.loc[:date]) so that the function is
    correct regardless of whether asset_returns and selection share the
    same index frequency.

    Falls back to equal weight when there is insufficient return history or
    all selected tickers have zero/NaN volatility. No lookahead.
    """
    result = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)

    for date in selection.index:
        cols = selection.columns[selection.loc[date]].tolist()
        if not cols:
            continue

        # Slice by date: last `window` observations up to and including `date`
        ret_window = asset_returns.loc[:date, cols].iloc[-window:].dropna(axis=1)

        # Need at least 2 observations to estimate vol; fall back to equal weight
        if ret_window.shape[0] < 2 or ret_window.empty:
            result.loc[date, cols] = 1.0 / len(cols)
            continue

        vols = ret_window.std().replace(0, np.nan).dropna()

        if vols.empty:
            result.loc[date, cols] = 1.0 / len(cols)
            continue

        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        result.loc[date, weights.index] = weights.values

    return result


def momentum_inv_vol_weight(
    selection: pd.DataFrame,
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Weight proportional to momentum rank × inverse volatility.

    For each selected ticker:
        composite_i = rank_i * (1 / σ_i)
    where rank_i is the cross-sectional momentum rank among selected tickers
    (1 = weakest, n = strongest) and σ_i is the trailing standard deviation
    over the last `window` observations of `asset_returns`.

    Using rank avoids negative composite values when some tickers have
    negative absolute momentum.

    `asset_returns` may be at any frequency; `window` must match that
    frequency (e.g. 252 for daily, 12 for monthly). Slicing uses date-based
    lookup so the function is correct across mismatched index frequencies.

    Falls back to equal weight when there is insufficient history or all
    selected tickers have zero/NaN volatility. No lookahead.
    """
    result = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)

    for date in selection.index:
        cols = selection.columns[selection.loc[date]].tolist()
        if not cols:
            continue

        # Cross-sectional momentum rank among selected tickers (1 = weakest, n = strongest)
        s = scores.loc[date, cols] if date in scores.index else pd.Series(dtype=float)
        ranks = s.rank(method="average") if not (s.empty or s.isna().all()) else pd.Series(1.0, index=cols)

        # Trailing volatility — date-based slice, frequency-agnostic
        ret_window = asset_returns.loc[:date, cols].iloc[-window:].dropna(axis=1)
        if ret_window.shape[0] < 2 or ret_window.empty:
            result.loc[date, cols] = 1.0 / len(cols)
            continue

        vols = ret_window.std().replace(0, np.nan).dropna()
        if vols.empty:
            result.loc[date, cols] = 1.0 / len(cols)
            continue

        common = ranks.index.intersection(vols.index)
        if common.empty:
            result.loc[date, cols] = 1.0 / len(cols)
            continue

        composite = ranks[common] * (1.0 / vols[common])
        weights = composite / composite.sum()
        result.loc[date, weights.index] = weights.values

    return result


def rebalance_weights(weights: pd.DataFrame, rebalance: str = "Q") -> pd.DataFrame:
    """Hold weights constant between rebalance dates (quarterly by default)."""
    freq = _FREQ_MAP.get(rebalance, rebalance)
    # Keep only the last weight observation in each rebalance period,
    # then forward-fill across the full monthly index.
    quarterly = weights.resample(freq).last()
    return quarterly.reindex(weights.index).ffill()


def staggered_rebalance_weights(
    weights: pd.DataFrame,
    rebalance: str = "Q",
    n_tranches: int = 3,
) -> pd.DataFrame:
    """
    Reduce rebalance timing risk by averaging N staggered tranches.

    Each tranche rebalances at a 1-month offset from the previous. For
    n_tranches=3 with quarterly rebalancing:
      Tranche 0 rebalances Mar, Jun, Sep, Dec  (standard Q-end)
      Tranche 1 rebalances Apr, Jul, Oct, Jan  (Q-end + 1 month)
      Tranche 2 rebalances May, Aug, Nov, Feb  (Q-end + 2 months)

    Each tranche uses the signal available at its own rebalance date —
    no lookahead. The portfolio is the equal-weight average of all tranches.

    n_tranches=1 returns identical results to rebalance_weights().

    Note: n_tranches > months-in-rebalance-period (e.g. 4 tranches with
    quarterly = 3 months) produces overlapping offsets; mathematically
    valid but weights some months more than others.
    """
    if n_tranches == 1:
        return rebalance_weights(weights, rebalance=rebalance)

    freq = _FREQ_MAP.get(rebalance, rebalance)
    tranche_list = []

    for offset in range(n_tranches):
        # Relabel the index back by `offset` months so that the date at
        # Q-end+offset aligns with a standard Q-end for the resampler.
        # The row VALUES are unchanged — no lookahead.
        w_temp = weights.copy()
        w_temp.index = weights.index - pd.DateOffset(months=offset)

        # Snap at standard Q-ends of the shifted index
        w_rebal = w_temp.resample(freq).last()

        # Shift snapped dates back to the original calendar
        w_rebal.index = w_rebal.index + pd.DateOffset(months=offset)

        tranche = w_rebal.reindex(weights.index).ffill()
        tranche_list.append(tranche)

    blended = sum(tranche_list) / n_tranches
    return blended.fillna(0.0)