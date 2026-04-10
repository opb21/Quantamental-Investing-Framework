"""
Signal quality analytics: Information Coefficient (IC) analysis.

IC measures the cross-sectional Spearman rank correlation between a signal
at time t and the returns earned in the next period. Mean IC > 0 means the
signal has predictive power on average; ICIR (mean/std) measures consistency.

Note on lookahead: IC analysis is a purely historical diagnostic. The
forward-looking shift used here is intentional — we want to measure
predictive power. This module must never be used for live signal generation.
"""
import numpy as np
import pandas as pd


def _cross_sectional_spearman(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """
    Row-wise Spearman rank correlation between two DataFrames.

    For each date, computes the rank correlation across all tickers that
    have valid values in both DataFrames. Dates with fewer than 2 valid
    common tickers are dropped.
    """
    ranked1 = df1.rank(axis=1)
    ranked2 = df2.rank(axis=1)
    return ranked1.corrwith(ranked2, axis=1).dropna()


def information_coefficient(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """
    Cross-sectional Spearman IC at each date.

    IC_t = rank_corr(scores[t], asset_returns[t+1])

    Measures whether the signal at time t predicts which stocks will
    outperform in the next period. A consistently positive IC indicates
    genuine predictive power.

    Parameters
    ----------
    scores : pd.DataFrame
        Signal values (dates × tickers). Higher = more favourable.
    asset_returns : pd.DataFrame
        Periodic asset returns (dates × tickers), same frequency as scores.

    Returns
    -------
    pd.Series
        IC at each date, named "ic". The last date is always NaN (no
        forward return available) and is dropped.
    """
    forward_returns = asset_returns.shift(-1)
    return _cross_sectional_spearman(scores, forward_returns).rename("ic")


def ic_decay(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    max_horizon: int = 12,
) -> pd.Series:
    """
    Mean IC at each forward horizon from 1 to max_horizon periods.

    IC_h = mean over all t of rank_corr(scores[t], compound_return[t → t+h])

    A fast decay (IC drops to ~0 by horizon 3) means the signal is only
    useful for short holding periods. A slow decay suggests the signal has
    lasting predictive power and can support longer rebalance intervals.

    Parameters
    ----------
    scores : pd.DataFrame
        Signal values (dates × tickers).
    asset_returns : pd.DataFrame
        Periodic returns (dates × tickers), same frequency as scores.
    max_horizon : int
        Maximum forward horizon in periods (default 12).

    Returns
    -------
    pd.Series
        Mean IC indexed by horizon (1, 2, ..., max_horizon), named "ic_decay".
    """
    result = {}
    for h in range(1, max_horizon + 1):
        # Compound h-period forward return from signal date t:
        # (1+r_{t+1}) * (1+r_{t+2}) * ... * (1+r_{t+h}) - 1
        fwd_h = (1 + asset_returns).shift(-1).rolling(h).apply(np.prod, raw=True) - 1
        result[h] = _cross_sectional_spearman(scores, fwd_h).mean()
    return pd.Series(result, name="ic_decay")


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summary statistics from an IC time series.

    Returns
    -------
    dict with keys:
        MeanIC    : average IC across all dates
        ICIR      : IC Information Ratio = MeanIC / std(IC)
        ICHitRate : fraction of dates with positive IC
        ICTStat   : t-statistic testing H0: MeanIC = 0
    """
    ic = ic_series.dropna()
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    n = len(ic)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_rate = float((ic > 0).mean())
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 0 else 0.0
    return {
        "MeanIC":    round(mean_ic, 4),
        "ICIR":      round(icir, 4),
        "ICHitRate": round(hit_rate, 4),
        "ICTStat":   round(t_stat, 4),
    }
