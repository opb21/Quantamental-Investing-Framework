import numpy as np
import pandas as pd

_PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "ME": 12, "Q": 4, "QE": 4, "A": 1, "Y": 1, "YE": 1}


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series (underwater equity curve).

    At each period, returns the percentage decline from the preceding peak.
    Values are <= 0; 0 means a new high-water mark.

    Parameters
    ----------
    returns : pd.Series
        Periodic return series.
    """
    cumulative = (1 + returns.dropna()).cumprod()
    drawdown = (cumulative / cumulative.cummax()) - 1
    return drawdown.reindex(returns.index).rename("drawdown")


def rolling_vol(returns: pd.Series, window: int = 12, freq: str = "M") -> pd.Series:
    """
    Annualised rolling portfolio volatility.

    Parameters
    ----------
    returns : pd.Series
        Periodic portfolio return series.
    window : int
        Rolling window in periods.
    freq : str
        Frequency alias used to annualise.
    """
    periods = _PERIODS_PER_YEAR.get(freq, 252)
    return (returns.rolling(window).std() * np.sqrt(periods)).rename("rolling_vol")


def avg_pairwise_correlation(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    window: int = 12,
) -> pd.Series:
    """
    Rolling average pairwise correlation of current portfolio holdings.

    At each period, uses the assets with non-zero weight and computes the
    mean of the upper-triangle of their trailing correlation matrix.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates × tickers). Non-zero entries are held assets.
    asset_returns : pd.DataFrame
        Periodic asset return series (dates × tickers).
    window : int
        Rolling window in periods.
    """
    results = []
    for i in range(len(weights)):
        if i < window - 1:
            results.append(np.nan)
            continue

        held = weights.iloc[i]
        held_tickers = held[held > 0].index.tolist()

        if len(held_tickers) < 2:
            results.append(np.nan)
            continue

        ret_window = (
            asset_returns.iloc[i - window + 1 : i + 1][held_tickers]
            .dropna(axis=1)
        )
        if ret_window.shape[1] < 2:
            results.append(np.nan)
            continue

        corr = ret_window.corr()
        n = len(corr)
        upper = corr.values[np.triu_indices(n, k=1)]
        results.append(float(np.nanmean(upper)))

    return pd.Series(results, index=weights.index, name="avg_pairwise_corr")


def contribution_to_vol(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    window: int = 12,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Per-asset percentage contribution to portfolio volatility at each period.

    Uses the identity: contribution_i = w_i * (Σw)_i / (w'Σw), where Σ is
    the annualised covariance matrix estimated over the trailing window.
    Values sum to 1.0 across held assets at each date.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates × tickers).
    asset_returns : pd.DataFrame
        Periodic asset return series (dates × tickers).
    window : int
        Rolling window used to estimate the covariance matrix.
    freq : str
        Frequency alias used to annualise the covariance matrix.
    """
    periods = _PERIODS_PER_YEAR.get(freq, 252)
    results = []

    for i in range(len(weights)):
        if i < window - 1:
            results.append(pd.Series(np.nan, index=weights.columns))
            continue

        w = weights.iloc[i]
        held_tickers = w[w > 0].index.tolist()

        if not held_tickers:
            results.append(pd.Series(0.0, index=weights.columns))
            continue

        ret_window = (
            asset_returns.iloc[i - window + 1 : i + 1][held_tickers]
            .dropna(axis=1)
        )
        available = ret_window.columns.tolist()
        w_active = w[available]

        if len(w_active) == 0 or w_active.sum() == 0:
            results.append(pd.Series(np.nan, index=weights.columns))
            continue

        cov = ret_window.cov() * periods
        port_var = float(w_active @ cov @ w_active)

        if port_var <= 0:
            results.append(pd.Series(np.nan, index=weights.columns))
            continue

        marginal = cov @ w_active
        contrib = w_active * marginal / port_var  # percentage contribution, sums to 1

        full_contrib = pd.Series(0.0, index=weights.columns)
        full_contrib[contrib.index] = contrib.values
        results.append(full_contrib)

    return pd.DataFrame(results, index=weights.index)
