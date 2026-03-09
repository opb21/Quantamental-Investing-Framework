import pandas as pd

_FREQ_MAP = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}


def select_top_n(scores: pd.DataFrame, n: int) -> pd.DataFrame:
    """Boolean selection matrix True for selected tickers at each date."""
    return scores.rank(axis=1, ascending=False, method="first") <= n


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """Convert selection matrix to weights that sum to 1 per date (or 0 if none)."""
    counts = selection.sum(axis=1)
    return selection.div(counts, axis=0).fillna(0.0)


def rebalance_weights(weights: pd.DataFrame, rebalance: str = "Q") -> pd.DataFrame:
    """Hold weights constant between rebalance dates (quarterly by default)."""
    freq = _FREQ_MAP.get(rebalance, rebalance)
    # Keep only the last weight observation in each rebalance period,
    # then forward-fill across the full monthly index.
    quarterly = weights.resample(freq).last()
    return quarterly.reindex(weights.index).ffill()