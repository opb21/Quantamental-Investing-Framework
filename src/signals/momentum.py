import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback_months: int = 12, skip_recent_months: int = 1) -> pd.DataFrame:
    """Cross-sectional momentum: (t-skip)/(t-lookback) - 1 on resampled prices."""
    return prices.shift(skip_recent_months) / prices.shift(lookback_months) - 1