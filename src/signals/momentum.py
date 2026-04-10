import pandas as pd


def momentum(prices: pd.DataFrame, lookback_months: int = 12, skip_recent_months: int = 1) -> pd.DataFrame:
    """Cross-sectional momentum: (t-skip)/(t-lookback) - 1 on resampled prices."""
    return prices.shift(skip_recent_months) / prices.shift(lookback_months) - 1


# Backward-compatible alias
momentum_12_1 = momentum


def blend_momentum(
    prices: pd.DataFrame,
    lookbacks: list,
    blend_weights: list | None = None,
    skip_recent_months: int = 1,
) -> pd.DataFrame:
    """
    Composite momentum score from multiple lookback periods.

    Each component is cross-sectionally z-scored before blending so that
    different lookback periods (with different absolute scales) contribute
    equally regardless of raw magnitude. blend_weights defaults to equal
    weighting and will be normalised to sum to 1 if provided.

    No lookahead: z-scoring is cross-sectional (across tickers at a single
    date), not time-series, so no forward-looking information is introduced.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly price matrix (dates x tickers).
    lookbacks : list[int]
        Lookback periods in months, e.g. [1, 3, 6, 12].
    blend_weights : list[float] | None
        Per-component weights. Normalised to sum to 1. Defaults to equal.
    skip_recent_months : int
        Applied uniformly to all components (standard = 1 month).
    """
    if blend_weights is None:
        blend_weights = [1.0 / len(lookbacks)] * len(lookbacks)
    else:
        total = sum(blend_weights)
        blend_weights = [w / total for w in blend_weights]

    composite = None
    for lb, w in zip(lookbacks, blend_weights):
        raw = momentum_12_1(prices, lookback_months=lb, skip_recent_months=skip_recent_months)
        # Cross-sectional z-score: normalise each date's scores across the universe
        std_row = raw.std(axis=1)
        z = raw.sub(raw.mean(axis=1), axis=0).div(std_row, axis=0)
        # Where std == 0 but data exists (e.g. degenerate lookback == skip_recent
        # producing a constant zero signal), treat as zero contribution rather
        # than propagating NaN through the composite. Rows with no data yet
        # (std is NaN because raw is all-NaN) are left as NaN.
        degenerate = (std_row == 0) & raw.notna().any(axis=1)
        if degenerate.any():
            z.loc[degenerate] = 0.0
        composite = w * z if composite is None else composite + w * z

    return composite