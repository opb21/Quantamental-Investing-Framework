import pandas as pd


def momentum(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_recent_months: int = 1,
    vol_adjust: bool = False,
    vol_method: str = "rolling",
    vol_window: int | None = None,
    ewma_lambda: float = 0.97,
    prices_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly price matrix (dates × tickers).
    lookback_months : int
        Return lookback window in months.
    skip_recent_months : int
        Most recent months to skip (standard = 1 to avoid short-term reversal).
    vol_adjust : bool
        If True, divide the raw return by realised annualised volatility
        (risk-adjusted / Sharpe momentum). Each stock's return is divided by
        its own vol estimate, so the signal is proportional to the stock's
        realised Sharpe ratio over the lookback window.
    vol_method : str
        Vol estimation method: ``"rolling"`` (default) or ``"ewma"``.
    vol_window : int | None
        Window for rolling vol. Interpreted as **trading days** when
        ``prices_daily`` is provided, or months when falling back to monthly
        data. ``None`` → dynamic: ``lookback_months × 21`` days (or
        ``lookback_months`` months in the monthly fallback).
    ewma_lambda : float
        Decay factor for EWMA vol (``vol_method="ewma"``). ``α = 1 − λ``.
        Default 0.97 (RiskMetrics convention: slow decay, ~33-month memory).
    prices_daily : pd.DataFrame | None
        Daily price matrix (dates × tickers). When provided, vol is estimated
        from daily returns (annualised by √252) and aligned back to the
        monthly index. When None, vol falls back to monthly returns (√12).
    """
    ret = prices.shift(skip_recent_months) / prices.shift(lookback_months) - 1
    if not vol_adjust:
        return ret

    if prices_daily is not None:
        rets = prices_daily.pct_change()
        ann_factor = 252 ** 0.5
        if vol_method == "ewma":
            raw_vol = rets.ewm(alpha=1 - ewma_lambda, adjust=False).std()
        else:  # rolling
            w = vol_window if vol_window is not None else lookback_months * 21
            raw_vol = rets.rolling(w).std()
        # Align daily vol series to monthly price dates
        vol = (raw_vol * ann_factor).reindex(prices.index).ffill()
    else:
        rets = prices.pct_change()
        ann_factor = 12 ** 0.5
        if vol_method == "ewma":
            raw_vol = rets.ewm(alpha=1 - ewma_lambda, adjust=False).std()
        else:  # rolling
            w = vol_window if vol_window is not None else lookback_months
            raw_vol = rets.rolling(w).std()
        vol = raw_vol * ann_factor

    # NaN where vol == 0 to avoid inf
    return ret / vol.where(vol > 0)


# Backward-compatible alias
momentum_12_1 = momentum


def blend_momentum(
    prices: pd.DataFrame,
    lookbacks: list,
    blend_weights: list | None = None,
    skip_recent_months: int = 1,
    vol_adjust: bool = False,
    vol_method: str = "rolling",
    vol_window: int | None = None,
    ewma_lambda: float = 0.97,
    prices_daily: pd.DataFrame | None = None,
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
        Lookback periods in months, e.g. [3, 6, 12].
    blend_weights : list[float] | None
        Per-component weights. Normalised to sum to 1. Defaults to equal.
    skip_recent_months : int
        Applied uniformly to all components (standard = 1 month).
    vol_adjust : bool
        If True, each component return is divided by realised vol before
        z-scoring (risk-adjusted momentum). Applied per-component at each
        lookback horizon independently.
    vol_method : str
        Vol estimation method passed to each ``momentum()`` call:
        ``"rolling"`` (default) or ``"ewma"``.
    vol_window : int | None
        Vol window passed to each ``momentum()`` call. Trading days when
        ``prices_daily`` is provided; months otherwise. ``None`` → each
        component uses its own lookback (dynamic per-component).
    ewma_lambda : float
        EWMA decay factor. Only used when ``vol_method="ewma"``.
    prices_daily : pd.DataFrame | None
        Daily price matrix for vol estimation. Passed through to each
        ``momentum()`` call unchanged.
    """
    if blend_weights is None:
        blend_weights = [1.0 / len(lookbacks)] * len(lookbacks)
    else:
        total = sum(blend_weights)
        blend_weights = [w / total for w in blend_weights]

    composite = None
    for lb, w in zip(lookbacks, blend_weights):
        raw = momentum(
            prices,
            lookback_months=lb,
            skip_recent_months=skip_recent_months,
            vol_adjust=vol_adjust,
            vol_method=vol_method,
            vol_window=vol_window,
            ewma_lambda=ewma_lambda,
            prices_daily=prices_daily,
        )
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