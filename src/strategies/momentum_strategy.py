import pandas as pd

from src.signals.momentum import momentum, blend_momentum
from src.portfolio.construction import (
    select_top_n, equal_weight, inverse_vol_weight,
    momentum_inv_vol_weight, rebalance_weights, staggered_rebalance_weights,
)


def run_momentum_strategy(
    prices: pd.DataFrame,
    cfg: dict,
    prices_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build momentum signal and portfolio weights from prices and config.

    Returns
    -------
    weights : pd.DataFrame
        Rebalanced portfolio weights (tickers x dates).
    scores : pd.DataFrame
        Raw momentum scores used for selection.
    """
    sig_cfg = cfg["signal"]
    port_cfg = cfg["portfolio"]

    # --- Signal ---
    vol_adjust = sig_cfg.get("vol_adjust", False)
    vol_method = sig_cfg.get("vol_method", "rolling")
    vol_window = sig_cfg.get("vol_window")  # None → dynamic (lookback × 21 days or lookback months)
    ewma_lambda = sig_cfg.get("ewma_lambda", 0.97)

    if sig_cfg.get("name") == "blend_momentum":
        scores = blend_momentum(
            prices,
            lookbacks=sig_cfg["lookbacks"],
            blend_weights=sig_cfg.get("blend_weights"),
            skip_recent_months=sig_cfg.get("skip_recent_months", 1),
            vol_adjust=vol_adjust,
            vol_method=vol_method,
            vol_window=vol_window,
            ewma_lambda=ewma_lambda,
            prices_daily=prices_daily,
        )
    else:
        scores = momentum(
            prices,
            lookback_months=sig_cfg.get("lookback_months", 12),
            skip_recent_months=sig_cfg.get("skip_recent_months", 1),
            vol_adjust=vol_adjust,
            vol_method=vol_method,
            vol_window=vol_window,
            ewma_lambda=ewma_lambda,
            prices_daily=prices_daily,
        )

    # --- Selection & weighting ---
    selection = select_top_n(scores, n=int(port_cfg.get("n_positions", 12)))

    weighting = port_cfg.get("weighting", "equal")
    risk_window_months = int(cfg.get("risk", {}).get("window", 12))

    # Use daily returns for vol estimation when available — far more data points
    # (~252/yr vs 12/yr) gives much more stable and accurate volatility estimates.
    # Falls back to monthly returns if daily prices were not provided.
    if prices_daily is not None:
        vol_returns = prices_daily.pct_change()
        vol_window = risk_window_months * 21  # ~21 trading days per month
    else:
        vol_returns = prices.pct_change()
        vol_window = risk_window_months

    if weighting == "inv_vol":
        w_raw = inverse_vol_weight(selection, vol_returns, window=vol_window)
    elif weighting == "momentum_inv_vol":
        w_raw = momentum_inv_vol_weight(selection, scores, vol_returns, window=vol_window)
    else:
        w_raw = equal_weight(selection)

    # --- Rebalancing ---
    n_tranches = int(port_cfg.get("n_tranches", 1))
    weights = staggered_rebalance_weights(
        w_raw, rebalance=port_cfg.get("rebalance", "Q"), n_tranches=n_tranches
    )

    return weights, scores
