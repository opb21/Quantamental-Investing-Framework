import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute one-way turnover per period from weight changes."""
    return weights.diff().abs().sum(axis=1) / 2


def apply_costs(returns: pd.Series, turnover: pd.Series, commission_bps: float, slippage_bps: float) -> pd.Series:
    """Apply simple linear costs: cost = turnover * (commission+slippage) in bps."""
    total_bps = commission_bps + slippage_bps
    return returns - turnover * total_bps / 10_000