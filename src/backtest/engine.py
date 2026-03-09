import pandas as pd
from src.backtest.costs import turnover as compute_turnover, apply_costs


def backtest_long_only(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Returns DataFrame with:
      - portfolio_returns (gross)
      - costs
      - portfolio_returns_net
      - cumulative_net
      - turnover
    """
    prices, weights = prices.align(weights, join="inner", axis=0)

    asset_returns = prices.pct_change()

    # weights.shift(1): positions are set at the close of period T and earn
    # the return from T to T+1, so we lag weights by one period.
    portfolio_returns = (asset_returns * weights.shift(1)).sum(axis=1)

    to = compute_turnover(weights)
    portfolio_returns_net = apply_costs(portfolio_returns, to, commission_bps, slippage_bps)

    return pd.DataFrame(
        {
            "portfolio_returns": portfolio_returns,
            "costs": portfolio_returns_net - portfolio_returns,
            "portfolio_returns_net": portfolio_returns_net,
            "cumulative_net": (1 + portfolio_returns_net).cumprod(),
            "turnover": to,
        }
    )
