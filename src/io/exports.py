import pandas as pd


def current_leaders(scores: pd.DataFrame, asof_date: pd.Timestamp, top_k: int = 25) -> pd.DataFrame:
    """Return DataFrame of top_k tickers and scores at the most recent date <= asof_date."""
    row = scores.loc[scores.index <= asof_date].iloc[-1]
    return row.nlargest(top_k).rename_axis("ticker").reset_index(name="score")