import pandas as pd


def current_leaders(scores: pd.DataFrame, asof_date: pd.Timestamp, top_k: int = 25) -> pd.DataFrame:
    """Return DataFrame of top_k tickers and scores at the most recent date <= asof_date."""
    valid = scores.loc[scores.index <= asof_date]
    if valid.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    row = valid.iloc[-1]
    return row.nlargest(top_k).rename_axis("ticker").reset_index(name="score")