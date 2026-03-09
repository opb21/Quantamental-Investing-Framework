import pandas as pd


def generate_signals(data: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Adds moving average and signal columns to the DataFrame.
    Signal = 1 if Close > MA, else 0.
    """
    data = data.copy()

    data["MA"] = data["Close"].rolling(window=window).mean()
    data["signal"] = (data["Close"] > data["MA"]).astype(int)

    return data
