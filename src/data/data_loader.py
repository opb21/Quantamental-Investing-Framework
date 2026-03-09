import pandas as pd
import yfinance as yf


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily close price data for a single ticker.
    Returns a DataFrame indexed by date with a single 'Close' column (float).
    """
    df = yf.download(ticker, start=start, end=end, progress=False, group_by="column", auto_adjust=False)

    if df is None or df.empty:
        raise ValueError("No data downloaded. Check ticker or date range.")

    # If yfinance returns MultiIndex columns, drop the top level
    # Example: ('Close', 'SPY') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(f"'Close' not found in downloaded data columns: {df.columns}")

    out = df[["Close"]].copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna()
    out.index = pd.to_datetime(out.index)

    return out
