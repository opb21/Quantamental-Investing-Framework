from pathlib import Path
import pandas as pd


def load_universe(
    csv_path: str | Path,
    ticker_column: str = "ticker",
    ticker_suffix: str = "",
) -> list[str]:
    """Load tickers from a CSV file.

    Parameters
    ----------
    ticker_suffix : str
        Optional suffix appended to every ticker (e.g. ".L" for LSE).
        Trailing dots on the raw ticker are stripped before appending,
        so "TW." + ".L" correctly yields "TW.L" not "TW..L".
    """
    df = pd.read_csv(csv_path)
    tickers = df[ticker_column].dropna().str.strip().tolist()
    if ticker_suffix:
        tickers = [t.rstrip(".") + ticker_suffix for t in tickers]
    return tickers