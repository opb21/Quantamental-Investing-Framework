from pathlib import Path
import pandas as pd


def load_universe(csv_path: str | Path, ticker_column: str = "ticker") -> list[str]:
    """Load tickers from a CSV file."""
    df = pd.read_csv(csv_path)
    return df[ticker_column].dropna().str.strip().tolist()