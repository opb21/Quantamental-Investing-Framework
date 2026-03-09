import pandas as pd
import yfinance as yf

# pandas 3.x renamed period-end aliases: 'M' -> 'ME', 'Q' -> 'QE', etc.
_FREQ_MAP = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}


def load_prices_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    price_field: str = "Adj Close",
) -> pd.DataFrame:
    """Return daily prices DataFrame indexed by date, columns=tickers."""
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker download: columns are (field, ticker)
        prices = raw[price_field]
    else:
        # Single-ticker edge case: flat columns
        prices = raw[[price_field]].rename(columns={price_field: tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    return prices.dropna(how="all")


def resample_prices(prices: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    """Resample to period end (e.g., month-end) using last available price."""
    freq = _FREQ_MAP.get(frequency, frequency)
    return prices.resample(freq).last()