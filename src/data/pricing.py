import pandas as pd
import yfinance as yf

# pandas 3.x renamed period-end aliases: 'M' -> 'ME', 'Q' -> 'QE', etc.
_FREQ_MAP = {"M": "ME", "Q": "QE", "SA": "6ME", "A": "YE", "Y": "YE"}


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


def clip_extreme_returns(prices: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Replace price observations that imply extreme period-on-period returns with
    forward-filled values from the prior period.

    Protects backtests against corrupted price data (e.g. unadjusted corporate
    actions in yfinance where a stock alternates between pre- and post-split
    prices, creating artificial returns of thousands of percent).

    A return of ±500% in a single month is the default threshold — well above
    any plausible genuine move for a listed equity.

    Parameters
    ----------
    prices : pd.DataFrame
        Period-end price matrix (dates x tickers).
    threshold : float
        Maximum allowed absolute return per period (e.g. 5.0 = 500%).
        Price observations that imply a larger move are set to NaN and
        forward-filled from the prior period.
    """
    returns = prices.pct_change()
    extreme = returns.abs() > threshold
    if not extreme.any().any():
        return prices
    cleaned = prices.copy()
    cleaned[extreme] = float("nan")
    return cleaned.ffill()



def load_benchmark_yfinance(
    ticker: str,
    start: str,
    end: str,
    price_field: str = "Adj Close",
) -> pd.Series:
    """Return daily price Series for a single benchmark ticker (e.g. '^FTSC')."""
    raw = yf.download([ticker], start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_field].iloc[:, 0]
    else:
        prices = raw[price_field]
    prices.index = pd.to_datetime(prices.index)
    return prices.dropna().rename(ticker)