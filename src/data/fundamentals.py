"""
Fundamental/metadata fetching.

Fetches company metadata and key financial metrics from yfinance.
Failures for individual tickers are handled silently — the ticker is still
included with "Unknown" / NaN values so downstream consumers don't need to
guard against missing rows.
"""
import yfinance as yf
import pandas as pd


def fetch_ticker_info(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch static metadata and fundamental metrics for a list of tickers.

    Returns
    -------
    pd.DataFrame
        Indexed by ticker, columns:
        name, sector, industry,
        marketCap, trailingPE, priceToBook, enterpriseToEbitda,
        trailingEps, earningsGrowth, revenueGrowth, profitMargins, returnOnEquity,
        debtToEquity, currentRatio, dividendYield.

        Numeric fields are NaN where yfinance returns no data.
    """
    _numeric_fields = (
        "marketCap", "trailingPE", "priceToBook", "enterpriseToEbitda",
        "trailingEps", "earningsGrowth", "revenueGrowth", "profitMargins",
        "returnOnEquity", "debtToEquity", "currentRatio", "dividendYield",
    )

    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            row = {
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or ticker,
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
                "website": info.get("website"),
            }
            for field in _numeric_fields:
                row[field] = info.get(field)
        except Exception:
            row = {
                "ticker": ticker,
                "name": ticker,
                "sector": "Unknown",
                "industry": "Unknown",
                "website": None,
            }
            for field in _numeric_fields:
                row[field] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")
