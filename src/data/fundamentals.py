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
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or ticker,
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
                "website": info.get("website"),
                "marketCap": info.get("marketCap"),
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "trailingEps": info.get("trailingEps"),
                "earningsGrowth": info.get("earningsGrowth"),
                "revenueGrowth": info.get("revenueGrowth"),
                "profitMargins": info.get("profitMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "debtToEquity": info.get("debtToEquity"),
                "currentRatio": info.get("currentRatio"),
                "dividendYield": info.get("dividendYield"),
            })
        except Exception:
            rows.append({
                "ticker": ticker,
                "name": ticker,
                "sector": "Unknown",
                "industry": "Unknown",
                "website": None,
                "marketCap": None,
                "trailingPE": None,
                "priceToBook": None,
                "enterpriseToEbitda": None,
                "trailingEps": None,
                "earningsGrowth": None,
                "revenueGrowth": None,
                "profitMargins": None,
                "returnOnEquity": None,
                "debtToEquity": None,
                "currentRatio": None,
                "dividendYield": None,
            })
    return pd.DataFrame(rows).set_index("ticker")
