"""
Trading212 CSV export parser.

Trading212 lets users export their full transaction history from:
  Invest tab → History → (menu) → Export CSV

The CSV contains one row per transaction. This module parses it into
a current-positions DataFrame and provides a helper to fetch live prices.
"""
import io

import pandas as pd
import yfinance as yf


# Actions T212 uses for buys and sells
_BUY_ACTIONS = {"market buy", "limit buy", "stop buy"}
_SELL_ACTIONS = {"market sell", "limit sell", "stop sell"}


def parse_t212_csv(
    file_obj,
    ticker_suffix: str = ".L",
) -> pd.DataFrame:
    """
    Parse a Trading212 transaction history CSV export into current positions.

    Parameters
    ----------
    file_obj : file-like object or path
        The CSV exported from Trading212 (History → Export CSV).
    ticker_suffix : str
        Suffix appended to each ticker so it matches yfinance / model format.
        Default ".L" for LSE-listed UK stocks.

    Returns
    -------
    pd.DataFrame
        Indexed by ticker (with suffix), columns:
            shares          : net shares currently held (buys - sells)
            avg_cost        : average buy price per share (account currency)
            total_invested  : total spent on buys (account currency)
        Only rows where net shares > 0 are returned.

    Notes
    -----
    Average cost = total_spent_on_buys / total_shares_bought.
    This matches the "average cost" method shown in the T212 app.
    """
    if isinstance(file_obj, (str, bytes)):
        df = pd.read_csv(file_obj)
    else:
        # Streamlit UploadedFile — read bytes then wrap
        content = file_obj.read()
        df = pd.read_csv(io.BytesIO(content))

    # Normalise column names: strip whitespace, lower for matching
    df.columns = df.columns.str.strip()

    # Identify required columns (T212 uses consistent names)
    _required = ["Action", "Ticker", "No. of shares", "Price / share"]
    missing = [c for c in _required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in T212 CSV: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["_action_lower"] = df["Action"].str.strip().str.lower()
    is_buy = df["_action_lower"].isin(_BUY_ACTIONS)
    is_sell = df["_action_lower"].isin(_SELL_ACTIONS)

    df["_shares"] = pd.to_numeric(df["No. of shares"], errors="coerce").fillna(0)
    df["_price"] = pd.to_numeric(df["Price / share"], errors="coerce").fillna(0)
    df["_ticker"] = df["Ticker"].str.strip()

    # Apply suffix (skip if already ends with it)
    if ticker_suffix:
        df["_ticker"] = df["_ticker"].apply(
            lambda t: t if t.endswith(ticker_suffix) else t + ticker_suffix
        )

    rows = []
    for ticker, group in df.groupby("_ticker"):
        buys = group[is_buy.reindex(group.index, fill_value=False)]
        sells = group[is_sell.reindex(group.index, fill_value=False)]

        total_bought_shares = buys["_shares"].sum()
        total_sold_shares = sells["_shares"].sum()
        net_shares = total_bought_shares - total_sold_shares

        if net_shares <= 1e-8:
            continue  # position fully closed

        total_invested = (buys["_shares"] * buys["_price"]).sum()
        avg_cost = total_invested / total_bought_shares if total_bought_shares > 0 else 0.0

        rows.append({
            "ticker": ticker,
            "shares": round(net_shares, 6),
            "avg_cost": round(avg_cost, 4),
            "total_invested": round(total_invested, 2),
        })

    if not rows:
        return pd.DataFrame(
            columns=["shares", "avg_cost", "total_invested"],
            index=pd.Index([], name="ticker"),
        )

    result = pd.DataFrame(rows).set_index("ticker")
    return result.sort_index()


def fetch_current_prices(tickers: list[str]) -> pd.Series:
    """
    Fetch the latest available close price for each ticker via yfinance.

    Uses a 5-day window and takes the last non-NaN value, which handles
    weekends and public holidays without raising errors.

    Returns
    -------
    pd.Series
        Indexed by ticker. NaN for any ticker that could not be fetched.
    """
    if not tickers:
        return pd.Series(dtype=float)

    raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)

    if raw.empty:
        return pd.Series({t: float("nan") for t in tickers})

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Last valid price per ticker
    return close.apply(lambda col: col.dropna().iloc[-1] if col.dropna().shape[0] > 0 else float("nan"))
