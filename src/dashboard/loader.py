"""
Data loading layer for the dashboard.

All disk I/O is centralised here. Both public functions are decorated with
@st.cache_data so Streamlit avoids redundant reads on rerun.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.fundamentals import fetch_ticker_info as _fetch_info
from src.io.run_db import compare_runs


@st.cache_data(ttl=60)
def list_runs(db_path: str) -> pd.DataFrame:
    """
    Return all runs from the registry as a DataFrame.
    Returns an empty DataFrame if the database does not exist.
    """
    return compare_runs(db_path)


@st.cache_data
def load_run_data(run_dir: str) -> dict:
    """
    Load all artifacts for a single run from disk.

    Returns a dict with keys:
        metrics            : dict
        returns            : pd.DataFrame (portfolio_returns_net, cumulative_net,
                             turnover, and optionally rolling_sharpe_12m,
                             drawdown, rolling_vol)
        weights            : pd.DataFrame (dates x tickers, full universe)
        leaders            : pd.DataFrame
        benchmark_returns  : pd.Series | None
        avg_pairwise_corr  : pd.Series | None
        contrib_to_vol     : pd.DataFrame | None
    """
    d = Path(run_dir)

    with open(d / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    returns = pd.read_csv(d / "returns.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(d / "weights.csv", index_col=0, parse_dates=True)
    leaders = pd.read_csv(d / "leaders.csv")

    def _load_optional_series(filename: str) -> pd.Series | None:
        path = d / filename
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]

    def _load_optional_df(filename: str) -> pd.DataFrame | None:
        path = d / filename
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    return {
        "metrics": metrics,
        "returns": returns,
        "weights": weights,
        "leaders": leaders,
        "benchmark_returns": _load_optional_series("benchmark_returns.csv"),
        "avg_pairwise_corr": _load_optional_series("avg_pairwise_corr.csv"),
        "contrib_to_vol": _load_optional_df("contrib_to_vol.csv"),
        "holdings_stats": _load_optional_df("holdings_stats.csv"),
        "stock_attribution": _load_optional_df("stock_attribution.csv"),
        "asset_returns": _load_optional_df("asset_returns.csv"),
        "scores": _load_optional_df("scores.csv"),
        "ic_series": _load_optional_series("ic_series.csv"),
        "ic_decay": _load_optional_series("ic_decay.csv"),
    }


@st.cache_data(ttl=86400)
def fetch_ticker_info(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Cached wrapper around src.data.fundamentals.fetch_ticker_info.

    Accepts a tuple (for hashability) and refreshes at most once per day.
    Returns a DataFrame indexed by ticker with columns: name, sector, industry.
    """
    return _fetch_info(list(tickers))


@st.cache_data
def load_run_returns(run_dir: str) -> pd.DataFrame | None:
    """Load only returns.csv for a run — lightweight loader for multi-run comparison."""
    path = Path(run_dir) / "returns.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_active_tickers(weights: pd.DataFrame) -> list[str]:
    """
    Return all tickers with any nonzero weight, sorted by most-recently-held
    descending. This ensures current holdings appear first so the chart layer's
    top-N truncation always captures the live portfolio rather than the
    longest-tenured historical positions.
    """
    w = weights.fillna(0)
    active = w.columns[(w > 0).any()]
    last_held = {col: w[col].where(w[col] > 0).last_valid_index() for col in active}
    return sorted(active, key=lambda c: last_held[c], reverse=True)
