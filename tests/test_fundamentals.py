"""Tests for src/data/fundamentals.py"""
import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.fundamentals import fetch_ticker_info

_FULL_INFO = {
    "shortName": "Test Corp",
    "sector": "Technology",
    "industry": "Software",
    "website": "https://testcorp.example.com",
    "marketCap": 500_000_000,
    "trailingPE": 18.5,
    "priceToBook": 2.3,
    "enterpriseToEbitda": 12.1,
    "trailingEps": 1.25,
    "earningsGrowth": 0.08,
    "revenueGrowth": 0.12,
    "profitMargins": 0.09,
    "returnOnEquity": 0.18,
    "debtToEquity": 45.0,
    "currentRatio": 1.8,
    "dividendYield": 0.025,
}

_EXPECTED_COLUMNS = [
    "name", "sector", "industry", "website",
    "marketCap", "trailingPE", "priceToBook", "enterpriseToEbitda",
    "trailingEps", "earningsGrowth", "revenueGrowth", "profitMargins", "returnOnEquity",
    "debtToEquity", "currentRatio", "dividendYield",
]


def _mock_ticker(info_dict):
    mock = MagicMock()
    mock.info = info_dict
    return mock


def test_returns_all_columns():
    with patch("yfinance.Ticker", return_value=_mock_ticker(_FULL_INFO)):
        df = fetch_ticker_info(["TEST.L"])
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert df.loc["TEST.L", "name"] == "Test Corp"
    assert df.loc["TEST.L", "sector"] == "Technology"
    assert df.loc["TEST.L", "website"] == "https://testcorp.example.com"
    assert df.loc["TEST.L", "marketCap"] == 500_000_000
    assert df.loc["TEST.L", "trailingPE"] == 18.5
    assert df.loc["TEST.L", "profitMargins"] == 0.09


def test_missing_numeric_fields_are_nan():
    sparse_info = {"shortName": "Sparse Co"}
    with patch("yfinance.Ticker", return_value=_mock_ticker(sparse_info)):
        df = fetch_ticker_info(["SPARSE.L"])
    assert df.loc["SPARSE.L", "name"] == "Sparse Co"
    assert df.loc["SPARSE.L", "sector"] == "Unknown"
    numeric_cols = [c for c in _EXPECTED_COLUMNS if c not in ("name", "sector", "industry")]
    for col in numeric_cols:
        val = df.loc["SPARSE.L", col]
        assert val is None or (isinstance(val, float) and math.isnan(val)), (
            f"Expected NaN for {col}, got {val!r}"
        )


def test_exception_path_included():
    with patch("yfinance.Ticker", side_effect=Exception("network error")):
        df = fetch_ticker_info(["ERR.L"])
    assert "ERR.L" in df.index
    assert df.loc["ERR.L", "name"] == "ERR.L"
    assert df.loc["ERR.L", "sector"] == "Unknown"
    numeric_cols = [c for c in _EXPECTED_COLUMNS if c not in ("name", "sector", "industry")]
    for col in numeric_cols:
        val = df.loc["ERR.L", col]
        assert val is None or (isinstance(val, float) and math.isnan(val)), (
            f"Expected NaN for {col} on exception path, got {val!r}"
        )


def test_backward_compatible_columns():
    """name, sector, industry must always be present regardless of yfinance content."""
    for info in [_FULL_INFO, {"shortName": "X"}, {}]:
        with patch("yfinance.Ticker", return_value=_mock_ticker(info)):
            df = fetch_ticker_info(["T.L"])
        for col in ("name", "sector", "industry"):
            assert col in df.columns, f"Missing column: {col}"
