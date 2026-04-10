import pandas as pd
import pytest
from pathlib import Path

from src.data.universe import load_universe


def test_load_universe_returns_list(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("ticker\nAAPL\nMSFT\nGOOG\n")
    result = load_universe(csv)
    assert result == ["AAPL", "MSFT", "GOOG"]


def test_load_universe_strips_whitespace(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("ticker\n AAPL \n MSFT\n")
    result = load_universe(csv)
    assert result == ["AAPL", "MSFT"]


def test_load_universe_custom_column(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("symbol\nAAPL\nMSFT\n")
    result = load_universe(csv, ticker_column="symbol")
    assert result == ["AAPL", "MSFT"]


def test_load_universe_drops_na(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("ticker\nAAPL\n\nMSFT\n")
    result = load_universe(csv)
    assert "" not in result
    assert len(result) == 2


def test_load_universe_applies_suffix(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("ticker\nBEZ\nWEIR\nDPLM\n")
    result = load_universe(csv, ticker_suffix=".L")
    assert result == ["BEZ.L", "WEIR.L", "DPLM.L"]


def test_load_universe_strips_trailing_dot_before_suffix(tmp_path):
    csv = tmp_path / "tickers.csv"
    csv.write_text("ticker\nTW.\nQQ.\nAO.\n")
    result = load_universe(csv, ticker_suffix=".L")
    assert result == ["TW.L", "QQ.L", "AO.L"]
