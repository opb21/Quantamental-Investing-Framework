import pytest
from pathlib import Path

from src.io.run_db import init_db, log_run, compare_runs


def _sample_cfg():
    return {
        "strategy_name": "momentum",
        "market": "UK",
        "start_date": "2015-01-01",
        "end_date": "2024-01-01",
        "signal": {"name": "momentum_12_1", "lookback_months": 12, "skip_recent_months": 1},
        "portfolio": {"n_positions": 12, "rebalance": "Q"},
        "costs": {"commission_bps": 10, "slippage_bps": 50},
        "benchmark": {"ticker": "^FTSC"},
    }


def _sample_metrics():
    return {
        "CAGR": 0.138, "Volatility": 0.15, "Sharpe": 0.75,
        "Sortino": 1.1, "Calmar": 0.38, "MaxDrawdown": -0.36,
        "HitRate": 0.58, "AvgWin": 0.03, "AvgLoss": -0.025,
        "BenchmarkCAGR": 0.043, "ExcessCAGR": 0.095,
        "Beta": 1.09, "TrackingError": 0.128, "InformationRatio": 0.78,
    }


def test_init_db_creates_table(tmp_path):
    db = tmp_path / "runs.db"
    init_db(db)
    import sqlite3
    with sqlite3.connect(db) as con:
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("runs",) in tables


def test_log_run_inserts_row(tmp_path):
    db = tmp_path / "runs.db"
    log_run(db, "20240101_120000", "UK_momentum_Q", tmp_path, _sample_cfg(), _sample_metrics())
    df = compare_runs(db)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "20240101_120000"


def test_log_run_stores_strategy_name(tmp_path):
    db = tmp_path / "runs.db"
    log_run(db, "20240101_120001", "UK_momentum_Q", tmp_path, _sample_cfg(), _sample_metrics())
    df = compare_runs(db)
    assert df.iloc[0]["strategy_name"] == "momentum"


def test_log_run_stores_metrics(tmp_path):
    db = tmp_path / "runs.db"
    log_run(db, "20240101_120002", "UK_momentum_Q", tmp_path, _sample_cfg(), _sample_metrics())
    df = compare_runs(db)
    assert abs(df.iloc[0]["sharpe"] - 0.75) < 1e-6
    assert abs(df.iloc[0]["excess_cagr"] - 0.095) < 1e-6


def test_log_run_replace_on_duplicate_id(tmp_path):
    db = tmp_path / "runs.db"
    log_run(db, "20240101_120003", "run_a", tmp_path, _sample_cfg(), _sample_metrics())
    metrics2 = {**_sample_metrics(), "Sharpe": 0.99}
    log_run(db, "20240101_120003", "run_a", tmp_path, _sample_cfg(), metrics2)
    df = compare_runs(db)
    assert len(df) == 1
    assert abs(df.iloc[0]["sharpe"] - 0.99) < 1e-6


def test_compare_runs_returns_empty_when_no_db(tmp_path):
    df = compare_runs(tmp_path / "nonexistent.db")
    assert df.empty


def test_compare_runs_filters_by_strategy(tmp_path):
    db = tmp_path / "runs.db"
    cfg_a = {**_sample_cfg(), "strategy_name": "momentum"}
    cfg_b = {**_sample_cfg(), "strategy_name": "value"}
    log_run(db, "id_a", "run_a", tmp_path, cfg_a, _sample_metrics())
    log_run(db, "id_b", "run_b", tmp_path, cfg_b, _sample_metrics())
    df = compare_runs(db, strategy_name="momentum")
    assert len(df) == 1
    assert df.iloc[0]["strategy_name"] == "momentum"


def test_compare_runs_sorts_by_sharpe(tmp_path):
    db = tmp_path / "runs.db"
    log_run(db, "id_1", "run_1", tmp_path, _sample_cfg(), {**_sample_metrics(), "Sharpe": 0.5})
    log_run(db, "id_2", "run_2", tmp_path, _sample_cfg(), {**_sample_metrics(), "Sharpe": 0.9})
    log_run(db, "id_3", "run_3", tmp_path, _sample_cfg(), {**_sample_metrics(), "Sharpe": 0.7})
    df = compare_runs(db, sort_by="sharpe", ascending=False)
    assert list(df["sharpe"]) == [0.9, 0.7, 0.5]
