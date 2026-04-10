import sqlite3
from pathlib import Path
import pandas as pd


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    run_name        TEXT NOT NULL,
    strategy_name   TEXT,
    run_dir         TEXT,
    created_at      TEXT,
    market          TEXT,
    start_date      TEXT,
    end_date        TEXT,
    signal_name     TEXT,
    lookback_months INTEGER,
    skip_recent_months INTEGER,
    n_positions     INTEGER,
    rebalance       TEXT,
    commission_bps  REAL,
    slippage_bps    REAL,
    benchmark_ticker TEXT,
    cagr            REAL,
    volatility      REAL,
    sharpe          REAL,
    sortino         REAL,
    calmar          REAL,
    max_drawdown    REAL,
    hit_rate        REAL,
    avg_win         REAL,
    avg_loss        REAL,
    benchmark_cagr  REAL,
    excess_cagr     REAL,
    beta            REAL,
    tracking_error  REAL,
    information_ratio REAL,
    avg_vol         REAL,
    vol_of_vol      REAL,
    weighting       TEXT,
    blend_lookbacks TEXT,
    n_tranches      INTEGER,
    mean_ic         REAL,
    icir            REAL,
    ic_tstat        REAL,
    avg_turnover        REAL,
    annual_cost_drag    REAL,
    wf_group        TEXT,
    wf_period       TEXT
)
"""


_NEW_COLUMNS = ["avg_vol REAL", "vol_of_vol REAL", "weighting TEXT", "blend_lookbacks TEXT", "n_tranches INTEGER", "mean_ic REAL", "icir REAL", "ic_tstat REAL", "avg_turnover REAL", "annual_cost_drag REAL", "wf_group TEXT", "wf_period TEXT"]


def init_db(db_path: str | Path) -> None:
    """Create the runs table if it does not exist; migrate missing columns."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(_CREATE_TABLE)
        for col_def in _NEW_COLUMNS:
            col_name = col_def.split()[0]
            try:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists


def log_run(
    db_path: str | Path,
    run_id: str,
    run_name: str,
    run_dir: Path,
    cfg: dict,
    metrics: dict,
    wf_group: str | None = None,
    wf_period: str | None = None,
) -> None:
    """Insert a completed run into the registry. Replaces on duplicate run_id."""
    init_db(db_path)

    sig = cfg.get("signal", {})
    port = cfg.get("portfolio", {})
    costs = cfg.get("costs", {})
    bench = cfg.get("benchmark", {})

    row = {
        "run_id":             run_id,
        "run_name":           run_name,
        "strategy_name":      cfg.get("strategy_name"),
        "run_dir":            str(run_dir),
        "created_at":         run_id,
        "market":             cfg.get("market"),
        "start_date":         str(cfg.get("start_date", "")),
        "end_date":           str(cfg.get("end_date", "")),
        "signal_name":        sig.get("name"),
        "lookback_months":    sig.get("lookback_months"),
        "skip_recent_months": sig.get("skip_recent_months"),
        "n_positions":        port.get("n_positions"),
        "rebalance":          port.get("rebalance"),
        "commission_bps":     costs.get("commission_bps"),
        "slippage_bps":       costs.get("slippage_bps"),
        "benchmark_ticker":   bench.get("ticker"),
        "cagr":               metrics.get("CAGR"),
        "volatility":         metrics.get("Volatility"),
        "sharpe":             metrics.get("Sharpe"),
        "sortino":            metrics.get("Sortino"),
        "calmar":             metrics.get("Calmar"),
        "max_drawdown":       metrics.get("MaxDrawdown"),
        "hit_rate":           metrics.get("HitRate"),
        "avg_win":            metrics.get("AvgWin"),
        "avg_loss":           metrics.get("AvgLoss"),
        "benchmark_cagr":     metrics.get("BenchmarkCAGR"),
        "excess_cagr":        metrics.get("ExcessCAGR"),
        "beta":               metrics.get("Beta"),
        "tracking_error":     metrics.get("TrackingError"),
        "information_ratio":  metrics.get("InformationRatio"),
        "avg_vol":            metrics.get("AvgVol"),
        "vol_of_vol":         metrics.get("VolOfVol"),
        "weighting":          cfg.get("portfolio", {}).get("weighting", "equal"),
        "blend_lookbacks":    ",".join(map(str, sig.get("lookbacks", []))) or None,
        "n_tranches":         cfg.get("portfolio", {}).get("n_tranches", 1),
        "mean_ic":            metrics.get("MeanIC"),
        "icir":               metrics.get("ICIR"),
        "ic_tstat":           metrics.get("ICTStat"),
        "avg_turnover":       metrics.get("AvgTurnover"),
        "annual_cost_drag":   metrics.get("AnnualCostDrag"),
        "wf_group":           wf_group,
        "wf_period":          wf_period,
    }

    columns = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    sql = f"INSERT OR REPLACE INTO runs ({columns}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as con:
        con.execute(sql, list(row.values()))


def compare_runs(
    db_path: str | Path,
    strategy_name: str | None = None,
    sort_by: str = "sharpe",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Load runs from the registry as a DataFrame.

    Parameters
    ----------
    strategy_name : str, optional
        Filter to a specific strategy. If None, returns all runs.
    sort_by : str
        Column to sort by (default: 'sharpe').
    ascending : bool
        Sort direction.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    query = "SELECT * FROM runs"
    params: list = []
    if strategy_name is not None:
        query += " WHERE strategy_name = ?"
        params.append(strategy_name)

    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(query, con, params=params)

    if sort_by in df.columns and not df.empty:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    return df
