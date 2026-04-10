from pathlib import Path
import numpy as np
import yaml
import pandas as pd

from src.data.universe import load_universe
from src.data.pricing import load_prices_yfinance, resample_prices, clip_extreme_returns, load_benchmark_yfinance
from src.strategies.momentum_strategy import run_momentum_strategy
from src.backtest.engine import backtest_long_only
from src.analytics.performance import calculate_performance, calculate_relative_performance, rolling_sharpe
from src.analytics.risk import drawdown_series, rolling_vol, avg_pairwise_correlation, contribution_to_vol
from src.analytics.signal_analysis import information_coefficient, ic_decay as compute_ic_decay, ic_summary
from src.io.results_store import create_run_dir, save_run_artifacts
from src.io.run_db import log_run
from src.io.exports import current_leaders


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def resolve_config_paths(cfg: dict, base_dir: Path) -> None:
    """
    Resolve relative file/directory paths in a config dict to absolute paths.

    Mutates cfg in place. base_dir is the reference point for resolution
    (typically the project root). Callers — both the CLI main() and the
    dashboard — pass their own base_dir so this function stays pure of any
    CWD assumptions.
    """
    def _abs(p: str) -> str:
        path = Path(p)
        return str(base_dir / path) if not path.is_absolute() else p

    if "universe" in cfg and "file" in cfg["universe"]:
        cfg["universe"]["file"] = _abs(cfg["universe"]["file"])
    if "outputs" in cfg:
        for key in ("runs_dir", "db_path"):
            if key in cfg["outputs"]:
                cfg["outputs"][key] = _abs(cfg["outputs"][key])


def load_prices_for_config(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """
    Fetch and resample universe prices and benchmark returns for a config.

    Separating data loading from pipeline execution allows a sweep to download
    prices once and reuse them across all variants (which share the same
    universe, date range, and pricing frequency).

    Returns
    -------
    prices : pd.DataFrame
        Month-end (or configured frequency) prices, columns = tickers.
    prices_daily : pd.DataFrame
        Raw daily prices — retained for daily-return vol estimation in
        portfolio construction (more observations → more stable vol).
    benchmark_returns : pd.Series or None
        Periodic benchmark returns, or None if no benchmark configured.
    """
    universe_cfg = cfg["universe"]
    tickers = load_universe(
        universe_cfg["file"],
        universe_cfg.get("ticker_column", "ticker"),
        universe_cfg.get("ticker_suffix", ""),
    )
    freq = cfg["pricing"].get("frequency", "M")
    prices_daily = load_prices_yfinance(
        tickers=tickers,
        start=cfg["start_date"],
        end=cfg["end_date"],
        price_field=cfg["pricing"].get("price_field", "Adj Close"),
    )
    prices = resample_prices(prices_daily, frequency=freq)
    prices = clip_extreme_returns(prices)

    benchmark_returns = None
    bench_cfg = cfg.get("benchmark", {})
    if bench_cfg:
        bench_daily = load_benchmark_yfinance(
            ticker=bench_cfg["ticker"],
            start=cfg["start_date"],
            end=cfg["end_date"],
            price_field=cfg["pricing"].get("price_field", "Adj Close"),
        )
        benchmark_returns = resample_prices(bench_daily.to_frame(), frequency=freq).iloc[:, 0].pct_change()

    return prices, prices_daily, benchmark_returns


def run_pipeline(
    cfg: dict,
    cfg_text: str = "",
    prices: pd.DataFrame | None = None,
    prices_daily: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    wf_group: str | None = None,
    wf_period: str | None = None,
) -> dict:
    """
    Run the full pipeline for a given config dict.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Pre-loaded monthly prices. If None, fetched from yfinance.
    prices_daily : pd.DataFrame, optional
        Pre-loaded daily prices. Used for vol estimation in portfolio
        construction. If None, fetched alongside prices.
    benchmark_returns : pd.Series, optional
        Pre-loaded benchmark return series. If None, fetched from yfinance
        (unless no benchmark is configured).

    Returns a dict with keys: 'metrics', 'run_dir', 'run_id'.
    """
    if prices is None:
        prices, prices_daily, benchmark_returns = load_prices_for_config(cfg)

    # Strategy
    weights, scores = run_momentum_strategy(prices, cfg, prices_daily=prices_daily)

    # Backtest + costs
    costs_cfg = cfg.get("costs", {})
    results = backtest_long_only(
        prices=prices,
        weights=weights,
        commission_bps=float(costs_cfg.get("commission_bps", 0.0)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 0.0)),
    )

    freq = cfg["pricing"].get("frequency", "M")
    risk_window = cfg.get("risk", {}).get("window", 12)
    _ppy = {"D": 252, "W": 52, "M": 12, "ME": 12, "Q": 4, "QE": 4, "SA": 2, "6ME": 2, "A": 1, "Y": 1, "YE": 1}
    periods = _ppy.get(freq, 12)

    metrics = calculate_performance(results["portfolio_returns_net"], freq=freq)
    _gross_metrics = calculate_performance(results["portfolio_returns"], freq=freq)
    metrics["AnnualCostDrag"] = round(_gross_metrics["CAGR"] - metrics["CAGR"], 4)
    results["rolling_sharpe_12m"] = rolling_sharpe(results["portfolio_returns_net"], window=12, freq=freq)

    relative_metrics = {}
    if benchmark_returns is not None:
        relative_metrics = calculate_relative_performance(results["portfolio_returns_net"], benchmark_returns, freq=freq)

    # Risk analytics
    asset_returns = prices.pct_change()
    dd = drawdown_series(results["portfolio_returns_net"])
    results["drawdown"] = dd
    rv = rolling_vol(results["portfolio_returns_net"], window=risk_window, freq=freq)
    results["rolling_vol"] = rv
    avg_corr = avg_pairwise_correlation(weights, asset_returns, window=risk_window)
    ctv = contribution_to_vol(weights, asset_returns, window=risk_window, freq=freq)

    rv_clean = rv.dropna()
    risk_metrics: dict = {}
    if not rv_clean.empty:
        risk_metrics["AvgVol"] = float(rv_clean.mean())
        risk_metrics["VolOfVol"] = float(rv_clean.std())

    to_series = results["turnover"].replace(0, float("nan")).dropna()
    if not to_series.empty:
        risk_metrics["AvgTurnover"] = float(to_series.mean())

    # Signal quality analytics
    ic_series = information_coefficient(scores, asset_returns)
    ic_decay_curve = compute_ic_decay(scores, asset_returns, max_horizon=risk_window)
    signal_metrics = ic_summary(ic_series)

    all_metrics = {**metrics, **relative_metrics, **risk_metrics, **signal_metrics}

    # Current holdings snapshot with per-stock stats
    last_weights_date = weights.dropna(how="all").index[-1]
    last_weights = weights.loc[last_weights_date]
    current_tickers = last_weights[last_weights > 0].index.tolist()
    daily_returns = prices_daily.pct_change() if prices_daily is not None else None
    holdings_rows = []
    for t in current_tickers:
        # Use prices directly for returns to avoid NaN-gap distortion from pct_change.
        t_prices_nm = prices[t].iloc[-(risk_window + 1):].dropna()
        period_return = (
            float(t_prices_nm.iloc[-1] / t_prices_nm.iloc[0] - 1)
            if len(t_prices_nm) >= 2 else None
        )
        t_prices_1m = prices[t].iloc[-2:].dropna()
        return_1m = (
            float(t_prices_1m.iloc[-1] / t_prices_1m.iloc[0] - 1)
            if len(t_prices_1m) >= 2 else None
        )
        # Volatility: prefer daily returns (more observations, more stable estimate).
        if daily_returns is not None and t in daily_returns.columns:
            t_rets_vol = daily_returns[t].iloc[-(risk_window * 21):].dropna()
            t_vol = float(t_rets_vol.std() * np.sqrt(252)) if len(t_rets_vol) > 1 else None
        else:
            t_rets_window = asset_returns[t].iloc[-risk_window:].dropna()
            t_vol = float(t_rets_window.std() * np.sqrt(periods)) if len(t_rets_window) > 1 else None
        holdings_rows.append({
            "ticker": t,
            f"return_{risk_window}m": round(period_return, 4) if period_return is not None else None,
            "return_1m": round(return_1m, 4) if return_1m is not None else None,
            "volatility": round(t_vol, 4) if t_vol is not None else None,
        })
    if holdings_rows:
        holdings_stats = pd.DataFrame(holdings_rows).set_index("ticker")
    else:
        holdings_stats = pd.DataFrame(
            columns=[f"return_{risk_window}m", "return_1m", "volatility"]
        )
        holdings_stats.index.name = "ticker"

    # Stock-level return attribution (period × held tickers)
    held_tickers = [t for t in weights.columns if (weights[t] > 0).any()]
    stock_attribution = (weights[held_tickers].shift(1) * asset_returns[held_tickers]).dropna(how="all")

    # Leaders export
    last_date = scores.dropna(how="all").index.max()
    leaders = current_leaders(scores, asof_date=last_date, top_k=25)

    # Save artifacts
    sig_cfg = cfg["signal"]
    port_cfg = cfg["portfolio"]
    run_name = f"{cfg['market']}_{sig_cfg['name']}_{port_cfg['rebalance']}"
    if wf_period:
        run_name = f"{wf_period.upper()}_{run_name}"
    run_dir, run_id = create_run_dir(cfg["outputs"]["runs_dir"], run_name=run_name)

    if not cfg_text:
        cfg_text = yaml.dump(cfg)

    save_run_artifacts(
        run_dir=run_dir,
        config_text=cfg_text,
        metrics=all_metrics,
        weights=weights,
        returns=results,
        leaders=leaders,
        benchmark_returns=benchmark_returns,
        avg_pairwise_corr=avg_corr,
        contrib_to_vol=ctv,
        holdings_stats=holdings_stats,
        stock_attribution=stock_attribution,
        asset_returns=asset_returns[held_tickers],
        scores=scores,
        ic_series=ic_series,
        ic_decay=ic_decay_curve,
    )

    # Log to registry
    db_path = cfg["outputs"].get("db_path", Path(cfg["outputs"]["runs_dir"]).parent / "runs.db")
    log_run(
        db_path=db_path,
        run_id=run_id,
        run_name=run_name,
        run_dir=run_dir,
        cfg=cfg,
        metrics=all_metrics,
        wf_group=wf_group,
        wf_period=wf_period,
    )

    return {"metrics": all_metrics, "run_dir": run_dir, "run_id": run_id}


def main() -> None:
    _project_root = Path(__file__).resolve().parents[1]
    cfg_path = _project_root / "config/uk_smallcap.yaml"
    cfg_text = read_text(cfg_path)
    cfg = yaml.safe_load(cfg_text)
    resolve_config_paths(cfg, _project_root)

    result = run_pipeline(cfg, cfg_text=cfg_text)
    m = result["metrics"]

    print("Run saved to:", result["run_dir"])
    print("Portfolio :", {k: m[k] for k in ("CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "MaxDrawdown") if k in m})
    if "ExcessCAGR" in m:
        print("vs Benchmark:", {k: m[k] for k in ("BenchmarkCAGR", "ExcessCAGR", "Beta", "TrackingError", "InformationRatio") if k in m})
    if "AvgVol" in m:
        print("Risk        :", {k: m[k] for k in ("AvgVol", "VolOfVol") if k in m})


if __name__ == "__main__":
    main()
