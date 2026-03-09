from pathlib import Path
import yaml
import pandas as pd

from src.data.universe import load_universe
from src.data.pricing import load_prices_yfinance, resample_prices
from src.signals.momentum import momentum_12_1
from src.portfolio.construction import select_top_n, equal_weight, rebalance_weights
from src.backtest.engine import backtest_long_only
from src.analytics.performance import calculate_performance
from src.io.results_store import create_run_dir, save_run_artifacts
from src.io.exports import current_leaders


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    cfg_path = Path("config/uk_smallcap.yaml")
    cfg_text = read_text(cfg_path)
    cfg = yaml.safe_load(cfg_text)

    # Universe
    universe_cfg = cfg["universe"]
    tickers = load_universe(universe_cfg["file"], universe_cfg.get("ticker_column", "ticker"))

    # Prices
    prices_daily = load_prices_yfinance(
        tickers=tickers,
        start=cfg["start_date"],
        end=cfg["end_date"],
        price_field=cfg["pricing"].get("price_field", "Adj Close"),
    )
    prices = resample_prices(prices_daily, frequency=cfg["pricing"].get("frequency", "M"))

    # Signal
    sig_cfg = cfg["signal"]
    scores = momentum_12_1(
        prices,
        lookback_months=sig_cfg.get("lookback_months", 12),
        skip_recent_months=sig_cfg.get("skip_recent_months", 1),
    )

    # Portfolio construction
    port_cfg = cfg["portfolio"]
    selection = select_top_n(scores, n=int(port_cfg.get("n_positions", 12)))
    w_raw = equal_weight(selection)
    weights = rebalance_weights(w_raw, rebalance=port_cfg.get("rebalance", "Q"))

    # Backtest + costs
    costs_cfg = cfg.get("costs", {})
    results = backtest_long_only(
        prices=prices,
        weights=weights,
        commission_bps=float(costs_cfg.get("commission_bps", 0.0)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 0.0)),
    )

    metrics = calculate_performance(results["portfolio_returns_net"])

    # Leaders export for Notion review
    last_date = scores.dropna(how="all").index.max()
    leaders = current_leaders(scores, asof_date=last_date, top_k=25)

    # Save run artifacts
    run_dir = create_run_dir(cfg["outputs"]["runs_dir"], run_name=f"{cfg['market']}_{sig_cfg['name']}_{port_cfg['rebalance']}")
    save_run_artifacts(
        run_dir=run_dir,
        config_text=cfg_text,
        metrics=metrics,
        weights=weights,
        returns=results,
        leaders=leaders,
    )

    print("Run saved to:", run_dir)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()