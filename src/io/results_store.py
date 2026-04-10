from datetime import datetime
from pathlib import Path
import json
import pandas as pd


def create_run_dir(runs_dir: str | Path, run_name: str) -> tuple[Path, str]:
    """Create a timestamped run directory. Returns (run_dir, run_id)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def save_run_artifacts(
    run_dir: Path,
    config_text: str,
    metrics: dict,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    leaders: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    avg_pairwise_corr: pd.Series | None = None,
    contrib_to_vol: pd.DataFrame | None = None,
    holdings_stats: pd.DataFrame | None = None,
    stock_attribution: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    ic_series: pd.Series | None = None,
    ic_decay: pd.Series | None = None,
) -> None:
    """Persist config, metrics, weights, returns, leaders and risk artifacts to disk."""
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    weights.to_csv(run_dir / "weights.csv")
    returns.to_csv(run_dir / "returns.csv")
    leaders.to_csv(run_dir / "leaders.csv", index=False)
    if benchmark_returns is not None:
        benchmark_returns.to_csv(run_dir / "benchmark_returns.csv", header=True)
    if avg_pairwise_corr is not None:
        avg_pairwise_corr.to_csv(run_dir / "avg_pairwise_corr.csv", header=True)
    if contrib_to_vol is not None:
        contrib_to_vol.to_csv(run_dir / "contrib_to_vol.csv")
    if holdings_stats is not None:
        holdings_stats.to_csv(run_dir / "holdings_stats.csv")
    if stock_attribution is not None:
        stock_attribution.to_csv(run_dir / "stock_attribution.csv")
    if asset_returns is not None:
        asset_returns.to_csv(run_dir / "asset_returns.csv")
    if scores is not None:
        scores.to_csv(run_dir / "scores.csv")
    if ic_series is not None:
        ic_series.to_csv(run_dir / "ic_series.csv", header=True)
    if ic_decay is not None:
        ic_decay.to_csv(run_dir / "ic_decay.csv", header=True)