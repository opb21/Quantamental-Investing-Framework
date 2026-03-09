from datetime import datetime
from pathlib import Path
import json
import pandas as pd


def create_run_dir(runs_dir: str | Path, run_name: str) -> Path:
    """Create a timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    config_text: str,
    metrics: dict,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    leaders: pd.DataFrame,
) -> None:
    """Persist config, metrics, weights, returns, leaders to disk."""
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    weights.to_csv(run_dir / "weights.csv")
    returns.to_csv(run_dir / "returns.csv")
    leaders.to_csv(run_dir / "leaders.csv", index=False)