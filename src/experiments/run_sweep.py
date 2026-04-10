"""
Parameter sweep runner.

Usage:
    python -m src.experiments.run_sweep config/experiments/momentum_sweep.yaml
"""
import copy
import itertools
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _set_nested(d: dict, key_path: str, value) -> None:
    """Set a nested dict value in-place using dot notation (e.g. 'signal.lookback_months')."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _generate_variants(base_cfg: dict, grid: dict) -> list[dict]:
    """Return all config dicts from the Cartesian product of the parameter grid."""
    if not grid:
        return [copy.deepcopy(base_cfg)]

    keys = list(grid.keys())
    values = list(grid.values())
    variants = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)
        for key, val in zip(keys, combo):
            _set_nested(cfg, key, val)
        variants.append(cfg)
    return variants


def run_sweep(sweep_cfg_path: str | Path) -> list[dict]:
    """
    Run all parameter combinations defined in a sweep config YAML.

    Prices and benchmark returns are fetched once from the base config and
    reused across all variants. This assumes all variants share the same
    universe, date range, and pricing frequency — vary only signal and
    portfolio parameters in the grid.

    Returns a list of result dicts (one per variant).
    """
    from src.main import run_pipeline, load_prices_for_config, resolve_config_paths

    sweep_path = Path(sweep_cfg_path)
    if not sweep_path.is_absolute():
        sweep_path = _PROJECT_ROOT / sweep_path
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))

    base_cfg_path = Path(sweep["base_config"])
    if not base_cfg_path.is_absolute():
        base_cfg_path = _PROJECT_ROOT / base_cfg_path
    base_cfg_text = base_cfg_path.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(base_cfg_text)
    resolve_config_paths(base_cfg, _PROJECT_ROOT)

    for key_path, value in sweep.get("base_overrides", {}).items():
        _set_nested(base_cfg, key_path, value)

    grid = sweep.get("grid", {})
    variants = _generate_variants(base_cfg, grid)

    print(f"Loading prices for {len(variants)} variants (downloaded once)...")
    prices, prices_daily, benchmark_returns = load_prices_for_config(base_cfg)
    print(f"Prices loaded. Running {len(variants)} variants...\n")

    results = []
    for i, cfg in enumerate(variants, 1):
        sig = cfg.get("signal", {})
        port = cfg.get("portfolio", {})
        label = (
            f"[{i}/{len(variants)}] "
            f"lookback={sig.get('lookback_months')} "
            f"n_positions={port.get('n_positions')} "
            f"rebalance={port.get('rebalance')}"
        )
        print(label)
        try:
            result = run_pipeline(cfg, prices=prices, prices_daily=prices_daily, benchmark_returns=benchmark_returns)
            m = result["metrics"]
            print(
                f"  CAGR={m.get('CAGR'):.3f}  "
                f"Sharpe={m.get('Sharpe'):.3f}  "
                f"ExcessCAGR={m.get('ExcessCAGR', float('nan')):.3f}"
            )
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nSweep complete. {len(results)}/{len(variants)} variants succeeded.")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_sweep <sweep_config.yaml>")
        sys.exit(1)
    run_sweep(sys.argv[1])
