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


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep copy of cfg with dot-notation overrides applied."""
    cfg = copy.deepcopy(cfg)
    for key_path, value in overrides.items():
        _set_nested(cfg, key_path, value)
    return cfg


def _generate_variants(base_cfg: dict, grid: dict, variants: list | None = None) -> list[dict]:
    """Return all config dicts to run.

    If ``variants`` is provided, each entry is a named override set — a dict
    of dot-notation key paths plus an optional ``label`` key.  The label is
    written to ``strategy_name`` so it appears in the run DB.

    If ``grid`` is also provided, every variant is crossed with the full
    Cartesian product of the grid (variants × grid combinations).

    If only ``grid`` is provided (legacy behaviour), the Cartesian product is
    applied to the single base config.
    """
    # Build per-variant base configs
    if variants:
        base_cfgs = []
        for v in variants:
            overrides = {k: val for k, val in v.items() if k != "label"}
            cfg = _apply_overrides(base_cfg, overrides)
            if "label" in v:
                cfg["strategy_name"] = v["label"]
            base_cfgs.append(cfg)
    else:
        base_cfgs = [copy.deepcopy(base_cfg)]

    if not grid:
        return base_cfgs

    # Cross each base config with the Cartesian product of the grid
    keys = list(grid.keys())
    values = list(grid.values())
    result = []
    for b_cfg in base_cfgs:
        for combo in itertools.product(*values):
            cfg = copy.deepcopy(b_cfg)
            for key, val in zip(keys, combo):
                _set_nested(cfg, key, val)
            result.append(cfg)
    return result


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
    named_variants = sweep.get("variants")
    all_variants = _generate_variants(base_cfg, grid, named_variants)

    print(f"Loading prices for {len(all_variants)} variants (downloaded once)...")
    prices, prices_daily, benchmark_returns = load_prices_for_config(base_cfg)
    print(f"Prices loaded. Running {len(all_variants)} variants...\n")

    results = []
    for i, cfg in enumerate(all_variants, 1):
        sig = cfg.get("signal", {})
        port = cfg.get("portfolio", {})
        strategy_name = cfg.get("strategy_name", "")
        label = (
            f"[{i}/{len(all_variants)}] "
            + (f"{strategy_name}  " if strategy_name else "")
            + f"n={port.get('n_positions')}  "
            + f"rebalance={port.get('rebalance')}"
        )
        print(label)
        try:
            result = run_pipeline(cfg, prices=prices, prices_daily=prices_daily, benchmark_returns=benchmark_returns)
            m = result["metrics"]
            print(
                f"  CAGR={m.get('CAGR', float('nan')):.3f}  "
                f"Sharpe={m.get('Sharpe', float('nan')):.3f}  "
                f"ExcessCAGR={m.get('ExcessCAGR', float('nan')):.3f}"
            )
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nSweep complete. {len(results)}/{len(all_variants)} variants succeeded.")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_sweep <sweep_config.yaml>")
        sys.exit(1)
    run_sweep(sys.argv[1])
