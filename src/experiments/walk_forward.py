"""
Walk-forward validation runner.

Usage:
    python -m src.experiments.walk_forward config/experiments/walk_forward.yaml
"""
import copy
import sys
from datetime import datetime
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_walk_forward(wf_cfg_path: str | Path) -> dict:
    """
    1. Run the parameter grid sweep on the in-sample (IS) period.
    2. Select the top-N configs by selection_metric.
    3. Re-run those configs on the out-of-sample (OOS) period.
    4. Return IS results, OOS results, and a shared wf_group tag.

    Prices are downloaded once for the full date range and sliced — no
    redundant downloads.

    Returns dict with keys:
        is_results  : list of run dicts from IS sweep
        oos_results : list of run dicts for top-N IS configs on OOS
        wf_group    : str timestamp linking IS and OOS runs in the DB
    """
    from src.main import run_pipeline, load_prices_for_config, resolve_config_paths
    from src.experiments.run_sweep import _generate_variants

    wf_path = Path(wf_cfg_path)
    if not wf_path.is_absolute():
        wf_path = _PROJECT_ROOT / wf_path
    wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))

    base_cfg_path = Path(wf["base_config"])
    if not base_cfg_path.is_absolute():
        base_cfg_path = _PROJECT_ROOT / base_cfg_path
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    resolve_config_paths(base_cfg, _PROJECT_ROOT)

    is_end = str(wf["is_end_date"])
    oos_start = is_end
    selection_metric = wf.get("selection_metric", "sharpe")
    top_n = int(wf.get("top_n", 3))
    grid = wf.get("grid", {})
    wf_group = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Download full history once, then slice ─────────────────────────────
    print("Downloading full price history...")
    prices_full, prices_daily_full, bench_full = load_prices_for_config(base_cfg)

    prices_is = prices_full.loc[:is_end]
    prices_daily_is = prices_daily_full.loc[:is_end]
    bench_is = bench_full.loc[:is_end] if bench_full is not None else None

    prices_oos = prices_full.loc[oos_start:]
    prices_daily_oos = prices_daily_full.loc[oos_start:]
    bench_oos = bench_full.loc[oos_start:] if bench_full is not None else None

    # ── IS sweep ───────────────────────────────────────────────────────────
    is_base = copy.deepcopy(base_cfg)
    is_base["end_date"] = is_end
    variants = _generate_variants(is_base, grid)

    print(f"Running {len(variants)} IS variants...")
    is_results = []
    for i, cfg in enumerate(variants, 1):
        print(f"  IS variant {i}/{len(variants)}...")
        try:
            result = run_pipeline(
                cfg,
                prices=prices_is,
                prices_daily=prices_daily_is,
                benchmark_returns=bench_is,
                wf_group=wf_group,
                wf_period="is",
            )
            is_results.append(result)
        except Exception as e:
            print(f"  IS variant {i} failed: {e}")

    # ── Select top-N IS configs ────────────────────────────────────────────
    _metric_key = {
        "sharpe": "Sharpe",
        "cagr": "CAGR",
        "sortino": "Sortino",
        "excess_cagr": "ExcessCAGR",
        "information_ratio": "InformationRatio",
        "max_drawdown": "MaxDrawdown",
    }
    metric_key = _metric_key.get(selection_metric.lower(), selection_metric)
    ascending = selection_metric.lower() in {"max_drawdown", "maxdrawdown", "volatility"}
    is_results.sort(
        key=lambda r: r["metrics"].get(metric_key, float("-inf")),
        reverse=not ascending,
    )
    top_is = is_results[:top_n]

    print(f"\nTop {top_n} IS configs selected by {selection_metric}. Running OOS...")

    # ── OOS runs for top IS configs ────────────────────────────────────────
    oos_results = []
    for i, r in enumerate(top_is, 1):
        run_dir = Path(r["run_dir"])
        cfg_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
        oos_cfg = yaml.safe_load(cfg_text)
        resolve_config_paths(oos_cfg, _PROJECT_ROOT)
        oos_cfg["start_date"] = oos_start
        # end_date stays as base_cfg["end_date"] (already in the saved config)
        print(f"  OOS run {i}/{len(top_is)} (IS Sharpe={r['metrics'].get('Sharpe', float('nan')):.3f})...")
        try:
            result = run_pipeline(
                oos_cfg,
                prices=prices_oos,
                prices_daily=prices_daily_oos,
                benchmark_returns=bench_oos,
                wf_group=wf_group,
                wf_period="oos",
            )
            result["is_run_id"] = r["run_id"]
            oos_results.append(result)
        except Exception as e:
            print(f"  OOS run {i} failed: {e}")

    print(f"\nWalk-forward complete — {len(is_results)} IS variants, {len(oos_results)} OOS runs.")
    return {"is_results": is_results, "oos_results": oos_results, "wf_group": wf_group}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.walk_forward <walk_forward_config.yaml>")
        sys.exit(1)
    results = run_walk_forward(sys.argv[1])
    print(f"\nGroup: {results['wf_group']}")
    print(f"IS runs : {len(results['is_results'])}")
    print(f"OOS runs: {len(results['oos_results'])}")
    if results["oos_results"]:
        print("\nOOS results (top configs):")
        for r in results["oos_results"]:
            m = r["metrics"]
            print(
                f"  {r['run_id']}  "
                f"CAGR={m.get('CAGR', float('nan')):.3f}  "
                f"Sharpe={m.get('Sharpe', float('nan')):.3f}"
            )
