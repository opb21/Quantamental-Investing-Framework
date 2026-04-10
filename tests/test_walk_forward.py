"""Tests for walk-forward validation runner."""
import copy
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.experiments.run_sweep import _generate_variants


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_base_cfg(start="2015-01-01", end="2024-01-01"):
    return {
        "market": "test",
        "start_date": start,
        "end_date": end,
        "universe": {"file": "data/universe.csv"},
        "pricing": {"frequency": "M"},
        "signal": {"name": "momentum", "lookback_months": 12, "skip_recent_months": 1},
        "portfolio": {"n_positions": 20, "rebalance": "Q", "weighting": "equal", "n_tranches": 1},
        "costs": {"commission_bps": 0, "slippage_bps": 0},
        "outputs": {"runs_dir": "reports/runs", "db_path": "reports/runs.db"},
    }


def _make_mock_result(run_id, sharpe=1.0, cagr=0.1):
    return {
        "run_id": run_id,
        "run_dir": f"reports/runs/{run_id}",
        "metrics": {"Sharpe": sharpe, "CAGR": cagr, "MaxDrawdown": -0.2},
    }


# ── Tests for _generate_variants (used internally by walk_forward) ─────────────

def test_generate_variants_count():
    base = _make_base_cfg()
    grid = {
        "signal.lookback_months": [6, 9, 12],
        "portfolio.n_positions": [10, 20],
    }
    variants = _generate_variants(base, grid)
    assert len(variants) == 6


def test_generate_variants_values():
    base = _make_base_cfg()
    grid = {"signal.lookback_months": [6, 12]}
    variants = _generate_variants(base, grid)
    lookbacks = [v["signal"]["lookback_months"] for v in variants]
    assert sorted(lookbacks) == [6, 12]


def test_generate_variants_does_not_mutate_base():
    base = _make_base_cfg()
    original_lookback = base["signal"]["lookback_months"]
    grid = {"signal.lookback_months": [6]}
    _generate_variants(base, grid)
    assert base["signal"]["lookback_months"] == original_lookback


# ── Tests for run_walk_forward ─────────────────────────────────────────────────

def _build_mock_prices():
    idx = pd.date_range("2015-01-01", "2024-01-01", freq="ME")
    return pd.DataFrame({"A": 1.0, "B": 2.0}, index=idx)


def _write_mock_config(tmp_path, base_cfg_path, is_end="2020-01-01", top_n=2):
    wf_yaml = tmp_path / "wf.yaml"
    wf_yaml.write_text(
        f"base_config: '{base_cfg_path}'\n"
        f"is_end_date: '{is_end}'\n"
        f"selection_metric: 'sharpe'\n"
        f"top_n: {top_n}\n"
        "grid:\n"
        "  signal.lookback_months: [6, 12]\n"
    )
    return str(wf_yaml)


def test_walk_forward_returns_structure(tmp_path):
    """Return dict must have is_results, oos_results, wf_group."""
    base_cfg = _make_base_cfg()
    base_path = tmp_path / "base.yaml"
    import yaml
    base_path.write_text(yaml.dump(base_cfg))

    prices = _build_mock_prices()
    prices_daily = prices.copy()

    call_count = {"n": 0}

    def mock_run_pipeline(cfg, **kwargs):
        call_count["n"] += 1
        run_id = f"run_{call_count['n']:04d}"
        # Write a minimal config.yaml so OOS can re-read it
        import os
        run_dir = str(tmp_path / run_id)
        os.makedirs(run_dir, exist_ok=True)
        (tmp_path / run_id / "config.yaml").write_text(yaml.dump(cfg))
        return {"run_id": run_id, "run_dir": run_dir, "metrics": {"Sharpe": 1.0, "CAGR": 0.1}}

    wf_cfg_path = _write_mock_config(tmp_path, str(base_path), top_n=2)

    with (
        patch("src.experiments.walk_forward.load_prices_for_config", return_value=(prices, prices_daily, None)),
        patch("src.experiments.walk_forward.resolve_config_paths"),
        patch("src.experiments.walk_forward.run_pipeline", side_effect=mock_run_pipeline),
    ):
        from src.experiments.walk_forward import run_walk_forward
        result = run_walk_forward(wf_cfg_path)

    assert "is_results" in result
    assert "oos_results" in result
    assert "wf_group" in result
    assert isinstance(result["wf_group"], str)
    assert len(result["wf_group"]) > 0


def test_walk_forward_top_n_respected(tmp_path):
    """OOS runs should be <= top_n."""
    base_cfg = _make_base_cfg()
    base_path = tmp_path / "base.yaml"
    import yaml
    base_path.write_text(yaml.dump(base_cfg))

    prices = _build_mock_prices()
    prices_daily = prices.copy()
    call_count = {"n": 0}

    def mock_run_pipeline(cfg, **kwargs):
        call_count["n"] += 1
        run_id = f"run_{call_count['n']:04d}"
        import os
        run_dir = str(tmp_path / run_id)
        os.makedirs(run_dir, exist_ok=True)
        (tmp_path / run_id / "config.yaml").write_text(yaml.dump(cfg))
        return {"run_id": run_id, "run_dir": run_dir, "metrics": {"Sharpe": float(call_count["n"]), "CAGR": 0.1}}

    # Grid has 3 variants, top_n=2
    wf_cfg_path = _write_mock_config(tmp_path, str(base_path), top_n=2)

    with (
        patch("src.experiments.walk_forward.load_prices_for_config", return_value=(prices, prices_daily, None)),
        patch("src.experiments.walk_forward.resolve_config_paths"),
        patch("src.experiments.walk_forward.run_pipeline", side_effect=mock_run_pipeline),
    ):
        from importlib import reload
        import src.experiments.walk_forward as wf_mod
        reload(wf_mod)
        result = wf_mod.run_walk_forward(wf_cfg_path)

    assert len(result["oos_results"]) <= 2


def test_walk_forward_oos_dates_correct(tmp_path):
    """OOS configs should have start_date == is_end_date."""
    base_cfg = _make_base_cfg()
    base_path = tmp_path / "base.yaml"
    import yaml
    base_path.write_text(yaml.dump(base_cfg))

    prices = _build_mock_prices()
    prices_daily = prices.copy()

    captured_cfgs = []
    call_count = {"n": 0}

    def mock_run_pipeline(cfg, **kwargs):
        call_count["n"] += 1
        run_id = f"run_{call_count['n']:04d}"
        import os
        run_dir = str(tmp_path / run_id)
        os.makedirs(run_dir, exist_ok=True)
        (tmp_path / run_id / "config.yaml").write_text(yaml.dump(cfg))
        if kwargs.get("wf_period") == "oos":
            captured_cfgs.append(copy.deepcopy(cfg))
        return {"run_id": run_id, "run_dir": run_dir, "metrics": {"Sharpe": 1.0, "CAGR": 0.1}}

    is_end = "2020-01-01"
    wf_cfg_path = _write_mock_config(tmp_path, str(base_path), is_end=is_end, top_n=1)

    with (
        patch("src.experiments.walk_forward.load_prices_for_config", return_value=(prices, prices_daily, None)),
        patch("src.experiments.walk_forward.resolve_config_paths"),
        patch("src.experiments.walk_forward.run_pipeline", side_effect=mock_run_pipeline),
    ):
        import importlib, src.experiments.walk_forward as wf_mod
        importlib.reload(wf_mod)
        wf_mod.run_walk_forward(wf_cfg_path)

    assert len(captured_cfgs) >= 1
    for cfg in captured_cfgs:
        assert str(cfg["start_date"]) == is_end
