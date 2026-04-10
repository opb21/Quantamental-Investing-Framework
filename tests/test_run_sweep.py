import copy
import pytest

from src.experiments.run_sweep import _set_nested, _generate_variants


def test_set_nested_top_level():
    d = {"signal": {"lookback_months": 12}}
    _set_nested(d, "signal.lookback_months", 6)
    assert d["signal"]["lookback_months"] == 6


def test_set_nested_creates_missing_keys():
    d = {}
    _set_nested(d, "portfolio.rebalance", "M")
    assert d["portfolio"]["rebalance"] == "M"


def test_generate_variants_count():
    base = {"signal": {"lookback_months": 12}, "portfolio": {"n_positions": 12, "rebalance": "Q"}}
    grid = {"signal.lookback_months": [6, 9, 12], "portfolio.rebalance": ["M", "Q"]}
    variants = _generate_variants(base, grid)
    assert len(variants) == 6  # 3 × 2


def test_generate_variants_values_applied():
    base = {"signal": {"lookback_months": 12}}
    grid = {"signal.lookback_months": [6, 9]}
    variants = _generate_variants(base, grid)
    lookbacks = [v["signal"]["lookback_months"] for v in variants]
    assert set(lookbacks) == {6, 9}


def test_generate_variants_does_not_mutate_base():
    base = {"signal": {"lookback_months": 12}}
    grid = {"signal.lookback_months": [6, 9]}
    original = copy.deepcopy(base)
    _generate_variants(base, grid)
    assert base == original


def test_generate_variants_empty_grid_returns_base():
    base = {"signal": {"lookback_months": 12}}
    variants = _generate_variants(base, {})
    assert len(variants) == 1
    assert variants[0]["signal"]["lookback_months"] == 12
