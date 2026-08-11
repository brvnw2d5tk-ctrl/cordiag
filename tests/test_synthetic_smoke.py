"""Synthetic, data-free public-package smoke coverage."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_runner():
    script_path = PUBLIC_ROOT / "scripts" / "run_synthetic_smoke.py"
    assert script_path.is_file(), "synthetic smoke runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_synthetic_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_synthetic_smoke


def test_synthetic_smoke_is_deterministic():
    run_synthetic_smoke = _load_smoke_runner()
    first = run_synthetic_smoke(seed=42)
    second = run_synthetic_smoke(seed=42)

    assert first == second
    assert set(first) == {"seed", "n_samples", "zpg_statistic", "tg_statistic"}
