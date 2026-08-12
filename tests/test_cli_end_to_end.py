"""Data-free end-to-end contracts for the documented command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def _write_inputs(tmp_path: Path, *, mismatch: bool = False) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    n = 24
    sample_ids = [f"sample_{i:02d}" for i in range(n)]
    condition = np.repeat(["control", "treated"], n // 2)
    latent = rng.normal(size=(n, 3))

    rna = pd.DataFrame({"sample_id": sample_ids})
    protein = pd.DataFrame({"sample_id": sample_ids})
    for index, name in enumerate(("feature_a", "feature_b", "feature_c")):
        rna[name] = latent[:, index] + 0.15 * (condition == "treated")
        protein[name] = 0.9 * latent[:, index] + rng.normal(scale=0.05, size=n)
    zpg_protein = pd.DataFrame({
        "sample_id": sample_ids,
        "protein": 1.1 * latent[:, 0] - 0.6 * latent[:, 1] + rng.normal(scale=0.05, size=n),
    })
    design = pd.DataFrame({"sample_id": sample_ids, "condition": condition, "batch": "batch_1"})
    if mismatch:
        protein.loc[0, "sample_id"] = "wrong_sample"

    rna.to_csv(tmp_path / "rna.csv", index=False)
    protein.to_csv(tmp_path / "protein.csv", index=False)
    zpg_protein.to_csv(tmp_path / "zpg_protein.csv", index=False)
    design.to_csv(tmp_path / "design.csv", index=False)
    (tmp_path / "zpg.yaml").write_text(
        "rna_csv: rna.csv\nprotein_csv: zpg_protein.csv\ndesign_csv: design.csv\n"
        "protein_column: protein\nn_perms: 8\nseed: 42\ncv: loocv\n",
        encoding="utf-8",
    )
    (tmp_path / "tg.yaml").write_text(
        "rna_csv: rna.csv\nprotein_csv: protein.csv\ndesign_csv: design.csv\n"
        "n_permutations: 4\nn_bootstrap: 4\nn_subsamples: 3\nmin_n: 8\nseed: 42\n",
        encoding="utf-8",
    )
    return tmp_path / "zpg.yaml", tmp_path / "tg.yaml"


def _run_cli(command: str, config: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_root = env.get("CORDIAG_TEST_SOURCE_ROOT", str(PUBLIC_ROOT))
    if source_root:
        env["PYTHONPATH"] = source_root
    else:
        env.pop("PYTHONPATH", None)
    executable = env.get("CORDIAG_CLI_PYTHON", sys.executable)
    return subprocess.run(
        [executable, "-B", "-m", "cordiag", command, "--config", str(config)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_zpg_cli_runs_relative_config_from_another_working_directory(tmp_path):
    zpg_config, _ = _write_inputs(tmp_path)

    completed = _run_cli("zpg", zpg_config, tmp_path.parent)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["analysis"] == "zpg"
    assert result["n_samples"] == 24
    assert result["seed"] == 42
    assert result["n_permutations"] == 8


def test_tg_cli_is_deterministic_for_documented_synthetic_schema(tmp_path):
    _, tg_config = _write_inputs(tmp_path)

    first = _run_cli("tg", tg_config, tmp_path.parent)
    second = _run_cli("tg", tg_config, tmp_path.parent)

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert first.stdout == second.stdout
    result = json.loads(first.stdout)
    assert result["analysis"] == "tg"
    assert result["n_samples"] == 24
    assert result["seed"] == 42
    assert result["n_pairs"] > 0


def test_tg_cli_rejects_mismatched_sample_ids(tmp_path):
    _, tg_config = _write_inputs(tmp_path, mismatch=True)

    completed = _run_cli("tg", tg_config, tmp_path.parent)

    assert completed.returncode == 2
    assert "sample_id" in completed.stderr
    assert "Traceback" not in completed.stderr
