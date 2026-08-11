"""Contract tests for the GitHub-safe public release tree."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable

import pytest


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = PUBLIC_ROOT / "tools" / "verify_github_release_allowlist.py"
CI_WORKFLOW_PATH = PUBLIC_ROOT / ".github" / "workflows" / "core-ci.yml"


def load_verify_tree() -> Callable[[Path], list[str]]:
    spec = importlib.util.spec_from_file_location("github_release_allowlist", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_tree


def test_verifier_cli_accepts_allowlisted_tree_and_reports_rejected_file(tmp_path):
    (tmp_path / "GITHUB_RELEASE_ALLOWLIST.json").write_text(
        '{"include": ["GITHUB_RELEASE_ALLOWLIST.json"]}\n', encoding="utf-8"
    )

    allowed = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert allowed.returncode == 0
    assert allowed.stdout == ""

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "real.csv").write_text("x\n1\n", encoding="utf-8")

    rejected = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 1
    assert rejected.stdout == "data/real.csv\n"


def test_public_tree_rejects_real_data(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "real.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "GITHUB_RELEASE_ALLOWLIST.json").write_text(
        '{"include": ["GITHUB_RELEASE_ALLOWLIST.json"]}\n', encoding="utf-8"
    )

    verify_tree = load_verify_tree()

    assert verify_tree(tmp_path) == ["data/real.csv"]


def test_verifier_rejects_allowlisted_symlink(tmp_path):
    """An allowlisted name must not conceal content through a filesystem link."""
    (tmp_path / "GITHUB_RELEASE_ALLOWLIST.json").write_text(
        '{"include": ["GITHUB_RELEASE_ALLOWLIST.json", "cordiag/module.py"]}\n',
        encoding="utf-8",
    )
    controlled_file = tmp_path / "controlled-data.py"
    controlled_file.write_text("secret = True\n", encoding="utf-8")
    link = tmp_path / "cordiag" / "module.py"
    try:
        link.parent.mkdir()
        os.symlink(controlled_file, link)
    except OSError as exc:
        link.parent.rmdir()
        controlled_dir = tmp_path / "controlled-directory"
        controlled_dir.mkdir()
        (controlled_dir / "module.py").write_text("secret = True\n", encoding="utf-8")
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link.parent), str(controlled_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert junction.returncode == 0, (
            "filesystem denied both symlink and directory-junction contract setup: "
            f"symlink error={exc}; junction error={junction.stderr}"
        )

    verify_tree = load_verify_tree()

    assert any(
        problem.startswith("reparse point: cordiag")
        for problem in verify_tree(tmp_path)
    )


def test_public_tree_matches_explicit_allowlist():
    verify_tree = load_verify_tree()

    assert verify_tree(PUBLIC_ROOT) == []


def test_public_tree_has_no_data_or_output_directories():
    assert not (PUBLIC_ROOT / "data").exists()
    assert not (PUBLIC_ROOT / "output").exists()


def test_ci_runs_release_contract():
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "verify_github_release_allowlist.py" in workflow
    assert "run_synthetic_smoke.py --seed 42" in workflow
