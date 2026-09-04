"""Contracts for paper parameter overlays and public CLI/API boundaries."""

from __future__ import annotations

import inspect
from pathlib import Path
import re

import pytest

from cordiag import cli, tg, zpg


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def test_paper_zpg_overlay_uses_the_cli_and_api_parameter_name():
    config = cli.parse_config(str(PUBLIC_ROOT / "configs" / "paper_zpg.yaml"))

    assert config == {"n_perms": 1000, "seed": 42, "cv": "loocv"}
    assert cli.validated_parameters("zpg", config) == config
    assert "n_perms" in inspect.signature(zpg.compute_zpg).parameters
    assert "groups" not in inspect.signature(zpg.compute_zpg).parameters


def test_paper_tg_overlay_uses_cli_and_api_parameter_names():
    config = cli.parse_config(str(PUBLIC_ROOT / "configs" / "paper_tg.yaml"))

    assert config == {
        "n_permutations": 1000,
        "n_bootstrap": 1000,
        "n_subsamples": 20,
        "seed": 42,
    }
    assert cli.validated_parameters("tg", config) == config
    signature = inspect.signature(tg.compute_transportability_gap).parameters
    assert {"n_permutations", "n_bootstrap", "n_subsamples", "groups"} <= set(signature)


@pytest.mark.parametrize("command", ["zpg", "tg"])
def test_cli_rejects_unknown_config_keys_instead_of_silently_ignoring_them(command):
    with pytest.raises(ValueError, match="unsupported .* config key"):
        cli.validated_parameters(command, {"permutations": 1000})


def test_release_metadata_is_fixed_at_0_1_2():
    pyproject = (PUBLIC_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (PUBLIC_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert re.search(r'^version = "0\.1\.2"$', pyproject, flags=re.MULTILINE)
    assert re.search(r"^version: 0\.1\.2$", citation, flags=re.MULTILINE)
    assert re.search(r"^date-released: 2026-09-04$", citation, flags=re.MULTILINE)
    assert __import__("cordiag").__version__ == "0.1.2"


def test_readme_separates_fast_demo_paper_settings_and_controlled_data():
    readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
    prose = " ".join(readme.split())

    for heading in (
        "## Fast synthetic demo settings",
        "## Paper frozen settings",
        "## Input fields and controlled-data boundary",
    ):
        assert heading in readme
    assert "not a paper rerun" in prose
    assert "cannot independently reconstruct the manuscript's real-data results" in prose
    assert "n_perms: 1000" in readme
    assert "n_permutations: 1000" in readme
    assert "n_bootstrap: 1000" in readme
    assert "n_subsamples: 20" in readme


def test_documented_zpg_decision_and_groups_contract_matches_executable_code():
    readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
    prose = " ".join(readme.split())

    assert "NOT_IDENTIFIABLE_DESIGN takes precedence" in prose
    assert "finite zPG > 0 that does not meet GO" in prose
    assert "n < 12 is a low-confidence, exploratory setting" in prose
    assert "Group-aware CV is an implemented TG capability" in prose
    assert "zPG implementation uses stratum-conditioned LOOCV and does not expose a public `groups` parameter" in prose


def test_tg_endpoint_terminology_does_not_change_interpretation_primary_api():
    readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
    tg_source = (PUBLIC_ROOT / "cordiag" / "tg.py").read_text(encoding="utf-8")
    readme_prose = " ".join(readme.split())
    tg_prose = " ".join(tg_source.split())

    required = "TG_raw is the primary decision-scale effect and classification scale"
    assert required in readme_prose
    assert "TG_log is the scale-free reporting and ranking companion" in readme_prose
    assert required in tg_prose
    assert "TG_log = primary" not in tg_source
    assert "TG_log (primary)" not in tg_source
    assert "interpretation_primary" in tg_source
