"""Command-line interface for data-free zPG and TG analyses."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


EXIT_OK = 0
EXIT_NOT_READY = 1
EXIT_USAGE = 2


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    lowered = token.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _parse_list(raw: str) -> List[Any]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [_parse_scalar(item) for item in raw.split(",") if item.strip()]


def parse_config(path: str) -> Dict[str, Any]:
    """Parse the documented flat ``key: value`` YAML subset."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")
    config: Dict[str, Any] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: expected 'key: value'")
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if not key:
                raise ValueError(f"{path}:{line_number}: empty key")
            config[key] = _parse_list(value) if value.startswith("[") and value.endswith("]") else _parse_scalar(value)
    return config


def _load_module(module_name: str):
    """Import a package module with a concise error for a missing package part."""
    try:
        return __import__(f"cordiag.{module_name}", fromlist=[module_name])
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("cordiag."):
            sys.stderr.write(f"[cordiag] unable to import cordiag.{module_name}: {exc}\n")
            raise SystemExit(EXIT_NOT_READY) from exc
        raise


def _load_data_config(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Load all required tables, resolving relative paths from the config file."""
    import pandas as pd

    config_dir = Path(config_path).resolve().parent
    loaded: Dict[str, Any] = {}
    for config_key, short_name in (
        ("rna_csv", "rna"),
        ("protein_csv", "protein"),
        ("design_csv", "design"),
    ):
        value = config.get(config_key)
        if value is None or not str(value).strip():
            raise ValueError(f"{config_key} is required")
        path = Path(str(value))
        if not path.is_absolute():
            path = config_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"{config_key}={path} not found")
        loaded[short_name] = pd.read_csv(path)
    return loaded


def _align_input_tables(data: Dict[str, Any]) -> Dict[str, Any]:
    """Require exact sample identity and align tables to the RNA row order."""
    for name, table in data.items():
        if "sample_id" not in table.columns:
            raise ValueError(f"{name}_csv must contain a sample_id column")
        if table["sample_id"].isna().any() or not table["sample_id"].is_unique:
            raise ValueError(f"{name}_csv sample_id values must be present and unique")

    reference_ids = data["rna"]["sample_id"].astype(str).tolist()
    reference_set = set(reference_ids)
    for name, table in data.items():
        ids = table["sample_id"].astype(str).tolist()
        if set(ids) != reference_set or len(ids) != len(reference_ids):
            raise ValueError(f"{name}_csv sample_id values must exactly match rna_csv")
        data[name] = (
            table.assign(sample_id=table["sample_id"].astype(str))
            .set_index("sample_id")
            .loc[reference_ids]
            .reset_index()
        )

    design = data["design"]
    for column in ("condition", "batch"):
        if column not in design.columns:
            raise ValueError(f"design_csv must contain a {column} column")
        if design[column].isna().any() or (design[column].astype(str).str.strip() == "").any():
            raise ValueError(f"design_csv {column} values must be present")
    return data


def _numeric_measurements(table, table_name: str) -> Dict[str, np.ndarray]:
    """Read finite numeric measurement columns while excluding ``sample_id``."""
    import pandas as pd

    columns = [column for column in table.columns if column != "sample_id"]
    if not columns:
        raise ValueError(f"{table_name}_csv must contain at least one measurement column")
    measurements: Dict[str, np.ndarray] = {}
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{table_name}_csv column {column!r} must be finite numeric values")
        measurements[str(column)] = values
    return measurements


def _json_value(value: Any) -> Any:
    """Convert NumPy objects and non-finite values to strict JSON values."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _run_zpg(args) -> int:
    config = parse_config(args.config)
    zpg = _load_module("zpg")
    data = _align_input_tables(_load_data_config(config, args.config))
    params = {key: value for key, value in config.items() if key in ("n_perms", "seed", "cv", "repeats") and value is not None}
    params.setdefault("n_perms", 200)
    params.setdefault("seed", 42)

    rna_modules = _numeric_measurements(data["rna"], "rna")
    protein_column = str(config.get("protein_column", ""))
    if not protein_column or protein_column not in data["protein"].columns:
        raise ValueError("protein_column must name a column in protein_csv")
    protein = _numeric_measurements(data["protein"], "protein")[protein_column]
    with contextlib.redirect_stdout(sys.stderr):
        result = zpg.compute_zpg(rna_modules, protein, data["design"], **params)
    print(json.dumps(_json_value({
        "analysis": "zpg",
        "n_samples": len(data["design"]),
        "seed": int(params["seed"]),
        "n_permutations": int(params["n_perms"]),
        "result": result,
    }), allow_nan=False, sort_keys=True))
    return EXIT_OK


def _run_tg(args) -> int:
    config = parse_config(args.config)
    tg = _load_module("tg")
    data = _align_input_tables(_load_data_config(config, args.config))
    params = {key: value for key, value in config.items() if key in (
        "n_permutations", "n_bootstrap", "seed", "min_n", "n_subsamples", "groups"
    ) and value is not None}
    params.setdefault("n_permutations", 100)
    params.setdefault("n_bootstrap", 50)
    params.setdefault("seed", 42)

    rna_data = _numeric_measurements(data["rna"], "rna")
    protein_data = _numeric_measurements(data["protein"], "protein")
    common = sorted(set(rna_data) & set(protein_data))
    if not common:
        raise ValueError("rna_csv and protein_csv must share at least one measurement column")
    with contextlib.redirect_stdout(sys.stderr):
        result = tg.compute_transportability_gap(
            {name: rna_data[name] for name in common},
            {name: protein_data[name] for name in common},
            data["design"],
            **params,
        )
    pairs = []
    for protein_name in sorted(result):
        for (source, target), pair in sorted(result[protein_name].items()):
            pairs.append({
                "protein": protein_name,
                "source_condition": source,
                "target_condition": target,
                "tg_raw": pair.tg_raw,
                "permutation_p_raw": pair.permutation_p_raw,
                "fdr_global": pair.fdr_global,
                "interpretation": pair.interpretation_primary,
            })
    print(json.dumps(_json_value({
        "analysis": "tg",
        "n_samples": len(data["design"]),
        "seed": int(params["seed"]),
        "n_permutations": int(params["n_permutations"]),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }), allow_nan=False, sort_keys=True))
    return EXIT_OK


def _cmd_version(args) -> int:
    import cordiag
    print(f"cordiag {cordiag.__version__}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cordiag",
        description="zPG and TG diagnostics for paired molecular measurements.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    zpg_parser = subparsers.add_parser("zpg", help="run zPG from synthetic or user-supplied CSV inputs")
    zpg_parser.add_argument("--config", required=True, help="flat YAML config with CSV paths")
    zpg_parser.set_defaults(func=_run_zpg)
    tg_parser = subparsers.add_parser("tg", help="run TG from synthetic or user-supplied CSV inputs")
    tg_parser.add_argument("--config", required=True, help="flat YAML config with CSV paths")
    tg_parser.set_defaults(func=_run_tg)
    version_parser = subparsers.add_parser("version", help="print package version")
    version_parser.set_defaults(func=_cmd_version)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"[cordiag] {exc}\n")
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
