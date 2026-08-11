"""cordiag 命令行接口 (argparse 骨架).

命令:
    cordiag zpg --config config.yaml    — 跑 zPG 诊断 (调 zpg.compute_zpg / compute_rank_zPG)
    cordiag tg  --config config.yaml    — 跑 TG 诊断 (调 tg.compute_tg_pair 家族)
    cordiag version                     — 打印版本

config.yaml 为最小实现解析 (见 _parse_config): 平铺 ``key: value`` 行,
支持 # 注释 / 引号字符串 / int / float / bool / 逗号或中括号列表。

zpg.py / tg.py modules: 模块或入口函数未就位时,
CLI 给出明确报错 (exit 1), 而不是栈回溯。
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Any, Dict, List

EXIT_OK = 0
EXIT_NOT_READY = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------- #
# 最小 YAML 子集解析 (不引入 pyyaml 依赖)
# --------------------------------------------------------------------------- #

def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return None
    if token[0] == '"' and token[-1] == '"' or token[0] == "'" and token[-1] == "'":
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_list(raw: str) -> List[Any]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [_parse_scalar(x) for x in raw.split(",") if x.strip()]


def parse_config(path: str) -> Dict[str, Any]:
    """最小 config 解析: 平铺 ``key: value`` (YAML 子集, 支持列表/注释)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")
    cfg: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{lineno}: expected 'key: value', got: {line!r}")
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if not key:
                raise ValueError(f"{path}:{lineno}: empty key")
            if value.startswith("[") and value.endswith("]"):
                cfg[key] = _parse_list(value)
            else:
                cfg[key] = _parse_scalar(value)
    return cfg


# --------------------------------------------------------------------------- #
# 子命令实现
# --------------------------------------------------------------------------- #

def _load_module(modname: str):
    """Import ``cordiag.<modname>`` and report a concise import failure."""
    try:
        return __import__(f"cordiag.{modname}", fromlist=[modname])
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name.startswith("cordiag."):
            sys.stderr.write(
                f"[cordiag] unable to import cordiag.{modname}: {exc}\n"
                f"  Verify that the installed package includes cordiag/{modname}.py.\n"
            )
            raise SystemExit(EXIT_NOT_READY) from exc
        raise  # missing third-party dependency — surface the real traceback


def _call_with_matching_kwargs(func, kwargs: Dict[str, Any]) -> Any:
    """Call func passing only the kwargs its signature accepts."""
    try:
        sig = inspect.signature(func)
        accepted = set(sig.parameters)
    except (TypeError, ValueError):
        return func(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return func(**filtered)


def _find_entry(module, names: List[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _load_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load rna_csv / protein_csv / design_csv from config into DataFrames (optional)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(EXIT_NOT_READY) from exc
    loaded: Dict[str, Any] = {}
    for key, col in (("rna_csv", "rna"), ("protein_csv", "protein"), ("design_csv", "design")):
        path = cfg.get(key)
        if path:
            if not os.path.isfile(str(path)):
                raise SystemExit(f"[cordiag] {key}={path} not found (exit {EXIT_USAGE})")
            loaded[col] = pd.read_csv(path)
    return loaded


def _run_zpg(args) -> int:
    cfg = parse_config(args.config)
    zpg = _load_module("zpg")
    entry = _find_entry(zpg, ["compute_zpg", "compute_rank_zPG"])
    if entry is None:
        sys.stderr.write(
            f"[cordiag] cordiag.zpg exists but neither compute_zpg nor "
            f"compute_rank_zPG is defined but could not be resolved.\n"
        )
        return EXIT_NOT_READY

    data = _load_data_config(cfg)
    params = {k: v for k, v in cfg.items()
              if k in ("n_perms", "seed", "n_folds") and v is not None}
    params.setdefault("n_perms", 200)
    params.setdefault("seed", 42)

    if not data:
        print("[cordiag] zpg config (data not provided — supply rna_csv/protein_csv/design_csv "
              "in config to run):")
        print(f"  n_perms={params['n_perms']}  seed={params['seed']}")
        print(f"  other keys: {', '.join(sorted(k for k in cfg if k not in params)) or '(none)'}")
        return EXIT_OK

    result = _call_with_matching_kwargs(entry, {**data, **params})
    print("[cordiag] zpg result:")
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"  {k}: {v!r}")
    else:
        print(f"  {result!r}")
    return EXIT_OK


def _run_tg(args) -> int:
    cfg = parse_config(args.config)
    tg = _load_module("tg")
    entry = _find_entry(tg, ["compute_tg_pair", "compute_transportability_gap"])
    if entry is None:
        sys.stderr.write(
            f"[cordiag] cordiag.tg exists but neither compute_tg_pair nor "
            f"compute_transportability_gap is defined but could not be resolved.\n"
        )
        return EXIT_NOT_READY

    data = _load_data_config(cfg)
    params = {k: v for k, v in cfg.items()
              if k in ("n_permutations", "n_bootstrap", "seed", "min_n",
                       "n_subsamples", "batch", "strata") and v is not None}
    params.setdefault("n_permutations", 100)
    params.setdefault("n_bootstrap", 50)
    params.setdefault("seed", 42)
    if "strata" in cfg and isinstance(cfg["strata"], list):
        params["strata"] = cfg["strata"]

    if not data:
        print("[cordiag] tg config (data not provided — supply rna_csv/protein_csv/design_csv "
              "in config to run):")
        print(f"  n_permutations={params['n_permutations']}  n_bootstrap={params['n_bootstrap']}")
        print(f"  other keys: {', '.join(sorted(k for k in cfg if k not in params)) or '(none)'}")
        return EXIT_OK

    result = _call_with_matching_kwargs(entry, {**data, **params})
    print("[cordiag] tg result:")
    print(f"  {result!r}")
    return EXIT_OK


def _cmd_version(args) -> int:
    import cordiag
    print(f"cordiag {cordiag.__version__}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cordiag",
        description="cordial diagnostics — zPG + TG: RNA→protein translation "
                    "transferability across conditions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_zpg = sub.add_parser("zpg", help="run the zPG diagnostic")
    p_zpg.add_argument("--config", required=True,
                       help="config.yaml with n_perms/seed/strata and optional "
                            "rna_csv/protein_csv/design_csv")
    p_zpg.set_defaults(func=_run_zpg)

    p_tg = sub.add_parser("tg", help="run the TG diagnostic")
    p_tg.add_argument("--config", required=True,
                      help="config.yaml with n_permutations/n_bootstrap/seed/strata/"
                           "batch and optional rna_csv/protein_csv/design_csv")
    p_tg.set_defaults(func=_run_tg)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[cordiag] {exc}\n")
        return EXIT_USAGE
    except ValueError as exc:
        sys.stderr.write(f"[cordiag] config error: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:
        raise
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
