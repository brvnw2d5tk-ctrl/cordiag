"""Verify that a public GitHub release tree contains only allowlisted files."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path


ALLOWLIST_FILENAME = "GITHUB_RELEASE_ALLOWLIST.json"


def _is_reparse_point(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse point."""
    mode = path.lstat().st_mode
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return stat.S_ISLNK(mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def verify_tree(root: Path) -> list[str]:
    """Return release-boundary violations for *root* and its explicit allowlist."""
    root = Path(root)
    if _is_reparse_point(root):
        return ["reparse point: ."]

    allowlist_path = root / ALLOWLIST_FILENAME
    if _is_reparse_point(allowlist_path):
        return [f"reparse point: {ALLOWLIST_FILENAME}"]
    allowed = set(
        json.loads(allowlist_path.read_text(encoding="utf-8"))["include"]
    )
    found: set[str] = set()
    problems: list[str] = []

    def inspect(directory: Path) -> None:
        for path in directory.iterdir():
            relative_path = path.relative_to(root).as_posix()
            # Git metadata is not part of the versioned release tree.
            if directory == root and relative_path == ".git":
                continue
            if _is_reparse_point(path):
                problems.append(f"reparse point: {relative_path}")
                continue
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                inspect(path)
            elif stat.S_ISREG(mode):
                found.add(relative_path)
            else:
                problems.append(f"non-regular file: {relative_path}")

    inspect(root)
    problems.extend(sorted(found - allowed))
    problems.extend(f"missing allowlisted file: {path}" for path in sorted(allowed - found))
    return sorted(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()

    unexpected = verify_tree(args.repo_root)
    for path in unexpected:
        print(path)
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
