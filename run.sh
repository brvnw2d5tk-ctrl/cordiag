#!/usr/bin/env sh
set -eu

PYTHON="${PYTHON:-python}"

"$PYTHON" -m pytest -p no:cacheprovider tests
"$PYTHON" scripts/run_synthetic_smoke.py --seed 42
