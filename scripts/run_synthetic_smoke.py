"""Run a deterministic, in-memory smoke check against cordiag's public APIs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from cordiag import tg, zpg


def _finite_statistic(value: float, name: str) -> float:
    """Return a JSON-safe statistic or fail loudly if the synthetic design breaks."""
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"synthetic {name} statistic is not finite")
    return float(np.round(value, 12))


def run_synthetic_smoke(seed: int = 42) -> dict[str, int | float]:
    """Compute deterministic zPG and TG statistics from 24 generated samples."""
    rng = np.random.default_rng(seed)
    n_per_condition = 12
    n_samples = n_per_condition * 2
    conditions = np.repeat(["control", "treated"], n_per_condition)
    design = pd.DataFrame(
        {"condition": conditions, "batch": np.repeat("batch_1", n_samples)}
    )

    latent = rng.normal(size=(n_samples, 3))
    treatment_shift = (conditions == "treated").astype(float)
    module_one = latent[:, :2] + treatment_shift[:, None] * np.array([0.2, -0.1])
    module_two = latent[:, 2:] + treatment_shift[:, None] * 0.15
    protein = 1.2 * latent[:, 0] - 0.7 * latent[:, 1] + rng.normal(
        scale=0.08, size=n_samples
    )

    source_mask = conditions == "control"
    target_mask = ~source_mask
    protein_source = 1.1 * latent[source_mask, 0] - 0.6 * latent[source_mask, 1]
    protein_target = 1.1 * latent[target_mask, 0] - 0.6 * latent[target_mask, 1]
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        warnings.filterwarnings(
            "ignore", message=r"R\^2 score is not well-defined with less than two samples"
        )
        warnings.filterwarnings(
            "ignore", message="One or more of the test scores are non-finite"
        )
        zpg_result = zpg.compute_zpg(
            {"module_one": module_one, "module_two": module_two},
            protein,
            design,
            n_perms=12,
            seed=seed,
            cv="loocv",
        )
        tg_result = tg.compute_tg_pair(
            protein="synthetic_protein",
            source_cond="control",
            target_cond="treated",
            batch="batch_1",
            P_source=protein_source,
            P_target=protein_target,
            X_source=latent[source_mask, :2],
            X_target=latent[target_mask, :2],
            source_strata=np.repeat("control_batch_1", n_per_condition),
            target_strata=np.repeat("treated_batch_1", n_per_condition),
            full_design=design,
            cv_alphas=[0.1, 1.0, 10.0],
            n_permutations=10,
            n_bootstrap=10,
            min_n=8,
            n_subsamples=3,
        )

    return {
        "seed": int(seed),
        "n_samples": n_samples,
        "zpg_statistic": _finite_statistic(zpg_result["zPG_rank"], "zPG"),
        "tg_statistic": _finite_statistic(tg_result.tg_raw, "TG"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_synthetic_smoke(seed=args.seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
