# cordiag

`cordiag` provides two diagnostics for paired RNA and protein measurements:

- zPG (z-score of Paired Gain), a within-stratum permutation diagnostic for
  individual-level RNA-to-protein predictability.
- TG (Transportability Gap), a cross-condition diagnostic for the stability of
  RNA-to-protein prediction.

## Install

```bash
pip install .
```

## Command line

```bash
python -m cordiag --help
cordiag version
```

The `zpg` and `tg` commands accept a flat YAML-style configuration file. Every
input table must contain a unique `sample_id`; the three tables must name
exactly the same samples. Paths are resolved relative to the configuration
file. The design table additionally requires `condition` and `batch`.

The tracked files in [`examples/`](examples/) are purely synthetic and execute
the complete CSV-to-JSON interface. Unsupported configuration keys are rejected
so misspelled statistical parameters cannot be silently ignored.

## Fast synthetic demo settings

The examples are deliberately small, fast interface checks and are not a paper
rerun. `examples/zpg_synthetic.yaml` uses 8 permutations;
`examples/tg_synthetic.yaml` uses 4 permutations, 4 bootstrap resamples, and 3
matched-subsample repetitions.

```bash
cordiag zpg --config examples/zpg_synthetic.yaml
cordiag tg --config examples/tg_synthetic.yaml
```

## Paper frozen settings

The path-free overlays [`configs/paper_zpg.yaml`](configs/paper_zpg.yaml) and
[`configs/paper_tg.yaml`](configs/paper_tg.yaml) record the manuscript settings:

```yaml
# zPG
n_perms: 1000

# TG
n_permutations: 1000
n_bootstrap: 1000
n_subsamples: 20
```

These files record parameters only. They contain no study-data paths and are
not standalone paper rerun recipes. To apply an overlay, combine its parameter
keys with an authorized local configuration containing the required input
paths.

## Statistical contracts

The public zPG implementation uses stratum-conditioned LOOCV and does not
expose a public `groups` parameter. Group-aware CV is an implemented TG
capability through its `groups` argument.

The executable zPG decision order is fixed: NOT_IDENTIFIABLE_DESIGN takes
precedence; GO requires zPG > 1 and FDR < 0.1; INCONCLUSIVE is assigned to a
finite zPG > 0 that does not meet GO; and NO_GO covers all remaining cases.
The setting n < 12 is a low-confidence, exploratory setting, but sample size
does not change these executable classification semantics.

TG_raw is the primary decision-scale effect and classification scale. TG_log
is the scale-free reporting and ranking companion. The API field
`interpretation_primary` remains the primary classification field; its name
does not imply that TG_log is the primary statistical endpoint.

## Input fields and controlled-data boundary

For zPG, `rna_csv` contains one or more module-score columns and
`protein_column` selects the target column in `protein_csv`. For TG, the RNA
and protein tables must share one or more measurement-column names. Config
files use only one `key: value` entry per line; nested YAML is not supported.

The public repository contains only general software, tests, synthetic inputs,
the CLI, and parameter overlays. Controlled study matrices, frozen manuscript
results, source-data tables, paper figures, and submission files are maintained
outside GitHub under their applicable access controls. Consequently, GitHub
cannot independently reconstruct the manuscript's real-data results or the
complete paper; it verifies the public software interface and synthetic
execution boundary only.

## Synthetic smoke test

The repository contains no study data, generated outputs, figures, cached
results, or golden files. A self-contained check creates 24 samples in memory:

```bash
python scripts/run_synthetic_smoke.py --seed 42
```

Identical seeds produce identical JSON summaries. The script neither reads
project data nor uses environment variables.

## Scope

This public package supplies software only. The supported release boundary and
the data-free verification policy are specified in
[`REPRODUCIBILITY_SCOPE.md`](REPRODUCIBILITY_SCOPE.md).

## License

MIT; see [`LICENSE`](LICENSE).
