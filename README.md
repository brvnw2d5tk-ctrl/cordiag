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
the complete CSV-to-JSON interface:

```bash
cordiag zpg --config examples/zpg_synthetic.yaml
cordiag tg --config examples/tg_synthetic.yaml
```

For zPG, `rna_csv` contains one or more module-score columns and
`protein_column` selects the target column in `protein_csv`. For TG, the RNA
and protein tables must share one or more measurement-column names. Config
files use only one `key: value` entry per line; nested YAML is not supported.

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
