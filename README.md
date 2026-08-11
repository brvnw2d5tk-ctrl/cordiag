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

The `zpg` and `tg` commands accept a small YAML-style configuration file. See
their command help for the supported options.

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
