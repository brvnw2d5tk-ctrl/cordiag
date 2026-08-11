# Reproducibility scope

This repository is a GitHub-safe software release. It contains package source,
documentation, tests, and a deterministic synthetic smoke runner only.

It intentionally excludes study data, source-data tables, generated outputs,
figures, caches, environment snapshots, and golden regression artifacts. The
smoke runner constructs all 24 observations in memory with
`numpy.random.default_rng(seed)` and does not read files, environment
variables, or project-specific paths.

The public release can verify installation, import, command-line discovery,
and deterministic execution on synthetic inputs. Reproducing any study-level
result requires separately governed data access and is outside this release.
