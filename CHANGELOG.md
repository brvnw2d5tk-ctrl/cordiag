# Changelog

## 0.1.3 - 2026-09-05

- Added `tg_cross_study_residual` as the canonical descriptive name for the
  second TG decomposition component; `tg_rna` remains available as a
  deprecated compatibility alias.
- Replaced causal RNA-coupling interpretation text with a cross-study residual
  description that does not assign a biological or technical cause.
- No TG numerical algorithm, decision rule, or frozen paper result changed.

## 0.1.2 - 2026-09-04

- Corrected the TG restricted-permutation documentation to describe the
  existing, tested matched-null estimator contract and to distinguish the
  fallback-null helpers.
- Added an explicit release contract confirming that a matched null calls the
  matched within, transfer, and crossed estimators for every realization.
- No TG numerical algorithm or frozen paper result changed from v0.1.1.

## 0.1.1 - 2026-08-14

- Added path-free paper parameter overlays with tested CLI/API names.
- Made unsupported configuration keys fail explicitly instead of being ignored.
- Aligned documentation with the executable zPG decision rule and the TG_raw
  decision scale without changing numerical algorithms or frozen results.

## 0.1.0 - 2026-08-11

- Initial public, source-only release of the zPG and TG core package.
- Added a deterministic, 24-sample synthetic smoke runner.
