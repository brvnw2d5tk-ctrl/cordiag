# Changelog

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
