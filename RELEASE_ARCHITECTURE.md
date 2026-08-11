# Release architecture

## GitHub-only software release

This repository is the complete public GitHub software release for `cordiag`.
GitHub is the sole release surface: no Zenodo record, DOI archive, or external
artifact mirror is part of this release contract.

The committed tree is constrained by `GITHUB_RELEASE_ALLOWLIST.json`. The CI
workflow verifies that boundary before installation and testing, so a file that
is not explicitly allowlisted fails the public-release contract.

## What the release verifies

The repository distributes installable Python source, command-line access,
documentation, contract tests, and a deterministic in-memory synthetic smoke
runner. CI uses Python 3.12 to check the allowlist, run the test suite, execute
the smoke runner with seed 42, import the package, and display `cordiag --help`.

These checks establish a software reproducibility boundary: the released code
can be installed and its documented synthetic behavior can be re-executed from
GitHub.

## Data and manuscript boundary

No real study data, source-data tables, generated outputs, figures, caches,
environment snapshots, or golden regression artifacts are released here. Access
to real data remains separately governed and is not granted, packaged, or
implied by this repository.

Accordingly, this software release cannot reconstruct study-level analyses,
figures, numerical results, or manuscript conclusions. It supports inspection
and execution of the software only; reproducing the paper requires the
separately governed data and its study-specific analysis context.
