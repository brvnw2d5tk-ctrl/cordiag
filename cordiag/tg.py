"""Transportability-gap statistics for paired molecular measurements."""
import contextlib
import functools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings


# ═══════════════════════════════════════════════════════════════════
# Shared M1 primitives
#
# These primitives come from cordiag.m1 so TG uses one prediction and seeding
# implementation:
#   m0_stratum_means_loocv → _compute_stratum_means_loocv
#   m1_loocv               → _m1_loocv with the TG fallback policy
#   m1_train_test          → _m1_train_test
#   ridge_edf              → _ridge_edf
#   _spearmanr             → _spearmanr
# Underscore-prefixed aliases remain available to the calibration module.
# ═══════════════════════════════════════════════════════════════════
from cordiag.m1 import (
    m0_stratum_means_loocv as _compute_stratum_means_loocv,
    m1_loocv as _m1_loocv_impl,
    m1_train_test as _m1_train_test,
    ridge_edf as _ridge_edf,
    _spearmanr,
    derive_seed,
    subsample_seed,
)


# ═══════════════════════════════════════════════════════════════════
# Warning scope: suppress only warnings emitted inside numerical helpers.
#
# RuntimeWarning and FutureWarning messages from numerical dependencies are
# suppressed only during internal calculations. Package UserWarning messages
# remain visible.
# ═══════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_internal_warnings():
    """Scope warning suppression to internal library calls.

    RuntimeWarning and FutureWarning messages are suppressed while internal
    computation runs; package UserWarning messages remain visible.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        yield


def _quiet_internal(func):
    """Decorator: run func under :func:`_suppress_internal_warnings`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _suppress_internal_warnings():
            return func(*args, **kwargs)
    return wrapper


def _m1_loocv(
    P: np.ndarray,
    X: np.ndarray,
    strata: np.ndarray,
    cv_alphas: List[float],
    eval_indices: Optional[np.ndarray] = None,
    fixed_alpha: Optional[float] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Run shared M1 LOOCV with the TG fallback policy for unseen strata."""
    return _m1_loocv_impl(
        P, X, strata, cv_alphas,
        eval_indices=eval_indices,
        fixed_alpha=fixed_alpha,
        groups=groups,
        unseen_stratum='fallback',
    )


# ═══════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TGResult:
    """Transportability Gap for one protein, one source→target pair."""

    # Identity
    protein: str
    source_condition: str
    target_condition: str
    batch: str

    # Sample info
    n_source: int
    n_target: int
    size_ratio: float
    size_ratio_directional: float         # n_source / max(n_target, 1) — for asymmetry flag
    size_ratio_symmetric: float           # max(n_source, n_target) / min(...) (>= 1)

    # TG_raw is the primary decision-scale effect and classification scale;
    # TG_log is its scale-free reporting and ranking companion.
    q2_within_b: float                # Q² within target condition
    q2_a_to_b: float                  # Q² train on source → predict on target
    tg_raw: float                     # PRIMARY: Q²_within_b - Q²_a_to_b
    tg_relative: float                # tg_raw / max(q2_within_b, 0.01)

    # Decomposed TG
    q2_crossed: float                 # Q² from pooled a+b training
    tg_design: float                  # Q²_within_b - Q²_crossed (stratum shift)
    tg_rna: float                     # Deprecated storage name for the cross-study residual
    tg_design_fraction: float         # tg_design / max(tg_raw, 1e-10)

    # Stratum baseline
    mse_stratum_b: float

    # Statistical significance
    permutation_p_raw: float
    permutation_p_design: float
    permutation_p_rna: float
    interaction_pvalue: float = 1.0    # condition-label permutation interaction P value (supplementary, report-only)
    ztg: float = 0.0                    # standardized TG_raw
    tg_log: float = 0.0                 # Reporting/ranking: log cross/within MSE

    # Bootstrap CI (B=1000)
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    # FDR
    fdr_per_protein: float = 1.0      # BH across pairs within this protein
    fdr_global: float = 1.0           # BH across all proteins × pairs

    # Quality flags
    estimable: bool = True
    weak_baseline: bool = False
    asymmetric: bool = False
    ridge_edf: float = 0.0            # effective df of Ridge (avg over LOOCV folds)
    cramers_v: float = 0.0            # condition × batch association
    js_divergence: float = 0.0        # stratum distribution divergence

    # Interpretation
    interpretation_primary: str = ''  # TRANSPORTABLE | PARTIALLY | NON | NOT_ESTIMABLE
    interpretation_secondary: str = ''  # STRATUM_SHIFT | CROSS_STUDY_RESIDUAL | MIXED | ...
    interpretation_text: str = ''     # Human-readable summary

    # Repro
    seed: int = 42
    n_perms: int = 100
    n_bootstrap: int = 1000

    # Methodology flags (defaults last — dataclass ordering requirement)
    group_aware: bool = False         # Whether group-aware LOOCV was used
    ridge_alpha_within_b: float = 1.0  # Alpha used for within-b LOOCV, passed to crossed
    cv_mode: str = 'loocv'            # 'matched_subsample' | 'loocv' — CV strategy for within-b
    n_subsamples: int = 0             # Number of subsample reps (0 when LOOCV)

    @property
    def tg_cross_study_residual(self) -> float:
        """Return the descriptive cross-study residual.

        ``tg_rna`` is retained as the stored field for backward compatibility.
        Neither name assigns a biological or technical cause to the residual.
        """
        return self.tg_rna


@dataclass
class TGMatrixResult:
    """Aggregated TG results across all condition pairs for one protein."""
    protein: str
    conditions: List[str]
    matrix: Dict[Tuple[str, str], TGResult]  # (source, target) → result
    n_valid_pairs: int
    n_transportable: int
    n_non_transportable: int
    mean_tg_raw: float
    max_tg_raw: float
    worst_pair: Tuple[str, str]


@dataclass
class TGEnsembleResult:
    """Aggregated TG across all proteins."""
    proteins: List[str]
    conditions: List[str]
    per_protein: Dict[str, TGMatrixResult]
    # Cross-protein summary
    mean_matrix: np.ndarray            # conditions × conditions, mean TG_raw
    transportable_fraction_matrix: np.ndarray  # fraction TRANSPORTABLE per pair
    n_total_pairs: int
    n_estimable_pairs: int


# ═══════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════
# Shared M1 primitives are imported above. TG-specific design diagnostics, Q²
# estimators, permutation tests, bootstrap intervals, and interpretation rules
# remain private to this module.

def _compute_cramers_v(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> float:
    """
    Cramer's V between two categorical label arrays.

    Measures association between condition and batch variables.
    V = 1.0 means perfect confounding (every batch has exactly one condition).

    Parameters
    ----------
    labels1 : np.ndarray of strings
    labels2 : np.ndarray of strings

    Returns
    -------
    float in [0, 1]
    """
    from scipy.stats import chi2_contingency

    uniq1 = np.unique(labels1)
    uniq2 = np.unique(labels2)
    n1 = len(uniq1)
    n2 = len(uniq2)

    if n1 <= 1 or n2 <= 1:
        # One of the variables is constant → no association
        return 0.0

    # Build contingency table
    table = np.zeros((n1, n2), dtype=np.float64)
    for i, u1 in enumerate(uniq1):
        for j, u2 in enumerate(uniq2):
            table[i, j] = float(np.sum((labels1 == u1) & (labels2 == u2)))

    # If any row or column is all zeros, chi2 is degenerate
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        return 0.0

    try:
        chi2, p_val, dof, expected = chi2_contingency(table)
    except Exception:
        return 0.0

    n = len(labels1)
    k = min(n1, n2) - 1
    if k <= 0 or chi2 <= 0 or n == 0:
        return 0.0

    v = np.sqrt(chi2 / (n * k))
    return float(min(v, 1.0))


def _compute_js_divergence(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Jensen-Shannon divergence between source and target RNA distributions,
    averaged across all RNA features.

    Higher values indicate more divergent RNA distributions, which may drive
    stratum shift in the TG decomposition.

    Parameters
    ----------
    X_source : np.ndarray (n_source, p)
    X_target : np.ndarray (n_target, p)
    n_bins : int
        Number of bins for histogram-based density estimation.

    Returns
    -------
    float — mean JS divergence across features. NaN if computation fails.
    """
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    n_features = X_source.shape[1]

    if n_source < 2 or n_target < 2 or n_features == 0:
        return float('nan')

    js_values = []
    for j in range(n_features):
        col_src = X_source[:, j].astype(np.float64)
        col_tgt = X_target[:, j].astype(np.float64)

        pooled = np.concatenate([col_src, col_tgt])
        pooled_min = float(np.min(pooled))
        pooled_max = float(np.max(pooled))

        if pooled_max - pooled_min < 1e-12:
            # Constant column — JS divergence is 0
            js_values.append(0.0)
            continue

        # Quantile-based bin edges for better coverage
        bin_edges = np.percentile(pooled, np.linspace(0, 100, n_bins + 1))
        # Ensure last bin edge includes max
        bin_edges[-1] += 1e-12

        try:
            p_src, _ = np.histogram(col_src, bins=bin_edges, density=True)
            p_tgt, _ = np.histogram(col_tgt, bins=bin_edges, density=True)

            p_src = p_src.astype(np.float64)
            p_tgt = p_tgt.astype(np.float64)
            p_src /= np.sum(p_src) + 1e-15
            p_tgt /= np.sum(p_tgt) + 1e-15

            m = 0.5 * (p_src + p_tgt)
            kl_sm = np.sum(p_src * np.log2((p_src + 1e-15) / (m + 1e-15)))
            kl_tm = np.sum(p_tgt * np.log2((p_tgt + 1e-15) / (m + 1e-15)))
            js = 0.5 * (kl_sm + kl_tm)
            js_values.append(float(js))
        except Exception:
            continue

    if len(js_values) == 0:
        return float('nan')
    return float(np.mean(js_values))


# ═══════════════════════════════════════════════════════════════════
# Q² computation functions
# ═══════════════════════════════════════════════════════════════════

def _compute_q2_within(
    P_target: np.ndarray,
    X_target: np.ndarray,
    strata_target: np.ndarray,
    cv_alphas: List[float],
    groups_target: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Compute Q²_within_b — LOOCV prediction accuracy of M1 within target condition.

    Parameters
    ----------
    groups_target : np.ndarray or None
        Patient/donor group IDs. When provided, all same-group samples are
        excluded from each LOOCV fold (prevents patient-level leakage).

    Returns
    -------
    q2_within_b : float
    mse_stratum_b : float
    mse_model_b : float
    avg_edf : float
    best_alpha : float
        Average Ridge alpha selected across LOOCV folds (or 1.0 fallback).
        Used to lock alpha for Q²_crossed to eliminate Ridge bonus confound.
    """
    # Stratum baseline for target
    _, mse_stratum_b = _compute_stratum_means_loocv(P_target, strata_target, groups=groups_target)

    # LOOCV M1 on target
    _, mse_model, avg_edf, best_alpha = _m1_loocv(
        P_target, X_target, strata_target, cv_alphas, groups=groups_target,
    )

    if mse_stratum_b > 1e-10 and not np.isnan(mse_model):
        q2 = float(1.0 - mse_model / mse_stratum_b)
    else:
        q2 = float('nan')

    return q2, mse_stratum_b, mse_model, avg_edf, best_alpha


@_quiet_internal
def _compute_q2_within_matched(
    P_target: np.ndarray,
    X_target: np.ndarray,
    target_strata: np.ndarray,
    n_source: int,
    cv_alphas: List[float],
    n_subsamples: int = 100,
    seed: Optional[int] = None,
    groups_target: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Q² within target using matched training size = n_source.

    Draws n_source samples from target, trains M1, predicts on remaining
    n_target - n_source. Repeated n_subsamples times. Directly comparable
    to a→b cross-condition prediction which also trains on n_source.

    Eliminates the LOOCV bias: in standard Q²_within_b (LOOCV), the model
    trains on n_target-1 samples, while Q²_a→b trains on only n_source.
    Even with identical RNA→protein coupling, the larger training set gives
    LOOCV an advantage, inflating TG_raw under the null.

    Falls back to LOOCV (_compute_q2_within) when n_target - n_source < 3
    (no meaningful holdout set) or when too few subsamples succeed.

    Returns
    -------
    q2_within_b : float
    mse_stratum_b : float  (from LOOCV stratum means on FULL target)
    mse_model_b : float    (average MSE across subsamples)
    avg_edf : float        (0.0 — not applicable for matched subsample)
    best_alpha : float     (1.0 default — RidgeCV selects alpha per subsample)
    """
    n_target = len(P_target)

    # Fallback: not enough target samples for a meaningful holdout set
    if n_target - n_source < 3:
        return _compute_q2_within(
            P_target, X_target, target_strata, cv_alphas,
            groups_target=groups_target,
        )

    # Stratum baseline on FULL target (LOOCV stratum means, same as standard within-b)
    _, mse_stratum_b = _compute_stratum_means_loocv(P_target, target_strata,
                                                     groups=groups_target)
    if mse_stratum_b < 1e-10:
        # Degenerate stratum baseline — unify with the LOOCV path
        # (_compute_q2_within returns q2 = NaN here, not 0.0; 0.0 would look
        # like a perfect Q² and distort TG_raw/TG_log).
        # mse_model_b remains at its initialized zero because no subsamples ran.
        return np.nan, mse_stratum_b, 0.0, 0.0, 1.0

    mse_vals = []

    for rep in range(n_subsamples):
        # Per-subsample deterministic seed (seeded per subsample index)
        rep_seed = subsample_seed(seed, rep)  # m1 chain: within/crossed (offset=0)
        rng = np.random.default_rng(rep_seed)

        # Draw n_source training indices from target
        train_idx = rng.choice(n_target, size=n_source, replace=False)
        test_idx = np.setdiff1d(np.arange(n_target), train_idx)

        P_tr = P_target[train_idx].astype(np.float64)
        X_tr = X_target[train_idx].astype(np.float64)
        strata_tr = target_strata[train_idx]

        P_te = P_target[test_idx].astype(np.float64)
        X_te = X_target[test_idx].astype(np.float64)
        strata_te = target_strata[test_idx]

        # Train M1 on subsample (same pattern as _compute_q2_cross / _m1_train_test)
        predictions = _m1_train_test(
            P_tr, X_tr, strata_tr,
            P_te, X_te, strata_te,
            cv_alphas,
        )

        valid = ~np.isnan(predictions)
        if valid.sum() >= 1:
            mse = float(np.mean((predictions[valid] - P_te[valid]) ** 2))
            mse_vals.append(mse)

    if len(mse_vals) < 3:
        # Insufficient valid subsamples — fall back to LOOCV
        return _compute_q2_within(
            P_target, X_target, target_strata, cv_alphas,
            groups_target=groups_target,
        )

    avg_mse = float(np.mean(mse_vals))
    q2 = float(1.0 - avg_mse / mse_stratum_b)

    # best_alpha = 1.0 default; RidgeCV selects alpha per subsample via _m1_train_test
    return q2, mse_stratum_b, avg_mse, 0.0, 1.0


def _compute_q2_cross(
    P_source: np.ndarray,
    X_source: np.ndarray,
    strata_source: np.ndarray,
    P_target: np.ndarray,
    X_target: np.ndarray,
    strata_target: np.ndarray,
    mse_stratum_b: float,
    cv_alphas: List[float],
) -> Tuple[float, float]:
    """
    Compute Q²_a_to_b — train M1 on source condition, predict on target.

    Returns
    -------
    q2_a_to_b : float
    mse_a_to_b : float
    """
    predictions = _m1_train_test(
        P_source, X_source, strata_source,
        P_target, X_target, strata_target,
        cv_alphas,
    )
    valid = ~np.isnan(predictions)
    if valid.sum() < 1:
        return float('nan'), float('nan')

    mse = float(np.mean((predictions[valid] - P_target[valid]) ** 2))

    if mse_stratum_b > 1e-10:
        q2 = float(1.0 - mse / mse_stratum_b)
    else:
        q2 = float('nan')

    return q2, mse


def _compute_q2_cross_matched(
    P_source: np.ndarray,
    X_source: np.ndarray,
    strata_source: np.ndarray,
    P_target: np.ndarray,
    X_target: np.ndarray,
    strata_target: np.ndarray,
    train_size: int,
    mse_stratum_b: float,
    cv_alphas: List[float],
    n_subsamples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Q² a→b with matched training size by subsampling the source cohort.

    Draws `train_size` samples from source, trains M1, predicts on ALL target.
    Repeated `n_subsamples` times. Averaged Q² is directly comparable to
    within-b matched subsample (both train on `train_size` samples).

    Eliminates the training-size asymmetry that causes NULL TG bias at n≥50.
    """
    n_source = len(P_source)
    if n_source <= train_size:
        return _compute_q2_cross(
            P_source, X_source, strata_source,
            P_target, X_target, strata_target,
            mse_stratum_b, cv_alphas,
        )
    mse_vals = []
    for rep in range(n_subsamples):
        rep_seed = subsample_seed(seed, rep, offset=10000)  # m1 chain: cross a→b
        rng = np.random.default_rng(rep_seed)
        tr_idx = rng.choice(n_source, size=train_size, replace=False)
        preds = _m1_train_test(
            P_source[tr_idx], X_source[tr_idx], strata_source[tr_idx],
            P_target, X_target, strata_target, cv_alphas,
        )
        valid = ~np.isnan(preds)
        if valid.sum() >= 1:
            mse_vals.append(float(np.mean((preds[valid] - P_target[valid]) ** 2)))
    if len(mse_vals) < 3:
        return _compute_q2_cross(
            P_source, X_source, strata_source,
            P_target, X_target, strata_target,
            mse_stratum_b, cv_alphas,
        )
    avg_mse = float(np.mean(mse_vals))
    q2 = float(1.0 - avg_mse / mse_stratum_b) if mse_stratum_b > 1e-10 else float('nan')
    return q2, avg_mse


def _compute_interaction_pvalue(
    P_pooled: np.ndarray,
    X_pooled: np.ndarray,
    condition_labels: np.ndarray,
    train_size: int,
    cv_alphas: List[float],
    seed: int = 42,
    n_permutations: int = 100,
    cov_type: str = 'nonrobust',
) -> float:
    """
    Interaction P value via condition-label restricted permutation.

    Fits Y ~ Y_hat * condition as an OLS interaction model on a fixed
    train/test split, where Y_hat is the Ridge-predicted protein value from
    RNA (trained on the training fold only). Tests whether the interaction
    coefficient is zero.

    Under the null, condition labels are permuted *within the test fold*. The train/test partition (the
    "fold structure") is kept fixed and the observed within-fold class sizes
    are preserved. Y_hat depends only on (X, P) of the training samples, never
    on condition labels, so under the interaction null the labels on the test
    fold are exchangeable and the permutation p-value is exact. The statistic
    is |t| of the interaction coefficient; the two-sided p-value follows the
    TG main-test convention (_permutation_test_tg):

        p = (1 + #(|t_perm| >= |t_obs|)) / (1 + n_valid)

    with n_valid valid permutations, so p = 0 is unreachable (continuity-corrected
    permutation p-value); the
    minimum achievable value is 1 / (1 + n_valid). Permutations with fewer
    than 3 samples of either condition on the test fold are discarded, and
    fewer than 10 valid permutations returns 1.0 (same n_valid < 10
    convention as _permutation_test_tg).

    Resampling observed data with replacement would preserve the signal under
    test. Label permutation instead provides the required exchangeable null.

    Parameters
    ----------
    n_permutations : int
        Number of label permutations for the null distribution (default 100;
        callers pass the same count as the TG main permutation test).
    cov_type : str
        'nonrobust' — classical OLS standard errors (default; the permutation
        distribution is valid regardless of heteroskedasticity under label
        exchangeability).
        'HC3' — MacKinnon–White heteroskedasticity-robust standard errors
        (optional robustness refinement of the |t| statistic).

    Returns
    -------
    float : two-sided permutation p-value in (0, 1]
    """
    try:
        from sklearn.linear_model import RidgeCV
        import statsmodels.api as sm
    except ImportError:
        return 1.0

    n = len(P_pooled)
    if n < 10:
        return 1.0
    rng = np.random.default_rng(seed)

    def _interaction_abs_t(P, X, cond, tr_idx, te_idx, cov_type='nonrobust'):
        """|t| of the OLS interaction coefficient for a given split/labels.

        cov_type : str
            'nonrobust' — classical OLS standard errors.
            'HC3' — MacKinnon–White heteroskedasticity-robust standard
            errors.
        """
        scaler_x = StandardScaler().fit(X[tr_idx])
        scaler_y = StandardScaler().fit(P[tr_idx].reshape(-1, 1))
        ridge = RidgeCV(alphas=cv_alphas, cv=min(5, len(tr_idx))).fit(
            scaler_x.transform(X[tr_idx]),
            scaler_y.transform(P[tr_idx].reshape(-1, 1)).ravel(),
        )
        y_hat = scaler_y.inverse_transform(
            ridge.predict(scaler_x.transform(X[te_idx])).reshape(-1, 1)
        ).ravel()
        cond_dummy = (cond[te_idx] == cond[te_idx][0]).astype(float)
        X_ols = np.column_stack([
            np.ones(len(te_idx)), y_hat, cond_dummy, y_hat * cond_dummy
        ])
        try:
            ols = sm.OLS(P[te_idx], X_ols).fit(cov_type=cov_type)
            return float(abs(ols.tvalues[3]))
        except Exception:
            return float('nan')

    # ── Fixed train/test split (fold structure preserved across permutations) ──
    tr_idx = rng.choice(n, size=min(train_size, n - 3), replace=False)
    te_idx = np.setdiff1d(np.arange(n), tr_idx)

    # ── Observed statistic ──
    t_obs = _interaction_abs_t(P_pooled, X_pooled, condition_labels, tr_idx, te_idx,
                               cov_type=cov_type)
    if np.isnan(t_obs):
        return 1.0

    # ── Condition-label restricted permutation null ──
    # Shuffle labels only within the test fold, preserving the observed
    # per-class sizes (balanced/restricted permutation — exact exchangeability
    # under the null). The training fold's labels never enter Y_hat, so the
    # observed partition is the conditioning structure.
    counts = np.bincount(condition_labels[te_idx], minlength=2)
    if counts.min() < 3:
        # Interaction term not estimable on the observed test fold
        return 1.0
    n_ones = int(counts[1])
    n_te = len(te_idx)

    n_valid = 0
    n_exceed = 0
    for _ in range(n_permutations):
        # Restricted shuffle: choose n_ones positions within the test fold to
        # hold class 1, preserving the observed class sizes. The inner helper
        # indexes labels by te_idx, so the permuted labels are written back
        # into a full-length copy.
        perm_ones = rng.choice(n_te, size=n_ones, replace=False)
        cond_perm = condition_labels.copy()
        cond_perm[te_idx] = 0
        cond_perm[te_idx[perm_ones]] = 1
        t_perm = _interaction_abs_t(P_pooled, X_pooled, cond_perm, tr_idx, te_idx,
                                    cov_type=cov_type)
        if np.isnan(t_perm):
            continue
        n_valid += 1
        if t_perm >= t_obs:
            n_exceed += 1

    if n_valid < 10:
        # Same convention as _permutation_test_tg: insufficient null support
        return 1.0

    # TG main-test convention: (count + 1) / (n + 1) — continuity-corrected
    # permutation p-value (never p = 0).
    return float((n_exceed + 1) / (n_valid + 1))


def _compute_q2_crossed(
    P_pooled: np.ndarray,
    X_pooled: np.ndarray,
    strata_pooled: np.ndarray,
    target_indices: np.ndarray,
    mse_stratum_b: float,
    cv_alphas: List[float],
    fixed_alpha: Optional[float] = None,
    groups_pooled: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Compute Q²_crossed — LOOCV on target samples after training on pooled
    source + target data. The model sees the union of stratum distributions.

    Uses `fixed_alpha` (from within_b) when provided, to eliminate the
    "Ridge bonus" confound where more training data selects a different
    optimal alpha than the within-b model had.

    For each held-out target sample:
      Training = all pooled samples minus the held-out target sample
      (or all pooled samples minus all same-group samples when groups_pooled
      is provided)

    Parameters
    ----------
    fixed_alpha : float or None
        If provided, use this alpha for Ridge instead of RidgeCV.
        Should come from _compute_q2_within()'s best_alpha.
    groups_pooled : np.ndarray or None
        Group IDs for pooled data. When provided, all samples sharing the
        held-out sample's group ID are excluded from training, preventing
        patient-level leakage in LOOCV.

    Returns
    -------
    q2_crossed : float
    mse_crossed : float
    avg_edf : float  (average effective df for the crossed model folds)
    """
    preds, mse, avg_edf, _ = _m1_loocv(
        P_pooled, X_pooled, strata_pooled,
        cv_alphas,
        eval_indices=target_indices,
        fixed_alpha=fixed_alpha,
        groups=groups_pooled,
    )

    if mse_stratum_b > 1e-10 and not np.isnan(mse):
        q2 = float(1.0 - mse / mse_stratum_b)
    else:
        q2 = float('nan')

    return q2, mse, avg_edf


@_quiet_internal
def _compute_q2_crossed_matched(
    P_pooled: np.ndarray,
    X_pooled: np.ndarray,
    strata_pooled: np.ndarray,
    target_indices: np.ndarray,
    n_source: int,
    mse_stratum_b: float,
    cv_alphas: List[float],
    n_subsamples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Q²_crossed using matched subsample CV (mirrors _compute_q2_within_matched).

    Draws n_source samples from pooled (a+b) data, trains M1, predicts on
    target b samples not in the training set. Repeated n_subsamples times.

    This is the direct counterpart to _compute_q2_within_matched: both train
    on n_source samples, but crossed sees both conditions' strata while
    within-b sees only target strata.

    Falls back to LOOCV (_compute_q2_crossed) when too few valid subsamples.

    Returns
    -------
    q2_crossed : float
    mse_crossed : float
    avg_edf : float  (0.0 — not applicable)
    """
    n_pooled = len(P_pooled)

    # Build boolean mask for target_b samples in pooled array
    target_mask = np.zeros(n_pooled, dtype=bool)
    target_mask[target_indices] = True

    # Fallback if too few target samples
    n_target_b = int(target_mask.sum())
    if n_target_b - n_source < 3:
        return _compute_q2_crossed(
            P_pooled, X_pooled, strata_pooled, target_indices,
            mse_stratum_b, cv_alphas,
        )

    mse_vals = []

    for rep in range(n_subsamples):
        # Per-subsample deterministic seed
        rep_seed = subsample_seed(seed, rep)  # m1 chain: within/crossed (offset=0)
        rng = np.random.default_rng(rep_seed)

        # Draw n_source from pooled
        train_idx = rng.choice(n_pooled, size=n_source, replace=False)

        # Target b samples NOT in training set
        test_mask = target_mask.copy()
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        if len(test_idx) < 3:
            continue

        P_tr = P_pooled[train_idx].astype(np.float64)
        X_tr = X_pooled[train_idx].astype(np.float64)
        strata_tr = strata_pooled[train_idx]

        P_te = P_pooled[test_idx].astype(np.float64)
        X_te = X_pooled[test_idx].astype(np.float64)
        strata_te = strata_pooled[test_idx]

        predictions = _m1_train_test(
            P_tr, X_tr, strata_tr,
            P_te, X_te, strata_te,
            cv_alphas,
        )

        valid = ~np.isnan(predictions)
        if valid.sum() >= 1:
            mse = float(np.mean((predictions[valid] - P_te[valid]) ** 2))
            mse_vals.append(mse)

    if len(mse_vals) < 3:
        return float('nan'), float('nan'), 0.0

    avg_mse = float(np.mean(mse_vals))

    if mse_stratum_b > 1e-10 and not np.isnan(avg_mse):
        q2 = float(1.0 - avg_mse / mse_stratum_b)
    else:
        q2 = float('nan')

    return q2, avg_mse, 0.0


# ═══════════════════════════════════════════════════════════════════
# Permutation test
# ═══════════════════════════════════════════════════════════════════

def _empirical_upper_tail_pvalue(null_values: np.ndarray, observed: float) -> float:
    """Continuity-corrected upper-tail p-value for positive TG loss."""
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0 or not np.isfinite(observed):
        return 1.0
    return float((np.sum(valid >= observed) + 1) / (len(valid) + 1))

def _permutation_test_tg(
    P_source: np.ndarray,
    P_target: np.ndarray,
    X_source: np.ndarray,
    X_target: np.ndarray,
    source_strata: np.ndarray,
    target_strata: np.ndarray,
    cv_alphas: List[float],
    n_permutations: int,
    rng: np.random.Generator,
    tg_raw_obs: float,
    tg_design_obs: float,
    tg_rna_obs: float,
    best_alpha: float = 1.0,
    source_groups: Optional[np.ndarray] = None,
    target_groups: Optional[np.ndarray] = None,
    cv_mode: str = 'loocv',
    n_subsamples: int = 100,
    n_source: int = 0,
    train_size: Optional[int] = None,
    matched_seed: Optional[int] = None,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Permutation test for TG (restricted permutation).

    H0: condition labels are exchangeable within the pool (same batch).
    Procedure:
      1. Pool source and target samples
      2. FIX the CV fold partition BEFORE any shuffling: source occupies
         pooled positions [0, n_source), target occupies [n_source, n_pooled)
      3. Shuffle pooled samples (equivalent to shuffling condition labels
         under H0 exchangeability) with the partition positions held fixed
         (block-restricted: shuffle within groups when groups are provided)
      4. Recompute the full TG decomposition (raw + design + RNA). In
         matched mode, use the same matched within, transfer, and crossed
         estimators as the observed statistic. In fallback mode, use the
         null's LOOCV within, transfer, and crossed helpers described below
      5. Repeat n_permutations times

    Restricted permutation guarantees that every permutation uses the EXACT
    same train/test partition sizes and fold structure as the observed data —
    the permuted partitions cannot drift in size.

    In ``matched_subsample`` mode, every null realization uses the observed
    matched training size and the same matched within, transfer, and crossed
    estimators as the observed statistic. In fallback mode, the null uses
    ``_compute_q2_within``, ``_compute_q2_cross``, and
    ``_compute_q2_crossed``. The observed fallback path uses LOOCV for within
    and crossed but retains ``_compute_q2_cross_matched`` for the transfer
    arm; therefore estimator identity is asserted only for matched mode.
    Partition positions and effective source/target sample counts remain
    fixed in either mode.

    When `source_groups` and `target_groups` are provided, the permutation
    uses block-restricted permutation: samples are shuffled WITHIN each group.
    This preserves the within-group correlation structure under H0.
    Groups with samples in only one condition stay fixed (they carry no
    information about the condition transition).

    Parameters
    ----------
    P_source, P_target : np.ndarray
    X_source, X_target : np.ndarray
    source_strata, target_strata : np.ndarray — original strata (condition+batch)
    cv_alphas : list of float
    n_permutations : int
    rng : np.random.Generator
    tg_raw_obs, tg_design_obs, tg_rna_obs : float
        Observed TG values to compare against.
    source_groups, target_groups : np.ndarray or None
        Patient/donor group IDs. When provided, block permutation within
        groups is used instead of global shuffle.
    cv_mode : str
        'loocv' (default) or 'matched_subsample'. Selects the matched-null
        contract or the explicitly documented fallback-null helpers above.
    n_subsamples : int
        Number of matched-subsample repetitions per null realization in
        ``matched_subsample`` mode; unused in LOOCV mode.
    n_source : int
        Legacy compatibility argument. The executable function derives the
        effective fixed source-partition size from ``len(P_source)``; matched
        training size is controlled by ``train_size``.
    matched_seed : int or None
        Base seed for deterministic matched-subsample draws in the null;
        unused in LOOCV mode.

    Returns
    -------
    p_raw : float
    p_design : float
    p_rna : float
    theta : float  — max(0.05, 95th %ile of null TG_raw)
    null_raw_vals : np.ndarray
    """
    # Pool data
    n_source = len(P_source)
    n_target = len(P_target)
    matched_train_size = int(train_size if train_size is not None else n_source)

    P_pooled = np.concatenate([P_source, P_target]).astype(np.float64)
    X_pooled = np.concatenate([X_source, X_target], axis=0).astype(np.float64)
    n_pooled = len(P_pooled)

    # Original strata concatenated
    pooled_strata = np.concatenate([source_strata, target_strata])

    # Condition labels (0 = source, 1 = target)
    cond_labels = np.zeros(n_pooled, dtype=int)
    cond_labels[n_source:] = 1

    # Pooled groups (for block permutation)
    has_groups = (source_groups is not None and target_groups is not None)
    pooled_groups = None
    if has_groups:
        pooled_groups = np.concatenate([source_groups, target_groups])

    # ── Restricted permutation setup ──
    # The CV fold partition is FIXED before any shuffling: source occupies
    # pooled positions [0, n_source), target occupies [n_source, n_pooled).
    # Each permutation shuffles the pooled SAMPLES (equivalent to shuffling
    # condition labels under H0 exchangeability) while the partition positions
    # stay fixed, so every permutation uses the EXACT same train/test sizes
    # and fold structure — the permuted partitions cannot drift in size.
    null_raw = []
    null_design = []
    null_rna = []
    n_valid_perms = 0

    for permutation_index in range(n_permutations):
        # Shuffle pooled samples (restricted permutation; positions fixed)
        if pooled_groups is not None:
            # Block-restricted permutation: shuffle sample indices within
            # each group that contains BOTH conditions
            perm_idx = np.arange(n_pooled)
            for g in np.unique(pooled_groups):
                g_idx = np.where(pooled_groups == g)[0]
                g_cond_0 = np.sum(cond_labels[g_idx] == 0)
                g_cond_1 = np.sum(cond_labels[g_idx] == 1)
                if g_cond_0 > 0 and g_cond_1 > 0:
                    perm_idx[g_idx] = g_idx[rng.permutation(len(g_idx))]
                # Groups with all samples in one condition: position stays fixed
        else:
            perm_idx = rng.permutation(n_pooled)

        # Fixed partition positions → permuted partition sizes are ALWAYS
        # exactly (n_source, n_target); no size drift possible.
        src_pos = perm_idx[:n_source]
        tgt_pos = perm_idx[n_source:]

        # Extract permuted partitions
        P_ps = P_pooled[src_pos]
        P_pt = P_pooled[tgt_pos]
        X_ps = X_pooled[src_pos]
        X_pt = X_pooled[tgt_pos]
        strata_ps = pooled_strata[src_pos]
        strata_pt = pooled_strata[tgt_pos]

        # Extract groups for permuted partitions
        pt_groups = pooled_groups[tgt_pos] if pooled_groups is not None else None
        pooled_perm_groups = None
        if pooled_groups is not None:
            pooled_perm_groups = np.concatenate([
                pooled_groups[src_pos],
                pooled_groups[tgt_pos]
            ])

        P_cross = np.concatenate([P_ps, P_pt])
        X_cross = np.concatenate([X_ps, X_pt], axis=0)
        strata_cross = np.concatenate([strata_ps, strata_pt])
        tgt_idx = np.arange(n_source, n_source + len(P_pt))
        if cv_mode == 'matched_subsample':
            permutation_seed = (0 if matched_seed is None else int(matched_seed)) + 100000 + permutation_index
            q2_w, _, mse_s_b, _, _ = _compute_q2_within_matched(
                P_pt, X_pt, strata_pt, matched_train_size, cv_alphas,
                n_subsamples=n_subsamples, seed=permutation_seed, groups_target=pt_groups,
            )
            if np.isnan(q2_w) or mse_s_b < 1e-10:
                continue
            q2_ab, _ = _compute_q2_cross_matched(
                P_ps, X_ps, strata_ps, P_pt, X_pt, strata_pt,
                matched_train_size, mse_s_b, cv_alphas,
                n_subsamples=n_subsamples, seed=permutation_seed,
            )
            q2_cross, _, _ = _compute_q2_crossed_matched(
                P_cross, X_cross, strata_cross, tgt_idx, matched_train_size,
                mse_s_b, cv_alphas, n_subsamples=n_subsamples, seed=permutation_seed,
            )
        else:
            q2_w, mse_s_b, _, _, _ = _compute_q2_within(
                P_pt, X_pt, strata_pt, cv_alphas, groups_target=pt_groups)
            if np.isnan(q2_w) or mse_s_b < 1e-10:
                continue
            q2_ab, _ = _compute_q2_cross(
                P_ps, X_ps, strata_ps, P_pt, X_pt, strata_pt, mse_s_b, cv_alphas,
            )
            q2_cross, _, _ = _compute_q2_crossed(
                P_cross, X_cross, strata_cross, tgt_idx, mse_s_b, cv_alphas,
                fixed_alpha=1.0, groups_pooled=pooled_perm_groups,
            )
        if np.isnan(q2_ab):
            continue
        if np.isnan(q2_cross):
            continue

        # --- TG decomposition ---
        tg_r = q2_w - q2_ab
        tg_d = q2_w - q2_cross
        tg_rna_perm = q2_cross - q2_ab

        null_raw.append(float(tg_r))
        null_design.append(float(tg_d))
        null_rna.append(float(tg_rna_perm))
        n_valid_perms += 1

    null_raw = np.array(null_raw, dtype=np.float64)
    null_design = np.array(null_design, dtype=np.float64)
    null_rna = np.array(null_rna, dtype=np.float64)

    print(f"[cordiag] permutation test: n_valid_perms = {n_valid_perms} / {n_permutations}")

    if n_valid_perms < 10:
        # Insufficient valid permutations — return null p-values
        return 1.0, 1.0, 1.0, 0.05, null_raw

    p_raw = _empirical_upper_tail_pvalue(null_raw, tg_raw_obs)
    p_design = _empirical_upper_tail_pvalue(null_design, tg_design_obs)
    p_rna = _empirical_upper_tail_pvalue(null_rna, tg_rna_obs)

    # Theta: simulation-calibrated threshold
    theta = float(max(0.05, np.percentile(null_raw, 95)))

    return p_raw, p_design, p_rna, theta, null_raw


# ═══════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════

def _bootstrap_ci_tg(
    P_source: np.ndarray,
    P_target: np.ndarray,
    X_source: np.ndarray,
    X_target: np.ndarray,
    source_strata: np.ndarray,
    target_strata: np.ndarray,
    cv_alphas: List[float],
    n_bootstrap: int,
    rng: np.random.Generator,
    mse_stratum_b: float,
    n_target: int,
    source_groups: Optional[np.ndarray] = None,
    target_groups: Optional[np.ndarray] = None,
    cv_mode: str = 'loocv',
    train_size: int = 0,
    n_subsamples: int = 100,
    matched_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for TG_raw.

    Paired out-of-bag bootstrap (all n_target):
      1. Resample the source and target conditions with replacement (B times)
      2. Fit the within-b model on the resampled target and the a->b model on
         the resampled source; evaluate BOTH arms out-of-sample on the target
         samples NOT drawn in the resample (out-of-bag set), with the
         stratum-mean baseline recomputed on the OOB set
      3. TG_raw_boot = Q²_within_b^b - Q²_a->b^b
      4. 95% CI = [2.5th, 97.5th] percentile of TG_raw_boot

    The OOB evaluation removes the LOOCV-with-duplicates optimism of the
    previous design, because bootstrap resamples were evaluated in-sample via
    LOOCV, which can inflate Q²_within_b and
    deflate TG_raw). Resamples whose OOB set has fewer than 5 samples are
    discarded; pairs with fewer than 20 valid resamples return NaN (the
    equivalence classification then fails closed to PARTIALLY_TRANSPORTABLE).

    Parameters
    ----------
    P_source, P_target : np.ndarray
    X_source, X_target : np.ndarray
    source_strata, target_strata : np.ndarray
    cv_alphas : list of float
    n_bootstrap : int
    rng : np.random.Generator
    mse_stratum_b : float
    n_target : int
    source_groups, target_groups : np.ndarray or None
        Patient/donor group IDs. Passed through to LOOCV calls to prevent
        patient-level information leakage in bootstrap samples.

    Returns
    -------
    ci_lower : float
    ci_upper : float
    ztg_bootstrap : float  — standardized TG_raw (z-score)
    """
    # ── Q²_within_b: use same CV strategy as main analysis ──
    if cv_mode == 'matched_subsample' and train_size >= 8 and n_target - train_size >= 3:
        q2_w_orig, _, mse_w_orig, _, _ = _compute_q2_within_matched(
            P_target, X_target, target_strata, train_size, cv_alphas,
            n_subsamples=n_subsamples, seed=matched_seed,
            groups_target=target_groups,
        )
    else:
        q2_w_orig, _, mse_w_orig, _, _ = _compute_q2_within(
            P_target, X_target, target_strata, cv_alphas,
            groups_target=target_groups,
        )

    # ── Q²_a→b: train on source, predict on target ──
    matched_mode = cv_mode == 'matched_subsample' and train_size >= 8 and n_target - train_size >= 3
    if matched_mode:
        q2_ab_orig, _ = _compute_q2_cross_matched(
            P_source, X_source, source_strata,
            P_target, X_target, target_strata,
            train_size, mse_stratum_b, cv_alphas,
            n_subsamples=n_subsamples, seed=matched_seed,
        )
    else:
        preds_ab = _m1_train_test(
            P_source, X_source, source_strata,
            P_target, X_target, target_strata,
            cv_alphas,
        )
        valid_ab = ~np.isnan(preds_ab)
        if valid_ab.sum() < 1:
            return float('nan'), float('nan'), float('nan')
        mse_ab_orig = float(np.mean((preds_ab[valid_ab] - P_target[valid_ab]) ** 2))
        q2_ab_orig = 1.0 - mse_ab_orig / mse_stratum_b if mse_stratum_b > 1e-10 else float('nan')
    tg_orig = q2_w_orig - q2_ab_orig

    # Bootstrap — paired out-of-bag design (see docstring)
    tg_boot = np.full(n_bootstrap, np.nan, dtype=np.float64)
    n_oob_min = 5
    for b in range(n_bootstrap):
        t_idx = rng.integers(0, n_target, size=train_size if matched_mode else n_target)
        s_idx = rng.integers(0, len(P_source), size=train_size if matched_mode else len(P_source))
        oob_idx = np.array(sorted(set(range(n_target)) - set(t_idx.tolist())))
        if len(oob_idx) < n_oob_min:
            continue

        P_oob = P_target[oob_idx].astype(np.float64)
        X_oob = X_target[oob_idx].astype(np.float64)
        strata_oob = target_strata[oob_idx]

        # Stratum-mean baseline on the OOB evaluation set
        _, mse_s_oob = _compute_stratum_means_loocv(P_oob, strata_oob)
        if mse_s_oob < 1e-10:
            continue

        # Q²_within^b: fit on resampled target -> evaluate on OOB target
        preds_w = _m1_train_test(
            P_target[t_idx], X_target[t_idx], target_strata[t_idx],
            P_oob, X_oob, strata_oob,
            cv_alphas,
        )
        valid_w = ~np.isnan(preds_w)
        if valid_w.sum() < 1:
            continue
        mse_w_boot = float(np.mean((preds_w[valid_w] - P_oob[valid_w]) ** 2))
        q2_w_boot = 1.0 - mse_w_boot / mse_s_oob

        # Q²_a→b^b: fit on resampled source -> evaluate on OOB target
        preds_ab_boot = _m1_train_test(
            P_source[s_idx], X_source[s_idx], source_strata[s_idx],
            P_oob, X_oob, strata_oob,
            cv_alphas,
        )
        valid_boot = ~np.isnan(preds_ab_boot)
        if valid_boot.sum() < 1:
            continue
        mse_ab_boot = float(np.mean((preds_ab_boot[valid_boot] - P_oob[valid_boot]) ** 2))
        q2_ab_boot = 1.0 - mse_ab_boot / mse_s_oob
        tg_boot[b] = q2_w_boot - q2_ab_boot

    tg_boot = tg_boot[~np.isnan(tg_boot)]
    n_valid_boot = len(tg_boot)

    if n_valid_boot < 20:
        return float('nan'), float('nan'), float('nan')

    ci_lower = float(np.percentile(tg_boot, 2.5))
    ci_upper = float(np.percentile(tg_boot, 97.5))

    # zTG from bootstrap distribution
    boot_mean = float(np.mean(tg_boot))
    boot_std = float(np.std(tg_boot))
    ztg = (tg_orig - boot_mean) / boot_std if boot_std > 1e-10 else float('nan')

    return ci_lower, ci_upper, ztg


# ═══════════════════════════════════════════════════════════════════
# Interpretation logic
# ═══════════════════════════════════════════════════════════════════

def _interpret_tg(
    estimable: bool,
    weak_baseline: bool,
    asymmetric: bool,
    tg_raw: float,
    q2_within_b: float,
    permutation_p_raw: float,
    tg_design_fraction: float,
    theta: float,
    ci_upper: float = float('inf'),
) -> Tuple[str, str, str]:
    """
    Map TG metrics to primary and secondary interpretation codes.

    Primary (can it transfer?):
      NOT_ESTIMABLE         — quality gate failed
      TRANSPORTABLE         — upper 95% bootstrap bound of TG_raw < theta
                              (equivalence/non-inferiority test against the
                              null-calibrated margin theta; the old
                              "TG_raw < theta AND p > 0.05" rule used absence
                              of evidence, which is not a valid equivalence
                              criterion)
      NON_TRANSPORTABLE     — TG_raw >= 3*theta AND p < 0.05
      PARTIALLY_TRANSPORTABLE — otherwise (intermediate)

    Secondary (why?):
      Only if primary in {PARTIALLY, NON_TRANSPORTABLE}
      STRATUM_SHIFT         — TG_design_fraction > 0.6
      CROSS_STUDY_RESIDUAL  — TG_design_fraction < 0.4
      MIXED                 — 0.4 <= fraction <= 0.6
      WEAK_BASELINE         — weak baseline flag set
      SAMPLE_ASYMMETRY      — asymmetric flag set

    Returns
    -------
    primary : str
    secondary : str
    text : str  — human-readable summary
    """
    # ── Primary ──
    if not estimable:
        primary = 'NOT_ESTIMABLE'
        secondary = ''
        text = (
            f'Quality gate failed: cannot estimate TG for this pair. '
            f'Possible causes: n<min_n, Q²_within_b<0, or condition×batch confounding.'
        )
        return primary, secondary, text

    if np.isnan(tg_raw):
        primary = 'NOT_ESTIMABLE'
        secondary = ''
        text = 'TG_raw is NaN — cannot estimate transportability.'
        return primary, secondary, text

    # ── Significantly negative TG: RNA unexpectedly helps ──
    if not np.isnan(tg_raw) and tg_raw < 0 and permutation_p_raw < 0.05:
        primary = 'RNA_UNEXPECTEDLY_HELPFUL'
        secondary_parts = []
        if weak_baseline:
            secondary_parts.append('WEAK_BASELINE')
        if asymmetric:
            secondary_parts.append('SAMPLE_ASYMMETRY')
        secondary = ' | '.join(secondary_parts) if secondary_parts else ''
        text = (
            f'RNA→protein relationship is unexpectedly helpful: source model predicts '
            f'target proteins BETTER than target model '
            f'(TG_raw={tg_raw:.4f} < 0, p={permutation_p_raw:.4f}).'
        )
        return primary, secondary, text

    if ci_upper < theta:
        primary = 'TRANSPORTABLE'
    elif tg_raw >= 3.0 * theta and permutation_p_raw < 0.05:
        primary = 'NON_TRANSPORTABLE'
    elif tg_raw < theta and permutation_p_raw < 0.05:
        primary = 'PARTIALLY_TRANSPORTABLE'  # directional evidence but small effect size
    else:
        primary = 'PARTIALLY_TRANSPORTABLE'

    # ── Secondary ──
    secondary_parts = []

    if weak_baseline:
        secondary_parts.append('WEAK_BASELINE')
    if asymmetric:
        secondary_parts.append('SAMPLE_ASYMMETRY')

    if primary in ('PARTIALLY_TRANSPORTABLE', 'NON_TRANSPORTABLE'):
        if not np.isnan(tg_design_fraction):
            if tg_design_fraction > 0.6:
                secondary_parts.append('STRATUM_SHIFT')
            elif tg_design_fraction < 0.4:
                secondary_parts.append('CROSS_STUDY_RESIDUAL')
            else:
                secondary_parts.append('MIXED')

    if secondary_parts:
        secondary = ' | '.join(secondary_parts)
    else:
        secondary = ''

    # ── Human text ──
    if primary == 'TRANSPORTABLE':
        text = (
            f'RNA→protein relationship is transportable from source to target '
            f'(upper 95% bootstrap bound of TG_raw = {ci_upper:.4f} < theta = '
            f'{theta:.4f}; equivalence margin on the TG_log scale '
            f'epsilon = ln(1 + theta/(1 - Q2_within_b))). '
            f'The M1 model generalizes within the calibrated equivalence margin.'
        )
    elif primary == 'NON_TRANSPORTABLE':
        base = (
            f'RNA→protein relationship is NOT transportable '
            f'(TG_raw={tg_raw:.4f} >= {3 * theta:.4f}, p={permutation_p_raw:.4f}). '
        )
        if weak_baseline:
            base += 'Baseline predictability is weak (Q²<0.1) — TG may be unreliable. '
        if asymmetric:
            base += f'Sample sizes are asymmetric. '
        if 'STRATUM_SHIFT' in secondary:
            base += 'The stratum-shift component is the larger descriptive component. '
        if 'CROSS_STUDY_RESIDUAL' in secondary:
            base += ('The cross-study residual is the larger descriptive component; '
                     'its cause is not identified. ')
        text = base.strip()
    elif primary == 'PARTIALLY_TRANSPORTABLE':
        base = (
            f'RNA→protein relationship partially transportable '
            f'(TG_raw={tg_raw:.4f}, p={permutation_p_raw:.4f}). '
        )
        if 'STRATUM_SHIFT' in secondary:
            base += 'The stratum-shift component contributes to the observed transfer gap. '
        if 'CROSS_STUDY_RESIDUAL' in secondary:
            base += 'The cross-study residual contributes to the observed transfer gap. '
        text = base.strip()
    else:
        text = 'NOT_ESTIMABLE — insufficient data or failed quality gates.'

    return primary, secondary, text


# ═══════════════════════════════════════════════════════════════════
# FDR correction
# ═══════════════════════════════════════════════════════════════════

def _apply_tg_fdr(
    results: Dict[str, Dict[Tuple[str, str], TGResult]],
) -> None:
    """
    Apply BH-FDR to TG results: per-protein and global.

    Per-protein FDR: BH correction across all valid condition pairs for one protein.
    Global FDR: BH correction across all proteins × condition pairs.

    Modifies TGResult fields fdr_per_protein and fdr_global in place.
    """
    from statsmodels.stats.multitest import multipletests

    # ── Per-protein FDR ──
    for prot, pair_results in results.items():
        pairs = sorted(pair_results.keys())
        pvals = [pair_results[p].permutation_p_raw for p in pairs]

        if len(pvals) > 0 and not all(np.isnan(x) for x in pvals):
            pvals_clean = [1.0 if np.isnan(x) else float(x) for x in pvals]
            try:
                _, p_adj, _, _ = multipletests(pvals_clean, method='fdr_bh')
                for i, p in enumerate(pairs):
                    pair_results[p].fdr_per_protein = float(p_adj[i])
            except Exception:
                for p in pairs:
                    pair_results[p].fdr_per_protein = 1.0

    # ── Global FDR ──
    all_pvals = []
    all_keys = []  # (protein, pair)
    for prot, pair_results in results.items():
        for pair, tg in pair_results.items():
            pv = tg.permutation_p_raw
            all_pvals.append(1.0 if np.isnan(pv) else float(pv))
            all_keys.append((prot, pair))

    if len(all_pvals) > 0 and not all(np.isnan(x) for x in all_pvals):
        try:
            _, p_adj_global, _, _ = multipletests(all_pvals, method='fdr_bh')
            for i, (prot, pair) in enumerate(all_keys):
                results[prot][pair].fdr_global = float(p_adj_global[i])
        except Exception:
            for prot, pair in all_keys:
                results[prot][pair].fdr_global = 1.0


# ═══════════════════════════════════════════════════════════════════
# Per-pair TG computation (internal)
# ═══════════════════════════════════════════════════════════════════

@_quiet_internal
def _compute_tg_pair(
    protein: str,
    source_cond: str,
    target_cond: str,
    batch: str,
    P_source: np.ndarray,
    P_target: np.ndarray,
    X_source: np.ndarray,
    X_target: np.ndarray,
    source_strata: np.ndarray,
    target_strata: np.ndarray,
    full_design: pd.DataFrame,
    cv_alphas: List[float],
    n_permutations: int,
    n_bootstrap: int,
    min_n: int,
    base_seed: int = 42,
    source_groups: Optional[np.ndarray] = None,
    target_groups: Optional[np.ndarray] = None,
    n_subsamples: int = 100,
) -> TGResult:
    """
    Compute full TG decomposition for one (protein, source→target, batch) triple.

    This is the workhorse function called by compute_transportability_gap().
    All internal scipy/numpy/sklearn warnings are scoped by @_quiet_internal
    (see _suppress_internal_warnings); intentional UserWarnings still surface.
    """
    # Group-aware flag
    group_aware = (source_groups is not None or target_groups is not None)

    n_source = len(P_source)
    n_target = len(P_target)
    size_ratio = float(n_source / max(n_target, 1))
    size_ratio_directional = size_ratio
    size_ratio_symmetric = float(max(n_source, n_target) / max(min(n_source, n_target), 1))

    # Per-pair deterministic RNG (not from parent RNG)
    # Derive a stable, pair-specific seed from the shared M1 helper.
    per_pair_seed = derive_seed(f'{protein}_{source_cond}_{target_cond}_{batch}')
    pair_rng = np.random.default_rng(per_pair_seed)

    # ── Quality gate: Cramér's V on FULL design (global condition×batch) ──
    # NOT per-pair within a single batch (which is always 0 because batch is constant).
    cramers_v = _compute_cramers_v(
        full_design['condition'].values.astype(str),
        full_design['batch'].values.astype(str),
    )

    # ── JS divergence between source and target RNA ──
    js_div = _compute_js_divergence(X_source, X_target)

    # ── Check basic estimability ──
    n_est_check = (n_source >= min_n) and (n_target >= min_n)
    conf_check = cramers_v < 1.0 - 1e-12

    # ── Compute stratum baseline for target ──
    _, mse_stratum_b = _compute_stratum_means_loocv(P_target, target_strata, groups=target_groups)

    # ── Q²_within_b (matched-subsample CV with LOOCV fallback for small n) ──
    n_subsamples = n_subsamples
    # Training size: match n_source when source is smaller, else use half of target.
    # Minimum train_size=8 prevents numerical catastrophe at n<16 where stratum
    # splits can produce degenerate (single-sample) strata within a subsample,
    # causing StandardScaler to divide by zero and MSE to become unstable.
    # n_subsamples is passed from the function parameter (default 100).
    train_size = min(n_source, max(n_target // 2, n_target - 10))  # match source, min 10 test
    if train_size >= 8 and n_target - train_size >= 3:
        cv_mode = 'matched_subsample'
        q2_within_b, _, mse_within_b, avg_edf, best_alpha = _compute_q2_within_matched(
            P_target, X_target, target_strata, train_size, cv_alphas,
            n_subsamples=n_subsamples, seed=per_pair_seed,
            groups_target=target_groups,
        )
    else:
        cv_mode = 'loocv'
        q2_within_b, _, mse_within_b, avg_edf, best_alpha = _compute_q2_within(
            P_target, X_target, target_strata, cv_alphas,
            groups_target=target_groups,
        )

    # ── Check if within-b model is useful ──
    q2_within_valid = (not np.isnan(q2_within_b)) and (q2_within_b >= 0)
    estimable = n_est_check and conf_check and q2_within_valid

    if not estimable:
        # Return NOT_ESTIMABLE result with minimal computations
        weak_bl = (np.isnan(q2_within_b) or q2_within_b < 0.1 or
                   n_target < 15 or avg_edf > max(n_target / 2, 1))
        asym = (size_ratio > 2.0 or size_ratio < 0.5)

        primary, secondary, text = _interpret_tg(
            estimable=False, weak_baseline=weak_bl, asymmetric=asym,
            tg_raw=float('nan'), q2_within_b=q2_within_b,
            permutation_p_raw=1.0, tg_design_fraction=float('nan'),
            theta=0.05,
        )

        return TGResult(
            protein=protein, source_condition=source_cond,
            target_condition=target_cond, batch=batch,
            n_source=n_source, n_target=n_target, size_ratio=size_ratio,
            size_ratio_directional=size_ratio_directional,
            size_ratio_symmetric=size_ratio_symmetric,
            q2_within_b=q2_within_b if not np.isnan(q2_within_b) else float('-inf'),
            q2_a_to_b=float('nan'), tg_raw=float('nan'),
            tg_log=float('nan'),
            tg_relative=float('nan'),
            q2_crossed=float('nan'), tg_design=float('nan'),
            tg_rna=float('nan'), tg_design_fraction=float('nan'),
            mse_stratum_b=mse_stratum_b,
            permutation_p_raw=1.0, permutation_p_design=1.0,
            permutation_p_rna=1.0, interaction_pvalue=1.0, ztg=float('nan'),
            ci_lower=float('nan'), ci_upper=float('nan'),
            fdr_per_protein=1.0, fdr_global=1.0,
            group_aware=group_aware,
            ridge_alpha_within_b=best_alpha if not np.isnan(best_alpha) else 1.0,
            estimable=False, weak_baseline=weak_bl, asymmetric=asym,
            ridge_edf=avg_edf, cramers_v=cramers_v, js_divergence=js_div,
            interpretation_primary=primary,
            interpretation_secondary=secondary,
            interpretation_text=text,
            seed=per_pair_seed,
            n_perms=n_permutations, n_bootstrap=n_bootstrap,
            cv_mode=cv_mode,
            n_subsamples=n_subsamples if cv_mode == 'matched_subsample' else 0,
        )

    # ═══════════════════════════════════════════════════════════════
    # Estimable: full TG computation
    # ═══════════════════════════════════════════════════════════════

    # Q²_a→b: subsample the source cohort to the matched training size.
    q2_a_to_b, mse_a_to_b = _compute_q2_cross_matched(
        P_source, X_source, source_strata,
        P_target, X_target, target_strata,
        train_size, mse_stratum_b, cv_alphas,
        n_subsamples=n_subsamples, seed=per_pair_seed,
    )

    # ── TG_log: scale-free reporting/ranking companion ──
    # Scale-free (MSE ratio, log-compressed): immune to the TG_raw explosion
    # when MSE_stratum is tiny on Z-scored data.
    # Epsilon guard: floor MSE_within_b at 1e-10 to avoid log(0).
    if (not np.isnan(mse_a_to_b)) and (not np.isnan(mse_within_b)) and mse_a_to_b > 0.0:
        tg_log = float(np.log(mse_a_to_b / max(mse_within_b, 1e-10)))
    else:
        tg_log = float('nan')

    # ── Q²_crossed: pooled LOOCV on target ──
    P_pooled = np.concatenate([P_source, P_target]).astype(np.float64)
    X_pooled = np.concatenate([X_source, X_target], axis=0).astype(np.float64)
    strata_pooled = np.concatenate([source_strata, target_strata])
    n_src = len(P_source)
    target_idx = np.arange(n_src, n_src + n_target)

    groups_pooled = np.concatenate([source_groups, target_groups]) if source_groups is not None and target_groups is not None else None

    if cv_mode == 'matched_subsample':
        q2_crossed, mse_crossed, _ = _compute_q2_crossed_matched(
            P_pooled, X_pooled, strata_pooled,
            target_idx, train_size, mse_stratum_b, cv_alphas,
            n_subsamples=n_subsamples, seed=per_pair_seed,
        )
    else:
        q2_crossed, mse_crossed, _ = _compute_q2_crossed(
            P_pooled, X_pooled, strata_pooled,
            target_idx, mse_stratum_b, cv_alphas,
            fixed_alpha=best_alpha,
            groups_pooled=groups_pooled,
        )

    # ── TG decomposition ──
    if not np.isnan(q2_within_b) and not np.isnan(q2_a_to_b):
        tg_raw = float(q2_within_b - q2_a_to_b)
    else:
        tg_raw = float('nan')

    if not np.isnan(q2_within_b):
        tg_relative = float(tg_raw / max(q2_within_b, 0.01))
    else:
        tg_relative = float('nan')

    if not np.isnan(q2_within_b) and not np.isnan(q2_crossed):
        tg_design = float(q2_within_b - q2_crossed)
    else:
        tg_design = float('nan')

    if not np.isnan(q2_crossed) and not np.isnan(q2_a_to_b):
        tg_rna = float(q2_crossed - q2_a_to_b)
    else:
        tg_rna = float('nan')

    if not np.isnan(tg_raw) and abs(tg_raw) > 1e-10:
        tg_design_fraction = float(tg_design / tg_raw)
        # Clamp to [0, 1] for interpretability
        tg_design_fraction = max(0.0, min(1.0, tg_design_fraction))
    else:
        tg_design_fraction = 0.5  # neutral when TG_raw is near zero

    # ── TG decomposition identity check ──
    if not np.isnan(tg_raw) and not np.isnan(tg_design) and not np.isnan(tg_rna):
        if not abs(tg_raw - tg_design - tg_rna) < 1e-4:
            warnings.warn(
                f"TG decomposition violated: {tg_raw} != {tg_design} + {tg_rna} "
                f"(diff={abs(tg_raw - tg_design - tg_rna):.6g})"
            )

    # ── Quality flags ──
    weak_baseline = (
        (not np.isnan(q2_within_b) and q2_within_b < 0.1) or
        n_target < 15 or
        avg_edf > max(n_target / 2.0, 1.0)
    )
    asymmetric = (size_ratio > 2.0 or size_ratio < 0.5)

    # ── Permutation test (decision significance test; p_raw/p_design/p_rna
    #    feed _interpret_tg — see TGResult) ──
    p_raw, p_design, p_rna, theta, null_raw = _permutation_test_tg(
        P_source, P_target, X_source, X_target,
        source_strata, target_strata,
        cv_alphas, n_permutations, pair_rng,
        tg_raw, tg_design, tg_rna,
        best_alpha=best_alpha,
        source_groups=source_groups,
        target_groups=target_groups,
        cv_mode=cv_mode,
        n_subsamples=n_subsamples,
        n_source=n_source,
        train_size=train_size,
        matched_seed=per_pair_seed,
    )

    # Interaction P value (supplementary, report-only column;
    #    the decision significance test is the permutation test above) ──
    P_pooled = np.concatenate([P_source, P_target]).astype(np.float64)
    X_pooled = np.concatenate([X_source, X_target], axis=0).astype(np.float64)
    cond_labels = np.concatenate([
        np.zeros(len(P_source), dtype=int),
        np.ones(len(P_target), dtype=int),
    ])
    interaction_pvalue = _compute_interaction_pvalue(
        P_pooled, X_pooled, cond_labels, train_size, cv_alphas,
        seed=per_pair_seed, n_permutations=n_permutations,
    )

    # ── zTG from permutation null distribution ──
    null_valid = null_raw[~np.isnan(null_raw)]
    if len(null_valid) >= 10:
        null_mean = float(np.mean(null_valid))
        null_std = float(np.std(null_valid))
        ztg = (tg_raw - null_mean) / null_std if null_std > 1e-10 else float('nan')
    else:
        ztg = float('nan')

    # ── Bootstrap CI (uses same CV mode as main analysis) ──
    ci_lower, ci_upper, ztg_boot = _bootstrap_ci_tg(
        P_source, P_target, X_source, X_target,
        source_strata, target_strata,
        cv_alphas, n_bootstrap, pair_rng,
        mse_stratum_b, n_target,
        source_groups=source_groups,
        target_groups=target_groups,
        cv_mode=cv_mode,
        train_size=train_size,
        n_subsamples=n_subsamples,
        matched_seed=per_pair_seed,
    )

    # ── Interpretation ──
    primary, secondary, text = _interpret_tg(
        estimable=True, weak_baseline=weak_baseline, asymmetric=asymmetric,
        tg_raw=tg_raw, q2_within_b=q2_within_b,
        permutation_p_raw=p_raw,
        tg_design_fraction=tg_design_fraction,
        theta=theta,
        ci_upper=ci_upper,
    )

    # Deterministic per-pair seed for repro (computed early at top)
    seed_val = per_pair_seed

    return TGResult(
        protein=protein, source_condition=source_cond,
        target_condition=target_cond, batch=batch,
        n_source=n_source, n_target=n_target, size_ratio=size_ratio,
        size_ratio_directional=size_ratio_directional,
        size_ratio_symmetric=size_ratio_symmetric,
        q2_within_b=q2_within_b,
        q2_a_to_b=q2_a_to_b if not np.isnan(q2_a_to_b) else float('-inf'),
        tg_raw=tg_raw,
        tg_log=tg_log,
        tg_relative=tg_relative,
        q2_crossed=q2_crossed,
        tg_design=tg_design,
        tg_rna=tg_rna,
        tg_design_fraction=tg_design_fraction,
        mse_stratum_b=mse_stratum_b,
        permutation_p_raw=p_raw,
        permutation_p_design=p_design,
        permutation_p_rna=p_rna,
        interaction_pvalue=interaction_pvalue,
        ztg=ztg,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        fdr_per_protein=1.0,    # filled in by _apply_tg_fdr
        fdr_global=1.0,         # filled in by _apply_tg_fdr
        group_aware=group_aware,
        estimable=True,
        weak_baseline=weak_baseline,
        asymmetric=asymmetric,
        ridge_edf=avg_edf,
        ridge_alpha_within_b=best_alpha,
        cramers_v=cramers_v,
        js_divergence=js_div,
        interpretation_primary=primary,
        interpretation_secondary=secondary,
        interpretation_text=text,
        seed=seed_val,
        n_perms=n_permutations,
        n_bootstrap=n_bootstrap,
        cv_mode=cv_mode,
        n_subsamples=n_subsamples if cv_mode == 'matched_subsample' else 0,
    )


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def compute_transportability_gap(
    rna_data: Dict[str, np.ndarray],
    protein_data: Dict[str, np.ndarray],
    design: pd.DataFrame,
    n_permutations: int = 100,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cv_alphas: List[float] = None,
    min_n: int = 8,
    groups: Optional[Union[str, np.ndarray]] = None,
    n_subsamples: int = 100,
) -> Dict[str, Dict[Tuple[str, str], TGResult]]:
    """
    Compute Transportability Gap for all proteins × within-batch condition pairs.

    For each protein m:
      1. Identify all within-batch condition pairs (a, b) where n_a >= min_n
         and n_b >= min_n.
      2. Compute TG_raw (primary decision scale), TG_log (scale-free reporting
         and ranking companion), TG_design, and TG_RNA.
      3. Permutation test (condition label shuffle within batch).
      4. Bootstrap CI for TG_raw.
      5. Quality gates and interpretation.
      6. BH-FDR correction (per-protein and global).

    Parameters
    ----------
    rna_data : dict {protein: np.ndarray (n_samples,)}
        RNA module scores. For protein m, RNA features = all OTHER proteins'
        module scores (same as IB).
    protein_data : dict {protein: np.ndarray (n_samples,)}
        Protein target values. Must have same keys as rna_data.
    design : pd.DataFrame (n_samples, 2+)
        Must contain columns ['condition', 'batch'].
    n_permutations : int
        Number of permutations for significance testing.
    n_bootstrap : int
        Number of bootstrap samples for CI.
    seed : int
        Random seed for reproducibility.
    cv_alphas : list of float or None
        Ridge regularization strengths for inner CV.
        Default: [0.01, 0.1, 1.0, 10.0, 100.0].
    min_n : int
        Minimum sample size per condition+batch group. Pairs below this
        threshold are NOT_ESTIMABLE.
    groups : str or np.ndarray or None
        Patient/donor group IDs for group-aware LOOCV. When a string is
        provided, it is treated as a column name in the design matrix.
        When a numpy array is provided, it must have length == n_samples.
        When None, auto-detects columns named 'group', 'patient', or 'donor'
        in the design matrix.
    n_subsamples : int
        Number of subsamples for stratum-mean LOOCV. Larger values yield
        more stable Q² estimates at the cost of runtime.

    Returns
    -------
    dict {protein: {(source_cond, target_cond): TGResult}}
        Only within-batch condition pairs where both conditions have
        >= min_n samples are computed.
    """
    if cv_alphas is None:
        cv_alphas = [0.1, 1.0, 10.0]  # Z-score scale: 3 alphas for speed

    protein_names = sorted(protein_data.keys())

    if len(protein_names) == 0:
        return {}

    n_samples = len(design)
    _verify_input_consistency(rna_data, protein_data, design)

    # Ensure float64 throughout
    design = design.copy()

    # ── Pre-compute stratum labels ──
    strata = (design['condition'].astype(str) + '_' +
              design['batch'].astype(str)).values

    # ── Resolve group IDs ──
    group_arr: Optional[np.ndarray] = None
    if groups is not None:
        if isinstance(groups, str):
            if groups in design.columns:
                group_arr = design[groups].values.astype(str)
            else:
                raise ValueError(
                    f"groups column '{groups}' not found in design matrix. "
                    f"Available: {list(design.columns)}"
                )
        elif isinstance(groups, np.ndarray):
            if len(groups) != len(design):
                raise ValueError(
                    f"groups array length {len(groups)} != design rows {len(design)}"
                )
            group_arr = groups.astype(str)
        else:
            raise TypeError(f"groups must be str, np.ndarray, or None, got {type(groups)}")
    else:
        # Auto-detect common group column names
        for col in ['group', 'patient', 'donor']:
            if col in design.columns:
                group_arr = design[col].values.astype(str)
                break

    results: Dict[str, Dict[Tuple[str, str], TGResult]] = {}

    # ── Multi-batch guard ──
    unique_batches = design['batch'].unique()
    if len(unique_batches) > 1:
        import warnings
        warnings.warn(
            f"Design has {len(unique_batches)} batches ({list(unique_batches)}). "
            f"TG is computed within each batch separately. "
            f"When multiple batches contain the same condition pair, "
            f"only the LAST batch's result is retained. "
            f"For multi-batch designs, use the 'batch' parameter to select a "
            f"single batch, or run separately per batch. "
            f"See docstring for details."
        )

    for prot in protein_names:
        P = protein_data[prot].astype(np.float64)

        # RNA features: all OTHER proteins
        other_prots = [p for p in protein_names if p != prot]
        if len(other_prots) == 0:
            continue

        X = np.column_stack([
            rna_data[p].astype(np.float64) for p in other_prots
        ])

        results[prot] = {}

        # ── Iterate over batches → within-batch condition pairs ──
        for batch_val in unique_batches:
            batch_mask = design['batch'] == batch_val
            batch_conditions = design.loc[batch_mask, 'condition'].unique()

            if len(batch_conditions) < 2:
                # Need at least 2 conditions in the batch for a pair
                continue

            for i, src_cond in enumerate(batch_conditions):
                for j, tgt_cond in enumerate(batch_conditions):
                    if i == j:
                        continue

                    # Extract source indices
                    src_mask = ((design['condition'] == src_cond) &
                                (design['batch'] == batch_val))
                    tgt_mask = ((design['condition'] == tgt_cond) &
                                (design['batch'] == batch_val))

                    src_idx = np.where(src_mask.values)[0] if hasattr(src_mask, 'values') else np.where(src_mask)[0]
                    tgt_idx = np.where(tgt_mask.values)[0] if hasattr(tgt_mask, 'values') else np.where(tgt_mask)[0]

                    n_source = len(src_idx)
                    n_target = len(tgt_idx)

                    if n_source < min_n or n_target < min_n:
                        continue

                    # Extract data
                    P_source = P[src_idx].astype(np.float64)
                    P_target = P[tgt_idx].astype(np.float64)
                    X_source = X[src_idx].astype(np.float64)
                    X_target = X[tgt_idx].astype(np.float64)

                    source_strata = strata[src_idx]
                    target_strata = strata[tgt_idx]

                    # Slice group IDs
                    source_groups = group_arr[src_idx] if group_arr is not None else None
                    target_groups = group_arr[tgt_idx] if group_arr is not None else None

                    # Compute TG
                    tg = _compute_tg_pair(
                        protein=prot,
                        source_cond=str(src_cond),
                        target_cond=str(tgt_cond),
                        batch=str(batch_val),
                        P_source=P_source, P_target=P_target,
                        X_source=X_source, X_target=X_target,
                        source_strata=source_strata,
                        target_strata=target_strata,
                        full_design=design,
                        cv_alphas=cv_alphas,
                        n_permutations=n_permutations,
                        n_bootstrap=n_bootstrap,
                        min_n=min_n,
                        base_seed=seed,
                        source_groups=source_groups,
                        target_groups=target_groups,
                        n_subsamples=n_subsamples,
                    )
                    results[prot][(str(src_cond), str(tgt_cond))] = tg

    # ── Apply FDR corrections ──
    _apply_tg_fdr(results)

    return results


# ═══════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════

def _verify_input_consistency(
    rna_data: Dict[str, np.ndarray],
    protein_data: Dict[str, np.ndarray],
    design: pd.DataFrame,
) -> None:
    """Verify that all inputs are consistent in shape and content."""
    required_cols = ['condition', 'batch']
    for col in required_cols:
        if col not in design.columns:
            raise ValueError(f"Design matrix must contain column '{col}'")

    n = len(design)
    if n == 0:
        raise ValueError("Design matrix is empty")

    common_prots = set(rna_data.keys()) & set(protein_data.keys())
    if len(common_prots) == 0:
        raise ValueError(
            "No common proteins between rna_data and protein_data"
        )

    for prot in common_prots:
        rn = len(rna_data[prot])
        pn = len(protein_data[prot])
        if rn == 0:
            raise ValueError(
                f"rna_data['{prot}'] is empty"
            )
        if pn == 0:
            raise ValueError(
                f"protein_data['{prot}'] is empty"
            )
        if rn != n:
            raise ValueError(
                f"rna_data['{prot}'] has {rn} samples, expected {n}"
            )
        if pn != n:
            raise ValueError(
                f"protein_data['{prot}'] has {pn} samples, expected {n}"
            )
        if np.any(np.isnan(rna_data[prot])):
            raise ValueError(
                f"rna_data['{prot}'] contains NaN values"
            )
        if np.any(np.isnan(protein_data[prot])):
            raise ValueError(
                f"protein_data['{prot}'] contains NaN values"
            )


# ═══════════════════════════════════════════════════════════════════
# Matrix and ensemble summary
# ═══════════════════════════════════════════════════════════════════

def compute_pairwise_tg_matrix(
    protein: str,
    pair_results: Dict[Tuple[str, str], TGResult],
) -> TGMatrixResult:
    """
    Aggregate TG results across all condition pairs for one protein.

    Parameters
    ----------
    protein : str
        Protein name.
    pair_results : dict {(source, target): TGResult}
        Results from compute_transportability_gap() for one protein.

    Returns
    -------
    TGMatrixResult
    """
    conditions = sorted(set(
        c for pair in pair_results.keys() for c in pair
    ))

    n_valid = 0
    n_transportable = 0
    n_non_transportable = 0
    tg_vals = []

    worst_pair = (None, None)
    worst_tg = -float('inf')

    for (src, tgt), tg in pair_results.items():
        if tg.estimable:
            n_valid += 1
            if tg.interpretation_primary == 'TRANSPORTABLE':
                n_transportable += 1
            elif tg.interpretation_primary == 'NON_TRANSPORTABLE':
                n_non_transportable += 1

            if not np.isnan(tg.tg_raw):
                tg_vals.append(tg.tg_raw)
                if tg.tg_raw > worst_tg:
                    worst_tg = tg.tg_raw
                    worst_pair = (src, tgt)

    mean_tg = float(np.mean(tg_vals)) if tg_vals else float('nan')
    max_tg = float(np.max(tg_vals)) if tg_vals else float('nan')

    if worst_pair[0] is None:
        worst_pair = ('', '')

    return TGMatrixResult(
        protein=protein,
        conditions=conditions,
        matrix=pair_results,
        n_valid_pairs=n_valid,
        n_transportable=n_transportable,
        n_non_transportable=n_non_transportable,
        mean_tg_raw=mean_tg,
        max_tg_raw=max_tg,
        worst_pair=worst_pair,
    )


def ensemble_summary(
    all_results: Dict[str, Dict[Tuple[str, str], TGResult]],
) -> TGEnsembleResult:
    """
    Aggregate TG results across all proteins.

    Parameters
    ----------
    all_results : dict {protein: {(source, target): TGResult}}
        Output from compute_transportability_gap().

    Returns
    -------
    TGEnsembleResult
    """
    proteins = sorted(all_results.keys())

    conditions = sorted(set(
        c for prot_results in all_results.values()
        for pair in prot_results.keys()
        for c in pair
    ))

    # Per-protein matrices
    per_protein: Dict[str, TGMatrixResult] = {}
    for prot in proteins:
        per_protein[prot] = compute_pairwise_tg_matrix(
            prot, all_results[prot]
        )

    # Cross-protein summary matrices
    n_cond = len(conditions)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}

    mean_matrix = np.full((n_cond, n_cond), np.nan, dtype=np.float64)
    transportable_fraction = np.full((n_cond, n_cond), np.nan, dtype=np.float64)
    sum_matrix = np.zeros((n_cond, n_cond), dtype=np.float64)
    count_matrix = np.zeros((n_cond, n_cond), dtype=np.float64)
    tp_count = np.zeros((n_cond, n_cond), dtype=np.float64)

    for prot in proteins:
        for (src, tgt), tg in all_results[prot].items():
            if src in cond_to_idx and tgt in cond_to_idx:
                i, j = cond_to_idx[src], cond_to_idx[tgt]
                if tg.estimable and not np.isnan(tg.tg_raw):
                    sum_matrix[i, j] += tg.tg_raw
                    count_matrix[i, j] += 1.0
                    if tg.interpretation_primary == 'TRANSPORTABLE':
                        tp_count[i, j] += 1.0

    for i in range(n_cond):
        for j in range(n_cond):
            if count_matrix[i, j] > 0:
                mean_matrix[i, j] = sum_matrix[i, j] / count_matrix[i, j]
                transportable_fraction[i, j] = (
                    tp_count[i, j] / count_matrix[i, j]
                )

    n_total = sum(len(r) for r in all_results.values())
    n_estimable = sum(
        sum(1 for tg in r.values() if tg.estimable)
        for r in all_results.values()
    )

    return TGEnsembleResult(
        proteins=proteins,
        conditions=conditions,
        per_protein=per_protein,
        mean_matrix=mean_matrix,
        transportable_fraction_matrix=transportable_fraction,
        n_total_pairs=n_total,
        n_estimable_pairs=n_estimable,
    )


# ═══════════════════════════════════════════════════════════════════
# DataFrame export helpers
# ═══════════════════════════════════════════════════════════════════

def tg_results_to_dataframe(
    all_results: Dict[str, Dict[Tuple[str, str], TGResult]],
) -> pd.DataFrame:
    """
    Flatten TG results to a DataFrame for CSV export and visualization.

    Parameters
    ----------
    all_results : dict {protein: {(source, target): TGResult}}
        Output from compute_transportability_gap().

    Returns
    -------
    pd.DataFrame with one row per (protein, source, target) triple.
    """
    rows = []
    for prot, pair_results in all_results.items():
        for (src, tgt), tg in pair_results.items():
            rows.append({
                'protein': prot,
                'source_condition': src,
                'target_condition': tgt,
                'batch': tg.batch,
                'n_source': tg.n_source,
                'n_target': tg.n_target,
                'size_ratio': round(tg.size_ratio, 4),
                'size_ratio_directional': round(tg.size_ratio_directional, 4),
                'size_ratio_symmetric': round(tg.size_ratio_symmetric, 4),
                'q2_within_b': round(tg.q2_within_b, 4),
                'q2_a_to_b': round(tg.q2_a_to_b, 4),
                'tg_raw': round(tg.tg_raw, 6),
                'tg_log': round(tg.tg_log, 6),
                'tg_relative': round(tg.tg_relative, 4),
                'q2_crossed': round(tg.q2_crossed, 4),
                'tg_design': round(tg.tg_design, 6),
                'tg_cross_study_residual': round(tg.tg_cross_study_residual, 6),
                'tg_rna': round(tg.tg_rna, 6),  # Deprecated compatibility alias
                'tg_design_fraction': round(tg.tg_design_fraction, 4),
                'mse_stratum_b': round(tg.mse_stratum_b, 6),
                'permutation_p_raw': round(tg.permutation_p_raw, 4),
                'interaction_pvalue': round(tg.interaction_pvalue, 4),
                'permutation_p_design': round(tg.permutation_p_design, 4),
                'permutation_p_rna': round(tg.permutation_p_rna, 4),
                'ztg': round(tg.ztg, 4) if not np.isnan(tg.ztg) else '',
                'ci_lower': round(tg.ci_lower, 6) if not np.isnan(tg.ci_lower) else '',
                'ci_upper': round(tg.ci_upper, 6) if not np.isnan(tg.ci_upper) else '',
                'fdr_per_protein': round(tg.fdr_per_protein, 4),
                'fdr_global': round(tg.fdr_global, 4),
                'estimable': tg.estimable,
                'weak_baseline': tg.weak_baseline,
                'asymmetric': tg.asymmetric,
                'ridge_edf': round(tg.ridge_edf, 2),
                'ridge_alpha_within_b': round(tg.ridge_alpha_within_b, 4),
                'cramers_v': round(tg.cramers_v, 4),
                'js_divergence': round(tg.js_divergence, 4),
                'interpretation_primary': tg.interpretation_primary,
                'interpretation_secondary': tg.interpretation_secondary,
                'interpretation_text': tg.interpretation_text,
                'seed': tg.seed,
                'cv_mode': tg.cv_mode,
                'n_subsamples': tg.n_subsamples,
            })
    return pd.DataFrame(rows)


def tg_ensemble_to_dataframe(ensemble: TGEnsembleResult) -> Dict[str, pd.DataFrame]:
    """
    Convert TGEnsembleResult to a dict of DataFrames for export.

    Returns
    -------
    dict with keys:
      - 'summary_matrix': mean TG_raw matrix (conditions × conditions)
      - 'transportable_fraction': fraction TRANSPORTABLE per pair
      - 'per_protein': per-protein summary
    """
    conditions = ensemble.conditions

    # Mean TG_raw matrix
    mean_df = pd.DataFrame(
        ensemble.mean_matrix,
        index=conditions, columns=conditions,
    )
    mean_df.index.name = 'source'
    mean_df.columns.name = 'target'

    # Transportable fraction matrix
    frac_df = pd.DataFrame(
        ensemble.transportable_fraction_matrix,
        index=conditions, columns=conditions,
    )
    frac_df.index.name = 'source'
    frac_df.columns.name = 'target'

    # Per-protein summary
    pp_rows = []
    for prot, matrix in ensemble.per_protein.items():
        pp_rows.append({
            'protein': prot,
            'n_valid_pairs': matrix.n_valid_pairs,
            'n_transportable': matrix.n_transportable,
            'n_non_transportable': matrix.n_non_transportable,
            'mean_tg_raw': round(matrix.mean_tg_raw, 6)
                if not np.isnan(matrix.mean_tg_raw) else '',
            'max_tg_raw': round(matrix.max_tg_raw, 6)
                if not np.isnan(matrix.max_tg_raw) else '',
            'worst_pair': f"{matrix.worst_pair[0]}→{matrix.worst_pair[1]}",
        })
    pp_df = pd.DataFrame(pp_rows)

    return {
        'summary_matrix': mean_df,
        'transportable_fraction': frac_df,
        'per_protein': pp_df,
    }


# ═══════════════════════════════════════════════════════════════════
# cordiag public API aliases
# ═══════════════════════════════════════════════════════════════════
# Public aliases reference the same function objects as their private names.
compute_tg_pair = _compute_tg_pair
apply_tg_fdr = _apply_tg_fdr
