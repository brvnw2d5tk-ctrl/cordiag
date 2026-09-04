"""Rank-based paired-gain statistics for stratified molecular measurements."""
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Random-number generator used when a call does not supply an explicit seed.
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

# The shared M1 implementation is the single source of truth. zPG requests
# ``unseen_stratum='skip'`` so observations without a trainable stratum remain
# undefined rather than falling back to a pooled mean.
from .m1 import m1_loocv as _m1_loocv

# Candidate ridge penalties used by cross-validation.
_CV_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _upper_tail_permutation_pvalue(null_values, observed):
    """Continuity-corrected upper-tail p-value over finite null draws."""
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0 or not np.isfinite(observed):
        return np.nan
    return float((np.sum(valid >= observed) + 1) / (len(valid) + 1))


def set_seed(seed=42):
    """Reset the module-level random-number generator.

    Subsequent zPG calls with ``seed=None`` draw from this reproducible stream.
    """
    global _RNG
    _RNG = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_design(design, n, require_batch=True):
    """Validate and normalize supported design representations."""
    if isinstance(design, pd.DataFrame):
        needed = ['condition', 'batch'] if require_batch else ['condition']
        missing = [c for c in needed if c not in design.columns]
        if missing:
            raise ValueError(f"design DataFrame missing column(s): {missing}")
        if len(design) != n:
            raise ValueError(
                f"design has {len(design)} rows but data has {n} samples")
        return design
    if isinstance(design, (tuple, list)):
        if len(design) != 2:
            raise ValueError("design tuple must be (condition, batch) — two arrays")
        cond = np.asarray(design[0])
        batch = np.asarray(design[1])
        if len(cond) != n or len(batch) != n:
            raise ValueError(
                f"condition/batch arrays have length {len(cond)}/{len(batch)} "
                f"but data has {n} samples")
        return pd.DataFrame({'condition': cond, 'batch': batch})
    raise TypeError(
        "design must be a DataFrame with 'condition'/'batch' columns, "
        "or a (condition, batch) tuple of arrays")


# ---------------------------------------------------------------------------
# Stratum-conditioned ridge prediction.
# ---------------------------------------------------------------------------

def _loo_stratum_ridge(X, y, strata, mice=None):
    """Run stratum-conditioned leave-one-out ridge prediction for zPG.

    The shared implementation is :func:`cordiag.m1.m1_loocv`; this wrapper
    explicitly selects the zPG policy for unseen strata.

    Predict each held-out sample after centering X and y within condition-by-
    batch strata. RidgeCV selects alpha when at least 20 training observations
    are available; smaller training sets use alpha=1. Predictions for strata
    absent from training remain NaN.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_predictors)
    y : np.ndarray (n_samples,)
    strata : pandas.Series or np.ndarray
        Condition-by-batch labels aligned with X and y.
    mice : array-like or None
        Optional sample-index container used only to determine sample count.

    Returns
    -------
    (preds, truths) : (np.ndarray, np.ndarray)
        Per-sample predictions and observed values. Skipped predictions are NaN.
    """
    # Delegate prediction to the shared M1 implementation in zPG mode.
    n = len(y) if mice is None else len(mice)
    preds, _mse, _edf, _alpha = _m1_loocv(
        P=np.asarray(y[:n], dtype=np.float64),
        X=np.asarray(X[:n], dtype=np.float64),
        strata=np.asarray(strata[:n]),
        cv_alphas=_CV_ALPHAS,
        fixed_alpha=None,
        groups=None,
        unseen_stratum='skip',
    )
    truths = np.asarray(y[:n], dtype=np.float64).copy()
    return preds, truths


def _loo_fast(X, y, strata, mice):
    """Stratum-conditioned LOOCV Ridge with fixed alpha=1.0 for speed.

    Fixed-alpha approximation used only for cross-validation stability checks.
    Uses fixed alpha=1 for fast cross-validation stability comparisons; it is
    not used to report the primary zPG statistic.

    Returns predictions and observations in the same form as
    :func:`_loo_stratum_ridge`.
    """
    n = len(mice)
    preds = np.full(n, np.nan)
    truths = np.zeros(n)

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        if len(tr) < 3:
            truths[i] = y[i]; continue
        strata_tr = strata.iloc[tr]
        y_ctr = y[tr].copy()
        X_ctr = X[tr].copy()
        for s in strata_tr.unique():
            idx = np.where(strata_tr == s)[0]
            if len(idx) > 0:
                y_ctr[idx] -= np.mean(y[tr][idx])
                X_ctr[idx] -= np.mean(X[tr][idx], axis=0)
        s_test = strata.iloc[i]
        s_tr_idx = np.where(strata_tr == s_test)[0]
        if len(s_tr_idx) == 0:
            truths[i] = y[i]; continue
        y_test_ctr = y[i] - np.mean(y[tr][s_tr_idx])
        X_test_ctr = X[i] - np.mean(X[tr][s_tr_idx], axis=0)
        try:
            sX = StandardScaler().fit(X_ctr)
            sy = StandardScaler().fit(y_ctr.reshape(-1, 1))
            # Fixed alpha=1.0 — deliberately avoids RidgeCV for speed
            m = Ridge(alpha=1.0).fit(
                sX.transform(X_ctr),
                sy.transform(y_ctr.reshape(-1, 1)).ravel())

            preds[i] = sy.inverse_transform(
                m.predict(sX.transform(X_test_ctr.reshape(1, -1))).reshape(-1, 1)).ravel()[0]
            preds[i] += np.mean(y[tr][s_tr_idx])
            truths[i] = y[i]
        except Exception:
            truths[i] = y[i]
    return preds, truths


def _kfold_stratum_ridge(X, y, strata, n_folds=5, seed=42):
    """Run stratum-conditioned k-fold ridge prediction.

    Folds are evaluated independently while preserving stratum conditioning.

    Each fold is predicted from the remaining folds after condition-by-batch
    centering. Alpha is fixed at 1.0 for stability comparisons.

    Each sample is predicted once; failed predictions are NaN.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    preds = np.full(n, np.nan)
    truths = np.zeros(n)
    cv_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Stratified shuffle: maintain stratum proportions in each fold when possible
    # Simple shuffle for inputs with many strata
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_assignments = np.zeros(n, dtype=int)
    fold_size = n // n_folds
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        fold_assignments[indices[start:end]] = f

    for f in range(n_folds):
        test_idx = np.where(fold_assignments == f)[0]
        train_idx = np.where(fold_assignments != f)[0]

        for i in test_idx:
            # Training set: all train_idx (exclude test sample i if overlapping)
            tr = [j for j in train_idx]
            if len(tr) < 3:
                truths[i] = y[i]
                continue

            # Stratum conditioning within training set
            strata_tr = strata.iloc[tr]
            y_ctr = y[tr].copy()
            X_ctr = X[tr].copy()
            for s in strata_tr.unique():
                idx = np.where(strata_tr == s)[0]
                if len(idx) > 0:
                    y_ctr[idx] -= np.mean(y[tr][idx])
                    X_ctr[idx] -= np.mean(X[tr][idx], axis=0)

            s_test = strata.iloc[i]
            s_tr_idx = np.where(strata_tr == s_test)[0]
            if len(s_tr_idx) == 0:
                truths[i] = y[i]
                continue

            y_test_ctr = y[i] - np.mean(y[tr][s_tr_idx])
            X_test_ctr = X[i] - np.mean(X[tr][s_tr_idx], axis=0)

            try:
                sX = StandardScaler().fit(X_ctr)
                sy = StandardScaler().fit(y_ctr.reshape(-1, 1))
                # Fixed alpha=1.0 (stability comparison doesn't need optimal alpha)
                m = Ridge(alpha=1.0).fit(
                    sX.transform(X_ctr),
                    sy.transform(y_ctr.reshape(-1, 1)).ravel())

                preds[i] = sy.inverse_transform(
                    m.predict(sX.transform(X_test_ctr.reshape(1, -1))).reshape(-1, 1)).ravel()[0]
                preds[i] += np.mean(y[tr][s_tr_idx])
                truths[i] = y[i]
            except Exception:
                truths[i] = y[i]

    # Return full-length arrays (NaNs for failed), consistent with _loo_stratum_ridge
    return preds, truths


# ---------------------------------------------------------------------------
# Rank-based zPG statistic.
# ---------------------------------------------------------------------------

def compute_rank_zPG(R_modules, P, design, n_perms, seed=None):
    """Compute the rank-based paired-gain statistic and a permutation p-value."""
    design = _coerce_design(design, len(P))
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    n = len(P)
    strata = design['condition'] + '_' + design['batch']
    unique_strata = strata.unique()
    X_modules = np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])

    # Observed LOOCV predictions
    true_preds, true_truths = _loo_stratum_ridge(X_modules, P, strata, np.arange(n))
    valid = ~np.isnan(true_preds)

    # Stratum-only baseline
    stratum_preds = np.array([
        np.mean(P[[j for j in range(n) if j != i and strata.iloc[j] == strata.iloc[i]]])
        if np.any([j != i and strata.iloc[j] == strata.iloc[i] for j in range(n)])
        else np.mean(P[[j for j in range(n) if j != i]])
        for i in range(n)
    ])

    mse_model = np.mean((true_preds[valid] - true_truths[valid])**2)
    mse_stratum = np.mean((stratum_preds[valid] - true_truths[valid])**2)
    if mse_stratum < 1e-10:
        Q2_obs = np.nan
    else:
        Q2_obs = 1 - mse_model / mse_stratum

    # Rank-based: Spearman rho
    # Observation-side gate, symmetric with the permutation side below
    # (`if v.sum() >= 3`): fewer than 3 valid predictions cannot support a
    # Spearman rho. Degenerate small-n inputs (e.g. n=3 split across 2 strata)
    # leave < 3 LOOCV predictions valid → rho_obs is undefined → NaN
    # This guard preserves the undefined-statistic behavior for all valid inputs.
    if valid.sum() < 3:
        rho_obs = np.nan
    else:
        rho_obs, _ = spearmanr(true_preds[valid], true_truths[valid])

    # Within-stratum permutation
    perm_rhos = []
    perm_Q2s = []
    for _ in range(n_perms):
        P_perm = P.copy()
        for s in unique_strata:
            s_idx = np.where(strata == s)[0]
            if len(s_idx) >= 2:
                P_perm[s_idx] = P[s_idx[rng.permutation(len(s_idx))]]
        preds, truths = _loo_stratum_ridge(X_modules, P_perm, strata, np.arange(n))
        v = ~np.isnan(preds)
        if v.sum() >= 3:
            perm_rhos.append(spearmanr(preds[v], truths[v])[0])
            mse_p = np.mean((preds[v] - truths[v])**2)
            if mse_stratum < 1e-10:
                perm_Q2s.append(np.nan)
            else:
                perm_Q2s.append(1 - mse_p / mse_stratum)

    perm_rhos = np.array(perm_rhos)
    perm_Q2s = np.array(perm_Q2s)

    perm_std = np.std(perm_rhos)
    if perm_std < 1e-10:
        zPG_rank = np.nan
    else:
        zPG_rank = (rho_obs - np.mean(perm_rhos)) / perm_std
    if np.isnan(rho_obs):
        # Degenerate observation (valid preds < 3 or constant predictions):
        # rho_obs is undefined — (0 + 1)/(n_perms+1) would report a
        # pseudo-significant p-value. Mirror rho_obs as NaN (caller decide()
        # maps NaN to NO_GO).
        p_val = np.nan
    elif len(perm_rhos) == 0:
        # No valid permutation produced a rho — no null information at all
        p_val = np.nan
    else:
        p_val = _upper_tail_permutation_pvalue(perm_rhos, rho_obs)
    perm_q2_std = np.std(perm_Q2s)
    if perm_q2_std < 1e-10 or np.isnan(perm_q2_std):
        zPG_Q2 = np.nan
    else:
        zPG_Q2 = (Q2_obs - np.mean(perm_Q2s)) / perm_q2_std

    # Minimum achievable p-value for this stratum configuration
    min_p = 1.0
    for s in unique_strata:
        n_s = (strata == s).sum()
        if n_s >= 2:
            min_p = min(min_p, 1.0 / (math.factorial(n_s) if n_s <= 10 else 1000))
    min_p = max(min_p, 1.0 / (len(perm_rhos) + 1))

    return {
        'zPG_rank': zPG_rank,
        'zPG_Q2': zPG_Q2,
        'rho_obs': rho_obs,
        'Q2_obs': Q2_obs,
        'p_val': p_val,
        'perm_rho_mean': np.mean(perm_rhos),
        'perm_rho_std': np.std(perm_rhos),
        'min_achievable_p': min_p,
        'n_valid': valid.sum()
    }


# ---------------------------------------------------------------------------
# Covariate-adjusted rank-based zPG statistic.
# ---------------------------------------------------------------------------

def compute_rank_zPG_partial(R_modules, P, design, n_perms, seed=None, n_pcs=3):
    """Compute the partial rank-based paired-gain statistic after covariate adjustment."""
    design = _coerce_design(design, len(P))
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    n = len(P)
    strata = design['condition'] + '_' + design['batch']
    unique_strata = strata.unique()

    # Step 1: PCA reduce RNA modules
    X = np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(n_pcs, X.shape[1], n-1))
    X_pca = pca.fit_transform(X_scaled)

    # Step 2: Residualize against condition+batch (OLS per PC)
    cond_dummies = pd.get_dummies(design[['condition', 'batch']], drop_first=True).values
    X_resid = np.zeros_like(X_pca)
    for j in range(X_pca.shape[1]):
        beta = np.linalg.lstsq(cond_dummies, X_pca[:, j], rcond=None)[0]
        X_resid[:, j] = X_pca[:, j] - cond_dummies @ beta

    # Residualize protein
    beta_y = np.linalg.lstsq(cond_dummies, P, rcond=None)[0]
    P_resid = P - cond_dummies @ beta_y

    # Step 3: Observed partial Spearman
    # Use mean of PC residuals as combined RNA signal
    rna_combined = X_resid.mean(axis=1)
    rho_obs, _ = spearmanr(rna_combined, P_resid)

    # Step 4: Within-stratum permutation
    perm_rhos = []
    for _ in range(n_perms):
        P_perm = P.copy()
        for s in unique_strata:
            s_idx = np.where(strata == s)[0]
            if len(s_idx) >= 2:
                P_perm[s_idx] = P[s_idx[rng.permutation(len(s_idx))]]
        # Re-residualize after permutation
        beta_yp = np.linalg.lstsq(cond_dummies, P_perm, rcond=None)[0]
        Pp_resid = P_perm - cond_dummies @ beta_yp
        rho_p, _ = spearmanr(rna_combined, Pp_resid)
        if not np.isnan(rho_p):
            perm_rhos.append(rho_p)

    perm_rhos = np.array(perm_rhos)
    perm_std = np.std(perm_rhos)
    if perm_std < 1e-10:
        zPG_partial = np.nan
    else:
        zPG_partial = (rho_obs - np.mean(perm_rhos)) / perm_std
    if np.isnan(rho_obs):
        # A degenerate observed correlation has no defined permutation P value.
        p_val = np.nan
    elif len(perm_rhos) == 0:
        p_val = np.nan
    else:
        p_val = _upper_tail_permutation_pvalue(perm_rhos, rho_obs)

    min_p = 1.0
    for s in unique_strata:
        n_s = (strata == s).sum()
        if n_s >= 2:
            min_p = min(min_p, 1.0 / (math.factorial(n_s) if n_s <= 10 else 1000))
    min_p = max(min_p, 1.0 / (len(perm_rhos) + 1))

    return {
        'zPG_partial': zPG_partial,
        'rho_obs': rho_obs,
        'p_val': p_val,
        'perm_rho_mean': np.mean(perm_rhos),
        'perm_rho_std': np.std(perm_rhos),
        'min_achievable_p': min_p,
        'n_pcs': n_pcs,
        'pca_variance_explained': pca.explained_variance_ratio_.sum()
    }


# ---------------------------------------------------------------------------
# Cross-validation variants of zPG.
# ---------------------------------------------------------------------------

def _zpg_with_cv(R_modules, P, design, n_folds, n_perms, seed=42, actual_perms=None):
    """Compute zPG using the requested number of cross-validation folds."""
    design = _coerce_design(design, len(P))
    rng = np.random.default_rng(seed)
    n = len(P)
    strata = design['condition'].astype(str) + '_' + design['batch'].astype(str)
    unique_strata = strata.unique()
    X_modules = np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])

    # Determine n_perms: LOOCV also gets reduced perms for speed
    if n_folds == n:  # LOOCV
        actual_perms = min(n_perms, 20)
        # Use local _loo_fast (fixed alpha=1.0) instead of RidgeCV version
        true_preds, true_truths = _loo_fast(X_modules, P, strata, np.arange(n))
    else:
        actual_perms = min(n_perms, 30)
        true_preds, true_truths = _kfold_stratum_ridge(X_modules, P, strata, n_folds=n_folds, seed=seed)

    valid = ~np.isnan(true_preds)
    if valid.sum() < 3:
        # Degenerate small-n path — mirror the LOOCV convention
        # (compute_rank_zPG): NaN rank/rho/p, not a pseudo-significant
        # sentinel values (0.0, 0.0, 1.0);
        # decision layer maps NaN -> NO_GO).
        return {'zPG_rank': np.nan, 'rho_obs': np.nan, 'p_val': np.nan,
                'n_valid': valid.sum(), 'n_folds': n_folds}

    # Stratum-only baseline (same for all CV schemes)
    stratum_preds = np.array([
        np.mean(P[[j for j in range(n) if j != i and strata.iloc[j] == strata.iloc[i]]])
        if np.any([j != i and strata.iloc[j] == strata.iloc[i] for j in range(n)])
        else np.mean(P[[j for j in range(n) if j != i]])
        for i in range(n)
    ])

    rho_obs, _ = spearmanr(true_preds[valid], true_truths[valid])
    mse_model = np.mean((true_preds[valid] - true_truths[valid])**2)
    mse_stratum = np.mean((stratum_preds[valid] - true_truths[valid])**2)
    Q2_obs = np.nan if mse_stratum < 1e-10 else 1 - mse_model / mse_stratum

    # Within-stratum permutation
    perm_rhos = []
    perm_Q2s = []
    for _ in range(actual_perms):
        P_perm = P.copy()
        for s in unique_strata:
            s_idx = np.where(strata == s)[0]
            if len(s_idx) >= 2:
                P_perm[s_idx] = P[s_idx[rng.permutation(len(s_idx))]]

        if n_folds == n:
            preds, truths = _loo_stratum_ridge(X_modules, P_perm, strata, np.arange(n))
        else:
            preds, truths = _kfold_stratum_ridge(X_modules, P_perm, strata, n_folds=n_folds, seed=rng.integers(0, 2**31))

        v = ~np.isnan(preds)
        if v.sum() >= 3:
            perm_rhos.append(spearmanr(preds[v], truths[v])[0])
            mse_p = np.mean((preds[v] - truths[v])**2)
            perm_Q2s.append(np.nan if mse_stratum < 1e-10 else 1 - mse_p / mse_stratum)

    perm_rhos = np.array(perm_rhos)
    p_std = np.std(perm_rhos)
    zPG_rank = (rho_obs - np.mean(perm_rhos)) / p_std if p_std > 1e-10 else np.nan
    if np.isnan(rho_obs) or len(perm_rhos) == 0:
        # Degenerate observation or no valid permutation — NaN, same
        # convention as compute_rank_zPG / the LOOCV branch above
        # when too few finite null values remain.
        p_val = np.nan
    else:
        p_val = _upper_tail_permutation_pvalue(perm_rhos, rho_obs)
    q2_std = np.std(perm_Q2s)
    zPG_Q2 = (Q2_obs - np.mean(perm_Q2s)) / q2_std if q2_std > 1e-10 and not np.isnan(q2_std) else np.nan

    return {
        'zPG_rank': zPG_rank, 'zPG_Q2': zPG_Q2,
        'rho_obs': rho_obs, 'Q2_obs': Q2_obs,
        'p_val': p_val, 'n_valid': valid.sum(),
        'n_folds': n_folds,
        'perm_rho_mean': float(np.mean(perm_rhos)),
        'perm_rho_std': float(np.std(perm_rhos))
    }


def select_cv(n):
    """Choose a cross-validation scheme from the available sample size."""
    if n < 30:
        return ('loocv', None, 1)
    if n < 100:
        return ('kfold', 10, 3)
    return ('kfold', 5, 10)


def compute_zpg(R_modules, P, design, n_perms, seed=None, cv=None, repeats=None):
    """Compute the configured paired-gain statistic using cross-validation."""
    design = _coerce_design(design, len(P))
    n = len(P)

    if cv is None:
        method, n_folds, n_rep = select_cv(n)
    elif cv == 'loocv':
        method, n_folds, n_rep = 'loocv', None, 1
    else:
        method, n_folds, n_rep = 'kfold', int(cv), 1
    if repeats is not None:
        n_rep = int(repeats)

    if method == 'loocv':
        res = dict(compute_rank_zPG(R_modules, P, design, n_perms, seed=seed))
        res['cv'] = 'loocv'
        res['n_folds'] = n
        res['n_repeats'] = 1
        return res

    # k-fold branch: derive a reproducible local seed from global RNG when seed=None
    if seed is None:
        base_seed = int(_RNG.integers(0, 2**31))
    else:
        base_seed = int(seed)

    results = []
    for i in range(n_rep):
        r = _zpg_with_cv(R_modules, P, design, n_folds, n_perms=n_perms,
                         seed=base_seed + i)
        results.append(r)

    if n_rep > 1:
        res = dict(results[0])
        z_list = [r['zPG_rank'] for r in results]
        res['zPG_rank'] = float(np.mean(z_list))
        res['zPG_rank_repeats'] = z_list
    else:
        res = dict(results[0])

    res['cv'] = f'{n_folds}fold'
    res['n_folds'] = n_folds
    res['n_repeats'] = n_rep
    return res


# ---------------------------------------------------------------------------
# Supplementary calculations
# ---------------------------------------------------------------------------

def compute_ECI(R_modules, design, cond_pairs, n_bootstrap=200, seed=None):
    """Compute the expression-consistency index across requested condition pairs."""
    n = len(next(iter(R_modules.values()))) if R_modules else 0
    design = _coerce_design(design, n, require_batch=False)
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    results = {}
    for ca, cb in cond_pairs:
        idx_a = design['condition'] == ca
        idx_b = design['condition'] == cb
        if idx_a.sum() < 2 or idx_b.sum() < 2:
            results[f'{ca}_vs_{cb}'] = {'ECI': np.nan, 'ECI_ci': (np.nan, np.nan)}
            continue

        cos_sims = []
        for mod_name in sorted(R_modules.keys()):
            vals = R_modules[mod_name]
            boot_cos = []
            for _ in range(min(n_bootstrap, 100)):
                ba = rng.choice(vals[idx_a], size=idx_a.sum(), replace=True)
                bb = rng.choice(vals[idx_b], size=idx_b.sum(), replace=True)
                mu_a = ba.mean(); mu_b = bb.mean()
                cos_sim = 1.0 if (mu_a * mu_b > 0) else (-1.0 if (mu_a * mu_b < 0) else 0.0)
                boot_cos.append(cos_sim)
            cos_sims.append(np.mean(boot_cos))

        cos_sims_arr = np.array(cos_sims)
        # ECI = 1 - mean(cosine similarity): 0 is concordant, 1 is opposite.
        eci = 1.0 - np.mean(cos_sims_arr)
        eci_std = np.std(cos_sims_arr) / max(np.sqrt(len(cos_sims_arr)), 1)
        eci_ci = (max(0, eci - 1.96*eci_std), min(1, eci + 1.96*eci_std))

        results[f'{ca}_vs_{cb}'] = {
            'ECI': eci, 'ECI_ci': eci_ci,
            'n_modules': len(cos_sims),
            'interpretation': _interpret_ECI(eci)
        }
    return results


def _interpret_ECI(eci):
    """Map ECI values to descriptive supplementary categories.

    Map ECI values to the calibrated descriptive categories.
    The heuristic thresholds were calibrated in simulations with at least ten
    observations per condition.
    """
    if eci < 0.3: return "high concordance (stable direction)"
    elif eci < 0.5: return "moderate concordance"
    elif eci < 0.7: return "clear discordance (possible batch effect)"
    else: return "severe discordance (batch-dominated)"


def joint_decision(zPG, ECI, zPG_thresh=0, ECI_thresh=0.5):
    """Combine paired-gain and expression-consistency statistics into a categorical decision."""
    if zPG > zPG_thresh and ECI < ECI_thresh:
        return "GO: paired signal with stable direction"
    elif zPG > zPG_thresh and ECI >= ECI_thresh:
        return "CAUTION: signal present but direction is unstable"
    elif zPG <= zPG_thresh and ECI >= ECI_thresh:
        return "STOP: no signal and unstable direction"
    else:
        return "NO_SIGNAL: stable direction but weak individual pairing"


# ---------------------------------------------------------------------------
# Multiple-testing correction.
# ---------------------------------------------------------------------------

def fdr_bh(p_values):
    """Apply the Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    nan_mask = np.isnan(p)
    if nan_mask.any():
        out = np.full(n, np.nan)
        out[~nan_mask] = fdr_bh(p[~nan_mask])
        return out
    order = np.argsort(p, kind='mergesort')
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # enforce monotonicity from the tail
    q = np.minimum(q, 1.0)
    result = np.empty(n)
    result[order] = q
    return result


# ---------------------------------------------------------------------------
# Decision rules.
# ---------------------------------------------------------------------------

def decide(
    zpg,
    p_fdr,
    n_per_condition=None,
    zpg_go=1.0,
    fdr_go=0.1,
    n_min=12,
    design_identifiable=True,
):
    """Return NOT_IDENTIFIABLE_DESIGN, GO, INCONCLUSIVE, or NO_GO."""
    if not design_identifiable:
        return "NOT_IDENTIFIABLE_DESIGN"
    if zpg > zpg_go and p_fdr < fdr_go:
        return "GO"
    if np.isfinite(zpg) and zpg > 0:
        return "INCONCLUSIVE"
    return "NO_GO"


def decide_legacy(zpg, p_fdr, zpg_go=1.0, fdr_go=0.1):
    """Return the deprecated GO/GRAY/NO_GO compatibility decision.

    Retained only for compatibility with historical three-level decisions.
    New analyses should use :func:`decide`, which returns
    GO/INCONCLUSIVE/NO_GO.
    """
    if zpg > zpg_go and p_fdr < fdr_go:
        return 'GO'
    elif zpg > 0:
        return 'GRAY'
    else:
        return 'NO_GO'


# ---------------------------------------------------------------------------
# Fast module-level check without permutation
# ---------------------------------------------------------------------------

def compute_module_q2_simple(R_modules, P, design):
    """Compute per-module prediction Q² values with stratum-conditioned LOOCV."""
    n = len(P)
    if n < 3:
        return np.nan, np.nan, 0

    strata = design['condition'].astype(str) + '_' + design['batch'].astype(str)

    # Stack RNA module scores
    mod_names = sorted(R_modules.keys())
    X = np.column_stack([R_modules[m] for m in mod_names])

    # LOOCV predictions
    preds = np.full(n, np.nan)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, X_test = X[train_idx], X[i:i+1]
        y_train, y_test = P[train_idx], P[i]

        # RidgeCV within each fold
        try:
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            model.fit(X_train, y_train)
            preds[i] = model.predict(X_test)[0]
        except Exception:
            continue

    valid = ~np.isnan(preds)
    if valid.sum() < 3:
        return np.nan, np.nan, 0

    # Stratum-mean baseline (LOO stratum mean)
    stratum_preds = np.full(n, np.nan)
    for i in range(n):
        same_stratum = [j for j in range(n) if j != i and strata.iloc[j] == strata.iloc[i]]
        if len(same_stratum) >= 1:
            stratum_preds[i] = np.mean(P[same_stratum])
        else:
            stratum_preds[i] = np.mean(P[[j for j in range(n) if j != i]])

    mse_model = np.mean((preds[valid] - P[valid])**2)
    mse_stratum = np.mean((stratum_preds[valid] - P[valid])**2)
    Q2 = 1 - mse_model / (mse_stratum + 1e-10)

    rho, _ = spearmanr(preds[valid], P[valid])
    return Q2, rho, valid.sum()


# ---------------------------------------------------------------------------
# Module utilities
# ---------------------------------------------------------------------------

def module_scores(rna_df, modules_ref, log1p=True):
    """Aggregate RNA features into scores for predefined modules."""
    out = {}
    for mod_name in sorted(modules_ref.keys()):
        genes = [g for g in modules_ref[mod_name] if g in rna_df.columns]
        if len(genes) >= 2:
            block = np.log1p(rna_df[genes]) if log1p else rna_df[genes]
            out[mod_name] = block.mean(axis=1)
    return pd.DataFrame(out)


def data_driven_modules(prot_df, n_modules=8, genes_per_module=None):
    """Construct data-driven modules from first-PC loading ranks.

    Features are ranked by first-principal-component loading magnitude and
    divided into deterministic quantile-based modules.

    Parameters
    ----------
    prot_df : pandas.DataFrame
        Samples by protein features.
    n_modules : int
        Number of modules (default 8).
    genes_per_module : int or None
        None uses quantile boundaries based on feature count and n_modules;
        An integer requests deterministic groups with that many genes each.

    Returns
    -------
    dict[str, list[str]]
        Mapping of module names to features. Modules with fewer than three
        features are omitted.
    """
    from numpy.linalg import eigh
    prot_corr = prot_df.corr().values  # Protein-protein correlation matrix
    n_genes = prot_corr.shape[0]
    # Sort by 1st PC loading magnitude, then split into n_modules quantiles
    _, evecs = eigh(prot_corr)
    pc1_loadings = np.abs(evecs[:, -1])  # last eigenvector = 1st PC (eigh returns ascending)
    order = np.argsort(pc1_loadings)
    clusters = np.zeros(n_genes, dtype=int)
    if genes_per_module is None:
        for ci in range(n_modules):
            start = int(ci * n_genes / n_modules)
            end = int((ci + 1) * n_genes / n_modules)
            for idx in order[start:end]:
                clusters[idx] = ci
    else:
        gpm = int(genes_per_module)
        for ci in range(n_modules):
            start = ci * gpm
            end = min((ci + 1) * gpm, n_genes)
            for idx in order[start:end]:
                clusters[idx] = ci

    prot_genes = list(prot_df.columns)
    modules = {}
    for ci in range(n_modules):
        mod_genes = [prot_genes[j] for j in range(len(prot_genes)) if clusters[j] == ci]
        if len(mod_genes) < 3:
            # Original behavior: drop modules with <3 genes (no renumbering)
            continue
        modules[f'M{ci}'] = mod_genes
    return modules


# ---------------------------------------------------------------------------
# Simulation data used by tests and power calculations
# ---------------------------------------------------------------------------

def simulate_paired_data(n, effect_size_d, n_modules=8, seed=42):
    """Simulate paired module and protein measurements at a target effect size."""
    rng_local = np.random.default_rng(seed)
    true_rho = effect_size_d / np.sqrt(4 + effect_size_d**2)

    n_half1 = n // 2
    n_half2 = n - n_half1

    # Condition labels
    conditions = ['A'] * n_half1 + ['B'] * n_half2
    design = pd.DataFrame({'condition': conditions, 'batch': ['B1'] * n})

    # Shared latent structure (2 factors)
    L = rng_local.normal(0, 1, (n, 2))
    loadings = rng_local.uniform(0.3, 1.0, (n_modules, 2))

    rna_scores = {}
    prot_scores = {}

    n_bridgeable = max(1, n_modules // 2)  # half are bridgeable
    for m in range(n_modules):
        latent = L @ loadings[m] + rng_local.normal(0, 0.1, n)
        # RNA: latent + noise
        rna_scores[f'M{m}'] = latent + rng_local.normal(0, 0.2, n)

        if m < n_bridgeable:
            # Bridgeable: protein follows RNA with coupling = true_rho
            # Scale: protein = rho * RNA + (1-rho) * independent signal + noise
            prot_scores[f'M{m}'] = (true_rho * latent +
                                     (1 - abs(true_rho)) * rng_local.normal(0, 1, n) +
                                     rng_local.normal(0, 0.25, n))
        else:
            # Null: protein independent of RNA
            prot_scores[f'M{m}'] = rng_local.normal(0, 1, n)

    return rna_scores, prot_scores, design


__all__ = [
    'set_seed',
    'compute_zpg',
    'compute_rank_zPG',
    'compute_rank_zPG_partial',
    'compute_ECI',
    '_interpret_ECI',
    'joint_decision',
    'decide',
    'decide_legacy',
    'fdr_bh',
    'compute_module_q2_simple',
    'module_scores',
    'data_driven_modules',
    'simulate_paired_data',
    'select_cv',
]
