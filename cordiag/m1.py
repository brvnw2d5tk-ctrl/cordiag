"""Stratum-aware ridge prediction utilities used by the diagnostic statistics."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

__all__ = [
    'derive_seed',
    'subsample_seed',
    'm0_stratum_means_loocv',
    'm1_loocv',
    'm1_train_test',
    'ridge_edf',
    'within_stratum_permute',
    'empirical_p',
    'zscore_stat',
]

# matched-subsample 采样流的固定偏移 (the package specification §4.1):
# within/crossed 用 offset=0, cross a→b 用 offset=10000 —— +10000 使
# cross 的采样流与 within/crossed 分离, 避免两条链共享同一采样序列。
SEED_OFFSET_CROSS = 10000


def derive_seed(key_str: str) -> int:
    """
    md5 派生 per-pair RNG seed (TG 家族语义)。

    Deterministic seed construction:
        per_pair_seed = int(hashlib.md5(
            f'{protein}_{source_cond}_{target_cond}_{batch}'.encode()
        ).hexdigest(), 16) % 2**31

    为什么有这个行为: 保证 pair 间独立、与并行分块顺序无关 (worker
    顺序不影响结果)。调用方负责构造 key_str 为
    '{protein}_{source_cond}_{target_cond}_{batch}'。

    Returns
    -------
    int in [0, 2**31)
    """
    return int(hashlib.md5(key_str.encode()).hexdigest(), 16) % 2**31


def subsample_seed(base_seed: Optional[int], rep: int, offset: int = 0) -> int:
    """
    matched-subsample 链上的 per-rep seed (TG 家族语义)。

    Matched-subsample seed convention:
    (the package specification §4.1-4.2):
      - within / crossed:  rng.choice(...) 用 seed = pair_seed + rep
                           (offset=0)
      - cross a→b:         rng.choice(...) 用 seed = pair_seed + rep + 10000
                           (offset=SEED_OFFSET_CROSS)
    原实现内联为 `(seed if seed is not None else 42) + rep` 与
    `+ rep + 10000`; 本函数将链约定显式化, 数值语义与原实现bit-identical。

    为什么有这个行为: matched-subsample 的 rng.choice(replace=False)
    结果必须跨重跑可复现; +10000 把 cross 的采样流与 within/crossed
    分离, 保证两条链互不共享采样序列。

    Returns
    -------
    int
    """
    base = base_seed if base_seed is not None else 42
    return base + rep + offset



# ═══════════════════════════════════════════════════════════════════
# Spearman correlation (shared primitive, the package specification §3.2)
#
# Shared Spearman-correlation primitive.
# Implementation detail.
# `from cordiag.m1 import _spearmanr` 使用 (下划线前缀 = 私有共享原语,
# 不进 __all__)。tie 处理依赖 scipy.stats.spearmanr 版本 —— 黄金回归
# 锁定 scipy 1.14.1 (pyproject.toml scipy>=1.10,<1.18)。
# ═══════════════════════════════════════════════════════════════════

def _spearmanr(x, y) -> Tuple[float, float]:
    """Spearman correlation using scipy.stats.spearmanr."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(pval)


def m0_stratum_means_loocv(
    P: np.ndarray,
    strata: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Compute leave-one-out predictions from stratum-specific means."""
    strata = np.asarray(strata)
    n = len(P)
    means = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        s = strata[i]
        same = np.where(strata == s)[0]
        if groups is not None:
            same_group = np.where(groups == groups[i])[0]
            others = same[~np.isin(same, same_group)]
        else:
            others = same[same != i]
        if len(others) > 0:
            means[i] = np.mean(P[others]).astype(np.float64)
        else:
            if groups is not None:
                others = np.array(
                    [j for j in range(n) if groups[j] != groups[i]], dtype=int
                )
            else:
                others = np.array([j for j in range(n) if j != i], dtype=int)
            means[i] = np.mean(P[others]).astype(np.float64)
    mse = float(np.nanmean((means - P) ** 2))
    return means, mse


def m1_loocv(
    P: np.ndarray,
    X: np.ndarray,
    strata: np.ndarray,
    cv_alphas: List[float],
    eval_indices: Optional[np.ndarray] = None,
    fixed_alpha: Optional[float] = None,
    groups: Optional[np.ndarray] = None,
    *,
    unseen_stratum: str,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute stratum-aware ridge leave-one-out predictions and diagnostics."""
    if unseen_stratum not in ('fallback', 'skip'):
        raise ValueError(
            "unseen_stratum must be 'fallback' (TG 语义) or 'skip' (zPG 语义), "
            f"got {unseen_stratum!r}; m1.py 禁止自动推断, 调用方必须显式选择"
        )
    strata = np.asarray(strata)
    n = len(P)
    if eval_indices is None:
        eval_indices = np.arange(n)
    n_eval = len(eval_indices)

    preds = np.full(n_eval, np.nan, dtype=np.float64)
    edf_list = []
    alpha_list = []

    for fold_idx in range(n_eval):
        i = int(eval_indices[fold_idx])
        if groups is not None:
            # Exclude ALL samples from the same group (patient-level LOOCV)
            tr = [j for j in range(n) if groups[j] != groups[i]]
        else:
            tr = [j for j in range(n) if j != i]
        if len(tr) < 3:
            continue

        X_tr = X[tr].astype(np.float64)
        P_tr = P[tr].astype(np.float64)
        strata_tr = strata[tr]
        s_test = strata[i]

        # Stratum-center training data
        y_ctr = P_tr.copy()
        X_ctr = X_tr.copy()
        stratum_means_y = {}
        stratum_means_X = {}
        for s in np.unique(strata_tr):
            idx = np.where(strata_tr == s)[0]
            if len(idx) > 0:
                my = np.mean(P_tr[idx]).astype(np.float64)
                mX = np.mean(X_tr[idx], axis=0).astype(np.float64)
                stratum_means_y[s] = my
                stratum_means_X[s] = mX
                y_ctr[idx] -= my
                X_ctr[idx] -= mX

        # Center test sample
        if s_test in stratum_means_y:
            test_mean_y = stratum_means_y[s_test]
            test_mean_X = stratum_means_X[s_test]
        elif unseen_stratum == 'fallback':
            # TG 语义: 全局训练 mean 回退 (core.py _m1_loocv L336-337)
            test_mean_y = np.mean(P_tr).astype(np.float64)
            test_mean_X = np.mean(X_tr, axis=0).astype(np.float64)
        else:
            # Implementation detail.
            #   if len(s_tr_idx) == 0: truths[i] = y[i]; continue
            # 该样本预测保持 NaN, 不进 MSE/edf/alpha 汇总
            preds[fold_idx] = np.nan
            continue

        y_test_ctr = P[i].astype(np.float64) - test_mean_y
        X_test_ctr = X[i].astype(np.float64) - test_mean_X

        try:
            sX = StandardScaler().fit(X_ctr)
            sy = StandardScaler().fit(y_ctr.reshape(-1, 1))

            n_tr = len(tr)
            if fixed_alpha is not None:
                # Locked alpha: use the provided regularization strength
                alpha = float(fixed_alpha)
                m = Ridge(alpha=alpha).fit(
                    sX.transform(X_ctr),
                    sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                )
            elif n_tr >= 20:
                try:
                    m = RidgeCV(alphas=cv_alphas, cv=min(5, n_tr)).fit(
                        sX.transform(X_ctr),
                        sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                    )
                    alpha = m.alpha_
                except Exception:
                    m = Ridge(alpha=1.0).fit(
                        sX.transform(X_ctr),
                        sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                    )
                    alpha = 1.0
            else:
                alpha = 1.0
                m = Ridge(alpha=alpha).fit(
                    sX.transform(X_ctr),
                    sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                )

            # Compute effective df for this fold
            X_scaled = sX.transform(X_ctr).astype(np.float64)
            edf_list.append(ridge_edf(X_scaled, alpha))
            alpha_list.append(alpha)

            pred_ctr = sy.inverse_transform(
                m.predict(sX.transform(X_test_ctr.reshape(1, -1))).reshape(-1, 1)
            ).ravel()[0]
            preds[fold_idx] = pred_ctr + test_mean_y
        except Exception:
            preds[fold_idx] = np.nan

    valid = ~np.isnan(preds)
    if valid.sum() >= 1:
        mse = float(np.nanmean((preds[valid] - P[eval_indices[valid]]) ** 2))
    else:
        mse = float('nan')

    avg_edf = float(np.mean(edf_list)) if edf_list else 0.0
    avg_alpha = float(np.mean(alpha_list)) if alpha_list else 1.0
    return preds, mse, avg_edf, avg_alpha


def m1_train_test(
    P_train: np.ndarray,
    X_train: np.ndarray,
    strata_train: np.ndarray,
    P_test: np.ndarray,
    X_test: np.ndarray,
    strata_test: np.ndarray,
    cv_alphas: List[float],
) -> np.ndarray:
    """
    Train M1 (stratum-conditioned Ridge) on source data,
    predict on target data (no retraining, direct prediction).

    Uses the package's shared train/test modelling semantics.

    为什么有这个行为: Q²_a→b (train source → predict target) 与
    matched-subsample 系列 (within/crossed) 都用它 —— 单次拟合 + 逐样本
    预测; 测试样本 stratum 不在训练集时回退全局训练 mean (与 m1_loocv
    的 'fallback' 模式同一约定; zPG 无 train-test 分割, 全部 LOOCV,
    故本函数无 'skip' 模式)。

    Parameters
    ----------
    P_train : np.ndarray (n_train,)
    X_train : np.ndarray (n_train, p)
    strata_train : np.ndarray (n_train,) — 训练样本 stratum 标签
    P_test : np.ndarray (n_test,)
    X_test : np.ndarray (n_test, p)
    strata_test : np.ndarray (n_test,) — 测试样本 stratum 标签
    cv_alphas : list of float

    Returns
    -------
    predictions : np.ndarray (n_test,)
        预测失败的样本为 NaN。
    """
    strata_train = np.asarray(strata_train)
    strata_test = np.asarray(strata_test)
    n_train = len(P_train)
    n_test = len(P_test)

    # Stratum-center training data
    y_ctr = P_train.copy().astype(np.float64)
    X_ctr = X_train.copy().astype(np.float64)
    stratum_means_y = {}
    stratum_means_X = {}

    for s in np.unique(strata_train):
        idx = np.where(strata_train == s)[0]
        if len(idx) > 0:
            my = float(np.mean(P_train[idx]))
            mX = np.mean(X_train[idx], axis=0).astype(np.float64)
            stratum_means_y[s] = my
            stratum_means_X[s] = mX
            y_ctr[idx] -= my
            X_ctr[idx] -= mX

    # Fit model once on centered+scaled training
    try:
        sX = StandardScaler().fit(X_ctr)
        sy = StandardScaler().fit(y_ctr.reshape(-1, 1))

        if n_train >= 20:
            try:
                m = RidgeCV(alphas=cv_alphas, cv=min(5, n_train)).fit(
                    sX.transform(X_ctr),
                    sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                )
            except Exception:
                m = Ridge(alpha=1.0).fit(
                    sX.transform(X_ctr),
                    sy.transform(y_ctr.reshape(-1, 1)).ravel(),
                )
        else:
            m = Ridge(alpha=1.0).fit(
                sX.transform(X_ctr),
                sy.transform(y_ctr.reshape(-1, 1)).ravel(),
            )
    except Exception:
        return np.full(n_test, np.nan, dtype=np.float64)

    # Predict on test samples
    predictions = np.full(n_test, np.nan, dtype=np.float64)
    for j in range(n_test):
        s = strata_test[j] if j < len(strata_test) else ''
        if s in stratum_means_y:
            test_mean_y = stratum_means_y[s]
            test_mean_X = stratum_means_X[s]
        else:
            # Fallback: global training mean
            test_mean_y = float(np.mean(P_train))
            test_mean_X = np.mean(X_train, axis=0).astype(np.float64)

        y_te_ctr = P_test[j].astype(np.float64) - test_mean_y
        X_te_ctr = X_test[j].astype(np.float64) - test_mean_X

        try:
            pred_ctr = sy.inverse_transform(
                m.predict(sX.transform(X_te_ctr.reshape(1, -1))).reshape(-1, 1)
            ).ravel()[0]
            predictions[j] = pred_ctr + test_mean_y
        except Exception:
            predictions[j] = np.nan

    return predictions


def ridge_edf(X_scaled: np.ndarray, alpha: float) -> float:
    """Helper for the paired-data diagnostic calculation."""
    n, p = X_scaled.shape
    k = min(n, p)
    if k == 0:
        return 0.0
    try:
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        svals = s[:k]
        return float(np.sum(svals ** 2 / (svals ** 2 + alpha)))
    except np.linalg.LinAlgError:
        # SVD failed: rough approximation
        return float(np.trace(X_scaled @ X_scaled.T) /
                     (np.trace(X_scaled @ X_scaled.T) + alpha * n))


def within_stratum_permute(
    P: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute values independently within each stratum."""
    strata = np.asarray(strata)
    P_perm = P.copy()
    for s in _strata_unique_order(strata):
        s_idx = np.where(strata == s)[0]
        if len(s_idx) >= 2:
            P_perm[s_idx] = P[s_idx[rng.permutation(len(s_idx))]]
    return P_perm


def empirical_p(
    null_vals: np.ndarray,
    obs: float,
    two_sided: bool = True,
    denominator: Optional[int] = None,
) -> float:
    """Helper for the paired-data diagnostic calculation."""
    null_arr = np.asarray(null_vals, dtype=np.float64)
    null_arr = null_arr[~np.isnan(null_arr)]
    if two_sided:
        cnt = int(np.sum(np.abs(null_arr) >= np.abs(obs)))
    else:
        cnt = int(np.sum(null_arr >= obs))
    if denominator is None:
        denom = len(null_arr) + 1
    else:
        denom = denominator + 1
    return float((cnt + 1) / denom)


def zscore_stat(obs: float, null_vals: np.ndarray) -> float:
    """Helper for the paired-data diagnostic calculation."""
    null_arr = np.asarray(null_vals, dtype=np.float64)
    null_arr = null_arr[~np.isnan(null_arr)]
    if len(null_arr) < 1:
        return float('nan')
    mu = float(np.mean(null_arr))
    sd = float(np.std(null_arr))
    if np.isnan(sd) or sd <= 1e-10:
        return float('nan')
    return float((obs - mu) / sd)


# ── 内部工具 ────────────────────────────────────────────────────────

def _strata_unique_order(strata: np.ndarray) -> np.ndarray:
    """Helper for the paired-data diagnostic calculation."""
    _, first_idx = np.unique(strata, return_index=True)
    return strata[np.sort(first_idx)]
