# -*- coding: utf-8 -*-
"""
cordiag.calibration — TG simulation scenarios (8 ground-truth mechanisms).
==========================================================================

Threshold calibration suite for the Transportability Gap (TG) diagnostic.
The suite is self-contained and preserves the package's established numerical
semantics.

Tests and calibrates Transportability Gap (TG) thresholds under controlled
synthetic conditions. Generates paired RNA-protein data where the RNA-to-protein
coupling is either identical or deliberately different between two conditions
(a and b).

8 Scenarios:
  1. NULL_IDENTICAL         — a and b have identical RNA->protein relationships
  2. NULL_STRATUM_SHIFT     — same RNA coupling, different stratum distributions
  3. TRANSPORTABLE          — same relationship, same stratum (null for Type I)
  4. NON_TRANSPORTABLE_WEAK — RNA coupling differs by Deltarho ~ 0.1
  5. NON_TRANSPORTABLE_STRONG — RNA coupling differs by Deltarho ~ 0.3
  6. SAMPLE_ASYMMETRY       — same relationship, n_a = 50, n_b = 10
  7. HIGH_DIM               — p_features = 50 > n = 20 per condition
  8. REALISTIC              — moderate, heterogeneous synthetic parameters

Data generation:
  - Design: condition (a/b) + batch (1-3 levels)
  - RNA: n x p multivariate normal with condition-specific means
  - Protein: P = beta_0 + beta_stratum * Z + beta_rna * X_rna + epsilon

TG spec v2 (double-reviewer revision), 2026-07-28.

The module-level tuning grid remains patchable so callers can run controlled
simulation studies with explicitly selected repetition and subsampling counts.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import contextlib
import functools
import warnings
import os


# ── TG core imports (single source of truth) ────────────────────────────
# Shared core component
# m1.py 已落地 (the package specification §3.2), 实际状态 (功能正确, 无需再改):
#   - 4/6 名字是 m1 原语经 tg.py re-export: _compute_stratum_means_loocv
#     (= m0_stratum_means_loocv 别名), _m1_loocv (TG 包装, unseen_stratum
#     固定 'fallback'), _m1_train_test (= m1_train_test 别名),
#     _spearmanr (= m1._spearmanr)
#   - 2/6 名字是 tg.py 自持 (m1.py 不重实现): _compute_q2_within_matched /
#     _compute_q2_crossed_matched (matched-subsample 阈值链,
#     见 m1.py docstring 不变量 #2)
# 调用语义零变化 —— 等价性已由 the package specification §5.4 验证。
from cordiag.tg import (_compute_stratum_means_loocv, _m1_loocv, _m1_train_test,
                     _compute_q2_within_matched, _compute_q2_cross_matched,
                     _compute_q2_crossed_matched,
                     _spearmanr)


# ═══════════════════════════════════════════════════════════════════════════
# Warning 作用域 (工程审核 2026-08-01: 模块级 filterwarnings('ignore')
# 已移除 — 它会吞掉调用方进程级的所有警告; 同 tg.py 方案)
#
# 替代方案: 仅在内部计算期间 (scipy/numpy/sklearn 数值噪声) 静默
# RuntimeWarning/FutureWarning; 本模块刻意发出的 UserWarning 仍正常显示。
# 装饰目标 = 数值重入口 (simulate_one_rep / run_calibration); TG 核心调用
# (_compute_q2_within_matched / _compute_q2_crossed_matched) 已在 tg.py 侧
# 自带 @_quiet_internal (2026-08-01 扩展)。
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_internal_warnings():
    """Scope warning suppression to internal library calls (tg.py 同款)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        yield


def _quiet_internal(func):
    """Decorator: run *func* under :func:`_suppress_internal_warnings`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _suppress_internal_warnings():
            return func(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Pure-numpy utilities
# ═══════════════════════════════════════════════════════════════════════════

def _pearsonr(x, y):
    """Pure numpy Pearson correlation, float64."""
    xm = x - x.mean()
    ym = y - y.mean()
    num = np.dot(xm, ym)
    den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    if den < 1e-15:
        return 0.0
    return np.clip(num / den, -1.0, 1.0)


def _random_orthogonal(v, rng):
    """Return a random unit vector orthogonal to v (same length)."""
    p = len(v)
    u = rng.normal(0, 1, p).astype(np.float64)
    u -= np.dot(u, v) * v / max(np.dot(v, v), 1e-15)
    nu = np.linalg.norm(u)
    if nu > 1e-15:
        u /= nu
    return u


def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


SCENARIO_NAMES = [
    'NULL_IDENTICAL',
    'NULL_STRATUM_SHIFT',
    'TRANSPORTABLE',
    'NON_TRANSPORTABLE_WEAK',
    'NON_TRANSPORTABLE_STRONG',
    'SAMPLE_ASYMMETRY',
    'HIGH_DIM',
    'REALISTIC',
]


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TGSimConfig:
    """Configuration for one TG simulation run (one scenario, one sample size)."""
    scenario: str = 'NULL_IDENTICAL'
    n_a: int = 30                      # samples in condition a
    n_b: int = 30                      # samples in condition b
    p_features: int = 20               # number of RNA features
    n_batches: int = 2                 # batch levels (1-3)
    effect_size: float = 0.5           # base RNA->protein signal strength
    delta_rho_target: float = 0.0      # target coupling diff (only for NON_TRANSPORTABLE)
    noise_scale: float = 0.5           # residual noise std
    batch_imbalance: float = 0.0       # 0 = balanced, >0 = batch shift (NULL_STRATUM_SHIFT)
    seed: int = 42

    # Transparencies (populated during generation)
    realized_rho_a: Optional[float] = None
    realized_rho_b: Optional[float] = None
    realized_delta_rho: Optional[float] = None
    n_a_actual: Optional[int] = None
    n_b_actual: Optional[int] = None


@dataclass
class TGSimResult:
    """Results from one TG simulation rep (one generated dataset + TG computation)."""
    config: TGSimConfig
    scenario: str = ''

    # Ground truth
    delta_rho_true: float = 0.0        # actual Deltarho in generated data
    rho_a_true: float = 0.0
    rho_b_true: float = 0.0
    coupling_identical: bool = True    # True in NULL / TRANSPORTABLE scenarios

    # TG components
    q2_within_b: float = float('nan')
    q2_a_to_b: float = float('nan')
    q2_crossed: float = float('nan')
    tg_raw: float = float('nan')
    tg_design: float = float('nan')
    tg_rna: float = float('nan')
    tg_relative: float = float('nan')
    tg_design_fraction: float = float('nan')
    mse_stratum_b: float = float('nan')

    # Sample info
    n_source: int = 0
    n_target: int = 0
    size_ratio: float = 1.0

    # Quality
    estimable: bool = True
    ridge_alpha: float = 1.0
    ridge_alpha_within_b: float = 1.0  # Avg alpha from within-b, locked for crossed

    # Transparency for calibration
    scenario_group: str = ''  # 'null_identical' | 'transportable' | 'non_transportable' | 'asymmetry' | 'high_dim' | 'realistic'


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_autoregressive_cov(p, rho=0.3):
    """Generate AR(1) correlation matrix: rho^{|i-j|}."""
    rows, cols = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    return np.float64(rho ** np.abs(rows - cols))


def _make_design_matrix(n_a, n_b, n_batches, batch_imbalance, rng):
    """
    Build design DataFrame for both conditions.

    Parameters
    ----------
    n_a, n_b : int
        Samples per condition.
    n_batches : int
        Number of batch levels.
    batch_imbalance : float
        0 = balanced batches across conditions.
        >0 = condition b is shifted toward later batches (stratum shift).
    rng : np.random.Generator

    Returns
    -------
    design : pd.DataFrame
        Columns: condition, batch.
    idx_a, idx_b : np.ndarray
        Boolean masks for conditions a and b.
    """
    n_total = n_a + n_b

    conditions = np.array(['a'] * n_a + ['b'] * n_b)
    batches = np.empty(n_total, dtype=object)

    if batch_imbalance == 0.0:
        # Balanced: each condition spread evenly across batches
        for cond_start, cond_end in [(0, n_a), (n_a, n_total)]:
            c_n = cond_end - cond_start
            per_batch = max(1, c_n // n_batches)
            remaining = c_n
            for bi in range(n_batches):
                n_this = min(per_batch, remaining) if bi < n_batches - 1 else remaining
                start = cond_start + bi * per_batch
                end = min(start + n_this, cond_end)
                batches[start:end] = f'B{bi}'
                remaining -= n_this
    else:
        # Imbalanced: condition a concentrated in early batches, b in later
        for ci, (cond_label, cond_start, cond_end) in enumerate(
            [('a', 0, n_a), ('b', n_a, n_total)]
        ):
            c_n = cond_end - cond_start
            # Shift: for condition b, skew toward higher batch indices
            shift = batch_imbalance if cond_label == 'b' else 0.0
            batch_probs = np.ones(n_batches, dtype=np.float64)
            batch_probs += shift * np.arange(n_batches)
            batch_probs /= batch_probs.sum()
            assigned = rng.choice([f'B{b}' for b in range(n_batches)],
                                  size=c_n, p=batch_probs)
            batches[cond_start:cond_end] = assigned

    design = pd.DataFrame({'condition': conditions, 'batch': batches})
    idx_a = np.array([c == 'a' for c in conditions])
    idx_b = np.array([c == 'b' for c in conditions])

    return design, idx_a, idx_b


def _build_stratum_encoding(design):
    """One-hot encode condition + batch."""
    Z = pd.get_dummies(design[['condition', 'batch']], drop_first=True).values.astype(np.float64)
    return Z


def _build_stratum_labels(design):
    """Build combined condition_batch string labels."""
    return (design['condition'].astype(str) + '_' + design['batch'].astype(str)).values


def generate_scenario(
    scenario_name: str,
    n_per_condition: Union[int, Tuple[int, int]] = 30,
    p_features: int = 20,
    n_batches: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate synthetic paired RNA-protein data for one TG scenario.

    Parameters
    ----------
    scenario_name : str
        One of the 8 scenario names (case-insensitive; underscores or hyphens).
    n_per_condition : int or (int, int)
        Samples per condition. If int, both conditions get n. If tuple, (n_a, n_b).
    p_features : int
        Number of RNA features.
    n_batches : int
        Number of batch levels (1-3).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        rna_a : ndarray (n_a, p)
        rna_b : ndarray (n_b, p)
        protein_a : ndarray (n_a,)
        protein_b : ndarray (n_b,)
        design_a : pd.DataFrame
        design_b : pd.DataFrame
        config : TGSimConfig
        ground_truth : dict
    """
    rng = np.random.default_rng(seed)

    # Normalize scenario name
    sname = scenario_name.upper().replace('-', '_')

    # Sample sizes
    if isinstance(n_per_condition, (tuple, list)):
        n_a, n_b = int(n_per_condition[0]), int(n_per_condition[1])
    else:
        n_a = n_b = int(n_per_condition)

    # Override for specific scenarios
    if sname == 'SAMPLE_ASYMMETRY':
        n_a, n_b = 50, 10

    # Build config
    config = TGSimConfig(
        scenario=sname,
        n_a=n_a,
        n_b=n_b,
        p_features=p_features,
        n_batches=n_batches,
        seed=seed,
    )

    # Scenario-specific parameters
    batch_imbalance = 0.0
    delta_rho_target = 0.0
    coupling_strength = 1.0  # base RNA->protein signal (beta norm scaled for variance)
    noise_scale = 0.3        # residual noise std (lower = cleaner signal)

    # Override synthetic example parameters for REALISTIC
    if sname == 'REALISTIC':
        p_features = max(p_features, 30)
        coupling_strength = 0.4
        noise_scale = 0.6
        batch_imbalance = 0.0

    if sname == 'NON_TRANSPORTABLE_WEAK':
        delta_rho_target = 0.1

    if sname == 'NON_TRANSPORTABLE_STRONG':
        delta_rho_target = 0.3

    if sname in ('NULL_STRATUM_SHIFT',):
        batch_imbalance = 1.5  # strong batch imbalance

    if sname == 'HIGH_DIM':
        p_features = max(p_features, 50)
        n_a = n_b = min(n_a, 20)  # ensure p > n

    config.p_features = p_features
    config.n_a = n_a
    config.n_b = n_b
    config.batch_imbalance = batch_imbalance
    config.delta_rho_target = delta_rho_target
    config.noise_scale = noise_scale

    # ── Design matrix ──
    design, idx_a, idx_b = _make_design_matrix(n_a, n_b, n_batches, batch_imbalance, rng)
    n_total = n_a + n_b

    # ── Stratum encoding ──
    Z = _build_stratum_encoding(design)
    beta_stratum = rng.normal(0, 0.5, Z.shape[1]).astype(np.float64)
    stratum_effect = Z @ beta_stratum

    # ── RNA features (AR(1) correlation) ──
    cov_rna = _generate_autoregressive_cov(p_features, rho=0.3)
    L = np.linalg.cholesky(cov_rna)
    rna_raw = (L @ rng.normal(0, 1, (p_features, n_total))).T.astype(np.float64)

    # ── Build RNA->protein beta coefficients ──
    norm_base = coupling_strength / np.sqrt(p_features)

    if sname == 'HIGH_DIM':
        # Only 3 features carry signal
        beta_shared = np.zeros(p_features, dtype=np.float64)
        beta_shared[:3] = rng.normal(0, coupling_strength, 3)
    else:
        beta_shared = rng.normal(0, norm_base, p_features).astype(np.float64)

    beta_shared = beta_shared.astype(np.float64)
    coupling_identical = True

    if sname in ('NON_TRANSPORTABLE_WEAK', 'NON_TRANSPORTABLE_STRONG'):
        coupling_identical = False
        # Rotation approach: keep ||beta|| same, change direction.
        # beta_b = norm * (cos(theta) * unit(beta_a) + sin(theta) * orth(beta_a))
        # theta = 0 -> identical (transportable); theta > 0 -> rotated (non-transportable)
        # TG_raw contribution ≈ rho^2 * sin^2(theta) where rho^2 = signal/noise ratio
        norm_beta = max(np.linalg.norm(beta_shared), 1e-15)
        v = beta_shared / norm_beta
        u = _random_orthogonal(beta_shared, rng)
        # Map target to rotation angle: theta = pi * delta_rho_target
        #   delta_rho_target=0.1 → 18° rotation, 0.3 → 54° rotation
        theta = np.pi * delta_rho_target
        beta_b = norm_beta * (np.cos(theta) * v + np.sin(theta) * u)
    else:
        beta_b = beta_shared.copy()

    beta_a = beta_shared.copy()

    # ── Compute RNA effects ──
    rna_effect_a = rna_raw[idx_a] @ beta_a
    rna_effect_b = rna_raw[idx_b] @ beta_b

    # ── Noise ──
    noise_a = rng.normal(0, noise_scale, n_a).astype(np.float64)
    noise_b = rng.normal(0, noise_scale, n_b).astype(np.float64)

    # ── Protein = stratum effect + RNA effect + noise ──
    protein_a = stratum_effect[idx_a] + rna_effect_a + noise_a
    protein_b = stratum_effect[idx_b] + rna_effect_b + noise_b

    # ── Compute realized coupling (semi-partial rho: RNA vs Y after stratum removal) ──
    def _semi_partial_r(rna_eff, protein, Z_cond):
        """Semi-partial correlation of RNA effect with protein, controlling for Z."""
        # Regress out Z from both rna_effect and protein
        try:
            beta_rna = np.linalg.lstsq(Z_cond, rna_eff, rcond=None)[0]
            beta_prot = np.linalg.lstsq(Z_cond, protein, rcond=None)[0]
            rna_resid = rna_eff - Z_cond @ beta_rna
            prot_resid = protein - Z_cond @ beta_prot
            return _pearsonr(rna_resid, prot_resid)
        except np.linalg.LinAlgError:
            return _pearsonr(rna_eff, protein)

    Za = Z[idx_a]
    Zb = Z[idx_b]
    rho_a = abs(_semi_partial_r(rna_effect_a, protein_a, Za))
    rho_b = abs(_semi_partial_r(rna_effect_b, protein_b, Zb))
    delta_rho_realized = abs(rho_a - rho_b)

    config.realized_rho_a = rho_a
    config.realized_rho_b = rho_b
    config.realized_delta_rho = delta_rho_realized

    # ── Design dataframes per condition ──
    design_a = design.iloc[idx_a].reset_index(drop=True)
    design_b = design.iloc[idx_b].reset_index(drop=True)

    # ── Ground truth metadata ──
    ground_truth = {
        'scenario': sname,
        'coupling_identical': coupling_identical,
        'delta_rho_target': delta_rho_target,
        'delta_rho_realized': float(delta_rho_realized),
        'rho_a': float(rho_a),
        'rho_b': float(rho_b),
        'beta_a': beta_a,
        'beta_b': beta_b,
        'n_a': n_a,
        'n_b': n_b,
        'p_features': p_features,
        'n_batches': n_batches,
        'batch_imbalance': batch_imbalance,
        'noise_scale': noise_scale,
        'coupling_strength': coupling_strength,
        'seed': seed,
    }

    return {
        'rna_a': rna_raw[idx_a],
        'rna_b': rna_raw[idx_b],
        'protein_a': protein_a,
        'protein_b': protein_b,
        'design_a': design_a,
        'design_b': design_b,
        'design_combined': design,
        'Z_combined': Z,
        'idx_a': idx_a,
        'idx_b': idx_b,
        'config': config,
        'ground_truth': ground_truth,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TG computation (inline Ridge LOOCV)
# ═══════════════════════════════════════════════════════════════════════════

_RIDGE_ALPHAS = np.logspace(-3, 3, 21)




def _compute_q2_components(
    rna_a, rna_b, protein_a, protein_b, design_a, design_b, design_combined,
    n_subsamples: int = 100,
):
    """
    Compute all Q2 and TG components for one simulation run.

    Uses tg_core's LOOCV stratum means and M1 Ridge model (single source of
    truth).  All Q² values share the same denominator (target M0 LOOCV MSE).
    This is the correct approach per TG spec — without it, TG decomposition
    identity (TG_raw = TG_design + TG_RNA) does not hold.

    Parameters
    ----------
    rna_a : ndarray (n_a, p)
    rna_b : ndarray (n_b, p)
    protein_a : ndarray (n_a,)
    protein_b : ndarray (n_b,)
    design_a, design_b : pd.DataFrame
    design_combined : pd.DataFrame

    Returns
    -------
    TGSimResult with filled TG components.
    """
    n_a = len(protein_a)
    n_b = len(protein_b)
    result = TGSimResult(config=None)
    result.n_source = n_a
    result.n_target = n_b
    result.size_ratio = max(n_a, n_b) / max(min(n_a, n_b), 1)
    result.ridge_alpha = 1.0  # placeholder

    # ── Build stratum labels ──
    strata_a = (design_a['condition'].astype(str) + '_' +
                design_a['batch'].astype(str)).values
    strata_b = (design_b['condition'].astype(str) + '_' +
                design_b['batch'].astype(str)).values
    strata_pooled = (design_combined['condition'].astype(str) + '_' +
                     design_combined['batch'].astype(str)).values

    # ── M0 denominator: LOOCV stratum means on target (b) ──
    # This single MSE is used as denominator for ALL three Q² values,
    # ensuring TG_raw = TG_design + TG_RNA holds identically.
    _, mse_stratum_b = _compute_stratum_means_loocv(protein_b, strata_b)
    result.mse_stratum_b = float(mse_stratum_b)

    if mse_stratum_b < 1e-10:
        # Degenerate baseline — all Q² default to 0, TG = 0
        result.q2_within_b = 0.0
        result.q2_a_to_b = 0.0
        result.q2_crossed = 0.0
        result.tg_raw = 0.0
        result.tg_design = 0.0
        result.tg_rna = 0.0
        result.tg_design_fraction = 0.5
        return result

    # ── Q²_within_b: matched-subsample CV (LOOCV fallback for small n) ──
    train_size = min(n_a, max(n_b // 2, n_b - 10))  # match source, min 10 test (sync with tg_core)
    sim_seed = 42
    if train_size >= 8 and n_b - train_size >= 3:
        q2_w, _, _, _, within_alpha = _compute_q2_within_matched(
            protein_b, rna_b, strata_b, train_size, _RIDGE_ALPHAS.tolist(),
            n_subsamples=n_subsamples, seed=sim_seed,
        )
    else:
        _, mse_w, _, within_alpha = _m1_loocv(
            protein_b, rna_b, strata_b, _RIDGE_ALPHAS.tolist(),
        )
        q2_w = float(1.0 - mse_w / mse_stratum_b)
    result.q2_within_b = float(q2_w) if not np.isnan(q2_w) else 0.0
    result.ridge_alpha_within_b = float(within_alpha)

    # ── Q²_a→b: train M1 on source (a), predict on target (b) ──
    if train_size >= 8 and n_b - train_size >= 3:
        q2_ab, _ = _compute_q2_cross_matched(
            protein_a, rna_a, strata_a,
            protein_b, rna_b, strata_b,
            train_size, mse_stratum_b, _RIDGE_ALPHAS.tolist(),
            n_subsamples=n_subsamples, seed=sim_seed,
        )
        result.q2_a_to_b = float(q2_ab)
    else:
        preds_ab = _m1_train_test(
            protein_a, rna_a, strata_a,
            protein_b, rna_b, strata_b,
            _RIDGE_ALPHAS.tolist(),
        )
        valid_ab = ~np.isnan(preds_ab)
        if valid_ab.sum() >= 1:
            mse_ab = float(np.mean((preds_ab[valid_ab] - protein_b[valid_ab]) ** 2))
            result.q2_a_to_b = float(1.0 - mse_ab / mse_stratum_b)
        else:
            result.q2_a_to_b = float('nan')

    # ── Q²_crossed: matched-subsample pooled (a+b), evaluate on target ──
    rna_pooled = np.vstack([rna_a, rna_b])
    protein_pooled = np.concatenate([protein_a, protein_b])
    b_indices = np.arange(n_a, n_a + n_b)
    q2_cr, mse_crossed, _ = _compute_q2_crossed_matched(
        protein_pooled, rna_pooled, strata_pooled,
        b_indices, train_size, mse_stratum_b, _RIDGE_ALPHAS.tolist(),
        n_subsamples=n_subsamples, seed=sim_seed + 1,
    )
    result.q2_crossed = float(q2_cr) if not np.isnan(q2_cr) else 0.0

    # ── TG decomposition ──
    result.tg_raw = result.q2_within_b - result.q2_a_to_b
    result.tg_design = result.q2_within_b - result.q2_crossed
    result.tg_rna = result.q2_crossed - result.q2_a_to_b
    result.tg_relative = result.tg_raw / max(abs(result.q2_within_b), 0.01)
    tg_denom = max(abs(result.tg_raw), 1e-10)
    result.tg_design_fraction = result.tg_design / tg_denom

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Single-rep simulation
# ═══════════════════════════════════════════════════════════════════════════

@_quiet_internal
def simulate_one_rep(
    scenario_name: str,
    n_per_condition: Union[int, Tuple[int, int]] = 30,
    p_features: int = 20,
    n_batches: int = 2,
    seed: int = 42,
    n_subsamples: int = 100,
) -> TGSimResult:
    """
    Run one TG simulation repetition: generate data + compute TG.

    Parameters
    ----------
    scenario_name : str
        One of the 8 scenario names.
    n_per_condition : int or (int, int)
        Samples per condition. If int, both conditions get n.
    p_features : int
        Number of RNA features.
    n_batches : int
        Number of batch levels.
    seed : int
        Random seed.

    Returns
    -------
    TGSimResult
    """
    data = generate_scenario(scenario_name, n_per_condition, p_features, n_batches, seed)

    result = _compute_q2_components(
        data['rna_a'], data['rna_b'],
        data['protein_a'], data['protein_b'],
        data['design_a'], data['design_b'],
        data['design_combined'],
        n_subsamples=n_subsamples,
    )

    gt = data['ground_truth']
    result.config = data['config']
    result.scenario = gt['scenario']
    result.coupling_identical = gt['coupling_identical']
    result.delta_rho_true = gt['delta_rho_realized']
    result.rho_a_true = gt['rho_a']
    result.rho_b_true = gt['rho_b']

    # Scenario group for calibration
    sname = gt['scenario']
    if sname in ('NULL_IDENTICAL',):
        result.scenario_group = 'null_identical'
    elif sname == 'NULL_STRATUM_SHIFT':
        result.scenario_group = 'null_stratum_shift'
    elif sname == 'TRANSPORTABLE':
        result.scenario_group = 'transportable'
    elif sname in ('NON_TRANSPORTABLE_WEAK', 'NON_TRANSPORTABLE_STRONG'):
        result.scenario_group = 'non_transportable'
    elif sname == 'SAMPLE_ASYMMETRY':
        result.scenario_group = 'asymmetry'
    elif sname == 'HIGH_DIM':
        result.scenario_group = 'high_dim'
    elif sname == 'REALISTIC':
        result.scenario_group = 'realistic'

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Summary collection
# ═══════════════════════════════════════════════════════════════════════════

def summarize_reps(results: List[TGSimResult]) -> pd.DataFrame:
    """
    Aggregate multiple TG simulation results into a summary DataFrame.

    Parameters
    ----------
    results : list of TGSimResult

    Returns
    -------
    pd.DataFrame with one row per rep, columns for all TG components + metadata.
    """
    rows = []
    for i, r in enumerate(results):
        cfg = r.config
        rows.append({
            'rep': i,
            'scenario': r.scenario,
            'scenario_group': r.scenario_group,
            'n_a': r.n_source,
            'n_b': r.n_target,
            'n_total': r.n_source + r.n_target,
            'size_ratio': r.size_ratio,
            'p_features': cfg.p_features if cfg else 20,
            'n_batches': cfg.n_batches if cfg else 2,
            'tg_raw': r.tg_raw,
            'tg_design': r.tg_design,
            'tg_rna': r.tg_rna,
            'tg_relative': r.tg_relative,
            'tg_design_fraction': r.tg_design_fraction,
            'q2_within_b': r.q2_within_b,
            'q2_a_to_b': r.q2_a_to_b,
            'q2_crossed': r.q2_crossed,
            'mse_stratum_b': r.mse_stratum_b,
            'delta_rho_true': r.delta_rho_true,
            'rho_a_true': r.rho_a_true,
            'rho_b_true': r.rho_b_true,
            'coupling_identical': r.coupling_identical,
            'ridge_alpha': r.ridge_alpha,
            'ridge_alpha_within_b': r.ridge_alpha_within_b,
            'estimable': r.estimable,
            'seed': cfg.seed if cfg else 0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_simulation(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Type I error, Power, FDR, threshold percentiles from simulation.

    For each (scenario_group, n_a, n_b, p_features) combination, computes:
      - 95th / 99th percentile of TG_raw under NULL_IDENTICAL (threshold candidates)
      - Mean TG_raw, TG_design, TG_RNA
      - Mean delta_rho_true
      - Count of reps

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_reps().

    Returns
    -------
    pd.DataFrame one row per group.
    """
    group_cols = ['scenario', 'scenario_group', 'n_total']
    # Add n_a, n_b if present
    for c in ['n_a', 'n_b', 'p_features', 'n_batches']:
        if c in summary_df.columns:
            group_cols.append(c)

    results = []
    for group_vals, grp in summary_df.groupby(group_cols, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row = dict(zip(group_cols, group_vals))

        tg_raw = grp['tg_raw'].dropna().values
        tg_design = grp['tg_design'].dropna().values
        tg_rna = grp['tg_rna'].dropna().values
        delta_rho = grp['delta_rho_true'].dropna().values

        row['n_reps'] = len(tg_raw)
        if len(tg_raw) > 0:
            row['tg_raw_mean'] = float(np.mean(tg_raw))
            row['tg_raw_std'] = float(np.std(tg_raw))
            row['tg_raw_p50'] = float(np.percentile(tg_raw, 50))
            row['tg_raw_p90'] = float(np.percentile(tg_raw, 90))
            row['tg_raw_p95'] = float(np.percentile(tg_raw, 95))
            row['tg_raw_p99'] = float(np.percentile(tg_raw, 99))
            row['tg_raw_max'] = float(np.max(tg_raw))
            row['tg_design_mean'] = float(np.mean(tg_design))
            row['tg_rna_mean'] = float(np.mean(tg_rna))
            row['delta_rho_mean'] = float(np.mean(delta_rho))
        else:
            row['tg_raw_mean'] = float('nan')
            row['tg_raw_std'] = float('nan')
            row['tg_raw_p50'] = float('nan')
            row['tg_raw_p90'] = float('nan')
            row['tg_raw_p95'] = float('nan')
            row['tg_raw_p99'] = float('nan')
            row['tg_raw_max'] = float('nan')
            row['tg_design_mean'] = float('nan')
            row['tg_rna_mean'] = float('nan')
            row['delta_rho_mean'] = float('nan')

        # Power / detection rate for non-transportable scenarios
        if row.get('scenario_group') in ('non_transportable',):
            n_total_reps = len(tg_raw)
            # Count how many have positive TG_raw (correct detection)
            n_pos = int(np.sum(tg_raw > 0))
            row['detection_rate'] = n_pos / max(n_total_reps, 1)
            # Count how many have TG_raw > 0.05 (practical significance)
            n_above_005 = int(np.sum(tg_raw > 0.05))
            row['practical_detection_005'] = n_above_005 / max(n_total_reps, 1)
            n_above_010 = int(np.sum(tg_raw > 0.10))
            row['practical_detection_010'] = n_above_010 / max(n_total_reps, 1)
        else:
            row['detection_rate'] = float('nan')
            row['practical_detection_005'] = float('nan')
            row['practical_detection_010'] = float('nan')

        # Type I error: for null/transportable, fraction with TG_raw > threshold
        if row.get('scenario_group') in ('null_identical', 'null_stratum_shift', 'transportable'):
            n_total_reps = len(tg_raw)
            # Under true null, TG_raw > 0.05 is Type I error
            n_type1_005 = int(np.sum(tg_raw > 0.05))
            n_type1_010 = int(np.sum(tg_raw > 0.10))
            row['type_i_005'] = n_type1_005 / max(n_total_reps, 1)
            row['type_i_010'] = n_type1_010 / max(n_total_reps, 1)
            row['type_i_p95'] = int(np.sum(tg_raw > row.get('tg_raw_p95', 0.05))) / max(n_total_reps, 1)
        else:
            row['type_i_005'] = float('nan')
            row['type_i_010'] = float('nan')
            row['type_i_p95'] = float('nan')

        results.append(row)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Threshold calibration
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_tg_thresholds(
    summary_df: pd.DataFrame,
    target_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Calibrate TG thresholds theta from simulation results.

    For each sample size (n_total), computes:
      - theta = max(0.05, 95th percentile of NULL_IDENTICAL TG_raw)

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_reps().
    target_alpha : float
        Target Type I error rate (default 0.05).

    Returns
    -------
    pd.DataFrame with columns: n_total, theta_raw, theta, n_reps_null, n_reps_total.
    """
    null_df = summary_df[summary_df['scenario'] == 'NULL_IDENTICAL'].copy()

    if len(null_df) == 0:
        # Fallback: use all null-like scenarios
        null_df = summary_df[summary_df['scenario_group'].isin(['null_identical', 'transportable'])].copy()

    thresholds = []
    for n_total, grp in null_df.groupby('n_total', sort=True):
        tg_vals = grp['tg_raw'].dropna().values
        if len(tg_vals) < 3:
            continue

        theta_raw = float(np.percentile(tg_vals, 100 * (1 - target_alpha)))
        theta = max(0.05, theta_raw)

        thresholds.append({
            'n_total': n_total,
            'n_a': int(grp['n_a'].iloc[0]) if 'n_a' in grp.columns else n_total // 2,
            'n_b': int(grp['n_b'].iloc[0]) if 'n_b' in grp.columns else n_total // 2,
            'theta_raw': theta_raw,
            'theta': theta,
            'theta_99': float(np.percentile(tg_vals, 99)),
            'tg_raw_mean_null': float(np.mean(tg_vals)),
            'tg_raw_std_null': float(np.std(tg_vals)),
            'n_reps': len(tg_vals),
        })

    return pd.DataFrame(thresholds)


# ═══════════════════════════════════════════════════════════════════════════
# Power curve computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_power_curves(
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute power (detection rate) vs. effect size across sample sizes.

    For each (n_total, scenario), computes the fraction of reps where
    TG_raw > calibrated threshold.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-rep results.
    threshold_df : pd.DataFrame
        Calibrated thresholds from calibrate_tg_thresholds().

    Returns
    -------
    pd.DataFrame with columns: n_total, scenario, delta_rho_mean, power_theta, n_reps.
    """
    theta_map = dict(zip(threshold_df['n_total'], threshold_df['theta']))

    results = []
    for (n_total, scenario), grp in summary_df.groupby(['n_total', 'scenario'], sort=False):
        theta = theta_map.get(n_total, 0.05)
        tg_vals = grp['tg_raw'].dropna().values
        delta_rhos = grp['delta_rho_true'].dropna().values

        if len(tg_vals) == 0:
            continue

        n_detect = int(np.sum(tg_vals > theta))
        n_total_reps = len(tg_vals)

        # Power curve variants
        results.append({
            'n_total': n_total,
            'scenario': scenario,
            'delta_rho_mean': float(np.mean(delta_rhos)),
            'delta_rho_std': float(np.std(delta_rhos)),
            'power_theta': n_detect / max(n_total_reps, 1),
            'power_theta_2x': int(np.sum(tg_vals > 2 * theta)) / max(n_total_reps, 1),
            'power_005': int(np.sum(tg_vals > 0.05)) / max(n_total_reps, 1),
            'tg_raw_mean': float(np.mean(tg_vals)),
            'n_reps': n_total_reps,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Full calibration suite
# ═══════════════════════════════════════════════════════════════════════════

# Default sample size grid (matches spec: 8, 11, 15, 20, 30, 50, 100)
DEFAULT_N_SAMPLES = (8, 11, 15, 20, 30, 50, 100)

# Scenarios for full calibration
CALIBRATION_SCENARIOS = (
    'NULL_IDENTICAL',
    'NULL_STRATUM_SHIFT',
    'TRANSPORTABLE',
    'NON_TRANSPORTABLE_WEAK',
    'NON_TRANSPORTABLE_STRONG',
    'SAMPLE_ASYMMETRY',
    'HIGH_DIM',
    'REALISTIC',
)


@_quiet_internal
def run_calibration(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Full calibration suite. Runs all 8 scenarios x sample sizes.

    Parameters
    ----------
    config : dict or None
        Override defaults. Keys:
          - n_samples_grid : list[int] (default DEFAULT_N_SAMPLES)
          - n_reps : int (default 100)
          - scenarios : list[str] (default CALIBRATION_SCENARIOS)
          - p_features : int (default 20)
          - n_batches : int (default 2)
          - seed : int (default 42)
          - output_dir : str

    Returns
    -------
    dict with keys:
        summary_df : pd.DataFrame  — per-rep results
        eval_df : pd.DataFrame     — evaluation metrics
        threshold_df : pd.DataFrame — calibrated thresholds
        power_df : pd.DataFrame    — power curves
    """
    if config is None:
        config = {}

    n_samples_grid = config.get('n_samples_grid', DEFAULT_N_SAMPLES)
    n_reps = config.get('n_reps', 100)
    scenarios = config.get('scenarios', CALIBRATION_SCENARIOS)
    p_features = config.get('p_features', 20)
    n_batches = config.get('n_batches', 2)
    n_subsamples = config.get('n_subsamples', 100)
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', os.path.join(os.getcwd(), 'output'))
    output_suffix = config.get('output_suffix', '')
    _ensure_dir(output_dir)

    all_results = []
    total_combos = len(scenarios) * len(n_samples_grid)

    combo_idx = 0
    for scenario in scenarios:
        for n_idx, n in enumerate(n_samples_grid):
            combo_idx += 1

            # Handle asymmetric sample sizes
            if scenario == 'SAMPLE_ASYMMETRY':
                n_per_cond = (50, 10)
            elif scenario == 'HIGH_DIM':
                n_per_cond = (min(20, n), min(20, n))
            else:
                n_per_cond = (n, n)

            p_actual = 50 if scenario == 'HIGH_DIM' else p_features

            print(f"\n[{combo_idx}/{total_combos}] Scenario: {scenario}  "
                  f"n_per_cond={n_per_cond}  p={p_actual}")

            rep_results = []
            for rep in range(n_reps):
                rep_seed = seed + combo_idx * 1000 + rep * 7
                res = simulate_one_rep(
                    scenario_name=scenario,
                    n_per_condition=n_per_cond,
                    p_features=p_actual,
                    n_batches=n_batches,
                    seed=rep_seed,
                    n_subsamples=n_subsamples,
                )
                rep_results.append(res)

            df_rep = summarize_reps(rep_results)
            df_rep['n_total'] = df_rep['n_a'] + df_rep['n_b']
            all_results.append(df_rep)

            # Print quick summary
            tg_raw_vals = df_rep['tg_raw'].dropna().values
            if len(tg_raw_vals) > 0:
                print(f"  TG_raw: mean={np.mean(tg_raw_vals):.4f}  "
                      f"std={np.std(tg_raw_vals):.4f}  "
                      f"p95={np.percentile(tg_raw_vals, 95):.4f}  "
                      f"delta_rho_mean={df_rep['delta_rho_true'].mean():.4f}")

    # Combine all results
    summary_df = pd.concat(all_results, ignore_index=True)

    # ── Evaluation ──
    eval_df = evaluate_simulation(summary_df)

    # ── Threshold calibration ──
    threshold_df = calibrate_tg_thresholds(summary_df, target_alpha=0.05)

    # ── Power curves ──
    power_df = compute_power_curves(summary_df, threshold_df)

    # ── Save CSVs ──
    summary_path = os.path.join(output_dir, f'tg_simulation_summary{output_suffix}.csv')
    eval_path = os.path.join(output_dir, f'tg_simulation_eval{output_suffix}.csv')
    threshold_path = os.path.join(output_dir, f'tg_simulation_thresholds{output_suffix}.csv')
    power_path = os.path.join(output_dir, f'tg_simulation_power{output_suffix}.csv')

    summary_df.to_csv(summary_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    power_df.to_csv(power_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved to {output_dir}/")
    print(f"  tg_simulation_summary.csv      ({len(summary_df)} rows)")
    print(f"  tg_simulation_eval.csv          ({len(eval_df)} rows)")
    print(f"  tg_simulation_thresholds.csv    ({len(threshold_df)} rows)")
    print(f"  tg_simulation_power.csv         ({len(power_df)} rows)")

    return {
        'summary_df': summary_df,
        'eval_df': eval_df,
        'threshold_df': threshold_df,
        'power_df': power_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
