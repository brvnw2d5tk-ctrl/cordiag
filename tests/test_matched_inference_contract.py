import numpy as np
import pandas as pd

from cordiag import calibration, tg, zpg


def test_zpg_upper_tail_pvalue_uses_only_valid_null_draws():
    null = np.array([-0.1, 0.0, 0.1, 0.2, np.nan])

    assert zpg._upper_tail_permutation_pvalue(null, observed=0.15) == 0.4


def test_zpg_decision_distinguishes_identifiability_and_inconclusive_state():
    assert zpg.decide(2.0, 0.001, n_per_condition=53, design_identifiable=False) == "NOT_IDENTIFIABLE_DESIGN"
    assert zpg.decide(1.1, 0.09, n_per_condition=53) == "GO"
    assert zpg.decide(0.2, 0.80, n_per_condition=53) == "INCONCLUSIVE"
    assert zpg.decide(-0.1, 0.01, n_per_condition=53) == "NO_GO"


def test_matched_tg_permutation_uses_the_observed_matched_estimators(monkeypatch):
    called = []

    def within(*args, **kwargs):
        called.append(("within", args[3], kwargs["n_subsamples"], kwargs["seed"]))
        return 0.5, 0.0, 1.0, 1.0, 1.0

    def cross(*args, **kwargs):
        called.append(("cross", args[6], kwargs["n_subsamples"], kwargs["seed"]))
        return 0.1, 1.0

    def crossed(*args, **kwargs):
        called.append(("crossed", args[4], kwargs["n_subsamples"], kwargs["seed"]))
        return 0.3, 1.0, 0.0

    def forbidden(*args, **kwargs):
        raise AssertionError("LOOCV estimator entered a matched null")

    monkeypatch.setattr(tg, "_compute_q2_within_matched", within)
    monkeypatch.setattr(tg, "_compute_q2_cross_matched", cross)
    monkeypatch.setattr(tg, "_compute_q2_crossed_matched", crossed)
    monkeypatch.setattr(tg, "_compute_q2_within", forbidden)
    monkeypatch.setattr(tg, "_compute_q2_cross", forbidden)
    monkeypatch.setattr(tg, "_compute_q2_crossed", forbidden)

    tg._permutation_test_tg(
        np.linspace(-1, 1, 8), np.linspace(-0.5, 1.5, 12),
        np.arange(16, dtype=float).reshape(8, 2), np.arange(24, dtype=float).reshape(12, 2),
        np.repeat("source_batch", 8), np.repeat("target_batch", 12), [1.0], 10,
        np.random.default_rng(7), 0.4, 0.2, 0.2,
        cv_mode="matched_subsample", n_subsamples=3, n_source=8,
        train_size=6, matched_seed=11,
    )

    expected = []
    for permutation_index in range(10):
        seed = 100011 + permutation_index
        expected.extend([
            ("within", 6, 3, seed),
            ("cross", 6, 3, seed),
            ("crossed", 6, 3, seed),
        ])
    assert called == expected


def test_matched_bootstrap_uses_the_observed_training_size(monkeypatch):
    train_lengths = []
    monkeypatch.setattr(tg, "_compute_q2_within_matched", lambda *args, **kwargs: (0.5, 0.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(tg, "_compute_q2_cross_matched", lambda *args, **kwargs: (0.1, 1.0))
    monkeypatch.setattr(tg, "_compute_stratum_means_loocv", lambda *args, **kwargs: (np.zeros(len(args[0])), 1.0))

    def record(P_train, X_train, strata_train, P_test, X_test, strata_test, alphas):
        train_lengths.append(len(P_train))
        return np.zeros(len(P_test))

    monkeypatch.setattr(tg, "_m1_train_test", record)
    tg._bootstrap_ci_tg(
        np.linspace(-1, 1, 12), np.linspace(-0.5, 1.5, 20),
        np.arange(24, dtype=float).reshape(12, 2), np.arange(40, dtype=float).reshape(20, 2),
        np.repeat("source_batch", 12), np.repeat("target_batch", 20), [1.0], 20,
        np.random.default_rng(8), 1.0, 20, cv_mode="matched_subsample",
        train_size=10, n_subsamples=3, matched_seed=12,
    )
    assert train_lengths and set(train_lengths) == {10}


def test_calibration_uses_matched_cross_estimator(monkeypatch):
    called = []
    monkeypatch.setattr(calibration, "_compute_stratum_means_loocv", lambda *args, **kwargs: (np.zeros(len(args[0])), 1.0))
    monkeypatch.setattr(calibration, "_compute_q2_within_matched", lambda *args, **kwargs: (0.5, 0.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(calibration, "_compute_q2_crossed_matched", lambda *args, **kwargs: (0.3, 1.0, 0.0))

    def matched_cross(*args, **kwargs):
        called.append(True)
        return 0.1, 1.0

    def forbidden(*args, **kwargs):
        raise AssertionError("calibration used a full-source cross estimator")

    monkeypatch.setattr(calibration, "_compute_q2_cross_matched", matched_cross, raising=False)
    monkeypatch.setattr(calibration, "_m1_train_test", forbidden)
    design_a = pd.DataFrame({"condition": ["a"] * 10, "batch": ["b1"] * 10})
    design_b = pd.DataFrame({"condition": ["b"] * 20, "batch": ["b1"] * 20})
    calibration._compute_q2_components(
        np.arange(20, dtype=float).reshape(10, 2), np.arange(40, dtype=float).reshape(20, 2),
        np.linspace(-1, 1, 10), np.linspace(-1, 1, 20),
        design_a, design_b, pd.concat([design_a, design_b], ignore_index=True), n_subsamples=3,
    )
    assert called == [True]
