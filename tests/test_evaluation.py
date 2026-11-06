"""Split, cross-validation, metrics and statistics (evaluation.py)."""
import numpy as np
import pandas as pd

from tests.helpers import fitted_lr, cleveland_target


def test_make_split_is_stratified_disjoint_and_deterministic():
    from cardiotriage import data, evaluation
    df = data.load_cleveland()
    X_tr, X_te, y_tr, y_te = evaluation.make_split(df)
    assert (len(X_tr), len(X_te)) == (242, 61), (len(X_tr), len(X_te))
    assert not set(X_tr.index) & set(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02, (y_tr.mean(), y_te.mean())
    X_tr2, X_te2, _, _ = evaluation.make_split(df)
    assert list(X_te.index) == list(X_te2.index)


def test_make_cv_gives_50_identical_folds_each_time():
    from cardiotriage import data, evaluation
    df = data.load_cleveland()
    X, _, y, _ = evaluation.make_split(df)
    a = [tuple(te) for _, te in evaluation.make_cv().split(X, y)]
    b = [tuple(te) for _, te in evaluation.make_cv().split(X, y)]
    assert len(a) == 50 and a == b


def test_metrics_at_matches_hand_computed_toy_example():
    from cardiotriage import evaluation
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = evaluation.metrics_at(y_true, y_pred)
    assert np.isclose(m["recall"], 3 / 4) and np.isclose(m["specificity"], 5 / 6)
    assert np.isclose(m["precision"], 3 / 4) and np.isclose(m["accuracy"], 8 / 10)


def test_cv_scores_returns_one_row_per_fold_with_all_metrics():
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import data, evaluation, models
    X, _, y, _ = evaluation.make_split(data.load_cleveland())
    s = evaluation.cv_scores(models.build_pipeline(LogisticRegression(max_iter=1000)), X, y)
    assert len(s) == 50 and set(evaluation.METRICS) <= set(s.columns), s.columns
    assert s["roc_auc"].between(0.5, 1).all() and s["brier"].between(0, 1).all()
    assert "pr_auc" in evaluation.METRICS and s["pr_auc"].between(0, 1).all(), s.get("pr_auc")


def test_corrected_ttest_p_value_is_a_probability_and_detects_a_big_gap():
    from cardiotriage import evaluation
    rng = np.random.default_rng(0)
    a = rng.normal(0.85, 0.03, 50)
    p_same = evaluation.corrected_ttest(a, a + rng.normal(0, 0.01, 50), n_train=194, n_test=48)
    p_gap = evaluation.corrected_ttest(a + 0.15, a, n_train=194, n_test=48)
    assert 0 <= p_same <= 1 and p_gap < 0.05, (p_same, p_gap)


def test_select_model_prefers_lr_within_one_sd_of_best():
    from cardiotriage import evaluation
    t = pd.DataFrame({"roc_auc_mean": [0.90, 0.88, 0.80], "roc_auc_sd": [0.04, 0.05, 0.05]},
                     index=["XGBoost", "Logistic Regression", "KNN"])
    assert evaluation.select_model(t) == "Logistic Regression"
    t.loc["Logistic Regression", "roc_auc_mean"] = 0.80
    assert evaluation.select_model(t) == "XGBoost"


def test_format_table_shows_mean_plus_minus_sd():
    from cardiotriage import evaluation
    t = pd.DataFrame({"roc_auc_mean": [0.8944], "roc_auc_sd": [0.0381]}, index=["LR"])
    assert evaluation.format_table(t).loc["LR", "roc_auc"] == "0.894 ± 0.038"


def test_threshold_sweep_reports_rates_and_per_100_counts():
    from cardiotriage import evaluation
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    row = evaluation.threshold_sweep(y, p, [0.5]).iloc[0]
    assert np.isclose(row["recall"], 0.5) and np.isclose(row["specificity"], 0.5)
    assert np.isclose(row["flagged_per_100"], 50) and np.isclose(row["caught_per_100"], 25)
    assert np.isclose(row["missed_per_100"], 25)


def test_net_benefit_matches_formula_for_perfect_model_and_treat_all():
    from cardiotriage import evaluation
    y = np.array([1, 1, 0, 0, 0])
    nb = evaluation.net_benefit(y, y.astype(float), [0.2]).iloc[0]
    assert np.isclose(nb["model"], 0.4)                      # perfect model: NB = prevalence
    assert np.isclose(nb["treat_all"], 0.4 - 0.6 * 0.2 / 0.8)
    assert nb["treat_none"] == 0


def test_oof_proba_is_deterministic_and_covers_every_patient():
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import data, evaluation, models
    X, _, y, _ = evaluation.make_split(data.load_cleveland())
    pipe = models.build_pipeline(LogisticRegression(max_iter=1000))
    a = evaluation.oof_proba(pipe, X, y, n_repeats=2)
    b = evaluation.oof_proba(pipe, X, y, n_repeats=2)
    assert a.shape == (len(X),) and np.array_equal(a, b) and ((a > 0) & (a < 1)).all()


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    from cardiotriage import evaluation
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 120), 0, 1)
    a = evaluation.bootstrap_ci(y, p, threshold=0.5, n_boot=300)
    b = evaluation.bootstrap_ci(y, p, threshold=0.5, n_boot=300)
    assert a.equals(b)
    assert {"roc_auc", "recall", "specificity", "ppv", "npv", "brier"} <= set(a.index), a.index
    assert ((a["lo"] <= a["point"]) & (a["point"] <= a["hi"])).all(), a


def test_bootstrap_ci_skips_one_class_resamples_without_warnings():
    # with 6 patients ~1 in 32 resamples has a single class: AUC is undefined (sklearn warns, returns NaN),
    # so those resamples must be skipped whole, keeping every metric on the same draws and the run warning-free
    import warnings
    from cardiotriage import evaluation
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([.9, .7, .4, .6, .3, .1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ci = evaluation.bootstrap_ci(y, p, threshold=0.5, n_boot=200)
    assert np.isfinite(ci[["lo", "hi"]].to_numpy()).all(), ci


def test_subgroup_recall_counts_patients_per_group():
    from cardiotriage import evaluation
    df = pd.DataFrame({"group": ["a", "a", "a", "b", "b"], "y": [1, 1, 0, 1, 0], "p": [0.9, 0.1, 0.8, 0.7, 0.2]})
    t = evaluation.subgroup_recall(df["group"], df["y"], df["p"], threshold=0.5)
    assert t.loc["a", "patients"] == 3 and t.loc["a", "with_disease"] == 2 and np.isclose(t.loc["a", "recall"], 0.5)
    assert np.isclose(t.loc["b", "recall"], 1.0)


def test_confusion_has_clinical_labels_with_true_rows():
    # T11: rows are the true status, columns the model's call
    from cardiotriage import evaluation
    cm = evaluation.confusion([1, 1, 0, 0], np.array([1, 0, 0, 0]) == 1)
    assert list(cm.index) == ["Truly no disease", "Truly disease"], cm.index
    assert cm.loc["Truly disease", "Flagged"] == 1 and cm.loc["Truly disease", "Not flagged"] == 1


def test_exact_ci_gives_a_real_interval_when_every_case_is_caught():
    # a bootstrap of 28/28 collapses to [1, 1]; Clopper-Pearson does not
    from cardiotriage import evaluation
    lo, hi = evaluation.exact_ci(28, 28)
    assert abs(lo - 0.025 ** (1 / 28)) < 1e-6 and hi == 1.0, (lo, hi)    # n of n: lower bound = (alpha/2)^(1/n)
    lo2, hi2 = evaluation.exact_ci(14, 18)
    assert 0 < lo2 < 14 / 18 < hi2 < 1


def test_permutation_table_and_coefficients_cover_the_features():
    from cardiotriage import config as cfg, evaluation, triage
    pipe, X = fitted_lr()
    imp = evaluation.permutation_table(pipe, X, cleveland_target(), n_repeats=3)
    assert set(imp.index) == set(cfg.FEATURES) and imp.index[0] == imp.abs().idxmax()
    coef = evaluation.coefficients(pipe)
    assert len(coef) == len(pipe[:-1].get_feature_names_out()) and "age" in coef.index


def test_operating_points_cut_each_model_at_its_own_recall_target():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from cardiotriage import config as cfg, data, evaluation, models
    X, _, y, _ = evaluation.make_split(data.load_cleveland())
    pipes = {"LR": models.build_pipeline(LogisticRegression(max_iter=1000)),
             "KNN": models.build_pipeline(KNeighborsClassifier(n_neighbors=25))}
    table, flags = evaluation.operating_points(pipes, X, y)
    assert list(table.index) == ["LR", "KNN"] and list(flags.columns) == ["LR", "KNN"]
    assert (table["recall"] >= cfg.RECALL_TARGET).all(), table["recall"]
    assert table["pr_auc"].between(0, 1).all() and table["specificity"].between(0, 1).all()
    assert flags.index.equals(y.index) and flags.dtypes.eq(bool).all()
    # the table's numbers are the flags' numbers
    sick = (y == 1).to_numpy()
    for m in pipes:
        assert np.isclose(table.loc[m, "recall"], flags[m][sick].mean())
        assert np.isclose(table.loc[m, "missed_per_100"], 100 * (~flags[m] & sick).sum() / len(y))


def test_nested_cv_gives_one_unbiased_score_per_outer_fold_and_is_deterministic():
    from cardiotriage import data, evaluation
    X, _, y, _ = evaluation.make_split(data.load_cleveland())
    a = evaluation.nested_cv("Logistic Regression", X, y, n_iter=3)
    b = evaluation.nested_cv("Logistic Regression", X, y, n_iter=3)
    assert len(a) == 5 and np.all((a >= 0) & (a <= 1)), a
    assert np.array_equal(a, b)


def test_queue_gain_matches_a_hand_computed_queue():
    from cardiotriage import evaluation
    # 6 patients, 3 sick; the model ranks them 0 > 1 > 2 > 3 > 4 > 5
    y = [1, 0, 1, 0, 0, 1]
    p = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    g = evaluation.queue_gain(y, p, [1 / 3, 0.5, 1.0]).set_index("slots")
    # first 2 slots: patients 0, 1 -> 1 of 3 sick; first 3: 0, 1, 2 -> 2 of 3; all 6 -> 3 of 3
    assert np.allclose(g["model"], [1 / 3, 2 / 3, 1.0]), g
    assert np.allclose(g["first_come_first_served"], [1 / 3, 0.5, 1.0])


def test_queue_gain_of_a_perfect_ranker_serves_every_sick_patient_first():
    from cardiotriage import evaluation
    y = [0, 1, 0, 1, 0, 1]
    g = evaluation.queue_gain(y, y, [0.5])
    assert g["model"].iloc[0] == 1.0
