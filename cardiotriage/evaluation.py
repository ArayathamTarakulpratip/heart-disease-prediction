"""Measure how good a model is: split, cross-validation, metrics, significance, tuning, learning curve."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, make_scorer,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, cross_validate, learning_curve,
                                     train_test_split)

from cardiotriage import config as cfg, data, models, triage

# Threshold metrics (recall, specificity, F1) use the default 0.5 cut-off here; the clinical cut-off is
# chosen later from out-of-fold predictions. ROC-AUC, PR-AUC and Brier score do not depend on any cut-off.
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "f1": "f1",
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
}
METRICS = list(SCORING)


def make_split(df: pd.DataFrame):
    """De-duplicate, then a stratified 80/20 split. The test part is locked until the final evaluation."""
    df = data.deduplicate(df)
    return train_test_split(df[cfg.FEATURES], df[cfg.TARGET], test_size=cfg.TEST_SIZE,
                            stratify=df[cfg.TARGET], random_state=cfg.SEED)


def make_cv(n_splits: int = cfg.CV_SPLITS, n_repeats: int = cfg.CV_REPEATS, seed: int = cfg.SEED):
    """Repeated stratified k-fold. Same seed -> same folds, so every model sees identical splits."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def metrics_at(y_true, y_pred) -> dict:
    """Recall, specificity, precision and accuracy for hard predictions (1 = disease)."""
    return {
        "recall": recall_score(y_true, y_pred),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def cv_scores(pipe, X, y, cv=None) -> pd.DataFrame:
    """One row per fold, one column per metric (Brier reported as a positive error)."""
    if cv is None:
        cv = make_cv()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=cfg.N_JOBS)
    out = pd.DataFrame({m: res[f"test_{m}"] for m in METRICS})
    out["brier"] = -out["brier"]
    return out


def cv_table(scores: dict) -> pd.DataFrame:
    """{model name: cv_scores frame} -> one row per model with mean and SD of every metric."""
    rows = {}
    for name, s in scores.items():
        rows[name] = {f"{m}_{stat}": getattr(s[m], stat)() for m in METRICS for stat in ("mean", "std")}
    t = pd.DataFrame(rows).T
    return t.rename(columns=lambda c: c.replace("_std", "_sd"))


def format_table(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """cv_table output -> one 'mean ± sd' string column per metric, for display."""
    metrics = [c[:-5] for c in table.columns if c.endswith("_mean")]
    return pd.DataFrame({m: [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(table[f"{m}_mean"], table[f"{m}_sd"])]
                         for m in metrics}, index=table.index)


def corrected_ttest(a, b, n_train: int, n_test: int) -> float:
    """Two-sided p-value for 'models a and b score the same' over the same CV folds.

    Nadeau & Bengio (2003) corrected resampled t-test: CV folds overlap, so the plain t-test is
    over-confident; the variance is inflated by n_test / n_train.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def fold_sizes(X) -> tuple[int, int]:
    """(n_train, n_test) of one fold of make_cv() on X, for corrected_ttest."""
    n_test = int(round(len(X) / cfg.CV_SPLITS))
    return len(X) - n_test, n_test


def select_model(table: pd.DataFrame, metric: str = "roc_auc", prefer: str = "Logistic Regression") -> str:
    """Best mean `metric` wins, but `prefer` (the interpretable model) wins if within 1 SD of the best."""
    best = table[f"{metric}_mean"].idxmax()
    if prefer in table.index and table.loc[prefer, f"{metric}_mean"] >= (
            table.loc[best, f"{metric}_mean"] - table.loc[best, f"{metric}_sd"]):
        return prefer
    return best


def _search(name: str, n_iter: int, use_fe: bool, drop=()) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        models.build_pipeline(models.MODEL_ZOO[name], use_fe, drop=drop), models.SEARCH_SPACES[name],
        n_iter=n_iter, scoring="roc_auc",
        cv=make_cv(n_repeats=2, seed=cfg.SEED + 1), random_state=cfg.SEED, n_jobs=cfg.N_JOBS, refit=True)


def tune(name: str, X, y, n_iter: int = 30, use_fe: bool = False, drop=()) -> RandomizedSearchCV:
    """Random search over models.SEARCH_SPACES[name], scored by ROC-AUC on a 5x2 CV of the training set.

    Uses different folds (seed + 1) from the 5x10 comparison CV, so tuning and reporting do not share splits.
    """
    return _search(name, n_iter, use_fe, drop=drop).fit(X, y)


def nested_cv(name: str, X, y, n_iter: int = 30, use_fe: bool = False) -> np.ndarray:
    """ROC-AUC of the whole tuning procedure on 5 outer folds (seed + 2): each outer fold is scored by a model
    whose settings were searched on the other 4 folds only, so the estimate is not inflated by tuning."""
    outer = StratifiedKFold(cfg.CV_SPLITS, shuffle=True, random_state=cfg.SEED + 2)
    return cross_val_score(_search(name, n_iter, use_fe), X, y, cv=outer, scoring="roc_auc")


def calibrated(pipe, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Wrap a pipeline in Platt (sigmoid) calibration fitted with an inner 5-fold CV."""
    return CalibratedClassifierCV(pipe, method=method, cv=5)


def learning_curve_table(pipe, X, y, sizes=(0.2, 0.4, 0.6, 0.8, 1.0)) -> pd.DataFrame:
    """ROC-AUC on training and validation folds as the training set grows."""
    n, tr, va = learning_curve(pipe, X, y, train_sizes=list(sizes), cv=make_cv(n_repeats=4),
                               scoring="roc_auc", n_jobs=cfg.N_JOBS, shuffle=True, random_state=cfg.SEED)
    return pd.DataFrame({"train_size": n, "train_mean": tr.mean(1), "train_sd": tr.std(1),
                         "valid_mean": va.mean(1), "valid_sd": va.std(1)})


# ---------------------------------------------------------------- decision layer and final evaluation

def oof_proba(pipe, X, y, n_repeats: int = cfg.CV_REPEATS) -> np.ndarray:
    """Out-of-fold P(disease) for every training patient, averaged over `n_repeats` 5-fold splits.

    Each prediction comes from a model that never saw that patient. Selecting and evaluating
    cut-offs on this same set of predictions is still optimistic; use nested_decision_cv for
    an independent assessment of the complete decision rule.
    """
    runs = [cross_val_predict(pipe, X, y, method="predict_proba", n_jobs=cfg.N_JOBS,
                              cv=StratifiedKFold(cfg.CV_SPLITS, shuffle=True, random_state=cfg.SEED + r))[:, 1]
            for r in range(n_repeats)]
    return np.mean(runs, axis=0)


def nested_decision_cv(name: str, X, y, *, drop=(), n_iter: int = 10) -> pd.DataFrame:
    """Assess tuning, threshold selection and tier assignment on separate outer folds.

    Each outer training fold tunes C, obtains inner out-of-fold probabilities to set both
    cut-offs, then scores patients in the untouched outer fold. The held-out test set is
    excluded. This estimates the rule as implemented, conditional on choosing LR beforehand.
    """
    outer = StratifiedKFold(cfg.CV_SPLITS, shuffle=True, random_state=cfg.SEED + 3)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        search = _search(name, n_iter, use_fe=False, drop=drop).fit(X_tr, y_tr)
        pipe = search.best_estimator_
        inner_p = oof_proba(pipe, X_tr, y_tr, n_repeats=1)
        cuts = triage.pick_thresholds(y_tr, inner_p)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        rates = _rates(y_te, p >= cuts["low"])
        high_flag = p >= cuts["high"]
        rows.append({"fold": fold, "low": cuts["low"], "high": cuts["high"],
                     "recall": rates["recall"], "specificity": rates["specificity"],
                     "ppv": rates["ppv"], "low_patients": int((p < cuts["low"]).sum()),
                     "disease_in_low": int(((p < cuts["low"]) & (y_te.to_numpy() == 1)).sum()),
                     "high_patients": int(high_flag.sum()),
                     "disease_in_high": int((high_flag & (y_te.to_numpy() == 1)).sum()),
                     "test_patients": len(te), "test_disease": int(y_te.sum())})
    return pd.DataFrame(rows).set_index("fold")


def operating_points(pipes: dict, X, y) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each model at its OWN clinical cut-off: the highest one that keeps out-of-fold recall >= RECALL_TARGET.

    Returns one row per model (cut-off, recall, specificity, PPV, missed per 100, out-of-fold PR-AUC) and a
    frame of out-of-fold flags (True = at or above that model's own cut-off), one column per model, indexed like y.
    """
    rows, flags = {}, {}
    for name, pipe in pipes.items():
        p = oof_proba(pipe, X, y)
        cut = triage.pick_thresholds(y, p)["low"]
        row = threshold_sweep(y, p, [cut]).iloc[0]
        rows[name] = {"threshold": cut, "recall": row["recall"], "specificity": row["specificity"],
                      "ppv": row["ppv"], "missed_per_100": row["missed_per_100"],
                      "pr_auc": average_precision_score(y, p)}
        flags[name] = p >= cut
    return pd.DataFrame(rows).T, pd.DataFrame(flags, index=y.index)


def _rates(y, flag) -> dict:
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    tp, fp = (flag & y).sum(), (flag & ~y).sum()
    fn, tn = (~flag & y).sum(), (~flag & ~y).sum()
    div = lambda a, b: a / b if b else np.nan
    return {"recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_sweep(y, p, thresholds) -> pd.DataFrame:
    """Rates at each cut-off ("flag" = P(disease) >= cut-off), plus what it means per 100 patients."""
    n, rows = len(y), []
    for t in thresholds:
        r = _rates(y, np.asarray(p) >= t)
        rows.append({"threshold": t, "recall": r["recall"], "specificity": r["specificity"], "ppv": r["ppv"],
                     "npv": r["npv"], "flagged_per_100": 100 * (r["tp"] + r["fp"]) / n,
                     "caught_per_100": 100 * r["tp"] / n, "missed_per_100": 100 * r["fn"] / n})
    return pd.DataFrame(rows)


def net_benefit(y, p, thresholds) -> pd.DataFrame:
    """Decision-curve analysis (Vickers & Elkin 2006): benefit of acting on the model at each risk cut-off,
    against acting on everyone (treat all) or no one (treat none)."""
    y, p, n = np.asarray(y), np.asarray(p), len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        w = t / (1 - t)                                    # how many false alarms one true case is worth
        flag = p >= t
        rows.append({"threshold": t,
                     "model": (flag & (y == 1)).sum() / n - (flag & (y == 0)).sum() / n * w,
                     "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(rows)


def queue_gain(y, p, fractions) -> pd.DataFrame:
    """Simulate disease share in early risk-ordered slots against a proportional benchmark.

    The data contain neither arrival order nor waiting times. The benchmark assumes disease
    is spread evenly through an imagined arrival sequence; it is not an observed queue.
    """
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p, float), kind="stable")    # highest risk first; ties keep input order
    served = np.cumsum(y[order]) / y.sum()
    rows = [{"slots": f, "model": served[max(round(f * len(y)), 1) - 1], "proportional_benchmark": f}
            for f in fractions]
    return pd.DataFrame(rows)


def curves(y, p, n_bins: int = 5) -> dict:
    """ROC, precision-recall and calibration curves for plotting (plots.py cannot import sklearn)."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {"roc": (fpr, tpr, roc_auc_score(y, p)), "pr": (rec, prec, average_precision_score(y, p), float(np.mean(y))),
            "calibration": (prob_pred, prob_true)}


def confusion(y, flag) -> pd.DataFrame:
    """2x2 table with the TRUE status as rows and the model's call as columns (fixes trap T11)."""
    r = _rates(y, flag)
    return pd.DataFrame([[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                        index=["Truly no disease", "Truly disease"], columns=["Not flagged", "Flagged"])


def _metrics(y, p, threshold) -> dict:
    r = _rates(y, p >= threshold)
    return {"roc_auc": roc_auc_score(y, p), "recall": r["recall"], "specificity": r["specificity"],
            "ppv": r["ppv"], "npv": r["npv"], "brier": brier_score_loss(y, p)}


def bootstrap_ci(y, p, threshold: float, n_boot: int = 1000, seed: int = cfg.SEED) -> pd.DataFrame:
    """Point estimate and percentile 95% CI of each metric over patient-level bootstrap resamples."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():                       # a one-class resample has no AUC; skip it
            continue
        draws.append(_metrics(y[i], p[i], threshold))
    d = pd.DataFrame(draws)
    point = _metrics(y, p, threshold)
    return pd.DataFrame({"point": point, "lo": d.quantile(0.025), "hi": d.quantile(0.975)})


def subgroup_recall(group, y, p, threshold: float) -> pd.DataFrame:
    """Recall of the flag (P >= threshold) inside each subgroup, with the counts it rests on."""
    df = pd.DataFrame({"group": np.asarray(group), "y": np.asarray(y), "flag": np.asarray(p) >= threshold})
    sick = df[df["y"] == 1]
    out = pd.DataFrame({"patients": df.groupby("group").size(),
                        "with_disease": sick.groupby("group").size(),
                        "caught": sick.groupby("group")["flag"].sum()}).fillna(0).astype(int)
    out["recall"] = out["caught"] / out["with_disease"].replace(0, np.nan)
    return out


def exact_ci(successes: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% interval for a proportion. Unlike a bootstrap it stays honest at 0 or n of n."""
    ci = stats.binomtest(int(successes), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


# ---------------------------------------------------------------- interpretation

def permutation_table(pipe, X, y, n_repeats: int = 30) -> pd.Series:
    """Drop in ROC-AUC when each clinical feature is shuffled (bigger = the model relies on it more)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=cfg.SEED, n_jobs=cfg.N_JOBS)
    s = pd.Series(r.importances_mean, index=X.columns)
    return s.reindex(s.abs().sort_values(ascending=False).index)


def coefficients(pipe) -> pd.Series:
    """Logistic-regression coefficients on the standardised / one-hot inputs, largest magnitude first.

    Numeric inputs are standardised, so a coefficient is the change in log-odds per 1 SD (fixes trap T10).
    """
    s = pd.Series(pipe[-1].coef_[0], index=pipe[:-1].get_feature_names_out())
    return s.reindex(s.abs().sort_values(ascending=False).index)
