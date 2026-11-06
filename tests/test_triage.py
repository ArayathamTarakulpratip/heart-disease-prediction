"""Cut-offs, tiers and per-patient contributions (triage.py)."""
import numpy as np

from tests.helpers import fitted_lr


def test_pick_thresholds_meets_recall_target_and_orders_cuts():
    from cardiotriage import triage
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.35 + rng.uniform(0, 0.65, 400), 0, 1)
    t = triage.pick_thresholds(y, p, recall_target=0.9, ppv_target=0.85)
    assert ((p >= t["low"])[y == 1]).mean() >= 0.9
    assert ((p >= np.nextafter(t["low"], 1) + 1e-9)[y == 1]).mean() < 0.9 or t["low"] == p.max()
    assert t["high"] >= t["low"]


def test_pick_thresholds_high_is_the_lowest_cut_reaching_the_ppv_target():
    # PPV of "flag if p >= cut", by cut: .9->1, .8->1, .7->2/3, .6->3/4, .5->4/5, .4->4/6, .3->5/7
    from cardiotriage import triage
    y = np.array([1, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    t = triage.pick_thresholds(y, p, recall_target=1.0, ppv_target=0.78)
    assert t == {"low": 0.3, "high": 0.5}, t                  # .9, .8 and .5 qualify; the lowest is .5
    assert y[p >= t["high"]].mean() >= 0.78
    # no cut reaches the target -> fall back to the cut with the best PPV
    t = triage.pick_thresholds(y, p, recall_target=1.0, ppv_target=1.01)
    assert t["high"] == 0.9, t


def test_triage_tier_boundaries():
    from cardiotriage import triage
    th = {"low": 0.2, "high": 0.7}
    assert triage.triage(0.7, th)["tier"] == "High"
    assert triage.triage(0.2, th)["tier"] == "Medium"
    assert triage.triage(0.1999, th)["tier"] == "Low"
    assert all(triage.triage(p, th)["action"] for p in (0.1, 0.5, 0.9))


def test_contributions_are_exact_linear_shap_values():
    from cardiotriage import config as cfg, triage
    pipe, X = fitted_lr()
    base = triage.baseline(pipe, X)
    patient = X.iloc[[5]]
    c = triage.contributions(pipe, patient, base)
    assert set(c.index) == set(cfg.FEATURES), c.index
    model = pipe.named_steps["model"]
    logit = pipe.decision_function(patient)[0]
    base_logit = model.intercept_[0] + model.coef_[0] @ base
    assert abs(c.sum() - (logit - base_logit)) < 1e-6, (c.sum(), logit - base_logit)


def test_contributions_reject_non_linear_models():
    from sklearn.ensemble import RandomForestClassifier
    from cardiotriage import config as cfg, data, models, triage
    df = data.load_cleveland()
    rf = models.build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0)).fit(df[cfg.FEATURES], df[cfg.TARGET])
    try:
        triage.contributions(rf, df[cfg.FEATURES].iloc[[0]], np.zeros(5))
    except ValueError:
        pass
    else:
        raise AssertionError("contributions must refuse a non-linear model")
