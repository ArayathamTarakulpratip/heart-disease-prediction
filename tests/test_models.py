"""Features, pipelines and the model zoo (features.py, models.py)."""
import subprocess
import sys

import numpy as np
import pandas as pd


def test_domain_features_values():
    from cardiotriage import features
    x = pd.DataFrame({"age": [60.0, 40.0], "thalach": [128.0, 180.0], "trestbps": [130.0, 120.0],
                      "chol": [240.0, 200.0], "oldpeak": [1.5, 1.0]})
    out = features.add_domain_features(x)
    assert np.allclose(out["pct_max_hr"], [128 / 160, 180 / 180]), out["pct_max_hr"]
    assert out["hypertension"].tolist() == [1, 0]
    assert out["high_chol"].tolist() == [1, 0]
    assert out["st_depression_gt1"].tolist() == [1, 0]          # strictly greater than 1 mm
    assert "pct_max_hr" not in x.columns, "must not mutate the caller's frame"


def test_build_pipeline_fits_and_predicts_with_missing_values():
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import config as cfg, data, models
    df = data.load_cleveland()                                   # contains NaN in ca and thal
    for use_fe in (False, True):
        pipe = models.build_pipeline(LogisticRegression(max_iter=1000), use_fe=use_fe)
        pipe.fit(df[cfg.FEATURES], df[cfg.TARGET])
        p = pipe.predict_proba(df[cfg.FEATURES])[:, 1]
        assert p.shape == (303,) and (p >= 0).all() and (p <= 1).all()
    names = list(pipe[:-1].get_feature_names_out())
    assert "fe" in dict(pipe.steps) and any("pct_max_hr" in n for n in names), names


def test_pipeline_joblib_roundtrip_needs_only_features_module():
    import tempfile
    import joblib
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import config as cfg, data, models
    df = data.load_cleveland()
    pipe = models.build_pipeline(LogisticRegression(max_iter=1000), use_fe=True).fit(df[cfg.FEATURES], df[cfg.TARGET])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.joblib"
        joblib.dump(pipe, path)   # loading below is safe: the file was written by this test, never from outside
        code = (
            "import sys, joblib; from cardiotriage import config as cfg, data;"
            f"p = joblib.load(r'{path}');"
            "x = data.load_cleveland()[cfg.FEATURES].head(3); print(p.predict_proba(x)[:, 1].round(3));"
            "loaded = sorted(m for m in sys.modules if m.startswith('cardiotriage.'));"
            "extra = set(loaded) - {'cardiotriage.config', 'cardiotriage.data', 'cardiotriage.features'};"
            "print(loaded); sys.exit(1 if extra else 0)"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr[-500:]


def test_model_zoo_has_five_families_with_fixed_seeds():
    from cardiotriage import config as cfg, models
    assert list(models.MODEL_ZOO) == ["Dummy (prior)", "Logistic Regression", "KNN", "Random Forest", "XGBoost"]
    for name, est in models.MODEL_ZOO.items():
        params = est.get_params()
        if "random_state" in params:
            assert params["random_state"] == cfg.SEED, (name, params["random_state"])


def test_tune_returns_a_fitted_search_with_best_params():
    from cardiotriage import data, evaluation, models
    X, _, y, _ = evaluation.make_split(data.load_cleveland())
    search = evaluation.tune("Logistic Regression", X, y, n_iter=3)
    assert "model__C" in search.best_params_ and 0.5 < search.best_score_ <= 1


def test_pipeline_with_dropped_columns_scores_hungary_despite_missing_values():
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import config as cfg, data, models
    cle, hun = data.load_cleveland(), data.load_hungarian()
    drop = ["ca", "thal", "slope"]
    pipe = models.build_pipeline(LogisticRegression(max_iter=1000), drop=drop).fit(cle[cfg.FEATURES], cle[cfg.TARGET])
    p = pipe.predict_proba(hun[cfg.FEATURES])[:, 1]
    assert len(p) == 294 and np.isfinite(p).all()
    assert not any(c in drop for c in pipe[:-1].get_feature_names_out())
