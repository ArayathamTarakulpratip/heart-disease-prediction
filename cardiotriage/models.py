"""What we train: the five model families, their tuning ranges, and the full pipeline."""
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from cardiotriage import config as cfg, features

# One estimator per family, in the order the notebook presents them. Workers run in parallel at the
# CV level (cfg.N_JOBS), so each model uses a single thread to stay deterministic and avoid oversubscription.
MODEL_ZOO = {
    "Dummy (prior)": DummyClassifier(strategy="prior", random_state=cfg.SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=cfg.SEED),
    "KNN": KNeighborsClassifier(),                      # deterministic: no random_state parameter
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=cfg.SEED, n_jobs=1),
    "XGBoost": XGBClassifier(random_state=cfg.SEED, n_jobs=1, eval_metric="logloss"),
}

# RandomizedSearchCV spaces (prefixed with the pipeline step name "model__").
SEARCH_SPACES = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    },
    "KNN": {
        "model__n_neighbors": randint(3, 51),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Random Forest": {
        "model__max_depth": [None, 3, 5, 8],
        "model__min_samples_leaf": randint(1, 20),
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "model__n_estimators": randint(50, 400),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}


def build_pipeline(model, use_fe: bool = False, drop=()) -> Pipeline:
    """Raw 13 columns -> [domain features] -> impute/scale/one-hot -> model. `model` is cloned, never mutated."""
    steps = [("fe", features.domain_step())] if use_fe else []
    steps += [("pre", features.build_preprocessor(use_fe, drop)), ("model", clone(model))]
    return Pipeline(steps)


def zoo_pipeline(name: str, use_fe: bool = False) -> Pipeline:
    return build_pipeline(MODEL_ZOO[name], use_fe)
