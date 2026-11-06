"""Turn raw columns into model inputs: clinical domain features and the preprocessing step.

Imports only `config` from the package: the saved model contains objects from this module, and the
app must be able to unpickle them without loading training code. Functions are module-level
(never lambdas) so the fitted pipeline can be pickled.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cardiotriage import config as cfg

# engineered columns: (name, group). Thresholds: ACC/AHA 2017 stage-1 hypertension (systolic >= 130),
# NCEP high total cholesterol (>= 240 mg/dl), ST depression > 1 mm as a marked ischaemic response.
DOMAIN_NUMERIC = ["pct_max_hr"]
DOMAIN_BINARY = ["hypertension", "high_chol", "st_depression_gt1"]


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with the 4 clinical features added."""
    X = X.copy()
    X["pct_max_hr"] = X["thalach"] / (220 - X["age"])      # share of the age-predicted maximum heart rate
    X["hypertension"] = (X["trestbps"] >= 130).astype(int)
    X["high_chol"] = (X["chol"] >= 240).astype(int)
    X["st_depression_gt1"] = (X["oldpeak"] > 1).astype(int)
    return X


def domain_feature_names(transformer, input_features):
    """`feature_names_out` hook so the pipeline can report column names after the FE step."""
    return list(input_features) + DOMAIN_NUMERIC + DOMAIN_BINARY


def domain_step() -> FunctionTransformer:
    return FunctionTransformer(add_domain_features, feature_names_out=domain_feature_names)


def build_preprocessor(use_fe: bool = False, drop=()) -> ColumnTransformer:
    """Impute, scale and one-hot encode. Fitted inside each CV fold, so nothing leaks from validation data.

    `drop` removes columns entirely (used for external validation on a hospital that lacks them).
    Binary columns are imputed too: Cleveland has no gaps there, but other hospitals do.
    """
    numeric = [c for c in cfg.NUMERIC + cfg.ORDINAL + (DOMAIN_NUMERIC if use_fe else []) if c not in drop]
    nominal = [c for c in cfg.NOMINAL if c not in drop]
    binary = [c for c in cfg.BINARY + (DOMAIN_BINARY if use_fe else []) if c not in drop]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), nominal),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ], verbose_feature_names_out=False)
