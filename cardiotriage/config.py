"""Constants only. Imports nothing from the package, so every module can depend on it."""
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEVELAND = RAW / "processed.cleveland.data"      # UCI original: the source of truth
HUNGARIAN = RAW / "processed.hungarian.data"      # UCI, external validation only
KAGGLE = RAW / "heart-disease.csv"                # re-coded, label-flipped copy: QA evidence only
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "heart_disease_pipeline.joblib"

# ---- reproducibility and targets ----
SEED = 42
RECALL_TARGET = 0.90                              # recall on the disease class (SPEC §0.5)

# ---- columns (UCI order); `num` is angiographic status 0-4, `disease` = num > 0 ----
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + ORDINAL + NOMINAL + BINARY   # the 13 model inputs
TARGET = "disease"
ALLOW_MISSING = ["ca", "thal"]                    # the only columns UCI leaves blank ('?')

# ---- plausible physiological ranges (inclusive); the Cleveland data sits well inside ----
RANGES = {
    "age": (18, 100),
    "trestbps": (60, 260),     # resting blood pressure, mm Hg
    "chol": (80, 700),         # serum cholesterol, mg/dl
    "thalach": (50, 250),      # max heart rate achieved, bpm
    "oldpeak": (0.0, 10.0),    # ST depression
}

# ---- official UCI code meanings (also the allowed values for validate) ----
LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "fbs": {0: "fasting glucose <= 120 mg/dl", 1: "fasting glucose > 120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    "exang": {0: "no exercise angina", 1: "exercise angina"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}

# ---- how the Kaggle copy re-codes UCI (kaggle code -> UCI code); proven row-by-row in data.kaggle_vs_uci ----
NAN = float("nan")
KAGGLE_TO_UCI = {
    "cp": {0: 4, 1: 2, 2: 3, 3: 1},
    "restecg": {0: 2, 1: 0, 2: 1},
    "slope": {0: 3, 1: 2, 2: 1},
    "thal": {0: NAN, 1: 6, 2: 3, 3: 7},
    "ca": {0: 0, 1: 1, 2: 2, 3: 3, 4: NAN},
}
KAGGLE_MATCH_KEYS = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]
