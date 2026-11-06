"""Shared test fixtures used by more than one test file."""
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def fitted_lr():
    """A Logistic Regression pipeline fitted on all of Cleveland, plus its feature frame."""
    from sklearn.linear_model import LogisticRegression
    from cardiotriage import config as cfg, data, models
    df = data.load_cleveland()
    pipe = models.build_pipeline(LogisticRegression(max_iter=1000)).fit(df[cfg.FEATURES], df[cfg.TARGET])
    return pipe, df[cfg.FEATURES]


def cleveland_target():
    from cardiotriage import config as cfg, data
    return data.load_cleveland()[cfg.TARGET]
