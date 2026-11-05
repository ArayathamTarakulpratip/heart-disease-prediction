"""Load, clean and check the data. This is the data contract the notebook, the app and the tests share."""
import pandas as pd

from cardiotriage import config as cfg


def _load_uci(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=cfg.COLUMNS, na_values="?")
    df[cfg.TARGET] = (df["num"] > 0).astype(int)      # any vessel narrowing >50% counts as disease
    return df


def load_cleveland() -> pd.DataFrame:
    """UCI Cleveland (303 patients). '?' becomes NaN (only `ca` and `thal` have any)."""
    return _load_uci(cfg.CLEVELAND)


def load_hungarian() -> pd.DataFrame:
    """UCI Hungary (294 patients), used for external validation only."""
    return _load_uci(cfg.HUNGARIAN)


def load_kaggle() -> pd.DataFrame:
    """The Kaggle copy the original notebook used. Re-coded and label-flipped: never train on it."""
    return pd.read_csv(cfg.KAGGLE)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows so one patient cannot sit in both train and test."""
    return df.drop_duplicates().reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` breaks the data contract; return None when it is fine.

    Rules: all 13 feature columns present; numeric values inside cfg.RANGES; categorical values
    inside cfg.LABELS; NaN allowed only in `ca` and `thal` (the pipeline imputes those).
    """
    missing = [c for c in cfg.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    problems = []
    for col in cfg.FEATURES:
        s = df[col]
        nan = s.isna()
        if nan.any() and col not in cfg.ALLOW_MISSING:
            problems.append(f"{col}: {int(nan.sum())} missing value(s) (only {cfg.ALLOW_MISSING} may be missing)")
        present = s[~nan]
        if col in cfg.RANGES:
            lo, hi = cfg.RANGES[col]
            bad = present[(present < lo) | (present > hi)]
            if len(bad):
                problems.append(f"{col}: {len(bad)} value(s) outside [{lo}, {hi}], e.g. {bad.iloc[0]}")
        else:
            bad = present[~present.isin(cfg.LABELS[col].keys())]
            if len(bad):
                problems.append(f"{col}: {len(bad)} invalid code(s), e.g. {bad.iloc[0]}; allowed {sorted(cfg.LABELS[col])}")
    if problems:
        raise ValueError("data contract violated: " + "; ".join(problems))


def kaggle_vs_uci() -> dict:
    """Prove the Kaggle copy is label-flipped and re-coded, by matching every row back to UCI.

    Returns n_kaggle, n_matched, crosstabs (Kaggle value vs UCI value, per column) and mismatches
    (rows where the documented mapping in cfg.KAGGLE_TO_UCI does not give the UCI value; all 0).
    """
    k, u = load_kaggle(), load_cleveland()
    m = k.merge(u, on=cfg.KAGGLE_MATCH_KEYS, suffixes=("_kaggle", "_uci"))

    crosstabs = {"target": pd.crosstab(m["target"], m[cfg.TARGET])}
    mismatches = {"target": int(((1 - m["target"]) != m[cfg.TARGET]).sum())}   # Kaggle target=1 means NO disease
    for col, mapping in cfg.KAGGLE_TO_UCI.items():
        kag, uci = m[f"{col}_kaggle"], m[f"{col}_uci"]
        crosstabs[col] = pd.crosstab(kag, uci.fillna(-1))                         # -1 stands for '?'
        mapped = kag.map(mapping)
        mismatches[col] = int((~((mapped == uci) | (mapped.isna() & uci.isna()))).sum())
    return {"n_kaggle": len(k), "n_matched": len(m), "crosstabs": crosstabs, "mismatches": mismatches}
