"""Use the model on a patient: cut-offs, risk -> tier + action, and why a patient scored as they did.

This is the inference module shared by the notebook and the demo app. It imports only `config` from
the package, never training code (`models`, `evaluation`) or plotting.
"""
import numpy as np
import pandas as pd

from cardiotriage import config as cfg


def pick_thresholds(y, p, recall_target: float = cfg.RECALL_TARGET, ppv_target: float = cfg.PPV_TARGET) -> dict:
    """Choose the two tier cut-offs from out-of-fold probabilities.

    low:  the HIGHEST cut-off that still flags >= recall_target of patients with disease
          (Medium or High tier), so as few healthy patients as possible are flagged.
    high: the LOWEST cut-off >= low whose flagged group has PPV >= ppv_target; if none reaches it,
          the cut-off >= low with the best PPV.
    """
    y, p = np.asarray(y).astype(bool), np.asarray(p, float)
    cuts = np.unique(p)[::-1]                                   # high -> low
    recall = np.array([(p[y] >= t).mean() for t in cuts])
    low = float(cuts[np.argmax(recall >= recall_target)])        # first (highest) cut meeting the target
    above = cuts[cuts >= low]
    ppv = np.array([y[p >= t].mean() for t in above])
    ok = above[ppv >= ppv_target]
    high = float(ok.min()) if len(ok) else float(above[np.nanargmax(ppv)])
    return {"low": low, "high": high}


def triage(p: float, thresholds: dict) -> dict:
    """Risk -> tier and recommended action."""
    tier = "High" if p >= thresholds["high"] else "Medium" if p >= thresholds["low"] else "Low"
    return {"risk": float(p), "tier": tier, "action": cfg.TIER_ACTIONS[tier]}


def baseline(pipe, X: pd.DataFrame) -> np.ndarray:
    """Average training patient in the model's input space; the reference point for contributions."""
    return np.asarray(pipe[:-1].transform(X)).mean(axis=0)


def _source_feature(name: str) -> str:
    """Map a model input column (e.g. 'cp_4.0') back to the clinical feature it came from ('cp')."""
    if name in cfg.FEATURES:
        return name
    return next((f for f in cfg.FEATURES if name.startswith(f + "_")), name)


def contributions(pipe, patient: pd.DataFrame, base: np.ndarray) -> pd.Series:
    """Per-feature contribution to this patient's log-odds of disease, relative to the average patient.

    For a linear model these are exact SHAP values: coef * (x - average x), summed over the one-hot
    columns of each clinical feature. They add up to logit(patient) - logit(average patient).
    Positive = pushes risk up. Only defined for linear models.
    """
    model = pipe[-1]
    if not hasattr(model, "coef_"):
        raise ValueError(f"contributions need a linear model, got {type(model).__name__}")
    x = np.asarray(pipe[:-1].transform(patient))[0]
    names = pipe[:-1].get_feature_names_out()
    parts = pd.Series(model.coef_[0] * (x - base), index=[_source_feature(n) for n in names])
    out = parts.groupby(level=0).sum()
    return out.reindex(out.abs().sort_values(ascending=False).index)
