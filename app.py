"""Heart Disease Triage: live demo of the angiography-queue triage model.

Run from the repo root:  .venv/Scripts/streamlit run app.py

Imports only the inference side of the package (config, data.validate, triage), never training code,
so the demo cannot drift from the model the notebook trained and saved.
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from cardiotriage import config as cfg, data, triage

st.set_page_config(page_title="Heart Disease Triage", layout="wide")

# What the form will physically accept. Deliberately wider than the clinical data contract
# (cfg.RANGES) so that implausible values reach data.validate() and are rejected there, visibly.
FORM_LIMITS = {"age": (1.0, 120.0, 1.0), "trestbps": (40.0, 300.0, 1.0), "chol": (50.0, 1000.0, 1.0),
               "thalach": (30.0, 260.0, 1.0), "oldpeak": (0.0, 10.0, 0.1)}
NOT_MEASURED = "not measured"
TIER_STYLE = {"High": ("#E45756", "🔴"), "Medium": ("#F2A541", "🟠"), "Low": ("#4C78A8", "🔵")}
SECTIONS = {
    "Patient": ["age", "sex", "trestbps", "chol", "fbs"],
    "Symptoms and resting ECG": ["cp", "restecg"],
    "Exercise stress test": ["thalach", "exang", "oldpeak", "slope"],
    "Imaging": ["ca", "thal"],
}


@st.cache_resource
def load_artifact():
    # Our own artifact, written by the notebook's last section. Never point this at an untrusted file:
    # unpickling can execute code.
    return joblib.load(cfg.MODEL_PATH)


def label(col, value) -> str:
    """Human-readable value for a feature, e.g. cp=4 -> 'asymptomatic'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NOT_MEASURED
    if col in cfg.LABELS:
        return cfg.LABELS[col][int(value)]
    return f"{value:g}"


def load_preset():
    """Copy the chosen preset into the form widgets (runs before the widgets are drawn)."""
    patient = artifact["presets"][st.session_state["preset"]]["patient"]
    for col, value in patient.items():
        if col in cfg.LABELS:
            st.session_state[col] = None if value is None or np.isnan(value) else int(value)
        else:
            st.session_state[col] = float(value)


if not cfg.MODEL_PATH.exists():
    st.error(f"Model file not found: {cfg.MODEL_PATH}. Run the notebook to the end first; its last section saves it.")
    st.stop()
artifact = load_artifact()
presets, thresholds = artifact["presets"], artifact["thresholds"]

if "preset" not in st.session_state:                       # first visit: start from the first preset
    st.session_state["preset"] = next(iter(presets))
    load_preset()

# ---------------------------------------------------------------- sidebar: patient form
with st.sidebar:
    st.header("Inputs")
    st.selectbox("Load a demo patient", list(presets), key="preset", on_change=load_preset)
    st.caption(presets[st.session_state["preset"]]["description"] + ". Edit any value below to explore what-ifs.")
    for section, cols in SECTIONS.items():
        st.subheader(section)
        for col in cols:
            if col in cfg.LABELS:
                options = list(cfg.LABELS[col]) + ([None] if col in cfg.ALLOW_MISSING else [])
                st.selectbox(cfg.DESCRIPTIONS[col], options, key=col, format_func=lambda v, c=col: label(c, v))
            else:
                lo, hi, step = FORM_LIMITS[col]
                st.number_input(cfg.DESCRIPTIONS[col], min_value=lo, max_value=hi, step=step, key=col,
                                format="%.1f" if step < 1 else "%.0f")

# ---------------------------------------------------------------- main: risk, tier, explanation
st.title("Heart Disease Triage")
st.caption("Prioritising the confirmatory angiography queue after the initial cardiac work-up.")
st.warning("**Decision support, not diagnosis.** A proof of concept trained on 1980s UCI Cleveland data; "
           "the clinician makes the decision.", icon="⚠️")

patient = pd.DataFrame([{col: (np.nan if st.session_state[col] is None else float(st.session_state[col]))
                         for col in cfg.FEATURES}])
try:
    data.validate(patient)                                  # the same data contract the training data passed
except ValueError as err:
    st.error(f"This patient cannot be scored: {err}")
    st.stop()

pipe = artifact["pipeline"]
risk = float(pipe.predict_proba(patient[cfg.FEATURES])[0, 1])
result = triage.triage(risk, thresholds)
contrib = triage.contributions(pipe, patient[cfg.FEATURES], artifact["baseline"])

last = st.session_state.get("last")
if last is not None and not pd.Series(last["inputs"]).equals(patient.iloc[0]):   # NaN-safe compare
    st.session_state["previous"] = last          # only a real change of inputs moves the "vs previous" reference
previous = st.session_state.get("previous")
st.session_state["last"] = {"risk": risk, "tier": result["tier"], "inputs": patient.iloc[0].to_dict()}

color, icon = TIER_STYLE[result["tier"]]
left, right = st.columns([1, 2])
with left:
    delta = None if previous is None else f"{100 * (risk - previous['risk']):+.1f} pts vs previous"
    st.metric("Estimated risk of significant coronary disease", f"{risk:.1%}", delta=delta, delta_color="inverse")
    st.markdown(f"<div style='background:{color};color:white;padding:0.8rem 1rem;border-radius:0.5rem;"
                f"font-size:1.3rem'>{icon} <b>{result['tier']} tier</b><br>{result['action']}</div>",
                unsafe_allow_html=True)
    st.caption(f"Tiers: Low below {thresholds['low']:.1%} · Medium from {thresholds['low']:.1%} · "
               f"High from {thresholds['high']:.1%}. The low cut-off was set to catch at least "
               f"{cfg.RECALL_TARGET:.0%} of sick patients in validation.")
    if previous is not None:
        changed = [c for c in cfg.FEATURES
                   if label(c, previous["inputs"][c]) != label(c, patient.iloc[0][c])]
        if len(changed) > 4:
            st.markdown(f"**A different patient:** {len(changed)} inputs changed since the previous result.")
        elif changed:
            st.markdown("**Changed since the previous result:** " + "; ".join(
                f"{cfg.DESCRIPTIONS[c]} (`{c}`): {label(c, previous['inputs'][c])} → {label(c, patient.iloc[0][c])}"
                for c in changed))
    if patient.iloc[0]["sex"] == 0:
        st.info("In validation the model caught a smaller share of sick women than of sick men. "
                "Treat a Low result for a woman with extra caution.")

with right:
    st.subheader("Why this risk?")
    chart = pd.DataFrame({
        "factor": [f"{cfg.DESCRIPTIONS[c]}: {label(c, patient.iloc[0][c])}" for c in contrib.index],
        "contribution": contrib.to_numpy(),
    })
    chart["direction"] = np.where(chart["contribution"] >= 0, "raises risk", "lowers risk")
    st.altair_chart(
        alt.Chart(chart).mark_bar().encode(
            x=alt.X("contribution:Q", title="Push on log-odds vs the average training patient"),
            # explicit order: with sort=None Vega keeps the previous chart's order for repeated labels
            y=alt.Y("factor:N", sort=list(chart["factor"]), title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                           range=["#E45756", "#4C78A8"]), legend=None),
            tooltip=["factor", alt.Tooltip("contribution:Q", format="+.2f")],
        ).properties(height=380),
        width="stretch",
    )
    top = contrib.head(3)
    st.markdown("**Strongest factors:** " + "; ".join(
        f"{cfg.DESCRIPTIONS[c].lower()} ({label(c, patient.iloc[0][c])}) {'raises' if v > 0 else 'lowers'} the risk"
        for c, v in top.items()))

with st.expander("About this model"):
    m = artifact["metrics"]
    st.markdown(
        f"- **Model:** {m['selected_model']} (C = {m['C']:.3f}) on the 13 UCI clinical features\n"
        f"- **Cross-validated ROC-AUC:** {m['cv_roc_auc']['mean']:.3f} ± {m['cv_roc_auc']['sd']:.3f}\n"
        f"- **Held-out test set:** ROC-AUC {m['test']['roc_auc']['point']:.3f}; recall "
        f"{m['test']['recall']['point']:.3f} (exact 95% CI {m['test_recall_exact_ci'][0]:.3f} to "
        f"{m['test_recall_exact_ci'][1]:.3f}); specificity {m['test']['specificity']['point']:.3f}\n"
        f"- **Explanations:** exact per-feature contributions for a linear model (SHAP values), relative to "
        f"the average training patient\n"
        f"- **Not for:** first-contact screening, the general population, or decisions without a clinician\n"
        f"- **Built with:** " + ", ".join(f"{k} {v}" for k, v in artifact["versions"].items()))
