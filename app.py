"""Heart Disease Triage: pre-angiography demo using the reduced-input model.

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
    "Exercise stress test": ["thalach", "exang", "oldpeak"],
    "Retrospective comparison inputs (not used by the main score)": ["slope", "ca", "thal"],
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
if "no_imaging" not in artifact or "baseline" not in artifact["no_imaging"]:
    st.error("The saved model predates the pre-angiography demo. Re-run the notebook to rebuild artifacts.")
    st.stop()
presets = artifact["presets"]
main = artifact["no_imaging"]
thresholds = main["thresholds"]
active_features = [c for c in cfg.FEATURES if c not in main["dropped"]]

if "preset" not in st.session_state:                       # first visit: start from the first preset
    st.session_state["preset"] = next(iter(presets))
    load_preset()

# ---------------------------------------------------------------- sidebar: patient form
with st.sidebar:
    st.header("Inputs")
    st.selectbox("Load a demo patient", list(presets), key="preset", on_change=load_preset)
    st.caption(presets[st.session_state["preset"]]["description"] + ". Edit any value below to explore what-ifs.")
    st.toggle("Compare with 13-feature retrospective model", key="full_model")
    for section, cols in SECTIONS.items():
        if section.startswith("Retrospective") and not st.session_state["full_model"]:
            continue
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
st.caption("Pre-angiography prioritisation using 10 recorded features. The score excludes `ca`, `thal` and `slope`.")
st.warning("**Decision support, not diagnosis.** A proof of concept trained on 1980s UCI Cleveland data; "
           "the clinician makes the decision. Suspected acute coronary syndrome needs urgent assessment "
           "regardless of this score.", icon="⚠️")

patient = pd.DataFrame([{col: (np.nan if st.session_state[col] is None else float(st.session_state[col]))
                         for col in cfg.FEATURES}])
try:
    data.validate(patient)                                  # the same data contract the training data passed
except ValueError as err:
    st.error(f"This patient cannot be scored: {err}")
    st.stop()

pipe = main["pipeline"]
risk = float(pipe.predict_proba(patient[cfg.FEATURES])[0, 1])
result = triage.triage(risk, thresholds)
contrib = triage.contributions(pipe, patient[cfg.FEATURES], main["baseline"])

last = st.session_state.get("last")
if last is not None and not pd.Series(last["inputs"])[active_features].equals(patient.iloc[0][active_features]):
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
               f"High from {thresholds['high']:.1%}. These cut-offs target {cfg.RECALL_TARGET:.0%} recall "
               "on training predictions; outer-fold validation assesses the full rule separately.")
    if result["tier"] == "Low":
        st.info("**Low safety-net:** A clinician must review symptoms and tests before any deferral, "
                "record a follow-up date and who will contact the patient, and give clear instructions "
                "to seek urgent assessment for new or worsening chest pain or other concerning symptoms. "
                "A Low score does not rule out disease.")
    if st.session_state["full_model"]:
        full_risk = float(artifact["pipeline"].predict_proba(patient[cfg.FEATURES])[0, 1])
        full_tier = triage.triage(full_risk, artifact["thresholds"])["tier"]
        st.metric("Retrospective 13-feature risk", f"{full_risk:.1%}",
                  delta=f"{100 * (full_risk - risk):+.1f} pts vs pre-angiography model", delta_color="off")
        st.caption(f"{full_tier} tier on separate cut-offs. This comparison uses fluoroscopy (`ca`), "
                   "thallium imaging (`thal`) and stress ECG slope, so it is not the queue decision.")
    if previous is not None:
        changed = [c for c in active_features
                   if label(c, previous["inputs"][c]) != label(c, patient.iloc[0][c])]
        if len(changed) > 4:
            st.markdown(f"**A different patient:** {len(changed)} inputs changed since the previous result.")
        elif changed:
            st.markdown("**Changed since the previous result:** " + "; ".join(
                f"{cfg.DESCRIPTIONS[c]} (`{c}`): {label(c, previous['inputs'][c])} → {label(c, patient.iloc[0][c])}"
                for c in changed))
    if patient.iloc[0]["sex"] == 0:
        st.info("The main model's sex subgroup estimates are based on small historical groups. "
                "Review Low results carefully; notebook Section 13.3 reports counts and uncertainty.")

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
        f"- **Main model:** logistic regression on 10 recorded features; Cleveland cross-validated ROC-AUC "
        f"{m['no_imaging_cv_roc_auc']:.3f}; Hungary external ROC-AUC {m['hungary_roc_auc_10_features']:.3f} "
        f"and recall {m['hungary_recall_10_features']:.3f}, below the {cfg.RECALL_TARGET:.0%} target\n"
        f"- **Retrospective 13-feature model:** ROC-AUC {m['cv_roc_auc']['mean']:.3f} ± "
        f"{m['cv_roc_auc']['sd']:.3f}; its held-out test results do not describe the main model\n"
        f"- **Decision rule:** outer-fold recall {m['decision_cv']['recall']:.3f}, specificity "
        f"{m['decision_cv']['specificity']:.3f}, High-tier PPV {m['decision_cv']['high_ppv']:.3f}; "
        f"estimates remain uncertain in this small cohort\n"
        f"- **Explanations:** exact per-feature contributions for a linear model (SHAP values), relative to "
        f"the average training patient\n"
        f"- **Not for:** first-contact screening, the general population, or decisions without a clinician\n"
        f"- **Built with:** " + ", ".join(f"{k} {v}" for k, v in artifact["versions"].items()))
