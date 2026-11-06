"""The saved artifact and the Streamlit app (needs the notebook to have been run)."""
import subprocess
import sys

from tests.helpers import APP


def test_saved_artifact_scores_presets_with_expected_tiers_in_fresh_process():
    from cardiotriage import config as cfg
    assert cfg.MODEL_PATH.exists(), f"{cfg.MODEL_PATH} missing: run the notebook first"
    code = (
        "import sys, joblib, pandas as pd; from cardiotriage import config as cfg, data, triage;"
        "a = joblib.load(cfg.MODEL_PATH);"     # our own artifact, written by the notebook; never load untrusted files
        "assert {'pipeline','thresholds','baseline','features','metrics','versions','presets','no_imaging'} <= set(a), set(a);"
        "bad = [];"
        "[bad.append(n) for n, pr in a['presets'].items()"
        " if triage.triage(a['pipeline'].predict_proba(pd.DataFrame([pr['patient']])[cfg.FEATURES])[0, 1], a['thresholds'])['tier'] != pr['expected_tier']];"
        "loaded = [m for m in sys.modules if m in ('cardiotriage.evaluation','cardiotriage.models','cardiotriage.plots')];"
        "print(len(a['presets']), bad, loaded); sys.exit(1 if bad or loaded or len(a['presets']) != 4 else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr[-500:]


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_presets_land_in_their_expected_tiers():
    import joblib
    from cardiotriage import config as cfg
    presets = joblib.load(cfg.MODEL_PATH)["presets"]       # our own artifact, written by the notebook
    at = _app().run()
    assert not at.exception, at.exception
    for name, pr in presets.items():
        at.selectbox(key="preset").set_value(name).run()
        assert not at.exception and not at.error, (name, at.exception, [e.value for e in at.error])
        assert at.session_state["last"]["tier"] == pr["expected_tier"], (name, at.session_state["last"])
        assert abs(at.session_state["last"]["risk"] - pr["risk"]) < 1e-3, (name, at.session_state["last"])


def test_no_imaging_model_is_blind_to_imaging_features():
    import joblib
    import pandas as pd
    from cardiotriage import config as cfg
    a = joblib.load(cfg.MODEL_PATH)                           # our own artifact, written by the notebook
    patient = pd.DataFrame([a["presets"]["Clear high risk"]["patient"]])[cfg.FEATURES]
    blinded = patient.assign(ca=0.0, thal=3.0, slope=1.0)    # healthiest imaging values
    pipe = a["no_imaging"]["pipeline"]
    assert pipe.predict_proba(patient)[0, 1] == pipe.predict_proba(blinded)[0, 1]
    assert a["pipeline"].predict_proba(patient)[0, 1] != a["pipeline"].predict_proba(blinded)[0, 1]


def test_app_circularity_check_shows_the_no_imaging_risk():
    import joblib
    from cardiotriage import config as cfg
    pr = joblib.load(cfg.MODEL_PATH)["presets"]["Clear high risk"]
    at = _app().run()
    at.selectbox(key="preset").set_value("Clear high risk").run()
    assert len(at.metric) == 1, "check must be off by default"
    at.toggle(key="no_imaging").set_value(True).run()
    assert not at.exception and len(at.metric) == 2, at.exception
    assert at.metric[1].value == f"{pr['risk_no_imaging']:.1%}", (at.metric[1].value, pr["risk_no_imaging"])
    assert abs(at.session_state["last"]["risk"] - pr["risk"]) < 1e-3, "the main risk must not change"


def test_app_rejects_a_value_outside_the_data_contract():
    at = _app().run()
    at.number_input(key="age").set_value(10.0).run()          # the form allows it; the contract (18-100) must not
    assert not at.exception, at.exception
    assert any("age" in e.value for e in at.error), [e.value for e in at.error]


def test_app_shows_what_if_delta_after_an_edit():
    at = _app().run()
    at.selectbox(key="preset").set_value("Borderline").run()
    before = at.session_state["last"]["risk"]
    at.number_input(key="thalach").set_value(100.0).run()   # a much lower peak heart rate should raise risk
    after = at.session_state["last"]["risk"]
    assert after > before, (before, after)
    assert at.metric[0].delta not in (None, ""), at.metric[0].delta
    assert any("thalach" in m.value or "heart rate" in m.value for m in at.markdown), "changed input not listed"


def test_app_what_if_delta_survives_a_rerun_without_changes():
    # "vs previous" means the previous *different* inputs, not the previous rerun
    at = _app().run()
    at.selectbox(key="preset").set_value("Borderline").run()
    at.number_input(key="thalach").set_value(100.0).run()
    delta = at.metric[0].delta
    at.run()                                                  # e.g. a rerun triggered by something other than the form
    assert at.metric[0].delta == delta, (delta, at.metric[0].delta)


def test_app_renders_without_deprecation_notices():
    code = ("from streamlit.testing.v1 import AppTest;"
            f"at = AppTest.from_file(r'{APP}', default_timeout=60).run();"
            "[at.selectbox(key='preset').set_value(n).run() for n in at.selectbox(key='preset').options]")
    r = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code], capture_output=True, text=True)
    noise = (r.stdout + r.stderr).lower()
    assert r.returncode == 0, r.stderr[-500:]
    assert "deprecat" not in noise and "will be removed" not in noise, noise[-500:]
