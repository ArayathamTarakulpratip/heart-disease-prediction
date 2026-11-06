# Demo run-sheet: Heart Disease Triage app

A 3-minute live walkthrough of `app.py` on patients the model never trained on. Every figure below was produced by driving the real app with Streamlit's `AppTest` (2026-11-06); re-check them after re-running the notebook, because a re-trained model can move them.

**Start:** `.venv/Scripts/streamlit run app.py` from the repo root. The app opens on the first preset.

**Tier cut-offs:** Low below 28.7% · Medium from 28.7% · High from 50.9% (chosen on out-of-fold training predictions so that at least 90% of sick patients are flagged).

| Step | Do this | You should see | Say this |
|---|---|---|---|
| 1 | Preset **Clear high risk** (test patient: man, 58, asymptomatic chest pain, 3 vessels on fluoroscopy, reversible thallium defect, exercise angina, ST depression 2.2 mm) | **99.3%**, red **High tier: Priority angiography**. Strongest factors: 3 vessels, reversible defect, asymptomatic chest pain | "Every signal points the same way; this patient goes to the front of the angiography queue." |
| 2 | Preset **Clear low risk** (test patient: woman, 63, non-anginal pain, 0 vessels, normal thallium, no exercise angina) | **4.0%**, blue **Low tier: routine follow-up with safety-netting advice**; delta −95.2 pts; a blue note that the model caught fewer sick women in validation | "Low risk, but not discharged without a plan. The app also warns that recall was lower for women, a known gap we report rather than hide." |
| 3 | Preset **Borderline** (test patient: man, 60, non-anginal pain, 0 vessels, normal thallium, ST depression 3.0 mm) | **29.6%**, orange **Medium tier: cardiology review or further testing**. Normal imaging lowers the risk; marked ST depression pushes it just over the 28.7% cut-off | "This is the close call: the patient lands just above the low cut-off, so a cardiologist reviews them." |
| 4 | **What-if:** keep Borderline, change *Major vessels coloured by fluoroscopy* from 0 to **2** | **74.7%**, **High tier**; delta **+45.1 pts**; the app lists exactly what changed (`ca`: 0 vessels → 2 vessels) | "One imaging finding moves this patient from review to priority angiography, and the explanation shows why." |
| 5 | Preset **Missed by the model** (training patient, scored out-of-fold: man, 47, non-anginal pain, 0 vessels, normal thallium, no exercise angina, no ST depression) | **6.3%**, **Low tier**, although this patient **has** disease | "No model is perfect. Every recorded signal looks reassuring here, which is why the Low tier carries safety-netting advice and why the factors are shown: a clinician can see what the score rests on." |
| 6 | **Circularity check:** re-select preset **Borderline**, then switch on *Circularity check: also score without imaging* | A second metric: **57.0%** without imaging, **+27.3 pts vs full model**, *Medium tier on its own cut-offs*; caption: ROC-AUC **0.846** without imaging vs **0.900** with it. (Clear high risk: 99.3% → 89.6%) | "Is the model just reading the angiogram? No. Without any imaging it still ranks patients with ROC-AUC 0.846, and 0.880 at a hospital in Budapest it never saw. Here, history and the stress test alone say Medium; the normal imaging is what pulls this patient down. The angiogram still makes the diagnosis; we only order the queue." |
| 7 | Set **Age** to **10** | Red error: *This patient cannot be scored: data contract violated: age: 1 value(s) outside [18, 100], e.g. 10.0*; no risk is shown | "The same data contract that checked the training data guards the app: impossible inputs are refused, not scored." |

**Why these patients are fair to show.** Presets 1–3 come from the locked test set, which the model never saw during training, tuning or cut-off selection. No test patient with disease was missed, so preset 5 is the sick training patient with the lowest out-of-fold risk (scored by models that never saw them). Its live risk (6.3%) comes from the final model, which did train on this patient; the out-of-fold risk was 6.1% (notebook Section 12.3).

**Always say:** decision support, not diagnosis. The clinician makes the call.
