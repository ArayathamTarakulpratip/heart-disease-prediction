# Demo run-sheet: pre-angiography triage

These figures come from the rebuilt artifact and a live Streamlit run on 2026-11-07. Recheck them after retraining. Presets 1–3 are from the 61-patient Cleveland test set; preset 4 is a diseased **training** patient selected because reduced-input out-of-fold scoring placed them Low. Its displayed score comes from the final fitted model, which has seen that patient.

**Start:** `.venv/Scripts/streamlit run app.py` from the repo root. The first preset opens automatically. Main-model cut-offs: Low below **26.3%**, Medium from **26.3%**, High from **66.2%**. The main form asks for only ten inputs. Opening the retrospective comparison reveals `ca`, `thal` and `slope`; the main score excludes them. `ca` may be measured during the target angiogram, so the 13-feature score is only a retrospective comparison. `slope` is exercise ECG, excluded for cross-hospital completeness.

| Step | Do this | Expected live result | What to say |
|---|---|---|---|
| 1 | Start on **Clear high risk** | Main risk **97.5%**, **High: Priority angiography** | "History and the recorded exercise-test measures put this patient in High. A clinician decides priority." |
| 2 | Choose **Clear low risk** | Main risk **2.1%**, **Low: Clinician review and a documented safety-net plan**. The app specifies a follow-up date, responsible contact and urgent return instructions. | "Low does not rule out disease or authorize discharge. A clinician reviews the case before any deferral." |
| 3 | Choose **Borderline** | Main risk **30.3%**, **Medium: Cardiology review or further testing** | "This test patient is just above the 26.3% Low boundary." |
| 4 | With **Borderline** selected, change maximum exercise heart rate from its preset value to **100 bpm** | Main risk **73.3%**, **High**; the changed-input list names heart rate | "A pre-angiography measurement changes the main tier. This is a what-if, not a measured patient outcome." |
| 5 | Reload **Borderline**, enable **Compare with 13-feature retrospective model**, then change `ca` from its preset value to **2 vessels** | The main risk stays **30.3% Medium**. The retrospective score changes from **13.2%** to **51.6%**. | "The main queue decision ignores `ca`. The optional comparison moves because it uses fluoroscopy; it cannot be the pre-angiography decision if that finding comes from the target procedure." |
| 6 | Choose **Missed by the model**, then set **Age** to **10** | The diseased training example shows **5.5% Low** with the safety plan. Age 10 produces a data-contract error and no score. | "A real miss remains possible. Cleveland separate-fold recall was 0.901, and Hungary recall fell to 0.887, below our target. The data contract also rejects implausible input." |

The retrospective 13-feature queue and Medicare figures in the notebook are **simulations**. The dataset records no arrival sequence, waiting time or actual decision to withhold angiography. The payment difference assumes Low-tier patients receive no angiography; it is not a measured saving. The retrospective simulation leaves **45.5 patients with disease per 1,000 referred** in Low.

**Always say:** decision support, not diagnosis. Suspected acute coronary syndrome requires urgent assessment regardless of score. The clinician makes the call.
