# Demo run-sheet — 2:30, recorded live

Speaker **M6**. Screen capture only, no slides behind it. Figures below are the live values from the saved artifact
(`artifacts/heart_disease_pipeline.joblib`, verified 2026-11-12) and match [`../DEMO.md`](../DEMO.md).
**Re-verify them after any retraining — they move.**

**Pre-flight (do this before you hit record)**

1. `jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb` — rebuilds the artifact.
2. `python -m tests.check_notebook` → must print `clean`; `python -m tests.smoke_test` → all pass.
3. `streamlit run app.py`, browser at 100% zoom, window 1600×900, sidebar open. The first preset loads itself.
4. Confirm the caption reads **Low below 26.3% · Medium from 26.3% · High from 66.2%**. If it doesn't, the artifact is stale.
5. Record **two takes**. Keep the better one; keep the other as the fallback cut.

> **Note on the drafted run-sheet.** The pasted version used 99.3% / 4.1% / 29.6% / 6.3% and a `ca` 0→2 what-if.
> Those are the **retrospective 13-feature** model's numbers from before the pre-angiography pivot. The what-if below
> therefore moves **maximum heart rate**, because that is an input the shipped model actually uses; `ca` appears
> afterwards precisely to show the main score *not* moving.

---

| Time | Step | Action on screen | Live result | Say this |
|---|---|---|---|---|
| 0:00–0:15 | 0 | Point at the banner and the sidebar | "Decision support, not diagnosis" banner; ten inputs | "Ten inputs — history, resting ECG, and the recorded exercise-test measures. No fluoroscopy, no thallium scan. These are what the clinician already has." |
| 0:15–0:40 | 1 | Preset **Clear high risk** (loads on open) | Main risk **97.5%** → **High: Priority angiography** | "Our highest-risk test patient. Ninety-seven and a half percent, High tier: priority angiography. The panel on the right shows which inputs drove it — the clinician can see *why*, not just *what*." |
| 0:40–1:00 | 2 | Preset **Clear low risk** | **2.1%** → **Low**, safety-net box appears | "Two percent, Low. And watch what Low actually says: clinician review, a documented follow-up date, a named contact, and instructions to come back urgently if the chest pain changes. Low is not a discharge." |
| 1:00–1:20 | 3 | Preset **Borderline** | **30.3%** → **Medium: cardiology review or further testing** | "Thirty point three — just above the 26.3% Low boundary, so: Medium. Cardiology review or further testing. The model refuses to pretend this one is clear." |
| 1:20–1:45 | 4 | Still on Borderline: set **maximum exercise heart rate** to **100 bpm** | **73.3%** → **High**; the "changed since the previous result" line names heart rate | "One pre-angiography measurement — peak heart rate down to 100 — and the same patient moves from Medium to High. That is a what-if, not a measured outcome for this patient." |
| 1:45–2:05 | 5 | Reload **Borderline**, tick **Compare with 13-feature retrospective model**, set `ca` to **2 vessels** | Main risk stays **30.3% Medium**; retrospective score moves **13.2% → 51.6%** | "Here is the discipline. I have just told it two vessels are narrowed on fluoroscopy, and the queue decision does not move at all — because that finding may come from the very angiogram we are scheduling. Only the retrospective comparison moves." |
| 2:05–2:30 | 6 | Preset **Missed by the model**, then set **Age** to **10** | **5.5% Low** with the safety plan and the women-subgroup caution; age 10 → data-contract error, **no score at all** | "This patient has disease and our model puts them in Low at five and a half percent. A real miss. Separate-fold recall was 0.901 and at the second hospital it fell to 0.887, below our target — and recall for women is 0.833, which is why the app says so on screen. Last thing: age ten. The data contract refuses to score it. The model does not get to guess." |

**Hand back:** "Decision support, not diagnosis — and suspected acute coronary syndrome goes on the urgent pathway
whatever this screen says." → M3.

---

## If something breaks on the day

| Failure | What you see | Do this |
|---|---|---|
| Model file missing | "Model file not found" | Re-run the notebook; it writes `artifacts/` in its last section |
| Stale artifact | "The saved model predates the pre-angiography demo" | Same fix — re-run the notebook end to end |
| Numbers differ from this sheet | Any mismatch | Use the fallback recorded take; update this sheet before the next recording |
| Streamlit won't start | Port in use | `streamlit run app.py --server.port 8502` |
