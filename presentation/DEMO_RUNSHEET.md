# Demo run-sheet — 1:35, recorded live

Speaker **M6**, whose 2:10 block is this 1:35 demo plus the 0:35 close. Screen capture only, no slides behind it. Figures below are the live values from the saved artifact
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
| 0:00–0:15 | 1 | Preset **Clear high risk** (loads on open); point at the banner | Main risk **97.5%** → **High: Priority angiography** | "Ten inputs the clinician already has. This patient: 97.5 percent, High, priority angiography. The panel shows *why*." |
| 0:15–0:27 | 2 | Preset **Clear low risk** | **2.1%** → **Low**, safety-net box | "Two percent, Low. Low is not a discharge: review, follow-up date, named contact." |
| 0:27–0:39 | 3 | Preset **Borderline** | **30.3%** → **Medium** | "Just above the 26.3 percent boundary: Medium. The model won't pretend this one is clear." |
| 0:39–0:54 | 4 | On Borderline set **max heart rate** to **100 bpm** | **73.3%** → **High**; changed input named | "One measurement, peak heart rate to 100, and Medium becomes High. A what-if, not an outcome." |
| 0:54–1:12 | 5 | Reload **Borderline**, tick **Compare with 13-feature**, set `ca` to **2** | Main stays **30.3% Medium**; retrospective **13.2% → 51.6%** | "I've said two vessels are narrowed, and the queue decision doesn't move, because that finding may come from the angiogram itself." |
| 1:12–1:35 | 6 | Preset **Missed by the model**, then **Age** = **10** | **5.5% Low** plus women caution; age 10 → contract error, **no score** | "A real miss: disease, scored Low. Hungary recall was 0.887, and recall for women 0.833, so the app says so. And age ten: the contract refuses to score it." |

**Hand back:** "Decision support, not diagnosis — and suspected acute coronary syndrome goes on the urgent pathway
whatever this screen says." → slide 29 (the close).

---

## If something breaks on the day

| Failure | What you see | Do this |
|---|---|---|
| Model file missing | "Model file not found" | Re-run the notebook; it writes `artifacts/` in its last section |
| Stale artifact | "The saved model predates the pre-angiography demo" | Same fix — re-run the notebook end to end |
| Numbers differ from this sheet | Any mismatch | Use the fallback recorded take; update this sheet before the next recording |
| Streamlit won't start | Port in use | `streamlit run app.py --server.port 8502` |
