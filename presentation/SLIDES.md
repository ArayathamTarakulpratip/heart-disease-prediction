# Deck spec: 17 slides (revision 2)

A standard ML-project talk, in rubric order: problem → data → EDA → models → results → improvement → demo → conclusion.
No Kaggle-versus-UCI comparison, no cost or money figures, no country statistics, no queue simulation.
Every figure already exists in `heart_disease_pitch_gold_standard.ipynb` or `artifacts/metrics.json`.

**Time:** 13:00 = six speakers × 2:10. The live demo (slide 15) runs on screen capture and is 1:35 of M6's block.

## Part A — style (set once)

* **Canvas** 16:9, 1920×1080, white background, 96 px margins.
* **Layout** two layouts only: *chart left (7 cols) + 3 short bullets right (5 cols)*, or *full-width table*.
  No hero-number slides, no stat cards.
* **Colour** text `#111827`, muted `#6B7280` for sources, one accent `#B91C1C` for the single number or bar being
  discussed, `#E5E7EB` for everything else. Never more than one accent per slide.
* **Type** one sans-serif family. Title 40 pt bold (a plain topic or a short claim), bullets 26 pt, tables 22 pt,
  source line 15 pt.
* **Charts** regenerate with `cardiotriage.plots.set_style()`; label axes in words; no chart title (the slide title is
  the title); highlight one bar or line.
* **Footer** slide number `n / 17` on the right, speaker tag (M1–M6) bottom-left. Lower-third with name on each
  speaker's first slide, 3 s.
* **Rules** no code except in the demo; at most 3 bullets, one line each; one idea per slide; cite source and year for
  any external statistic (there are only two).

## Part B — speaker blocks

| Speaker | Slides | Time | Seconds per slide |
|---|---|---|---|
| M1 | 1–3 | 0:00–2:10 | 5 · 70 · 55 |
| M2 | 4–6 | 2:10–4:20 | 45 · 40 · 45 |
| M3 | 7–9 | 4:20–6:30 | 50 · 35 · 45 |
| M4 | 10–12 | 6:30–8:40 | 40 · 45 · 45 |
| M5 | 13–14 | 8:40–10:50 | 70 · 60 |
| M6 | 15–17 | 10:50–13:00 | 95 (demo) · 25 · 10 |

### M1 — Problem and data

| # | Title | Visual | Content |
|---|---|---|---|
| 1 | Heart Disease Triage: Prioritising Angiography with Machine Learning | Title, 6 names, course, date | Nothing else |
| 2 | Problem and objective | Left: decision-flow diagram `10 inputs → model → risk → tier → action`. Right: 3 bullets | • Ontario: **1.4%** of 8,030 patients waiting for catheterisation died or had a heart attack or heart failure first (Natarajan, *CMAJ* 2002) · • Only **37.6%** of 398,978 elective angiograms found obstructive disease (Patel, *NEJM* 2010) · • Goal: flag **≥ 90%** of patients who truly have disease. Recall is primary: missing a sick patient is the costly error |
| 3 | The dataset: UCI Heart Disease, Cleveland | Left: simplified data dictionary (13 features grouped 5 numeric / 1 ordinal / 4 categorical / 3 binary). Right: 3 bullets | • **303 patients, 13 attributes** (Janosi et al., 1989) · • Target `disease` = 1 when angiography shows > 50% narrowing: 139 sick, 164 healthy · • Binary classification, one row per patient |

### M2 — Data quality and EDA

| # | Title | Visual | Content |
|---|---|---|---|
| 4 | Data quality: checked before use | Left: `class_balance` chart. Right: valid-range table (age 18–100, BP 60–260, chol 80–700, HR 50–250, ST 0–10) | • All 303 patients pass the range checks, set wide on purpose to catch only impossible values · • 6 missing cells of 3,939 (`ca`, `thal`), imputed inside the pipeline · • 0 duplicates; **45.9%** with disease, so no resampling |
| 5 | EDA: sex and chest-pain type | `rate_by` for `sex` and `cp`, asymptomatic bar in accent | • Men **55.3%** vs women **25.8%** (only 97 women) · • Asymptomatic patients: **72.9%** with disease, the highest group, because all were already referred · • So chest pain is one-hot encoded |
| 6 | EDA: exercise test and imaging | `rate_by` for `exang` and `ca`, `ca` gradient in accent | • Exercise angina **76.8%** vs 30.9% · · Vessels `ca` 0 → 3: **26.1% → 85.0%** · • Blood pressure and cholesterol barely differ (median BP 130 in both groups) |

### M3 — Input timing, validation design, models

| # | Title | Visual | Content |
|---|---|---|---|
| 7 | Dropping `ca`, `thal` and `slope` | Left: two-row table 13 vs 10 features (CV ROC-AUC 0.900 vs 0.844). Right: 3 bullets | • `ca` is a fluoroscopy finding that may be read during the very angiogram we are scheduling · • UCI has no timestamps, so we cannot prove it is available earlier · • The shipped model uses the **10 features** available beforehand |
| 8 | Split and validation design | Diagram: 242 train / 61 locked test; 5-fold × 10 repeats | • Stratified 80/20 split; test set opened **once** · • **50** cross-validation folds, identical for every model · • All preprocessing inside the pipeline, so nothing leaks |
| 9 | Five models | 5-row table: model · idea in one line · role | Dummy (floor) · Logistic regression (weighted sum + sigmoid) · KNN (vote of nearest patients) · Random forest (averaged trees, bagging) · XGBoost (trees correcting errors, boosting). Footer: no deep learning with 303 rows |

### M4 — Comparison, tuning, results

| # | Title | Visual | Content |
|---|---|---|---|
| 10 | Model comparison | `cv_boxplot` of ROC-AUC, logistic regression in accent | • Tuned ROC-AUC: LR **0.900**, XGBoost 0.897, KNN 0.896, RF 0.894, Dummy 0.500 · • LR vs XGBoost: corrected t-test **p = 0.81**, no real difference · • We choose logistic regression: interpretable and calibrated |
| 11 | Tuning and the decision threshold | Left: before/after table. Right: `confusion_grid` | • Randomised search, 30 settings, inside CV: KNN +0.030, XGBoost +0.034, LR +0.007 · • At the same 90% recall, LR spares **89 of 131** healthy patients, XGBoost 88, KNN 79, RF 74 · • Cut-offs Low < 26.3%, High ≥ 66.2% |
| 12 | Results | Table of three checks | • Separate-fold decision check: recall **0.901**, specificity 0.542, High-tier PPV **0.857** · • Locked test set: **28 of 28** sick patients flagged (95% CI 0.877–1.000) · • External hospital (Budapest, 294 patients): ROC-AUC **0.877**, recall 0.887, below target |

### M5 — Improvement and limitations

| # | Title | Visual | Content |
|---|---|---|---|
| 13 | Improving the model | Left: `learning_curve`. Right: 3 bullets | • Nested CV: **0.909** vs 0.900 tuned, so no tuning optimism · • Learning curve still rising: **more data** helps; a 100-setting search added nothing · • **6 sick patients** are missed by all four models, so they need new inputs (e.g. troponin), not a new algorithm |
| 14 | Limitations | 4 short lines | • Recall for women **0.833** vs men 0.914 (18 sick women) · • One hospital, 1980s, already-referred patients · • External recall 0.887 is below our 90% target · • `ca` timing still needs checking on a timestamped cohort |

### M6 — Demo and conclusion

| # | Title | Visual | Content |
|---|---|---|---|
| 15 | Live demo | Screen capture of `app.py`, no slide | Six steps, see [`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md) |
| 16 | Conclusion | 3 bullets, last line in accent | • Logistic regression on 10 inputs: recall **0.901**, CV ROC-AUC 0.844, external ROC-AUC 0.877 · • Simple, interpretable models were as good as XGBoost here · • **Decision support, not diagnosis** |
| 17 | Credits | The table from [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) | Name · section · slides. Hold 8 s |

## Part C — production checklist

1. Re-run the notebook (`jupyter nbconvert --execute --inplace`), then `python -m tests.check_notebook` → `clean`.
2. Export the charts named above from the rebuilt notebook.
3. Re-verify the six demo figures against the rebuilt artifact. They move when the model retrains.
4. Record the demo twice and keep one as the fallback cut.
5. Record at 1x with a visible timer. Do not speed up in the edit.
6. Watch once on a phone; if a number is unreadable, enlarge it.
