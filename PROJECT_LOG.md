# Project Log: Heart Disease Triage

A phase-by-phase record of what we built, what we checked, what we found, and what we changed along the way. For the layout of the files see `STRUCTURE.md`; for requirements see `SPEC.md`; for the task list see `tasks/todo.md`.

**One-paragraph summary.** We turned a course notebook (three models, accuracy on a 61-patient test set) into a tested pipeline: a Python package (`cardiotriage/`), an executed report notebook, a saved model, and a Streamlit demo app. The central finding is that the Kaggle copy of the dataset the original notebook used has **inverted labels, re-mapped category codes, and one patient swapped for a duplicate**, so we rebuilt everything on the original UCI file. The final model is a tuned logistic regression that, on the locked test set, flagged 28 of 28 patients who had disease.

---

## At a glance

| Phase | Goal | Checkpoint result | Smoke tests |
|---|---|---|---|
| 0 Foundation | Environment, data in the repo, data contract | Passed | 14 |
| 1 Story and QA | Notebook sections 0–5: framing, provenance proof, data QA, EDA | Passed; narrative reviewed and approved by the user | 22 |
| 2 Modelling | Validation design, feature test, 5-model comparison, tuning, selection | Passed; tables identical across runs | 36 |
| 3 Decision layer and evaluation | Cut-offs, decision curve, locked-test evaluation, safety, external validation, saved model | Passed | 53 |
| 4 Demo | Streamlit app and final verification | All six SPEC §12 criteria met | 58 |

The smoke-test count grows because every task adds tests first. The notebook has 107 cells (34 of them code) and runs headless in about 58 seconds.

---

## Before Phase 0: analysis and decisions

**What we started from.** `end-to-end-heart-disease-classification (1).ipynb`: Logistic Regression, KNN and Random Forest on `heart-disease.csv`, judged by accuracy on a 61-patient test set, with a 95% accuracy goal. The reference for quality was `Assignment.md` (the internship report), which we studied for its structure: executive summary, stated assumptions, data QA before analysis, a baseline, an imbalance-aware metric, a leakage-aware split, and a "markdown → code → Insight" rhythm.

**Gap analysis against the rubric.** The original notebook had no problem statistics, no model explanations, no preprocessing, no feature engineering, tuned on the test set, and had no demo. The full table is in `SPEC.md` §2.

**Decisions made with the user**
- Scope is **implementation and fixes only**: the pitch deck, script, team roles and a Thailand-specific angle are out of scope.
- Python **3.11** virtual environment; the original UCI data is the source of truth.
- Recall target: **recall ≥ 0.90** on the disease class.
- Structure: a **package of small modules** plus one combining notebook, instead of one big notebook (see the next section).
- A production stack (feature store, protobuf/gRPC, React or mobile apps) was considered and **rejected** as unnecessary for a one-week proof of concept; the code already has the production-shaped pieces (config, data contract, saved model, shared inference code).

**Why a package, not one notebook.** The notebook, the demo app and the tests import the same code, so the demo cannot drift from the trained model; logic can be unit-tested; and the notebook stays a readable story of markdown and short calls.

**Checked health statistics** (each carries its year in the notebook)

| Fact | Source |
|---|---|
| About 19.8 million cardiovascular deaths worldwide in 2022 (about 32% of all deaths) | WHO cardiovascular diseases fact sheet, updated 31 July 2025 |
| 919,032 US cardiovascular deaths in 2023; heart disease cost more than $168 billion in 2021–2022 | CDC Heart Disease Facts |
| 19 of 889 acute heart attack patients (2.1%) mistakenly sent home from the emergency department | Pope et al., *N Engl J Med* 2000;342:1163–70 |

---

## Phase 0: Foundation

**Built**
- `.venv` on Python 3.11.4, with `requirements.txt` pinned to the installed versions.
- `data/raw/` with the UCI Cleveland file (303 patients), the UCI Hungarian file (294) and the Kaggle CSV.
- `cardiotriage/config.py`: constants only (paths, seed, feature groups, official UCI code meanings, the Kaggle→UCI recode maps). It imports nothing from the package.
- `cardiotriage/data.py`: loaders, `deduplicate()`, `validate()` (the data contract) and `kaggle_vs_uci()`.
- `tests/smoke_test.py`: assert-based, no pytest.

**Checked.** 14/14 smoke tests; importing `data.py` pulls in no machine-learning or plotting library (and the detector was shown to fire when one is imported); loading, validating and running the Kaggle comparison raise no warnings with every warning turned into an error. Tests were written first: 12 of 13 failed before the code existed.

**Notes**
- Validation ranges (for example age 18–100) are our choice and live in `config.RANGES`.
- Pandas is version 3, a major version newer than the original notebook used; nothing broke.

---

## Phase 1: Story and QA (notebook sections 0–5)

**Built**
- Notebook sections 0–5: executive summary and problem importance with cited statistics; objective and who uses the model; the cost of each type of error; the reproducibility cell; data dictionary; the Kaggle-vs-UCI proof; missing values, duplicates, the data-contract check and class balance; six hypothesis → chart → Insight EDA blocks.
- `cardiotriage/plots.py` (charts) and `tests/check_notebook.py`, which fails an executed notebook if any cell errored or warned, or if an **Insight** quotes a number that its cells did not print.

**The central finding: the Kaggle file is not the original data.** All 303 Kaggle rows were matched back to UCI patients using six continuous measurements.

| Problem in the Kaggle copy | Evidence |
|---|---|
| **Labels inverted** | `target = 1` means *no* disease, for all 165 such rows; `target = 0` means disease for all 138 |
| **Category codes re-mapped** | `cp`, `restecg`, `slope`, `thal` all use different numbers than UCI; after applying the documented recode there are 0 mismatches |
| **Missing values hidden as fake codes** | `ca = 4` and `thal = 0` stand in for UCI's missing values |
| **One patient swapped** | Kaggle lost UCI patient 184 (a woman with disease) and duplicated a healthy man; UCI itself has no duplicates |

Consequence: every clinical conclusion in the original notebook was read backwards, and its data dictionary described codes the file did not use. The data contract rejects the Kaggle file, which the notebook shows as a negative control.

**Other findings**
- Asymptomatic chest pain has the highest disease rate (72.9%) and typical angina only 30.4%. This contradicts textbook expectation and is presented as a likely referral-selection effect.
- Only 6 of 3,939 values are missing, so patients are kept and imputed inside the model pipeline.
- Disease is 45.9% of patients, so no resampling is needed; "always predict no disease" would score 54.1% accuracy while catching nobody, which is why accuracy is not the headline metric.

**Checked.** Notebook check clean; two runs identical; 22/22 smoke tests. **Checkpoint B included a human review of the story, which the user approved.** After it, the decision was made that **the `.ipynb` itself is the source of truth** (cells are edited in place, not regenerated from a script).

**Lessons**
- A seaborn box plot triggered a matplotlib 3.11 deprecation warning; `plots.py` now draws box plots with matplotlib directly instead of hiding the warning.
- The notebook checker confirms a quoted number was printed, not which variable it belongs to (it let through `oldpeak` 0.43 when the cell printed 0.42). Every Insight was also read against its output by hand.

---

## Phase 2: Modelling (sections 6–9)

**Built**
- `features.py`: four clinical features (share of age-predicted maximum heart rate, hypertension flag, high-cholesterol flag, ST depression above 1 mm) and the preprocessing step (impute, scale, one-hot encode, all fitted inside each fold).
- `models.py`: five model families (Dummy baseline, Logistic Regression, KNN, Random Forest, XGBoost), their search ranges and `build_pipeline()`.
- `evaluation.py`: the locked 80/20 split (242 train / 61 test), 50 shared cross-validation folds (5-fold × 10 repeats), metrics, the corrected t-test, tuning, calibration and the learning curve.

**Design.** The test set is touched once, at the end. Tuning uses its own folds (a different seed) from the comparison folds. Model settings are chosen by cross-validation only.

**Results** (ROC-AUC, mean over the 50 folds)

| | Logistic regression | Random forest | KNN | XGBoost | Dummy |
|---|---|---|---|---|---|
| Default settings | 0.894 | 0.889 | 0.867 | 0.863 | 0.500 |
| Tuned | **0.900** | 0.894 | 0.896 | 0.897 | — |

- The four clinical features **did not help** (ROC-AUC −0.004 for logistic regression, −0.008 for random forest, neither significant), so they were dropped.
- **Selected model: tuned logistic regression** (`C = 0.216`). Its lead over XGBoost is not significant (p = 0.81), so the simpler, explainable model wins.
- Extra probability calibration was skipped: the Brier score was 0.1275 with it and 0.1264 without.
- The learning curve is still rising, so more patients would help.

**Checked.** Comparison and tuning tables identical across independent runs, compared cell by cell; 36/36 smoke tests; a prototype run with every warning as an error found none.

**Notes**
- Sections 6 and 7 swapped (validation design before feature engineering), because the feature test needs the folds.
- Tuned scores are slightly optimistic (the same 242 patients both choose and score the settings); the locked test set gives the honest number.
- Mistake caught: my first determinism comparison was broken by a file-encoding error and compared nothing; it was redone properly before the checkpoint was ticked.

---

## Phase 3: Decision layer and evaluation (sections 10–16)

**Built**
- `triage.py`: chooses the two tier cut-offs, turns a risk into a tier and action, and explains a patient's score with exact linear (SHAP-equivalent) contributions relative to the average training patient. It imports only `config`.
- More of `evaluation.py` and `plots.py`: out-of-fold risks, threshold sweep, decision-curve analysis, bootstrap and exact confidence intervals, subgroup recall, permutation importance.
- Notebook sections 10–16, ending with a model card and the saved model: `artifacts/heart_disease_pipeline.joblib` (pipeline, cut-offs, average patient, four demo presets, metrics, versions) and `artifacts/metrics.json`.

**Results**
- **Cut-offs**, from out-of-fold predictions on the training set only: low **0.287**, high **0.509**. Tiers: High → priority angiography; Medium → cardiology review or further testing; Low → routine follow-up with safety-netting advice.
- Per 100 patients the model misses **4.5** sick patients, against 9.9 at the default 0.5 cut-off. The High tier holds 102 training patients at an 85.3% disease rate; the Low tier 100 patients at 11.0%.
- **Decision-curve analysis:** acting on the model beats both "angiography for everyone" and "angiography for no one" at every risk level from 0.05 to 0.80.
- **Locked test set, used once:** all **28 of 28** sick patients flagged (exact 95% CI 0.877–1.000); specificity 0.667; ROC-AUC 0.961. The notebook says the cross-validated 0.900 is the realistic expectation, since this small test set was on the easy side.
- **Another hospital:** in Hungary a 10-feature version (without `ca`, `thal`, `slope`) reached ROC-AUC **0.880** and recall 0.906 with no refitting.
- **Interpretation:** the model relies most on `ca` (vessels seen on fluoroscopy), then `thal`, chest-pain type and sex; coefficients point the clinically expected way.

**Safety findings (recorded as known gaps, not hidden)**
- Out-of-fold recall is **0.778 for women (14 of 18) against 0.925 for men (86 of 93)**. With so few sick women the intervals overlap, but the gap matches the missed-diagnosis literature. The app shows a caution note for women.
- 11 of 111 sick training patients would land in the Low tier, mostly with normal imaging and no exercise angina. Patient 184 is one of them.
- `ca` and `thal` come late in the work-up, so the model is for prioritising the angiography queue **after** the work-up, not for first-contact screening.

**Checked.** A full run from a deleted `artifacts/` folder re-created both files; 0 notebook problems; all 18 traps referenced (the checker now fails if any is missing); 53/53 smoke tests; all 34 code cells identical to the previous run.

**Changes from the plan**
- Explanations compare a patient with the *average training patient* rather than the model's built-in offset, which reads correctly for categories like chest pain.
- No test patient was missed, so the demo's "missed by the model" patient comes from the training-set out-of-fold predictions.
- Binary columns are now imputed too (Hungary has gaps there); Phase 2 results were unchanged.
- A bootstrap interval collapses at 28 of 28, so exact (Clopper–Pearson) intervals are used for recall.

**Problems and fixes**
- **Console tracebacks.** Running the notebook with `nbconvert` on Windows prints 81 joblib `resource_tracker` tracebacks to the console. They never appear in a cell, and results are unaffected; they were there since Phase 2 but my Phase 2 check missed them. Disabling memmapping and shutting the worker pool down did not remove them. Setting `N_JOBS = 1` in `config.py` would (slower run).
- Two of my tests had wrong formulas (the code was right both times).
- An Insight overstated calibration; the test curve shows mid-range risk being over-stated, and the text now says so.

---

## Phase 4: Demo (`app.py`)

**Built.** A Streamlit app (`.venv/Scripts/streamlit run app.py`): a form using clinical names, four demo patients, risk percentage, a coloured tier with its action, a chart of which features raised or lowered this patient's risk, a what-if view (the change in risk and which inputs changed), a "decision support, not diagnosis" banner, a caution note for women, and an "About this model" panel. Values outside the clinical ranges are rejected with a clear error. It imports only `config`, `data` and `triage`, never training code.

**Checked**
- Streamlit's headless `AppTest`: every preset lands in its expected tier and risk, the contract rejects an age of 10, the what-if delta works, the app imports no training code, and it prints no deprecation notices.
- A real browser (Chrome DevTools): the High preset showed 99.3% and the "Missed by the model" preset 6.3% (Low), a change of −93.0 points.

**Browser testing found three things the headless tests did not:** truncated chart labels, a chart that kept the previous patient's bar order (so small effects appeared at the top), and an unreadably long list when switching to a different patient. All three were fixed.

**Final verification (Checkpoint E).** From a deleted `artifacts/`: 58-second run, notebook check clean, 58/58 smoke tests, all 34 code cells identical to the previous run, the original notebook unchanged, installed versions matching `requirements.txt` (which now also pins `altair`, used directly by the app).

| SPEC §12 criterion | Evidence |
|---|---|
| 1. Label flip proved | Kaggle↔UCI mismatches are 0 in every column |
| 2. All 18 traps handled | All 18 are in the notebook's trap table |
| 3. Five model families compared | Mean ± SD on the same 50 folds, plus a significance test |
| 4. Recall ≥ 0.90 | 0.901 out-of-fold (specificity 0.679); test 28/28, exact 95% CI 0.877–1.000 |
| 5. No errors or warnings; model saved | 0 notebook problems; model file written |
| 6. App works; tests pass | 58/58, plus the browser check |

---

## Post-build code review

A five-axis review (correctness, readability, architecture, security, performance) found no critical problems. To check whether the tests really guard the code, we deliberately broke it in three places:

| Deliberate break | Before the fixes | After |
|---|---|---|
| High tier takes the *highest* qualifying cut-off instead of the lowest | Not caught | Caught by a new hand-computed test |
| A patient exactly at the low cut-off is no longer flagged | Caught | Caught |
| The bootstrap stops skipping one-class resamples | Not caught | Caught: the new test fails on sklearn's warning |

**Fixed**
- **High cut-off now pinned.** A new test uses 8 hand-computed patients where three cut-offs reach the PPV target, and checks that the lowest one is chosen. It also checks the fallback when no cut-off reaches the target.
- **Bootstrap guard now pinned.** In sklearn 1.9, a resample with only sick (or only healthy) patients makes ROC-AUC return NaN with a warning; it does not crash. The guard skips such resamples entirely, so the run stays free of warnings and every metric is computed on the same draws. A 6-patient test now checks this.
- **`deduplicate()` keeps the original row ids.** It used to renumber rows, so a duplicate early in a file would have shifted every later patient id (for example, "patient 184" would have pointed at someone else). Cleveland has no duplicates, so no result changes.
- **Tests split by area.** The 698-line `tests/smoke_test.py` is now one file per area (`tests/test_data.py`, `test_triage.py`, `test_app.py`, and so on). `smoke_test.py` runs them all, or only the areas you name: `python -m tests.smoke_test triage`.

- **Small fixes.**
  - The app's "vs previous" change now compares with the last *different* inputs. A rerun with nothing changed used to reset it to +0.0; a new test checks this.
  - The chart function `plots.test_curves` is renamed `roc_pr_calibration`, because its old name made it look like a test.
  - Smaller cleanups: an outdated module description, an `is None` check, the list of what the saved model file contains (in SPEC), and the test runner now fails if it finds no tests.

Smoke tests went from 58 to 62. The notebook was re-run after these changes; the text output of all 34 code cells is identical.

---

## Phases 5–9: rubric-audit follow-up (2026-11-06)

A strict audit against the 30-mark rubric scored Problem Statement and Implementation at 21.3 of 24. These phases close every gap it found (Presentation was out of scope). Decisions: no move of the original notebook; nested CV in; pooling Hungary into training stays a roadmap item; nothing committed.

| Phase | What changed | Key result |
|---|---|---|
| 5 Evidence and framing | Section 0 rebuilt in two layers (why CVD matters; why the angiography queue matters); one-line problem statement; formal prediction task; UCI dataset DOI | Patel et al. 2010: only 37.6% of 398,978 elective angiograms found obstructive disease. Al-Hijji et al. 2019: 8.2 major complications per 10,000 diagnostic catheterisations. Pope 2000 wording corrected (the "women under 55" finding covers all acute ischaemia). No citable cost per procedure, so impact is in patients and slots |
| 6 Comparison and math | PR-AUC in every CV table; one-line math per model family; new §9.2 compares the tuned models at their own recall-0.90 cut-off | At recall 0.901, specificity: logistic regression 0.679, XGBoost 0.672, KNN 0.603, random forest 0.565. 6 sick patients are missed by all four models, so the misses come from the data |
| 7 Honest tuning | Nested CV (§9.3) for the selected model and the runner-up; caveat on the cut-offs in §10; §15.3 split into accuracy levers (each with its evidence) and deployment | Nested ROC-AUC 0.909 vs tuned 0.900 (XGBoost 0.905 vs 0.897): no detectable tuning optimism. All four models would have added ~2 minutes, over the 3-minute budget |
| 8 Impact | §10.3 queue-gain curve and per-1,000 table; §15.2 now a code cell plus checked Insight, framed as benefit (earlier care, 368 healthy patients per 1,000 spared a priority angiogram) against price (45.5 sick patients per 1,000 left in Low, recall 0.778 for women vs 0.925 for men); avoided complications are only ≈ 0.3 per 1,000, so the gain is time and slots, not avoided harm | First 40% of slots serve 76.6% of sick patients (40% in arrival order); per 1,000 referred patients, 421.5 are High (359.5 sick) and 413.2 Low (45.5 sick) |
| 9 Demo and repo | `README.md`, `DEMO.md` (figures produced by driving the app with `AppTest`), duplicate import removed, SPEC v3 and STRUCTURE updated | Demo: High 99.3%, Low 4.0% (with the caution note for women), Borderline 29.6% → 74.7% when `ca` goes 0 → 2, missed patient 6.3%, age 10 rejected |

**New code (tests first):** `evaluation.operating_points`, `evaluation.nested_cv` (reuses the search that `tune` builds), `evaluation.queue_gain`, `plots.confusion_grid` (shares its drawing with `plots.confusion`), `plots.queue_gain`; `SCORING` gained `pr_auc`. `evaluation` now also imports `triage` (for `pick_thresholds`); there is no cycle.

**Checked.** Smoke tests went from 62 to 68 (each new function's test was written first and seen failing; three deliberate code breaks were all caught). After every phase: headless run, `check_notebook` clean, all smoke tests pass, and every earlier code cell's text output compared with the previous run (identical, apart from the new PR-AUC column).


**Phase 10 (2026-11-07): problem importance and clarity.** Section 0 gained the waiting-list harm figure (Natarajan *CMAJ* 2002: 1.4% of 8,030 patients had a major event while waiting) and a Thailand layer (IHD mortality trend, survival linked to angiography access, 36.1% normal elective angiograms in one provincial hospital). §1 now frames the work as two stages, with Stage 1 (10 features, no imaging, validated in Hungary in §13) and Stage 2 (13 features, the main model), which answers "if imaging is done, why predict the angiogram?". Markdown only; no code or outputs changed. A public-payer price per angiogram still could not be cited (SPEC §6), so impact stays in patients and slots.

**Phase 11 (2026-11-07): second rubric audit (importance, clarity, tuning).**

| Gap | What changed | Key result |
|---|---|---|
| No money in the cost/benefit argument; queue evidence 1998–2008 | §0 gained the CMS CY 2026 Medicare payment per angiogram (CPT 93458, APC 5191: $3,312 OPPS, $1,708 ASC) and the BHF/NHS England waiting list (Sept 2025); new cost table after §15.2 | Angiography for High + Medium tiers only: $1,368,518 saved per 1,000 referred (OPPS), cost per sick patient found $7,220 → $4,703, with 45.5 sick per 1,000 deferred to follow-up |
| EDA never showed why blood pressure and cholesterol are weak; Stage 1/2 dense | new §5.7 box plots with Mann-Whitney tests (reuses `plots.by_disease`); "What we carry into modelling" is now an EDA-finding → design-choice table; one-sentence summary of the two stages in §1 | Median blood pressure 130 in both groups, cholesterol 249 vs 234.5 (p ≈ 0.03), against p ≈ 1e-13 for max heart rate and ST depression |
| No feature selection; only 30 search settings | new §9.5 drop-column CV + one-step backward elimination; §9.6 30 vs 100 settings; §15.3 item 3 updated | Only `ca` is indispensable (−0.030, p = 0.016); the 6-feature-smaller model is +0.007, p = 0.47, so all 13 kept. 100 settings: LR unchanged, XGBoost +0.003 |
| Overclaim: "LR is best at the operating point" | §9.2 now prints healthy patients spared per model; Insight and README say LR and XGBoost are tied | 89 vs 88 of 131 healthy patients (one patient) |

**Checked.** No package code changed. Headless run (about 1 min 45 s), `check_notebook` clean, smoke tests pass, `artifacts/metrics.json` byte-identical to before, so the shipped model, cut-offs and DEMO.md figures are unchanged.

**Phase 11 (2026-11-07): recent and Thai evidence; one problem statement plus a circularity check.** An audit docked importance (queue-harm evidence 16–24 years old, money case US-only, Thai yield from one hospital) and clarity (`ca` "close to how the label is measured" and two "stages" read as two problem statements).

| Gap | Change | Evidence |
|---|---|---|
| Old queue-harm evidence | §0 adds Tam et al. 2021 | 584,341 Ontario referrals; pandemic-era PCI referral HR 1.83 (1.47–2.27) for death while waiting |
| Thai figure from one hospital | §0 adds national NHSO data, volume and capacity | Khananuraksa 2024: 37.6% of 141,605 UC angiograms led to no PCI or CABG; NHSO FY2025: 33,667 CAG-only; 72 PCI hospitals (Chandavimol 2025), about 1 per million vs 5.7 cath labs per million in ESC countries |
| No Thai price | Searched NHSO, CGD fee schedule (scanned, unreadable), SSO 2026; none citable. §0 now names the transferable figure (angiograms avoided per 1,000) | SPEC §6 lists every source and what was not found |
| "Stage 1 / Stage 2" = two problem statements | §1 rewritten: one problem (order the angiography queue after imaging) and a **no-imaging check**; new cell *Is the model reading the answer?* before §6; §13 retitled | 0.900 with imaging vs 0.846 without; 0.880 in Budapest |
| Circularity defended only on paper | The no-imaging model and its cut-offs are saved in the artifact; the app has a *Circularity check* switch; DEMO step 6 | Borderline preset 29.6% with imaging, 57.0% without; the no-imaging model gives identical risk whatever `ca`, `thal`, `slope` are (new test) |

**Checked.** Headless run 1 min 40 s, `check_notebook` clean, smoke tests 70/70 (2 new). Every existing figure in `metrics.json` unchanged; it gains `no_imaging_cv_roc_auc` 0.8459. App driven through the agent API with no console warnings; mutation (app scoring with the full model) caught by the new app test.

---

## How to reproduce

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb
.venv/Scripts/python -m tests.check_notebook     # must print "clean"
.venv/Scripts/python -m tests.smoke_test         # must pass all tests
.venv/Scripts/streamlit run app.py               # the demo
```

Run all commands from the repo root.

---

## Known limitations and open items

- **One hospital, one era.** Cleveland, 1980s, patients already referred for angiography. The model is not for screening the general population.
- **Small data.** 303 patients; the test set has 61. Test-set intervals are wide and the cross-validated figures are the more stable guide.
- **Recall gap for women** (0.778 vs 0.925); too few sick women to be certain, so it is a known gap to monitor.
- **Calibration** should be re-checked on any new hospital before showing risk as an exact probability.
- **Console noise** when running the notebook (81 joblib tracebacks); harmless, with a documented option (`N_JOBS = 1`) to silence it.
- **Not built, by decision:** the pitch deck and script, team roles, a deployed app, and a production service stack.

## Next steps (if the project continued)
1. Pool more data (the other UCI hospitals, modern records); the learning curve was still rising.
2. Run a prospective, silent pilot comparing the model's tiers with real outcomes.
3. Improve recall for women: more data, a review of missed cases.
4. Re-calibrate per hospital and monitor drift after deployment.
5. Production path: records-system integration, a model service with a typed input contract (our `data.validate`), monitoring.
