# Tasks: Heart Disease Triage PoC

Layout: the logic lives in the `cardiotriage/` package; the notebook (`nb` = `heart_disease_pitch_gold_standard.ipynb`) is the only combiner (see SPEC §8).
"Headless run" means `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace <nb>`, followed by `.venv/Scripts/python -m tests.check_notebook` reporting "clean" (no errors, no Warning/Error/Traceback on stderr, and every number in an **Insight:** cell is printed by the cells above it).
Notebook source: the `.ipynb` itself (Option B, decided at Checkpoint B). Edit cells in place, never regenerate it; after each edit run the headless run and `tests.check_notebook`, and commit after each finished section so a bad edit can be rolled back.
Smoke test: `.venv/Scripts/python -m tests.smoke_test` (run from the repo root).
Import rule: `config` ← `data`, `features` ← `models` ← `evaluation` (which also uses `data`); `triage` ← `config` (allowed `data`, not needed); `plots` ← `config`. No cycles. `__init__.py` stays empty.

## Phase 0: Foundation

- [x] **Task 1: Environment, data, package skeleton** (S)
  - Acceptance: `.venv` uses Python 3.11; `requirements.txt` is pinned to the installed versions; `data/raw/` holds the UCI Cleveland and Hungarian files and the Kaggle CSV; `.gitignore` excludes `.venv/`; `cardiotriage/__init__.py` exists and is empty; `artifacts/` exists.
  - Verify: `.venv/Scripts/python -c "import sklearn, xgboost, streamlit, nbconvert, seaborn, cardiotriage"` exits 0.
  - Files: `requirements.txt`, `.gitignore`, `data/raw/*`, `cardiotriage/__init__.py`
  - Deps: none

- [x] **Task 2: `config.py` and `data.py`: the data contract** (M)
  - Acceptance: `config.py` holds paths, `SEED`, feature groups, the UCI code→label maps, and `RECALL_TARGET`, and imports nothing from the package. `data.py` has `load_cleveland()` (`?`→NaN, `disease = num>0`), `load_hungarian()`, `load_kaggle()`, `kaggle_vs_uci()` (returns the crosstabs that prove T1/T2), and `validate()` (raises `ValueError` on out-of-range values or unknown codes; NaN allowed only in `ca`/`thal`).
  - Verify: `tests/smoke_test.py` asserts Cleveland is 303 rows with 139 positives, `validate()` passes on it and rejects `cp=5` and `age=-1`, and `kaggle_vs_uci()` shows 0 target mismatches.
  - Files: `cardiotriage/config.py`, `cardiotriage/data.py`, `tests/smoke_test.py`
  - Deps: 1

### Checkpoint A
- [x] Smoke test passes (14/14 on Python 3.11); `import cardiotriage.data` does not import sklearn, xgboost, matplotlib, or seaborn.

## Phase 1: Story and QA

- [x] **Task 3: nb §0–4: framing, reproducibility, provenance proof, integrity QA** (M)
  - Acceptance: exec summary; cited stats with years (SPEC §6); objective, persona, and FN/FP cost table; seed and version cell; `data.kaggle_vs_uci()` output for target and all re-mapped codes (T1/T2); official dictionary; duplicates, missing values, and range checks via `data.validate()`; class balance. The notebook cells only call package functions.
  - Verify: headless run is clean; the target crosstab shows 0 off-diagonal mismatches.
  - Files: `nb`
  - Deps: 2

- [x] **Task 4: `plots.py` EDA charts + nb §5** (S)
  - Acceptance: 5–6 hypothesis → chart → Insight blocks on the corrected target, each chart one `plots.*` call; each Insight quotes a printed number; correct legends and labels (T11, T12). `plots.py` imports only `config` and matplotlib/seaborn.
  - Verify: headless run is clean; every Insight number appears in the cell output above it.
  - Files: `cardiotriage/plots.py`, `nb`
  - Deps: 3

### Checkpoint B
- [x] Headless run is clean (`python -m tests.check_notebook` → clean; 0 errors, 0 warnings in cell outputs).
- [x] **Human reviews the narrative (§0–5) before modelling.** Approved by the user. Decision: the `.ipynb` is now the source of truth (Option B); the builder script is retired.

## Phase 2: Modelling

- [x] **Task 5: `features.py`, `build_pipeline`, validation design (nb §6–7)** (M) — §6 is now validation design and §7 feature engineering (ablation needs the folds). Domain features dropped: ROC-AUC −0.004 (LR, p = 0.13), −0.008 (RF, p = 0.15).
  - Acceptance: `features.py` has module-level (picklable) domain-feature functions (% age-predicted max HR, `trestbps ≥ 130`, `chol ≥ 240`, `oldpeak > 1`) and `build_preprocessor()`, and imports only `config`. `models.build_pipeline(model, use_fe)` chains them. `evaluation.make_split()` (de-duplicate, then stratified 80/20) and `evaluation.make_cv()` (`RepeatedStratifiedKFold(5×10)`). A CV ablation decides whether the domain features are kept. A toy confusion-matrix sanity check in the notebook.
  - Verify: headless run is clean; the smoke test fits and predicts with `build_pipeline(LogisticRegression())`, round-trips it through `joblib`, and asserts the unpickle only needs `cardiotriage.features`.
  - Files: `cardiotriage/features.py`, `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `nb`, `tests/smoke_test.py`
  - Deps: 4

- [x] **Task 6: Five-family comparison (nb §8)** (S)
  - Acceptance: `models.MODEL_ZOO` (Dummy, LR, KNN, RF, XGBoost, all with `random_state=SEED`); `evaluation.cv_table()` gives AUC, recall, specificity, F1, and Brier as mean ± SD on identical folds; `evaluation.corrected_ttest()` (Nadeau–Bengio) for the best vs the runner-up; a "how it works / why here" paragraph for each family; box plot via `plots`.
  - Verify: headless run is clean; a rerun reproduces the same numbers; the corrected t-test returns a p-value in [0, 1] in the smoke test (synthetic scores).
  - Files: `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`
  - Deps: 5

- [x] **Task 7: Tuning, learning curve, calibration, selection (nb §9)** (M)
  - Acceptance: `models.SEARCH_SPACES`; `evaluation.tune()` runs `RandomizedSearchCV` per model inside CV; before/after table; `evaluation.learning_curve_table()`; calibration only if the Brier score improves; the selection rule from SPEC §4 is applied and printed.
  - Verify: headless run is clean, with no ConvergenceWarning or FutureWarning.
  - Files: `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`
  - Deps: 6

### Checkpoint C
- [x] Comparison and tuning tables are identical across 2 headless runs (all 22 code cells, compared per cell from the executed JSON). Selected model recorded.
  - **Selected: tuned Logistic Regression**, `C = 0.216`, `class_weight = None`, 13 raw features (no domain features, no extra calibration). CV ROC-AUC 0.900 ± 0.035; runner-up XGBoost 0.897 (corrected t-test p = 0.81). Brier 0.1264 (Platt calibration 0.1275, so not used).

## Phase 3: Decision layer and evaluation

- [x] **Task 8: Thresholds, decision curve, `triage.py` (nb §10)** (M) — cut-offs low 0.287 / high 0.509 from 10×5-fold OOF risks (OOF recall 0.901, specificity 0.679). Contributions changed to exact linear SHAP: `coef × (x − average training patient)`, summing to logit(patient) − logit(average patient).
  - Acceptance: `evaluation.threshold_sweep()` on OOF predictions and `evaluation.net_benefit()`; a per-100-patients table; a DCA plot. `triage.py` has `pick_thresholds()` (low cut at recall ≥ `RECALL_TARGET`, high cut at PPV ≥ 0.85 or the closest achievable), `triage(p, thresholds)` → tier + action, and `contributions(pipe, patient)`. `triage.py` imports only `config` and `data`.
  - Verify: the smoke test asserts the tier boundaries, that contributions sum to logit − intercept (±1e-6), and that importing `cardiotriage.triage` does not import `evaluation`, `models`, or `plots`.
  - Files: `cardiotriage/triage.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`, `tests/smoke_test.py`
  - Deps: 7

- [x] **Task 9: Final test evaluation and clinical safety (nb §11–12)** (M) — test recall 28/28 (bootstrap CI collapses to 1.0, so exact Clopper–Pearson 0.877–1.000 added), specificity 0.667, ROC-AUC 0.961. No test patient missed, so the "missed" case study uses the OOF training miss. Known gap: OOF recall women 0.778 (14/18) vs men 0.925 (86/93).
  - Acceptance: the test set is used once; `evaluation.bootstrap_ci()` (1,000 resamples) gives recall, specificity, PPV, and NPV with 95% CIs; ROC, PR, confusion matrix, and calibration curve via `plots`; FN table; 3 case studies using `triage.contributions()`; recall by sex and age band.
  - Verify: headless run is clean; the CI columns are printed; the smoke test checks `bootstrap_ci` returns lo ≤ point ≤ hi on synthetic data.
  - Files: `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`, `tests/smoke_test.py`
  - Deps: 8

- [x] **Task 10: External validation on Hungary (nb §13), STRETCH** (S) — 10-feature model: Cleveland CV ROC-AUC 0.846, Hungary 0.880 (0.833–0.921), recall 0.906 at the Cleveland cut-off.
  - Acceptance: the 10-feature model is trained on Cleveland and scored with `data.load_hungarian()`; AUC and recall are compared with the internal CV; the drop is interpreted honestly.
  - Verify: headless run is clean.
  - Files: `cardiotriage/evaluation.py` (a small helper only if needed), `nb`
  - Deps: 9. **Can be cut** if the schedule slips.

- [x] **Task 11: Interpretation, limitations, model card, serialize (nb §14–16)** (S) — artifact also holds `baseline` and `presets`; the 4th preset ("Missed by the model") is a training patient because no test patient was missed. Trap index table (T1–T18) in §15.4; the notebook checker now fails if any trap id is unreferenced.
  - Acceptance: permutation importance + standardized coefficients; Limitations (T15–T17); impact; next steps; model card; `joblib.dump` to `artifacts/heart_disease_pipeline.joblib` of the pipeline, thresholds, features, metrics, and versions, plus `artifacts/metrics.json`.
  - Verify: the smoke test loads the joblib and scores 4 preset patients with the expected tiers.
  - Files: `nb`, `tests/smoke_test.py`, `artifacts/*`
  - Deps: 9 (10 optional)

### Checkpoint D
- [x] A full headless run gives 0 errors and 0 warnings; the joblib file is written; the smoke test passes. T1–T18 are each referenced in the notebook.
  - Verified from a deleted `artifacts/` folder (the notebook re-created both files); `check_notebook` clean incl. trap check; 53/53 smoke; 34 code cells identical to the previous run; run time ~58 s.
  - Known noise **outside** the notebook: the `nbconvert` console shows 81 joblib/loky `resource_tracker` KeyError tracebacks (Windows, parallel workers in a Jupyter kernel) plus two Jupyter warnings. None reach a cell output; results are unaffected. Present since Phase 2. Fix if wanted: `config.N_JOBS = 1` (slower run).

## Phase 4: Demo

- [x] **Task 12: Streamlit `app.py`** (S) — verified headlessly with Streamlit `AppTest` (5 tests: import rule, all 4 presets → expected tier and risk, contract rejects age 10, what-if delta + changed-input list, no deprecation notices) and in a real browser (Chrome DevTools: High preset 99.3%, "Missed by the model" 6.3% Low, delta −93.0 pts). Browser check found and fixed: truncated chart labels, chart order kept from the previous patient, duplicate "Patient" heading. Form limits are wider than the data contract on purpose, so `validate()` is what rejects implausible values.
  - Acceptance: imports only `config`, `data.validate`, and `triage` (plus streamlit/joblib); a form with clinical labels and ranges; 4 presets; risk %; tier with action; top contributions; what-if delta vs the previous run; disclaimer; `st.cache_resource`; invalid input shows `st.error` instead of crashing.
  - Verify: `streamlit run app.py`; click each preset and capture a screenshot; enter one invalid value; the smoke test asserts `app.py` has no import of `evaluation`, `models`, or `plots`.
  - Files: `app.py`, `tests/smoke_test.py`
  - Deps: 11

- [x] **Task 13: Final verification** (XS) — from a deleted `artifacts/`: headless run 58 s, `check_notebook` clean, smoke 58/58, all 34 code cells identical to the previous run, original notebook unchanged since commit 801bab5, installed versions match `requirements.txt` (now also pins `altair`, which `app.py` imports directly), `pip check` clean.
  - Acceptance: a fresh headless run, the smoke test, and an app check all pass; the original notebook is unchanged (`git diff --stat` shows no change to it).
  - Verify: the commands in SPEC §7.
  - Files: none
  - Deps: 12

### Checkpoint E: Complete
- [x] All SPEC §12 success criteria are met. Ready for review and commit (commit only on request).
  - Criteria 1–5 checked by a script against the executed notebook's printed outputs: Kaggle↔UCI mismatches all 0; trap table has T1–T18 and no id missing; 5 families with mean ± SD and corrected t-test (p = 0.70); OOF recall 0.901 / specificity 0.679, test recall exact CI 0.877–1.000; 0 notebook problems and the artifact written. Criterion 6: app presets, contract rejection and what-if covered by `AppTest` in the smoke test, plus a real-browser check.
  - Still true: the `nbconvert` console prints 81 known joblib tracker tracebacks (nothing else); see plan.md risks.
