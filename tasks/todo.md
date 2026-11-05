# Tasks: Heart Disease Triage PoC

The notebook is `nb` = `heart_disease_pitch_gold_standard.ipynb`. "Headless run" means
`.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace <nb>`, followed by a grep of the outputs for `Warning|Error|Traceback` that returns 0 hits.

## Phase 0: Foundation

- [ ] **Task 1: Environment and data in the repo** (S)
  - Acceptance: `.venv` uses Python 3.11; `requirements.txt` is pinned to the installed versions; `data/` holds the UCI Cleveland and Hungarian files and the Kaggle CSV; `.gitignore` excludes `.venv/`.
  - Verify: `.venv/Scripts/python -c "import sklearn, xgboost, streamlit, nbconvert, seaborn"` exits 0.
  - Files: `requirements.txt`, `.gitignore`, `data/*`
  - Deps: none

- [ ] **Task 2: `heart.py` data contract** (S)
  - Acceptance: `COLUMNS`, the feature groups, the UCI code→label maps, `load_cleveland()` (with `?`→NaN and `disease = num>0`), and `validate()` (raises `ValueError` on out-of-range values or unknown codes; NaN allowed only in `ca`/`thal`).
  - Verify: `tests/smoke_test.py` asserts that the Cleveland data loads as 303×14 + `disease` with 139 positives, that `validate()` passes on it, and that `validate()` rejects `cp=5` and `age=-1`.
  - Files: `heart.py`, `tests/smoke_test.py`
  - Deps: 1

### Checkpoint A
- [ ] Smoke test passes; imports are clean.

## Phase 1: Story and QA

- [ ] **Task 3: nb §0–4: framing, reproducibility, provenance proof, integrity QA** (M)
  - Acceptance: exec summary; cited stats with years (SPEC §6); objective, persona, and FN/FP cost table; seed and version cell; Kaggle↔UCI crosstabs for target and all re-mapped codes (T1/T2); official dictionary; duplicates, missing values, and range checks; class balance.
  - Verify: headless run is clean; the target crosstab shows 0 off-diagonal mismatches.
  - Files: `nb`
  - Deps: 2

- [ ] **Task 4: nb §5: focused EDA on the corrected target** (S)
  - Acceptance: 5–6 hypothesis → chart → Insight blocks, each Insight quoting a printed number; correct legends and labels (T11, T12).
  - Verify: headless run is clean; every Insight number appears in the cell output above it.
  - Files: `nb`
  - Deps: 3

### Checkpoint B
- [ ] Headless run is clean. **Human reviews the narrative (§0–5) before modelling.**

## Phase 2: Modelling

- [ ] **Task 5: Pipeline, feature engineering ablation, validation design (nb §6–7)** (M)
  - Acceptance: `heart.build_pipeline()`; 4 domain features with rationale; a CV ablation decides whether they are kept; stratified 80/20 locked test set after de-duplication; a `RepeatedStratifiedKFold(5×10)` object reused everywhere; a toy confusion-matrix sanity check.
  - Verify: headless run is clean; the smoke test also asserts that `build_pipeline(LogisticRegression())` fits and predicts on the Cleveland data.
  - Files: `heart.py`, `nb`, `tests/smoke_test.py`
  - Deps: 4

- [ ] **Task 6: Five-family comparison (nb §8)** (S)
  - Acceptance: Dummy, LR, KNN, RF, and XGBoost on identical folds; a "how it works / why here" paragraph for each; table of AUC, recall, specificity, F1, and Brier as mean ± SD; box plot; Nadeau–Bengio corrected t-test for the best vs the runner-up.
  - Verify: headless run is clean; KNN AUC beats the old unscaled 69% accuracy baseline story; a rerun reproduces the same numbers.
  - Files: `nb`
  - Deps: 5

- [ ] **Task 7: Tuning, learning curve, calibration, selection (nb §9)** (M)
  - Acceptance: `RandomizedSearchCV` per model inside CV; before/after table; learning curve; calibration only if the Brier score improves; the selection rule from SPEC §4 is applied and printed.
  - Verify: headless run is clean, with no ConvergenceWarning or FutureWarning.
  - Files: `nb`
  - Deps: 6

### Checkpoint C
- [ ] Comparison and tuning tables are identical across 2 headless runs. Selected model recorded.

## Phase 3: Decision layer and evaluation

- [ ] **Task 8: Thresholds, decision curve, triage tiers (nb §10)** (S)
  - Acceptance: an OOF threshold sweep; a per-100-patients table; a DCA plot (net benefit vs treat-all/none); `heart.triage()` with 2 thresholds (low cut at recall ≥ 0.90, high cut at PPV ≥ 0.85, or the closest achievable); `heart.contributions()`.
  - Verify: the smoke test asserts tier boundaries and that contributions sum to logit − intercept (±1e-6).
  - Files: `heart.py`, `nb`, `tests/smoke_test.py`
  - Deps: 7

- [ ] **Task 9: Final test evaluation and clinical safety (nb §11–12)** (M)
  - Acceptance: the test set is used once; ROC, PR, confusion matrix, and calibration curve; recall, specificity, PPV, and NPV with bootstrap 95% CIs (1,000 resamples); FN table; 3 case studies with contributions; recall by sex and age band.
  - Verify: headless run is clean; the CI columns are printed.
  - Files: `nb`
  - Deps: 8

- [ ] **Task 10: External validation on Hungary (nb §13), STRETCH** (S)
  - Acceptance: the 10-feature model is trained on Cleveland and scored on `processed.hungarian.data`; AUC and recall are compared with the internal CV; the drop is interpreted honestly.
  - Verify: headless run is clean.
  - Files: `nb`
  - Deps: 9. **Can be cut** if the schedule slips.

- [ ] **Task 11: Interpretation, limitations, model card, serialize (nb §14–16)** (S)
  - Acceptance: permutation importance + standardized coefficients; Limitations (T15–T17); impact; next steps; model card; `joblib.dump` of the pipeline, thresholds, features, metrics, and versions.
  - Verify: the smoke test loads the joblib and scores 4 preset patients with the expected tiers.
  - Files: `nb`, `tests/smoke_test.py`, `heart_disease_pipeline.joblib`
  - Deps: 9 (10 optional)

### Checkpoint D
- [ ] A full headless run gives 0 errors and 0 warnings; the joblib file is written; the smoke test passes. T1–T18 are each referenced in the notebook.

## Phase 4: Demo

- [ ] **Task 12: Streamlit `app.py`** (S)
  - Acceptance: a form with clinical labels and ranges; 4 presets; risk %; tier with action; top contributions; what-if delta vs the previous run; disclaimer; `st.cache_resource`; invalid input shows `st.error` instead of crashing.
  - Verify: `streamlit run app.py`; click each preset and capture a screenshot; enter one invalid value.
  - Files: `app.py`
  - Deps: 11

- [ ] **Task 13: Final verification** (XS)
  - Acceptance: a fresh headless run, the smoke test, and an app check all pass; the original notebook is unchanged (`git diff --stat` shows no change to it).
  - Verify: the commands in SPEC §7.
  - Files: none
  - Deps: 12

### Checkpoint E: Complete
- [ ] All SPEC §12 success criteria are met. Ready for review and commit (commit only on request).
