# Tasks: Heart Disease Triage PoC

Layout: the logic lives in the `cardiotriage/` package; the notebook (`nb` = `heart_disease_pitch_gold_standard.ipynb`) is the only combiner (see SPEC §8).
"Headless run" means `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace <nb>`, followed by a grep of the outputs for `Warning|Error|Traceback` that returns 0 hits.
Smoke test: `.venv/Scripts/python -m tests.smoke_test` (run from the repo root).
Import rule: `config` ← `data`, `features` ← `models` ← `evaluation`; `triage` ← `config`, `data`; `plots` ← `config`. No cycles. `__init__.py` stays empty.

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

- [ ] **Task 3: nb §0–4: framing, reproducibility, provenance proof, integrity QA** (M)
  - Acceptance: exec summary; cited stats with years (SPEC §6); objective, persona, and FN/FP cost table; seed and version cell; `data.kaggle_vs_uci()` output for target and all re-mapped codes (T1/T2); official dictionary; duplicates, missing values, and range checks via `data.validate()`; class balance. The notebook cells only call package functions.
  - Verify: headless run is clean; the target crosstab shows 0 off-diagonal mismatches.
  - Files: `nb`
  - Deps: 2

- [ ] **Task 4: `plots.py` EDA charts + nb §5** (S)
  - Acceptance: 5–6 hypothesis → chart → Insight blocks on the corrected target, each chart one `plots.*` call; each Insight quotes a printed number; correct legends and labels (T11, T12). `plots.py` imports only `config` and matplotlib/seaborn.
  - Verify: headless run is clean; every Insight number appears in the cell output above it.
  - Files: `cardiotriage/plots.py`, `nb`
  - Deps: 3

### Checkpoint B
- [ ] Headless run is clean. **Human reviews the narrative (§0–5) before modelling.**

## Phase 2: Modelling

- [ ] **Task 5: `features.py`, `build_pipeline`, validation design (nb §6–7)** (M)
  - Acceptance: `features.py` has module-level (picklable) domain-feature functions (% age-predicted max HR, `trestbps ≥ 130`, `chol ≥ 240`, `oldpeak > 1`) and `build_preprocessor()`, and imports only `config`. `models.build_pipeline(model, use_fe)` chains them. `evaluation.make_split()` (de-duplicate, then stratified 80/20) and `evaluation.make_cv()` (`RepeatedStratifiedKFold(5×10)`). A CV ablation decides whether the domain features are kept. A toy confusion-matrix sanity check in the notebook.
  - Verify: headless run is clean; the smoke test fits and predicts with `build_pipeline(LogisticRegression())`, round-trips it through `joblib`, and asserts the unpickle only needs `cardiotriage.features`.
  - Files: `cardiotriage/features.py`, `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `nb`, `tests/smoke_test.py`
  - Deps: 4

- [ ] **Task 6: Five-family comparison (nb §8)** (S)
  - Acceptance: `models.MODEL_ZOO` (Dummy, LR, KNN, RF, XGBoost, all with `random_state=SEED`); `evaluation.cv_table()` gives AUC, recall, specificity, F1, and Brier as mean ± SD on identical folds; `evaluation.corrected_ttest()` (Nadeau–Bengio) for the best vs the runner-up; a "how it works / why here" paragraph for each family; box plot via `plots`.
  - Verify: headless run is clean; a rerun reproduces the same numbers; the corrected t-test returns a p-value in [0, 1] in the smoke test (synthetic scores).
  - Files: `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`
  - Deps: 5

- [ ] **Task 7: Tuning, learning curve, calibration, selection (nb §9)** (M)
  - Acceptance: `models.SEARCH_SPACES`; `evaluation.tune()` runs `RandomizedSearchCV` per model inside CV; before/after table; `evaluation.learning_curve_table()`; calibration only if the Brier score improves; the selection rule from SPEC §4 is applied and printed.
  - Verify: headless run is clean, with no ConvergenceWarning or FutureWarning.
  - Files: `cardiotriage/models.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`
  - Deps: 6

### Checkpoint C
- [ ] Comparison and tuning tables are identical across 2 headless runs. Selected model recorded.

## Phase 3: Decision layer and evaluation

- [ ] **Task 8: Thresholds, decision curve, `triage.py` (nb §10)** (M)
  - Acceptance: `evaluation.threshold_sweep()` on OOF predictions and `evaluation.net_benefit()`; a per-100-patients table; a DCA plot. `triage.py` has `pick_thresholds()` (low cut at recall ≥ `RECALL_TARGET`, high cut at PPV ≥ 0.85 or the closest achievable), `triage(p, thresholds)` → tier + action, and `contributions(pipe, patient)`. `triage.py` imports only `config` and `data`.
  - Verify: the smoke test asserts the tier boundaries, that contributions sum to logit − intercept (±1e-6), and that importing `cardiotriage.triage` does not import `evaluation`, `models`, or `plots`.
  - Files: `cardiotriage/triage.py`, `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`, `tests/smoke_test.py`
  - Deps: 7

- [ ] **Task 9: Final test evaluation and clinical safety (nb §11–12)** (M)
  - Acceptance: the test set is used once; `evaluation.bootstrap_ci()` (1,000 resamples) gives recall, specificity, PPV, and NPV with 95% CIs; ROC, PR, confusion matrix, and calibration curve via `plots`; FN table; 3 case studies using `triage.contributions()`; recall by sex and age band.
  - Verify: headless run is clean; the CI columns are printed; the smoke test checks `bootstrap_ci` returns lo ≤ point ≤ hi on synthetic data.
  - Files: `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `nb`, `tests/smoke_test.py`
  - Deps: 8

- [ ] **Task 10: External validation on Hungary (nb §13), STRETCH** (S)
  - Acceptance: the 10-feature model is trained on Cleveland and scored with `data.load_hungarian()`; AUC and recall are compared with the internal CV; the drop is interpreted honestly.
  - Verify: headless run is clean.
  - Files: `cardiotriage/evaluation.py` (a small helper only if needed), `nb`
  - Deps: 9. **Can be cut** if the schedule slips.

- [ ] **Task 11: Interpretation, limitations, model card, serialize (nb §14–16)** (S)
  - Acceptance: permutation importance + standardized coefficients; Limitations (T15–T17); impact; next steps; model card; `joblib.dump` to `artifacts/heart_disease_pipeline.joblib` of the pipeline, thresholds, features, metrics, and versions, plus `artifacts/metrics.json`.
  - Verify: the smoke test loads the joblib and scores 4 preset patients with the expected tiers.
  - Files: `nb`, `tests/smoke_test.py`, `artifacts/*`
  - Deps: 9 (10 optional)

### Checkpoint D
- [ ] A full headless run gives 0 errors and 0 warnings; the joblib file is written; the smoke test passes. T1–T18 are each referenced in the notebook.

## Phase 4: Demo

- [ ] **Task 12: Streamlit `app.py`** (S)
  - Acceptance: imports only `config`, `data.validate`, and `triage` (plus streamlit/joblib); a form with clinical labels and ranges; 4 presets; risk %; tier with action; top contributions; what-if delta vs the previous run; disclaimer; `st.cache_resource`; invalid input shows `st.error` instead of crashing.
  - Verify: `streamlit run app.py`; click each preset and capture a screenshot; enter one invalid value; the smoke test asserts `app.py` has no import of `evaluation`, `models`, or `plots`.
  - Files: `app.py`, `tests/smoke_test.py`
  - Deps: 11

- [ ] **Task 13: Final verification** (XS)
  - Acceptance: a fresh headless run, the smoke test, and an app check all pass; the original notebook is unchanged (`git diff --stat` shows no change to it).
  - Verify: the commands in SPEC §7.
  - Files: none
  - Deps: 12

### Checkpoint E: Complete
- [ ] All SPEC §12 success criteria are met. Ready for review and commit (commit only on request).
