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

---

# Phases 5–9: close every non-Presentation rubric gap (audit of 2026-11-06)

Decisions (user, 2026-11-06): no legacy move of the original notebook; nested CV is in (it makes the tuned score valid); pooling Hungary into training stays a roadmap item; **do not commit**.
Every task ends with: headless run → `tests.check_notebook` clean → `tests.smoke_test` all pass. Cell numbers refer to the notebook before Phase 5 (0-indexed).

## Phase 5: Evidence and framing

- [x] **Task 14: Verify use-case statistics** (S) — Patel 2010 (37.6% of 398,978), Al-Hijji 2019 (8.2/10,000), Pope 2000 (19/889; women <55 OR 6.7 is for all acute ischaemia, wording fixed), WHO, CDC verified; no citable cost or waiting-time figure, so impact is in patients and slots; no `config` constant.: Patel et al. NEJM 2010 angiography yield; diagnostic-catheterisation complication rate; cost per procedure (currency + year); capacity/waiting figure if reliable; re-confirm WHO, CDC, Pope.
  - Acceptance: every figure has URL, year and the exact source number; unverifiable figures are dropped. Recorded in SPEC §6 (and a `config` cost constant only if a cost is verified).
  - Files: `SPEC.md` (`cardiotriage/config.py` only if a cost is verified)
- [x] **Task 15: Reframe nb cells [0], [1], [2], [5]** (S): problem-statement callout (Who/When/Predicts/Action/Success); two-layer stats table (why CVD matters / why queue order matters); bolded task sentence; UCI citation (doi:10.24432/C52P4X).
  - Acceptance: every statistic has a source and year; the callout quotes only numbers printed later.
  - Files: `nb`

### Checkpoint F (Phase 5)
- [x] Headless run clean (96 s); smoke 62/62; all 34 code-cell outputs identical.
- [x] Human review of the new medical wording (Section 0 and §10.3): done by the user, 2026-11-06.

## Phase 6: Model comparison and the math

- [x] **Task 16: PR-AUC in the CV tables** (XS): `"pr_auc": "average_precision"` in `evaluation.SCORING`; test first.
  - Files: `cardiotriage/evaluation.py`, `tests/test_evaluation.py`, `nb`
- [x] **Task 17: One-line math per model in nb cell [58]** (XS), linked to the tuned parameter (C, k, max_depth, learning_rate).
  - Files: `nb`
- [x] **Task 18: Comparison at the clinical cut-off (new §9.2)** (M) — at recall 0.901: specificity LR 0.679, XGB 0.672, KNN 0.603, RF 0.565; 6 sick patients missed by all 4 models (6 of LR's 11 misses). Learning curve renumbered §9.3.: `evaluation.operating_points()` (each tuned model at its own OOF recall-0.90 cut-off: threshold, recall, specificity, PPV, missed per 100, OOF PR-AUC), `plots.confusion_grid()`, miss-overlap error analysis; tests first.
  - Acceptance: Insight quotes the printed specificities and the overlap count; the selected model is still justified, or the text says it is not.
  - Files: `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `tests/test_evaluation.py`, `tests/test_plots.py`, `nb`

### Checkpoint G (Phase 6)
- [x] Headless run clean (95 s); smoke 64/64; 32/34 original code cells identical, the other 2 differ only by the new `pr_auc` column.

## Phase 7: Honest tuning and an accuracy roadmap

- [x] **Task 19: Nested CV (new §9.3)** (M) — all 4 models would add ~117 s (RF alone 79 s), so per the acceptance rule it runs for LR + runner-up (XGBoost): nested 0.909 / 0.905 vs tuned 0.900 / 0.897, optimism −0.008 / −0.007 (none detectable). `tune()` now builds its search via `_search()`, which `nested_cv()` reuses. Learning curve renumbered §9.4.: `evaluation.nested_cv(name, X, y)` = outer 5-fold around `tune()`; table tuned vs nested ROC-AUC; test first.
  - Acceptance: Insight quotes the optimism; notebook runs ≤ 3 min, else only LR + runner-up (and the notebook says so).
  - Files: `cardiotriage/evaluation.py`, `tests/test_evaluation.py`, `nb`
- [x] **Task 20: Caveat in cell [71]; split §15.3 into "Improving accuracy" (each lever with its evidence) and "Path to deployment"** (S).
  - Files: `nb`

### Checkpoint H (Phase 7)
- [x] Headless run clean; runtime 78 s (≤ 3 min); smoke 65/65; all 35 previous code-cell outputs identical.

## Phase 8: Quantified real-world impact

- [x] **Task 21: Queue impact (new §10.3)** (M) — first 40% of slots serve 76.6% of sick patients (vs 40.0%); High tier 42.1% of patients, 78.4% of sick, disease rate 85.3% vs 45.9%; per 1,000 referred: 421.5 High (359.5 sick), 413.2 Low (45.5 sick). No cost column (none verified).: `evaluation.queue_gain()` (share of sick patients served in the first k% of slots, model order vs first-come-first-served), `plots.queue_gain()`, per-1,000-referred-patients table from OOF tiers (cost column only if Task 14 verified a cost); tests first (hand-computed 6-patient example; perfect ranker).
  - Files: `cardiotriage/evaluation.py`, `cardiotriage/plots.py`, `tests/test_evaluation.py`, `tests/test_plots.py`, `nb`
- [x] **Task 22: §15.2 impact with numbers** (S) — extended after review with a benefit-vs-harm view (no money figure): 368 healthy patients per 1,000 spared a priority angiogram (≈ 0.3 major complications avoided at 8.2 per 10,000), 45.5 sick patients left in Low, recall women 0.778 vs men 0.925.: one code cell collecting the printed headline numbers + a checked Insight.
  - Files: `nb`

### Checkpoint I (Phase 8)
- [x] Headless run clean (84 s); smoke 68/68; all earlier code-cell outputs identical.

## Phase 9: Demo, repo and documentation

- [x] **Task 23: `DEMO.md` run-sheet** (S) — the caution note for women shows on the Clear-low-risk preset (a woman); the missed preset is a man.: unseen-patient walkthrough (3 test presets, borderline `ca` 0→2 what-if, missed case + caution for women, age 10 rejected); every figure produced by `AppTest`.
- [x] **Task 24: `README.md`** (S): pitch, Kaggle finding, results from `metrics.json`, how to run, demo, repo map, disclaimer.
- [x] **Task 25: Remove the duplicate `import platform` in nb cell [105]** (XS). (Legacy move: declined.)
- [x] **Task 26: Documentation** (S): SPEC v3 (§4 functions, §12 criteria), STRUCTURE.md, PROJECT_LOG Phase 5–9 entry.

### Checkpoint J (Phase 9, final)
- [x] From a deleted `artifacts/`: headless run 81 s, checker clean, smoke 68/68 (incl. `AppTest` presets), second run identical in all 38 code cells, rebuilt artifacts byte-identical to the committed ones, original notebook untouched. Re-run on the final notebook (after the §15.2 benefit-vs-harm change): 80 s, checker clean, smoke 68/68, second run identical in all 38 code cells, artifacts identical to the committed ones.
