# Implementation Plan: Heart Disease Triage PoC

## Overview
Build `heart_disease_pitch_gold_standard.ipynb` (structured like Assignment.md), `heart_disease_pipeline.joblib`, and a Streamlit `app.py` on top of the **UCI Cleveland original**. The Kaggle CSV the old notebook used has inverted labels and re-mapped codes (proven: 303/303 rows matched). Scope is implementation and fixes only; the pitch deck, script, and team roles are out of scope. The full requirements are in `SPEC.md`.

## Architecture decisions
- **UCI `processed.cleveland.data` is the source of truth.** Fixing the Kaggle CSV by remapping would mean 6 hand-written maps; using the original needs none. The Kaggle file stays only as QA evidence.
- **`heart.py` is a shared module.** Column lists, code labels, `validate()`, `build_pipeline()`, `contributions()`, and `triage()` live there. The notebook, app, and test import it, so the app cannot drift from the training code.
- **One sklearn `Pipeline` from raw columns to probability.** Imputation, scaling, and one-hot encoding are fitted inside CV, so nothing leaks, and the joblib file is self-contained.
- **The notebook is generated** by a builder script in the scratchpad (nbformat) and executed with nbconvert. The repo holds only the executed `.ipynb`; cells are edited section by section and re-executed after each task.
- **Prefer LR when it is within 1 SD of the best.** That gives exact per-patient contributions (linear SHAP values without the `shap` dependency) and good calibration.
- **Assert-based `tests/smoke_test.py`**, so there is no pytest dependency.

## Task list
See `tasks/todo.md` (13 tasks, 4 checkpoints).

| Phase | Tasks | Checkpoint |
|---|---|---|
| P0 Foundation | 1 env and data · 2 `heart.py` data contract | Imports OK, smoke test (validate part) passes |
| P1 Story and QA | 3 nb §0–4 · 4 nb §5 EDA | Headless run is clean; T1/T2 proof printed; **human reviews the narrative** |
| P2 Modelling | 5 pipeline, feature engineering, CV design · 6 five-family comparison · 7 tuning and selection | Comparison table is stable across 2 runs |
| P3 Decision and evaluation | 8 thresholds, DCA, tiers · 9 test evaluation and safety · 10 Hungary external validation (stretch) · 11 interpretation, model card, joblib | 0 errors / 0 warnings; joblib written; smoke test passes |
| P4 Demo | 12 `app.py` · 13 final verification | All presets work in the browser |

## Risks and mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Recall ≥ 0.90 forces specificity very low | Med | The threshold comes from OOF; show the trade-off in a DCA plot and per-100 table; document rather than hide |
| "0 warnings" under new sklearn/xgboost versions (FutureWarning, convergence) | Med | Fix at the cause (`max_iter`, `solver`, explicit params); pin versions in `requirements.txt` |
| XGBoost on n=242 overfits or is unstable | Low | Shallow trees, tuning inside CV, report SD; LR is the likely winner anyway |
| A tree model wins by more than 1 SD, so SHAP is needed | Low | Ask before adding `shap` (SPEC boundary) |
| Hungary has different missingness (ca/thal ~all missing) | Low | Use a 10-feature model for that section only; mark it a stretch task that can be cut |
| Headless run is slow (5×10 CV × 5 models × search) | Low | `n_jobs=-1`, `n_iter` ≤ 30; target < 5 min |

## Open questions
None blocking.
