# Heart Disease Triage

Predicts the risk of significant coronary heart disease from 13 clinical measurements (UCI Cleveland data) and turns the risk into a **High / Medium / Low** tier, to help a cardiologist decide who gets confirmatory angiography first.

> **Decision support, not diagnosis.** A proof of concept trained on 1980s data from one hospital. It is not for screening the general population.

## Results

| | |
|---|---|
| Model | Logistic Regression (tuned), chosen because it is within 1 SD of the best model and fully explainable |
| Cross-validated ROC-AUC | 0.900 ± 0.035 (5×10 repeated stratified CV; XGBoost 0.897, p = 0.81) |
| Recall (out-of-fold) | 0.901, set by the tier cut-off to meet a 0.90 target (specificity 0.679) |
| Locked test set (61 patients) | Recall 28/28 (exact 95% CI 0.877–1.000), specificity 0.667, ROC-AUC 0.961 |
| External check (Hungary) | ROC-AUC 0.880, recall 0.906 (10 features) |

Known gap: recall is lower for women (0.778, 14 of 18) than for men (0.925). There are too few sick women to be sure, so it is flagged for monitoring.

## A finding worth knowing

The popular Kaggle copy of this dataset (`heart-disease.csv`) is not the original. Checked row by row against the UCI file, its **target is inverted** (`1` means *no* disease), four categorical columns are re-coded, it contains codes UCI does not (`ca=4`, `thal=0`), and it is missing one sick patient while duplicating a healthy one. This project trains on the original UCI data and keeps the Kaggle file only as evidence.

## Quick start

Needs Python 3.11. Run everything from the repo root.

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# run the notebook top to bottom; this also writes artifacts/
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb

.venv/Scripts/python -m tests.check_notebook   # executed notebook is clean and honest
.venv/Scripts/python -m tests.smoke_test       # all tests (add an area to run one: ... smoke_test triage)
.venv/Scripts/streamlit run app.py             # the demo
```

On macOS/Linux use `.venv/bin/` instead of `.venv/Scripts/`. A full notebook run takes about a minute.

While it runs, Windows may print many `joblib ... resource_tracker` tracebacks. They are harmless and never appear in the notebook; set `N_JOBS = 1` in `cardiotriage/config.py` to silence them (slower).

## The demo app

`app.py` loads the saved model and lets you enter a patient (or pick one of four examples). It shows the risk, the tier and recommended action, which factors raised or lowered the risk, and what changed since the previous patient or edit. Inputs outside clinical ranges are rejected with a clear message.

## How it is organised

All logic lives in the `cardiotriage/` package, one file per job. The notebook tells the story and calls the package; the app and tests use the same code, so the demo cannot disagree with the trained model.

| Path | What it is |
|---|---|
| `cardiotriage/` | `config`, `data`, `features`, `models`, `evaluation`, `triage`, `plots` |
| `heart_disease_pitch_gold_standard.ipynb` | The analysis: data checks, modelling, evaluation, interpretation, limitations |
| `app.py` | Streamlit demo (imports only `config`, `data`, `triage`) |
| `tests/` | `test_<area>.py` files plus `smoke_test.py` (runner) and `check_notebook.py` |
| `artifacts/` | Saved model (`.joblib`) and `metrics.json`, written by the notebook |
| `data/raw/` | UCI Cleveland (source of truth), UCI Hungary (external check), Kaggle copy (evidence only) |

Details: [STRUCTURE.md](STRUCTURE.md) (file map and import rules), [SPEC.md](SPEC.md) (what we built and why), [PROJECT_LOG.md](PROJECT_LOG.md) (phase-by-phase record), [tasks/](tasks/) (plan and checklist).

## Method in brief

- Stratified 80/20 split; the test set is touched once, at the end. All tuning happens inside cross-validation.
- Five model families compared on identical folds, with a corrected resampled t-test (Nadeau–Bengio).
- Two tier cut-offs chosen from out-of-fold predictions: the lower one meets the recall target, the upper one targets a positive predictive value of 0.85.
- Bootstrap and exact confidence intervals, decision-curve analysis, permutation importance, and exact per-patient contributions (`coefficient × difference from the average patient`).

## Limitations

- One hospital, one era; patients were already referred for angiography.
- 303 patients: test-set intervals are wide, so trust the cross-validated figures more.
- Re-check calibration on any new hospital before showing risk as an exact probability.
- Not built, by decision: a deployed service, a pitch deck, and a production stack.

## Data sources

UCI Heart Disease repository (Cleveland and Hungarian files); see the notebook for the source and year of every statistic quoted.
