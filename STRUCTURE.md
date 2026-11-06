# Project Structure Guide

A map of the repo for team members. Read this first, then open the file you need.

> **Status:** this is the planned layout (see `SPEC.md` and `tasks/todo.md`). Files marked 🔜 are not written yet; they appear as the tasks are completed.

## What the project does

It predicts the risk of coronary heart disease from 13 clinical measurements (UCI Cleveland data) and turns the risk into a triage tier: high, medium or low. A cardiologist uses the tier to decide who gets the confirmatory angiography first. It is decision support, not a diagnosis.

## The big idea

All the real code lives in one Python package, `cardiotriage/`, with one file per job. Three things use that same code:

| Consumer | What it does with the package |
|---|---|
| **The notebook** | Tells the story (markdown) and calls package functions to produce the results and charts |
| **`app.py`** | The live demo; loads the saved model and uses the same validation and triage code |
| **`tests/test_*.py`** | Check that the package works (one file per module; `tests/smoke_test.py` runs them) |

Because everyone imports the same functions, the demo cannot disagree with the model we trained.

## Folder map

```
heart-disease-prediction/
├── cardiotriage/                  The package: ALL logic lives here
│   ├── __init__.py                Empty on purpose (keeps the app's startup light)
│   ├── config.py                  Constants only: paths, SEED, feature groups, code→label maps, recall target
│   ├── data.py                    Load data, clean it, check it (the "data contract")
│   ├── features.py                Domain features and the preprocessing step
│   ├── models.py                  The five models, their tuning ranges, and the full pipeline
│   ├── evaluation.py              Split, cross-validation, metrics, significance test, tuning, thresholds, confidence intervals
│   ├── triage.py                  Risk → tier + action, and "why this patient scored high"
│   └── plots.py                   Every chart
│
├── heart_disease_pitch_gold_standard.ipynb   The story: markdown + short calls into the package (sections 0–16 done)
├── app.py                          Streamlit demo (run: .venv/Scripts/streamlit run app.py)
├── tests/smoke_test.py             Runner: runs every tests/test_*.py (or only the areas you name)
├── tests/test_<area>.py            The checks, one file per area: data, imports, models, evaluation,
│                                   triage, plots, notebook, app (shared fixtures in tests/helpers.py)
├── tests/check_notebook.py         Checks the executed notebook: no errors/warnings, Insights only quote printed numbers
├── data/raw/                       The three input files (see below)
├── artifacts/                      Trained model (.joblib) and metrics.json, written by the notebook's last section
├── requirements.txt                Pinned library versions (Python 3.11)
├── SPEC.md                         What we are building and why
├── STRUCTURE.md                    This file
├── tasks/plan.md, tasks/todo.md    The implementation plan and task checklist
└── end-to-end-heart-disease-classification (1).ipynb   The original notebook (do not edit)
```

## What each module does

| File | One-line job | Typical contents |
|---|---|---|
| `config.py` | Single place for constants | `SEED = 42`, file paths, lists of numeric/categorical columns, the official meaning of each UCI code, the recall target |
| `data.py` | Get clean, trustworthy data | `load_cleveland()`, `load_hungarian()`, `load_kaggle()`, `validate()` (rejects impossible values), `kaggle_vs_uci()` (proves the Kaggle labels are flipped) |
| `features.py` | Turn raw columns into model inputs | Domain features such as % of age-predicted max heart rate and the hypertension flag; scaling and one-hot encoding |
| `models.py` | Define what we train | Dummy, Logistic Regression, KNN, Random Forest and XGBoost; tuning ranges; `build_pipeline()` |
| `evaluation.py` | Measure how good a model is | Repeated cross-validation, the comparison table, the significance test, threshold sweep, decision-curve analysis, bootstrap confidence intervals |
| `triage.py` | Use the model on a patient | Pick the cut-offs, map a probability to a tier and action, list the factors behind a score |
| `plots.py` | Draw the charts | EDA charts and evaluation charts, so notebook cells stay short |

## Who may import whom

Imports go one way only. A file may import the files to its left, never to its right, so there are no circular imports.

```
config ◄── data
config ◄── features ◄── models ◄── evaluation   (evaluation also uses data)
config ◄── plots
config ◄── triage
```

Rules to keep this working:
1. `config.py` imports nothing from the package.
2. `features.py` imports only `config`. The saved model contains objects from this file, and the app has to load them, so it must stay light.
3. `triage.py` imports only `config` (it is allowed `data`, but does not need it).
4. **`app.py` imports only `config`, `data` and `triage`.** It must never import `models`, `evaluation` or `plots`; the smoke test checks this.
5. Keep `cardiotriage/__init__.py` empty.
6. Write feature functions at module level in `features.py`. Do not use `lambda`, because the saved model would not load in the app.

## Where does my change go?

| I want to… | Edit |
|---|---|
| Add a new valid-range check or a missing-value rule | `data.py` |
| Change a constant (seed, recall target, column group) | `config.py` |
| Add or change a clinical feature | `features.py` |
| Add a model or change its tuning range | `models.py` |
| Add a metric or an evaluation method | `evaluation.py` |
| Change the tier cut-offs or the wording of actions | `triage.py` |
| Add or restyle a chart | `plots.py` |
| Change the demo's look or inputs | `app.py` |
| Change the narrative, insights or markdown | the notebook |

Put logic in the package and keep notebook cells short: a cell should be a call plus its output, not a 40-line function.

## Data files (`data/raw/`)

| File | Role |
|---|---|
| `processed.cleveland.data` | **Source of truth**: the original UCI Cleveland data (303 patients) |
| `processed.hungarian.data` | UCI Hungary (294 patients); used only for external validation |
| `heart-disease.csv` | The Kaggle copy the original notebook used. **Its labels are inverted** (`target=1` means no disease) and its category codes are re-mapped. We keep it only as evidence in the notebook; never train on it. |

## How to run (from the repo root)

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# build everything: runs the notebook top to bottom and writes artifacts/
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb

.venv/Scripts/python -m tests.check_notebook # is the executed notebook clean and honest?
.venv/Scripts/python -m tests.smoke_test     # all quick checks (run from the repo root)
.venv/Scripts/python -m tests.smoke_test triage data   # only tests/test_triage.py and tests/test_data.py
.venv/Scripts/streamlit run app.py           # the demo
```

Always run commands from the repo root, so `import cardiotriage` works.

## Working in the notebook

Put this in the first code cell so edits to a module show up without restarting:

```python
%load_ext autoreload
%autoreload 2
```

If something looks stale, restart the kernel and run all cells.

**Expected console noise.** When you run the notebook with `jupyter nbconvert --execute` on Windows, the terminal prints many `KeyError ... joblib_memmapping_folder` tracebacks from `resource_tracker.py`, plus two Jupyter warnings (zmq event loop, kernel over TCP). They come from parallel workers inside Jupyter, never appear inside the notebook, and do not change any result. The real check is `python -m tests.check_notebook`, which must say "clean". If you want a silent console, set `N_JOBS = 1` in `cardiotriage/config.py` (the run gets slower).

## Team rules

- Never edit the original notebook (`end-to-end-heart-disease-classification (1).ipynb`).
- The test set is used once, at the end. All tuning happens inside cross-validation.
- Every statistic in the notebook needs a source and a year.
- Do not silence warnings to get a clean run; fix their cause.
- The app is labelled "decision support, not diagnosis". Do not change that.
