# Project brief — end to end

**Purpose of this document.** Every team member should be able to answer any question about any part of this project
from this one file. It follows `heart_disease_pitch_gold_standard.ipynb` section by section and quotes the real
numbers. Where a number appears here it also appears in the notebook, in `artifacts/metrics.json`, or in `README.md`.

For the exam-style version of the same material see [`QA_BANK.md`](QA_BANK.md); for the machine-learning theory see
[`ML_CONCEPTS.md`](ML_CONCEPTS.md).

---

## 1. What is this project?

A proof-of-concept clinical **decision-support** tool that prioritises the coronary angiography waiting list.

Given ten results a clinician already has for a patient who has **already been referred** for angiography, it estimates
the probability that the angiogram will find more than 50% narrowing in at least one major coronary vessel, and turns
that probability into one of three actions.

It is **not** a diagnostic tool, **not** a screening tool for the general public, and **not** a medical device. A
cardiologist makes every decision; the model only suggests an order.

**Deliverables in the repo**

| Thing | Where |
|---|---|
| The analysis and report | `heart_disease_pitch_gold_standard.ipynb` (16 sections, 141 cells) |
| All reusable logic | `cardiotriage/` — 7 modules, one job each |
| The live demo | `app.py` (Streamlit) |
| The trained model + its metrics | `artifacts/heart_disease_pipeline.joblib`, `artifacts/metrics.json` |
| Tests | `tests/` — `python -m tests.smoke_test`, `python -m tests.check_notebook` |
| The original course notebook, unchanged | `end-to-end-heart-disease-classification (1).ipynb` |

## 2. Which dataset, and why that one

**UCI Heart Disease, Cleveland subset.** Janosi, Steinbrunn, Pfisterer & Detrano (1989), Cleveland Clinic Foundation.
DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). File: `data/raw/processed.cleveland.data`.

303 patients × 13 clinical attributes + the angiographic outcome.

Two other files are in `data/raw/` and they are **not** training data:

| File | Role | Why |
|---|---|---|
| `processed.hungarian.data` | **External validation only** — 294 patients, Hungarian Institute of Cardiology, Budapest | Tests whether the model survives a move to a different hospital and a different era of practice |
| `heart-disease.csv` | **Evidence only** — the popular Kaggle copy | Its labels are inverted; we use it to prove that, never to train |

**Why this dataset.** It is the canonical small-tabular clinical benchmark, the label is an objective procedural
result rather than a clinician's opinion, and the UCI collection ships four hospitals — which gives us a free external
validation set, something almost no other course-sized dataset offers.

## 3. Goal and objective

**Objective (notebook §1).** From the 10 attributes retained for the pre-angiography decision, estimate
P(`disease` = 1) and place the patient in a triage tier.

**Target.** Flag at least **90% of patients who truly have disease** (recall ≥ 0.90) while keeping as many healthy
patients as possible out of the priority queue.

**Prediction task.** Binary classification. `disease` = 1 means > 50% narrowing of at least one major coronary vessel
on angiography, i.e. UCI `num > 0`.

**Who uses it and when.** A cardiologist or chest-pain-unit physician who already has the work-up — history, resting
ECG, recorded exercise-test measures — and must decide the order of the queue.

**Decision flow.** `available results (10 attributes) → data contract → model → risk % → triage tier → action`

| Tier | Rule | Action |
|---|---|---|
| High | risk ≥ 66.2% | Priority angiography |
| Medium | between the cut-offs | Cardiology review or further testing |
| Low | risk < 26.3% | Clinician review **and a documented safety-net plan** |

**Why recall is primary.** The two errors cost different amounts:

| Error | Consequence | Severity |
|---|---|---|
| False negative — sick patient placed Low | delayed angiography; risk of heart attack or death while waiting (Ontario: 1.4% of 8,030 waiting patients had such an event) | **severe — the error we minimise** |
| False positive — healthy patient placed High | takes a slot; ~8.2 major complications per 10,000 procedures; $3,312 | moderate — accepted to protect recall |

Accuracy is reported only as a secondary number, because "always predict no disease" already scores 54.1%.

## 4. Initial settings, and why each one

Notebook §2, implemented in `cardiotriage/config.py` — a constants-only module that imports nothing from the package,
so every other module can depend on it and there are no circular imports.

| Setting | Value | Why |
|---|---|---|
| `SEED` | 42 | One seed for the split, the CV folds, the randomised searches and every model with randomness. Re-running reproduces identical numbers; the original notebook set no seeds (trap T13). |
| `TEST_SIZE` | 0.20 | 61 patients held out. Big enough to be a real check, small enough to leave 242 for learning. |
| `CV_SPLITS`, `CV_REPEATS` | 5, 10 | 50 folds. 5-fold keeps ~194 patients per training fold; 10 repeats average away the luck of any one partition. |
| `RECALL_TARGET` | 0.90 | The clinical requirement: miss no more than 1 in 10 patients with disease. |
| `PPV_TARGET` | 0.85 | The High tier must be worth a priority slot — at least 85% of the patients in it should truly have disease. |
| `N_JOBS` | −1 | Parallel CV. Each model is held to a single thread so results stay deterministic. |
| Python | 3.11+ (verified on 3.13) | Versions pinned in `requirements.txt`, so a teammate gets identical numbers. |
| `ALLOW_MISSING` | `["ca","thal"]` | The only two columns UCI leaves blank; anything else missing is a data error. |

**Reproducibility is a rubric item, not housekeeping.** `python -m tests.check_notebook` verifies the executed
notebook has no errors, no warnings, and that every "Insight" sentence only quotes numbers the cells actually printed.

## 5. What the data looks like

`df.shape` → 303 rows. 13 feature columns + `num` (0–4) + `disease` (0/1).

### The 13 features

| # | Column | Meaning | Type | Codes / valid range |
|---|---|---|---|---|
| 1 | `age` | Age in years | numeric | 18–100 |
| 2 | `sex` | Sex | binary | 0 = female, 1 = male |
| 3 | `cp` | Chest pain type | nominal | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| 4 | `trestbps` | Resting blood pressure on admission (mm Hg) | numeric | 60–260 |
| 5 | `chol` | Serum cholesterol (mg/dl) | numeric | 80–700 |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | binary | 0 = ≤ 120, 1 = > 120 |
| 7 | `restecg` | Resting ECG result | nominal | 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved on the exercise test (bpm) | numeric | 50–250 |
| 9 | `exang` | Angina induced by exercise | binary | 0 = no, 1 = yes |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest (mm) | numeric | 0.0–10.0 |
| 11 | `slope` | Slope of the peak-exercise ST segment | nominal | 1 = upsloping, 2 = flat, 3 = downsloping |
| 12 | `ca` | Major vessels (0–3) coloured by fluoroscopy | ordinal | 0, 1, 2, 3 |
| 13 | `thal` | Thallium stress-test result | nominal | 3 = normal, 6 = fixed defect, 7 = reversible defect |

By type: **5 numeric, 1 ordinal, 4 nominal, 3 binary.**

**The 10 the shipped model uses:** all of the above **except `ca`, `thal` and `slope`** (see §9 below).

### The target

`num` is the UCI angiographic status, 0–4 (number of major vessels with > 50% narrowing). We binarise it the standard
way: `disease = num > 0`. Rationale: the clinical decision — does this patient need a priority angiogram — is binary,
and the 1–4 classes are too small individually (139 positives split four ways) to model reliably.

### Why those valid ranges

The ranges in `config.RANGES` are **physiological plausibility bounds, deliberately wider than the observed data**:

| Column | Observed min–median–max | Allowed | Reason for the bound |
|---|---|---|---|
| `age` | 29 – 56 – 77 | 18–100 | Adult cardiology population; 18 excludes paediatric records, 100 catches transposed digits |
| `trestbps` | 94 – 130 – 200 | 60–260 | Below 60 or above 260 mm Hg is a measurement error or an emergency, not a clinic reading |
| `chol` | 126 – 241 – 564 | 80–700 | Spans familial hypercholesterolaemia at the top; below 80 mg/dl is not a survivable serum value |
| `thalach` | 71 – 153 – 202 | 50–250 | Physiological heart-rate limits on a treadmill |
| `oldpeak` | 0.0 – 0.8 – 6.2 | 0.0–10.0 | ST depression is non-negative by definition; 10 mm is far beyond any real tracing |

**Why we chose it this way.** A contract that merely re-states the observed min and max would reject any new patient
who is slightly unusual — useless in a hospital. A contract this wide rejects only values that are *impossible*, which
is exactly the class of error that matters: typos, unit mix-ups, and placeholder codes. It is what catches the Kaggle
file, and in the demo it is what refuses to score a 10-year-old. Categorical columns are validated against the
official UCI code sets in `config.LABELS`, which is how `ca = 4` and `thal = 0` are caught as disguised missing values.

## 6. QA and cleaning — what we found

Notebook §3.2 and §4. Four findings.

**(a) The popular Kaggle copy is not the same data.** We matched all 303 rows back to UCI on age, sex, blood pressure,
cholesterol, heart rate and ST depression — 303 of 303 matched. Then:

| Problem | Evidence |
|---|---|
| **Labels inverted** | Kaggle `target` = 1 ↔ UCI no disease (165 rows); `target` = 0 ↔ UCI disease (138 rows). Zero rows disagree. |
| **Four categoricals re-coded** | `cp` 0↔4, 1↔2, 2↔3, 3↔1; `restecg`, `slope`, `thal` similarly permuted. The recode reproduces UCI exactly, so it is systematic, not noise. |
| **Missing values disguised** | `ca = 4` (5 patients) and `thal = 0` (2 patients) are UCI's `?`, presented as real codes |
| **One patient lost, one duplicated** | Patient 184 — a 60-year-old woman with disease — is absent; the Kaggle file contains 1 exact duplicate row |

Any result trained on that file has its predictions reversed. This is the project's most memorable finding.

**(b) Missing values in UCI.** 6 cells of 3,939 (0.15%) — `ca` 4, `thal` 2. Six patients (2.0%) have at least one.
Decision: **keep the patients, impute inside the pipeline** (median for numeric, most-frequent for categorical), so the
imputer is fitted on each training fold only and never sees the validation fold. Dropping 6 patients from 303 would
cost more than it buys.

**(c) Duplicates.** 0 exact duplicates and 0 feature-identical rows in UCI. `deduplicate()` still runs before the
split, as a guard against future data.

**(d) The contract passes.** All 303 patients pass; the Kaggle file is rejected with 355 invalid codes (143 `cp`, 186 `thal`, 21 `slope`, 5 `ca`).

### Is it balanced?

**Nearly.** 139 of 303 patients have disease — **45.9%** — against 164 (54.1%) without.

So no resampling (no SMOTE, no class weighting as a default): the imbalance is not severe enough to distort learning.
What we do instead is treat the **decision threshold** as the place where the error trade-off is made, because the
asymmetry is in the *costs*, not in the class counts.

Two consequences that drive the rest of the project:
1. **Accuracy is a trap.** "Always predict no disease" = 54.1% accuracy, 0% recall. We optimise recall.
2. **The default 0.5 threshold is arbitrary.** At 0.5 the selected model's recall is 0.784 — well below target. We
   choose the cut-offs ourselves from out-of-fold predictions (§10).

**Next step after QA:** exploratory analysis, to decide what to encode, what to engineer, what to check for fairness,
and — as it turned out — what to throw away.

## 7. EDA — what we did and what we found

Notebook §5, **seven sub-sections, nine charts**, each with a stated hypothesis before the chart and a numbered
insight after it. All plotting lives in `cardiotriage/plots.py` so notebook cells stay one line long.

| § | What we looked at | Finding |
|---|---|---|
| 5.1 | Sex and age | Men **55.3%** with disease (114/206) vs women **25.8%** (25/97). Only 97 of 303 patients are women. Mean age 56.6 with disease vs 52.6 without. |
| 5.2 | Chest pain type | **The paradox:** `asymptomatic` has the *highest* disease rate, **72.9%** (105/144), vs typical angina 30.4%, non-anginal 20.9%, atypical 18.0%. Explanation: a referred population — asymptomatic patients here were referred for some other reason. |
| 5.3 | Age vs max heart rate | Patients with disease reach 139.3 bpm vs 158.4; **47.5%** fall below 85% of their age-predicted maximum vs 15.9%. |
| 5.4 | Exercise stress test | Exercise angina: **76.8%** vs 30.9%. ST depression (`oldpeak`) mean 1.57 mm vs 0.59; 59% vs 25% above 1 mm. ST slope: flat 65.0%, downsloping 57.1%, upsloping 25.4%. |
| 5.5 | Imaging | `ca`: 0 vessels **26.1%** → 1 vessel 67.7% → 2 vessels 81.6% → 3 vessels **85.0%**. `thal`: normal 22.3%, fixed defect 66.7%, reversible defect 76.1%. A clean monotone gradient. |
| 5.6 | Univariate strength | Correlation with disease: `ca` **0.46**, `exang` 0.43, `oldpeak` 0.42, `thalach` −0.42, `sex` 0.28, `age` 0.22, `trestbps` 0.15, `chol` 0.09, `fbs` 0.03. |
| 5.7 | The weak classics | Blood pressure: medians **130 in both groups**, Mann-Whitney p = 0.026. Cholesterol: 249 vs 234.5, p = 0.035. Max HR p = 1.9e-13; ST depression p = 6.8e-13. **Statistically significant ≠ clinically useful.** |

### What we did *with* the EDA (§5.8) — every design choice traces to a finding

| Finding | Design choice |
|---|---|
| Imaging is the strongest signal — and `ca` may come from the target procedure | **The pivot:** ship the 10-feature model; keep the 13-feature one as retrospective comparison (§5.9, §13) |
| Classes nearly balanced (45.9%) | **No resampling.** Handle the asymmetry with the decision threshold instead |
| Women under-represented, half the disease rate | **Check recall by sex** before claiming anything; warn in the app |
| The asymptomatic paradox — category order carries no meaning | **One-hot encode** `cp`, `restecg`, `slope`, `thal` (fixes trap T9) |
| Cholesterol in the hundreds, ST depression 0–6 | **Standardise** inside the pipeline — KNN uses distances, regularised LR needs comparable scales (trap T8) |
| BP and cholesterol weak alone | Keep them, but **test removal** by drop-column CV (§9.5) |
| All values physiologically plausible | **Keep outliers.** No trimming |

### §5.9 — the circularity check (the single most important slide)

The target is the angiographic result. `ca` is a fluoroscopy finding that **may be read during that procedure**; UCI
has no timestamps, so earlier availability is unproven. `thal` is imaging with unverified timing. `slope` is exercise
ECG, dropped for a different reason: the external hospital recorded 65% of it as missing.

| | Retrospective 13-feature | **Main 10-feature (shipped)** |
|---|---|---|
| Cleveland CV ROC-AUC | 0.900 ± 0.035 | **0.844 ± 0.045** |
| External (Hungary) | **Not evaluable** — 99% of `ca` missing there | **ROC-AUC 0.877, recall 0.887** |
| Usable before the target angiogram | unproven | yes |

### Feature engineering (§7) — and why we dropped it

We built four domain features in `cardiotriage/features.py`: `pct_max_hr` (achieved HR ÷ 220 − age),
`st_depression_gt1` (ST depression > 1 mm), `hypertension` (BP ≥ 140), `high_chol` (cholesterol ≥ 240).

Ablation on the same 50 folds:

| Model | without | with | change | corrected p |
|---|---|---|---|---|
| Logistic Regression | 0.894 | 0.890 | **−0.004** | 0.13 |
| Random Forest | 0.889 | 0.880 | **−0.008** | 0.15 |

**Decision: `USE_FE = False`.** The models already receive age, heart rate, BP, cholesterol and ST depression as raw
values; a fixed clinical cut-point adds no information a model can't learn, and when two options tie the simpler one
wins. Reporting a *negative* feature-engineering result is itself a finding — it shows the ablation was real.

## 8. Split and validation design

Notebook §6, implemented in `cardiotriage/evaluation.py`.

1. **Lock a stratified 20% test set** → 242 train / 61 test, both 45.9% positive. Opened exactly **once**, in §11.
2. **Compare and tune on the 242 only**, with `RepeatedStratifiedKFold(5 splits × 10 repeats, seed 42)` = **50 folds**,
   each training on ~194 patients and validating on ~48. Every model sees the *identical* 50 folds.
3. **Report mean ± SD** across folds, and test whether a difference is real with the **corrected resampled t-test**
   (Nadeau & Bengio 2003), because repeated folds share training data and the ordinary t-test is over-confident.
4. **All preprocessing inside the pipeline** — impute → scale → one-hot → model — so it is refitted on each training
   fold and never sees the validation fold.

**Why not a single train/test split?** With 61 test patients, one patient moves accuracy by 1.6 points. The original
course notebook tuned KNN *on* the test set (trap T5) and then ran "cross-validation" that included it (T7); its
reported numbers are therefore meaningless.

**Why stratified?** To keep 45.9% positives in every fold; with 139 positives, an unstratified fold could drift several
points and make folds non-comparable.

**Tuning uses different folds** (5 × 2, a different seed) from the 50 comparison folds, so the comparison is not scored
on the folds that chose the settings.

## 9. The five models

`cardiotriage/models.py` → `MODEL_ZOO`. One estimator per family, deliberately spanning four different inductive
biases plus a floor.

| Model | What it is / how it works | Pros | Cons | Role here |
|---|---|---|---|---|
| **Dummy (prior)** | Always predicts the majority class / base rate. No learning. | Establishes the floor; proves a metric is meaningful | Predicts nothing useful | The baseline. Its ROC-AUC is **0.500** — if a model can't beat it, nothing else matters |
| **Logistic Regression** | Weighted sum of inputs → sigmoid → probability. Fitted by maximising likelihood with an L2 penalty (`C` = inverse regularisation strength). Linear decision boundary in feature space. | Interpretable (each coefficient is a log-odds effect); naturally calibrated probabilities; very stable at small n; milliseconds to fit | Cannot represent interactions or non-linearity unless you add them by hand | **The shipped model.** Best CV ROC-AUC, exact per-patient contributions for the app's "why this risk?" panel |
| **KNN** | No training. To predict, find the *k* most similar patients in the training set and take a (distance-weighted) vote. | No assumptions about the shape of the boundary; trivially simple | Needs scaling or big-valued features dominate the distance; slow at predict time; no interpretable model | A **distance-based contrast** to the parametric and tree families. Tuned to k = 46, Manhattan distance |
| **Random Forest** | Hundreds of decision trees, each on a bootstrap sample of patients and a random subset of features at each split; predictions averaged. **Bagging** — reduces variance. | Captures interactions and non-linearity automatically; robust to outliers and scale; gives feature importances | Less interpretable; can overfit small data if depth isn't capped | The **bagging / non-linear** representative. Tuned to depth 5, min 9 samples per leaf |
| **XGBoost** | Gradient boosting: trees added one at a time, each fitted to the *residual errors* of the ensemble so far, with shrinkage (learning rate) and L2 regularisation. | Usually the strongest family on tabular data; handles missing values natively; many regularisation knobs | Many hyperparameters; can overfit at n = 242; needs care to stay calibrated | The **boosting** representative and the serious challenger — it finishes 0.003 behind |

**Why these five and not deep learning?** 303 patients of structured, heterogeneous clinical features is precisely the
regime where linear and tree models match or beat neural networks; the models must be interpretable for a clinician;
they must run in milliseconds; and their probabilities must be checkable for calibration. A neural network on 242
training rows would be all variance and no explanation.

### The pipeline every model sits in

```
raw 13 columns
  → [optional domain features]            # disabled: ablation showed no gain
  → ColumnTransformer:
       numeric   : median impute  → StandardScaler
       ordinal   : median impute → StandardScaler (ca stays one numeric column, in the numeric group)
       nominal   : most-frequent impute → OneHotEncoder
       binary    : most-frequent impute
  → model
```

Identical for all five families, so the comparison is fair. Built by `models.build_pipeline()`; `drop=[...]` is how
the 10-feature variant removes `ca`, `thal`, `slope` without duplicating any code.

## 10. Training, tuning and selection

### How we train

`RandomizedSearchCV` on the 242 training patients, **30 settings per model**, scored by ROC-AUC on 5-fold × 2-repeat
CV with its own seed. The search's own `best_estimator_` is used directly — the original notebook re-typed the best
`C` wrongly (4.89 instead of 0.204; trap T6).

**Search spaces** (`models.SEARCH_SPACES`)

| Model | Searched |
|---|---|
| Logistic Regression | `C` log-uniform 1e-3…1e2; `class_weight` {None, balanced} |
| KNN | `n_neighbors` 3…50; `weights` {uniform, distance}; `p` {1, 2} |
| Random Forest | `max_depth` {None,3,5,8}; `min_samples_leaf` 1…19; `max_features` {sqrt, 0.5}; `class_weight` |
| XGBoost | `n_estimators` 50…399; `max_depth` 2…5; `learning_rate` log-uniform 0.01…0.3; `subsample`, `colsample_bytree` 0.6…1.0; `min_child_weight` 1…9; `reg_lambda` log-uniform 0.1…10 |

**Why randomised rather than grid search?** With 7 XGBoost hyperparameters a grid is combinatorially impossible; random
search finds comparable optima in a fixed budget (Bergstra & Bengio 2012). We then *verified* the budget was enough
(§9.6): 100 settings instead of 30 added **0.000** to logistic regression and **0.003** to XGBoost.

### Results — tuning before/after, on the shared 50 folds

| Model | default ROC-AUC | tuned ROC-AUC | change |
|---|---|---|---|
| Logistic Regression | 0.894 | **0.900 ± 0.035** | +0.007 |
| XGBoost | 0.863 | 0.897 ± 0.034 | +0.034 |
| KNN | 0.867 | 0.896 ± 0.034 | +0.030 |
| Random Forest | 0.889 | 0.894 ± 0.038 | +0.005 |
| Dummy | 0.500 | — | — |

Tuning rescued the two models that needed it and did not change the ranking.

### Which model we selected, and why

**Tuned Logistic Regression, `C` = 0.216, `class_weight` = None, no domain features, no extra calibration.**

1. Best mean CV ROC-AUC (0.900).
2. The gap to XGBoost (0.897) is **not statistically detectable**: corrected t-test **p = 0.81**. The selection rule in
   `SPEC.md` is explicit — *if LR is within 1 SD of the best, ship LR*.
3. It is interpretable: exact per-patient log-odds contributions power the app's "why this risk?" panel.
4. It is already calibrated: Brier 0.1264 as-is vs 0.1275 with Platt scaling → **calibration step not added**.
5. At the clinically relevant operating point it is also the best (below).

**The 10-feature shipped model** is the same family, independently tuned on the ten shared columns (`C` = 0.985),
because that is the input set available before the angiogram.

## 11. How we evaluate

### Primary metric and why

**Recall (sensitivity) on the disease class**, with a target of ≥ 0.90.

Because the two errors have different costs (§3): a missed sick patient waits, and waiting is measurably dangerous.
Recall is the only metric that directly counts that error.

### The supporting metrics, and what each one is for

| Metric | What it answers | Why we need it |
|---|---|---|
| **Specificity** | of the healthy patients, how many did we keep out of the priority queue? | Recall alone is maximised by flagging everyone. Specificity is the constraint that makes recall meaningful |
| **ROC-AUC** | how well does the model *rank* patients, at every possible threshold? | Threshold-free, so it is the right metric for model *comparison* and for a queue-ordering problem |
| **PR-AUC** | ranking quality focused on the positive class | A second view that is more sensitive to the minority class |
| **PPV / precision** | of the patients we flagged High, how many really have disease? | Decides whether a High tier deserves a priority slot (target 0.85) |
| **NPV** | of the patients we placed Low, how many really are disease-free? | The safety number for deferral |
| **Brier score** | are the probabilities *honest*, not just correctly ordered? | A "73%" shown to a clinician must mean 73%. Brier = mean squared error of the probability |
| **Accuracy** | overall share correct | Reported only, never optimised — the 54.1% baseline shows why |

A **metric sanity check** (§6.1) computes recall, specificity, precision and accuracy by hand on a 10-patient toy
example and asserts the values, so nobody is reading a mislabelled number.

### The decision layer (§10)

A probability is not a decision. Both cut-offs are chosen from **out-of-fold predictions** on the training set (every
patient scored by a model that never saw them, 5-fold averaged over 10 repeats) — never the test set.

* **Low cut-off** = the highest threshold that still flags ≥ 90% of patients with disease.
* **High cut-off** = the lowest threshold at which ≥ 85% of the patients above it truly have disease.

Retrospective 13-feature model: low **0.287**, high **0.509**. Shipped 10-feature model: low **0.263**, high **0.662**.

Chosen vs default threshold, on out-of-fold predictions:

| Cut-off | recall | specificity | PPV | flagged/100 | **missed/100** |
|---|---|---|---|---|---|
| chosen 0.287 | **0.901** | 0.679 | 0.704 | 58.7 | **4.5** |
| default 0.5 | 0.784 | 0.885 | 0.853 | 42.1 | **9.9** |

Moving the threshold halves the number of sick patients missed. That is the single most valuable step in the project,
and it costs no modelling at all.

**Tiers on out-of-fold predictions** (242 patients): High 102 patients, 85.3% disease → priority angiography. Medium
40 patients, 32.5% → cardiology review. Low 100 patients, 11.0% → clinician review + safety-net plan.

**Decision-curve analysis** (§10.1): the model's net benefit beats both "angiogram for everyone" and "angiogram for
nobody" at every risk threshold from 0.05 to 0.80 — e.g. at 0.30, 0.340 vs 0.227 vs 0.
*This is what answers "is acting on the model actually worth it?"*

**Separate-fold validation of the whole rule** (§13.2) — the strongest evidence we have. Each of 5 outer folds retunes
`C` **and** re-chooses both cut-offs on inner out-of-fold predictions, then scores untouched outer patients:
**recall 0.901, specificity 0.542, High-tier PPV 0.857.** This is the number to quote, because it validates the
procedure, not just the model.

### The locked test set (§11, used once)

Retrospective 13-feature model on 61 patients:

| Metric | Point | 95% CI |
|---|---|---|
| Sick patients flagged | **28 of 28** | exact (Clopper–Pearson) 0.877–1.000 |
| ROC-AUC | 0.961 | 0.902–0.998 |
| Specificity | 0.667 | 0.500–0.829 |
| PPV | 0.718 | 0.579–0.854 |
| NPV | 1.000 | 1.000–1.000 |
| Brier | 0.084 | 0.049–0.128 |

Tiers on the test set: High 26 disease / 5 no disease; Medium 2 / 6; **Low 0 / 22**.
Reduced-input secondary check: 28/28, ROC-AUC 0.957, specificity 0.636.

**Say this out loud:** 28/28 on 61 patients has a lower confidence bound of 0.877. The cross-validated estimate is the
more stable number; the test set is a consistency check, not a triumph.

### External validation (§13)

The 10-feature model, trained on Cleveland, scored on **294 Budapest patients with no refitting**:
ROC-AUC **0.877** (0.828–0.918), recall **0.887** (0.820–0.944), specificity 0.628, Brier 0.131.

Recall is **below our 0.90 target.** We report that as a known gap, not a success.

### Clinical safety checks (§12)

* **Who do we miss?** 11 of 111 sick training patients land in Low out-of-fold (test set: 0). Their profile is
  consistent: mostly `ca` = 0, normal thallium, no exercise angina, ST depression ≈ 0 — patients who look clean on
  every recorded measure. 7 male / 4 female.
* **Subgroups.** Retrospective model: women recall **0.778** (14/18, CI 0.524–0.936) vs men 0.925 (86/93). Shipped
  10-feature model: women **0.833** (15/18, CI 0.586–0.964) vs men 0.914 (85/93). Age bands 0.857–0.917.
  The cut-off was chosen on these same patients, so this is **descriptive, not a fairness guarantee**.
* **Three patients up close.** Most certain (risk 0.993, driven by `ca`), borderline (0.296, Medium, truly diseased),
  hardest miss (0.063, Low, truly diseased) — each with its per-feature contributions.

## 12. Improvement: what we tried, what worked, what didn't

Notebook §9.3–9.6 and §15.3. This is the 5% tuning criterion, and the honest answers are the valuable ones.

| Check | Question | Result | Verdict |
|---|---|---|---|
| **Nested CV** (§9.3) | Is the tuned score optimistic because the same patients chose and scored the settings? | LR nested **0.909** vs tuned 0.900 → optimism **−0.008**. XGBoost 0.905 vs 0.897 | **No optimism.** Our number is if anything conservative |
| **Learning curve** (§9.4) | Would more data help? | Validation ROC-AUC 0.861 → 0.874 → 0.889 → 0.896 → **0.900** at 38/77/115/154/193 patients. Still rising | **Yes — more patients is the strongest lever** |
| **Drop-column selection** (§9.5) | Does every feature earn its place? | Only `ca` matters (−0.030, p = 0.016). A 7-feature variant scored **+0.007, p = 0.47** | **Keep all 13.** Not reliably better; selection on 242 patients is noisy |
| **Search budget** (§9.6) | Was 30 settings enough? | 100 settings: LR +0.000, XGBoost +0.003 | **Yes.** More search is not the lever |
| **Calibration** (§9.1) | Do we need Platt scaling? | Brier 0.1264 raw vs 0.1275 calibrated | **No.** LR was already calibrated |
| **Domain features** (§7) | Do clinical cut-point features help? | −0.004 (p = 0.13) and −0.008 (p = 0.15) | **No.** Dropped |
| **Shared failures** (§9.2) | Is a different algorithm the answer? | **6 sick patients are missed by all four tuned models** | **No.** They need new *inputs*, not a new algorithm |

**Roadmap (§15.3).** 1) Pool the other UCI hospitals and modern records with missing-indicators. 2) Add inputs for the
6 shared misses — troponin, symptom duration, coronary calcium score, the full stress-test trace. 3) Bayesian search
(Optuna) for XGBoost, scored by nested CV, once more patients are pooled. 4) Stacking LR + XGBoost, or XGBoost with
monotonic constraints — adopted only if it beats the shipped model *at the recall-0.90 operating point*. 5) Cost-weighted
cut-offs from clinician-agreed costs instead of a fixed 90% target. 6) Close the sex gap: collect more sick women,
review every missed case, re-check subgroup recall after every change.
**Path to deployment:** prospective silent pilot → per-hospital re-calibration and drift monitoring → EHR integration
with the typed input contract.

## 13. What the model learned (§14)

**Permutation importance** (drop in training ROC-AUC when one feature is shuffled):
`ca` 0.067, `thal` 0.037, `cp` 0.028, `sex` 0.016, `slope` 0.010, `thalach` 0.009, `trestbps` 0.005, `oldpeak` 0.004,
`exang` 0.003, `restecg` 0.003, `chol` 0.001, `age` 0.001, `fbs` 0.000.

**Standardised coefficients** (log-odds): `ca` +0.859, `sex` +0.767, `cp_4` (asymptomatic) +0.708,
`thal_7` (reversible defect) +0.645, `thal_3` (normal) −0.585, `exang` +0.455, `slope_2` (flat) +0.395,
`cp_3` −0.371, `thalach` −0.342, `slope_1` (upsloping) −0.323.

Both views agree, and both agree with cardiology: more diseased vessels, a reversible perfusion defect, male sex,
exercise-induced angina and a flat ST segment raise risk; a higher achievable heart rate lowers it. Unlike the original
notebook, which ranked raw coefficients of unscaled features (trap T10), these are comparable across features.

## 14. Impact — and the honest labels on it (§15.2)

**Simulated** figures on the retrospective model. The dataset records no arrival order, no waiting time and no decision
to withhold a test.

| Simulated measure | Value |
|---|---|
| Sick patients missed per 100 — chosen cut-off vs default 0.5 | **4.5 vs 9.9** |
| Disease rate in High tier vs all referred | 85.3% vs 45.9% |
| Share of sick patients in the first 40% of risk-ordered slots | **76.6% vs 40%** (proportional-arrival benchmark) |
| Per 1,000 referred: sick in High / left in Low | 360 / **45.5** |
| Healthy patients spared a priority angiogram per 1,000 | 368 (≈0.3 major complications avoided at 8.2/10,000) |
| Net benefit at a 0.3 risk threshold — model vs angiogram-for-all | 0.340 vs 0.227 |
| Payment difference per 1,000 referred (US Medicare 2026 OPPS, $3,312) | **$1,368,518** for 413 fewer angiograms |
| Cost per sick patient found | $7,220 → **$4,703** |

The money figures are **payments, not hospital costs**, exclude the physician fee, and assume Low-tier patients receive
no angiogram — which is not what the Low tier recommends. No Thai or UK public price per diagnostic angiogram was
found, so no saving transfers. Quote the "45.5 sick patients left in Low" alongside every favourable number.

## 15. Limitations (§15.1)

1. **Input timing (T15).** `ca` and `thal` are the strongest inputs and come late in the work-up; `ca` is close to how
   the diagnosis itself is made. Hence the 10-feature shipped model. This assumption needs checking in a modern,
   timestamped cohort.
2. **Spectrum and selection bias (T16).** Patients already referred for angiography, at one hospital, in the 1980s.
   Disease is far more common here (45.9%) than in the general population. Not a screening tool.
3. **Small sample.** 303 patients, 61 in the test set. Test CIs are wide; the CV estimate is the stable one.
4. **Under-served subgroup.** Recall is lower for women (0.833 shipped, 0.778 retrospective). Small groups, wide
   intervals, but the gap points the same way as the missed-diagnosis literature (women under 55 had 6.7× the odds of
   being sent home — Pope, *NEJM* 2000).
5. **Calibration (T17).** LR was already calibrated, so we checked rather than fixed. A new hospital needs re-calibration.
6. **External recall below target.** Hungary 0.887 < 0.90.
7. **Simulations, not outcomes.** Queue gain and money are simulated; no clinical outcome was observed.

## 16. The 18 traps from the original notebook

The original course notebook (`end-to-end-heart-disease-classification (1).ipynb`, kept unchanged) contains 18 defects
we catalogue and fix — labels inverted (T1), codes re-mapped (T2), missing values as fake codes (T3), duplicate
patient (T4), KNN tuned on the test set (T5), wrong `C` re-typed (T6), CV that included the test set (T7), no scaling
(T8), nominal codes as numbers (T9), importance from raw coefficients (T10), mislabelled confusion axes (T11), broken
legend (T12), no seeds (T13), a 95%-accuracy goal (T14), late-stage features (T15), referral bias (T16), uncalibrated
risk (T17), no environment file (T18). The full mapping of trap → fix is notebook §15.4.

Knowing this table is the fastest way to answer "what did you do differently?"

## 17. How to run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb
.venv/Scripts/python -m tests.check_notebook    # must print "clean"
.venv/Scripts/python -m tests.smoke_test        # all tests must pass
.venv/Scripts/streamlit run app.py              # the demo
```

Always run from the repo root so `import cardiotriage` resolves. On Linux/macOS use `.venv/bin/` instead of
`.venv/Scripts/`.

## 18. Code architecture — who may import whom

```
config ◄── data
config ◄── features ◄── models ◄── evaluation     (evaluation also uses data and triage)
config ◄── plots
config ◄── triage
```

Imports go one way only, so there are no cycles. `app.py` may import **only** `config`, `data` and `triage` — never
`models`, `evaluation` or `plots` — and the smoke test enforces it, which is what keeps the demo's startup fast and
guarantees the app and the training code share one source of truth. Details in `STRUCTURE.md`.
