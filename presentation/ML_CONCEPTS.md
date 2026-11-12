# AI / ML concepts in this project

Written for the Applied AI and Machine Learning course. Every concept is paired with **where it appears in this
project** and **the actual number it produced**, so you can answer a theory question and a project question with the
same sentence.

---

## 1. What kind of learning problem is this?

**Supervised learning → classification → binary classification.**

* *Supervised*: every training patient comes with the right answer (the angiographic result).
* *Classification*, not regression: the output is a class, not a quantity.
* *Binary*: two classes — disease (1) / no disease (0).

We actually predict a **probability** and then apply a decision rule, which makes it *probabilistic classification*.
That distinction is the whole point of §10: the model outputs P(disease), and a separate, clinically-chosen threshold
turns it into an action. A classifier that only outputs 0/1 cannot be re-tuned for a different cost of error.

**Why not other paradigms?** Unsupervised (clustering) — we have labels, throwing them away wastes information.
Deep learning — 303 structured rows; neural networks need far more data to beat regularised linear and tree models on
tabular features, and they can't give a clinician a per-feature reason.

## 2. The machine-learning lifecycle, as this project walks it

| Stage | Notebook § | What happens |
|---|---|---|
| Problem framing | §1 | Objective, who uses it, cost of each error, the metric that follows from those costs |
| Data acquisition & provenance | §3 | Load UCI; prove the Kaggle copy is a different, corrupted dataset |
| Data QA / data contract | §4 | Missing values, duplicates, valid ranges, class balance |
| Exploratory data analysis | §5 | 9 charts, 7 sub-sections; every design choice traced back to a finding |
| Split design | §6 | Lock the test set, build shared CV folds |
| Feature engineering | §7 | 4 domain features, ablated, **rejected** |
| Model comparison | §8 | 5 families on identical folds |
| Tuning & selection | §9 | Randomised search inside CV, nested CV, learning curve, feature selection, budget check |
| Decision layer | §10 | Probability → threshold → tier → action; decision-curve analysis |
| Final evaluation | §11 | Locked test set, opened once |
| Error & fairness analysis | §12 | Who we miss, by sex and age; three case studies |
| External validation | §13 | Second hospital, no refitting |
| Interpretability | §14 | Permutation importance + standardised coefficients |
| Limitations, impact, model card | §15 | Honest gaps, simulated impact, roadmap |
| Serialisation / deployment | §16 | `joblib` artifact + metrics, consumed by the app |

Being able to name these stages in order is worth more in a viva than any single number.

## 3. Train / validation / test — and what each one is for

| Set | Size here | Used for | Used how often |
|---|---|---|---|
| **Training** | 242 patients | fitting parameters | every fold |
| **Validation** (the CV folds inside training) | ~48 per fold | choosing hyperparameters, choosing the threshold, comparing models | 50 folds |
| **Test** (locked) | 61 patients | one final unbiased estimate | **once** |

**The rule:** anything that *chooses* something is a training-time activity. The moment the test set influences a
choice, it stops being a test set and becomes a (very small) validation set. The original notebook broke this by
tuning KNN's `k` on the test score (trap T5) — its reported accuracy is not an estimate of anything.

## 4. Cross-validation

**k-fold CV**: split the training data into k parts; train on k−1 and validate on the held-out one; rotate; average.

**What we use:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` = **50 folds**.

* **Why stratified?** Each fold keeps the overall 45.9% positive rate. With only 139 positives, an unstratified fold
  could drift several points and the folds would no longer be comparable.
* **Why repeated?** One 5-fold partition is itself a random choice. Ten different partitions average out the luck of
  the split and give a usable standard deviation (±0.035 for the selected model).
* **Why 5 and not 10?** 5-fold leaves ~194 patients in each training fold. With 242 total, 10-fold would make the
  validation folds ~24 patients — too small for a stable ROC-AUC per fold.
* **Why the identical folds for every model?** So differences between models are not differences between splits. This
  is *paired* comparison and it is what makes the significance test valid.

**Other CV schemes and when you'd use them:** LOOCV (tiny datasets, very high variance), GroupKFold (repeated
measurements of the same subject — not needed here, patients are independent), TimeSeriesSplit (temporal data — not
applicable, UCI has no timestamps, which is exactly why we could not resolve the `ca` timing question).

### Nested cross-validation

If you tune on the same folds you then report, the reported score is optimistic — you have selected the settings that
happened to suit those folds. Nested CV fixes this: an **outer** loop holds out patients, and the **entire tuning
procedure** runs inside each outer training set.

Here: LR nested **0.909** vs tuned 0.900 → optimism **−0.008**; XGBoost 0.905 vs 0.897. Our tuned numbers are not
inflated. *That is the answer to "how do you know your 0.900 is real?"*

## 5. Data leakage — the five kinds and how each is blocked

Leakage is any situation where information that would not be available at prediction time influences the model.

| Kind | Example | Blocked by |
|---|---|---|
| **Preprocessing leakage** | fitting the scaler or imputer on all the data before splitting | All preprocessing lives **inside** the `Pipeline`, so it is refitted on each training fold |
| **Test-set leakage** | tuning on the test set | Test set opened once, in §11 |
| **Threshold leakage** | choosing the cut-off on the test set | Cut-offs chosen from **out-of-fold** predictions on training patients |
| **Target leakage** | a feature that is a proxy for, or derived from, the label | **The `ca` problem.** `ca` is a fluoroscopy finding possibly read during the target procedure → dropped from the shipped model |
| **Duplicate leakage** | the same patient in train and test | `deduplicate()` before splitting |

The `ca` case is the one worth memorising, because it is the kind of leakage no code check can find — you need to know
what the column *means* clinically. Cost of fixing it: ROC-AUC 0.900 → 0.844.

## 6. Preprocessing, and why each step is needed

| Step | Applied to | Why |
|---|---|---|
| **Median imputation** | numeric, ordinal | Keeps all 303 patients; median is robust to the skew in cholesterol. Fitted per fold |
| **Most-frequent imputation** | nominal, binary | A category has no mean |
| **StandardScaler** (z-score: (x − μ)/σ) | numeric | KNN measures Euclidean/Manhattan distance, so cholesterol in the hundreds would dominate ST depression in single digits. L2-regularised LR penalises coefficients, which is only fair if features share a scale. **Trees don't care** — but one pipeline for all five models keeps the comparison fair |
| **One-hot encoding** | `cp`, `restecg`, `slope`, `thal` | Their integers are *labels*, not quantities: `cp`=4 (asymptomatic) is not "twice" `cp`=2. Leaving them numeric imposes a false ordering (trap T9) |
| **`ca` left as a number** | ordinal | 0 < 1 < 2 < 3 vessels **is** a genuine ordering, and the disease rate rises monotonically (26.1 → 85.0%), so the ordering is real information worth keeping |
| **No outlier removal** | — | Every value is physiologically plausible; removing real patients to make a chart tidier is data manipulation |
| **No resampling** | — | 45.9/54.1 is near-balanced. Imbalance is handled at the *threshold*, not by inventing or discarding patients |

## 7. The five algorithms — the maths in one line each

### Dummy (prior)
Predicts the base rate. P(disease) = 139/303 for every patient. ROC-AUC exactly **0.500** by construction.
*Purpose*: without a floor, "0.90" means nothing.

### Logistic regression — *the shipped model*
```
z = β₀ + β₁x₁ + … + βₙxₙ
P(disease) = σ(z) = 1 / (1 + e^(−z))
```
Fitted by maximising the log-likelihood with an **L2 penalty**: minimise `−Σ log P(yᵢ) + (1/2C)·‖β‖²`.
`C` is the inverse regularisation strength — **small `C` = strong shrinkage**. We tuned `C` = 0.216, i.e. fairly heavy
shrinkage, which is what you expect at n = 242 with 20+ one-hot columns.

* Decision boundary: a hyperplane. Linear in the features, but the *probability* is non-linear in them.
* Each `βⱼ` is a log-odds effect: `exp(βⱼ)` is the odds ratio per unit (per SD, after scaling).
* **Pros**: interpretable, calibrated by construction (it optimises a proper scoring rule), stable at small n, fast.
* **Cons**: no interactions or curvature unless you add them manually.

### K-nearest neighbours
No parameters. To predict, find the k nearest training patients under a distance metric and take a vote, optionally
weighted by 1/distance. Tuned: **k = 46, p = 1 (Manhattan), distance-weighted**.
*k = 46 out of 242 is a very smooth model* — the search chose heavy smoothing, which is the KNN equivalent of strong
regularisation.
* **Pros**: no assumption about boundary shape. **Cons**: needs scaling, no interpretable model, prediction cost grows
  with the training set, and it degrades quickly as dimensions grow (curse of dimensionality).

### Random forest — bagging
Build B trees. Each tree sees a **bootstrap sample** of patients (sample n with replacement) and, at every split, a
**random subset of features** (`max_features='sqrt'`). Average the trees' probabilities.
Tuned: 300 trees, `max_depth=5`, `min_samples_leaf=9`.
* Why it works: individual deep trees are low-bias/high-variance; averaging B decorrelated trees cuts variance by
  roughly 1/B without raising bias. Feature subsampling is what decorrelates them.
* A tree splits to reduce **Gini impurity** `1 − Σpₖ²` (or entropy).
* **Pros**: interactions and non-linearity for free, scale-invariant, robust. **Cons**: less interpretable, and at
  n = 242 the depth must be capped or it memorises.

### XGBoost — gradient boosting
Trees added **sequentially**, each fitted to the gradient of the loss w.r.t. the current prediction — in practice, to
what the ensemble is still getting wrong:
```
F₀(x) = constant
Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)      where hₘ fits the residual/gradient
```
`η` is the learning rate (shrinkage). Regularised by tree depth, `min_child_weight`, `reg_lambda` (L2 on leaf weights),
and row/column subsampling. Tuned: 178 trees, depth 3, η = 0.037, subsample 0.894, colsample 0.876, λ = 2.246.
* **Bagging vs boosting:** bagging builds trees *independently* to reduce **variance**; boosting builds them
  *dependently* to reduce **bias**. That is the single most likely exam question about these two.
* **Pros**: usually state of the art on tabular data; native missing-value handling. **Cons**: many hyperparameters;
  overfits easily at small n; needs care to stay calibrated.

### The result that matters
After tuning, on identical folds: LR 0.900, XGBoost 0.897, KNN 0.896, RF 0.894. **The spread is 0.006.** At n = 242
the data, not the algorithm, is the binding constraint — which is exactly what the learning curve independently says.

## 8. Bias, variance, and regularisation

**Total error ≈ bias² + variance + irreducible noise.**

* **Bias** = error from the model being too simple to represent the truth (underfitting).
* **Variance** = error from the model being too sensitive to *which* patients it saw (overfitting).

Where you see it here:

| Evidence | Reading |
|---|---|
| Learning curve: training ROC-AUC 0.964 at 38 patients → 0.929 at 193; validation 0.861 → 0.900 | The gap closes as n grows → the early gap was **variance** |
| Tuned `C` = 0.216 (strong L2) chosen by search | The search *wanted* more shrinkage → variance was the binding problem |
| RF capped at depth 5, leaf ≥ 9; XGBoost at depth 3 | Same conclusion, reached independently by the tree families |
| A 7-feature model gained +0.007, p = 0.47 | Reducing capacity didn't reliably help → we're not badly overfitting either |

**Regularisation** is any constraint that trades a little bias for a lot less variance: L2 in LR, depth/leaf limits in
trees, large k in KNN, shrinkage + subsampling in boosting, and — in a sense — our refusal to add engineered features.

## 9. Hyperparameters vs parameters, and how to search

* **Parameters** are learned from data by fitting: the `β`s in LR, the split points in a tree.
* **Hyperparameters** are set before fitting and control *how* fitting happens: `C`, `k`, `max_depth`, `learning_rate`.

| Search strategy | How it works | When |
|---|---|---|
| **Grid search** | every combination on a grid | few hyperparameters, cheap model |
| **Randomised search** ← *used here* | sample N random settings from distributions | many hyperparameters; with 7 XGBoost knobs a grid is impossible. Bergstra & Bengio (2012): random search finds comparable optima in far fewer evaluations, because only a few hyperparameters usually matter |
| **Bayesian optimisation** (Optuna, TPE) | model the score surface and sample where improvement is likely | the roadmap item — worth it only once more data makes the signal bigger than the noise |

We used **30 settings scored by 5×2 CV**, and then *verified* the budget: 100 settings added **0.000** to LR and
**0.003** to XGBoost. Verifying your own search budget is unusual and judges notice it.

## 10. Evaluation metrics — definitions and this project's values

From the confusion matrix (**rows = truth, columns = the model's call** — the original notebook had these backwards,
trap T11):

|  | Flagged | Not flagged |
|---|---|---|
| **Truly disease** | TP | FN |
| **Truly no disease** | FP | TN |

| Metric | Formula | Plain meaning | Here (OOF, chosen cut-off) |
|---|---|---|---|
| **Recall / sensitivity / TPR** | TP/(TP+FN) | of the sick, how many did we catch? | **0.901** |
| **Specificity / TNR** | TN/(TN+FP) | of the healthy, how many did we spare? | 0.679 |
| **Precision / PPV** | TP/(TP+FP) | of those we flagged, how many were sick? | 0.704 |
| **NPV** | TN/(TN+FN) | of those we cleared, how many were truly clear? | 0.890 |
| **F1** | 2·P·R/(P+R) | harmonic mean of precision and recall | 0.807 (CV) |
| **Accuracy** | (TP+TN)/all | overall share correct | 0.832 (CV) — **reported, never optimised** |

**ROC curve**: TPR against FPR (= 1 − specificity) as the threshold sweeps 0→1. **ROC-AUC** = P(a random sick patient
is scored higher than a random healthy one). 0.5 = coin flip, 1.0 = perfect. **Threshold-free**, so it is the right
metric for comparing models and for a *ranking* problem like ordering a queue. Ours: 0.900 (13-feature CV),
0.844 (10-feature CV), 0.877 (external), 0.961 (test).

**PR curve / PR-AUC**: precision against recall. More informative than ROC when positives are rare, because it ignores
TN. Ours 0.895 (CV). With 45.9% positives both views agree here — in a 1%-prevalence screening problem they would not.

**Calibration**: are the probabilities *honest*? A model can rank perfectly (AUC 1.0) and still say "90%" for patients
who are only 60% likely. Measured by the **Brier score** = mean((p − y)²); lower is better, 0.25 is the uninformative
baseline at p = 0.5.
Ours: **0.126** CV, **0.084** test. Platt scaling made it *worse* (0.1275), so no calibration step was added — logistic
regression optimises log-loss, a proper scoring rule, so it tends to come out calibrated already.
**Why this matters clinically:** the app shows a number to a clinician. "73%" must mean 73%.

### Which metric to optimise vs which to report

Optimise **recall** (subject to specificity remaining usable), because the FN cost dominates. Report everything else.
Use **ROC-AUC** to *compare models* because it does not depend on the threshold you happen to pick. This split —
"the metric you optimise is not the metric you compare with" — is the single most examinable idea in the project.

## 11. Class imbalance, and why we did almost nothing about it

Options and our verdict:

| Technique | What it does | Verdict here |
|---|---|---|
| Oversampling / **SMOTE** | duplicates or synthesises minority examples | **Not used.** 45.9% is near-balanced; SMOTE on 139 clinical positives invents patients and distorts calibration |
| Undersampling | discards majority examples | **Not used.** We cannot afford to throw away 25 of 303 patients |
| `class_weight='balanced'` | re-weights the loss | **In the search space**, and the search chose `None` |
| **Threshold moving** ← *used* | keep the model, move the decision boundary | **This is the answer.** 0.5 → 0.287 took missed-per-100 from 9.9 to 4.5 |

The key insight: **the imbalance that matters is in the cost of errors, not in the count of classes** — and a threshold
is the correct tool for a cost asymmetry.

## 12. Comparing models properly — the corrected t-test

Fold scores from overlapping training sets are **not independent**, so the ordinary paired t-test underestimates the
variance and declares differences significant when they are not. The **corrected resampled t-test**
(Nadeau & Bengio 2003) inflates the variance by the ratio of validation-fold size to training-fold size.

Results here:
* LR vs RF (untuned): p = 0.70
* LR vs XGBoost (tuned): **p = 0.81**
* Domain features on/off: p = 0.13 and p = 0.15
* 7-feature vs 13-feature: p = 0.47
* Dropping `ca`: p = **0.016** ← the only significant result

So the honest statement is: *four tuned models are statistically indistinguishable on this data, and the only feature
whose removal we can detect is the one we removed on purpose.* Shipping the interpretable member of a statistical tie
is a defensible engineering decision, not a shortcut.

## 13. Interpretability

| Method | What it tells you | Caveat |
|---|---|---|
| **Permutation importance** | how much performance drops when one feature is shuffled → how much the model *relies* on it | Model-specific, not causal; correlated features share credit |
| **Standardised coefficients** | direction and size of each effect, per SD (numeric) or vs the other categories (one-hot) | Linear models only; raw coefficients of unscaled features are meaningless (trap T10) |
| **Per-patient contributions** | `βⱼ·(xⱼ − x̄ⱼ)` summed = logit(patient) − logit(average patient) | Exact for LR, which is *why* LR powers the app's explanation panel |
| **SPEC fallback** | if a tree model had won by > 1 SD, add SHAP | Not needed — LR won |

Note the honesty requirement: importance is **not** causality. "`ca` is important" means the model leans on it, not
that vessels cause disease.

## 14. Fairness and subgroup evaluation

A single average hides who the model fails. We report **recall per subgroup with exact (Clopper–Pearson) binomial
intervals**, because 18 sick women cannot support a normal approximation.

| Group | Shipped model | Retrospective model |
|---|---|---|
| Women | 0.833 (15/18), CI 0.586–0.964 | 0.778 (14/18), CI 0.524–0.936 |
| Men | 0.914 (85/93), CI 0.838–0.962 | 0.925 (86/93) |

**And we state the limitation of our own fairness check:** the threshold was chosen on these same patients, so this is
descriptive, not an independent fairness estimate. The intervals overlap, so we cannot prove a gap — but we cannot rule
one out either, and the direction matches the literature. Saying this is better than claiming the model is fair.

## 15. Diagnosing whether to get more data or a better model

**Learning curve** — performance vs training-set size:

| Patients | Train ROC-AUC | Validation ROC-AUC |
|---|---|---|
| 38 | 0.964 | 0.861 |
| 77 | 0.942 | 0.874 |
| 115 | 0.934 | 0.889 |
| 154 | 0.930 | 0.896 |
| 193 | 0.929 | **0.900** |

Reading it: the curves are **converging and the validation curve is still rising** at the full training set. That is
the signature of *variance limited by sample size*, not bias. Conclusion: **more data helps, a fancier model does not.**

The independent confirmation: **6 sick patients are missed by all four tuned models.** Same patients, four different
inductive biases. No fifth algorithm will find them — they need new *measurements* (troponin, symptom duration,
coronary calcium score).

## 16. From probability to decision — threshold selection and decision-curve analysis

A classifier's 0.5 default is a *convention*, not a clinical choice. We pick thresholds from a stated requirement:

* **Low cut-off** = highest threshold with recall ≥ 0.90 → 0.263 (shipped) / 0.287 (retrospective)
* **High cut-off** = lowest threshold with PPV ≥ 0.85 → 0.662 / 0.509

**Decision-curve analysis** (Vickers & Elkin 2006) answers a question no accuracy metric can: *is acting on this model
better than the two trivial policies?* Net benefit = (TP/n) − (FP/n)·(pₜ/(1−pₜ)), where pₜ is the risk threshold at
which a clinician would accept the test. Our model beats both "angiogram for everyone" and "angiogram for nobody" at
every threshold from 0.05 to 0.80 (at 0.30: 0.340 vs 0.227 vs 0).

## 17. Reproducibility and MLOps hygiene

| Practice | Here |
|---|---|
| One seed, one place | `config.SEED = 42`, used by the split, folds, searches and models |
| Pinned environment | `requirements.txt`, Python 3.11 |
| One source of truth for logic | `cardiotriage/` imported by the notebook, the app **and** the tests — the demo cannot disagree with training |
| Serialised artifact | `joblib` bundle: pipeline + thresholds + baseline + feature list + metrics + presets; the notebook asserts the reloaded pipeline reproduces all 61 test predictions |
| Metrics as data | `artifacts/metrics.json` — README and app read numbers from it instead of hard-coding them |
| Input contract at the boundary | `data.validate()` runs in the app before every prediction |
| Automated honesty check | `tests.check_notebook` fails if an "Insight" quotes a number the cells didn't print |
| **Model card** | §15.5 — intended use, excluded inputs, "not for", training data, cut-offs, every metric, known gaps |

A model card is standard practice for responsible ML (Mitchell et al. 2019) and is the artifact that distinguishes a
deployable model from a notebook result.

## 18. Responsible-AI framing

* **Human in the loop.** "Decision support, not diagnosis" is in the banner, the README, the app and the close.
* **Scope limits stated.** Not screening, not acute coronary syndrome triage, not for use without a clinician.
* **Safe failure mode.** The Low tier mandates clinician review, a documented follow-up date, a named contact and
  urgent-return instructions. An ML system's low-risk output must still have a safety net.
* **Input validation as a safety feature.** The data contract refuses to score an implausible patient rather than
  returning a confident wrong number — demonstrated live with `age = 10`.
* **Known gaps published, not buried.** Hungary recall 0.887 below target; women's recall below men's; input timing
  unverified; impact figures labelled "simulated".
