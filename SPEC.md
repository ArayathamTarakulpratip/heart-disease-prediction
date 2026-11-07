# Spec: Heart Disease Triage — Pitch-Grade Upgrade

Status: **v4 — 2026-11-07: pre-angiography input timing, Low-tier safety, simulated impact labels, and separate-fold decision-rule validation.** The video and presentation are deferred at the user’s request.
Sources read: `end-to-end-heart-disease-classification (1).ipynb` (87 cells, all outputs), `../Assignment.md`,
`heart-disease.csv` (profiled from `C:\Users\USER\OneDrive\Desktop\ml-course\proj11\`), and the UCI originals
(`processed.{cleveland,hungarian,switzerland,va}.data`).

---

## 0. Decisions

1. Deliverables: the `cardiotriage/` package (7 modules, §8), `heart_disease_pitch_gold_standard.ipynb` (the single combiner), `artifacts/heart_disease_pipeline.joblib`, `app.py`, `tests/smoke_test.py`, `requirements.txt`. The original notebook is not changed.
   **Why a package instead of one notebook (decided 2026-11-06):** the app and the test import the same code as training, so there is one source of truth; the logic is unit-testable; the notebook stays a readable story of markdown and short calls (Assignment.md style).
2. Python **3.11** venv in `.venv/` (installed at `C:\Users\USER\AppData\Local\Programs\Python\Python311`).
3. **The source of truth is UCI `processed.cleveland.data`** (original codes, `?` = missing, `disease = num > 0`). The Kaggle CSV is loaded only to prove that it is re-coded and label-flipped (T1/T2).
4. The use case is clinical decision support for triage, not diagnosis. A clinician stays in the loop.
5. Recall target: "reasonable", defined as **recall ≥ 0.90 on the disease class** at the chosen threshold.
6. Statistics: global WHO/CDC figures; no Thailand-specific angle.

---

## 1. Step 1: What makes `Assignment.md` executive-grade

| Pattern | How Assignment.md does it | Heart-disease equivalent |
|---|---|---|
| Executive summary first | A one-paragraph blockquote: goal, method, data | A blockquote: "Flag high-risk chest-pain patients for urgent cardiology work-up; recall-first model" |
| Objective + operational context | Retrieval vs ranking; latency budget rules out deep learning | Triage at ED intake vs after a stress test; explain why tabular ML beats DL at n=303 |
| Stated assumptions up front | Temporal consistency, ID integrity, implicit feedback, position bias | Label orientation, missing-value codes, referral-population bias, the 1988 single-site cohort |
| Reproducibility cell | One seed for all libraries, printed library versions | Same, plus `requirements.txt` |
| Data QA before analysis | Clicks ⊆ impressions check; CTR to measure imbalance | Duplicates, invalid codes (`ca=4`, `thal=0`), label-flip proof, range checks, class balance |
| Preprocessing rationale | Median vs mean, "unknown" as a signal, no imputing IDs | Mode/median imputation inside the pipeline (no leakage), one-hot for nominal codes, scaling for LR/KNN |
| Markdown ↔ code rhythm | Purpose → code → **Insight:** (with a number) → **Rationale:** | Every chart is preceded by a hypothesis and followed by an Insight that quotes a number |
| Metric choice justified by data | Imbalance → AUC, NDCG, MAP rather than accuracy | Unequal error costs → Recall/Sensitivity, NPV, PR-AUC, Brier score; accuracy is secondary |
| Progressive benchmarking | LR → LightGBM → XGBoost on the same split | Dummy → LR → KNN → RF → XGBoost on the same CV folds |
| Leakage-aware split | GroupShuffleSplit by session | Stratified split, de-duplicated first; tuning only inside CV; test set touched once |
| Quantitative + qualitative evaluation | Session-level NDCG; per-keyword case studies | Cohort metrics + per-patient case studies (a true positive, a missed patient, a borderline case) |
| Metric sanity check | Synthetic NDCG example | A toy confusion matrix that shows recall vs precision by hand |
| Limitations by system stage | Retrieval not modelled, objective mismatch, etc. | Spectrum bias, invasive features, n=303, sex imbalance, no external validation |
| Business translation + next steps | NDCG → orders and retention | Recall → missed-MI reduction; triage tiers → saved cath-lab and stress-test capacity |

Reviewer feedback to replicate: model-family trade-offs, stated assumptions, imbalance-aware metrics, data QA before analysis, a baseline, reasoned choice of loss, a leakage-aware split, and knowing which metric you optimise vs which you report.

Things Assignment.md did weakly (do not copy these): duplicated code blocks (3.1 and 3.2), filler prose, insights without numbers, quoted results that don't match the cells, `display()` without import, typos.

---

## 2. Step 2: Gap analysis

### 2a. Against the 30-mark rubric

| Criterion (weight) | Current notebook | Gap | Upgrade for full marks |
|---|---|---|---|
| Objective (10%) | "Can we predict heart disease?"; target is 95% accuracy | Not tied to a decision; wrong metric; unrealistic target | "At ED/clinic intake, rank chest-pain patients by disease risk so that ≥90% of true cases are flagged for urgent work-up." Give targets for recall, specificity and AUC |
| Problem importance (15%) | None | No facts, statistics, or costs | WHO/CDC mortality and cost figures, missed-diagnosis rate, capacity bottleneck (§6 sources) |
| Clarity (15%) | Template headings, typos, wrong data dictionary | Reader can't tell who uses the model or when | Exec summary, persona (triage nurse / ED physician), input → output → action diagram, corrected dictionary |
| Model identification (10%) | LR, KNN, RF with no rationale | No boosting family, no baseline, no "why" | Dummy, LR (linear), KNN (distance), RF (bagging), XGBoost (boosting). One paragraph each: how it works, pros/cons, and why it suits n=303 |
| Solution (10%) | Ends at a feature-importance bar chart | No path from model to clinical action | Risk score → 3 triage tiers with thresholds → recommended action; serialized pipeline; app |
| Model comparison (10%) | Accuracy on a 61-row test set; KNN unscaled | Noisy (1 patient = 1.6 pts), unfair to KNN/LR | Repeated stratified CV (5×10) with mean ± SD for AUC, recall, specificity, F1, Brier; one table + box plot |
| Tuning (5%) | KNN tuned **on the test set**; LR/RF via RandomizedSearch | Leakage; the CV step uses the wrong C (4.89 vs best 0.204) | Tune every model inside a CV pipeline; choose the threshold from out-of-fold predictions; add calibration; show before/after |
| Demonstration (5%) | None | No demo | Streamlit app with preset patients, risk gauge, triage tier, and the factors that drove the score |
| Charisma (10%) | n/a | No script | 12-minute script with timing and a hook story; slide outline mapped to the rubric |
| Member contribution (10%) | n/a | No split | Contribution matrix: each member owns a section and presents it on camera |

### 2b. Hidden point traps (verified in the data or code)

| # | Trap | Evidence | Fix |
|---|---|---|---|
| **T1** | **Target is inverted (PROVEN).** In this Kaggle CSV, `target=1` means **no disease** | All 303 rows matched to UCI on 6 numeric columns: `target=1` ↔ `num=0` for 165 rows, `target=0` ↔ `num>0` for 138 rows, with zero exceptions | Use UCI as the source with `disease = (num > 0)`; show the match crosstab in a QA cell. Every "insight" in the current notebook is backwards in clinical terms |
| **T2** | **Every categorical code is re-mapped (PROVEN)** | Kaggle→UCI: `cp` 0→4 (asymptomatic), 1→2, 2→3, 3→1 (typical); `slope` 0→3 (down), 1→2, 2→1 (up); `thal` 1→6 (fixed), 2→3 (normal), 3→7 (reversible), 0→NaN; **`restecg` 0→2 (LVH), 1→0 (normal), 2→1 (ST-T)** | Use UCI codes with the official dictionary; print the mapping table as evidence |
| T3 | Hidden missing values | `ca=4` (5 rows) and `thal=0` (2 rows) are invalid codes; UCI has `?` there | Set them to NaN and impute inside the pipeline |
| T4 | Duplicate row **(PROVEN: a Kaggle artefact)** | Kaggle has 1 exact duplicate (a healthy 38-year-old man) and is **missing UCI patient 184** (a 60-year-old woman with disease), which is why Kaggle counts are 165/138 vs UCI 164/139. UCI Cleveland itself has 0 duplicates | Use UCI; still run `deduplicate()` before splitting as a guard |
| T5 | Test-set tuning | KNN `n_neighbors` was chosen by test score | Use CV only; the test set is used once at the end |
| T6 | Wrong hyperparameters reused | `clf = LogisticRegression(C=4.89)` but `best_params_` gives C=0.204 | Use `best_estimator_` |
| T7 | "CV" that includes the test set | `cross_val_score(clf, X, y)` uses all rows | Run CV on train only and report the test set separately |
| T8 | No scaling | LR ConvergenceWarning; KNN at 69% | `ColumnTransformer` with StandardScaler + OneHotEncoder inside the `Pipeline` |
| T9 | Nominal codes treated as numbers | `cp`, `thal`, `slope`, `restecg` fed as integers | One-hot encode them |
| T10 | Feature "importance" from raw LR coefficients | Unscaled coefficients aren't comparable; `zip(df.columns, …)` is fragile | Permutation importance on CV/test + standardized coefficients |
| T11 | Confusion-matrix axes mislabelled | Rows are true labels in sklearn; the plot calls the x-axis "true" | Use `ConfusionMatrixDisplay` with labels No disease / Disease |
| T12 | Legend bug | `plt.legend("Female","Male")` gives a UserWarning and the wrong legend | `plt.legend(["Female","Male"])` |
| T13 | No `random_state` on models | Results differ between runs | Set `SEED` everywhere and print library versions |
| T14 | 95%-accuracy goal | Published Cleveland results are around 0.85–0.90 accuracy; accuracy is the wrong lens | A recall-first goal with a stated specificity floor |
| T15 | `ca` and `thal` need fluoroscopy or thallium imaging | These tests come late in the work-up; `ca` comes from angiography, which is close to how the label itself is made | Do **not** call the model "non-invasive". Frame it as prioritising the confirmatory angiography queue after the initial work-up, and state the point in Limitations |
| T16 | Spectrum and selection bias | Patients already referred for angiography; 68% male; Cleveland 1988 | Put it in limitations; show recall by sex |
| T17 | Uncalibrated probabilities shown as "risk %" | RF and boosting probabilities are not calibrated | Report the Brier score, show a calibration curve, use `CalibratedClassifierCV` |
| T18 | Presentation hygiene | Typos ("Defifnition", "proff"), 4 empty trailing cells, kernel tied to a local env | Clean the text, remove empty cells, add `requirements.txt` |

### 2c. What is missing from a real end-to-end view

| Stage | Missing | Fix |
|---|---|---|
| Problem framing | Users, decision, cost of errors | Persona, decision flow, FN vs FP costs |
| Data provenance | Source version, licence, label definition | UCI citation (CC BY 4.0), `num>0` definition, row-match proof |
| Data contract | Valid ranges and codes | An assert-based validation cell (that the app reuses) |
| Preprocessing | Fitted outside any pipeline | One sklearn `Pipeline` from raw columns to probability |
| Validation design | One split, test leakage | Locked 20% test set + repeated stratified CV + bootstrap 95% CIs |
| Decision layer | 0.5 cut-off by default | Threshold picked from OOF predictions for recall ≥ 0.90; 3 tiers |
| Error analysis | None | FN table, per-patient case studies, subgroup recall by sex and age band |
| Explainability | Raw coefficients | Permutation importance; per-patient contributions for the LR model |
| Packaging | None | `joblib` pipeline plus metadata (threshold, features, versions, metrics) |
| Serving and demo | None | Streamlit app with input validation |
| Testing | None | Smoke test: load the pipeline, score known patients, check the tier |
| Monitoring and next steps | "Try XGBoost?" | Drift, recalibration, external validation (Hungary/Switzerland/VA UCI subsets), prospective pilot |
| Ethics | None | "Not a diagnostic device" disclaimer, sex fairness, data privacy |

---

## 3. Objective

Turn the course notebook into a pitch-ready proof of concept that scores full marks on every rubric line.

- **User:** a cardiologist or chest-pain-unit physician with history, ECG and recorded exercise-test measures, deciding priority before confirmatory angiography. `ca` and `thal` timing is unverified; the main demo excludes them.
- **Prediction:** P(angiographic coronary disease, >50% narrowing) from the 10 retained inputs. The 13-feature model is a retrospective comparison.
- **Action:** High risk → priority angiography. Medium → cardiology review or further testing. Low → clinician review and a documented safety-net plan before any deferral.
- **Primary metric:** Recall (sensitivity) on the disease class, because a missed case can mean a missed MI. Constraints: specificity, ROC-AUC, and Brier score.

## 4. Scope

### `cardiotriage/` package (shared by the notebook, app, and test, so there is one source of truth)

| Module | Responsibility | Imports from package |
|---|---|---|
| `config.py` | Constants only: paths, `SEED`, split/CV sizes, `N_JOBS`, feature groups (`NUMERIC/ORDINAL/NOMINAL/BINARY`), descriptions, UCI code→label maps, `RECALL_TARGET`, `PPV_TARGET`, `TIER_ACTIONS` | nothing |
| `data.py` | `load_cleveland/hungarian/kaggle`, `?`→NaN, `deduplicate`, `disease = num>0`, `dictionary()`, `validate(df)` (the data contract; raises `ValueError`), `kaggle_vs_uci()` (T1–T4 proof) | `config` |
| `features.py` | Domain-feature functions (module-level, picklable) and `build_preprocessor(use_fe, drop)`. Binary columns are imputed too (most frequent): Cleveland has no gaps there, but other hospitals such as Hungary do. | `config` |
| `models.py` | `MODEL_ZOO` (Dummy, LR, KNN, RF, XGBoost), `SEARCH_SPACES`, `build_pipeline(model, use_fe, drop)`, `zoo_pipeline` | `config`, `features` |
| `evaluation.py` | `make_split`, `make_cv`, `fold_sizes`, `metrics_at`, `cv_scores` (incl. PR-AUC), `cv_table`, `format_table`, `corrected_ttest`, `select_model`, `tune`, `nested_cv`, `calibrated`, `learning_curve_table`, `oof_proba`, `operating_points`, `threshold_sweep`, `net_benefit`, `queue_gain`, `curves`, `confusion`, `bootstrap_ci`, `exact_ci`, `subgroup_recall`, `permutation_table`, `coefficients` | `config`, `data`, `models`, `triage` |
| `triage.py` | Inference: `pick_thresholds`, `triage(p, thresholds)` → tier + action, `baseline(pipe, X)`, `contributions(pipe, patient, baseline)` | `config` |
| `plots.py` | Every chart (EDA and evaluation); takes precomputed data, never imports scikit-learn | `config` |

`__init__.py` is empty, so `app.py` never loads `evaluation`, `models`, or `plots`. The smoke test enforces this.

### `heart_disease_pitch_gold_standard.ipynb` (section order mirrors Assignment.md)

0. **Executive summary** (blockquote) and **problem importance** with cited statistics (§6).
1. **Objective and operational context:** persona, decision flow (input → risk → tier → action), why tabular ML at n=303, a cost table for false negatives vs false positives.
2. **Environment and reproducibility:** `SEED=42`, version print-out.
3. **Acquire and inspect:** assumptions list; load UCI Cleveland; **Kaggle-vs-UCI QA cell** proving T1/T2; official data dictionary.
4. **Integrity QA and cleaning:** duplicates (T4), invalid codes and `?` → NaN (T3), `validate()`, range checks, class balance.
5. **Focused EDA:** 5–6 blocks, each hypothesis → chart → **Insight** with a number, on the *corrected* target.
6. **Split and validation design:** stratified 80/20 locked test set; `RepeatedStratifiedKFold(5×10)` on train; a toy confusion matrix to sanity-check the metrics (like the NDCG sanity check in Assignment.md). *(Moved before feature engineering: the ablation needs the split and folds.)*
7. **Domain feature engineering:** % of age-predicted max HR = `thalach / (220 − age)`; hypertension flag `trestbps ≥ 130` (ACC/AHA 2017); hypercholesterolaemia `chol ≥ 240`; `oldpeak > 1`. **CV ablation decides** whether they are kept, and the result is shown either way. *(Result: dropped — no significant gain.)*
8. **Models:** Dummy → LR → KNN → RF → XGBoost on identical folds; one "how it works / why here" paragraph each; mean ± SD table + box plot; a **paired significance check** (corrected resampled t-test, Nadeau–Bengio) for the best vs the runner-up.
9. **Tuning:** `RandomizedSearchCV` inside CV for every model; before/after table; calibration (`CalibratedClassifierCV`) when a model's Brier score improves; **9.2** every tuned model at its own recall-0.90 cut-off (specificity, PPV, PR-AUC, confusion grid, miss overlap); **9.3** nested CV for the selected model and the runner-up; **9.4 learning curve** (does more data help?); **9.5** drop-column feature selection with a one-step backward elimination (keep all 13 unless the smaller model wins with p < 0.05); **9.6** search-budget check (30 vs 100 settings for the selected model and the runner-up).
10. **Decision layer:** threshold sweep from OOF predictions → per-100-patients table (flagged / caught / missed); **decision-curve analysis** (net benefit vs treat-all / treat-none); 3 triage tiers; **10.3** simulated queue gain versus a proportional-arrival benchmark and retrospective-model tiers per 1,000 referred patients; no queue or wait outcomes are recorded.
11. **Final test evaluation (run once):** ROC and PR curves, confusion matrix at the threshold, recall/specificity/PPV/NPV with **bootstrap 95% CIs**, calibration curve.
12. **Clinical safety:** false-negative table; 3 per-patient case studies (true positive, missed patient, borderline) with per-feature contributions; recall by sex and age band.
13. **External validation (stretch):** train on Cleveland using the 10 features Hungary records reliably (drop `ca`, `thal`, `slope`); score `processed.hungarian.data` (n=294, a different hospital) and report the AUC/recall drop honestly. Switzerland and VA are not used (`chol` is coded 0 = missing, and >50% of the stress-test fields are missing).
14. **Interpretation:** permutation importance + standardized LR coefficients; check them against cardiology knowledge.
15. **Limitations** (T15–T17), **clinical/business impact**, **next steps**, and a short **model card** (intended use, out of scope, metrics, data, caveats).
16. **Serialize** `{"pipeline", "thresholds", "baseline", "features", "metrics", "presets", "versions"}` → `artifacts/heart_disease_pipeline.joblib` (+ `artifacts/metrics.json`).

Selection rule: the best CV ROC-AUC wins. **If LR is within 1 SD of the best, ship LR** (interpretable, exact per-patient contributions, well calibrated). If a tree model wins by more than 1 SD, add `shap` for its explanations (this is an ask-first dependency).

### `app.py` (Streamlit)
Sidebar form with clinical labels and valid ranges; 4 **preset patients** stored in the artifact (clear high-risk, clear low-risk and borderline from the test set; "missed by the model" from the out-of-fold training predictions, because no test patient was missed); outputs are risk %, a colour-coded tier with its action, the top contributing factors (exact linear SHAP values `coef × (x − average training patient)`, computed without the `shap` library), a **what-if** comparison against the previous input, and a disclaimer. The model loads once via `st.cache_resource`; inputs go through `cardiotriage.data.validate()`; it imports only `config`, `data`, and `triage`. Out of scope: auth, a database, batch upload, deployment.

## 5. Tech stack
Python 3.11 venv · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · joblib · streamlit (+ altair, its charting library, used directly by `app.py`) · jupyter + nbconvert. No SHAP unless the selection rule requires it. No imbalanced-learn (the classes are 54/46, so resampling is not justified; the notebook says so).

## 6. Facts to cite (checked against sources on 2026-11-06)
- **WHO CVD fact sheet (updated 31 Jul 2025):** about **19.8 M** CVD deaths in 2022, about 32% of global deaths; 85% of these are heart attack and stroke; about 80% occur in low- and middle-income countries. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
- **CDC Heart Disease Facts:** 919,032 US CVD deaths in 2023 (1 in 3 deaths), one death every ~34 s; heart disease cost **> $168 B** in health care services and medications in 2021–2022. https://www.cdc.gov/heart-disease/data-research/facts-stats/
- **Pope et al., NEJM 2000;342:1163–70:** of 889 acute MI patients, 19 (2.1%, 95% CI 1.1–3.1%) were mistakenly discharged from the ED. Among all acute cardiac ischaemia, women under 55 were more likely to be sent home (OR 6.7, 95% CI 1.4–32.5). This is the argument for recall-first. (Abstract re-checked via Europe PMC, PMID 10770981, 2026-11-06.)
- **Patel et al., NEJM 2010;362:886–95, "Low Diagnostic Yield of Elective Coronary Angiography":** 398,978 patients without known CAD having elective catheterisation at 663 US hospitals (2004–2008); 149,739 (37.6%) had obstructive CAD (≥50% left main or ≥70% major vessel); 39.2% had no CAD (<20% stenosis everywhere); a positive non-invasive test raised the yield only to 41.0% (vs 35.0% with no test). Authors: "better strategies for risk stratification are needed". This is the argument for prioritising the angiography queue. (Abstract via Europe PMC, PMID 20220183.)
- **Al-Hijji et al., Circ Cardiovasc Interv 2019;12:e007791 (Mayo Clinic, 2002–2013):** major complications within 72 h (death, MI, stroke, tamponade, dissection needing PCI, unplanned bypass) in 36 of 43,786 diagnostic left-heart catheterisations (0.082%, 8.2 per 10,000). Rare but not zero: every unnecessary angiography carries this risk. doi:10.1161/CIRCINTERVENTIONS.119.007791
- **Natarajan et al., CMAJ 2002;167:1233–40 (Hamilton, Ontario, Apr 1998–Mar 2000):** 8,030 consecutive patients referred for catheterisation; median wait 6 days (inpatients) and 60 days (outpatients); only 37% of procedures done within the requested wait; 50 (0.6%) died, 32 (0.4%) had an MI, 41 (0.5%) had CHF, 109 (1.4%) any major event while waiting. Authors call for "prioritization schemes that allow earlier access for patients at higher risk". This is the argument that the queue order matters. (Abstract via Europe PMC, PMID 12451076, 2026-11-07.)
- **Sakboonyarat & Rangsin, BMC Res Notes 2024;17:142 (Thai MOPH national data):** age-standardised IHD mortality 23.4 per 100,000 in 2012, peak 28.6 in 2016, 26.9 in 2021; under age 60 it rose 59.6% in men and 36.1% in women over the decade. doi:10.1186/s13104-024-06803-x (abstract via Europe PMC, PMID 38764092, 2026-11-07)
- **Kumwichar et al., Glob Health Res Policy 2024;9:48 (Thai Universal Coverage claims, first ACS in 2017, propensity-matched, observational):** 5-year mortality HR 0.55 (95% CI 0.50–0.62) for STEMI and 0.57 (0.54–0.61) for NSTE-ACS with vs without angiography and PCI. doi:10.1186/s41256-024-00390-x (abstract via Europe PMC, PMID 39563455, 2026-11-07)
- **Chiangrai Med J 2025;17(2):1–10 (Chiangrai Prachanukroh Hospital, 1 Jan–31 Dec 2023):** of 122 non-urgent (elective) coronary angiograms, 36.1% found normal coronary arteries. Single centre, small; used only as a local echo of Patel 2010. https://he01.tci-thaijo.org/index.php/crmjournal/article/view/276594 (abstract checked 2026-11-07)
- **BHF news, 13 Nov 2025 (NHS England data):** 397,478 people waiting for routine cardiac care in England at the end of September 2025; 148,882 (37%) waiting 18 weeks or more. https://www.bhf.org.uk/what-we-do/news-from-the-bhf/news-archive/2025/november/heart-care-waiting-list-at-two-year-low (checked 2026-11-07)
- **CMS CY 2026 OPPS/ASC final rule CMS-1834-FC (published 21 Nov 2025):** CPT 93458 (coronary angiography with left heart catheterisation and ventriculography), status J1, APC 5191: national unadjusted payment **$3,312** (hospital outpatient) and **$1,708** (ASC); physician fee $287 not used. Read from the 93458 row of Medtronic's 2026 Coronary Reimbursement Guide, which cites CMS-1834-FC; a second listing (quickintell APC table) gives APC 5191 = $3,312.15. The CMS Addendum B file itself sits behind the AMA CPT licence page and was not opened. (checked 2026-11-07)
- **Tam et al., Can J Cardiol 2021;37:1547–54 (Ontario, all first-time referrals Jan 2014–Sep 2020):** 584,341 referrals for PCI, CABG, valve surgery or TAVI, 37,718 (6.4%) during the pandemic. Referral during the pandemic predicted higher all-cause mortality while waiting for PCI (HR 1.83, 95% CI 1.47–2.27) and CABG (HR 1.96, 1.28–3.01), not for valve procedures. The recent counterpart to Natarajan 2002. doi:10.1016/j.cjca.2021.05.008 (abstract via Europe PMC, PMID 34600793, 2026-11-07)
- **Khananuraksa et al., J Health Sci Thai 2024;33(3):505–16 (NHSO Universal Coverage reimbursement data, fiscal years 2020–2023; the title says 2020–2022):** 53,300 patients had "CAG only" (angiography with no PCI or CABG afterwards) out of 141,605 CAG cases, 37.63% (37.36–38.05%); their 1-year mortality was 19.66%. "CAG only" includes sick patients managed medically, so it is an upper bound on unneeded angiograms, not a normal-artery rate. https://thaidj.org/index.php/JHS/article/view/15212 (abstract checked 2026-11-07)
- **NHSO annual report FY2025 (fiscal year 2568), as quoted by Thai PBS Policy Watch, 21 Aug 2026:** 65,744 Universal Coverage patients had coronary diagnosis or treatment (79,394 services); 33,667 were CAG only, 39,962 CAG + PCI, 5,864 CAG + CABG. Secondary source (the three parts add to 79,493, not 79,394), so only 65,744 and 33,667 are quoted. https://policywatch.thaipbs.or.th/article/life-365 (checked 2026-11-07)
- **Chandavimol et al., Catheter Cardiovasc Interv 2025;105(2):442–55 (Thai PCI Registry, 2018–2019):** "39 out of 72 facilities voluntarily joined", i.e. 72 PCI facilities nationwide. With the World Bank population of 71,522,271 (2019) that is about 1.0 per million. doi:10.1002/ccd.31334 (full text via Europe PMC, PMC11788963, 2026-11-07)
- **ESC Atlas of Cardiology, total cath labs per million people:** ESC 5.7 (2023), United Kingdom 4.1 (2018), Germany 10.1 (2020). Counts labs, not hospitals, so the Thai contrast is order-of-magnitude only. (checked 2026-11-07)
- **Not used:** an NHS tariff or Thai NHSO/CSMBS/Social Security price per diagnostic angiogram. Searched NHSO announcements, the Comptroller General's Department fee-schedule report (a scanned PDF with no text layer) and the Social Security Office 2026 rate announcement (per-episode lump sums, no per-angiogram figure); private-hospital package pages did not show a CAG price when opened. Money figures are therefore US-only and labelled as an illustration; the transferable figure is angiograms avoided per 1,000 referred patients.
- Rule: never mix 2022 and 2023 figures in one sentence, and give the year next to every number.

## 7. Commands
```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb
.venv/Scripts/python -m tests.check_notebook  # executed notebook is clean and its Insights quote printed numbers
.venv/Scripts/python -m tests.smoke_test      # run from the repo root so `cardiotriage` imports
.venv/Scripts/streamlit run app.py
```

## 8. Project structure
```
cardiotriage/                       the package: all logic (see §4)
  __init__.py                       empty on purpose
  config.py  data.py  features.py  models.py  evaluation.py  triage.py  plots.py
heart_disease_pitch_gold_standard.ipynb   the single combiner: story + calls into cardiotriage
app.py                              Streamlit demo (imports config, data, triage only)
README.md, DEMO.md                  entry point and live-demo run-sheet
tests/test_<area>.py                assert-based checks, one file per area; no pytest dependency
tests/smoke_test.py                 runner for all of them (`python -m tests.smoke_test [area ...]`)
data/raw/processed.cleveland.data   UCI original (source of truth)
data/raw/processed.hungarian.data   UCI, external validation
data/raw/heart-disease.csv          Kaggle copy (QA evidence only)
artifacts/heart_disease_pipeline.joblib, artifacts/metrics.json
requirements.txt
end-to-end-heart-disease-classification (1).ipynb   untouched
```

## 9. Code style
```python
# cardiotriage/config.py
SEED = 42
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]

# cardiotriage/models.py  (the preprocessor itself is built in features.py)
def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC + ORDINAL),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), NOMINAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([("pre", pre), ("model", model)])
```
Conventions: one responsibility per module; no cycles (§4 table); domain features are module-level functions, never lambdas, so the joblib unpickles in the app; the notebook cells only call package functions; every code cell is preceded by a *Purpose* line and followed by an **Insight:** that contains a number; markdown never hard-codes a result that the cells don't print.

## 10. Testing strategy
- The notebook runs headless via nbconvert with **0 errors and 0 warnings**. Grep the outputs for `Warning`. Fix warnings at the cause; blanket `filterwarnings("ignore")` is not allowed.
- `tests/smoke_test.py`: `validate()` accepts the clean data and rejects bad codes; the joblib loads in a fresh interpreter; the presets give probabilities in [0, 1] and the expected tiers; contributions sum to logit(patient) − logit(average training patient); importing `cardiotriage.triage` and reading `app.py` does not pull in `evaluation`, `models`, or `plots`.
- Manual: `streamlit run app.py`, click every preset, try one invalid input.

## 11. Boundaries
- **Always:** keep the original notebook untouched; set seeds; tune only inside CV; touch the test set once; give a source and year for every statistic; label the app "decision support, not diagnosis".
- **Ask first:** dependencies beyond §5 (for example `shap`); committing or pushing; changing the recall target.
- **Never:** claim results that the cells don't print; suppress warnings to reach "0 warnings"; present the model as diagnostic; commit secrets.

## 12. Success criteria
1. A notebook cell proves T1/T2 against UCI.
2. All of T1–T18 are addressed and each one is referenced in the notebook.
3. Five model families are compared on identical repeated-CV folds (mean ± SD), plus a significance check.
4. Recall ≥ 0.90 at the chosen threshold on OOF predictions; test recall is reported with a bootstrap 95% CI. If specificity falls below 0.60 at that point, document the trade-off in a decision-curve plot instead of hiding it.
   **Status: met.** OOF recall 0.901 at specificity 0.679 (≥ 0.60). Test: 28 of 28 sick patients flagged; because a bootstrap CI collapses at 28/28, the exact Clopper–Pearson interval is reported instead (0.877–1.000), and the bootstrap is kept for the other metrics.
5. Headless execution shows 0 errors and 0 warnings; the joblib file is produced.
   **Status: met for notebook outputs** (`tests.check_notebook` clean). The `nbconvert` *console* also prints 81 joblib/loky `resource_tracker` tracebacks (Windows); they never reach a cell and do not affect results (see plan.md risks).
6. `app.py` works with all presets; `tests/smoke_test.py` passes.
   **Status: met.** Smoke test 58/58, including 5 Streamlit `AppTest` checks (every preset lands in its expected tier, the contract rejects age 10, what-if delta works, no deprecation notices, no training-code imports); also checked in a real browser.

7. Every statistic in Section 0 is checked against its primary source, with a year (§6); the problem statement, prediction task and dataset citation are explicit.
   **Status: met.** Patel 2010, Al-Hijji 2019, Pope 2000, WHO and CDC verified. Added 2026-11-07: CMS CY 2026 angiogram payment (cost table in §15.2) and the BHF/NHS England waiting list (Sept 2025).
8. Models are compared at the clinical operating point as well as by threshold-free metrics (ROC-AUC, PR-AUC), with per-model confusion matrices and error overlap; the math of each family is stated.
   **Status: met.** At recall 0.901, specificity LR 0.679 > XGBoost 0.672 > KNN 0.603 > RF 0.565; 6 sick patients missed by all 4.
9. Tuning optimism is measured with nested CV within a ≤ 3-minute notebook run.
   **Status: met** for LR and the runner-up (all four would exceed the budget): no detectable optimism (nested 0.909 vs tuned 0.900).
10. The impact is quantified (queue gain, per 1,000 referred patients), and the repo has a README and a demo run-sheet whose figures come from the app.
    **Status: met.** First 40% of slots serve 76.6% of sick patients vs 40%; DEMO.md figures produced with `AppTest`.

## 13. Plan
See `tasks/plan.md` and `tasks/todo.md`.
