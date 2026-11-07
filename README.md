# Heart Disease Triage: prioritising the angiography queue

A proof of concept that estimates the probability that coronary angiography will show significant narrowing from 10 recorded pre-angiography features, then proposes a tier (High: priority angiography; Medium: cardiology review; Low: clinician review with a documented safety-net plan). It is **decision support, not diagnosis**: the clinician makes the call.

**Start here:** [`heart_disease_pitch_gold_standard.ipynb`](heart_disease_pitch_gold_standard.ipynb) is the report. `end-to-end-heart-disease-classification (1).ipynb` is the original course notebook, kept unchanged for comparison; its labels are inverted (see below), so do not reuse its conclusions.

## Why it matters

Cardiovascular disease killed about 19.8 million people in 2022 (WHO), and in a US registry of 398,978 elective angiograms only 37.6% found obstructive disease (Patel et al., *NEJM* 2010). Waiting is itself dangerous: of 8,030 Ontario patients referred for catheterisation, 1.4% died, had a heart attack or developed heart failure while on the list (Natarajan et al., *CMAJ* 2002), and when the pandemic disrupted Ontario's queues, patients referred for PCI had 1.83 times the hazard of dying while they waited (584,341 referrals; Tam et al., *Can J Cardiol* 2021). In Thailand, age-standardised ischaemic heart disease mortality rose from 23.4 to 26.9 per 100,000 between 2012 and 2021 (Sakboonyarat & Rangsin, *BMC Res Notes* 2024); of 141,605 Universal Coverage angiograms in 2020–2023, 37.6% of patients had no stent or bypass afterwards (Khananuraksa et al., *J Health Sci Thai* 2024); and the country had 72 hospitals able to do coronary intervention in 2018–2019, about one per million people (Chandavimol et al., *Catheter Cardiovasc Interv* 2025). Queues are still long: in England 148,882 people had waited 18 weeks or more for cardiac care at the end of September 2025 (BHF / NHS England). In 2026 US Medicare pays $3,312 for one diagnostic angiogram in a hospital outpatient department (CMS CY 2026 OPPS, CPT 93458). Risk ordering might move more patients with disease into early slots, but this dataset does not measure queue changes or safety outcomes. Full sources and years: notebook Section 0 and `SPEC.md` §6.

## The data, and a finding about it

UCI Heart Disease, Cleveland subset: 303 patients, 13 clinical attributes, target = more than 50% narrowing on angiography (`num > 0`). Janosi, Steinbrunn, Pfisterer, Detrano (1989), [doi:10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

The widely used **Kaggle copy is not the same data**: matching all 303 rows back to UCI shows its **labels are inverted** (`target = 1` means *no* disease), four categorical columns are re-coded, missing values are hidden as fake codes, and one sick patient was replaced by a duplicate of a healthy one. Everything here uses the original UCI file (notebook Section 3.2).

## Input availability

The UCI data provide no measurement timestamps. `ca` means vessels coloured by fluoroscopy and may be measured during the angiogram being prioritised. The demo therefore leads with the existing 10-feature model, excluding `ca`, `thal` and `slope`. `thal` is thallium imaging with unverified timing; `slope` is exercise ECG and is excluded for cross-hospital completeness. The 13-feature result is a retrospective comparison and must not decide pre-angiography priority. This timing assumption needs checking in a modern, timestamped cohort before clinical use. [American Heart Association catheterisation overview](https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/cardiac-catheterization).

A Low result does not clear the patient. Before deferral, a clinician reviews symptoms and tests, records a follow-up date and responsible contact, and gives instructions for urgent assessment if chest pain is new or worsens. Suspected acute coronary syndrome follows the urgent pathway regardless of score. This is a proposed safety process, not a validated clinical protocol. [NICE chest-pain assessment](https://www.nice.org.uk/guidance/CG95/chapter/recommendations).

## Results

| What | Result |
|---|---|
| Models compared (same 50 cross-validation folds) | Dummy, logistic regression, KNN, random forest, XGBoost; all tuned inside cross-validation |
| Retrospective 13-feature model | Logistic regression (`C` = 0.216): best cross-validated ROC-AUC **0.900 ± 0.035**, and at 90% recall it ties XGBoost on specificity (0.679 vs 0.672: 89 vs 88 of 131 healthy patients) while beating KNN (0.603) and random forest (0.565) |
| Retrospective cut-offs (same OOF patients selected and assessed them) | Low 0.287, High 0.509; out-of-fold recall 0.901 at specificity 0.679 |
| Retrospective 13-feature locked test (61 patients) | **28 of 28** sick patients flagged (exact 95% CI 0.877–1.000); ROC-AUC 0.961 (0.902–0.998); specificity 0.667 (0.500–0.829) |
| Main 10-feature model (no `ca`/`thal`/`slope`) | Cleveland cross-validated ROC-AUC 0.844; Hungary external ROC-AUC 0.877 and recall 0.887, below the 0.90 target. Separate-fold assessment of tuning and cut-offs reaches recall 0.901, specificity 0.542 and High-tier PPV 0.857 on Cleveland training folds. The app displays this model first. |
| Queue simulation (retrospective 13-feature risks) | First 40% of hypothetical risk-ordered slots contain 76.6% of sick patients, versus 40% under a proportional-arrival benchmark. No actual queue order or waiting time was observed. |
| Payment simulation (13-feature tiers, US Medicare 2026) | If Low patients received no angiography, the estimated payment difference would be $1,368,518 per 1,000 referred. This assumes 45.5 patients with disease per 1,000 would not undergo angiography; it is not a measured saving or a recommendation to deny testing. |
| Retrospective feature and search checks | dropping a single feature other than `ca` made no significant difference; a smaller full-model variant was not reliably better (+0.007, p = 0.47). These checks do not select the main reduced-input feature set. |
| Known gaps | Hungary recall for the main model was 0.887, below target. The retrospective model had lower out-of-fold recall for women (0.778) than men (0.925); the main model needs its own subgroup check. |

Model figures are read from [`artifacts/metrics.json`](artifacts/metrics.json); the separate-fold decision check is in notebook Section 13. The payment and queue simulations in Sections 10 and 15 use the retrospective model.

## Run it

Python 3.11 or newer (verified on 3.13), from the repo root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb   # rebuilds artifacts/
.venv/Scripts/python -m tests.check_notebook    # must print "clean"
.venv/Scripts/python -m tests.smoke_test        # all tests must pass
.venv/Scripts/streamlit run app.py              # the demo
```

## Demo

[`DEMO.md`](DEMO.md) is a six-step walkthrough of the Streamlit app: three test-set patients (High, Low, borderline), a what-if that changes the main risk, a patient the model may miss, the retrospective 13-feature comparison, and an impossible input that the data contract rejects.

## Repository

| Path | What it is |
|---|---|
| `cardiotriage/` | All the logic: data contract, features, models, evaluation, triage, plots (one module per job) |
| `heart_disease_pitch_gold_standard.ipynb` | The report: markdown and short calls into `cardiotriage` |
| `app.py` | Streamlit demo; uses only the inference side of the package |
| `tests/` | Assert-based tests (`python -m tests.smoke_test`) and the notebook checker |
| `artifacts/` | The saved model and its metrics, written by the notebook's last section |
| `data/raw/` | UCI Cleveland (source of truth), UCI Hungary (external validation), the Kaggle copy (evidence only) |

More detail: [`STRUCTURE.md`](STRUCTURE.md) (repo map and import rules), [`SPEC.md`](SPEC.md) (requirements and sources), [`PROJECT_LOG.md`](PROJECT_LOG.md) (what was built and checked, phase by phase).

## Limitations

One hospital in the 1980s, patients already referred for angiography, 303 patients in total. The main demo excludes `ca`, `thal` and `slope` because availability before the target angiogram is unverified for the imaging inputs and `slope` is missing at the external site. It is not for first-contact screening. The money figures use US Medicare 2026 payments (payments, not hospital costs, physician fee excluded); no published Thai (NHSO, Comptroller General's Department) or UK public price per diagnostic angiogram was found, so no savings claim transfers without observed procedure decisions and outcomes. Not a medical device.
