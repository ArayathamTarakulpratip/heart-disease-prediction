# Heart Disease Triage: prioritising the angiography queue

A proof of concept that estimates, for a patient who has finished the initial cardiac work-up, the probability that coronary angiography will show significant narrowing, and turns it into a triage tier (High → priority angiography, Medium → cardiology review, Low → routine follow-up with safety-netting). It is **decision support, not diagnosis**: the clinician makes the call.

**Start here:** [`heart_disease_pitch_gold_standard.ipynb`](heart_disease_pitch_gold_standard.ipynb) is the report. `end-to-end-heart-disease-classification (1).ipynb` is the original course notebook, kept unchanged for comparison; its labels are inverted (see below), so do not reuse its conclusions.

## Why it matters

Cardiovascular disease killed about 19.8 million people in 2022 (WHO), and in a US registry of 398,978 elective angiograms only 37.6% found obstructive disease (Patel et al., *NEJM* 2010). Waiting is itself dangerous: of 8,030 Ontario patients referred for catheterisation, 1.4% died, had a heart attack or developed heart failure while on the list (Natarajan et al., *CMAJ* 2002), and when the pandemic disrupted Ontario's queues, patients referred for PCI had 1.83 times the hazard of dying while they waited (584,341 referrals; Tam et al., *Can J Cardiol* 2021). In Thailand, age-standardised ischaemic heart disease mortality rose from 23.4 to 26.9 per 100,000 between 2012 and 2021 (Sakboonyarat & Rangsin, *BMC Res Notes* 2024); of 141,605 Universal Coverage angiograms in 2020–2023, 37.6% of patients had no stent or bypass afterwards (Khananuraksa et al., *J Health Sci Thai* 2024); and the country had 72 hospitals able to do coronary intervention in 2018–2019, about one per million people (Chandavimol et al., *Catheter Cardiovasc Interv* 2025). Queues are still long: in England 148,882 people had waited 18 weeks or more for cardiac care at the end of September 2025 (BHF / NHS England). In 2026 US Medicare pays $3,312 for one diagnostic angiogram in a hospital outpatient department (CMS CY 2026 OPPS, CPT 93458). Ranking referred patients by risk sends the sickest to angiography first without losing sick patients in the queue. Full sources and years: notebook Section 0 and `SPEC.md` §6.

## The data, and a finding about it

UCI Heart Disease, Cleveland subset: 303 patients, 13 clinical attributes, target = more than 50% narrowing on angiography (`num > 0`). Janosi, Steinbrunn, Pfisterer, Detrano (1989), [doi:10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

The widely used **Kaggle copy is not the same data**: matching all 303 rows back to UCI shows its **labels are inverted** (`target = 1` means *no* disease), four categorical columns are re-coded, missing values are hidden as fake codes, and one sick patient was replaced by a duplicate of a healthy one. Everything here uses the original UCI file (notebook Section 3.2).

## Results

| What | Result |
|---|---|
| Models compared (same 50 cross-validation folds) | Dummy, logistic regression, KNN, random forest, XGBoost; all tuned inside cross-validation |
| Selected model | Logistic regression (`C` = 0.216): best cross-validated ROC-AUC **0.900 ± 0.035**, and at 90% recall it ties XGBoost on specificity (0.679 vs 0.672: 89 vs 88 of 131 healthy patients) while beating KNN (0.603) and random forest (0.565) |
| Tier cut-offs (out-of-fold, training set only) | Low 0.287, High 0.509; out-of-fold recall 0.901 at specificity 0.679 |
| Locked test set (61 patients, used once) | **28 of 28** sick patients flagged (exact 95% CI 0.877–1.000); ROC-AUC 0.961 (0.902–0.998); specificity 0.667 (0.500–0.829) |
| Circularity check: the same model without imaging (10 features, no `ca`/`thal`/`slope`) | cross-validated ROC-AUC 0.846 (vs 0.900 with imaging); at another hospital (Hungary, 294 patients, no refitting) 0.880. Most of the signal is there before imaging, so the model is not just reading the answer; the app shows both risks side by side |
| Queue impact (out-of-fold) | the first 40% of angiography slots reach 76.6% of sick patients, against 40% in arrival order |
| Cost impact (US Medicare 2026, $3,312 per angiogram) | angiography for High + Medium tiers only: $1,368,518 saved per 1,000 referred patients; cost per sick patient found $7,220 → $4,703 (45.5 sick patients per 1,000 deferred to safety-netted follow-up) |
| Feature selection and search size | dropping any single feature other than `ca` gives no significant change; the 6-feature-smaller model is not reliably better (+0.007, p = 0.47), so all 13 stay; 100 search settings instead of 30 change nothing for LR and +0.003 for XGBoost |
| Known gap | out-of-fold recall is lower for women (0.778) than for men (0.925); the app warns about it |

Model and test figures are read from [`artifacts/metrics.json`](artifacts/metrics.json); the rest are printed in the notebook (Sections 9–12).

## Run it

Python 3.11, from the repo root:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb   # rebuilds artifacts/
.venv/Scripts/python -m tests.check_notebook    # must print "clean"
.venv/Scripts/python -m tests.smoke_test        # all tests must pass
.venv/Scripts/streamlit run app.py              # the demo
```

## Demo

[`DEMO.md`](DEMO.md) is a seven-step walkthrough of the Streamlit app: three test-set patients (High, Low, borderline), a what-if that moves the borderline patient to High, a patient the model misses, the circularity check (the same patient scored without imaging), and an impossible input that the data contract rejects.

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

One hospital in the 1980s, patients already referred for angiography, 303 patients in total. Two of the strongest inputs (`ca`, `thal`) come from imaging late in the work-up, so the model is for ordering the angiography queue after imaging; the no-imaging check (notebook Sections 1 and 13) shows how much survives without them. It is not for first-contact screening. The money figures use US Medicare 2026 payments (payments, not hospital costs, physician fee excluded); no published Thai (NHSO, Comptroller General's Department) or UK public price per diagnostic angiogram was found, so the transferable figure is the count of angiograms avoided per 1,000 referred patients. Not a medical device.
