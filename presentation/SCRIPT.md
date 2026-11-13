# Script — 13:00 at 1x, six speakers, 2:10 each

Replace `M1`–`M6` with real names. Every speaker has **one contiguous 2:10 block** (6 × 2:10 = 13:00), is on camera
throughout, and names the next speaker at the hand-off.
~125 words per minute with pauses on numbers; each block is 240–280 words. Say each number once. Never read the slide.
Record with a visible timer: 13:00 target, 13:30 hard stop, 15:00 cap.

| Block | Who | Time | Slides | Segment |
|---|---|---|---|---|
| 1 | M1 | 0:00–2:10 | 1–3 | Problem, objective, dataset |
| 2 | M2 | 2:10–4:20 | 4–6 | Data quality and EDA |
| 3 | M3 | 4:20–6:30 | 7–9 | Input timing, validation design, models |
| 4 | M4 | 6:30–8:40 | 10–12 | Comparison, tuning, results |
| 5 | M5 | 8:40–10:50 | 13–14 | Improvement and limitations |
| 6 | M6 | 10:50–13:00 | 15–17 | Live demo and conclusion |

---

## M1 — Problem, objective, dataset (0:00–2:10)

**[Slide 1, 5 s]** Good afternoon, we are team *[name]*. Our project is a model that helps prioritise patients waiting
for coronary angiography.

**[Slide 2, 70 s]** Angiography is the test that confirms whether a heart artery is narrowed, and patients often wait
for it. In a study of 8,030 patients waiting in Ontario, **1.4 percent** died, had a heart attack, or developed heart
failure before the test. Yet in the United States, of nearly 400,000 elective angiograms, only **37.6 percent** found
obstructive disease. So the order of the queue matters. Our goal: from ten results a clinician already has, estimate
the probability that angiography will show more than fifty percent narrowing, and flag at least **ninety percent** of
patients who truly have disease. That makes **recall** our main metric, because missing a sick patient is worse than
one extra test. Accuracy would mislead us: always answering "no disease" is 54 percent accurate and catches nobody.

**[Slide 3, 55 s]** Our data is the UCI Heart Disease dataset, the Cleveland subset: **303 patients and 13 attributes**,
published in 1989. Five are numeric, such as age and blood pressure, one is ordinal, four are categorical, like
chest-pain type, and three are binary. The target is whether angiography showed more than fifty percent narrowing:
139 patients have disease and 164 do not. So this is binary classification.

*[bridge]* **M2** will show how we checked the data.

---

## M2 — Data quality and EDA (2:10–4:20)

**[Slide 4, 45 s]** Before modelling we validated the data against a contract of plausible ranges: age 18 to 100,
blood pressure 60 to 260, cholesterol 80 to 700, heart rate 50 to 250. They are wide on purpose, to reject impossible
values rather than unusual patients. All 303 patients pass. There are six missing cells out of nearly four thousand, in
two columns, which we impute inside the model pipeline so nothing leaks. No duplicates. And **45.9 percent** have
disease, so the classes are nearly balanced and we do not resample.

**[Slide 5, 40 s]** Now the patterns. First, sex: **55 percent** of men have disease against **26 percent** of women,
and only 97 patients are women. Second, chest pain: asymptomatic patients have the highest rate, **72.9 percent**. That
looks backwards, but everyone here was already referred for angiography. So we one-hot encode chest-pain type.

**[Slide 6, 45 s]** Third, the exercise test: **76.8 percent** of patients with exercise-induced angina have disease,
against 31 percent without. Fourth, imaging: as the number of vessels seen on fluoroscopy rises from zero to three,
the disease rate climbs from **26 to 85 percent**. And blood pressure and cholesterol barely differ between groups,
with a median blood pressure of 130 in both.

*[bridge]* **M3**: what that imaging finding means for our model.

---

## M3 — Input timing, validation design, models (4:20–6:30)

**[Slide 7, 50 s]** That strongest feature is a problem. The vessel count, `ca`, comes from fluoroscopy and may be read
**during the very angiogram we are trying to schedule**. UCI has no timestamps, so we cannot prove it exists beforehand.
We therefore removed it, together with the thallium scan and the ST slope. Cross-validated ROC-AUC fell from **0.900
to 0.844**, and we accepted that. The model we ship uses the **ten features** available before the procedure.

**[Slide 8, 35 s]** Our validation pipeline. We check the data, keep the ten features, and lock away a stratified
20 percent, 61 patients. On the other 242, every model runs through the same **fifty folds**, and inside each fold
imputation, scaling and encoding are fitted on the training part only, so nothing leaks. Then we tune, select, set the
cut-offs from out-of-fold risks, check the second hospital, and open the test set **once**.

**[Slide 9, 45 s]** We compare five models. A **dummy** model predicts the base rate; it is our floor. **Logistic
regression** is a weighted sum passed through a sigmoid. **KNN** votes among the most similar patients. A **random
forest** averages many trees. **XGBoost** builds trees one after another, each correcting the last. We chose these
four real models over deep learning because with 303 rows, simple models are as good and explainable.

*[bridge]* **M4** will compare them.

---

## M4 — Comparison, tuning, results (6:30–8:40)

**[Slide 10, 40 s]** On the same fifty folds, tuned, logistic regression scores a ROC-AUC of **0.900**, XGBoost 0.897,
KNN 0.896, random forest 0.894, and the dummy 0.500. The gap at the top is 0.003 and a corrected t-test gives **p of
0.81**, so there is no real difference. We choose logistic regression because it is interpretable and well calibrated.

**[Slide 11, 45 s]** Tuning was a randomised search of thirty settings inside cross-validation. It lifted KNN by 0.030
and XGBoost by 0.034, but did not change the ranking. We then compared models at the same **ninety percent recall**:
logistic regression correctly spares **89 of 131** healthy patients, XGBoost 88, KNN 79, forest 74. We set the cut-offs
from out-of-fold predictions: low risk below 26.3 percent, high risk from 66.2 percent.

**[Slide 12, 45 s]** Three checks. Re-choosing the cut-offs inside each fold, recall is **0.901**, specificity 0.542,
and the high tier is **85.7 percent** true cases. On the locked test set, **28 of 28** sick patients were flagged,
though with 61 patients the interval starts at 0.877. And on a second hospital in Budapest, 294 patients and no
refitting, ROC-AUC is **0.877** but recall is 0.887, below our target.

*[bridge]* **M5** will talk about improving it.

---

## M5 — Improvement and limitations (8:40–10:50)

**[Slide 13, 70 s]** How can we improve it? First, is our tuned score flattering itself? Nested cross-validation gives
**0.909** against 0.900, so there is no tuning optimism, and a hundred-setting search added nothing. Second, the learning
curve is **still rising** at 242 patients, so more data is the strongest lever: we would pool the other UCI hospitals and
modern records. Third, **six sick patients are missed by all four models**. A fifth algorithm will not find them. New
inputs, like troponin or symptom duration, might. Other ideas: stacking logistic regression with XGBoost, and
cut-offs set from clinicians' costs rather than a fixed ninety percent.

**[Slide 14, 60 s]** Our limitations. Recall for women is **0.833** against 0.914 for men, based on only eighteen sick
women, and the cut-off was chosen on those patients, so it is descriptive. The data is one hospital from the 1980s, and
only patients already referred for angiography. Recall at the second hospital, 0.887, is below our target. And the
timing of `ca` still needs checking on a modern cohort with timestamps. This is a proof of concept.

*[bridge]* **M6** will demonstrate it.

---

## M6 — Live demo and conclusion (10:50–13:00)

**[Slide 15, 95 s]** Live demo. Follow [`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md), about 180 words over the screen
capture: High 97.5 percent; Low 2.1 percent with the safety plan; Borderline 30.3 percent Medium; heart rate to 100
moves it to High; `ca` to 2 does **not** move the main score; the missed patient at 5.5 percent; age 10 refused.

**[Slide 16, 25 s]** To conclude: logistic regression on ten inputs reaches recall **0.901**, cross-validated ROC-AUC
0.844, and **0.877** at a second hospital. A simple, interpretable model did as well as XGBoost here. And it is
**decision support, not diagnosis**: a clinician makes every call.

**[Slide 17, 10 s]** That was *[all six names]*. Thank you.

---

## Rehearsal checklist

- [ ] Each block read aloud with a timer, within ±8 s of 2:10.
- [ ] Every number matches `artifacts/metrics.json` and `DEMO_RUNSHEET.md` after the final notebook run.
- [ ] Each speaker names the next at the hand-off.
- [ ] Demo recorded twice, each 1:35 or under.
- [ ] Full playback at 1x totals 12:45–13:30.
