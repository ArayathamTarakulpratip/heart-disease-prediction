# Script — 13:00 at 1x, six speakers, 2:10 each

Replace `M1`–`M6` with real names. **Every speaker gets exactly one contiguous 2:10 block** (6 × 2:10 = 13:00), is on
camera for all of it, and names the next speaker at the hand-off.

**Pacing.** ~140 words per minute, so a 2:10 block is ~300 words. Each block below is sized to that (word count in the
heading). If you run long, cut an adjective, never a number. Say every number once and let it sit.
Say "simulated" out loud wherever it appears. Never read the slide; the slide carries the number.
Record with a visible timer: 13:00 target, 13:30 hard stop, 15:00 cap.

| Block | Who | Time | Slides | Segment |
|---|---|---|---|---|
| 1 | **M1** | 0:00–2:10 | 1–8 | Hook (the audit) + why it matters |
| 2 | **M2** | 2:10–4:20 | 9–12 | Objective + data, contract, balance |
| 3 | **M3** | 4:20–6:30 | 13–17 | EDA + the input-timing pivot |
| 4 | **M4** | 6:30–8:40 | 18–23 | Models, comparison, tuning, tiers |
| 5 | **M5** | 8:40–10:50 | 24–28 | Impact, validation, improvement, limits |
| 6 | **M6** | 10:50–13:00 | app + 29–31 | Live demo (1:35) + close (0:35) |

---

## M1 — Hook and why it matters (0:00–2:10, ≈300 words)

**[Slide 1, 4 s]** Good afternoon, we are team *[name]*. We built a triage model for the coronary angiography queue,
but first, something we found in the data.

**[Slide 2, 12 s]** Patient 184: a sixty-year-old woman with coronary disease. She is in the original UCI file. She is
**not** in the Kaggle copy most people download. She was dropped, and a duplicate of a healthy patient took her place.

**[Slide 3, 20 s]** We matched all 303 rows back to the original. Every label in that copy is **inverted**: in Kaggle,
`target` equals one means *no* disease. A hundred and thirty-eight sick patients labelled healthy, a hundred and
sixty-five healthy labelled sick. Not one row disagrees. Everything we show next uses the original file.

**[Slide 4, 10 s]** Why does this matter? Cardiovascular disease kills about **19.8 million** people a year, a third of
all deaths.

**[Slide 5, 22 s]** The queue is where ranking pays off. In Ontario, **1.4 percent** of 8,030 patients waiting for
catheterisation died, had a heart attack or developed heart failure *before their test*. In England, 148,882 people had
waited eighteen weeks or more for cardiac care by September 2025.

**[Slide 6, 20 s]** Yet of nearly 400,000 US patients sent for an elective angiogram, only **37.6 percent** had
obstructive disease. Most tests found nothing to treat, at about $3,300 each.

**[Slide 7, 12 s]** Thailand is under the same pressure: ischaemic heart disease deaths rose from **23.4 to 26.9** per
100,000, with about one intervention hospital per million people.

**[Slide 8, 18 s]** So the two errors aren't equal. Deferring a sick patient is severe; prioritising a healthy one is
tolerable. **Recall is our primary metric**, not accuracy. "Nobody is sick" is already 54 percent accurate.

*[bridge]* **M2** will state the objective and show the data.

---

## M2 — Objective, data, contract, balance (2:10–4:20, ≈300 words)

**[Slide 9, 30 s]** Our objective, in one sentence. From ten results a clinician already has, estimate the probability
that angiography shows more than fifty percent narrowing in a major vessel, and place the patient in a tier, flagging
at least **ninety percent** of patients who truly have disease. Inputs, a data contract, a risk, a tier, an action. The
clinician makes the call.

**[Slide 10, 30 s]** The data is UCI Heart Disease, Cleveland: **303 patients, 13 clinical attributes**. Five numeric,
one ordinal, four categorical, three binary. The label is the angiographic result, `disease` equals one when UCI's
outcome score is above zero.

**[Slide 11, 50 s]** Before any analysis we wrote a **data contract**. Every patient must pass it or isn't scored: age
18 to 100, blood pressure 60 to 260, cholesterol 80 to 700, heart rate 50 to 250. These are plausibility bounds, set
deliberately wider than the observed data, so they reject typos and impossible values, not unusual patients. All 303
UCI patients pass. Six missing cells out of nearly four thousand, imputed inside the model pipeline so nothing leaks
between folds. Zero duplicates. The Kaggle file is **rejected**: 143 invalid chest-pain codes, 186 invalid thallium
codes, and missing values disguised as real codes.

**[Slide 12, 20 s]** Finally, balance. **45.9 percent** of patients have disease, nearly balanced, so no resampling.
But "always predict no disease" scores 54.1 percent accuracy with **zero** recall. That is why we optimise recall and
treat accuracy as a footnote.

*[bridge]* **M3** will show what the data says.

---

## M3 — EDA and the input-timing pivot (4:20–6:30, ≈300 words)

**[Slide 13, 18 s]** Four patterns. First, sex: **55.3 percent** of men have disease against **25.8 percent** of women,
and only 97 of 303 patients are women. Remember that.

**[Slide 14, 25 s]** Second, the surprise. **Asymptomatic** patients have the highest rate, **72.9 percent**, against 30
percent for typical angina. It isn't backwards: everyone here was already referred, so the asymptomatic group were sent
for some other reason. It's why we one-hot encode chest pain.

**[Slide 15, 25 s]** Third, the exercise test earns its place: exercise angina, **76.8 percent** against 31. Peak heart
rate 139 beats against 158. Blood pressure and cholesterol barely differ, with a median blood pressure of 130 in
*both* groups. Significant is not useful.

**[Slide 16, 15 s]** Fourth, the strongest signal: imaging. Zero vessels on fluoroscopy, 26 percent have disease; three
vessels, **85 percent**.

**[Slide 17, 47 s]** And that is our problem. `ca`, vessels coloured by fluoroscopy, **may be measured during the very
angiogram we're trying to schedule**. UCI has no timestamps to prove otherwise. Ship it and the model needs the answer
to produce the answer. So we deleted vessels, the thallium scan and the ST slope. Cross-validated ROC-AUC fell from
**0.900 to 0.844**, on purpose. And here is what that bought: the thirteen-feature model **cannot be validated at a
second hospital**, because Budapest recorded 99 percent of vessel counts as missing. The ten-feature model scores
**0.877** there. The honest model is the only one that can be checked.

*[bridge]* **M4**: so which algorithm?

---

## M4 — Models, comparison, tuning, tiers (6:30–8:40, ≈300 words)

**[Slide 18, 18 s]** With 303 patients one split is noise. We lock away 61 patients, open them **once**, and score
every model on the same **fifty** cross-validation folds of the other 242. All preprocessing sits inside the pipeline.

**[Slide 19, 25 s]** Five families. **Dummy** predicts the base rate, the floor. **Logistic regression**: a weighted sum
through a sigmoid, every weight readable. **KNN**: a vote among the forty-six most similar patients. **Random forest**:
many trees on resamples, averaged. **XGBoost**: trees fitted one after another, each fixing the last one's errors. No
deep learning: 300 rows, and a clinician needs a reason.

**[Slide 20, 25 s]** Tuned, logistic regression **0.900**, XGBoost 0.897, KNN 0.896, forest 0.894, dummy 0.500. The gap
at the top is 0.003, with a corrected t-test **p of 0.81**: no detectable difference. So we ship the interpretable one.

**[Slide 21, 15 s]** Tuning, thirty settings inside cross-validation, lifted KNN by 0.030 and XGBoost by 0.034, and
changed no rankings.

**[Slide 22, 20 s]** But decisions aren't made at ROC-AUC. Forced to **90 percent recall**, logistic regression spares 89
of 131 healthy patients, XGBoost 88, KNN 79, forest 74.

**[Slide 23, 27 s]** A probability isn't a decision, so we use tiers: **Low below 26.3 percent, High from 66.2**, chosen
from out-of-fold predictions. Re-chosen inside each fold, recall holds at **0.901**, and the High tier is **85.7
percent** true cases. And **Low is not a discharge**: clinician review, a follow-up date, a named contact.

*[bridge]* **M5** will show what this does for the queue.

---

## M5 — Impact, validation, improvement, limits (8:40–10:50, ≈300 words)

**[Slide 24, 35 s]** What changes for the queue? Ordered by risk, the first forty percent of slots hold **76.6 percent**
of sick patients, against forty in arrival order. Per 1,000 patients, 368 healthy ones stay out of the priority queue,
and **45.5 sick patients remain in Low**; that is the cost of our own method. And the word is **simulated**: the data
has no arrival order or waiting time.

**[Slide 25, 35 s]** Two honest checks. The locked test set: **28 of 28** sick patients flagged, but on 61 patients the
confidence interval starts at 0.877. And Budapest, 294 patients, no refitting: ROC-AUC **0.877**, recall **0.887**,
**below our 90 percent target**. We say so because that's the number that matters in deployment.

**[Slide 26, 15 s]** Is the tuned score flattering itself? Nested cross-validation gives **0.909** against 0.900:
optimism of minus 0.008. A hundred-setting search added nothing.

**[Slide 27, 25 s]** So what helps? The learning curve is **still rising**, so more data does. And **six sick patients
are missed by all four models**. No fifth algorithm finds them; new inputs might, like troponin or a calcium score.

**[Slide 28, 20 s]** What we wouldn't trust yet: recall for women is **0.833** against 0.914 for men, on 18 sick women.
One hospital, the 1980s, referred patients only, and the timing assumption needs a modern cohort.

*[bridge]* **M6**, show them.

---

## M6 — Live demo and close (10:50–13:00, ≈290 words)

**Live demo, 1:35** — follow [`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md), about 180 spoken words over the screen capture.
Highlights to hit: High 97.5 percent; Low 2.1 percent with the safety net; Borderline 30.3 percent Medium; heart rate
to 100 moves it to High; `ca` to 2 does **not** move the main score; the missed patient at 5.5 percent; age 10 refused.

**[Slide 29, 18 s]** Three numbers to take away. **Nine in ten** patients with disease flagged. **368 per thousand**
healthy patients kept out of the priority queue, simulated. And **0.877** at a hospital the model had never seen.

**[Slide 30, 10 s]** One sentence. **Decision support, not diagnosis.** It doesn't diagnose or discharge anyone; a
cardiologist makes every call.

**[Slide 31, 8 s]** That was *[all six names]*. Everything is in the repository, including the audit that started it.
Thank you.

---

## Rehearsal checklist

- [ ] Each block read aloud with a timer, within ±8 s of 2:10.
- [ ] Every number matches the rebuilt artifact (`artifacts/metrics.json`, `DEMO_RUNSHEET.md`).
- [ ] "Simulated" audible on slides 24 and 29.
- [ ] Each speaker names the next at the hand-off.
- [ ] Demo recorded twice, each take 1:35 or under.
- [ ] Full playback at 1x totals 12:45–13:30.
