# Script — 13:00 at 1x, six speakers

Replace `M1`–`M6` with real names. Every speaker appears on camera for their own segment and says the next speaker's
name at the hand-off, so the contribution marks are evidenced in the video itself.

**Delivery rules**
* ~140 words per minute. The word counts below are sized for that; if you run long, cut an adjective, never a number.
* Say every number **once**, slowly, and let it sit on screen for a beat.
* Say the word "simulated" out loud wherever the script has it. It is the difference between a finding and a fabrication.
* Never read the slide. The slide carries the number; you carry the sentence.
* Record with a visible timer. 13:00 target, 13:30 hard stop, 15:00 cap.

**Running order**

| Time | Who | Slides | Segment |
|---|---|---|---|
| 0:00–1:10 | **M1** | 1–3 | Hook: the audit |
| 1:10–3:00 | **M2** | 4–9 | Q3 why it matters + Q1 objective |
| 3:00–4:40 | **M3** | 10–12 | Q1 data, QA, metric |
| 4:40–6:40 | **M4** | 13–17 | Q2 EDA + the pivot |
| 6:40–8:50 | **M5** | 18–23 | Q4 models + Q5 decision layer |
| 8:50–11:30 | **M6** | 24–25 + live app | Q5 impact, validation, **demo** |
| 11:30–12:20 | **M3** | 26–28 | Q6 improvement + limitations |
| 12:20–13:00 | **M1** | 29–31 | Close + credits |

---

## M1 — Hook (0:00–1:10)

**[Slide 1, hold 5 s]**

> Good afternoon. We are team *[name]*, and we built a triage model for the coronary angiography queue.
> But we want to start with something we found in the data, because it changed the whole project.

**[Slide 2 — hero "184", hold 15 s]**

> This is patient 184. A sixty-year-old woman. Asymptomatic chest pain. And on angiography, she has coronary disease.
> She exists in the original UCI Cleveland file. She does **not** exist in the copy of this dataset that most people
> actually download — the Kaggle version, with over a thousand public notebooks written on it. She was dropped, and a
> duplicate of a healthy patient was put in her place.

**[Slide 3 — the crosstab, hold 20 s]**

> Then we matched all three hundred and three rows back to the original. And here is the result.
> Every single label in that copy is **inverted**. In the Kaggle file, `target` equals one means the patient has *no*
> disease. Not one row disagrees. A hundred and thirty-eight sick patients labelled healthy, a hundred and sixty-five
> healthy patients labelled sick.
>
> So before we tell you what we predict, we had to establish what we were predicting *from*. Everything you see next
> uses the original UCI file.
>
> *[bridge]* **M2** will tell you why this queue is worth getting right.

*(≈165 words)*

---

## M2 — Why it matters, and what we are actually predicting (1:10–3:00)

**[Slide 4, hold 10 s]**

> Cardiovascular disease kills about **nineteen point eight million** people a year — roughly a third of all deaths
> on earth. That is the World Health Organization's 2022 figure. That is the headline everyone quotes, and on its own
> it does not tell you what to build.

**[Slide 5, hold 20 s]**

> So here is the number that does. In Hamilton, Ontario, eight thousand and thirty consecutive patients were referred
> for cardiac catheterisation. **One point four percent** of them — a hundred and nine people — died, had a heart
> attack, or developed heart failure *while they were still on the waiting list*. Fifty of them died. The median wait
> was sixty days, and only thirty-seven percent of procedures happened inside the window the referring doctor asked
> for. And this is not history: at the end of September 2025, a hundred and forty-eight thousand people in England had
> been waiting eighteen weeks or more for routine cardiac care.

**[Slide 6, hold 20 s]**

> Now the other half of the problem. Of nearly four hundred thousand American patients sent for an elective
> angiogram, only **thirty-seven point six percent** turned out to have obstructive disease. The majority of those
> procedures found nothing to treat. Each one is an invasive test with about eight major complications per ten
> thousand, and Medicare pays three thousand three hundred and twelve dollars for it next year.
>
> So the queue is simultaneously too slow for the people who need it and too full of people who don't.

**[Slide 7, hold 12 s]**

> And this is not someone else's problem. In Thailand, age-standardised deaths from ischaemic heart disease rose from
> **twenty-three point four to twenty-six point nine** per hundred thousand in a decade — and the country has
> seventy-two hospitals able to perform coronary intervention. About one per million people.

**[Slide 8, hold 18 s]**

> Which means the two mistakes our model can make are not equal. If we defer a patient who has disease, they wait, and
> the Ontario number is what waiting costs. If we prioritise a patient who turns out to be healthy, they take a slot
> and accept a small procedural risk. The first error is severe; the second is tolerable.
>
> So **recall is our primary metric** — the share of patients with disease that we flag. Not accuracy. On this data a
> model that says "nobody is sick" is already fifty-four percent accurate and catches nobody.

**[Slide 9, hold 15 s]**

> Here is the objective in one sentence. From ten results a clinician already has, estimate the probability that
> angiography will show more than fifty percent narrowing in at least one major vessel, and place the patient in a
> tier — flagging at least **ninety percent** of the patients who truly have disease.
>
> Inputs, a data contract, a risk, a tier, an action. A clinician makes the call.
>
> *[bridge]* **M3**, what is in those ten inputs?

*(≈430 words)*

---

## M3 — The data, the contract, and the metric (3:00–4:40)

**[Slide 10, hold 20 s]**

> The UCI Heart Disease dataset, Cleveland subset. **Three hundred and three patients, thirteen clinical attributes**,
> collected at the Cleveland Clinic Foundation and published in 1989. Five numeric — age, resting blood pressure,
> cholesterol, peak heart rate, ST depression. One ordinal: the number of major vessels seen on fluoroscopy. Four
> categorical, including chest-pain type and the thallium scan. Three binary.
>
> The label is the angiographic result. UCI records it from zero to four; we use the standard binary form —
> `disease` equals one when that number is greater than zero, meaning more than fifty percent narrowing.

**[Slide 11, hold 25 s]**

> Before any analysis, we wrote a **data contract**: every patient must pass it or they are not scored. Age between
> eighteen and a hundred, blood pressure sixty to two-sixty, cholesterol eighty to seven hundred, heart rate fifty to
> two-fifty, ST depression zero to ten. These are physiological plausibility bounds, not the observed range — we set
> them wide on purpose, so the contract catches data-entry errors without rejecting real patients. All three hundred
> and three UCI patients pass. Six missing cells out of nearly four thousand, all in the two imaging columns, and
> they're imputed *inside* the model pipeline so no information leaks between folds. Zero duplicates.
>
> And when we feed the Kaggle file to the same contract, it is **rejected** — a hundred and forty-three invalid
> chest-pain codes, a hundred and eighty-six invalid thallium codes, and five patients whose missing vessel count was
> disguised as the number four.

**[Slide 12, hold 15 s]**

> Last thing before the patterns. A hundred and thirty-nine of the three hundred and three patients have disease —
> **forty-five point nine percent**. So this is close to balanced, which sounds convenient and is actually a trap:
> "always predict no disease" scores fifty-four point one percent accuracy with **zero** recall. That is why we
> optimise recall and report accuracy only as a footnote.
>
> *[bridge]* **M4** will show you what the data actually says.

*(≈400 words)*

---

## M4 — What the data says, and the decision it forced on us (4:40–6:40)

**[Slide 13, hold 18 s]**

> Four patterns. First, sex. In this cohort **fifty-five point three percent** of men have disease against
> **twenty-five point eight percent** of women — and only ninety-seven of the three hundred and three patients are
> women. Remember that; it comes back at the end as a limitation we have not solved.

**[Slide 14, hold 25 s]**

> Second, and this is the one that surprised us. The highest-risk chest-pain category is **asymptomatic** —
> seventy-two point nine percent of those patients have disease, against thirty percent for textbook typical angina.
> That looks backwards. It isn't: everyone here was *already referred for angiography*, so the asymptomatic group are
> people a cardiologist was worried about for some other reason. It's a reminder that this is a referred population,
> not people off the street — and it's why we one-hot encode chest pain instead of treating those codes as a number.

**[Slide 15, hold 25 s]**

> Third, the exercise test earns its place. Angina brought on by exercise: **seventy-six point eight percent** have
> disease, versus thirty-one percent without. Peak heart rate averages a hundred and thirty-nine beats against a
> hundred and fifty-eight. And the classic risk factors — blood pressure and cholesterol — barely move. Both are
> statistically significant, but the median blood pressure is a hundred and thirty in *both* groups. Significant is
> not the same as useful.

**[Slide 16, hold 15 s]**

> Fourth, the strongest signal in the whole dataset: the imaging. Zero vessels on fluoroscopy, twenty-six percent have
> disease. Three vessels, **eighty-five percent**. The highest correlation with the outcome of any feature, 0.46.

**[Slide 17, hold 35 s]**

> And that is where we had a problem. Because `ca` — vessels coloured by fluoroscopy — **may be measured during the
> very angiogram we are trying to prioritise**. UCI records no timestamps, so we cannot prove it was available
> earlier. If we ship that model, we have built something that needs the answer in order to produce the answer.
>
> So we deleted our three strongest-timing-risk inputs: vessels, the thallium scan, and the ST slope. Cross-validated
> ROC-AUC fell from **0.900 to 0.844**. We paid that on purpose.
>
> And here is what we bought. The thirteen-feature model **cannot be externally validated at all**, because the second
> hospital in the UCI collection recorded ninety-nine percent of its vessel counts as missing. The ten-feature model
> can — and it scores **0.877** there. The honest model is the only one that can be checked.
>
> *[bridge]* **M5**: so which algorithm?

*(≈480 words)*

---

## M5 — Models, comparison, tuning, and the decision layer (6:40–8:50)

**[Slide 18, hold 20 s]**

> With three hundred and three patients, a single train-test split is noise — one patient moves accuracy by one and a
> half points. So: we lock away a stratified twenty percent, sixty-one patients, and open it **once**, at the end.
> Everything else happens on the remaining two hundred and forty-two with five-fold cross-validation repeated ten
> times — fifty folds, the same fifty folds for every model. All preprocessing sits inside the pipeline, refitted on
> each fold, so nothing leaks.

**[Slide 19, hold 25 s]**

> Five families. **Dummy** predicts the base rate — that's the floor, and without it no score means anything.
> **Logistic regression**: a weighted sum of the inputs squeezed through a sigmoid; a linear decision boundary, and
> every weight is readable. **K-nearest neighbours**: no model at all, just a vote among the forty-six most similar
> patients. **Random forest**: hundreds of decision trees on bootstrap samples, averaged, which cancels their
> variance. **XGBoost**: trees fitted one after another, each correcting the previous one's errors.
>
> No deep learning, deliberately. Three hundred patients of structured clinical data is where linear and tree models
> win, they run in milliseconds, and a clinician can see why a patient scored high.

**[Slide 20, hold 25 s]**

> Same fifty folds, after tuning. Logistic regression **0.900**, XGBoost 0.897, KNN 0.896, random forest 0.894. Dummy,
> 0.500, exactly as it should be.
>
> Now, the gap between first and second is 0.003. We tested it with a corrected t-test — corrected because repeated
> folds overlap and the ordinary test is over-confident — and got **p equals 0.81**. There is no detectable difference.
> So we ship the model that is interpretable and well calibrated. Simplest model that ties, wins.

**[Slide 21, hold 15 s]**

> Tuning was a randomised search of thirty settings, scored inside cross-validation, on *different* folds from the
> comparison. It helped the models that needed it — KNN plus 0.030, XGBoost plus 0.034 — and barely touched logistic
> regression, plus 0.007. It did not change the ranking.

**[Slide 22, hold 20 s]**

> But ROC-AUC is not where the decision is made. So we forced all four models to the same clinically relevant
> operating point — ninety percent recall — and compared what they cost. At identical recall, logistic regression
> correctly spares **eighty-nine of a hundred and thirty-one** healthy patients. XGBoost eighty-eight, KNN
> seventy-nine, random forest seventy-four. Same recall, fifteen fewer unnecessary priority referrals.

**[Slide 23, hold 25 s]**

> Finally: a probability is not a decision. We turn it into three tiers. **Low below twenty-six point three percent.
> High from sixty-six point two.** The low cut-off is the highest threshold that still catches ninety percent of sick
> patients; the high cut-off is the lowest at which eighty-five percent of those above it genuinely have disease.
> Both chosen from out-of-fold predictions, never the test set.
>
> And we validated the *whole rule* — retuning the model and re-choosing both cut-offs inside each outer fold.
> Recall **0.901**, specificity 0.542, and the High tier is **eighty-five point seven percent** true positives.
>
> One thing matters more than any of those numbers: **Low is not a discharge.** It requires clinician review, a
> documented follow-up date, a named contact, and instructions to return urgently if the pain changes.
>
> *[bridge]* **M6**, show them what it does.

*(≈520 words)*

---

## M6 — Impact, validation, and the live demo (8:50–11:30)

**[Slide 24, hold 25 s]**

> What changes for the queue? If you order by risk, the first forty percent of slots contain **seventy-six point six
> percent** of the patients who have disease, against forty percent if they just arrive in order. Per thousand
> referred patients: three hundred and sixty sick patients in the High tier, three hundred and sixty-eight healthy
> patients kept out of the priority queue — and **forty-five point five sick patients still sitting in Low**. We show
> that number because it is the cost of our own method.
>
> And the word on this slide is **simulated**. This dataset records no arrival order, no waiting time, and no decision
> to withhold a test. We are not claiming we shortened anyone's wait.

**[Slide 25, hold 25 s]**

> Two honest checks. The locked test set, opened once: **twenty-eight of twenty-eight** patients with disease flagged,
> exact confidence interval 0.877 to 1.0, ROC-AUC 0.961 — on sixty-one patients, so that interval is wide and we quote
> it. And the external hospital: two hundred and ninety-four patients in Budapest, no refitting, ROC-AUC **0.877**,
> recall **0.887** — which is **below our ninety percent target**. We are telling you our model missed its own target
> at the second hospital, because that is the number that would matter if you deployed it.

**[LIVE DEMO — 2:05, follow `DEMO_RUNSHEET.md` steps 0–6, no slide]**

> *[closing the demo]* Decision support, not diagnosis — and suspected acute coronary syndrome goes on the urgent
> pathway whatever this screen says.
>
> *[bridge]* **M3**: how do we make it better?

*(≈240 words + demo)*

---

## M3 (second slot) — Improvement and limitations (11:30–12:20)

**[Slide 26, hold 15 s]**

> First, is our tuned score flattering itself? We re-ran the entire tuning procedure inside five outer folds — nested
> cross-validation. Logistic regression scored **0.909** unbiased against 0.900 tuned. The optimism is **minus 0.008**
> — our number is, if anything, slightly pessimistic. And a hundred-setting search instead of thirty added exactly
> **zero**. More search is not the lever.

**[Slide 27, hold 20 s]**

> So what is? The learning curve is **still rising** at two hundred and forty-two patients, so more data is. Dropping
> features is not — only vessels mattered, and a smaller variant gained 0.007 at p equals 0.47. And the honest finding:
> **six sick patients are missed by all four tuned models.** The same six. A fifth algorithm will not find them. New
> *inputs* might — troponin, symptom duration, a coronary calcium score.

**[Slide 28, hold 15 s]**

> What we would not yet trust. Recall for women is **0.833** — fifteen of eighteen — against 0.914 for men, and the
> cut-off was chosen on these same patients, so that is descriptive, not a fairness guarantee. One hospital, in the
> 1980s, patients already referred, three hundred and three in total. And the input-timing assumption needs checking
> in a modern cohort with timestamps before anyone goes near a patient with this.
>
> *[bridge]* **M1** will close.

*(≈230 words)*

---

## M1 — Close (12:20–13:00)

**[Slide 29, hold 18 s]**

> Three numbers to take away. **Nine in ten** patients with disease flagged, under separate-fold validation.
> **Three hundred and sixty-eight healthy patients per thousand** kept out of the priority queue — simulated. And
> **0.877** at a hospital the model had never seen.

**[Slide 30, hold 10 s]**

> And one sentence. **Decision support, not diagnosis.** This is a proof of concept on 1980s data from one hospital.
> It does not diagnose anyone, it does not discharge anyone, and a cardiologist makes every call.
>
> What it does is put the right patient earlier in the queue — and tell you honestly how often it fails to.

**[Slide 31 — credits, hold 8 s]**

> That was *[six names]*. Everything you have seen is in the repository, including the audit that started it.
> Thank you.

*(≈130 words)*

---

## Rehearsal checklist

- [ ] Read aloud with a timer. Each segment within ±10 s of its slot.
- [ ] Every number in the script matches the rebuilt artifact (`artifacts/metrics.json`, `DEMO_RUNSHEET.md`).
- [ ] The word "simulated" is audible on slides 24 and 29.
- [ ] Each speaker names the next one at the hand-off.
- [ ] Demo recorded twice.
- [ ] Total runtime 12:45–13:30, played back at 1x.
