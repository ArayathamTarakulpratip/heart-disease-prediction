# Question bank — what judges and the lecturer will ask

Answers are short on purpose: **one sentence of claim, one number, one caveat.** That shape is what makes an answer
sound rehearsed in the good way. Depth is in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) and
[`ML_CONCEPTS.md`](ML_CONCEPTS.md).

Each question is tagged with the member who should field it first (see [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md)).
Anyone may answer anything — but somebody must answer *immediately*, and silence costs more marks than an imperfect
answer.

---

## A. The hostile questions — the ten that can actually hurt you

**A1. Isn't `ca` basically the answer? You're predicting an angiogram result from an angiogram finding.** *(M4)*
> Yes, and that is why the shipped model doesn't use it. `ca` is vessels coloured by fluoroscopy and UCI gives no
> timestamp proving it was available before the target procedure, so we dropped `ca`, `thal` and `slope`. ROC-AUC fell
> from 0.900 to 0.844. The 13-feature model stays in the report as a retrospective comparison only, and the app's main
> score ignores it — we show that live by setting `ca` to 2 vessels and watching the main score not move.

**A2. Your test recall is 28 out of 28 — that's 100%. Isn't that suspicious?** *(M6)*
> It is 28 of 28 on 61 patients, so the exact 95% confidence interval runs from 0.877 to 1.000 — consistent with a true
> recall of 88%. We quote the cross-validated 0.901 and the separate-fold 0.901 as our real numbers; the test set is a
> consistency check, not the headline.

**A3. Why should I believe your 0.900 isn't just tuning luck?** *(M3/M5)*
> Nested cross-validation. We re-ran the whole tuning procedure inside five outer folds: 0.909 unbiased versus 0.900
> tuned, an optimism of minus 0.008. If anything our number is conservative.

**A4. This is 1980s data from one hospital. Why does it matter today?** *(M2)*
> It doesn't prove a modern deployment; it proves the method. The problem is current — 148,882 people in England were
> waiting 18 weeks or more for cardiac care in September 2025, and Thai ischaemic heart disease mortality rose from
> 23.4 to 26.9 per 100,000 in a decade. Our next step is a prospective silent pilot on modern records; nothing here
> goes near a patient first.

**A5. You claim you'd save $1.37 million. Prove it.** *(M6)*
> We don't claim it — the slide says "simulated". It is US Medicare 2026 payments, not hospital costs, excludes the
> physician fee, and assumes Low-tier patients get no angiogram, which is *not* what our Low tier recommends. The
> transferable number is 45.5 patients with disease left in Low per 1,000 referred — and that is a cost, not a saving.

**A6. Logistic regression? That's not AI, it's 1950s statistics.** *(M5)*
> It is the model the evidence chose. We tested five families on identical folds; the spread after tuning is 0.006 and
> logistic regression versus XGBoost gives p = 0.81 — statistically indistinguishable. At the clinical operating point
> it is also the best, sparing 89 of 131 healthy patients against XGBoost's 88 and random forest's 74. Picking the
> interpretable member of a statistical tie is the right engineering decision, and our learning curve says the limit is
> the 242 patients, not the algorithm.

**A7. Your model is worse for women. Why would anyone deploy that?** *(M3)*
> Recall is 0.833 for women against 0.914 for men, and we put it on a slide and in the app. The intervals overlap, so
> we can't prove a gap — and we can't rule one out. We also can't call it an independent fairness estimate because the
> threshold was chosen on these patients. It is a blocker for deployment, it is on the roadmap, and it points the same
> way as the literature: women under 55 had 6.7 times the odds of being sent home with a missed cardiac event.

**A8. Why not deep learning?** *(M5)*
> 303 patients of structured tabular features is the regime where regularised linear and tree models match or beat
> neural networks, and a clinician needs a per-feature reason for the score. A network on 242 rows would be variance
> with no explanation. Our learning curve shows the binding constraint is sample size — a bigger model class cannot fix
> that.

**A9. Only 303 patients. Isn't that too small to conclude anything?** *(M3)*
> It is small, which is why we report mean ± SD over 50 folds rather than one split, use bootstrap and exact binomial
> confidence intervals, and ran the learning curve — still rising at 242 patients, so more data is our strongest lever.
> We also validated externally on 294 patients at a second hospital, which is a stronger check than any single split.

**A10. What happens if your model is wrong about a real patient?** *(M6)*
> A Low tier is not a discharge: it requires clinician review before any deferral, a documented follow-up date, a named
> responsible contact, and instructions to return urgently if chest pain is new or worsening. Suspected acute coronary
> syndrome goes on the urgent pathway whatever the score says. And the data contract refuses to score an implausible
> patient rather than guessing — we show that live with age 10. Decision support, not diagnosis.

---

## B. Dataset and problem framing

**B1. What exactly is the target?** *(M3)* — `disease = 1` when UCI's `num > 0`, i.e. more than 50% narrowing in at
least one major coronary vessel on angiography. `num` is 0–4; we binarise because the clinical decision is binary and
139 positives split four ways is too thin to model.

**B2. Why binarise instead of predicting 0–4?** *(M3)* — The decision we support ("priority angiogram or not") is
binary; the 1–4 classes have as few as a dozen patients each; and no clinical action differs between `num` = 2 and 3 in
our use case. A multiclass version is possible but would need far more data.

**B3. How many features and of what kinds?** *(M3)* — 13: 5 numeric, 1 ordinal (`ca`), 4 nominal, 3 binary. The shipped
model uses 10 — all except `ca`, `thal`, `slope`.

**B4. Why did you choose those valid ranges?** *(M3)* — They are physiological plausibility bounds, deliberately wider
than the observed data (observed BP 94–200, allowed 60–260). A contract set to the observed range would reject any
slightly unusual new patient; this one rejects only *impossible* values — typos, unit errors and placeholder codes.
That is exactly what caught `ca = 4` and `thal = 0` in the Kaggle file.

**B5. Why is the Kaggle copy wrong, and how do you know?** *(M1)* — We matched all 303 rows back to UCI on six
continuous fields; 303 of 303 matched. The label crosstab is perfectly anti-diagonal: `target` = 1 ↔ no disease (165),
`target` = 0 ↔ disease (138). Four categorical columns are systematically permuted — our documented recode reproduces
UCI exactly, with zero mismatches. Missing values appear as `ca = 4` and `thal = 0`. And patient 184 is missing, with
one duplicate row in her place.

**B6. Is the data balanced?** *(M3)* — Nearly: 139 of 303 (45.9%) have disease. So no resampling. But "always predict no
disease" is 54.1% accurate with 0% recall — which is why recall, not accuracy, is the target.

**B7. Why not use the Switzerland and Long Beach VA files too?** *(M3)* — They have far more missing data, including
most of the same columns. Pooling all four hospitals with missing-indicator features is roadmap item 1; doing it
properly requires handling site effects, which is a bigger job than this proof of concept.

---

## C. EDA

**C1. What was the most surprising pattern?** *(M4)* — Asymptomatic chest pain has the *highest* disease rate, 72.9%,
against 30.4% for textbook typical angina. It looks backwards until you remember these are patients already referred
for angiography, so the asymptomatic group were referred for some other reason. It is a spectrum-bias effect, and it
is why we one-hot encode `cp` instead of treating the codes as a scale.

**C2. Blood pressure and cholesterol are significant but you called them weak. Which is it?** *(M4)* — Both: Mann-Whitney
p = 0.026 and 0.035, yet median blood pressure is 130 in *both* groups. Statistically detectable, clinically
near-useless on their own. We kept them as inputs and tested removing them by drop-column CV; removal changed nothing
measurable.

**C3. Did EDA change any modelling decision?** *(M4)* — Every one of them. §5.8 is a table mapping each finding to a
choice: imaging strength → the input-timing pivot; near-balance → no resampling; the female under-representation →
subgroup recall checks; the chest-pain paradox → one-hot encoding; feature scale differences → standardisation; weak
classics → drop-column testing. The biggest was finding that our strongest feature might be unusable.

**C4. How many EDA charts, and why those?** *(M4)* — Nine charts across seven sub-sections: demographics, chest pain,
age vs max heart rate, the exercise stress test, imaging, a correlation view, and the weak classic risk factors. Each
one has a stated hypothesis before it and a numbered insight after it; none is decorative.

---

## D. Modelling, validation, tuning

**D1. How did you split the data?** *(M5)* — Stratified 80/20: 242 train, 61 locked test, both 45.9% positive. The test
set is opened exactly once, in §11. Everything else happens on the 242 with RepeatedStratifiedKFold, 5 splits × 10
repeats, seed 42 — 50 folds, identical for every model.

**D2. Why repeated stratified k-fold rather than a plain holdout?** *(M5)* — With 61 test patients, one patient moves
accuracy by 1.6 points. Fifty folds give a mean *and* a standard deviation, stratification keeps the class rate stable
in every fold, and using the same folds for every model makes the comparison paired, which is what licenses the
significance test.

**D3. Where does tuning happen, and on which folds?** *(M5)* — `RandomizedSearchCV`, 30 settings per model, scored on a
5-fold × 2-repeat CV of the *training set only*, with a different seed from the 50 comparison folds. The search's own
`best_estimator_` is used directly — the original notebook re-typed the best `C` wrongly.

**D4. Why randomised search rather than grid search?** *(M5)* — Seven XGBoost hyperparameters make a grid
combinatorially impossible, and random search finds comparable optima in a fixed budget because usually only a few
hyperparameters matter. We then verified the budget: 100 settings instead of 30 added 0.000 to logistic regression and
0.003 to XGBoost.

**D5. Which model did you select, and on what rule?** *(M5)* — Tuned logistic regression, `C` = 0.216. Rule stated in
advance: best CV ROC-AUC wins, but if logistic regression is within one standard deviation of the best, ship logistic
regression — because it is interpretable, calibrated, and gives exact per-patient contributions. It was in fact best
(0.900), and the gap to XGBoost is p = 0.81.

**D6. What's your primary metric and why?** *(M2/M5)* — Recall on the disease class, target ≥ 0.90, because a missed
sick patient waits and waiting is measurably dangerous — 1.4% of 8,030 Ontario patients had a major event on the list.
Specificity, ROC-AUC and the Brier score are constraints; accuracy is reported only, because the 54.1% baseline shows
it is misleading here.

**D7. Why compare on ROC-AUC if you optimise recall?** *(M5)* — Recall depends on the threshold, so comparing models by
recall compares threshold choices, not models. ROC-AUC is threshold-free and measures ranking quality, which is exactly
what a queue-ordering problem needs. We then *also* compare at the clinical operating point — all four models forced to
recall 0.901 — to check the ranking survives the decision.

**D8. How do you know logistic regression is really better than XGBoost?** *(M5)* — We don't, and we say so. Corrected
resampled t-test p = 0.81. They are indistinguishable on this data; we ship the interpretable one.

**D9. Why a *corrected* t-test?** *(M5)* — Repeated CV folds share training data, so fold scores aren't independent and
the ordinary t-test is over-confident. Nadeau & Bengio's correction inflates the variance by the validation-to-training
size ratio. The only difference that survives it is dropping `ca` (p = 0.016).

**D10. Did feature engineering help?** *(M4)* — No, and we report that. Four clinical features — percentage of
age-predicted max heart rate, ST depression > 1 mm, hypertension and high-cholesterol flags — changed ROC-AUC by −0.004
for logistic regression (p = 0.13) and −0.008 for random forest (p = 0.15). The models already get the raw values, so a
fixed cut-point adds nothing. We dropped them.

**D11. Did you try feature selection?** *(M3)* — Drop-column CV on all 13. Only `ca` mattered (−0.030, p = 0.016). A
7-feature variant scored +0.007 with p = 0.47, so we kept all 13: not reliably better, and selection on 242 patients is
noisy.

**D12. Is your model overfitting?** *(M3)* — The learning curve says no, but it is variance-limited: training ROC-AUC
falls from 0.964 at 38 patients to 0.929 at 193 while validation rises from 0.861 to 0.900. The curves converge and
validation is still climbing — the classic signature of "needs more data", not "needs a smaller model".

**D13. How did you handle missing values?** *(M3)* — Six cells of 3,939, all in `ca` and `thal`. Median imputation for
numeric and most-frequent for categorical, **inside the pipeline**, so the imputer is fitted on each training fold and
never sees the validation fold. We kept all six patients; dropping 2% of a 303-patient dataset costs more than it buys.

**D14. Why one-hot encode some columns and not `ca`?** *(M3)* — `cp`, `restecg`, `slope` and `thal` are nominal: their
integers are labels, so leaving them numeric imposes a false ordering. `ca` is genuinely ordinal — 0 < 1 < 2 < 3
vessels — and the disease rate rises monotonically from 26.1% to 85.0%, so the ordering is real information.

**D15. Why standardise if you're using trees?** *(M3)* — Trees don't need it, but KNN measures distance and
L2-regularised logistic regression penalises coefficients, both of which require comparable scales. One identical
pipeline for all five families is what keeps the comparison fair.

---

## E. Solution, decision layer, deployment

**E1. How do you get from a probability to a decision?** *(M5)* — Two cut-offs chosen from out-of-fold predictions on
the training set: the Low cut-off is the highest threshold still catching ≥ 90% of sick patients (0.263 for the shipped
model), the High cut-off is the lowest at which ≥ 85% of those above it truly have disease (0.662). Below Low →
clinician review plus a safety-net plan; between → cardiology review; above High → priority angiography.

**E2. Why not just use 0.5?** *(M5)* — Because 0.5 is a convention, not a clinical choice. At 0.5 the model misses 9.9
sick patients per 100; at our cut-off it misses 4.5. Moving the threshold halves the error that matters and costs no
modelling at all.

**E3. Isn't choosing the threshold on the same data you evaluate on cheating?** *(M5)* — It would be, so we validated
the whole rule separately: each of five outer folds retunes `C` *and* re-chooses both cut-offs on inner out-of-fold
predictions, then scores untouched patients. Recall 0.901, specificity 0.542, High-tier PPV 0.857. That is the number
we quote.

**E4. Is acting on the model actually better than just scanning everyone?** *(M6)* — Decision-curve analysis says yes
across the whole plausible range: net benefit beats both "angiogram for everyone" and "angiogram for nobody" at every
risk threshold from 0.05 to 0.80. At 0.30 it is 0.340 against 0.227 for angiograms-for-all.

**E5. What does external validation show?** *(M6)* — The 10-feature model, trained on Cleveland and scored on 294
Budapest patients with no refitting: ROC-AUC 0.877, recall 0.887. Recall is *below* our 0.90 target, and we report that
as a gap. The 13-feature model can't be externally validated at all — Budapest recorded 99% of `ca` as missing.

**E6. Are the probabilities trustworthy, or just well-ordered?** *(M5)* — Both. Brier score 0.126 in CV and 0.084 on
the test set, with a calibration curve. We tested Platt scaling and it made calibration *worse* (0.1275), so we didn't
add it — logistic regression optimises a proper scoring rule, so it tends to be calibrated already.

**E7. What would deployment actually require?** *(M6)* — A prospective silent pilot running alongside clinicians without
acting on the output; per-hospital re-calibration; drift monitoring on inputs and outcomes; EHR integration with the
typed input contract we already have in `data.validate`. None of that is done, which is why this is labelled a proof of
concept.

**E8. What's in the saved artifact?** *(M6)* — The fitted pipeline, both thresholds, the average-patient baseline for
contributions, the feature list, the metrics, and the four demo presets. The notebook asserts that the reloaded
pipeline reproduces all 61 test predictions, so the app and the training run cannot disagree.

---

## F. Improvement

**F1. If you had one more week, what would you do?** *(M3)* — Pool the other UCI hospitals with missing-value
indicators, because the learning curve is still rising. Not a new algorithm — six sick patients are missed by all four
tuned models, so the gap is information, not capacity.

**F2. What would you *not* bother doing?** *(M3)* — A wider hyperparameter search (100 settings added 0.000 to logistic
regression), more feature engineering (negative ablation), or feature selection (+0.007 at p = 0.47).

**F3. What about ensembling or stacking?** *(M5)* — On the roadmap: a stacked logistic regression + XGBoost, or XGBoost
with monotonic constraints so risk can only rise with more diseased vessels. We'd adopt it only if it beats the shipped
model *at the recall-0.90 operating point*, not on ROC-AUC — that is where the decision is made.

**F4. How would you close the sex gap?** *(M3)* — Collect more sick women, review every missed female case individually,
and re-check subgroup recall after every single change. Possibly sex-specific thresholds, but not on 18 sick women —
that would overfit the gap rather than close it.

**F5. Could you set the threshold from costs instead of a fixed 90%?** *(M5)* — Yes, and that is roadmap item 5: elicit
the cost of a missed patient versus an unnecessary angiogram from clinicians and derive the cut-off from the ratio.
The 0.90 target is a stand-in for a cost ratio nobody has given us yet.

---

## G. Process, tooling, and "who did what"

**G1. Why a Python package instead of one notebook?** *(any)* — So the notebook, the app and the tests import the *same*
code. The demo cannot disagree with the model we trained, the logic is unit-testable, and the notebook stays a readable
story of markdown plus one-line calls. Import direction is one-way, enforced by a test.

**G2. How do you know your notebook's claims match its outputs?** *(any)* — `tests/check_notebook.py` fails if the
executed notebook has any error or warning, or if an "Insight" sentence quotes a number the cells did not print.

**G3. Is it reproducible?** *(any)* — One seed in `config.py` used by the split, folds, searches and models; pinned
`requirements.txt`; `nbconvert --execute` rebuilds every artifact; `tests.check_notebook` must print `clean`.

**G4. What's the hardest thing you learned?** *(any — answer honestly)* — That the most valuable finding was negative.
Our best feature was the one we had to delete, and admitting that cost 0.056 of ROC-AUC but produced the only model that
can be validated at a second hospital.

---

## H. If you genuinely don't know

Say this, then stop:

> "That isn't something we measured. What we *can* say is [nearest thing you did measure], and it would need
> [the specific check] to answer properly."

Never invent a number. One honest "we didn't measure that" costs almost nothing; one fabricated figure that a judge
can check loses the room.
