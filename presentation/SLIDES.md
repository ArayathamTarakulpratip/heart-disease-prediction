# Deck spec: 31 slides

**Count:** 31 — 1 title, 29 content, 1 credits. The live demo runs on screen capture with **no slide behind it**.
**Budget:** 13:00 total − 2:30 demo = 10:30 of slides ÷ 29 ≈ **22 s per slide**. Most are a single number and take 10–15 s;
the five chart slides take 35–45 s. [`SCRIPT.md`](SCRIPT.md) gives the hold time for each. If a slide needs more than
45 s, split it. If you must cut, drop slides 7, 21 and 26 in that order — nothing else is load-bearing.
**Every figure below already exists** in `heart_disease_pitch_gold_standard.ipynb`; export it, do not redraw it.

---

## Part A — global style (set once, never vary)

**Canvas** 16:9, 1920×1080. Export the video at 1080p.

**Grid** 96 px outer margin. One 12-column grid, 24 px gutter. A slide uses either the full width, or 7+5 columns
(chart left, text right). Never three columns.

**Colour tokens** (light deck — a recorded video is watched on phones, light reads better)

| Token | Value | Used for |
|---|---|---|
| `bg` | `#FFFFFF` | background, always explicit |
| `ink` | `#111827` | headings and body |
| `muted` | `#6B7280` | sources, captions, footers |
| `accent` | `#B91C1C` | the one number that matters on each slide, High tier |
| `accent-2` | `#1D4ED8` | the comparison number, Medium tier |
| `ok` | `#047857` | Low tier, "fixed" states |
| `rule` | `#E5E7EB` | hairlines, table borders |

Red is reserved for the number being argued. If two numbers appear, the one you want remembered is `accent`, the other
is `ink`. Never more than two coloured numbers on a slide.

**Type** One family, two weights. Inter / Source Sans / Arial — whatever is installed on the recording machine.

| Role | Size | Weight |
|---|---|---|
| Hero number | 180 pt | Bold |
| Slide title | 44 pt | Bold |
| Sub-claim | 28 pt | Regular |
| Table / body | 22 pt | Regular |
| Source line | 15 pt | Regular, `muted` |

**Slide anatomy** — the same four zones on every content slide:

```
┌─────────────────────────────────────────────────────┐
│ 44pt TITLE: the claim, as a sentence                │
│                                                     │
│   [ hero number 180pt ]    or   [ chart, 7 cols ]   │
│   28pt one-line reading of that number              │
│                                                     │
│ 15pt Source, year  ·  §x.y            n / 31   ▌M4 │
└─────────────────────────────────────────────────────┘
```

* Title is a **claim**, not a label. "Asymptomatic patients have the highest disease rate", not "Chest pain".
* Footer right carries the slide number and a 6 px `accent` bar with the current speaker's tag, so the editor can cut on
  speaker changes.
* **Lower third** on the first slide of each speaker's segment: name + role, 3 s, 32 pt on a 70%-opacity `ink` bar.

**Charts** Regenerate with `cardiotriage.plots.set_style()` so all figures share one look. Then: no gridlines except a
light horizontal one, no chart titles (the slide title is the title), axis labels in words not column names, direct
labels on the bars instead of a legend wherever there are ≤4 series, and the single value being discussed in `accent`
with every other bar in `rule`-grey. Minimum 22 pt effective text — check by viewing the slide at 25% zoom.

**Hard rules**
1. One claim per slide. If you can't say it in one sentence, it is two slides.
2. No code, no notebook, no terminal — except inside the demo capture.
3. No bullet list longer than 4 lines, no sub-bullets.
4. Every statistic carries source + year in the footer.
5. Every simulated number carries the word **"simulated"** on the slide itself, not just in speech.
6. No clip art, no stock photos of doctors, no stethoscope icons.

---

## Part B — slide-by-slide

Times are cumulative and match [`SCRIPT.md`](SCRIPT.md). "§" refers to notebook sections.

### M1 — Hook (0:00–1:10)

| # | Title (the claim) | Visual | Content / numbers |
|---|---|---|---|
| 1 | **Heart Disease Triage — prioritising the angiography queue** | Title only. Project name, 6 names, course, date | No subtitle longer than 9 words. Hold 4 s. |
| 2 | Patient 184 is a 60-year-old woman with heart disease. The most popular copy of this dataset lost her. | Hero: **184**, with the one-row table beneath (age 60, female, asymptomatic, `num`=1) | She is replaced by a duplicate of a healthy patient. §3.2 |
| 3 | And every label in that copy is inverted. | The 2×2 crosstab, large, with the two zero cells greyed and the two filled cells in `accent` | Kaggle `target`=0 ↔ UCI disease=1 (138); `target`=1 ↔ UCI no disease (165). 303/303 rows matched. §3.2 |

### M2 — Why it matters + objective (1:10–3:00)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 4 | Cardiovascular disease kills about 19.8 million people a year. | Hero: **19.8M**, `muted` sub-line "≈32% of all deaths" | WHO CVD fact sheet, 2022 data, updated 31 Jul 2025. §0 Layer 1 |
| 5 | The queue is where ranking pays off: 1 in 70 patients has a major event before their test. | Hero: **1.4%**, sub-line "109 of 8,030 died, had a heart attack or developed heart failure while waiting" | Natarajan, *CMAJ* 2002. Second line, 22 pt: England, 148,882 waiting ≥18 weeks, Sept 2025 (BHF/NHS England). §0 Layer 2 |
| 6 | And most of those tests find nothing to treat. | Hero: **37.6%**, sub-line "of 398,978 elective angiograms found obstructive disease" | Patel, *NEJM* 2010. Footer adds the unit price: $3,312 per angiogram, CMS CY2026 OPPS, CPT 93458. **Say only this one 37.6%.** |
| 7 | In Thailand the same pressure is rising. | Two numbers side by side: **23.4 → 26.9** per 100,000 (`accent`) and **72** PCI hospitals ≈ 1 per million (`ink`) | Sakboonyarat & Rangsin, *BMC Res Notes* 2024; Thai PCI Registry 2018–19. §0 Layer 3 |
| 8 | The two errors are not equal, so recall is the metric. | 2-row table, FN row in `accent` | FN = sick patient deferred → **severe**, the error we minimise. FP = healthy patient takes a slot → moderate, accepted. Accuracy is misleading: "always say no disease" already scores 54.1%. §1, §4.4 |
| 9 | Our objective, in one sentence. | The decision-flow diagram, full width | *From 10 results a clinician already has, estimate P(>50% narrowing in ≥1 major vessel) and place the patient in a tier, flagging ≥90% of those who truly have disease.* `10 inputs → data contract → model → risk % → tier → action`. §1 |

### M3 — Data, QA, metric (3:00–4:40)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 10 | UCI Cleveland: 303 patients, 13 features, one angiographic label. | Hero: **303 × 13**; beside it the simplified dictionary (8 rows, grouped: 5 numeric / 1 ordinal / 4 nominal / 3 binary) | Target `disease` = `num > 0`. Janosi et al. 1989, doi:10.24432/C52P4X. §3, §3.1 |
| 11 | Before any analysis: a data contract every patient must pass. | Split slide — left: the valid-range table (age 18–100, BP 60–260, chol 80–700, HR 50–250, ST 0.0–10.0); right: the rejection message for the Kaggle file in `accent` | 303/303 UCI patients pass. 6 missing cells of 3,939 (`ca` 4, `thal` 2), 0 duplicates. Kaggle rejected: `cp` 143 invalid codes, `thal` 186, `slope` 21, `ca` 5. §4.1–4.3 |
| 12 | The classes are nearly balanced, so accuracy hides everything. | `class_balance` chart | 139 of 303 (45.9%) with disease. Baseline "always no disease": 54.1% accuracy, **0% recall**. §4.4 |

### M4 — EDA + the pivot (4:40–6:40)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 13 | Men in this cohort have twice the disease rate of women. | `rate_by(df, ["sex"])` | Male 55.3% (114/206), female 25.8% (25/97). Mean age 56.6 with disease vs 52.6 without. → we check recall by sex before claiming anything. §5.1 |
| 14 | The highest-risk chest pain is *no* chest pain. | `rate_by(df, ["cp"])`, the "asymptomatic" bar in `accent` | Asymptomatic **72.9%** (105/144) vs typical angina 30.4%, non-anginal 20.9%, atypical 18.0%. These are patients already referred, so this is referral spectrum, not the street. → one-hot encode `cp`. §5.2 |
| 15 | The exercise test separates the groups; blood pressure and cholesterol barely do. | Two panels: `exang`/`oldpeak` left, BP/chol box plots right | Exercise angina 76.8% vs 30.9%. Max HR 139 vs 158 bpm; 47.5% vs 15.9% below 85% of age-predicted max. BP p=0.026, chol p=0.035 — significant but medians almost identical (130/130, 249/234.5). §5.3, §5.4, §5.7 |
| 16 | The strongest signal of all is the imaging — and that is the problem. | `rate_by(df, ["ca","thal"])` with a `accent` arrow on the `ca` gradient | `ca`: 0 vessels 26.1% → 3 vessels 85.0%. Highest correlation with disease, 0.46. §5.5, §5.6 |
| 17 | `ca` may be measured **during** the angiogram we are trying to schedule. So we deleted it. | Two-column comparison table, the kept column in `ok` | Retrospective 13-feature: CV ROC-AUC **0.900**, cannot be externally validated (Hungary is 99% missing `ca`). Main 10-feature (no `ca`, `thal`, `slope`): CV ROC-AUC **0.844**, Hungary ROC-AUC **0.877**. We paid 0.056 for a model that can run before the procedure. §5.9, §13 |

### M5 — Models, tuning, decision layer (6:40–8:50)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 18 | 303 patients means one split is noise, so we score every model on the same 50 folds. | Split diagram: 242 train / 61 locked test, then 5-fold × 10 repeats | Test set opened **once**. All preprocessing inside the pipeline. Differences tested with a corrected t-test (Nadeau & Bengio 2003). §6 |
| 19 | Five families, one line of maths each. | 5-row table, one tiny schematic per row | Dummy (predicts the prior — the floor); Logistic regression (weighted sum → sigmoid); KNN (vote of the k nearest patients); Random forest (bagged trees, averaged); XGBoost (trees fitted to the previous trees' errors). Why no deep learning: n=303, tabular, interpretability required. §8, §1 |
| 20 | On the same folds, the simplest model is already the best. | `cv_boxplot` of ROC-AUC | Tuned: LR **0.900 ± 0.035**, XGBoost 0.897, KNN 0.896, RF 0.894, Dummy 0.500. LR vs XGBoost corrected t-test **p = 0.81** → not distinguishable, so ship the interpretable one. §8, §9.1 |
| 21 | Tuning moved the weak models, not the ranking. | Before/after bar pairs | ROC-AUC change: KNN +0.030, XGBoost +0.034, LR +0.007, RF +0.005. 30-setting randomised search inside CV, on different folds from the comparison. LR settles at C = 0.216. §9 |
| 22 | Judge models where the decision is made: at 90% recall. | `confusion_grid`, 4 panels, LR panel outlined in `accent` | All four forced to recall 0.901. Specificity: LR 0.679, XGBoost 0.672, KNN 0.603, RF 0.565. Healthy patients correctly spared, of 131: 89 / 88 / 79 / 74. §9.2 |
| 23 | A probability is not a decision. Three tiers are. | Tier table + the `threshold_tradeoff` curve | Main model: Low below **26.3%**, High from **66.2%**. Separate-fold check (cut-offs re-chosen inside each fold): recall **0.901**, specificity 0.542, High-tier PPV **0.857**. Low tier is never a discharge — it requires clinician review, a follow-up date and a named contact. §10, §13.2 |

### M6 — Impact, validation, live demo (8:50–11:30)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 24 | Simulated: risk order puts 76.6% of sick patients in the first 40% of slots. | `queue_gain` curve, benchmark line dashed | vs 40% for proportional arrival. Per 1,000 referred (simulated, retrospective model): 360 sick in High, 368 healthy kept out of the priority queue, **45.5 sick left in Low**. The word **SIMULATED** sits top-right: the data has no arrival order, no waiting time, no withheld test. §10.3, §15.2 |
| 25 | The locked test set, opened once — and a hospital the model never saw. | Confusion matrix left, Hungary metrics right | Retrospective 13-feature test: **28 of 28** sick patients flagged, exact 95% CI 0.877–1.000; ROC-AUC 0.961; specificity 0.667. Reduced-input secondary check: 28/28, ROC-AUC 0.957. Hungary, 294 patients, no refitting: ROC-AUC **0.877**, recall 0.887 — **below our 0.90 target**. §11, §13 |
| — | **LIVE DEMO** | Screen capture only, no slide | 6 steps, [`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md) |

### M3 (second slot) — Improvement (11:30–12:20)

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 26 | Is the tuned score flattering itself? No. | 3-row table | Nested CV (tuning repeated inside 5 outer folds): LR 0.909 vs tuned 0.900 → optimism **−0.008**; XGBoost 0.905 vs 0.897. A 100-setting search added 0.000 to LR and 0.003 to XGBoost. §9.3, §9.6 |
| 27 | More data would help. A different algorithm would not. | `learning_curve`, still rising at the right edge | Validation ROC-AUC 0.861 → 0.900 from 38 to 193 patients, still climbing. Drop-column test: only `ca` matters (−0.030, p=0.016); a 7-feature variant was +0.007, p=0.47 → keep all. **6 sick patients are missed by all four tuned models** — they need new inputs (troponin, symptom duration, calcium score), not a new algorithm. §9.4, §9.5, §9.2 |
| 28 | What we would not yet trust. | 4 lines, the sex gap in `accent` | Recall for women **0.833** (15/18, CI 0.586–0.964) vs men 0.914 — the cut-off was chosen on these patients. One hospital, 1980s, patients already referred, 303 in total. Hungary recall 0.887, below target. Input timing unverified. §15.1, §13.3 |

### M1 (close) — 12:20–13:00

| # | Title | Visual | Content / numbers |
|---|---|---|---|
| 29 | Three numbers. | Three hero numbers in a row | **9 in 10** patients with disease flagged (separate-fold recall 0.901) · **368 per 1,000** healthy patients kept out of the priority queue (simulated) · **0.877** ROC-AUC at a hospital it had never seen |
| 30 | Decision support, not diagnosis. | The sentence, 60 pt, centred, nothing else | Then cut to credits |
| 31 | Credits | The contribution matrix from [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md), 6 rows | Name · owned · presented. Hold 6 s. |

> Footer numbering runs `n / 31`. Slides 1 and 31 are title and credits; 2–30 are the 29 content slides.

---

## Part C — production checklist

1. Export every figure from a **freshly re-run notebook** (`jupyter nbconvert --execute --inplace`), then
   `python -m tests.check_notebook` must print `clean`.
2. Re-verify the six demo numbers against the rebuilt artifact before recording. They move when the model retrains.
3. Record the demo **twice**; keep the second take as the fallback cut.
4. Record at 1x with a visible timer for the presenter. Do not speed up in the edit — the rubric says normal speed.
5. Put each speaker's lower-third on their first slide, and cut on the `accent` speaker bar.
6. Watch the whole thing once on a phone before uploading. If a number is unreadable, it is too small.
