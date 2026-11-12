# Presentation plan: comparison of the drafts, and the finalised approach

Audience: pitch-competition judges. Deliverable: one YouTube video, 10–15 min at 1x, demo included, 6 presenters.
Everything here is keyed to numbers that exist in `heart_disease_pitch_gold_standard.ipynb`, `artifacts/metrics.json`,
`README.md` and `DEMO.md`. Nothing is invented.

---

## 1. What the five pasted drafts have in common

They converge more than they differ. The shared skeleton is sound and is kept:

| Shared element | Detail |
|---|---|
| Runtime | ~12.5–13 min at 1x, leaving 2 min under the 15-min cap |
| Order | Hook → Q3 importance → Q1 data → Q2 EDA → Q4 models → Q5 solution → demo → Q6 improvement → close |
| Hook style | One shocking number in the first 45 seconds |
| Demo placement | Late, 2–2.5 min, live screen capture, with a backup take |
| Slide discipline | One claim and one number per slide; no code on screen except in the demo |
| Close | "Decision support, not diagnosis" + a credits slide |
| Contribution marks | A `CONTRIBUTIONS.md` matrix shown as the final slide, each speaker on camera for their own segment |
| Speaker template | 4 people (A–D or M1–M4), with a note to resplit for other team sizes |

## 2. Where they disagree

| Axis | The competing choices |
|---|---|
| **Hook** | (a) "Patient 184 vanished and every label was flipped"; (b) "1.4% had a major event while waiting"; (c) "6 in 10 elective angiograms find no blocked artery" |
| **Q3 position** | Before Q1 (hook → importance → data) in most drafts; one draft puts Q1 second and importance split across two slots |
| **Which model is the star** | Every draft headlines the **13-feature retrospective** model (0.900 CV ROC-AUC, 28/28, 0.287/0.509 cut-offs) |
| **Money** | Two drafts put the $1.37M / $7,220→$4,703 figures on stage; two leave them out |
| **Demo figures** | 99.3% / 4.1% / 29.6% / 6.3%, change `ca` 0→2 — versus a draft that says "change max heart rate" |
| **Cell references** | Three different numbering schemes; several point at cells that have since moved |
| **Q6 content** | Nested CV + learning curve + roadmap in all; the "6 patients every model misses" only in some |

## 3. The defect all five share — and it is the one that loses marks

**Every draft pitches the retrospective 13-feature model as the product.** The repo deliberately does not.

* `ca` is "major vessels coloured by fluoroscopy". It may be read **during the very angiogram the model is prioritising**. UCI has no timestamps, so earlier availability is unproven (notebook §5.9, §1).
* The shipped demo is therefore the **10-feature model** that excludes `ca`, `thal` and `slope` (`artifacts/metrics.json` → `no_imaging_cv_roc_auc`, model card §15.5).
* Consequence: the drafts' headline numbers are the wrong model's. Stale values in the drafts vs the live artifact:

| Draft says | Live value | Which model |
|---|---|---|
| "Clear high risk 99.3%" | **97.5%** | 10-feature main (99.3% was patient 62 on the 13-feature model) |
| "Clear low risk 4.1%" | **2.1%** | 10-feature main |
| "Borderline 29.6%" | **30.3%** | 10-feature main (29.6% was the 13-feature score) |
| "Missed by the model 6.3%" | **5.5%** | 10-feature main |
| "Recall for women 0.778" | **0.833** (15/18) for the main model; 0.778 (14/18) is the retrospective one | both exist, label which |
| "Hungary ROC-AUC 0.880" | **0.877** | 10-feature main |
| Tiers "Low 0.287 / High 0.509" | **Low 0.263 / High 0.662** | main model cut-offs; 0.287/0.509 are retrospective |

If a judge asks "isn't `ca` the answer you're predicting?" and the team has headlined 0.900 and 28/28, the pitch collapses.
The repo already contains the correct answer; the script just has to use it.

Two smaller shared problems: all drafts are written for 4 speakers (we have **6**), and the queue-gain, missed-per-100
and dollar figures are **simulations on the retrospective model** — the dataset records no arrival order, no waiting
time and no decision to withhold a test. They must be said as simulations or they read as a fabricated claim.

## 4. My proposal: "the honest pivot"

Same skeleton, different spine. Instead of "we built an accurate classifier", the story is:

> Everyone's heart-disease demo is built on a Kaggle CSV whose labels are inverted. We went back to the original UCI
> file, proved the flip row by row, then found a second and worse problem in our own model: its strongest input may be
> measured during the procedure we are trying to schedule. So we **threw that input away**, gave up 0.056 of ROC-AUC,
> and shipped the model that a cardiologist can actually run **before** the angiogram — then proved it at a second
> hospital it had never seen.

Why this wins on the rubric rather than just sounding good:

* **Problem importance (15%)** — the problem is not "predict heart disease", it is "order a queue in which 1.4% of
  patients have a heart attack, heart failure or die before their test, while roughly 6 in 10 elective angiograms find
  no obstructive disease". That is a scheduling problem with a measured body count and a $3,312 unit cost.
* **Clarity (15%)** — one sentence: *given 10 results a clinician already has, estimate P(>50% narrowing) and sort the
  queue, flagging ≥90% of patients who truly have disease.* Inputs → contract → risk → tier → action, on one diagram.
* **Model identification + comparison (20%)** — five families on the same 50 folds, twice (13-feature and 10-feature),
  with a corrected t-test so "best" is a claim with a p-value, not a leaderboard position.
* **Solution (10%)** — the tier table, the separate-fold decision check, and a live app a judge can poke at.
* **Tuning (5%)** — randomised search inside CV, nested CV to prove the tuned score is not optimistic, a 100-setting
  budget check, and drop-column feature selection that we *rejected*.
* **Charisma (10%)** — the audit and the self-inflicted pivot are both stories with a villain and a decision. Judges
  remember "we deleted our best feature on purpose" far longer than "our AUC was 0.90".
* **Demonstration (5%)** — six steps already written in `DEMO.md`, including an input the data contract rejects.

The risk of my version: it spends ~45 s on an apparent negative (accuracy went down). Mitigation — frame the drop as
the price of validity and immediately pay it off with the external-hospital number, which the 13-feature model
**cannot produce at all** because Hungary has 99% of `ca` missing.

## 5. Finalised approach, and why

Take the drafts' **timing discipline and slide rules**, my **spine**, and fix the three defects.

| Decision | Choice | Why |
|---|---|---|
| Runtime | **13:00**, hard stop 13:30 | 2 min of safety under the cap; all drafts agree |
| Speakers | **6**, every one on camera, exactly 2:10 each | Matches the team; contribution is 10% of the mark |
| Hook | **The audit**: patient 184 + inverted labels, with the crosstab on screen | Unique to us; any team can quote WHO figures, no other team has an audit |
| Problem numbers | Ontario 1.4%, Patel 37.6% obstructive yield, England 148,882 waiting ≥18 weeks, Thailand 23.4→26.9/100,000 | Recent first, mechanism second (fixes `EVALUATION.md` Gap 1) |
| Headline model | **10-feature pre-angiography model** | The only one that is usable before the procedure; it is what the app ships |
| Headline numbers | recall **0.901** / specificity 0.542 / High-tier PPV 0.857 (separate-fold); Hungary ROC-AUC **0.877**, recall 0.887; Cleveland CV ROC-AUC 0.844 | All from `artifacts/metrics.json`; honest and defensible |
| 13-feature model | Kept, on **one** slide, explicitly labelled "retrospective comparison" | It is the strongest evidence that we understand leakage |
| Queue / money | **One** slide, every figure prefixed "simulated" | Impact without a false claim |
| Demo figures | 97.5 / 2.1 / 30.3 / 73.3 / 13.2→51.6 / 5.5 from `DEMO.md` | Re-verified against the saved artifact on 2026-11-12 |
| Demo what-if | **Change max heart rate to 100 bpm** on *Borderline* (30.3% → 73.3%) as the main what-if; the `ca` 0→2 step is the *second* what-if, used to show the main score ignoring it | A what-if must move the score the product actually uses |
| Two 37.6% figures | Say one only ("US obstructive yield 37.6%"); Thailand is quoted as "angiogram and nothing else" without the number | Fixes the confusion `EVALUATION.md` Gap 1 flags |
| Three closing numbers | 9 in 10 flagged · 368 healthy patients per 1,000 kept out of the priority queue (simulated) · 0.877 at a hospital it never saw | One per speaker-memorable beat |
| Slides | 24 (title + 22 content + credits); demo runs on screen capture, no slides | ~29 s per slide at one claim each |

**What we are deliberately not doing:** no notebook on screen, no code except in the app, no accuracy as a headline,
no claim of saved money or lives, no "AI diagnoses heart disease" framing.

---

Next: [`SLIDES.md`](SLIDES.md) (deck spec) · [`SCRIPT.md`](SCRIPT.md) (6-speaker script) ·
[`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md) (2.5-min demo) · [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) (end-to-end study
guide) · [`ML_CONCEPTS.md`](ML_CONCEPTS.md) · [`QA_BANK.md`](QA_BANK.md) · [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md)
