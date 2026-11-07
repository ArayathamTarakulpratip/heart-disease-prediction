# Historical evaluation audit (superseded)

This 2026-11-06 audit predates the pre-angiography demo, separate-fold decision validation, and simulation labels. Use `README.md` and the executed notebook for current results. The user deferred the video deliverable.

Audit of the repo against the 24-mark rubric (Problem Statement 12 + Implementation 12). Notebook cell numbers are **0-based** (cell 0 is the title cell of `heart_disease_pitch_gold_standard.ipynb`).

Verified when written: `python -m tests.check_notebook` prints "clean"; `python -m tests.smoke_test` passes 70/70. The notebook was **not** re-run end to end, and the app was not opened in a browser.

## Score: 23.5 / 24

| Criterion | Max | Score | Gap |
|---|---|---|---|
| Objective | 3.0 | 3.0 | none |
| Problem Importance | 4.5 | **4.25** | −0.25, see Gap 1 |
| Clarity of Problem Statement | 4.5 | **4.25** | −0.25, see Gap 2 |
| Model Identification | 3.0 | 3.0 | none |
| Solution | 3.0 | 3.0 | none |
| Model Comparison | 3.0 | 3.0 | none |
| Tuning and Improvement | 1.5 | 1.5 | none |
| Demonstration vs Scope | 1.5 | 1.5 | none (but see Gap 4: the demo is not recorded) |

The criteria above add up to 24, not 30. In `SPEC.md` §2a the other 6 marks are **Charisma (10%)** and **Member contribution (10%)**. Nothing in the repo covers them yet (Gap 3), and that is the biggest risk.

---

## Gap 1: Problem Importance (−0.25)

**Where:** cell 1 ("0. Why this matters") and the "Why it matters" paragraph in `README.md`.

**Why a strict judge docks it:**
1. The statistics that carry the argument are old: Pope 2000, Natarajan 2002, Patel 2010. The recent ones (Tam 2021, BHF Sept 2025, Thai 2012–2021 mortality) are there but come after them.
2. Two different "37.6%" figures appear in the same cell. Patel's is the share of angiograms that found obstructive disease. Khananuraksa's is the share of Thai angiograms with no stent or bypass afterwards. A listener will mix them up.
3. All money figures are US Medicare 2026 ($3,312 per angiogram). No Thai or UK price was found (stated in `SPEC.md` §6 and the cell 1 text).

**Fix:**
- Relabel the two figures wherever they appear (cell 1, `README.md`, `SPEC.md` §6): "obstructive yield 37.6% (US, Patel 2010)" and "angiogram-only 37.6% (Thailand, Khananuraksa 2024)". Say in the video that the match is a coincidence.
- Lead with the recent statistics: BHF Sept 2025 (148,882 waiting 18 weeks or more), Tam 2021 (HR 1.83), Thai mortality 23.4 to 26.9 per 100,000. Present Natarajan and Patel as the mechanism behind them.
- Make "angiograms avoided per 1,000 referred patients" (cell 125) the headline impact number, because it transfers between countries. Treat the dollar figure as an illustration.
- Optional: a Thai or UK angiogram price would close the money gap. Only add one you can open and cite; don't guess.

## Gap 2: Clarity of Problem Statement (−0.25)

**Where:** cells 50 ("What we carry into modelling") and 51 ("Is the model reading the answer?"). They are unnumbered headings between §5 and §6. The notebook has 133 cells in total.

**Why a strict judge docks it:**
1. The section outline breaks: §5.7 is followed by two unnumbered sections and then §6.
2. A viewer cannot follow 133 cells in a 10–15 minute video, and nothing in the repo is a short version of the story.

**Fix:**
- Number cell 50 as §5.8 and cell 51 as §5.9. Update every cross-reference (grep for "What we carry" and "circularity" in the notebook, `README.md`, `DEMO.md`). Re-run the notebook, then `python -m tests.check_notebook`.
- Build the video around three numbers: **0.900 cross-validated ROC-AUC, 28 of 28 sick test patients flagged, 76.6% of sick patients served in the first 40% of angiography slots** (cells 70, 101, 98).

## Gap 3: no presentation material (the 6 marks outside the 24)

**Where:** nowhere. `PROJECT_LOG.md:274` says "Not built, by decision: the pitch deck and script, team roles". `tasks/plan.md:4` marks them out of scope.

**Needed:**
- A time-boxed script for 10–15 minutes (13 minutes is a safe target), with the six questions in the order below.
- A slide outline that maps to Q1–Q6.
- A speaker split, with everyone on camera, and a contribution table.

**Suggested 13-minute structure** (4 speakers A–D; reassign for your team size):

| Time | Segment | Visual (notebook cell) | Speaker |
|---|---|---|---|
| 0:00–0:45 | Hook: 1.4% of patients waiting for angiography had a major event; most angiograms find no blockage | cell 1 | A |
| 0:45–2:15 | Q3 Problem importance (global, queue, Thailand) and the cost of each error | cells 1–2 | A |
| 2:15–3:30 | Q1 Dataset and target; the Kaggle labels are inverted | cells 8, 12–13 | B |
| 3:30–5:30 | Q2 EDA: sex gap, asymptomatic paradox, `ca`/`thal` gradient, weak BP and cholesterol; "what we carry" table | cells 30, 33, 42, 48, 50 | B |
| 5:30–7:30 | Q4 Five models and their maths, CV box plot, before and after tuning, confusion grid | cells 62–63, 66, 72 | C |
| 7:30–8:30 | Q5 Triage tiers, queue-gain curve, cost per 1,000 patients | cells 88, 97, 125 | C |
| 8:30–11:00 | Live demo, `DEMO.md` steps 1–7 | app | D |
| 11:00–12:30 | Q6 Nested CV, feature selection, learning curve, roadmap, limitations (sex gap, circularity) | cells 76, 79, 82, 110, 127 | D |
| 12:30–13:00 | Close: the three numbers, "decision support, not diagnosis" | title slide | A |

## Gap 4: demo risk (inside the 1.5 marks, so not yet docked)

- **The demo has not been recorded.** `DEMO.md` is a run-sheet whose figures were produced with Streamlit's `AppTest`, not on camera. Record a backup take.
- `app.py:123` reads `artifact["no_imaging"]`. If the saved model file predates the circularity feature, switching the toggle on raises a `KeyError` on screen. Re-run the notebook before recording (its last section rewrites `artifacts/heart_disease_pipeline.joblib`), or add a guard.
- Re-check every figure in `DEMO.md` after any notebook re-run, because a re-trained model can move them.

## Code review notes (uncommitted diff)

Verdict: approve. Nothing blocks.

- **Optional:** `evaluation.queue_gain` returns NaN when `y` has no sick patients, and with a fraction of 0 it reports the first slot instead of 0. The notebook cannot trigger either case. Add a test or an assert if the function is reused.
- **Nit:** `tests/test_app.py:16` uses a list comprehension for its side effects.
- **Nit:** `artifacts/metrics.json` has no trailing newline.
- **Required before submission:** 14 modified files plus untracked `README.md`, `DEMO.md` and `.gitignore` are not committed. A fresh clone has no README.

## Six-question coverage

| Question | Notebook | Video material |
|---|---|---|
| Q1 Dataset and target | full (cells 2, 6–14) | missing |
| Q2 EDA patterns | full (cells 25–50) | missing |
| Q3 Real-world problem | full (cells 1–2, 124–126) | missing |
| Q4 Models and how they work | full (cells 62–73) | missing |
| Q5 How the model solves the problem | full (cells 87–98, 121–126, app) | demo run-sheet only |
| Q6 Improvement | full (cells 74–85, 127) | missing |

## Checklist for whoever picks this up

1. Relabel the two 37.6% figures (Gap 1).
2. Reorder the Thai and recent statistics ahead of the old ones; make "angiograms avoided per 1,000" the headline (Gap 1).
3. Number cells 50 and 51 as §5.8 and §5.9, fix cross-references, re-run the notebook and `check_notebook` (Gap 2).
4. Write the script, slide outline, speaker split and contribution table (Gap 3).
5. Re-run the notebook, re-check `DEMO.md` figures, record the demo plus a backup take (Gap 4).
6. Commit everything.
