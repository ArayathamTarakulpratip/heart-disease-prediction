# Contributions

Team of **6**. This table is shown as the final slide and is the basis for the 10% member-contribution criterion.

**How to read it.** The *Owns* column records who is responsible for an area of the project going forward: knowing it
in full, re-running and verifying its numbers before recording, and fielding its questions. The *Presents* column is
what that member says on camera. Code-authorship history is in `git log` and is not what this table claims.

Replace `M1`–`M6` with real names before recording, and keep the row order the same as the running order in
[`presentation/SCRIPT.md`](presentation/SCRIPT.md).

| # | Member | Owns (area + files / notebook sections) | Presents | Slides | On camera |
|---|---|---|---|---|---|
| M1 | *name* | **Data provenance and the audit** — `cardiotriage/data.py`, §3, §3.2, §4.1–4.3; the Kaggle label-inversion proof and patient 184 | Hook + Close | 1–3, 29–31 | 0:00–1:10, 12:20–13:00 |
| M2 | *name* | **Problem and objective** — §0 (all three evidence layers and their sources), §1; every statistic, source and year | Why it matters, objective, error costs | 4–9 | 1:10–3:00 |
| M3 | *name* | **Data contract, QA, metrics and improvement** — `config.py`, `evaluation.py` (metrics, CIs, nested CV, learning curve), §4, §6.1, §9.3–9.6, §12.2 | Data + QA + metric, then improvement and limitations | 10–12, 26–28 | 3:00–4:40, 11:30–12:20 |
| M4 | *name* | **EDA and the input-timing pivot** — `plots.py`, `features.py`, §5.1–5.9, §7 | EDA (4 findings) and the circularity decision | 13–17 | 4:40–6:40 |
| M5 | *name* | **Modelling, tuning and the decision layer** — `models.py`, `triage.py`, §6, §8, §9, §10 | Five families, comparison, tuning, tiers | 18–23 | 6:40–8:50 |
| M6 | *name* | **Validation, impact and the demo** — `app.py`, `artifacts/`, `DEMO.md`, §11, §13, §14, §15.2 | Simulated impact, test + external validation, **live demo** | 24–25 + app | 8:50–11:30 |

## Shared responsibilities

| Task | Owner | Backup |
|---|---|---|
| Re-run the notebook end to end and confirm `tests.check_notebook` prints `clean` | M3 | M5 |
| Re-verify the six demo figures against the rebuilt artifact | M6 | M4 |
| Export all slide figures from the rebuilt notebook | M4 | M6 |
| Build the deck to [`presentation/SLIDES.md`](presentation/SLIDES.md) | M2 | M1 |
| Record, edit, upload | M6 | M1 |
| Timekeeping in rehearsal (13:00 target, 13:30 hard stop) | M1 | M3 |
| Record the backup demo take | M6 | M5 |

## Question routing

Each member fields the questions tagged with their number in [`presentation/QA_BANK.md`](presentation/QA_BANK.md).
The ten hostile questions in section A are rehearsed by **everyone**, because any of them can arrive mid-answer.

## Note for the lecturer

Every member is responsible for understanding the whole project, not only their own segment;
[`presentation/PROJECT_BRIEF.md`](presentation/PROJECT_BRIEF.md) is the end-to-end reference each member has worked
through, and [`presentation/QA_BANK.md`](presentation/QA_BANK.md) is the individual-questioning preparation.
