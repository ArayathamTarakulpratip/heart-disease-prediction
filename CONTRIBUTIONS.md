# Contributions

Team of **6**. This table is shown as the final slide and is the basis for the 10% member-contribution criterion.

**How to read it.** The *Owns* column records who is responsible for an area of the project going forward: knowing it
in full, re-running and verifying its numbers before recording, and fielding its questions. The *Presents* column is
what that member says on camera. Code-authorship history is in `git log` and is not what this table claims.

Replace `M1`–`M6` with real names before recording, and keep the row order the same as the running order in
[`presentation/SCRIPT.md`](presentation/SCRIPT.md).

| # | Member | Owns (area + files / notebook sections) | Presents | Slides | On camera (2:10 each) |
|---|---|---|---|---|---|
| M1 | *name* | **Data provenance and problem evidence** — `cardiotriage/data.py`, §0 (all statistics, sources, years), §3.2; the Kaggle label-inversion proof | Hook + why it matters | 1–8 | 0:00–2:10 |
| M2 | *name* | **Objective, data contract, QA** — `config.py`, §1, §3, §4 | Objective, data, contract, balance | 9–12 | 2:10–4:20 |
| M3 | *name* | **EDA and the input-timing pivot** — `plots.py`, `features.py`, §5.1–5.9, §7 | EDA and the `ca` decision | 13–17 | 4:20–6:30 |
| M4 | *name* | **Modelling and the decision layer** — `models.py`, `triage.py`, §6, §8, §9.1–9.2, §10 | Five families, comparison, tuning, tiers | 18–23 | 6:30–8:40 |
| M5 | *name* | **Evaluation, validation, improvement** — `evaluation.py`, §9.3–9.6, §11–§14, §15 | Impact, validation, improvement, limits | 24–28 | 8:40–10:50 |
| M6 | *name* | **App, demo and artifacts** — `app.py`, `artifacts/`, `DEMO.md` | Live demo (1:35) + close (0:35) | app, 29–31 | 10:50–13:00 |

## Shared responsibilities

| Task | Owner | Backup |
|---|---|---|
| Re-run the notebook end to end and confirm `tests.check_notebook` prints `clean` | M5 | M4 |
| Re-verify the six demo figures against the rebuilt artifact | M6 | M4 |
| Export all slide figures from the rebuilt notebook | M3 | M6 |
| Build the deck to [`presentation/SLIDES.md`](presentation/SLIDES.md) | M2 | M1 |
| Record, edit, upload | M6 | M1 |
| Timekeeping in rehearsal (13:00 target, 13:30 hard stop) | M1 | M2 |
| Record the backup demo take | M6 | M5 |

## Question routing

Each member fields the questions tagged with their number in [`presentation/QA_BANK.md`](presentation/QA_BANK.md).
The ten hostile questions in section A are rehearsed by **everyone**, because any of them can arrive mid-answer.

## Note for the lecturer

Every member is responsible for understanding the whole project, not only their own segment;
[`presentation/PROJECT_BRIEF.md`](presentation/PROJECT_BRIEF.md) is the end-to-end reference each member has worked
through, and [`presentation/QA_BANK.md`](presentation/QA_BANK.md) is the individual-questioning preparation.
