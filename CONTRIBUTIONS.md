# Contributions

Team of **6**. This table is shown as the final slide and is the basis for the 10% member-contribution criterion.

**How to read it.** The *Owns* column records who is responsible for an area of the project going forward: knowing it
in full, re-running and verifying its numbers before recording, and fielding its questions. The *Presents* column is
what that member says on camera. Code-authorship history is in `git log` and is not what this table claims.

Replace `M1`–`M6` with real names before recording, and keep the row order the same as the running order in
[`presentation/SCRIPT.md`](presentation/SCRIPT.md).

| # | Member | Owns (area + files / notebook sections) | Presents | Slides | On camera (2:10 each) |
|---|---|---|---|---|---|
| M1 | *name* | **Problem and dataset** — §0, §1, §3, `data.py` | Problem, objective, dataset | 1–3 | 0:00–2:10 |
| M2 | *name* | **Data quality and EDA** — §4, §5.1–5.7, `plots.py`, `config.py` | Data checks and 4 EDA findings | 4–6 | 2:10–4:20 |
| M3 | *name* | **Input timing, validation, models** — §5.8–5.9, §6–§8, `models.py`, `features.py` | The `ca` decision, split design, five models | 7–9 | 4:20–6:30 |
| M4 | *name* | **Comparison, tuning, results** — §9.1–9.2, §10, §11, §13, `triage.py` | Comparison, tuning, results | 10–12 | 6:30–8:40 |
| M5 | *name* | **Improvement and limitations** — §9.3–9.6, §12, §14, §15, `evaluation.py` | Improvement, limitations | 13–14 | 8:40–10:50 |
| M6 | *name* | **App and demo** — `app.py`, `artifacts/`, `DEMO.md` | Live demo (1:35) + conclusion | 15–17 | 10:50–13:00 |

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
