# Presentation pack

Everything needed to record the 10–15 minute pitch video with 6 presenters. Read in this order.

| File | What it is | Who needs it |
|---|---|---|
| [`PLAN.md`](PLAN.md) | The proposed approach, a comparison of the five drafted plans, the defects they share, and the finalised decisions with reasons | everyone, first |
| [`SLIDES.md`](SLIDES.md) | Deck spec: 17 slides in a standard rubric-order ML talk, plus layout, colour and chart rules | whoever builds the deck |
| [`SCRIPT.md`](SCRIPT.md) | The 13:00 script, segment by segment, with timings, hand-off lines and hold times | all 6 speakers |
| [`DEMO_RUNSHEET.md`](DEMO_RUNSHEET.md) | The 1:35 live demo, step by step, with the live figures and a failure playbook | the demo presenter |
| [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) | End-to-end account of the project: data, QA, EDA, split design, models, tuning, evaluation, impact, limitations | everyone, for individual questioning |
| [`ML_CONCEPTS.md`](ML_CONCEPTS.md) | The AI/ML theory behind every choice, each paired with the number it produced here | everyone |
| [`QA_BANK.md`](QA_BANK.md) | ~50 likely questions with short answers, including the ten that can actually hurt | everyone |
| [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) | Who owns what and who presents what — the final slide | everyone |

## Before you record

```bash
jupyter nbconvert --to notebook --execute --inplace heart_disease_pitch_gold_standard.ipynb
python -m tests.check_notebook    # must print "clean"
python -m tests.smoke_test        # all tests must pass
```

Then re-verify every figure in `DEMO_RUNSHEET.md` against the rebuilt artifact. **Retraining moves the demo numbers**,
and a number on screen that disagrees with the spoken script is the easiest mark to lose.

## The three facts that must never be contradicted on camera

1. The shipped model uses **10 features** and excludes `ca`, `thal` and `slope`. The 13-feature model is a
   retrospective comparison only.
2. The queue-gain and money figures are **simulated**.
3. **Decision support, not diagnosis.** A Low tier is not a discharge.
