"""Every chart in the project. Each function takes a DataFrame and returns a matplotlib Figure,
so notebook cells stay one line long. Imports only `config` from the package (no ML libraries)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cardiotriage import config as cfg

DISEASE_NAMES = {0: "No disease", 1: "Disease"}
PALETTE = {"No disease": "#4C78A8", "Disease": "#E45756"}   # blue = healthy, red = disease, used everywhere


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100


def _status(df: pd.DataFrame) -> pd.Series:
    return df[cfg.TARGET].map(DISEASE_NAMES)


def class_balance(df: pd.DataFrame):
    """Bar chart of patients with and without disease, with counts and shares on the bars."""
    counts = _status(df).value_counts().reindex(list(PALETTE))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.to_numpy(), color=[PALETTE[n] for n in counts.index])
    for x, n in enumerate(counts.to_numpy()):
        ax.text(x, n, f"{n} ({n / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title="Class balance (angiographic disease = num > 0)", ylabel="Patients")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def rate_by(df: pd.DataFrame, cols: list[str]):
    """One panel per categorical column: disease rate per category, labelled with clinical names and n."""
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.8), squeeze=False)
    overall = df[cfg.TARGET].mean()
    for ax, col in zip(axes[0], cols):
        g = df.groupby(col)[cfg.TARGET].agg(["mean", "size"])
        names = [cfg.LABELS.get(col, {}).get(int(c), str(c)) for c in g.index]
        ax.bar(range(len(g)), g["mean"].to_numpy(), color=PALETTE["Disease"], alpha=0.85)
        for x, (rate, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(x, rate, f"{rate:.0%}\nn={n}", ha="center", va="bottom", fontsize=9)
        ax.axhline(overall, ls="--", color="grey", lw=1, label=f"overall {overall:.0%}")
        ax.set_xticks(range(len(g)), names, rotation=20, ha="right")
        ax.set(title=f"Disease rate: {cfg.DESCRIPTIONS.get(col, col)}", ylim=(0, 1.15),
               ylabel="Share with disease")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def age_vs_max_hr(df: pd.DataFrame):
    """Age vs maximum heart rate on the stress test, coloured by disease, with the 220 − age reference line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(x=df["age"], y=df["thalach"], hue=_status(df), palette=PALETTE, alpha=0.75, ax=ax)
    ages = pd.Series(range(int(df["age"].min()), int(df["age"].max()) + 1))
    ax.plot(ages, 220 - ages, ls="--", color="grey", lw=1, label="age-predicted max (220 − age)")
    ax.set(title="Age vs maximum heart rate achieved", xlabel="Age (years)", ylabel="Max heart rate (bpm)")
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def by_disease(df: pd.DataFrame, cols: list[str]):
    """One box plot per numeric column, split by disease status."""
    # matplotlib directly: seaborn 0.13.2 boxplot passes the `vert` argument matplotlib 3.11 deprecated
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8), squeeze=False)
    status = _status(df)
    for ax, col in zip(axes[0], cols):
        groups = [df.loc[status == name, col].dropna() for name in PALETTE]
        box = ax.boxplot(groups, tick_labels=list(PALETTE), patch_artist=True, widths=0.6)
        for patch, color in zip(box["boxes"], PALETTE.values()):
            patch.set(facecolor=color, alpha=0.75)
        ax.set(title=cfg.DESCRIPTIONS.get(col, col))
    fig.tight_layout()
    return fig


def cv_boxplot(scores: dict, metric: str = "roc_auc"):
    """Distribution of one metric over the CV folds, one box per model."""
    names = list(scores)
    label = {"roc_auc": "ROC-AUC", "brier": "Brier score"}.get(metric, metric.capitalize())
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2, 4))
    ax.boxplot([scores[n][metric] for n in names], tick_labels=names, widths=0.6)
    ax.set(title=f"{label} across the 50 cross-validation folds", ylabel=label)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def learning_curve(table: pd.DataFrame):
    """Training vs validation ROC-AUC as the training set grows (output of evaluation.learning_curve_table)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for part, color in (("train", PALETTE["No disease"]), ("valid", PALETTE["Disease"])):
        m, s = table[f"{part}_mean"], table[f"{part}_sd"]
        ax.plot(table["train_size"], m, marker="o", color=color, label="training folds" if part == "train" else "validation folds")
        ax.fill_between(table["train_size"], m - s, m + s, color=color, alpha=0.15)
    ax.set(title="Learning curve", xlabel="Training patients", ylabel="ROC-AUC")
    ax.legend()
    fig.tight_layout()
    return fig


def threshold_tradeoff(sweep: pd.DataFrame, thresholds: dict):
    """Recall and specificity as the cut-off moves, with the chosen tier cut-offs marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["recall"], color=PALETTE["Disease"], label="recall (sick patients flagged)")
    ax.plot(sweep["threshold"], sweep["specificity"], color=PALETTE["No disease"], label="specificity (healthy not flagged)")
    for key, style in (("low", "--"), ("high", ":")):
        ax.axvline(thresholds[key], color="black", ls=style, lw=1, label=f"{key} cut-off {thresholds[key]:.2f}")
    ax.set(title="Recall vs specificity across cut-offs (out-of-fold)", xlabel="Cut-off on P(disease)",
           ylabel="Rate", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def decision_curve(nb: pd.DataFrame):
    """Decision-curve analysis: net benefit of the model vs treating everyone or no one."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nb["threshold"], nb["model"], color=PALETTE["Disease"], lw=2, label="model")
    ax.plot(nb["threshold"], nb["treat_all"], color="grey", ls="--", label="angiography for everyone")
    ax.plot(nb["threshold"], nb["treat_none"], color="black", lw=1, label="angiography for no one")
    ax.set(title="Decision curve", xlabel="Risk at which angiography is worth it", ylabel="Net benefit",
           ylim=(min(-0.05, nb["model"].min()), max(nb["model"].max(), nb["treat_all"].max()) * 1.1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def queue_gain(gain: pd.DataFrame):
    """Cumulative share of sick patients served as the angiography queue fills: risk order vs arrival order."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(100 * gain["slots"], 100 * gain["model"], color=PALETTE["Disease"], lw=2, label="queue ordered by model risk")
    ax.plot(100 * gain["slots"], 100 * gain["first_come_first_served"], color="grey", ls="--",
            label="first come, first served")
    ax.set(title="Who gets the first angiography slots? (out-of-fold)", xlabel="Share of queue served (%)",
           ylabel="Share of all sick patients served (%)", xlim=(0, 100), ylim=(0, 102))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def roc_pr_calibration(c: dict):
    """ROC, precision-recall and calibration curves from evaluation.curves()."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fpr, tpr, auc = c["roc"]
    axes[0].plot(fpr, tpr, color=PALETTE["Disease"], label=f"model (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", label="chance")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate (recall)")
    rec, prec, ap, prev = c["pr"]
    axes[1].plot(rec, prec, color=PALETTE["Disease"], label=f"model (AP {ap:.3f})")
    axes[1].axhline(prev, color="grey", ls="--", label=f"prevalence {prev:.2f}")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision (PPV)", ylim=(0, 1.02))
    pred, true = c["calibration"]
    axes[2].plot(pred, true, marker="o", color=PALETTE["Disease"], label="model")
    axes[2].plot([0, 1], [0, 1], color="grey", ls="--", label="perfect calibration")
    axes[2].set(title="Calibration", xlabel="Predicted risk", ylabel="Observed share with disease")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _draw_confusion(ax, cm: pd.DataFrame, title: str) -> None:
    values = cm.to_numpy()
    ax.imshow(values, cmap="Blues")
    for (i, j), v in pd.DataFrame(values).stack().items():
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                color="white" if v > values.max() / 2 else "black")      # readable on dark cells
    ax.set_xticks(range(cm.shape[1]), cm.columns)
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.set(title=title, xlabel="Model's call", ylabel="True status")
    ax.grid(False)


def confusion(cm: pd.DataFrame):
    """Confusion matrix with true status as rows and the model's call as columns (labels spelled out, T11)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    _draw_confusion(ax, cm, "Test set at the low cut-off")
    fig.tight_layout()
    return fig


def confusion_grid(cms: dict):
    """One confusion matrix per model, side by side ({model name: evaluation.confusion table})."""
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.6), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        _draw_confusion(ax, cm, name)
    for ax in axes[0][1:]:
        ax.set(ylabel=None, yticklabels=[])                               # row labels once, on the left
    fig.tight_layout()
    return fig


def importance(values: pd.Series, title: str):
    """Horizontal bar chart of per-feature importance or contribution (positive = red, negative = blue)."""
    v = values.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(v) + 1.2))
    ax.barh(v.index, v.to_numpy(), color=[PALETTE["Disease"] if x >= 0 else PALETTE["No disease"] for x in v])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=title)
    fig.tight_layout()
    return fig


def correlation(df: pd.DataFrame):
    """Pearson correlation heatmap of the numeric, ordinal and binary columns plus the target.

    Nominal codes (cp, restecg, slope, thal) are left out on purpose: their numbers are labels, not
    quantities, so a correlation on them would be meaningless (trap T9).
    """
    cols = cfg.NUMERIC + cfg.ORDINAL + cfg.BINARY + [cfg.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation (numeric, ordinal and binary columns only)")
    fig.tight_layout()
    return fig
