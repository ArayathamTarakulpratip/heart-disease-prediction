"""Every chart returns a Figure without warnings (plots.py)."""
import numpy as np
import pandas as pd


def _no_warning_figure(fn, *args):
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # any warning fails the test
        fig = fn(*args)
    assert isinstance(fig, Figure), type(fig)
    assert all(ax.has_data() for ax in fig.axes if ax.get_label() != "<colorbar>"), "empty axes"
    plt.close(fig)
    return fig


def test_plots_return_figures_without_warnings():
    from cardiotriage import data, plots
    df = data.load_cleveland()
    _no_warning_figure(plots.class_balance, df)
    _no_warning_figure(plots.rate_by, df, ["sex", "cp"])
    _no_warning_figure(plots.age_vs_max_hr, df)
    _no_warning_figure(plots.by_disease, df, ["thalach", "oldpeak"])
    _no_warning_figure(plots.correlation, df)


def test_rate_by_uses_clinical_labels_not_codes():
    from cardiotriage import data, plots
    import matplotlib.pyplot as plt
    fig = _no_warning_figure(plots.rate_by, data.load_cleveland(), ["cp"])
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "asymptomatic" in ticks, ticks
    plt.close("all")


def test_class_balance_legend_is_not_the_broken_string_legend():
    # T12: the old notebook passed plt.legend("Female", "Male"), which splits a string into letters
    from cardiotriage import data, plots
    fig = _no_warning_figure(plots.class_balance, data.load_cleveland())
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["No disease", "Disease"], labels


def test_model_plots_return_figures_without_warnings():
    from cardiotriage import plots
    rng = np.random.default_rng(0)
    scores = {m: pd.DataFrame({"roc_auc": rng.uniform(0.8, 0.95, 50)}) for m in ["A", "B"]}
    _no_warning_figure(plots.cv_boxplot, scores, "roc_auc")
    lc = pd.DataFrame({"train_size": [50, 100], "train_mean": [0.95, 0.92], "train_sd": [0.01, 0.01],
                       "valid_mean": [0.85, 0.88], "valid_sd": [0.03, 0.02]})
    _no_warning_figure(plots.learning_curve, lc)


def test_evaluation_plots_return_figures_without_warnings():
    from cardiotriage import evaluation, plots
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 200), 0, 1)
    sweep = evaluation.threshold_sweep(y, p, np.linspace(0.05, 0.95, 19))
    th = {"low": 0.3, "high": 0.7}
    _no_warning_figure(plots.threshold_tradeoff, sweep, th)
    _no_warning_figure(plots.decision_curve, evaluation.net_benefit(y, p, np.linspace(0.05, 0.6, 12)))
    _no_warning_figure(plots.roc_pr_calibration, evaluation.curves(y, p))
    _no_warning_figure(plots.confusion, evaluation.confusion(y, p >= 0.5))
    _no_warning_figure(plots.importance, pd.Series({"ca": 0.1, "cp": 0.05, "age": 0.01}), "Permutation importance")


def test_confusion_grid_draws_one_panel_per_model_without_warnings():
    from cardiotriage import evaluation, plots
    y = [1, 1, 0, 0, 1, 0]
    cms = {m: evaluation.confusion(y, flag) for m, flag in
           {"A": [1, 1, 0, 1, 0, 0], "B": [1, 1, 1, 1, 1, 0], "C": [0, 1, 0, 0, 1, 0]}.items()}
    fig = _no_warning_figure(plots.confusion_grid, cms)
    assert [ax.get_title() for ax in fig.axes] == ["A", "B", "C"]


def test_queue_gain_plot_draws_without_warnings():
    from cardiotriage import evaluation, plots
    g = evaluation.queue_gain([1, 0, 1, 0, 0, 1], [0.9, 0.8, 0.7, 0.3, 0.2, 0.1], [0.25, 0.5, 0.75, 1.0])
    _no_warning_figure(plots.queue_gain, g)
