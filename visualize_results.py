"""
Visualize and statistically analyze multi-run inference results.

python visualize_results.py --results_path "results/inference_summary.csv" \
                             --output_path  "results/"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats


METRICS = ["accuracy", "f1_macro", "f1_binary", "f1_class0", "f1_class1"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "f1_macro":  "F1 Macro",
    "f1_binary": "F1 Binary",
    "f1_class0": "F1 Class 0\n(non-stereo)",
    "f1_class1": "F1 Class 1\n(stereo)",
}

MODEL_ORDER  = ["baseline", "context", "cross_attention"]
MODEL_LABELS = {
    "baseline":        "Baseline",
    "context":         "Hierarchical",
    "cross_attention": "Cross-Attention",
}

PALETTE = {
    "baseline":        "#6C8EBF",
    "context":         "#D4763B",
    "cross_attention": "#55A868",
}

# All pairs for t-tests
MODEL_PAIRS = [
    ("baseline",  "context"),
    ("baseline",  "cross_attention"),
    ("context",   "cross_attention"),
]

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_LABEL  = {"fontsize": 10, "color": "#333333"}
FONT_TICK   = {"fontsize": 9,  "color": "#555555"}
FONT_ANNOT  = {"fontsize": 8.5,"color": "#333333"}

plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.8,
    "figure.facecolor":  "#fafafa",
    "axes.facecolor":    "#fafafa",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True, help="Path to inference_summary.csv")
    p.add_argument("--output_path",  required=True, help="Directory to save figures")
    return p.parse_args()


def compute_summary(df):
    """Mean and std per model per metric."""
    available = [m for m in MODEL_ORDER if m in df["model"].unique()]
    return df[df["model"].isin(available)].groupby("model")[METRICS].agg(["mean", "std"]).round(4)


def paired_ttests(df):
    """
    Paired t-test for each metric across all model pairs.
    Returns a DataFrame with t-statistic, p-value, and significance flag.
    """
    rows = []
    for (m1, m2) in MODEL_PAIRS:
        if m1 not in df["model"].unique() or m2 not in df["model"].unique():
            continue
        for metric in METRICS:
            vals1 = df[df["model"] == m1].sort_values("run")[metric].values
            vals2 = df[df["model"] == m2].sort_values("run")[metric].values
            t, p  = stats.ttest_rel(vals2, vals1)
            rows.append({
                "pair":       f"{MODEL_LABELS[m2]} vs {MODEL_LABELS[m1]}",
                "metric":     metric,
                "delta_mean": (vals2 - vals1).mean(),
                "t_stat":     t,
                "p_value":    p,
                "significant (p<0.05)": p < 0.05,
                "significant (p<0.10)": p < 0.10,
            })
    return pd.DataFrame(rows)


def plot_bar_with_ci(df, ax, metric):
    """Grouped bar chart with ± 1 std error bars for one metric."""
    available = [m for m in MODEL_ORDER if m in df["model"].unique()]
    summary   = df.groupby("model")[metric].agg(["mean", "std"])
    x         = np.arange(len(available))
    width     = 0.6 / len(available)

    for i, model in enumerate(available):
        mean = summary.loc[model, "mean"]
        std  = summary.loc[model, "std"]
        ax.bar(
            i, mean, width,
            color=PALETTE[model], alpha=0.85,
            yerr=std, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#444"},
            zorder=3,
        )
        ax.text(i, mean + std + 0.005, f"{mean:.3f}", ha="center", **FONT_ANNOT)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in available], **FONT_TICK)
    ax.set_title(METRIC_LABELS[metric], **FONT_TITLE)
    ax.set_ylim(
        max(0, summary["mean"].min() - summary["std"].max() - 0.05),
        min(1, summary["mean"].max() + summary["std"].max() + 0.04),
    )


def plot_run_lines(df, ax, metric):
    """Connected scatter across all 3 models per run."""
    available = [m for m in MODEL_ORDER if m in df["model"].unique()]
    runs      = sorted(df["run"].unique())
    x_pos     = {m: i for i, m in enumerate(available)}

    for run in runs:
        vals = {}
        for model in available:
            row = df[(df["model"] == model) & (df["run"] == run)]
            if not row.empty:
                vals[model] = row[metric].values[0]

        if len(vals) < 2:
            continue

        models_in_run = [m for m in available if m in vals]
        ys     = [vals[m] for m in models_in_run]
        xs     = [x_pos[m] for m in models_in_run]
        color  = "#2ecc71" if ys[-1] >= ys[0] else "#e74c3c"

        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.4, zorder=2)
        ax.scatter(xs, ys,
                   color=[PALETTE[m] for m in models_in_run],
                   s=40, zorder=3)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([MODEL_LABELS[m] for m in available], **FONT_TICK)
    ax.set_title(f"{METRIC_LABELS[metric]} — per run", **FONT_TITLE)

    up   = mpatches.Patch(color="#2ecc71", alpha=0.7, label="Improves left → right")
    down = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Degrades left → right")
    ax.legend(handles=[up, down], fontsize=8, loc="lower right")


def plot_delta_distribution(df, ax, metric):
    """
    Per-run delta vs baseline for each non-baseline model,
    shown as side-by-side strips.
    """
    available    = [m for m in MODEL_ORDER if m in df["model"].unique() and m != "baseline"]
    runs         = sorted(df["run"].unique())
    n_models     = len(available)
    width        = 0.3
    x_positions  = np.arange(n_models)

    ax.axhline(0, color="#aaa", linewidth=1, linestyle="--", zorder=1)

    for i, model in enumerate(available):
        deltas = []
        for run in runs:
            base_val  = df[(df["model"] == "baseline") & (df["run"] == run)][metric].values
            model_val = df[(df["model"] == model)      & (df["run"] == run)][metric].values
            if len(base_val) and len(model_val):
                deltas.append(model_val[0] - base_val[0])

        deltas = np.array(deltas)
        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

        jitter = np.linspace(-width / 2, width / 2, len(deltas))
        ax.scatter(i + jitter, deltas, color=colors, s=60, zorder=3, alpha=0.8)
        ax.hlines(deltas.mean(), i - width / 2, i + width / 2,
                  color=PALETTE[model], linewidth=2.5, zorder=2,
                  label=f"{MODEL_LABELS[model]} mean Δ = {deltas.mean():+.4f}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in available], **FONT_TICK)
    ax.set_title(f"Δ {METRIC_LABELS[metric]} vs Baseline", **FONT_TITLE)
    ax.legend(fontsize=8)


def plot_stat_table(ttest_df, ax):
    """Render all paired t-test results as a clean table."""
    ax.axis("off")
    col_labels = ["Comparison", "Metric", "Mean Δ", "t-stat", "p-value", "p < 0.05", "p < 0.10"]
    table_data = []
    for _, row in ttest_df.iterrows():
        table_data.append([
            row["pair"],
            METRIC_LABELS[row["metric"]].replace("\n", " "),
            f"{row['delta_mean']:+.4f}",
            f"{row['t_stat']:.3f}",
            f"{row['p_value']:.4f}",
            "✓" if row["significant (p<0.05)"] else "✗",
            "✓" if row["significant (p<0.10)"] else "✗",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1a1a2e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "#fafafa")
            if j == 5:
                val = table_data[i - 1][j]
                table[i, j].set_text_props(
                    color="#2ecc71" if val == "✓" else "#e74c3c",
                    fontweight="bold",
                )

    ax.set_title("Paired t-tests — All Model Pairs", **FONT_TITLE, pad=20)


def build_figure(df, ttest_df, output_path):
    focus_metric = "f1_macro"

    fig = plt.figure(figsize=(20, 16), facecolor="#fafafa")
    fig.suptitle(
        "Model Comparison — Multi-Run Evaluation",
        fontsize=16, fontweight="bold", color="#1a1a2e", y=0.98,
    )

    gs = GridSpec(3, 5, figure=fig, hspace=0.6, wspace=0.45)

    # Row 0: bar chart per metric
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        plot_bar_with_ci(df, ax, metric)

    # Row 1: per-run lines + delta strips
    ax_lines = fig.add_subplot(gs[1, 0:2])
    plot_run_lines(df, ax_lines, focus_metric)

    ax_delta = fig.add_subplot(gs[1, 2:5])
    plot_delta_distribution(df, ax_delta, focus_metric)

    # Row 2: stat table
    ax_table = fig.add_subplot(gs[2, :])
    plot_stat_table(ttest_df, ax_table)

    handles = [
        mpatches.Patch(color=PALETTE[m], alpha=0.85, label=MODEL_LABELS[m])
        for m in MODEL_ORDER if m in df["model"].unique()
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.98, 0.96))

    out = Path(output_path) / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"[saved] Figure → {out}")
    plt.close()


def main():
    args = parse_args()
    df   = pd.read_csv(args.results_path)

    df.columns = df.columns.str.strip()
    df["run"]  = df["run"].astype(int)

    available = [m for m in MODEL_ORDER if m in df["model"].unique()]
    print(f"Models found: {available}")

    print("\n===== Summary statistics =====")
    print(compute_summary(df).to_string())

    print("\n===== Paired t-tests =====")
    ttest_df = paired_ttests(df)
    print(ttest_df.to_string(index=False))

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ttest_df.to_csv(out_dir / "statistical_tests.csv", index=False)
    print(f"\n[saved] t-test results → {out_dir / 'statistical_tests.csv'}")

    build_figure(df, ttest_df, args.output_path)


if __name__ == "__main__":
    main()