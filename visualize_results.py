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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METRICS = ["accuracy", "f1_macro", "f1_binary", "f1_class0", "f1_class1"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "f1_macro":  "F1 Macro",
    "f1_binary": "F1 Binary",
    "f1_class0": "F1 Class 0\n(non-stereo)",
    "f1_class1": "F1 Class 1\n(stereo)",
}

PALETTE = {
    "baseline": "#6C8EBF",
    "context":  "#D4763B",
}

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse path to results foler and path to output folder.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True, help="Path to inference_summary.csv")
    p.add_argument("--output_path",  required=True, help="Directory to save figures")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_summary(df):
    """
    Mean and std per model per metric.
    """
    return df.groupby("model")[METRICS].agg(["mean", "std"]).round(4)


def paired_ttests(df):
    """
    Paired t-test for each metric: baseline vs context across runs.
    Returns a DataFrame with t-statistic, p-value, and significance flag.
    """
    rows = []
    for metric in METRICS:
        base = df[df["model"] == "baseline"].sort_values("run")[metric].values
        ctx  = df[df["model"] == "context"].sort_values("run")[metric].values
        t, p = stats.ttest_rel(ctx, base)
        delta_mean = (ctx - base).mean()
        rows.append({
            "metric":     metric,
            "delta_mean": delta_mean,
            "t_stat":     t,
            "p_value":    p,
            "significant (p<0.05)": p < 0.05,
            "significant (p<0.10)": p < 0.10,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bar_with_ci(df, ax, metric):
    """
    Grouped bar chart with ± 1 std error bars for one metric.
    """
    summary = df.groupby("model")[metric].agg(["mean", "std"])
    models  = ["baseline", "context"]
    x       = np.arange(len(models))
    width   = 0.45

    for i, model in enumerate(models):
        mean = summary.loc[model, "mean"]
        std  = summary.loc[model, "std"]
        bar  = ax.bar(
            i, mean, width,
            color=PALETTE[model], alpha=0.85,
            yerr=std, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#444"},
            label=model.capitalize(), zorder=3,
        )
        ax.text(i, mean + std + 0.005, f"{mean:.3f}", ha="center", **FONT_ANNOT)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Context"], **FONT_TICK)
    ax.set_title(METRIC_LABELS[metric], **FONT_TITLE)
    ax.set_ylim(
        max(0, summary["mean"].min() - summary["std"].max() - 0.05),
        min(1, summary["mean"].max() + summary["std"].max() + 0.04),
    )


def plot_run_lines(df, ax, metric):
    """
    Connected scatter: one line per run showing baseline → context.
    """
    runs = sorted(df["run"].unique())
    for run in runs:
        base_val = df[(df["model"] == "baseline") & (df["run"] == run)][metric].values[0]
        ctx_val  = df[(df["model"] == "context")  & (df["run"] == run)][metric].values[0]
        color    = "#2ecc71" if ctx_val >= base_val else "#e74c3c"
        ax.plot([0, 1], [base_val, ctx_val], color=color, alpha=0.5, linewidth=1.4, zorder=2)
        ax.scatter([0, 1], [base_val, ctx_val],
                   color=[PALETTE["baseline"], PALETTE["context"]],
                   s=40, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Context"], **FONT_TICK)
    ax.set_title(f"{METRIC_LABELS[metric]} — per run", **FONT_TITLE)
    # Legend for direction
    up   = mpatches.Patch(color="#2ecc71", alpha=0.7, label="Context ≥ Baseline")
    down = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Context < Baseline")
    ax.legend(handles=[up, down], fontsize=8, loc="lower right")


def plot_delta_distribution(df, ax, metric):
    """
    Per-run delta (context − baseline) as a strip + mean line.
    """
    runs   = sorted(df["run"].unique())
    deltas = []
    for run in runs:
        base_val = df[(df["model"] == "baseline") & (df["run"] == run)][metric].values[0]
        ctx_val  = df[(df["model"] == "context")  & (df["run"] == run)][metric].values[0]
        deltas.append(ctx_val - base_val)

    deltas = np.array(deltas)
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

    ax.axhline(0, color="#aaa", linewidth=1, linestyle="--", zorder=1)
    ax.scatter(range(len(deltas)), deltas, color=colors, s=70, zorder=3)
    ax.axhline(deltas.mean(), color=PALETTE["context"], linewidth=2,
               linestyle="-", zorder=2, label=f"Mean Δ = {deltas.mean():+.4f}")

    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"Run {r}" for r in runs], **FONT_TICK)
    ax.set_title(f"Δ {METRIC_LABELS[metric]} (context − baseline)", **FONT_TITLE)
    ax.legend(fontsize=8)


def plot_stat_table(ttest_df, ax):
    """
    Render the t-test results as a clean table.
    """
    ax.axis("off")
    col_labels = ["Metric", "Mean Δ", "t-stat", "p-value", "p < 0.05", "p < 0.10"]
    table_data = []
    for _, row in ttest_df.iterrows():
        table_data.append([
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
    table.set_fontsize(9.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1a1a2e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "#fafafa")
            # Color p<0.05 column
            if j == 4:
                val = table_data[i - 1][j]
                table[i, j].set_text_props(
                    color="#2ecc71" if val == "✓" else "#e74c3c",
                    fontweight="bold"
                )

    ax.set_title("Paired t-test: Context vs Baseline", **FONT_TITLE, pad=20)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(df, ttest_df, output_path):
    """
    Combine all metrics into a single figure.
    """
    focus_metric = "f1_macro"  # primary metric for the detailed plots

    fig = plt.figure(figsize=(18, 14), facecolor="#fafafa")
    fig.suptitle(
        "Context-Aware vs Baseline Model — Multi-Run Evaluation",
        fontsize=16, fontweight="bold", color="#1a1a2e", y=0.98,
    )

    gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.4)

    # Row 0: bar charts for all 5 metrics
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        plot_bar_with_ci(df, ax, metric)

    # Row 1: per-run lines, delta strip, stat table
    ax_lines = fig.add_subplot(gs[1, 0:2])
    plot_run_lines(df, ax_lines, focus_metric)

    ax_delta = fig.add_subplot(gs[1, 2:4])
    plot_delta_distribution(df, ax_delta, focus_metric)

    ax_table = fig.add_subplot(gs[2, :])
    plot_stat_table(ttest_df, ax_table)

    # Legend for bar charts
    handles = [
        mpatches.Patch(color=PALETTE["baseline"], alpha=0.85, label="Baseline"),
        mpatches.Patch(color=PALETTE["context"],  alpha=0.85, label="Context-aware"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.98, 0.96))

    out = Path(output_path) / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"[saved] Figure → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    df   = pd.read_csv(args.results_path)

    # Normalise column names
    df.columns = df.columns.str.strip()
    df["run"]  = df["run"].astype(int)

    print("\n===== Summary statistics =====")
    summary = compute_summary(df)
    print(summary.to_string())

    print("\n===== Paired t-tests (context vs baseline) =====")
    ttest_df = paired_ttests(df)
    print(ttest_df.to_string(index=False))

    # Save stats to CSV
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ttest_df.to_csv(out_dir / "statistical_tests.csv", index=False)
    print(f"\n[saved] t-test results → {out_dir / 'statistical_tests.csv'}")

    build_figure(df, ttest_df, args.output_path)


if __name__ == "__main__":
    main()