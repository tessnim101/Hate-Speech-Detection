"""
Visualize and compare baseline, hierarchical, and backtranslation model results.

Usage:
    python3 visualize.py --results_path results/
    python3 visualize.py --results_path /home/bechrifa/Hate-Speech-Detection/results/
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER  = ["baseline", "hierarchical", "cross_attention", "augmented"]
MODEL_LABELS = {"baseline": "Baseline", "hierarchical": "Hierarchical", "cross_attention": "Cross-Attention", "augmented": "BT-Augmented"}
COLORS       = {"baseline": "#4C72B0", "hierarchical": "#DD8452", "cross_attention": "#9EDBFF", "augmented": "#55A868"}

METRICS = {
    "eval_f1_macro":  "F1 Macro",
    "eval_accuracy":  "Accuracy",
    "eval_f1_class0": "F1 Class 0\n(non-hate)",
    "eval_f1_class1": "F1 Class 1\n(hate)",
}


def load_results(res_dir: Path):
    train_df = pd.read_csv(res_dir / "train_results.csv")
    test_df  = pd.read_csv(res_dir / "test_results.csv")

    # If multiple runs exist, keep only the most recent run per model
    for df in [train_df, test_df]:
        df["run_id"] = df["run_id"].astype(str)

    latest = test_df.groupby("model")["run_id"].max().to_dict()
    train_df = train_df[train_df.apply(lambda r: latest.get(r["model"]) == r["run_id"], axis=1)]
    test_df  = test_df[test_df.apply(lambda r: latest.get(r["model"]) == r["run_id"], axis=1)]

    available = [m for m in MODEL_ORDER if m in test_df["model"].values]
    train_df  = train_df[train_df["model"].isin(available)]
    test_df   = test_df[test_df["model"].isin(available)]

    return train_df, test_df, available


def plot_test_bars(test_df, available, ax):
    metrics   = list(METRICS.keys())
    n_metrics = len(metrics)
    n_models  = len(available)
    width     = 0.75 / n_models
    x         = np.arange(n_metrics)

    for i, model in enumerate(available):
        row    = test_df[test_df["model"] == model].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width,
                        label=MODEL_LABELS[model], color=COLORS[model],
                        alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Test Set — All Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_class_f1(test_df, available, ax):
    x      = np.arange(len(available))
    width  = 0.35
    labels = [MODEL_LABELS[m] for m in available]
    f1_0   = [test_df[test_df["model"] == m]["eval_f1_class0"].values[0] for m in available]
    f1_1   = [test_df[test_df["model"] == m]["eval_f1_class1"].values[0] for m in available]

    b0 = ax.bar(x - width / 2, f1_0, width, label="Class 0 (non-hate)", color="#7fbfff", edgecolor="white")
    b1 = ax.bar(x + width / 2, f1_1, width, label="Class 1 (hate)",     color="#ff8c69", edgecolor="white")

    for bars in [b0, b1]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Test Set — Per-Class F1", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_val_curve(train_df, available, ax, metric, title):
    for model in available:
        mdf = train_df[train_df["model"] == model].copy()
        if metric not in mdf.columns:
            continue
        mdf = mdf.dropna(subset=[metric]).sort_values("epoch")
        ax.plot(mdf["epoch"], mdf[metric],
                marker="o", markersize=4, linewidth=2,
                color=COLORS[model], label=MODEL_LABELS[model])
        best_idx = mdf[metric].idxmax()
        ax.scatter(mdf.loc[best_idx, "epoch"], mdf.loc[best_idx, metric],
                   s=90, zorder=5, color=COLORS[model],
                   edgecolors="black", linewidths=1.2)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def print_summary(test_df, available):
    print("\n─── Test Results Summary ───")
    rows = []
    for model in available:
        row = test_df[test_df["model"] == model].iloc[0]
        rows.append({
            "Model":           MODEL_LABELS[model],
            "F1 Macro":        f"{row['eval_f1_macro']:.4f}",
            "Accuracy":        f"{row['eval_accuracy']:.4f}",
            "F1 Class0":       f"{row['eval_f1_class0']:.4f}",
            "F1 Class1 (hate)":f"{row['eval_f1_class1']:.4f}",
        })
    print(pd.DataFrame(rows).set_index("Model").to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True)
    args    = parser.parse_args()
    res_dir = Path(args.results_path)

    train_df, test_df, available = load_results(res_dir)
    print(f"Models found: {available}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Hate Speech Detection — Model Comparison", fontsize=15, fontweight="bold", y=1.01)

    plot_test_bars(test_df,  available, axes[0, 0])
    plot_class_f1(test_df,  available, axes[0, 1])
    plot_val_curve(train_df, available, axes[1, 0], "eval_f1_macro",  "Validation F1 Macro per Epoch")
    plot_val_curve(train_df, available, axes[1, 1], "eval_f1_class1", "Validation F1 Class 1 (hate) per Epoch")

    plt.tight_layout()
    out = res_dir / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

    print_summary(test_df, available)


if __name__ == "__main__":
    main()
