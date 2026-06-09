"""
Visualize and compare baseline and context-aware models across multiple seeds.

python analysis/visualize.py \
    --results_path              "results/" \
    --results_path_baseline     "results/results_baseline/" \
    --results_path_hierarchical "results/results_hierarchical/" \
    --results_path_cross_attention "results/results_cross_attention/" \
    --results_path_augmented    "results/results_augmented/" \
    --output_path               "results/figures/"
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SEEDS = [42, 123, 456]

MODEL_ORDER  = ["baseline", "hierarchical", "augmented", "cross_attention"]
MODEL_LABELS = {
    "baseline":        "Baseline",
    "hierarchical":    "Hierarchical",
    "augmented":       "BT-Augmented",
    "cross_attention": "Cross-Attention",
}
COLORS = {
    "baseline":        "#4C72B0",
    "hierarchical":    "#DD8452",
    "augmented":       "#55A868",
    "cross_attention": "#8B5CF6",
}

METRICS = {
    "eval_f1_macro":  "F1 Macro",
    "eval_accuracy":  "Accuracy",
    "eval_f1_class0": "F1 Class 0\n(non-stereo)",
    "eval_f1_class1": "F1 Class 1\n(stereo)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_path",               required=True, help="Path to results/ containing all_test_results.csv")
    p.add_argument("--results_path_baseline",        required=True)
    p.add_argument("--results_path_hierarchical",    required=True)
    p.add_argument("--results_path_cross_attention", required=True)
    p.add_argument("--results_path_augmented",       required=True)
    p.add_argument("--output_path",                  required=True)
    return p.parse_args()


def load_test_results(results_path):
    """Load all_test_results.csv — one row per model per seed."""
    df = pd.read_csv(Path(results_path) / "all_test_results.csv")
    df["seed"] = df["run_id"].apply(lambda x: int(x.split("_seed")[-1]))
    available  = [m for m in MODEL_ORDER if m in df["model"].unique()]
    return df, available


def load_train_results(results_dir):
    """Load and concatenate train_results.csv from all seed subfolders."""
    dfs = []
    for seed in SEEDS:
        csv_path = Path(results_dir) / f"seed_{seed}" / "train_results.csv"
        if not csv_path.exists():
            print(f"  [warn] Not found: {csv_path} — skipping seed {seed}")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def compute_mean_std_per_epoch(df, metric):
    grouped  = df[df[metric].notna()].groupby("epoch")[metric]
    mean     = grouped.mean()
    std      = grouped.std().fillna(0)
    return mean, std


def plot_test_bars(test_df, available, ax):
    metrics   = list(METRICS.keys())
    n_metrics = len(metrics)
    n_models  = len(available)
    width     = 0.75 / n_models
    x         = np.arange(n_metrics)

    for i, model in enumerate(available):
        mdf    = test_df[test_df["model"] == model]
        means  = [mdf[m].mean() for m in metrics]
        stds   = [mdf[m].std()  for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width

        bars = ax.bar(
            x + offset, means, width,
            label=MODEL_LABELS[model], color=COLORS[model],
            alpha=0.88, edgecolor="white", linewidth=0.8,
            yerr=stds, capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#444"},
        )
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Test Set — All Metrics (mean ± std)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_class_f1(test_df, available, ax):
    x      = np.arange(len(available))
    width  = 0.35
    labels = [MODEL_LABELS[m] for m in available]

    f1_0_mean = [test_df[test_df["model"] == m]["eval_f1_class0"].mean() for m in available]
    f1_0_std  = [test_df[test_df["model"] == m]["eval_f1_class0"].std()  for m in available]
    f1_1_mean = [test_df[test_df["model"] == m]["eval_f1_class1"].mean() for m in available]
    f1_1_std  = [test_df[test_df["model"] == m]["eval_f1_class1"].std()  for m in available]

    b0 = ax.bar(x - width / 2, f1_0_mean, width,
                yerr=f1_0_std, capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#444"},
                label="Class 0 (non-stereo)", color="#7fbfff", edgecolor="white")
    b1 = ax.bar(x + width / 2, f1_1_mean, width,
                yerr=f1_1_std, capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#444"},
                label="Class 1 (stereo)", color="#ff8c69", edgecolor="white")

    for bars, means, stds in [(b0, f1_0_mean, f1_0_std), (b1, f1_1_mean, f1_1_std)]:
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Test Set — Per-Class F1 (mean ± std)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_val_curve(train_dfs, available, ax, metric, title):
    """Mean ± std band across seeds for validation metric per epoch."""
    for model in available:
        df = train_dfs.get(model)
        if df is None or metric not in df.columns:
            continue

        mean, std = compute_mean_std_per_epoch(df, metric)

        ax.plot(mean.index, mean.values,
                marker="o", markersize=4, linewidth=2,
                color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(mean.index,
                        mean - std,
                        mean + std,
                        color=COLORS[model], alpha=0.15)

        # Mark best mean epoch
        best_epoch = mean.idxmax()
        ax.scatter(best_epoch, mean[best_epoch],
                   s=90, zorder=5, color=COLORS[model],
                   edgecolors="black", linewidths=1.2)

        # Individual seed lines faded
        for seed in df["seed"].unique():
            seed_df = df[(df["seed"] == seed) & df[metric].notna()].sort_values("epoch")
            ax.plot(seed_df["epoch"], seed_df[metric],
                    color=COLORS[model], linewidth=0.7, alpha=0.25, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score",  fontsize=10)
    ax.set_title(title,     fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def print_summary(test_df, available):
    print("\n─── Test Results Summary (mean ± std) ───")
    rows = []
    for model in available:
        mdf = test_df[test_df["model"] == model]
        rows.append({
            "Model":            MODEL_LABELS[model],
            "F1 Macro":         f"{mdf['eval_f1_macro'].mean():.4f} ± {mdf['eval_f1_macro'].std():.4f}",
            "Accuracy":         f"{mdf['eval_accuracy'].mean():.4f} ± {mdf['eval_accuracy'].std():.4f}",
            "F1 Class0":        f"{mdf['eval_f1_class0'].mean():.4f} ± {mdf['eval_f1_class0'].std():.4f}",
            "F1 Class1 (stereo)":f"{mdf['eval_f1_class1'].mean():.4f} ± {mdf['eval_f1_class1'].std():.4f}",
        })
    print(pd.DataFrame(rows).set_index("Model").to_string())


def main():
    args = parse_args()

    # Load test results
    test_df, available = load_test_results(args.results_path)
    print(f"Models found: {available}")

    # Load train results per model
    train_dfs = {
        "baseline":        load_train_results(args.results_path_baseline),
        "hierarchical":    load_train_results(args.results_path_hierarchical),
        "cross_attention": load_train_results(args.results_path_cross_attention),
        "augmented":       load_train_results(args.results_path_augmented),
    }
    # Keep only available models
    train_dfs = {k: v for k, v in train_dfs.items() if k in available and v is not None}

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Stereotype Detection — Model Comparison (mean ± std across seeds)",
                 fontsize=15, fontweight="bold", y=1.01)

    plot_test_bars(test_df,  available,  axes[0, 0])
    plot_class_f1( test_df,  available,  axes[0, 1])
    plot_val_curve(train_dfs, available, axes[1, 0],
                   "eval_f1_macro",  "Validation F1 Macro per Epoch")
    plot_val_curve(train_dfs, available, axes[1, 1],
                   "eval_f1_class1", "Validation F1 Class 1 (stereo) per Epoch")

    plt.tight_layout()
    out = Path(args.output_path) / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

    print_summary(test_df, available)


if __name__ == "__main__":
    main()