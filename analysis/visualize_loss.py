"""
Visualize training and validation loss curves from multi-run training results.
Shows mean ± std band across all seeds per model.

python analysis/visualize_loss.py --results_path_baseline      "results/results_baseline/" \
                          --results_path_hierarchical  "results/results_hierarchical/" \
                          --results_path_cross_attention "results/results_cross_attention/" \
                          --results_path_augmented     "results/results_augmented/" \
                          --output_path                "figures/"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


SEEDS = [42, 123, 456]

PALETTE = {
    "train": "#6C8EBF",
    "val":   "#D4763B",
}

MODEL_LABELS = {
    "baseline":        "Baseline",
    "hierarchical":    "Hierarchical",
    "cross_attention": "Cross-Attention",
    "augmented":       "Augmented",
}

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_LABEL = {"fontsize": 10, "color": "#333333"}
FONT_TICK  = {"fontsize": 9,  "color": "#555555"}

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
    p.add_argument("--results_path_baseline",        required=True)
    p.add_argument("--results_path_hierarchical",    required=True)
    p.add_argument("--results_path_cross_attention", required=True)
    p.add_argument("--results_path_augmented",       required=True)
    p.add_argument("--output_path",                  required=True)
    return p.parse_args()


def load_model_data(results_dir):
    """
    Load and concatenate train_results.csv from all seed subfolders.
    Returns a dict: {seed: df} and a combined df with a 'seed' column.
    """
    dfs = []
    for seed in SEEDS:
        csv_path = Path(results_dir) / f"seed_{seed}" / "train_results.csv"
        if not csv_path.exists():
            print(f"  [warn] Not found: {csv_path} — skipping seed {seed}")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def compute_mean_std(df, col):
    """
    For a given column (loss or eval_loss), compute mean and std per epoch
    across seeds.
    """
    grouped = df[df[col].notna()].groupby("epoch")[col]
    mean = grouped.mean()
    std  = grouped.std().fillna(0)
    return mean, std


def plot_loss_curves(df, ax, model_name):
    """
    Plot mean train and val loss with ± std shaded band.
    """
    train_mean, train_std = compute_mean_std(df, "loss")
    val_mean,   val_std   = compute_mean_std(df, "eval_loss")

    # Train loss
    ax.plot(train_mean.index, train_mean.values,
            color=PALETTE["train"], linewidth=2.2, label="Train loss")
    ax.fill_between(
        train_mean.index,
        train_mean - train_std,
        train_mean + train_std,
        color=PALETTE["train"], alpha=0.2,
    )

    # Val loss
    ax.plot(val_mean.index, val_mean.values,
            color=PALETTE["val"], linewidth=2.2, label="Val loss")
    ax.fill_between(
        val_mean.index,
        val_mean - val_std,
        val_mean + val_std,
        color=PALETTE["val"], alpha=0.2,
    )

    # Individual seed lines (faded)
    for seed in df["seed"].unique():
        seed_df = df[df["seed"] == seed]
        train_s = seed_df[seed_df["loss"].notna()].sort_values("epoch")
        val_s   = seed_df[seed_df["eval_loss"].notna()].sort_values("epoch")
        ax.plot(train_s["epoch"], train_s["loss"],
                color=PALETTE["train"], linewidth=0.8, alpha=0.3, linestyle="--")
        ax.plot(val_s["epoch"], val_s["eval_loss"],
                color=PALETTE["val"], linewidth=0.8, alpha=0.3, linestyle="--")

    ax.set_title(MODEL_LABELS.get(model_name, model_name), **FONT_TITLE)
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Loss",  **FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK["fontsize"])


def build_figure(model_data, output_path):
    available = [(name, df) for name, df in model_data.items() if df is not None]
    n = len(available)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), facecolor="#fafafa")
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Training & Validation Loss — Mean ± Std across Seeds",
        fontsize=15, fontweight="bold", color="#1a1a2e", y=1.02,
    )

    for ax, (model_name, df) in zip(axes, available):
        plot_loss_curves(df, ax, model_name)

    # Legend
    train_line  = mlines.Line2D([], [], color=PALETTE["train"], linewidth=2,   label="Train loss (mean)")
    val_line    = mlines.Line2D([], [], color=PALETTE["val"],   linewidth=2,   label="Val loss (mean)")
    train_shade = mlines.Line2D([], [], color=PALETTE["train"], linewidth=6,   alpha=0.2, label="Train ± std")
    val_shade   = mlines.Line2D([], [], color=PALETTE["val"],   linewidth=6,   alpha=0.2, label="Val ± std")
    seed_line   = mlines.Line2D([], [], color="#888888",        linewidth=0.8, alpha=0.4, linestyle="--", label="Individual seeds")

    fig.legend(
        handles=[train_line, train_shade, val_line, val_shade, seed_line],
        loc="upper right", fontsize=9, framealpha=0.9,
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.tight_layout()
    out = Path(output_path) / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"[saved] Figure → {out}")
    plt.close()


def main():
    args = parse_args()

    model_data = {
        "baseline":        load_model_data(args.results_path_baseline),
        "hierarchical":    load_model_data(args.results_path_hierarchical),
        "cross_attention": load_model_data(args.results_path_cross_attention),
        "augmented":       load_model_data(args.results_path_augmented),
    }

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    build_figure(model_data, args.output_path)


if __name__ == "__main__":
    main()