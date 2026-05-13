"""
Visualize training and validation loss curves from multi-run training results.

python visualize_loss.py --results_path "results/train_results.csv" \
                          --output_path  "results/figures/"
                          [--run_id 2026-05-08_15-29-22]   # defaults to latest run
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PALETTE = {
    "train": "#6C8EBF",
    "val":   "#D4763B",
}

MODEL_LABELS = {
    "baseline":     "Baseline",
    "hierarchical": "Context-Aware",
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True, help="Path to train_results.csv")
    p.add_argument("--output_path",  required=True, help="Directory to save figures")
    p.add_argument("--run_id", default=None, help="Run ID to plot (defaults to latest run)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(df, ax, model_type):
    """
    For one model type, plot train and val loss curves for a single run.
    """
    run_df = df[df["model"] == model_type].sort_values("epoch")

    train = run_df[run_df["loss"].notna()][["epoch", "loss"]]
    val   = run_df[run_df["eval_loss"].notna()][["epoch", "eval_loss"]]

    ax.plot(train["epoch"], train["loss"],
            color=PALETTE["train"], linewidth=2.2, label="Train loss")
    ax.plot(val["epoch"], val["eval_loss"],
            color=PALETTE["val"], linewidth=2.2, label="Val loss")

    ax.set_title(MODEL_LABELS.get(model_type, model_type), **FONT_TITLE)
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Loss",  **FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK["fontsize"])


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(df, run_id, output_path):
    model_types = sorted(df["model"].unique())
    n = len(model_types)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), facecolor="#fafafa")
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Training and Validation Loss — Run {run_id}",
        fontsize=15, fontweight="bold", color="#1a1a2e", y=1.02,
    )

    for ax, model_type in zip(axes, model_types):
        plot_loss_curves(df, ax, model_type)

    train_line = mlines.Line2D([], [], color=PALETTE["train"], linewidth=2, label="Train loss")
    val_line   = mlines.Line2D([], [], color=PALETTE["val"],   linewidth=2, label="Val loss")
    fig.legend(
        handles=[train_line, val_line],
        loc="upper right", fontsize=9, framealpha=0.9,
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.tight_layout()
    out = Path(output_path) / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"[saved] Figure → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    df = pd.read_csv(args.results_path)
    df.columns = df.columns.str.strip()

    run_id = args.run_id or sorted(df["run_id"].unique())[-1]
    if run_id not in df["run_id"].values:
        raise ValueError(f"run_id '{run_id}' not found. Available: {sorted(df['run_id'].unique())}")

    df = df[df["run_id"] == run_id]
    print(f"[info] Plotting run: {run_id}")

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    build_figure(df, run_id, args.output_path)


if __name__ == "__main__":
    main()
