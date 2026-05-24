"""
Interpretability analysis for Hierarchical, BT-Augmented, and Cross-Attention models.

Analyses:
  1. Thread-level attention weights per position per class
     (Hierarchical and BT-Augmented: root/parent/tweet)
  2. Context segment importance per class
     (Cross-Attention: root/parent)
  3. Unified 3-model comparison plot

Usage:
    python3 interpretability_extended.py \
        --dataset_path          data/spanish_subset/ \
        --context_path          results/best_model_context/ \
        --bt_path               results/best_model_bt/ \
        --cross_attention_path  results/best_model_cross_attention/ \
        --results_path          figures/
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import CONFIG
from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text
from utils.inference_utils import enc, load_hierarchical, load_cross_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--context_path",         required=True)
    p.add_argument("--bt_path",              required=True)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--results_path",         default="figures/")
    p.add_argument("--batch_size",           type=int, default=32)
    return p.parse_args()


# Hierarchical: root / parent / tweet
POSITIONS_H  = ["Root tweet", "Parent tweet", "Target tweet"]
COLORS_H     = ["#4C72B0", "#DD8452", "#55A868"]

# Cross-attention segments: root / parent
POSITIONS_CA = ["Root tweet", "Parent tweet"]
COLORS_CA    = ["#4C72B0", "#DD8452"]


@torch.no_grad()
def run_hierarchical_attention(model, tokenizer, df, batch_size, device):
    all_probs, all_weights = [], []
    texts        = df["text"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_enc = enc(tokenizer, texts[i:i+batch_size],        CONFIG["max_len"], device)
        p_enc = enc(tokenizer, parent_texts[i:i+batch_size], CONFIG["max_len"], device)
        r_enc = enc(tokenizer, root_texts[i:i+batch_size],   CONFIG["max_len"], device)

        out = model(
            root_input_ids=        r_enc["input_ids"],
            root_attention_mask=   r_enc["attention_mask"],
            parent_input_ids=      p_enc["input_ids"],
            parent_attention_mask= p_enc["attention_mask"],
            tweet_input_ids=       t_enc["input_ids"],
            tweet_attention_mask=  t_enc["attention_mask"],
            return_attention_weights=True,
        )

        probs = torch.softmax(out["logits"], dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_weights.append(out["attention_weights"].cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_weights)


@torch.no_grad()
def run_cross_attention_weights(model, tokenizer, df, batch_size, device):
    all_weights, all_ctx_ids = [], []
    texts        = df["text"].tolist()
    root_texts   = df["root_text"].tolist()
    parent_texts = df["parent_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_b = texts[i:i+batch_size]
        r_b = root_texts[i:i+batch_size]
        p_b = parent_texts[i:i+batch_size]

        context = [
            tokenizer.sep_token.join(x for x in [r, p] if x and str(x).strip())
            for r, p in zip(r_b, p_b)
        ]

        ctx_enc = enc(tokenizer, context, CONFIG["max_len_context"], device)
        twt_enc = enc(tokenizer, t_b,     CONFIG["max_len_tweet"],   device)

        out = model(
            context_input_ids=       ctx_enc["input_ids"],
            context_attention_mask=  ctx_enc["attention_mask"],
            tweet_input_ids=         twt_enc["input_ids"],
            tweet_attention_mask=    twt_enc["attention_mask"],
            return_attention_weights=True,
        )

        avg = out["attention_weights"].cpu().numpy().mean(axis=1).mean(axis=1)
        all_weights.extend([avg[j] for j in range(len(t_b))])
        all_ctx_ids.extend(ctx_enc["input_ids"].cpu().tolist())

    return all_weights, all_ctx_ids


def get_segment_masks(tokenizer, root, parent, ctx_ids):
    """Return boolean masks over ctx_ids for root and parent segments."""
    masks = {}
    for name, text in [("root", root), ("parent", parent)]:
        mask = np.zeros(len(ctx_ids), dtype=bool)
        if text and str(text).strip():
            seg_ids = tokenizer.encode(text, add_special_tokens=False)
            for start in range(len(ctx_ids) - len(seg_ids) + 1):
                if ctx_ids[start : start + len(seg_ids)] == seg_ids:
                    mask[start : start + len(seg_ids)] = True
                    break
        masks[name] = mask
    return masks


def compute_ca_segment_importance(all_weights, all_ctx_ids, df, tokenizer):
    root_texts   = df["root_text"].tolist()
    parent_texts = df["parent_text"].tolist()
    results      = []

    for idx, (weights, ctx_ids) in enumerate(zip(all_weights, all_ctx_ids)):
        masks = get_segment_masks(tokenizer, root_texts[idx], parent_texts[idx], ctx_ids)
        total = weights.sum() + 1e-9
        results.append([
            float(weights[masks["root"]].sum()   / total),
            float(weights[masks["parent"]].sum() / total),
        ])
    return np.array(results)  # (N, 2)


def _draw_bars(ax, positions, means, stds, colors, ylabel=""):
    bars = ax.bar(
        positions, means, yerr=stds, capsize=5,
        color=colors, alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": "#444"},
    )
    ax.set_ylim(0, min(1.0, (np.array(means) + np.array(stds)).max() + 0.12))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    for bar, mean in zip(bars, means):
        if bar.get_height() > 0.06:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{mean:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
            )


def plot_aggregate_attention(weights, labels, model_name, out_dir, positions, colors):
    df_w          = pd.DataFrame(weights, columns=positions)
    df_w["label"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, (lbl, name) in zip(axes, [(0, "Not stereotype (class 0)"),
                                       (1, "Stereotype (class 1)")]):
        sub   = df_w[df_w["label"] == lbl][positions]
        _draw_bars(ax, positions, sub.mean().values, sub.std().values, colors,
                   ylabel="Mean attention weight" if lbl == 0 else "")
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.suptitle(f"{model_name} — Attention Weights by Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(out_dir) / f"attention_by_class_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


def plot_attention_distribution(weights, model_name, out_dir, positions, colors):
    n   = weights.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 4))

    parts = ax.violinplot(
        [weights[:, i] for i in range(n)],
        positions=list(range(1, n + 1)),
        showmeans=True, showmedians=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(list(range(1, n + 1)))
    ax.set_xticklabels(positions, fontsize=11)
    ax.set_ylabel("Attention weight", fontsize=11)
    ax.set_title(f"{model_name} — Distribution of Attention Weights", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = Path(out_dir) / f"attention_distribution_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


def plot_all_models_comparison(hier_weights, bt_weights, ca_weights, labels, out_dir):
    """
    3x2 grid:
      rows = Hierarchical / BT-Augmented / Cross-Attention
      cols = Not stereotype / Stereotype

    Hierarchical and BT: 3 bars (root/parent/tweet)
    Cross-Attention:      2 bars (root/parent segments)
    """
    classes = [(0, "Not stereotype (class 0)"), (1, "Stereotype (class 1)")]
    rows    = [
        ("Hierarchical",    hier_weights, POSITIONS_H,  COLORS_H),
        ("BT-Augmented",    bt_weights,   POSITIONS_H,  COLORS_H),
        ("Cross-Attention", ca_weights,   POSITIONS_CA, COLORS_CA),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        "Context Attention Weights by Class — All Models",
        fontsize=14, fontweight="bold",
    )

    for row_idx, (model_name, weights, positions, colors) in enumerate(rows):
        for col_idx, (lbl, class_name) in enumerate(classes):
            ax    = axes[row_idx][col_idx]
            mask  = labels == lbl
            means = weights[mask].mean(axis=0)
            stds  = weights[mask].std(axis=0)
            _draw_bars(ax, positions, means, stds, colors,
                       ylabel="Mean attention" if col_idx == 0 else "")
            ax.set_title(f"{model_name} — {class_name}", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = Path(out_dir) / "all_models_attention_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out_dir = Path(args.results_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, df_test = load_data(args.dataset_path)
    df_test    = filter_contextual_tweets(df_test)
    df_test    = ids_to_text(df_test.copy())
    labels     = df_test["stereotype"].values
    print(f"[data] Test set: {len(df_test)} samples")

    # Hierarchical
    print("\n[loading] Hierarchical...")
    hier_model, hier_tokenizer = load_hierarchical(args.context_path, device)
    print("[inference] Extracting hierarchical attention weights...")
    hier_probs, hier_weights = run_hierarchical_attention(
        hier_model, hier_tokenizer, df_test, args.batch_size, device
    )
    print("\n===== Hierarchical Attention Weights =====")
    for i, pos in enumerate(POSITIONS_H):
        print(f"  {pos:<15}  mean={hier_weights[:, i].mean():.3f}  std={hier_weights[:, i].std():.3f}")

    plot_aggregate_attention(hier_weights, labels, "Hierarchical", out_dir, POSITIONS_H, COLORS_H)
    plot_attention_distribution(hier_weights, "Hierarchical", out_dir, POSITIONS_H, COLORS_H)

    # BT augmented 
    print("\n[loading] BT-Augmented...")
    bt_model, bt_tokenizer = load_hierarchical(args.bt_path, device)
    print("[inference] Extracting BT-augmented attention weights...")
    bt_probs, bt_weights = run_hierarchical_attention(
        bt_model, bt_tokenizer, df_test, args.batch_size, device
    )
    print("\n===== BT-Augmented Attention Weights =====")
    for i, pos in enumerate(POSITIONS_H):
        print(f"  {pos:<15}  mean={bt_weights[:, i].mean():.3f}  std={bt_weights[:, i].std():.3f}")

    plot_aggregate_attention(bt_weights, labels, "BT-Augmented", out_dir, POSITIONS_H, COLORS_H)
    plot_attention_distribution(bt_weights, "BT-Augmented", out_dir, POSITIONS_H, COLORS_H)

    # cross attention
    print("\n[loading] Cross-Attention...")
    ca_model, ca_tokenizer = load_cross_attention(args.cross_attention_path, device)
    print("[inference] Extracting cross-attention weights...")
    ca_token_weights, ca_ctx_ids = run_cross_attention_weights(
        ca_model, ca_tokenizer, df_test, args.batch_size, device
    )
    print("[analysis] Computing segment importance...")
    ca_seg_weights = compute_ca_segment_importance(
        ca_token_weights, ca_ctx_ids, df_test, ca_tokenizer
    )
    print("\n===== Cross-Attention Segment Importance =====")
    for i, seg in enumerate(POSITIONS_CA):
        print(f"  {seg:<15}  mean={ca_seg_weights[:, i].mean():.3f}  std={ca_seg_weights[:, i].std():.3f}")
    for lbl, name in [(0, "Not stereotype"), (1, "Stereotype")]:
        mask = labels == lbl
        print(f"  {name}:")
        for i, seg in enumerate(POSITIONS_CA):
            print(f"    {seg:<15}  {ca_seg_weights[mask, i].mean():.3f}")

    plot_aggregate_attention(ca_seg_weights, labels, "Cross-Attention", out_dir, POSITIONS_CA, COLORS_CA)
    plot_attention_distribution(ca_seg_weights, "Cross-Attention", out_dir, POSITIONS_CA, COLORS_CA)
    plot_all_models_comparison(hier_weights, bt_weights, ca_seg_weights, labels, out_dir)

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()