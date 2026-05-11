"""
Interpretability analysis for the Hierarchical Context Model.

Two analyses:
  1. Thread-level attention weights — how much does the tweet attend
     to root vs parent vs itself across the test set
  2. Per-sample case studies — show attention weights for specific
     interesting examples (context_wins, baseline_wins, ties)

Usage:
    python interpretability.py \
        --dataset_path  data/spanish_subset/ \
        --context_path  results/best_model_context/ \
        --per_sample    results/inference_per_sample.csv \
        --results_path  results/figures/
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text
from modeling.models import HierarchicalContextModel
from config import CONFIG


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--context_path", required=True)
    p.add_argument("--per_sample",   required=True,
                   help="Path to inference_per_sample.csv")
    p.add_argument("--results_path", required=True)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--n_cases",      type=int, default=4,
                   help="Number of case studies per verdict category")
    return p.parse_args()


# ── Model with attention output ───────────────────────────────────────────────

class HierarchicalWithAttention(HierarchicalContextModel):
    """
    Subclass that returns attention weights from the thread attention layer.
    The original forward() discards them with `_`; here we capture them.
    """

    def forward_with_attn(
        self,
        root_input_ids,      root_attention_mask,
        parent_input_ids,    parent_attention_mask,
        tweet_input_ids,     tweet_attention_mask,
    ):
        root_cls   = self.encode(root_input_ids,   root_attention_mask)
        parent_cls = self.encode(parent_input_ids, parent_attention_mask)
        tweet_cls  = self.encode(tweet_input_ids,  tweet_attention_mask)

        positions = torch.arange(3, device=root_cls.device)
        pos_emb   = self.position_embeddings(positions).unsqueeze(0)
        thread    = torch.stack([root_cls, parent_cls, tweet_cls], dim=1) + pos_emb

        root_empty   = (root_attention_mask.sum(dim=1)   <= 2)
        parent_empty = (parent_attention_mask.sum(dim=1) <= 2)
        tweet_empty  = torch.zeros(root_empty.shape[0], dtype=torch.bool,
                                   device=root_cls.device)
        key_padding_mask = torch.stack([root_empty, parent_empty, tweet_empty], dim=1)

        # need_weights=True returns (attn_out, attn_weights)
        # average_attn_weights=True averages across the 8 heads → shape (B, 1, 3)
        attn_out, attn_weights = self.thread_attention(
            query=thread[:, 2:, :],
            key=thread,
            value=thread,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        pooled = attn_out.squeeze(1)
        logits = self.classifier(pooled)
        probs  = torch.softmax(logits, dim=-1)

        # attn_weights: (B, 1, 3) → squeeze to (B, 3)
        # columns: [root_weight, parent_weight, tweet_weight]
        return probs, attn_weights.squeeze(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(context_path, device):
    model = HierarchicalWithAttention(CONFIG["model_name"])
    weights = load_file(
        str(Path(context_path) / "model.safetensors"), device=str(device)
    )
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def encode(tokenizer, texts, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=CONFIG["max_len"],
        return_tensors="pt",
    ).to(device)


@torch.no_grad()
def run_attention(model, tokenizer, df, batch_size, device):
    all_probs, all_weights = [], []

    texts        = df["text"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t = encode(tokenizer, texts[i:i+batch_size],        device)
        p = encode(tokenizer, parent_texts[i:i+batch_size], device)
        r = encode(tokenizer, root_texts[i:i+batch_size],   device)

        probs, weights = model.forward_with_attn(
            root_input_ids=r["input_ids"],      root_attention_mask=r["attention_mask"],
            parent_input_ids=p["input_ids"],    parent_attention_mask=p["attention_mask"],
            tweet_input_ids=t["input_ids"],     tweet_attention_mask=t["attention_mask"],
        )

        all_probs.append(probs.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_weights)


# ── Plot 1: aggregate attention distribution ──────────────────────────────────

def plot_aggregate_attention(weights, labels, out_dir):
    """
    Bar chart: mean attention weight to root / parent / tweet
    split by true label (not hate vs hate).
    """
    POSITIONS = ["Root tweet", "Parent tweet", "Target tweet"]
    df = pd.DataFrame(weights, columns=POSITIONS)
    df["label"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (lbl, name, color) in zip(
        axes,
        [(0, "Not hate speech (class 0)", "#4C72B0"),
         (1, "Hate speech (class 1)",     "#DD8452")]
    ):
        sub   = df[df["label"] == lbl][POSITIONS]
        means = sub.mean()
        stds  = sub.std()

        bars = ax.bar(POSITIONS, means, yerr=stds, capsize=5,
                      color=color, alpha=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("Mean attention weight" if lbl == 0 else "")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Thread Attention Weights by Class\n"
                 "(how much the model attends to each thread position)",
                 fontsize=12)
    plt.tight_layout()
    out = Path(out_dir) / "attention_by_class.png"
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.close()


def plot_attention_distribution(weights, out_dir):
    """
    Violin plot showing distribution of attention weights across all samples.
    """
    POSITIONS = ["Root tweet", "Parent tweet", "Target tweet"]
    fig, ax = plt.subplots(figsize=(8, 4))

    data = [weights[:, i] for i in range(3)]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True,
                          showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(POSITIONS, fontsize=11)
    ax.set_ylabel("Attention weight", fontsize=11)
    ax.set_title("Distribution of Thread Attention Weights\n"
                 "(across all 860 test samples)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = Path(out_dir) / "attention_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.close()



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.results_path).mkdir(parents=True, exist_ok=True)

    # Load data
    _, df_test = load_data(args.dataset_path)
    df_test    = filter_contextual_tweets(df_test)
    df_test    = ids_to_text(df_test.copy())
    labels     = df_test["stereotype"].values

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.context_path)
    model     = load_model(args.context_path, device)
    print(f"[model] Loaded from {args.context_path}")

    # Run inference with attention
    print("[inference] Extracting attention weights...")
    probs, weights = run_attention(model, tokenizer, df_test,
                                   args.batch_size, device)
    print(f"[inference] Done. weights shape: {weights.shape}")  # (860, 3)

    # Load per-sample verdicts
    per_sample = pd.read_csv(args.per_sample)

    # Print aggregate stats
    print("\n===== Aggregate Attention Weights =====")
    for i, pos in enumerate(["Root", "Parent", "Tweet"]):
        print(f"  {pos:<8}  mean={weights[:, i].mean():.3f}  "
              f"std={weights[:, i].std():.3f}")

    print("\n  --- By class ---")
    for lbl, name in [(0, "Not hate"), (1, "Hate")]:
        mask = labels == lbl
        print(f"  {name}:")
        for i, pos in enumerate(["Root", "Parent", "Tweet"]):
            print(f"    {pos:<8}  {weights[mask, i].mean():.3f}")

    # Generate plots
    plot_aggregate_attention(weights, labels, args.results_path)
    plot_attention_distribution(weights, args.results_path)

    print("\nDone. All figures saved to", args.results_path)


if __name__ == "__main__":
    main()
