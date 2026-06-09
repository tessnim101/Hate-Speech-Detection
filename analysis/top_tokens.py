"""
Top attended context tokens analysis for the Cross-Attention model.

Aggregates cross-attention weights across the test set and identifies
which context tokens (root, parent) the model attends to most,
split by class. Outputs ranked lists and bar charts.

python analysis/top_tokens.py \
    --dataset_path         "data/spanish_subset_collapsed/" \
    --cross_attention_path "results/best_model_cross_attention/" \
    --results_path         "figures/" \
    --top_k                30
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text, clean_df
from config import CONFIG
from utils.inference_utils import enc, load_cross_attention

nltk.download("stopwords", quiet=True)
SPANISH_STOPWORDS = set(nltk.corpus.stopwords.words("spanish"))

SKIP_TOKENS = {"<pad>", "<s>", "</s>", "|", "▁|", "▁", ""}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--results_path",         default="figures/")
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--top_k",                type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def extract_attention(model, tokenizer, df, batch_size, device):
    """
    Run inference and collect averaged context token attention weights.
    Uses layer 1 (raw attention) from the cross-attention stack.

    Returns:
        all_weights:  list of (ctx_len,) arrays — averaged across heads and tweet tokens
        all_ctx_ids:  list of context token-id lists
        all_preds:    list of predicted class indices
    """
    all_weights, all_ctx_ids, all_preds = [], [], []

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

        # out["attention_weights"] is a list of 2 tensors (B, heads, tweet_len, ctx_len)
        # use layer 1 (raw attention) — avg heads + tweet tokens → (B, ctx_len)
        avg   = out["attention_weights"][0].cpu().numpy().mean(axis=1).mean(axis=1)
        preds = out.logits.argmax(dim=-1).cpu().tolist()

        all_weights.extend(list(avg))
        all_ctx_ids.extend(ctx_enc["input_ids"].cpu().tolist())
        all_preds.extend(preds)

    return all_weights, all_ctx_ids, all_preds


def aggregate_top_tokens(all_weights, all_ctx_ids, labels, tokenizer, top_k):
    """
    Aggregate attention weights per token per class.

    Ranks by discriminative score (stereo − non_stereo attention) so tokens
    meaningful for stereotype detection rank highest.

    Returns:
        dict: {class_label -> [(token, score), ...]}
        class_counts: {class_label -> n_samples}
    """
    class_token_attn = {0: defaultdict(float), 1: defaultdict(float)}
    class_counts     = {0: 0, 1: 0}

    for weights, ctx_ids, label in zip(all_weights, all_ctx_ids, labels):
        tokens = tokenizer.convert_ids_to_tokens(ctx_ids)
        for token, weight in zip(tokens, weights):
            clean = token.replace("▁", "").strip().lower()
            if clean in SKIP_TOKENS \
                or clean in SPANISH_STOPWORDS \
                or clean.isdigit() or len(clean) < 4 \
                or not token.startswith("▁") \
                or clean in {"url", "URL"}:
                continue
            class_token_attn[label][token] += weight
        class_counts[label] += 1

    normalized = {
        lbl: {t: v / class_counts[lbl] for t, v in attn.items()}
        for lbl, attn in class_token_attn.items()
    }

    all_tokens = set(normalized[0].keys()) | set(normalized[1].keys())

    stereo_disc     = {t: normalized[1].get(t, 0) - normalized[0].get(t, 0) for t in all_tokens}
    non_stereo_disc = {t: normalized[0].get(t, 0) - normalized[1].get(t, 0) for t in all_tokens}

    return {
        1: sorted(
            [(t, normalized[1].get(t, 0)) for t in all_tokens if stereo_disc[t] > 0],
            key=lambda x: stereo_disc[x[0]], reverse=True,
        )[:top_k],
        0: sorted(
            [(t, normalized[0].get(t, 0)) for t in all_tokens if non_stereo_disc[t] > 0],
            key=lambda x: non_stereo_disc[x[0]], reverse=True,
        )[:top_k],
    }, class_counts


def plot_top_tokens(top_tokens, class_counts, top_k, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_k * 0.4)))
    fig.suptitle(
        "Cross-Attention — Most Attended Context Tokens by Class",
        fontsize=13, fontweight="bold",
    )

    for ax, (lbl, name, color) in zip(axes, [
        (0, f"Not stereotype (class 0) — {class_counts[0]} samples", "#4C72B0"),
        (1, f"Stereotype (class 1) — {class_counts[1]} samples",     "#DD8452"),
    ]):
        tokens, scores = zip(*top_tokens[lbl])
        scores = np.array(scores)
        y_pos  = np.arange(len(tokens))

        bars = ax.barh(y_pos, scores, color=color, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("Mean attention weight", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", fontsize=7.5)

    plt.tight_layout()
    out = Path(out_dir) / "cross_attn_top_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


def save_csv(top_tokens, out_dir):
    rows = [
        {"class": lbl, "class_name": name, "rank": rank, "token": token, "mean_attn": round(score, 6)}
        for lbl, name in [(0, "not_stereotype"), (1, "stereotype")]
        for rank, (token, score) in enumerate(top_tokens[lbl], 1)
    ]
    out = Path(out_dir) / "cross_attn_top_tokens.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")


def print_summary(top_tokens, top_k):
    for lbl, name in [(0, "Not stereotype"), (1, "Stereotype")]:
        print(f"\n  {name} — top {top_k} tokens:")
        for rank, (token, score) in enumerate(top_tokens[lbl], 1):
            print(f"    {rank:>2}. {token:<20} {score:.5f}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out_dir = Path(args.results_path)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    df_train, df_test = load_data(args.dataset_path)
    df_train = clean_df(df_train)
    df_test  = clean_df(df_test)
    combined = pd.concat([df_train, df_test], ignore_index=True)
    df_train = ids_to_text(df_train, lookup_df=combined)
    df_test  = ids_to_text(df_test,  lookup_df=combined)
    df_test  = filter_contextual_tweets(df_test)
    labels   = df_test["stereotype"].values
    print(f"[data] Test set: {len(df_test)} samples")

    print("[loading] Cross-Attention model...")
    model, tokenizer = load_cross_attention(args.cross_attention_path, device)

    print("[inference] Extracting attention weights...")
    all_weights, all_ctx_ids, all_preds = extract_attention(
        model, tokenizer, df_test, args.batch_size, device
    )
    print(f"[inference] Done — {len(all_weights)} samples")

    print(f"\n[analysis] Computing top {args.top_k} tokens per class...")
    top_tokens, class_counts = aggregate_top_tokens(
        all_weights, all_ctx_ids, labels, tokenizer, args.top_k
    )

    print("\n===== Top Attended Context Tokens =====")
    print_summary(top_tokens, args.top_k)

    plot_top_tokens(top_tokens, class_counts, args.top_k, out_dir)
    save_csv(top_tokens, out_dir)

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()