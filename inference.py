"""
Inference and analysis script.

Evaluates baseline, hierarchical, and cross-attention models on the test set
and compares their predictions sample-by-sample.

python3 inference.py \
    --dataset_path          "data/spanish_subset/" \
    --baseline_path         "results/best_model_baseline/" \
    --context_path          "results/best_model_context/" \
    --cross_attention_path  "results/best_model_cross_attention/" \
    --results_path          "results/" \
    --run_id                "0"
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import CONFIG
from data.loader import load_data
from data.preprocessing import ids_to_text, filter_contextual_tweets
from utils.inference_utils import enc, load_hierarchical, load_cross_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--baseline_path",        required=True)
    p.add_argument("--context_path",         required=True)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--results_path",         required=True)
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--run_id",               type=str, default="0")
    return p.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


@torch.no_grad()
def predict_baseline(model, tokenizer, df, batch_size, device):
    """Batched inference for the baseline model (tweet text only)."""
    model.eval()
    all_preds, all_probs = [], []

    for batch in batch_iter(df["text"].tolist(), batch_size):
        e      = enc(tokenizer, batch, CONFIG["max_len"], device)
        logits = model(**e).logits
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def predict_hierarchical(model, tokenizer, df, batch_size, device):
    """
    Batched inference for the hierarchical model.
    Encodes tweet, parent, and root separately.
    level4 (hoax) is intentionally excluded.
    """
    model.eval()
    all_preds, all_probs = [], []

    texts        = df["text"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_enc = enc(tokenizer, texts[i:i+batch_size],        CONFIG["max_len"], device)
        p_enc = enc(tokenizer, parent_texts[i:i+batch_size], CONFIG["max_len"], device)
        r_enc = enc(tokenizer, root_texts[i:i+batch_size],   CONFIG["max_len"], device)

        out = model(
            tweet_input_ids=       t_enc["input_ids"],
            tweet_attention_mask=  t_enc["attention_mask"],
            parent_input_ids=      p_enc["input_ids"],
            parent_attention_mask= p_enc["attention_mask"],
            root_input_ids=        r_enc["input_ids"],
            root_attention_mask=   r_enc["attention_mask"],
        )

        logits = out["logits"]
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def predict_cross_attention(model, tokenizer, df, batch_size, device):
    """
    Batched inference for the cross-attention model.
    Context (root + parent) and tweet are encoded separately.
    level4 (hoax) is intentionally excluded.
    """
    model.eval()
    all_preds, all_probs = [], []

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
            context_input_ids=      ctx_enc["input_ids"],
            context_attention_mask= ctx_enc["attention_mask"],
            tweet_input_ids=        twt_enc["input_ids"],
            tweet_attention_mask=   twt_enc["attention_mask"],
        )

        logits = out["logits"]
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_binary": f1_score(labels, preds, average="binary", zero_division=0),
        "f1_class0": f1_score(labels, preds, average=None,     zero_division=0)[0],
        "f1_class1": f1_score(labels, preds, average=None,     zero_division=0)[1],
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro",    zero_division=0),
        "n_samples": len(labels),
    }


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")


def main():
    args   = parse_args()
    device = get_device()
    print(f"[device] {device}")

    _, df_test = load_data(args.dataset_path)
    df_test    = filter_contextual_tweets(df_test)
    df_test    = ids_to_text(df_test.copy())
    labels     = df_test["stereotype"].values

    # baseline
    print_section("Baseline")
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_path)
    baseline_model     = AutoModelForSequenceClassification.from_pretrained(
        args.baseline_path
    ).to(device).eval()

    baseline_preds, baseline_probs = predict_baseline(
        baseline_model, baseline_tokenizer, df_test, args.batch_size, device,
    )
    baseline_metrics = compute_metrics(labels, baseline_preds)
    print_metrics(baseline_metrics)
    print(classification_report(labels, baseline_preds, zero_division=0))

    # hierarchical model
    print_section("Hierarchical")
    hier_model, hier_tokenizer = load_hierarchical(args.context_path, device)

    hier_preds, hier_probs = predict_hierarchical(
        hier_model, hier_tokenizer, df_test, args.batch_size, device,
    )
    hier_metrics = compute_metrics(labels, hier_preds)
    print_metrics(hier_metrics)
    print(classification_report(labels, hier_preds, zero_division=0))

    # cross-attention model
    print_section("Cross-Attention")
    ca_model, ca_tokenizer = load_cross_attention(args.cross_attention_path, device)

    ca_preds, ca_probs = predict_cross_attention(
        ca_model, ca_tokenizer, df_test, args.batch_size, device,
    )
    ca_metrics = compute_metrics(labels, ca_preds)
    print_metrics(ca_metrics)
    print(classification_report(labels, ca_preds, zero_division=0))
    print_section("Deltas vs Baseline")
    for k in ["accuracy", "f1_macro", "f1_class0", "f1_class1"]:
        print(f"  {'Hierarchical':<20} {k:<15} {hier_metrics[k] - baseline_metrics[k]:+.4f}")
        print(f"  {'Cross-Attention':<20} {k:<15} {ca_metrics[k]  - baseline_metrics[k]:+.4f}")

    df_results = df_test[["text", "stereotype"]].copy()

    df_results["baseline_pred"]    = baseline_preds
    df_results["baseline_prob1"]   = baseline_probs[:, 1]
    df_results["hier_pred"]        = hier_preds
    df_results["hier_prob1"]       = hier_probs[:, 1]
    df_results["ca_pred"]          = ca_preds
    df_results["ca_prob1"]         = ca_probs[:, 1]

    df_results["baseline_correct"] = (baseline_preds == labels).astype(int)
    df_results["hier_correct"]     = (hier_preds     == labels).astype(int)
    df_results["ca_correct"]       = (ca_preds       == labels).astype(int)

    df_results["verdict"] = "tie"
    df_results.loc[
        (df_results["ca_correct"] == 1) & (df_results["baseline_correct"] == 0),
        "verdict"
    ] = "context_wins"
    df_results.loc[
        (df_results["ca_correct"] == 0) & (df_results["baseline_correct"] == 1),
        "verdict"
    ] = "baseline_wins"

    print("\nVerdict breakdown (cross-attention vs baseline):")
    print(df_results["verdict"].value_counts().to_string())

    res_dir = Path(args.results_path)
    res_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = res_dir / "inference_per_sample.csv"
    df_results.to_csv(per_sample_path, index=False)
    print(f"\n[saved] Per-sample results → {per_sample_path}")

    rows = [
        {"run_id": args.run_id, "model": "baseline",        **baseline_metrics},
        {"run_id": args.run_id, "model": "context",         **hier_metrics},
        {"run_id": args.run_id, "model": "cross_attention", **ca_metrics},
    ]
    summary_df   = pd.DataFrame(rows)
    summary_path = res_dir / "inference_summary.csv"
    file_exists  = summary_path.exists()
    summary_df.to_csv(summary_path, mode="a", header=not file_exists, index=False)
    print(f"[saved] Summary → {summary_path}")


if __name__ == "__main__":
    main()