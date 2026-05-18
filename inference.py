"""
Inference and analysis script.

Evaluates baseline and context-aware models on the test set and compares
their predictions sample-by-sample. Designed to be run once per training
seed so that results accumulate in inference_summary.csv for multi-run
statistical analysis (see visualize_results.py).

python inference.py --dataset_path  "data/spanish_subset/" \
                    --baseline_path "results/best_model_baseline" \
                    --context_path  "results/best_model_context" \
                    --results_path  "results/" \
                    --run_id        "0"
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from data.loader import load_data
from data.preprocessing import ids_to_text, filter_contextual_tweets
from modeling.models import HierarchicalContextModel , CrossAttentionHoaxModel
from config import CONFIG


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--baseline_path", required=True, help="Path to best_model_baseline/")
    p.add_argument("--context_path",  required=True, help="Path to best_model_context/")
    p.add_argument("--results_path",  required=True, help="Directory to write CSVs")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--run_id",        type=str, default="0",
                   help="Seed/run identifier — used as a key when aggregating across runs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_device():
    """
    Return a CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_texts(texts, batch_size):
    """
    Yield successive slices of texts of length batch_size.
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


@torch.no_grad()
def predict_baseline(model, tokenizer, texts, batch_size, device):
    """
    Run batched inference with the baseline AutoModelForSequenceClassification.

    Args:
        model: Loaded baseline model in eval mode.
        tokenizer: Matching tokenizer.
        texts: List of raw tweet strings.
        batch_size: Number of samples per forward pass.
        device: torch.device to run inference on.

    Returns:
        preds: Integer array of shape (N,) with predicted class indices.
        probs: Float array of shape (N, num_classes) with softmax probabilities.
    """
    model.eval()
    all_preds, all_probs = [], []

    for batch in batch_texts(texts, batch_size):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CONFIG["max_len"],
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def predict_hierarchical(model, tokenizer, df, batch_size, device):
    """
    Run batched inference with the hierarchical context-aware model.

    Each batch tokenizes tweet, parent, and root texts independently and passes
    them as separate inputs, matching the forward() signature of HierarchicalContextModel.

    Args:
        model: Loaded HierarchicalContextModel in eval mode.
        tokenizer: Matching tokenizer.
        df: DataFrame with "text", "parent_text", and "root_text" columns.
        batch_size: Number of samples per forward pass.
        device: torch.device to run inference on.

    Returns:
        preds: Integer array of shape (N,) with predicted class indices.
        probs: Float array of shape (N, num_classes) with softmax probabilities.
    """
    model.eval()
    all_preds, all_probs = [], []

    texts        = df["text"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_batch = texts[i        : i + batch_size]
        p_batch = parent_texts[i : i + batch_size]
        r_batch = root_texts[i   : i + batch_size]

        def enc(batch):
            """
            Tokenize a single text batch and move to device.
            """
            return tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=CONFIG["max_len"],
                return_tensors="pt",
            ).to(device)

        tweet_enc  = enc(t_batch)
        parent_enc = enc(p_batch)
        root_enc   = enc(r_batch)

        out = model(
            tweet_input_ids=tweet_enc["input_ids"],
            tweet_attention_mask=tweet_enc["attention_mask"],
            parent_input_ids=parent_enc["input_ids"],
            parent_attention_mask=parent_enc["attention_mask"],
            root_input_ids=root_enc["input_ids"],
            root_attention_mask=root_enc["attention_mask"],
        )

        logits = out["logits"]
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)

@torch.no_grad()
def predict_cross_attention(model, tokenizer, df, batch_size, device):
    """
    Run batched inference with the cross-attention model.

    Each batch tokenizes the context (hoax + root + parent) and tweet
    separately, matching the forward() signature of CrossAttentionHoaxModel.

    Args:
        model: Loaded CrossAttentionHoaxModel in eval mode.
        tokenizer: Matching tokenizer.
        df: DataFrame with "text", "hoax", "parent_text", and "root_text" columns.
        batch_size: Number of samples per forward pass.
        device: torch.device to run inference on.

    Returns:
        preds: Integer array of shape (N,) with predicted class indices.
        probs: Float array of shape (N, num_classes) with softmax probabilities.
    """
    model.eval()
    all_preds, all_probs = [], []

    texts        = df["text"].tolist()
    hoax_texts   = df["hoax"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    def enc(batch, max_length):
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

    for i in range(0, len(texts), batch_size):
        h_batch = hoax_texts[i   : i + batch_size]
        r_batch = root_texts[i   : i + batch_size]
        p_batch = parent_texts[i : i + batch_size]
        t_batch = texts[i        : i + batch_size]

        context = []
        for h, r, p in zip(h_batch, r_batch, p_batch):
            parts = []
            if h and str(h).strip(): parts.append(f"Hoax: {h}")
            if r and str(r).strip(): parts.append(f"Thread: {r}")
            if p and str(p).strip(): parts.append(f"Reply to: {p}")
            context.append(" | ".join(parts))

        context_enc = enc(context, CONFIG["max_len_context"])
        tweet_enc   = enc(t_batch, CONFIG["max_len_tweet"])

        out = model(
            context_input_ids=      context_enc["input_ids"],
            context_attention_mask= context_enc["attention_mask"],
            tweet_input_ids=        tweet_enc["input_ids"],
            tweet_attention_mask=   tweet_enc["attention_mask"],
        )

        logits = out["logits"]
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds):
    """
    Compute a standard set of classification metrics.

    Args:
        labels: Ground-truth integer array.
        preds: Predicted integer array.

    Returns:
        Dict of metric name → scalar value.
    """
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title):
    """
    Print a section header to stdout for readability.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metrics(metrics: dict):
    """
    Pretty-print a metrics dict, formatting floats to 4 decimal places.
    """
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device()
    print(f"[device] Using {device}")

    # ---- Load and filter test data ---------------------------------------
    # Drop rows without parent/root context to remain consistent with training
    _, df_test = load_data(args.dataset_path)
    df_test    = filter_contextual_tweets(df_test)
    labels     = df_test["stereotype"].values

    # ---- Baseline --------------------------------------------------------
    print_section("Baseline model")

    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_path)
    baseline_model     = AutoModelForSequenceClassification.from_pretrained(
        args.baseline_path
    ).to(device)

    baseline_preds, baseline_probs = predict_baseline(
        baseline_model, baseline_tokenizer,
        df_test["text"].tolist(), args.batch_size, device,
    )

    baseline_metrics = compute_metrics(labels, baseline_preds)
    print_metrics(baseline_metrics)
    print("\nClassification report:")
    print(classification_report(labels, baseline_preds, zero_division=0))

    # ---- Context-aware (hierarchical) ------------------------------------
    print_section("Context-aware model")

    context_tokenizer = AutoTokenizer.from_pretrained(args.context_path)
    context_model     = HierarchicalContextModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(args.context_path) / "model.safetensors"), device=str(device))
    context_model.load_state_dict(weights)

    df_test_ctx = ids_to_text(df_test.copy())

    context_preds, context_probs = predict_hierarchical(
        context_model, context_tokenizer,
        df_test_ctx, args.batch_size, device,
    )

    context_metrics = compute_metrics(labels, context_preds)
    print_metrics(context_metrics)
    print("\nClassification report:")
    print(classification_report(labels, context_preds, zero_division=0))

    # ---- Delta -----------------------------------------------------------
    print_section("Context vs Baseline")
    for k in ["accuracy", "f1_macro", "f1_binary", "f1_class0", "f1_class1"]:
        delta = context_metrics[k] - baseline_metrics[k]
        print(f"  {k:<25} {delta:+.4f}")

    # ---- Per-sample results ----------------------------------------------
    # verdict column classifies each sample into one of three outcomes:
    #   context_wins — context model correct, baseline wrong
    #   baseline_wins — baseline correct, context model wrong
    #   tie — both models agree (right or wrong)
    df_results = df_test[["text", "stereotype"]].copy()
    df_results["baseline_pred"] = baseline_preds
    df_results["baseline_prob1"] = baseline_probs[:, 1]  
    df_results["context_pred"] = context_preds
    df_results["context_prob1"] = context_probs[:, 1]
    df_results["baseline_correct"] = (baseline_preds == labels).astype(int)
    df_results["context_correct"] = (context_preds  == labels).astype(int)
    df_results["verdict"] = "tie"
    df_results.loc[
        (df_results["context_correct"] == 1) & (df_results["baseline_correct"] == 0),
        "verdict"
    ] = "context_wins"
    df_results.loc[
        (df_results["context_correct"] == 0) & (df_results["baseline_correct"] == 1),
        "verdict"
    ] = "baseline_wins"

    print("\nVerdict breakdown:")
    print(df_results["verdict"].value_counts().to_string())

    # ---- Save results ----------------------------------------------------
    res_dir = Path(args.results_path)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Full per-sample predictions (useful for error analysis)
    per_sample_path = res_dir / "inference_per_sample.csv"
    df_results.to_csv(per_sample_path, index=False)
    print(f"\n[saved] Per-sample results → {per_sample_path}")

    rows = [
        {"run_id": args.run_id, "model": "baseline", **baseline_metrics},
        {"run_id": args.run_id, "model": "context",  **context_metrics},
    ]
    summary_df   = pd.DataFrame(rows)
    summary_path = res_dir / "inference_summary.csv"
    file_exists  = summary_path.exists()
    summary_df.to_csv(summary_path, mode="a", header=not file_exists, index=False)
    print(f"[saved] Summary → {summary_path}")


if __name__ == "__main__":
    main()