"""
Inference and analysis script.
 
Evaluates baseline and context-aware models on the test set and analyses
whether context-awareness helps more for implicit vs explicit stereotypes.
 
python inference.py --dataset_path  "data/spanish_subset/" --baseline_path "results/best_model_baseline" --context_path  "results/best_model_context" --results_path  "results/"
"""
 
import argparse
from pathlib import Path
 
import torch
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
 
from data.loader import load_data
from data.preprocessing import ids_to_text
from modeling.models import HierarchicalContextModel
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
    return p.parse_args()
 
 
# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
 
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def batch_texts(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]
 
 
@torch.no_grad()
def predict_baseline(model, tokenizer, texts, batch_size, device):
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
 
 
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
 
def compute_metrics(labels, preds, prefix=""):
    return {
        f"{prefix}accuracy":    accuracy_score(labels, preds),
        f"{prefix}f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        f"{prefix}f1_binary":   f1_score(labels, preds, average="binary",   zero_division=0),
        f"{prefix}f1_class0":   f1_score(labels, preds, average=None,       zero_division=0)[0],
        f"{prefix}f1_class1":   f1_score(labels, preds, average=None,       zero_division=0)[1],
        f"{prefix}precision":   precision_score(labels, preds, average="macro", zero_division=0),
        f"{prefix}recall":      recall_score(labels, preds, average="macro",    zero_division=0),
    }
 
 
def metrics_by_group(labels, preds, groups, group_col="implicit"):
    """Return a DataFrame of metrics broken down by a binary group column."""
    rows = []
    for group_val in sorted(groups.unique()):
        mask = groups == group_val
        m = compute_metrics(labels[mask], preds[mask])
        m[group_col] = group_val
        m["n_samples"] = mask.sum()
        rows.append(m)
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
 
def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
 
 
def print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args   = parse_args()
    device = get_device()
    print(f"[device] Using {device}")
 
    # ---- Load test data --------------------------------------------------
    _, df_test = load_data(args.dataset_path)
 
    if "implicit" not in df_test.columns:
        raise ValueError(
            "Test set must have an 'implicit' column for implicit/explicit analysis."
        )
 
    labels   = df_test["stereotype"].values
    implicit = df_test["implicit"]
 
    # ---- Baseline --------------------------------------------------------
    print_section("Baseline model")
 
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_path)
    baseline_model     = AutoModelForSequenceClassification.from_pretrained(
        args.baseline_path
    ).to(device)
 
    baseline_preds, baseline_probs = predict_baseline(
        baseline_model,
        baseline_tokenizer,
        df_test["text"].tolist(),
        args.batch_size,
        device,
    )
 
    baseline_overall = compute_metrics(labels, baseline_preds, prefix="baseline_")
    print_metrics(baseline_overall)
    print("\nClassification report:")
    print(classification_report(labels, baseline_preds, zero_division=0))
 
    baseline_by_implicit = metrics_by_group(labels, baseline_preds, implicit)
    baseline_by_implicit.insert(0, "model", "baseline")
    print("\nMetrics by implicit/explicit:")
    print(baseline_by_implicit.to_string(index=False))
 
    # ---- Context-aware (hierarchical) ------------------------------------
    print_section("Context-aware model")
 
    context_tokenizer = AutoTokenizer.from_pretrained(args.context_path)
    context_model     = HierarchicalContextModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(args.context_path) / "model.safetensors"), device=str(device))
    context_model.load_state_dict(weights)
 
    df_test_ctx = ids_to_text(df_test.copy())
 
    context_preds, context_probs = predict_hierarchical(
        context_model,
        context_tokenizer,
        df_test_ctx,
        args.batch_size,
        device,
    )
 
    context_overall = compute_metrics(labels, context_preds, prefix="context_")
    print_metrics(context_overall)
    print("\nClassification report:")
    print(classification_report(labels, context_preds, zero_division=0))
 
    context_by_implicit = metrics_by_group(labels, context_preds, implicit)
    context_by_implicit.insert(0, "model", "context")
    print("\nMetrics by implicit/explicit:")
    print(context_by_implicit.to_string(index=False))
 
    # ---- Implicit vs explicit analysis -----------------------------------
    print_section("Context-awareness benefit: implicit vs explicit")
 
    comparison = pd.merge(
        baseline_by_implicit,
        context_by_implicit,
        on="implicit",
        suffixes=("_baseline", "_context"),
    )
    comparison["f1_macro_delta"] = (
        comparison["f1_macro_context"] - comparison["f1_macro_baseline"]
    )
    comparison["f1_binary_delta"] = (
        comparison["f1_binary_context"] - comparison["f1_binary_baseline"]
    )
 
    print(comparison[["implicit", "f1_macro_baseline", "f1_macro_context",
                       "f1_macro_delta", "f1_binary_delta"]].to_string(index=False))
 
    implicit_delta  = comparison.loc[comparison["implicit"] == 1, "f1_macro_delta"].values[0]
    explicit_delta  = comparison.loc[comparison["implicit"] == 0, "f1_macro_delta"].values[0]
 
    print(f"\n  f1_macro gain on implicit tweets:  {implicit_delta:+.4f}")
    print(f"  f1_macro gain on explicit tweets:  {explicit_delta:+.4f}")
 
    if implicit_delta > explicit_delta:
        print("\n  → Context helps MORE for implicit stereotypes (as hypothesised).")
    elif explicit_delta > implicit_delta:
        print("\n  → Context helps MORE for explicit stereotypes (unexpected).")
    else:
        print("\n  → Context helps equally for both.")
 
    # ---- Per-sample results ----------------------------------------------
    df_results = df_test[["text", "stereotype", "implicit"]].copy()
    df_results["baseline_pred"]  = baseline_preds
    df_results["baseline_prob1"] = baseline_probs[:, 1]
    df_results["context_pred"]   = context_preds
    df_results["context_prob1"]  = context_probs[:, 1]
    df_results["baseline_correct"] = (baseline_preds == labels).astype(int)
    df_results["context_correct"]  = (context_preds  == labels).astype(int)
    # Cases where context model helps vs hurts vs ties
    df_results["verdict"] = "tie"
    df_results.loc[
        (df_results["context_correct"] == 1) & (df_results["baseline_correct"] == 0),
        "verdict"
    ] = "context_wins"
    df_results.loc[
        (df_results["context_correct"] == 0) & (df_results["baseline_correct"] == 1),
        "verdict"
    ] = "baseline_wins"
 
    # ---- Save results ----------------------------------------------------
    res_dir = Path(args.results_path)
    res_dir.mkdir(parents=True, exist_ok=True)
 
    per_sample_path = res_dir / "inference_per_sample.csv"
    df_results.to_csv(per_sample_path, index=False)
    print(f"\n[saved] Per-sample results → {per_sample_path}")
 
    summary_path = res_dir / "inference_summary.csv"
    summary = pd.concat([baseline_by_implicit, context_by_implicit], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"[saved] Summary by implicit/explicit → {summary_path}")
 
    verdict_counts = df_results.groupby(["implicit", "verdict"]).size().unstack(fill_value=0)
    print("\nVerdict breakdown (where models disagree):")
    print(verdict_counts.to_string())
 
 
if __name__ == "__main__":
    main()