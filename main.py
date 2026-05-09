"""
Run training

Controlled ablation: same early-fusion architecture, three context levels.

python main.py \
    --dataset_path        "data/spanish_subset/" \
    --results_path        "results/" \
    --imbalance_strategy  "oversample"
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from safetensors.torch import save_file
import shutil

from data.loader import load_data
from data.preprocessing import (
    filter_contextual_tweets,
    split_train_validation,
    ids_to_text,
    tokenize_baseline,
    tokenize_hierarchical,
    tokenize_early_fusion,
)
from modeling.models import load_model, HierarchicalContextModel
from training.trainer_utils import build_trainer
from training.metrics import compute_metrics
from config import CONFIG

from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

"""
class UnfreezeEncoderCallback(TrainerCallback):
    #Freezes the shared encoder for epoch 1, unfreezes from epoch 2 onward.

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("[callback] Encoder frozen for epoch 1.")

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if state.epoch >= 1 and hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = True
            print(f"[callback] Encoder unfrozen at epoch {state.epoch}.")
"""

class EpochMetricsCallback(TrainerCallback):
    """
    Collects a single merged row per epoch containing both training
    and validation metrics, so they can be saved together.
    """
    def __init__(self):
        self.records  = []
        self._pending = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._pending.update({k: v for k, v in logs.items() if k != "total_flos"})

    def on_epoch_end(self, args, state, control, **kwargs):
        row = {"epoch": state.epoch, **self._pending}
        self.records.append(row)
        self._pending = {}


def freeze_encoder_bottom_layers(model, n_layers=6):
    # Handle both AutoModelForSequenceClassification (.roberta) 
    # and raw XLMRobertaModel (no wrapper)
    base = getattr(model, "roberta", model)

    for param in base.embeddings.parameters():
        param.requires_grad = False

    for i, layer in enumerate(base.encoder.layer):
        if i < n_layers:
            for param in layer.parameters():
                param.requires_grad = False

    total  = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[freeze] Froze bottom {n_layers} layers — "
          f"{frozen:,} / {total:,} params frozen "
          f"({100 * frozen / total:.1f}%)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",  required=True,  help="Dataset path")
    p.add_argument("--results_path",  required=True,  help="Output directory")
    p.add_argument(
        "--imbalance_strategy",
        choices=["class_weights", "oversample", "none"],
        default="class_weights",
    )
    args = p.parse_args()
    return args.dataset_path, args.results_path, args.imbalance_strategy


# ---------------------------------------------------------------------------
# Class-imbalance utilities
# ---------------------------------------------------------------------------

def compute_class_weights(df, label_col="stereotype"):
    counts  = df[label_col].value_counts().sort_index()
    freqs   = counts.values.astype(float)
    weights = 1.0 / freqs
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float)


def oversample_minority_classes(df, label_col="stereotype", random_state=42):
    max_count = df[label_col].value_counts().max()
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            group = group.sample(max_count, replace=True, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# Maps a context_level to a human-readable model name stored in result CSVs.
CONTEXT_LEVEL_NAMES = {
    "none":   "fusion_no_context",
    "parent": "fusion_parent_only",
    "full":   "fusion_full_context",
}

def build_prompt(df, context_level="full"):
    """
    Constructs the input prompt depending on how much context to include.

    none   → tweet only  (equivalent to baseline, same architecture)
    parent → tweet + direct parent reply
    full   → tweet + parent + root thread
    """
    base = "Classify this tweet: " + df["text"]
    if context_level == "none":
        return base
    if context_level == "parent":
        return base + " | Reply to: " + df["parent_text"]
    if context_level == "full":
        return (
            base
            + " | Reply to: "       + df["parent_text"]
            + " | Thread context: " + df["root_text"]
        )
    raise ValueError(f"Unknown context_level: {context_level!r}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def prepare_dataset(df, columns, label_col="stereotype"):
    df = df[columns + [label_col]].reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    return dataset.rename_column(label_col, "labels")


def format_dataset(dataset, model_type="baseline"):
    columns = {
        "baseline":     ["input_ids", "attention_mask", "labels"],
        "early_fusion": ["input_ids", "attention_mask", "labels"],
        "hierarchical": [
            "root_input_ids",   "root_attention_mask",
            "parent_input_ids", "parent_attention_mask",
            "tweet_input_ids",  "tweet_attention_mask",
            "labels",
        ],
    }
    dataset.set_format(type="torch", columns=columns[model_type])
    return dataset


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_train_results(records, filepath, extra_info=None):
    """Append per-epoch train+val metric records to a CSV — never overwrites."""
    df = pd.DataFrame(records)
    if extra_info is not None:
        for key, value in extra_info.items():
            df[key] = value
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def save_test_results(metrics, filepath):
    """Append test metrics to a CSV — never overwrites."""
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def save_hierarchical_model(trainer, tokenizer, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainer.model.state_dict(), save_dir / "model.safetensors")
    tokenizer.save_pretrained(save_dir)
    hconfig = {"model_name": CONFIG["model_name"], "num_labels": 2, "dropout": 0.05}
    with open(save_dir / "hierarchical_config.json", "w") as f:
        json.dump(hconfig, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_on_test(trainer, df_test, tokenizer, model_type, context_level="full"):
    if model_type == "baseline":
        test_ds = prepare_dataset(df_test, ["text"])
        test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, "baseline")

    elif model_type == "early_fusion":
        df_test = ids_to_text(df_test.copy())
        df_test["prompt"] = build_prompt(df_test, context_level)
        test_ds = prepare_dataset(df_test, ["prompt"])
        max_len = CONFIG.get("max_len_fusion", CONFIG["max_len"])
        test_ds = tokenize_early_fusion(test_ds, tokenizer, max_len)
        test_ds = format_dataset(test_ds, "early_fusion")

    elif model_type == "hierarchical":
        df_test = ids_to_text(df_test.copy())
        test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
        test_ds = tokenize_hierarchical(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, "hierarchical")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return trainer.evaluate(test_ds)


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_baseline(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    metrics_cb = EpochMetricsCallback()
    tokenizer  = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = prepare_dataset(df_train, ["text"])
    val_ds   = prepare_dataset(df_val,   ["text"])
    train_ds = tokenize_baseline(train_ds, tokenizer, CONFIG["max_len"])
    val_ds   = tokenize_baseline(val_ds,   tokenizer, CONFIG["max_len"])
    train_ds = format_dataset(train_ds, "baseline")
    val_ds   = format_dataset(val_ds,   "baseline")

    model = load_model(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate":  1.5e-5,
        "weight_decay":   0.025,
        "metrics_fn":     compute_metrics,
        "class_weights":  class_weights,
        "output_dir":     str(Path(res_dir) / "checkpoints"),
        "callbacks":      [metrics_cb, EarlyStoppingCallback(early_stopping_patience=4)],
    }
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    print(f"[baseline] Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"[baseline] Best metric     : {trainer.state.best_metric}")

    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info={"run_id": run_id, "model": "baseline", "context": False,
                    "max_len": CONFIG["max_len"]},
    )
    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, "baseline")
    test_metrics.update({"run_id": run_id, "split": "test", "model": "baseline"})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")

    trainer.save_model(Path(res_dir) / "best_model_baseline")
    tokenizer.save_pretrained(Path(res_dir) / "best_model_baseline")
    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)


def run_early_fusion(
    df_train, df_val, df_test, run_id, res_dir,
    context_level="full", class_weights=None,
):
    """
    Trains the early-fusion model for a given context_level.
    All three ablation variants share the same architecture and hyperparameters —
    the only difference is how much context is included in the prompt.

    context_level:
        "none"   — tweet only (no context)
        "parent" — tweet + direct parent reply
        "full"   — tweet + parent + root thread
    """
    model_name = CONTEXT_LEVEL_NAMES[context_level]
    max_len    = CONFIG.get("max_len_fusion", CONFIG["max_len"])

    print(f"\n[ablation] Running early-fusion variant: {model_name}  (max_len={max_len})")

    metrics_cb = EpochMetricsCallback()
    tokenizer  = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train_p = ids_to_text(df_train.copy())
    df_val_p   = ids_to_text(df_val.copy())

    df_train_p["prompt"] = build_prompt(df_train_p, context_level)
    df_val_p["prompt"]   = build_prompt(df_val_p,   context_level)

    train_ds = prepare_dataset(df_train_p, ["prompt"])
    val_ds   = prepare_dataset(df_val_p,   ["prompt"])
    train_ds = tokenize_early_fusion(train_ds, tokenizer, max_len)
    val_ds   = tokenize_early_fusion(val_ds,   tokenizer, max_len)
    train_ds = format_dataset(train_ds, "early_fusion")
    val_ds   = format_dataset(val_ds,   "early_fusion")

    model = load_model(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate":  1.5e-5,
        "weight_decay":   0.025,
        "warmup_ratio":   0.1,
        "max_grad_norm":  1.0,
        "metrics_fn":     compute_metrics,
        "class_weights":  class_weights,
        "output_dir":     str(Path(res_dir) / "checkpoints"),
        "callbacks":      [metrics_cb, EarlyStoppingCallback(early_stopping_patience=3)],
    }
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    print(f"[{model_name}] Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"[{model_name}] Best metric     : {trainer.state.best_metric}")

    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info={
            "run_id":        run_id,
            "model":         model_name,
            "context":       context_level != "none",
            "context_level": context_level,
            "max_len":       max_len,
        },
    )
    test_metrics = evaluate_on_test(
        trainer, df_test, tokenizer, "early_fusion", context_level
    )
    test_metrics.update({
        "run_id":        run_id,
        "split":         "test",
        "model":         model_name,
        "context_level": context_level,
    })
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")

    save_dir = Path(res_dir) / f"best_model_{model_name}"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)


def run_hierarchical(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    metrics_cb = EpochMetricsCallback()
    tokenizer  = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train_h = ids_to_text(df_train.copy())
    df_val_h   = ids_to_text(df_val.copy())

    train_ds = prepare_dataset(df_train_h, ["text", "parent_text", "root_text"])
    val_ds   = prepare_dataset(df_val_h,   ["text", "parent_text", "root_text"])
    train_ds = tokenize_hierarchical(train_ds, tokenizer, CONFIG["max_len"])
    val_ds   = tokenize_hierarchical(val_ds,   tokenizer, CONFIG["max_len"])
    train_ds = format_dataset(train_ds, "hierarchical")
    val_ds   = format_dataset(val_ds,   "hierarchical")

    model = HierarchicalContextModel(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model.encoder, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate":  2e-5,
        "weight_decay":   0.0025,
        "warmup_ratio":   0.1,
        "max_grad_norm":  1.0,
        "metrics_fn":     compute_metrics,
        "class_weights":  class_weights,
        "output_dir":     str(Path(res_dir) / "checkpoints"),
        "callbacks":      [metrics_cb, EarlyStoppingCallback(early_stopping_patience=4)],
    }
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    print(f"[hierarchical] Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"[hierarchical] Best metric     : {trainer.state.best_metric}")

    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info={"run_id": run_id, "model": "hierarchical", "context": True,
                    "max_len": CONFIG["max_len"]},
    )
    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, "hierarchical")
    test_metrics.update({"run_id": run_id, "split": "test", "model": "hierarchical"})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")

    save_hierarchical_model(trainer, tokenizer, Path(res_dir) / "best_model_context")
    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    DATADIR, RES_DIR, IMBALANCE_STRATEGY = parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(Path(RES_DIR) / "checkpoints", exist_ok=True)

    df_train, df_test = load_data(DATADIR)
    df_train = filter_contextual_tweets(df_train)
    df_test  = filter_contextual_tweets(df_test)
    df_train_split, df_val_split = split_train_validation(df_train)

    # ---- class-imbalance handling ----------------------------------------
    class_weights = None

    if IMBALANCE_STRATEGY == "oversample":
        print("[imbalance] Oversampling minority classes in training set.")
        df_train_split = oversample_minority_classes(df_train_split)

    elif IMBALANCE_STRATEGY == "class_weights":
        print("[imbalance] Computing inverse-frequency class weights.")
        class_weights = compute_class_weights(df_train_split)
        print(f"            weights → {class_weights.tolist()}")

    # ---- GPU setup -------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[device] No CUDA devices found — running on CPU.")
    else:
        print(f"[device] {n_gpus} GPU(s) detected: "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))

    # ---- Baseline --------------------------------------------------------
    run_baseline(
        df_train_split, df_val_split, df_test,
        run_id=RUN_ID, res_dir=RES_DIR, class_weights=class_weights,
    )

    # ---- Ablation: early fusion with increasing context ------------------
    # Same architecture and hyperparameters across all three variants.
    # Only the prompt content changes, making this a controlled comparison.
    """for level in ["none", "parent", "full"]:
        run_early_fusion(
            df_train_split, df_val_split, df_test,
            run_id=RUN_ID, res_dir=RES_DIR,
            context_level=level, class_weights=class_weights,
        )"""

    # ---- Hierarchical (late-fusion, for reference) -----------------------
    run_hierarchical(
        df_train_split, df_val_split, df_test,
        run_id=RUN_ID, res_dir=RES_DIR, class_weights=class_weights,
    )