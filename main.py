"""
Run training

python main.py --dataset_path "data/spanish_subset/" --results_path "results/" --imbalance_strategy "class_weights"

"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
import shutil

from data.loader import load_data
from data.preprocessing import (
    split_train_validation,
    ids_to_text,
    tokenize_baseline,
    tokenize_hierarchical,
)
from modeling.models import load_model, HierarchicalContextModel
from training.trainer_utils import build_trainer
from training.metrics import compute_metrics
from config import CONFIG


from transformers import TrainerCallback

class UnfreezeEncoderCallback(TrainerCallback):
    """Freezes the shared encoder for epoch 1, unfreezes from epoch 2 onward."""
    
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

class EpochMetricsCallback(TrainerCallback):
    """
    Collects a single merged row per epoch containing both training
    and validation metrics, so they can be saved together.

    The Trainer fires on_log multiple times per epoch (once per
    logging_steps for train, once at eval time for val). This callback
    accumulates those separately and merges them at epoch end.
    """
    def __init__(self):
        self.records = []
        self._pending = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Accumulate into the pending row for this epoch
        self._pending.update({k: v for k, v in logs.items() if k != "total_flos"})

    def on_epoch_end(self, args, state, control, **kwargs):
        row = {"epoch": state.epoch, **self._pending}
        self.records.append(row)
        self._pending = {}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",  required=True,  help="Dataset path")
    p.add_argument("--results_path",  required=True,  help="Output directory")
    p.add_argument(
        "--imbalance_strategy",
        choices=["class_weights", "oversample", "none"],
        default="class_weights",
        help=(
            "How to handle class imbalance. "
            "'class_weights' passes inverse-frequency loss weights to the model; "
            "'oversample' upsamples minority classes in the training set; "
            "'none' does nothing."
        ),
    )
    args = p.parse_args()
    return args.dataset_path, args.results_path, args.imbalance_strategy


# ---------------------------------------------------------------------------
# Class-imbalance utilities
# ---------------------------------------------------------------------------

def compute_class_weights(df, label_col="stereotype"):
    """Return a float tensor of inverse-frequency weights, one per class."""
    counts  = df[label_col].value_counts().sort_index()
    freqs   = counts.values.astype(float)
    weights = 1.0 / freqs
    weights = weights / weights.sum() * len(weights)   # normalise so mean == 1
    return torch.tensor(weights, dtype=torch.float)


def oversample_minority_classes(df, label_col="stereotype", random_state=42):
    """
    Upsample every class to match the majority-class count.
    Uses simple random sampling with replacement on the minority classes.
    """
    max_count = df[label_col].value_counts().max()
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            group = group.sample(max_count, replace=True, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def prepare_dataset(df, columns, label_col="stereotype"):
    df = df[columns + [label_col]].reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    return dataset.rename_column(label_col, "labels")


def format_dataset(dataset, model_type="baseline"):
    columns = {
        "baseline": ["input_ids", "attention_mask", "labels"],
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
    """Append per-epoch train+val metric records to a CSV file."""
    df = pd.DataFrame(records)
    if extra_info is not None:
        for key, value in extra_info.items():
            df[key] = value
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def save_test_results(metrics, filepath):
    """Append test metrics to a CSV file."""
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_on_test(trainer, df_test, tokenizer, model_type="baseline"):
    """Tokenize and evaluate a test split; supports both model types."""
    if model_type == "baseline":
        test_ds = prepare_dataset(df_test, ["text"])
        test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])
    elif model_type == "hierarchical":
        df_test = ids_to_text(df_test)
        test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
        test_ds = tokenize_hierarchical(test_ds, tokenizer, CONFIG["max_len"])
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    test_ds = format_dataset(test_ds, model_type)
    return trainer.evaluate(test_ds)


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_baseline(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    metrics_cb = EpochMetricsCallback()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = prepare_dataset(df_train, ["text"])
    val_ds   = prepare_dataset(df_val,   ["text"])

    train_ds = tokenize_baseline(train_ds, tokenizer, CONFIG["max_len"])
    val_ds   = tokenize_baseline(val_ds,   tokenizer, CONFIG["max_len"])

    train_ds = format_dataset(train_ds)
    val_ds   = format_dataset(val_ds)

    model = load_model(CONFIG["model_name"])

    run_config = {
        **CONFIG,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,   # None → standard CE loss
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks": [metrics_cb]
    }
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    extra_info = {
        "run_id":  run_id,
        "model":   "baseline",
        "context": False,
        "max_len": CONFIG["max_len"],
    }
    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info=extra_info,
    )

    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, model_type="baseline")
    test_metrics.update({"run_id": run_id, "split": "test", "model": "baseline"})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")
    trainer.save_model(Path(res_dir) / "best_model_baseline")
    tokenizer.save_pretrained(Path(res_dir) / "best_model_baseline")
    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)

def run_hierarchical(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    metrics_cb = EpochMetricsCallback()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train = ids_to_text(df_train)
    df_val   = ids_to_text(df_val)

    train_ds = prepare_dataset(df_train, ["text", "parent_text", "root_text"])
    val_ds   = prepare_dataset(df_val,   ["text", "parent_text", "root_text"])

    train_ds = tokenize_hierarchical(train_ds, tokenizer, CONFIG["max_len"])
    val_ds   = tokenize_hierarchical(val_ds,   tokenizer, CONFIG["max_len"])

    train_ds = format_dataset(train_ds, "hierarchical")
    val_ds   = format_dataset(val_ds,   "hierarchical")

    model = HierarchicalContextModel(CONFIG["model_name"])

    run_config = {
        **CONFIG,
        "learning_rate": 1e-5, # lower than baseline
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks": [metrics_cb, UnfreezeEncoderCallback(), EarlyStoppingCallback(early_stopping_patience=5)],
    }
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    extra_info = {
        "run_id":  run_id,
        "model":   "hierarchical",
        "context": True,
        "max_len": CONFIG["max_len"],
    }
    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info=extra_info,
    )

    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, model_type="hierarchical")
    test_metrics.update({"run_id": run_id, "split": "test", "model": "hierarchical"})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")
    trainer.save_model(Path(res_dir) / "best_model_context")
    tokenizer.save_pretrained(Path(res_dir) / "best_model_context")
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
    df_train_split, df_val_split = split_train_validation(df_train)

    # ---- class-imbalance handling ----------------------------------------
    class_weights = None

    if IMBALANCE_STRATEGY == "oversample":
        print("[imbalance] Oversampling minority classes in training set.")
        df_train_split = oversample_minority_classes(df_train_split)

    elif IMBALANCE_STRATEGY == "class_weights":
        print("[imbalance] Computing inverse-frequency class weights.")
        # Compute weights from the training split so val/test are untouched.
        class_weights = compute_class_weights(df_train_split)
        print(f"            weights → {class_weights.tolist()}")

    # ---- GPU setup -------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[device] No CUDA devices found — running on CPU.")
    else:
        print(f"[device] {n_gpus} GPU(s) detected: "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))

    # TrainingArguments picks up CUDA automatically via the Trainer.
    # For multi-GPU runs launch with:
    #   torchrun --nproc_per_node=<N_GPUS> train.py ...
    # Recommended GPU-specific CONFIG keys to set:
    #   "fp16": True                        # mixed-precision training
    #   "per_device_train_batch_size": 32   # scale up from CPU default
    #   "per_device_eval_batch_size":  64
    #   "dataloader_num_workers": 4         # parallel data loading

    # ---- Baseline --------------------------------------------------------
    """run_baseline(
        df_train_split, df_val_split, df_test,
        run_id=RUN_ID,
        res_dir=RES_DIR,
        class_weights=class_weights,
    )"""

    # ---- Hierarchical ----------------------------------------------------
    run_hierarchical(
        df_train_split, df_val_split, df_test,
        run_id=RUN_ID,
        res_dir=RES_DIR,
        class_weights=class_weights,
    )