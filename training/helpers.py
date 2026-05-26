"""
Shared utilities used across training runners and inference.
"""

import os
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer, TrainerCallback

from data.preprocessing import tokenize_baseline, tokenize_hierarchical, tokenize_cross_attention
from config import CONFIG


class EpochMetricsCallback(TrainerCallback):
    """
    Accumulates training and validation metrics into a single row per epoch.
    """

    def __init__(self):
        self.records  = []   # one dict per completed epoch
        self._pending = {}   # accumulates logs mid-epoch

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # total_flos is a cumulative counter — not useful per-epoch
        self._pending.update({k: v for k, v in logs.items() if k != "total_flos"})

    def on_epoch_end(self, args, state, control, **kwargs):
        row = {"epoch": state.epoch, **self._pending}
        self.records.append(row)
        self._pending = {}


def freeze_encoder_bottom_layers(model, n_layers=6):
    # Baseline: AutoModelForSequenceClassification has .roberta
    # Hierarchical/CrossAttention: model.encoder is passed directly,
    # which already has .embeddings and .encoder.layer at top level
    if hasattr(model, "roberta"):
        base = model.roberta
    else:
        base = model  # already the raw encoder

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


def prepare_dataset(df, columns, label_col="stereotype"):
    """
    Subset a DataFrame to the relevant columns and convert to a HuggingFace Dataset.

    Args:
        df: DataFrame.
        columns: Feature columns to keep.
        label_col: Name of the target column.

    Returns:
        HuggingFace Dataset with feature columns + "labels".
    """
    df = df[columns + [label_col]].reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    # rename label column to "labels" to match the name the HuggingFace Trainer expects when computing loss
    return dataset.rename_column(label_col, "labels")


def format_dataset(dataset, model_type="baseline"):
    """
    Set the dataset output format to torch tensors, keeping only model-relevant columns.

    Args:
        dataset:    Tokenized HuggingFace Dataset.
        model_type: "baseline", "hierarchical", "augmented", or "cross_attention".

    Returns:
        The same dataset with torch format applied in-place.
    """
    columns = {
        "baseline": ["input_ids", "attention_mask", "labels"],
        "hierarchical": [
            "root_input_ids",   "root_attention_mask",
            "parent_input_ids", "parent_attention_mask",
            "tweet_input_ids",  "tweet_attention_mask",
            "labels",
        ],
        "cross_attention": [
            "context_input_ids", "context_attention_mask",
            "tweet_input_ids",   "tweet_attention_mask",
            "labels",
        ],
    }
    # augmented uses the same architecture and columns as hierarchical
    key = "hierarchical" if model_type == "augmented" else model_type
    dataset.set_format(type="torch", columns=columns[key])
    return dataset


def save_train_results(records, filepath, extra_info=None):
    """
    Append per-epoch train + val metric records to a CSV file.

    Args:
        records: List of dicts from EpochMetricsCallback.
        filepath: Destination CSV path.
        extra_info: Optional dict of scalar values (e.g. run_id, model name) added as constant columns to every row.
    """
    df = pd.DataFrame(records)
    if extra_info is not None:
        for key, value in extra_info.items():
            df[key] = value
    file_exists = os.path.isfile(filepath)
    # Header written only if file doesn't exist yet
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def save_test_results(metrics, filepath):
    """
    Append a single row of test metrics to a CSV file.

    Args:
        metrics: Dict of metric name → value, typically from trainer.evaluate().
        filepath: Destination CSV path.
    """
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def save_hierarchical_model(trainer, tokenizer, save_dir):
    """
    Serialize the hierarchical model weights, tokenizer, and architecture config.

    HierarchicalContextModel is a custom nn.Module and can't be saved with
    trainer.save_model() (which expects a HuggingFace PreTrainedModel).

    Args:
        trainer: Trained Trainer instance holding the model.
        tokenizer: Tokenizer to save alongside the weights.
        save_dir: Directory to write model.safetensors, tokenizer files, and hierarchical_config.json.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainer.model.state_dict(), save_dir / "model.safetensors")
    tokenizer.save_pretrained(save_dir)
    hconfig = {"model_name": CONFIG["model_name"], "num_labels": 2, "dropout": CONFIG["dropout"]}
    with open(save_dir / "hierarchical_config.json", "w") as f:
        json.dump(hconfig, f, indent=2)



def evaluate_on_test(trainer, df_test, tokenizer, model_type):
    """
    Tokenize the test set for the given model type and run Trainer.evaluate().

    Args:
        trainer: Trained Trainer instance.
        df_test: Raw test DataFrame.
        tokenizer: Tokenizer matching the trained model.
        model_type: One of "baseline" or "hierarchical" (controls tokenization and dataset formatting).

    Returns:
        Dict of evaluation metrics from Trainer.evaluate().
    """
    if model_type == "baseline":
        test_ds = prepare_dataset(df_test, ["text"])
        test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, "baseline")

    elif model_type in ("hierarchical", "augmented"):
        test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
        test_ds = tokenize_hierarchical(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, model_type)

    elif model_type == "cross_attention":
        test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
        test_ds = tokenize_cross_attention(test_ds, tokenizer, CONFIG["max_len_tweet"], CONFIG["max_len_context"])
        test_ds = format_dataset(test_ds, "cross_attention")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return trainer.evaluate(test_ds)