import os
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer, TrainerCallback

from data.preprocessing import ids_to_text, tokenize_baseline, tokenize_hierarchical, tokenize_early_fusion
from config import CONFIG

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

def evaluate_on_test(trainer, df_test, tokenizer, model_type):
    if model_type == "baseline":
        test_ds = prepare_dataset(df_test, ["text"])
        test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, "baseline")


    elif model_type == "hierarchical":
        df_test = ids_to_text(df_test.copy())
        test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
        test_ds = tokenize_hierarchical(test_ds, tokenizer, CONFIG["max_len"])
        test_ds = format_dataset(test_ds, "hierarchical")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return trainer.evaluate(test_ds)

