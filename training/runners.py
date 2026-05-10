import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer, EarlyStoppingCallback
from safetensors.torch import save_file

from data.preprocessing import ids_to_text, tokenize_baseline, tokenize_hierarchical, tokenize_early_fusion
from modeling.models import load_model, HierarchicalContextModel
from training.trainer_utils import build_trainer
from training.metrics import compute_metrics
from config import CONFIG
from helpers import (
    EpochMetricsCallback,
    freeze_encoder_bottom_layers,
    prepare_dataset,
    format_dataset,
    save_train_results,
    save_test_results,
    save_hierarchical_model,
    evaluate_on_test
)


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