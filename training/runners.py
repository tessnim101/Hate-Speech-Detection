"""
Functions to run training for baseline and hierarchical models.

Each function (run_baseline, run_hierarchical) prepares its datasets, instantiates its model, 
and delegates to _run_training for the shared train → evaluate → save pipeline.
"""

import shutil
from pathlib import Path

from transformers import AutoTokenizer, EarlyStoppingCallback

from data.preprocessing import ids_to_text, tokenize_baseline, tokenize_hierarchical, augment_with_backtranslation
from modeling.models import load_model, HierarchicalContextModel
from training.trainer_utils import build_trainer
from training.trainer_utils import compute_metrics
from config import CONFIG
from training.helpers import (
    EpochMetricsCallback,
    freeze_encoder_bottom_layers,
    prepare_dataset,
    format_dataset,
    save_train_results,
    save_test_results,
    save_hierarchical_model,
    evaluate_on_test,
)


def _run_training(
    model, model_type, tokenizer,
    train_ds, val_ds, df_test,
    run_id, res_dir,
    run_config,
    save_fn,
):
    """
    Shared train → evaluate → save pipeline used by both run_ functions.

    Args:
        model: Model ready for training.
        model_type: "baseline" or "hierarchical".
        tokenizer: Tokenizer matching the model.
        train_ds: Tokenized and formatted training dataset.
        val_ds: Tokenized and formatted validation dataset.
        df_test: Raw test DataFrame (tokenized inside evaluate_on_test).
        run_id: Timestamp string used to identify this run in result CSVs.
        res_dir: Output directory for checkpoints and result files.
        run_config: Dict of hyperparameters passed to build_trainer.
        save_fn: Handles model saving.
    """
    metrics_cb = run_config["callbacks"][0] 

    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    print(f"[{model_type}] Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"[{model_type}] Best metric     : {trainer.state.best_metric}")

    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info={
            "run_id":   run_id,
            "model":    model_type,
            "context":  model_type == "hierarchical",
            "max_len":  CONFIG["max_len"],
        },
    )

    # Evaluate on test set and append to test_results.csv
    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, model_type)
    test_metrics.update({"run_id": run_id, "split": "test", "model": model_type})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")

    save_fn(trainer, tokenizer, res_dir)

    # Remove intermediate checkpoints, only the best model weights are kept
    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)


def run_baseline(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the XLM-RoBERTa baseline on tweet text only (no thread context).

    Uses AutoModelForSequenceClassification with the top 6 encoder layers unfrozen. 
    Serves as the no-context reference point against which the hierarchical model is compared.

    Args:
        df_train: Training DataFrame (raw, before tokenization).
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        run_id: Timestamp string used to identify this run in result CSVs.
        res_dir: Output directory for checkpoints and result files.
        class_weights: Optional weight tensor containing class weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = format_dataset(tokenize_baseline(prepare_dataset(df_train, ["text"]), tokenizer, CONFIG["max_len"]), "baseline")
    val_ds   = format_dataset(tokenize_baseline(prepare_dataset(df_val,   ["text"]), tokenizer, CONFIG["max_len"]), "baseline")

    model = load_model(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate": 1.5e-5,
        "weight_decay":  0.025,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=4)],
    }

    def save_fn(trainer, tokenizer, res_dir):
        trainer.save_model(Path(res_dir) / "best_model_baseline")
        tokenizer.save_pretrained(Path(res_dir) / "best_model_baseline")

    _run_training(
        model, "baseline", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config, save_fn,
    )


def run_hierarchical(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the hierarchical context-aware model on tweet + parent + root text.

    Encodes each thread level separately with a shared XLM-RoBERTa encoder,
    then combines representations via multi-head attention before classification.
    The bottom 6 encoder layers are frozen, the top 6 and the attention head are trained.

    Args:
        df_train: Training DataFrame (raw, before tokenization).
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        run_id: Timestamp string used to identify this run in result CSVs.
        res_dir: Output directory for checkpoints and result files.
        class_weights: Optional weight tensor containing class weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Resolve parent and root IDs to their actual tweet text before tokenizing
    df_train_h = ids_to_text(df_train.copy())
    df_val_h   = ids_to_text(df_val.copy())

    train_ds = format_dataset(tokenize_hierarchical(prepare_dataset(df_train_h, ["text", "parent_text", "root_text"]), tokenizer, CONFIG["max_len"]), "hierarchical")
    val_ds   = format_dataset(tokenize_hierarchical(prepare_dataset(df_val_h,   ["text", "parent_text", "root_text"]), tokenizer, CONFIG["max_len"]), "hierarchical")

    model = HierarchicalContextModel(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model.encoder, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate": 2e-5,
        "weight_decay":  0.0025,
        "warmup_ratio":  0.1,
        "max_grad_norm": 1.0,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=4)],
    }

    _run_training(
        model, "hierarchical", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config,
        save_fn=lambda trainer, tokenizer, res_dir: save_hierarchical_model(trainer, tokenizer, Path(res_dir) / "best_model_context"),
    )


def run_augmented(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the hierarchical model on a back-translation-augmented training set.

    Follows the data-augmentation approach from Beddiar et al. (2021): back-translated
    tweets are appended as new training rows with the same labels, expanding the training
    set. The model architecture and evaluation are identical to run_hierarchical — only
    the training data is larger.

    Args:
        df_train: Training DataFrame (must contain 'backtranslated_text').
        df_val: Validation DataFrame (not augmented).
        df_test: Test DataFrame.
        run_id: Timestamp string used to identify this run in result CSVs.
        res_dir: Output directory for checkpoints and result files.
        class_weights: Optional weight tensor containing class weights.
    """
    if df_train["backtranslated_text"].eq("").all():
        print("[augmented] 'backtranslated_text' column is empty — "
              "run translate.py first. Skipping run_augmented.")
        return

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train_aug = augment_with_backtranslation(ids_to_text(df_train.copy()))
    df_val_h     = ids_to_text(df_val.copy())

    train_ds = format_dataset(
        tokenize_hierarchical(
            prepare_dataset(df_train_aug, ["text", "parent_text", "root_text"]),
            tokenizer, CONFIG["max_len"],
        ),
        "hierarchical",
    )
    val_ds = format_dataset(
        tokenize_hierarchical(
            prepare_dataset(df_val_h, ["text", "parent_text", "root_text"]),
            tokenizer, CONFIG["max_len"],
        ),
        "hierarchical",
    )

    model = HierarchicalContextModel(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model.encoder, n_layers=6)

    run_config = {
        **CONFIG,
        "learning_rate": 2e-5,
        "weight_decay":  0.0025,
        "warmup_ratio":  0.1,
        "max_grad_norm": 1.0,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=4)],
    }

    _run_training(
        model, "augmented", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config,
        save_fn=lambda trainer, tokenizer, res_dir: save_hierarchical_model(trainer, tokenizer, Path(res_dir) / "best_model_bt"),
    )