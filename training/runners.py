"""
Functions to run training for baseline and hierarchical models.

Each function (run_baseline, run_hierarchical) prepares its datasets, instantiates its model,
and delegates to _run_training for the shared train → evaluate → save pipeline.
"""

import shutil
from pathlib import Path

from transformers import AutoTokenizer, EarlyStoppingCallback

from data.preprocessing import ids_to_text, tokenize_baseline, tokenize_hierarchical, augment_with_backtranslation, tokenize_cross_attention
from modeling.models import load_model, HierarchicalContextModel, CrossAttentionContextModel
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
    Shared train → evaluate → save pipeline used by all run_ functions.

    Args:
        model:       Model ready for training.
        model_type:  "baseline", "hierarchical", "augmented", or "cross_attention".
        tokenizer:   Tokenizer matching the model.
        train_ds:    Tokenized and formatted training dataset.
        val_ds:      Tokenized and formatted validation dataset.
        df_test:     Raw test DataFrame (tokenized inside evaluate_on_test).
        run_id:      Timestamp string used to identify this run in result CSVs.
        res_dir:     Output directory for checkpoints and result files.
        run_config:  Dict of hyperparameters passed to build_trainer.
        save_fn:     Handles model saving.
    """
    Path(run_config["output_dir"]).mkdir(parents=True, exist_ok=True)
    metrics_cb = run_config["callbacks"][0]

    trainer = build_trainer(model, train_ds, val_ds, tokenizer, run_config)
    trainer.train()

    print(f"[{model_type}] Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"[{model_type}] Best metric     : {trainer.state.best_metric}")

    save_train_results(
        metrics_cb.records,
        filepath=Path(res_dir) / "train_results.csv",
        extra_info={
            "run_id":  run_id,
            "model":   model_type,
            "context": model_type != "baseline",
            "max_len": CONFIG["max_len"],
        },
    )

    test_metrics = evaluate_on_test(trainer, df_test, tokenizer, model_type)
    test_metrics.update({"run_id": run_id, "split": "test", "model": model_type})
    save_test_results(test_metrics, filepath=Path(res_dir) / "test_results.csv")

    save_fn(trainer, tokenizer, res_dir)

    shutil.rmtree(Path(res_dir) / "checkpoints", ignore_errors=True)


def run_baseline(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the XLM-RoBERTa baseline on tweet text only (no thread context).

    Uses AutoModelForSequenceClassification with the bottom 6 encoder layers frozen.
    Serves as the no-context reference point against which context-aware models are compared.
    """
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = format_dataset(tokenize_baseline(prepare_dataset(df_train, ["text"]), tokenizer, CONFIG["max_len"]), "baseline")
    val_ds   = format_dataset(tokenize_baseline(prepare_dataset(df_val,   ["text"]), tokenizer, CONFIG["max_len"]), "baseline")

    model = load_model(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model, n_layers=6)

    run_config = {
        **CONFIG,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])],
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

    Encodes each thread level separately with a shared XLM-RoBERTa encoder, then
    combines representations via multi-head attention before classification.
    The bottom 6 encoder layers are frozen; the top 6 and the attention head are trained.
    level4 (hoax) is intentionally excluded.
    """
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train_h = ids_to_text(df_train.copy())
    df_val_h   = ids_to_text(df_val.copy())

    train_ds = format_dataset(
        tokenize_hierarchical(
            prepare_dataset(df_train_h, ["text", "parent_text", "root_text"]),
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
    freeze_encoder_bottom_layers(model.encoder, n_layers=4)

    run_config = {
        **CONFIG,
        "max_grad_norm": 1.0,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])],
    }

    _run_training(
        model, "hierarchical", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config,
        save_fn=lambda trainer, tokenizer, res_dir: save_hierarchical_model(
            trainer, tokenizer, Path(res_dir) / "best_model_context"
        ),
    )


def run_augmented(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the hierarchical model on a back-translation-augmented training set.

    Back-translated tweets are appended as new training rows with the same labels,
    expanding the training set (Beddiar et al., 2021). Architecture and evaluation
    are identical to run_hierarchical — only the training data is larger.
    level4 (hoax) is intentionally excluded.
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
        "augmented",
    )
    val_ds = format_dataset(
        tokenize_hierarchical(
            prepare_dataset(df_val_h, ["text", "parent_text", "root_text"]),
            tokenizer, CONFIG["max_len"],
        ),
        "augmented",
    )

    model = HierarchicalContextModel(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model.encoder, n_layers=4)

    run_config = {
        **CONFIG,
        "max_grad_norm": 1.0,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])],
    }

    _run_training(
        model, "augmented", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config,
        save_fn=lambda trainer, tokenizer, res_dir: save_hierarchical_model(
            trainer, tokenizer, Path(res_dir) / "best_model_bt"
        ),
    )


def run_cross_attention(df_train, df_val, df_test, run_id, res_dir, class_weights=None):
    """
    Fine-tune the token-level cross-attention model on tweet + parent + root text.

    Tweet tokens attend to concatenated root + parent context tokens via stacked
    cross-attention layers. The bottom 6 encoder layers are frozen.
    level4 (hoax) is intentionally excluded.
    """
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train_ca = ids_to_text(df_train.copy())
    df_val_ca   = ids_to_text(df_val.copy())

    train_ds = format_dataset(
        tokenize_cross_attention(
            prepare_dataset(df_train_ca, ["text", "parent_text", "root_text"]),
            tokenizer, CONFIG["max_len_tweet"], CONFIG["max_len_context"],
        ),
        "cross_attention",
    )
    val_ds = format_dataset(
        tokenize_cross_attention(
            prepare_dataset(df_val_ca, ["text", "parent_text", "root_text"]),
            tokenizer, CONFIG["max_len_tweet"], CONFIG["max_len_context"],
        ),
        "cross_attention",
    )

    model = CrossAttentionContextModel(CONFIG["model_name"])
    freeze_encoder_bottom_layers(model.encoder, n_layers=4)

    run_config = {
        **CONFIG,
        "max_grad_norm": 1.0,
        "metrics_fn":    compute_metrics,
        "class_weights": class_weights,
        "output_dir":    str(Path(res_dir) / "checkpoints"),
        "callbacks":     [EpochMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])],
    }

    _run_training(
        model, "cross_attention", tokenizer,
        train_ds, val_ds, df_test,
        run_id, res_dir, run_config,
        save_fn=lambda trainer, tokenizer, res_dir: save_hierarchical_model(
            trainer, tokenizer, Path(res_dir) / "best_model_cross_attention"
        ),
    )