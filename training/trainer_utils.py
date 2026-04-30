"""
Build trainer
"""

from transformers import Trainer, TrainingArguments, DefaultDataCollator
from modeling.models import HierarchicalContextModel


def build_trainer(model, train_ds, val_ds, tokenizer, config: dict) -> Trainer:
    args = TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints"),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 16),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        logging_steps=50,
        fp16=config.get("fp16", False),
        report_to="none",
    )

    # HierarchicalContextModel needs custom column forwarding
    """if isinstance(model, HierarchicalContextModel):
        remove_cols = ["root_text", "parent_text", "text"] if "text" in (train_ds.column_names or []) else []
    else:
        remove_cols = []"""

    collator = DefaultDataCollator() if isinstance(model, HierarchicalContextModel) else None

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=config.get("metrics_fn"),
    )