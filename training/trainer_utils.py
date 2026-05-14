"""
Trainer construction and custom loss for stereotype classification.

Selects the appropriate Trainer subclass depending on whether class weights 
are supplied and whether the model is hierarchical. All HuggingFace TrainingArguments
are centralised here so hyperparameters don't need to be scattered across callers.
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from sklearn.metrics import f1_score, accuracy_score
from modeling.models import HierarchicalContextModel


class WeightedLossTrainer(Trainer):
    """
    Trainer subclass that applies class weights to the cross-entropy loss.

    Args:
        class_weights: 1-D float tensor of length num_labels.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # HierarchicalContextModel returns a dict; standard models return a ModelOutput
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )(logits, labels)

        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Compute classification metrics from raw logits and ground-truth labels.

    Args:
        eval_pred: (logits, labels) tuple provided by the Trainer.

    Returns:
        Dict with accuracy, macro F1, binary F1, and per-class F1 scores.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro"),
        "f1_binary": f1_score(labels, preds),
        "f1_class0": f1_score(labels, preds, pos_label=0),
        "f1_class1": f1_score(labels, preds, pos_label=1),
    }


def build_trainer(model, train_ds, val_ds, tokenizer, config: dict) -> Trainer:
    """
    Construct and return a HuggingFace Trainer for the given model and datasets.

    Args:
        model: The model to train (baseline or HierarchicalContextModel).
        train_ds: Tokenized training dataset.
        val_ds: Tokenized validation dataset.
        tokenizer: Tokenizer used.
        config: Dict of hyperparameters (c.f. config.py).

    Returns:
        A configured Trainer.
    """
    args = TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints"),
        num_train_epochs=config.get("epochs", 3),
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,                  
        weight_decay=config.get("weight_decay", 0.0),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        warmup_ratio=config.get("warmup_ratio", 0.0),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",     
        logging_steps=50,
        fp16=config.get("fp16", False),
        report_to="none",  
        seed = 42,                       
    )

    _is_custom = isinstance(model, HierarchicalContextModel)
    collator = DefaultDataCollator() if _is_custom else None

    class_weights  = config.get("class_weights")
    is_hierarchical = _is_custom

    use_custom_trainer = class_weights is not None or is_hierarchical

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=config.get("metrics_fn"),
        callbacks=config.get("callbacks"),
    )

    if use_custom_trainer:
        if class_weights is None:
            class_weights = torch.ones(config.get("num_labels", 2))
        trainer_kwargs["class_weights"] = class_weights

    trainer_cls = WeightedLossTrainer if use_custom_trainer else Trainer
    return trainer_cls(**trainer_kwargs)