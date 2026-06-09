"""
Trainer construction and custom loss for stereotype classification.

Selects the appropriate Trainer subclass depending on whether class weights
are supplied and whether the model is hierarchical. All HuggingFace TrainingArguments
are centralised here so hyperparameters don't need to be scattered across callers.

Three imbalance strategies are supported:
    - "weighted_loss": static inverse-frequency class weights in cross-entropy
    - "oversample":    handled at data level in main.py, uses weighted loss here too
    - "focal_loss":    dynamic focal loss that down-weights easy majority examples
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from sklearn.metrics import f1_score, accuracy_score
from modeling.models import HierarchicalContextModel, CrossAttentionContextModel


class WeightedLossTrainer(Trainer):
    """
    Trainer subclass that applies class weights to the cross-entropy loss
    with label smoothing to reduce overconfidence on small datasets.

    Args:
        class_weights: 1-D float tensor of length num_labels.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=0.1,
        )(logits, labels)

        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return (loss, None, None)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return (loss, logits.detach(), labels.detach() if labels is not None else None)


class FocalLossTrainer(Trainer):
    """
    Trainer using Focal Loss for class imbalance.

    Focal loss down-weights easy majority examples dynamically and focuses
    training on hard misclassified ones — strictly better than static
    weighted cross-entropy for imbalanced noisy data.

    FL(p) = -alpha * (1 - p_correct)^gamma * log(p_correct)

    Args:
        class_weights: 1-D float tensor of length num_labels (alpha per class)
        focal_gamma:   focusing parameter γ (default 2.0)
    """

    def __init__(self, *args, class_weights: torch.Tensor, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma   = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # Per-sample CE with class weights
        ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            reduction="none",
        )(logits, labels)

        # Focal weight: (1 - p_correct)^gamma
        probs        = torch.softmax(logits, dim=-1)
        p_correct    = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_correct) ** self.focal_gamma

        loss = (focal_weight * ce_loss).mean()
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return (loss, None, None)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return (loss, logits.detach(), labels.detach() if labels is not None else None)


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
        "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_binary": f1_score(labels, preds, average="binary", zero_division=0),
        "f1_class0": f1_score(labels, preds, pos_label=0,      zero_division=0),
        "f1_class1": f1_score(labels, preds, pos_label=1,      zero_division=0),
    }


def build_trainer(model, train_ds, val_ds, tokenizer, config: dict) -> Trainer:
    """
    Construct and return a HuggingFace Trainer for the given model and datasets.

    Args:
        model:    The model to train.
        train_ds: Tokenized training dataset.
        val_ds:   Tokenized validation dataset.
        tokenizer: Tokenizer used.
        config:   Dict of hyperparameters (c.f. config.py).

    Returns:
        A configured Trainer instance.
    """
    train_size   = len(train_ds)
    steps_per_epoch = max(1, train_size // (config.get("train_bs", 16) * 2))  # *2 for grad accumulation
    total_steps     = steps_per_epoch * config.get("epochs", 10)
    warmup_steps    = max(50, int(0.06 * total_steps))
    args = TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints"),
        num_train_epochs=config.get("epochs", 7),
        learning_rate=config.get("learning_rate", 1e-5),
        per_device_train_batch_size=config.get("train_bs", 16),
        per_device_eval_batch_size=config.get("eval_bs", 16),
        #dataloader_num_workers=10,        # parallel data loading
        dataloader_pin_memory=True,        # faster GPU transfer
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        #warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type="linear",
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_class1",
        logging_strategy="epoch",
        logging_steps=50,
        fp16=config.get("fp16", False),
        report_to="none",
        seed=config.get("seed", 42),
    )

    _is_custom          = isinstance(model, (HierarchicalContextModel, CrossAttentionContextModel))
    collator            = DefaultDataCollator() if _is_custom else None
    class_weights       = config.get("class_weights")
    imbalance_strategy  = config.get("imbalance_strategy", "weighted_loss")

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

    # Always use custom trainer for custom models even without class weights
    use_custom = class_weights is not None or _is_custom

    if use_custom:
        if class_weights is None:
            class_weights = torch.ones(config.get("num_labels", 2))

        if imbalance_strategy == "focal_loss":
            trainer_kwargs["class_weights"] = class_weights
            trainer_kwargs["focal_gamma"]   = config.get("focal_gamma", 2.0)
            trainer_cls = FocalLossTrainer
        else:
            # covers "weighted_loss" and "oversample" (oversample is data-level,
            # weighted loss is still applied here for robustness)
            trainer_kwargs["class_weights"] = class_weights
            trainer_cls = WeightedLossTrainer
    else:
        trainer_cls = Trainer

    return trainer_cls(**trainer_kwargs)