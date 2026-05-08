"""
Build trainer
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from modeling.models import HierarchicalContextModel


class WeightedLossTrainer(Trainer):
    """
    Trainer subclass that applies class weights to the cross-entropy loss.

    The base Trainer.compute_loss() uses the loss returned by the model's
    forward() pass, which is unweighted. We recompute it here with the
    supplied weights so the minority classes contribute more to the gradient.

    Weights should be a 1-D float tensor of length num_labels, already on CPU
    (they are moved to the correct device inside compute_loss).
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # HierarchicalContextModel returns a dict; standard models return a ModelOutput
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        # Re-attach labels so downstream metrics callbacks still see them
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss


def build_trainer(model, train_ds, val_ds, tokenizer, config: dict) -> Trainer:
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
    )

    collator = DefaultDataCollator() if isinstance(model, HierarchicalContextModel) else None

    class_weights = config.get("class_weights")
    is_hierarchical = isinstance(model, HierarchicalContextModel)

    # Use WeightedLossTrainer if we have class weights OR if the model
    # doesn't return its own loss (hierarchical model)
    use_custom_trainer = class_weights is not None or is_hierarchical

    trainer_cls = WeightedLossTrainer if use_custom_trainer else Trainer
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
        # If no class weights, pass uniform weights so WeightedLossTrainer
        # still works but behaves identically to standard CE loss
        if class_weights is None:
            num_labels = config.get("num_labels", 2)
            class_weights = torch.ones(num_labels)
        trainer_kwargs["class_weights"] = class_weights

    return trainer_cls(**trainer_kwargs)