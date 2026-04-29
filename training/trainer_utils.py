"""
Build trainer
"""

from transformers import Trainer, TrainingArguments


def build_trainer(model, train_dataset, test_dataset, tokenizer, config):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["train_bs"],
        per_device_eval_batch_size=config["eval_bs"],
        num_train_epochs=config["epochs"],
        save_strategy="epoch",
        logging_dir=config["log_dir"],
        fp16=config.get("fp16", False),
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=config["metrics_fn"],
    )
