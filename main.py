"""
Run training
"""

from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
from datetime import datetime

from data.loader import load_data
from data.preprocessing import stratified_sample, ids_to_text
from modeling.model_utils import load_model
from training.trainer_utils import build_trainer
from training.metrics import compute_metrics
from config import CONFIG

def tokenize_baseline(dataset, tokenizer, max_len):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    return dataset.map(tokenize, batched=True)


def tokenize_context(dataset, tokenizer, max_len):
    def tokenize(example):
        combined = [
            r + " </s> " + p + " </s> " + t
            for r, p, t in zip(
                example["root_text"],
                example["parent_text"],
                example["text"]
            )
        ]

        return tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    return dataset.map(tokenize, batched=True)


def prepare_dataset(df, columns, label_col="stereotype"):
    df = df[columns + [label_col]]
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset = dataset.rename_column(label_col, "labels")
    return dataset


def format_dataset(dataset):
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def save_results(trainer, filepath="results/results.csv", extra_info=None):
    logs = trainer.state.log_history

    all_logs = [x for x in logs if "loss" in x or "eval_loss" in x]
    df = pd.DataFrame(all_logs)

    if extra_info is not None:
        for key, value in extra_info.items():
            df[key] = value

    file_exists = os.path.isfile(filepath)

    df.to_csv(
        filepath,
        mode="a",
        header=not file_exists,
        index=False
    )

def run_baseline(df_train, df_test, run_id):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = prepare_dataset(df_train, ["text"])
    test_ds = prepare_dataset(df_test, ["text"])

    train_ds = tokenize_baseline(train_ds, tokenizer, CONFIG["max_len"])
    test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])

    train_ds = format_dataset(train_ds)
    test_ds = format_dataset(test_ds)

    model = load_model(CONFIG["model_name"])

    CONFIG["metrics_fn"] = compute_metrics

    trainer = build_trainer(model, train_ds, test_ds, tokenizer, CONFIG)
    trainer.train()
    
    extra_info = {
        "run_id": run_id,
        "model": "baseline",
        "context": False,
        "max_len": CONFIG["max_len"]
    }

    save_results(trainer, extra_info=extra_info)


def run_context(df_train, df_test, run_id):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train = ids_to_text(df_train)
    df_test = ids_to_text(df_test)

    train_ds = prepare_dataset(df_train, ["text", "parent_text", "root_text"])
    test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])

    train_ds = tokenize_context(train_ds, tokenizer, CONFIG["max_len_context"])
    test_ds = tokenize_context(test_ds, tokenizer, CONFIG["max_len_context"])

    train_ds = format_dataset(train_ds)
    test_ds = format_dataset(test_ds)

    model = load_model(CONFIG["model_name"])

    CONFIG["metrics_fn"] = compute_metrics

    trainer = build_trainer(model, train_ds, test_ds, tokenizer, CONFIG)
    trainer.train()

    extra_info = {
        "run_id": run_id,
        "model": "context",
        "context": True,
        "max_len": CONFIG["max_len"]
    }

    save_results(trainer, extra_info=extra_info)


if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path_data = "data/spanish_subset/"

    df_train, df_test = load_data(path_data)

    # small CPU-friendly subsets
    df_train_small = stratified_sample(df_train, 100)
    df_test_small = stratified_sample(df_test, 30)

    run_baseline(df_train_small, df_test_small, run_id=RUN_ID)

    # context experiment
    df_train_small = stratified_sample(df_train, 200)
    df_test_small = stratified_sample(df_test, 50)

    #run_context(df_train_small, df_test_small, run_id=RUN_ID)