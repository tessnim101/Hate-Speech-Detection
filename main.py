"""
Run training
"""

from datasets import Dataset
from transformers import AutoTokenizer

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


def run_baseline(df_train, df_test):
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


def run_context(df_train, df_test):
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


if __name__ == "__main__":
    path_data = "data/spanish_subset/"

    df_train, df_test = load_data(path_data)

    # small CPU-friendly subsets
    df_train_small = stratified_sample(df_train, 400)
    df_test_small = stratified_sample(df_test, 60)

    run_baseline(df_train_small, df_test_small)

    # context experiment
    df_train_small = stratified_sample(df_train, 200)
    df_test_small = stratified_sample(df_test, 50)

    #run_context(df_train_small, df_test_small)