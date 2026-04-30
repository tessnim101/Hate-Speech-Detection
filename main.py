"""
Run training
"""

from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
from datetime import datetime

from data.loader import load_data
from data.preprocessing import *
from modeling.models import load_model, HierarchicalContextModel
from training.trainer_utils import build_trainer
from training.metrics import compute_metrics
from config import CONFIG


def prepare_dataset(df, columns, label_col="stereotype"):
    df = df[columns + [label_col]].reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    return dataset.rename_column(label_col, "labels")

def format_dataset(dataset, model_type="baseline"):
    columns = {
        "baseline":     ["input_ids", "attention_mask", "labels"],
        "hierarchical": [
            "root_input_ids",   "root_attention_mask",
            "parent_input_ids", "parent_attention_mask",
            "tweet_input_ids",  "tweet_attention_mask",
            "labels",
        ],
    }
    dataset.set_format(type="torch", columns=columns[model_type])
    return dataset

def save_train_results(trainer, filepath="results/train_results.csv", extra_info=None):
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

def save_test_results(metrics, filepath="results/test_results.csv"):
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile(filepath)

    df.to_csv(
        filepath,
        mode="a",
        header=not file_exists,
        index=False
    )

def evaluate_on_test(trainer, df_test, tokenizer):
    test_ds = prepare_dataset(df_test, ["text"])
    test_ds = tokenize_baseline(test_ds, tokenizer, CONFIG["max_len"])
    test_ds = format_dataset(test_ds)

    metrics = trainer.evaluate(test_ds)
    return metrics

def run_baseline(df_train, df_val, df_test, run_id):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = prepare_dataset(df_train, ["text"])
    val_ds = prepare_dataset(df_val, ["text"])

    train_ds = tokenize_baseline(train_ds, tokenizer, CONFIG["max_len"])
    val_ds = tokenize_baseline(val_ds, tokenizer, CONFIG["max_len"])

    train_ds = format_dataset(train_ds)
    val_ds = format_dataset(val_ds)

    model = load_model(CONFIG["model_name"])

    CONFIG["metrics_fn"] = compute_metrics

    trainer = build_trainer(model, train_ds, val_ds, tokenizer, CONFIG)
    trainer.train()
    
    extra_info = {
        "run_id": run_id,
        "model": "baseline",
        "context": False,
        "max_len": CONFIG["max_len"]
    }

    save_train_results(trainer, extra_info=extra_info)

    test_metrics = evaluate_on_test(trainer, df_test, tokenizer)
    test_metrics["run_id"] = run_id
    test_metrics["split"] = "test"
    test_metrics["model"] = "baseline"
    save_test_results(test_metrics)

def run_hierarchical(df_train, df_val, df_test, run_id):

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train = ids_to_text(df_train)
    df_val   = ids_to_text(df_val)
    df_test  = ids_to_text(df_test)

    train_ds = prepare_dataset(df_train, ["text", "parent_text", "root_text"])
    val_ds   = prepare_dataset(df_val,   ["text", "parent_text", "root_text"])

    train_ds = tokenize_hierarchical(train_ds, tokenizer, CONFIG["max_len"])
    val_ds   = tokenize_hierarchical(val_ds,   tokenizer, CONFIG["max_len"])

    train_ds = format_dataset(train_ds, "hierarchical")
    val_ds   = format_dataset(val_ds,   "hierarchical")

    model = HierarchicalContextModel(CONFIG["model_name"])

    CONFIG["metrics_fn"] = compute_metrics
    trainer = build_trainer(model, train_ds, val_ds, tokenizer, CONFIG)
    trainer.train()

    extra_info = {
        "run_id": run_id, 
        "model": "hierarchical", 
        "context": True
    }
    
    save_train_results(trainer, extra_info=extra_info)

    test_ds = prepare_dataset(df_test, ["text", "parent_text", "root_text"])
    test_ds = tokenize_hierarchical(test_ds, tokenizer, CONFIG["max_len"])
    test_ds = format_dataset(test_ds, "hierarchical")

    test_metrics = trainer.evaluate(test_ds)
    test_metrics.update({"run_id": run_id, "split": "test", "model": "hierarchical"})
    save_test_results(test_metrics)

"""def run_context(df_train, df_val, run_id):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    df_train = ids_to_text(df_train)
    df_val = ids_to_text(df_val)

    train_ds = prepare_dataset(df_train, ["text", "parent_text", "root_text"])
    val_ds = prepare_dataset(df_val, ["text", "parent_text", "root_text"])

    train_ds = tokenize_context(train_ds, tokenizer, CONFIG["max_len_context"])
    val_ds = tokenize_context(val_ds, tokenizer, CONFIG["max_len_context"])

    train_ds = format_dataset(train_ds)
    val_ds = format_dataset(val_ds)

    model = load_model(CONFIG["model_name"])

    CONFIG["metrics_fn"] = compute_metrics

    trainer = build_trainer(model, train_ds, val_ds, tokenizer, CONFIG)
    trainer.train()

    extra_info = {
        "run_id": run_id,
        "model": "context",
        "context": True,
        "max_len": CONFIG["max_len"]
    }

    save_results(trainer, extra_info=extra_info)

    test_metrics = evaluate_on_test(trainer, df_test, tokenizer)
    test_metrics["run_id"] = run_id
    test_metrics["split"] = "test"
    save_test_results(test_metrics)"""


if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path_data = "data/spanish_subset/"

    df_train, df_test = load_data(path_data)

    # small CPU-friendly subsets
    df_train_small = stratified_sample(df_train, 100)
    df_train_split, df_val_split = split_train_validation(df_train_small)
    df_test_small = stratified_sample(df_test, 50)

    #run_baseline(df_train_split, df_val_split, df_test_small, run_id=RUN_ID)

    # context experiment
    df_train_small = stratified_sample(df_train, 80)
    df_train_split, df_val_split = split_train_validation(df_train_small)
    df_test_small = stratified_sample(df_test, 40)

    run_hierarchical(df_train_split, df_val_split, df_test_small, run_id=RUN_ID)