"""
Run training for both the baseline and the hierearchical (context-aware) model.

python main.py \
    --dataset_path "data/spanish_subset/" \
    --results_path "results/" \
    --imbalance_strategy "oversample"
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import random
import numpy as np

import torch

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, split_train_validation
from utils.imbalance import compute_class_weights, oversample_minority_classes
from training.runners import run_baseline, run_hierarchical, run_augmented

def set_seed(seed: int):
    """
    Fix all sources of randomness for reproducible training.
    Does not eliminate run-to-run variance entirely on GPU (CUDA
    non-determinism) but makes results as stable as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forces cuDNN to use deterministic algorithms — small performance cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    """
    Parse path to dataset, path to results folder and imbalance
    strategy (oversample or class weights, see utils/imbalance.py)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--results_path",  required=True)
    p.add_argument(
        "--imbalance_strategy",
        choices=["class_weights", "oversample", "none"],
        default="class_weights",
    )
    args = p.parse_args()
    return args.dataset_path, args.results_path, args.imbalance_strategy


if __name__ == "__main__":
    set_seed(42)
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATADIR, RES_DIR, IMBALANCE_STRATEGY = parse_args()

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(Path(RES_DIR) / "checkpoints", exist_ok=True)

    df_train, df_test = load_data(DATADIR)
    # Keep only tweets that have both parent and root tweets
    df_train = filter_contextual_tweets(df_train)
    df_test  = filter_contextual_tweets(df_test)
    df_train_split, df_val_split = split_train_validation(df_train)

    # Class-imbalance handling
    class_weights = None
    if IMBALANCE_STRATEGY == "oversample":
        print("[imbalance] Oversampling minority classes.")
        df_train_split = oversample_minority_classes(df_train_split)
    elif IMBALANCE_STRATEGY == "class_weights":
        print("[imbalance] Computing inverse-frequency class weights.")
        class_weights = compute_class_weights(df_train_split)
        print(f"            weights → {class_weights.tolist()}")

    # GPU info
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[device] No CUDA devices — running on CPU.")
    else:
        print(f"[device] {n_gpus} GPU(s): "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))

    # run_baseline(
    #     df_train_split, df_val_split, df_test,
    #     run_id=RUN_ID, res_dir=RES_DIR, class_weights=class_weights,
    # )
    # run_hierarchical(
    #     df_train_split, df_val_split, df_test,
    #     run_id=RUN_ID, res_dir=RES_DIR, class_weights=class_weights,
    # )
    run_augmented(
        df_train_split, df_val_split, df_test,
        run_id=RUN_ID, res_dir=RES_DIR, class_weights=class_weights,
    )