"""
Run training for baseline, hierarchical, cross-attention, and augmented models.

python main.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --results_path "results/" \
    --imbalance_strategy "class_weights"
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import pandas as pd
import shutil

import torch

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, split_train_validation, clean_df, ids_to_text
from utils.imbalance import compute_class_weights, oversample_minority_classes
from training.runners import run_baseline, run_hierarchical, run_augmented, run_cross_attention
from config import CONFIG


def set_seed(seed: int):
    """
    Fix all sources of randomness for reproducible training.
    Note: cudnn.benchmark is disabled to avoid non-deterministic
    algorithm selection — this trades a small speed cost for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
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
    SEEDS = [42, 123, 456]
    DATADIR, RES_DIR, IMBALANCE_STRATEGY = parse_args()

    os.makedirs(RES_DIR, exist_ok=True)

    # Data loading & preprocessing (done once, outside the seed loop) 
    df_train, df_test = load_data(DATADIR)

    df_train = clean_df(df_train)
    df_test  = clean_df(df_test)

    combined = pd.concat([df_train, df_test], ignore_index=True)
    df_train = ids_to_text(df_train, lookup_df=combined)
    df_test  = ids_to_text(df_test,  lookup_df=combined)

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

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[device] No CUDA devices — running on CPU.")
    else:
        print(f"[device] {n_gpus} GPU(s): "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))

    # Multi-seed training loop 
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*60}\n")

        CONFIG["seed"] = seed
        set_seed(seed)

        RUN_ID = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{seed}"
        run_res_dir = str(Path(RES_DIR) / f"seed_{seed}")
        os.makedirs(run_res_dir, exist_ok=True)

        print("RUNNING BASELINE")
        run_baseline(
            df_train_split, df_val_split, df_test,
            run_id=RUN_ID, res_dir=run_res_dir, class_weights=class_weights,
        )
        print("RUNNING HIERARCHICAL")
        run_hierarchical(
            df_train_split, df_val_split, df_test,
            run_id=RUN_ID, res_dir=run_res_dir, class_weights=class_weights,
        )
        print("RUNNING CROSS-ATTENTION")
        run_cross_attention(
            df_train_split, df_val_split, df_test,
            run_id=RUN_ID, res_dir=run_res_dir, class_weights=class_weights,
        )
        print("RUNNING AUGMENTED")
        run_augmented(
            df_train_split, df_val_split, df_test,
            run_id=RUN_ID, res_dir=run_res_dir, class_weights=class_weights,
        )
 
        print("\n[results] Aggregating results across seeds...")

        result_files = list(Path(RES_DIR).glob("seed_*/test_results.csv"))
        if result_files:
            df_all = pd.concat([pd.read_csv(f) for f in result_files], ignore_index=True)
            df_all.to_csv(Path(RES_DIR) / "all_test_results.csv", index=False)

            metrics = ["eval_f1_macro", "eval_f1_binary", "eval_f1_class0", "eval_f1_class1", "eval_accuracy"]
            summary = df_all.groupby("model")[metrics].agg(["mean", "std"]).round(4)
            summary.to_csv(Path(RES_DIR) / "summary_mean_std.csv")
            print("\n", summary.to_string())

            # Copy median model  
            MODEL_FOLDER_MAP = {
                "baseline":        "best_model_baseline",
                "hierarchical":    "best_model_context",
                "cross_attention": "best_model_cross_attention",
                "augmented":       "best_model_bt",
            }

            for model_name, model_folder in MODEL_FOLDER_MAP.items():
                df_model = df_all[df_all["model"] == model_name].copy()
                if df_model.empty:
                    print(f"\n[median {model_name}] No results found — skipping.")
                    continue

                mean_f1     = df_model["eval_f1_macro"].mean()
                median_idx  = (df_model["eval_f1_macro"] - mean_f1).abs().idxmin()
                median_row  = df_model.loc[median_idx]
                median_seed = median_row["run_id"].split("_seed")[-1]

                print(f"\n[median {model_name}] mean={mean_f1:.4f}, "
                    f"selected seed={median_seed} (eval_f1_macro={median_row['eval_f1_macro']:.4f})")

                best_model_src = Path(RES_DIR) / f"seed_{median_seed}" / model_folder
                best_model_dst = Path(RES_DIR) / f"best_model_{model_name}"

                if best_model_src.exists():
                    shutil.copytree(best_model_src, best_model_dst, dirs_exist_ok=True)
                    print(f"[median {model_name}] saved to → {best_model_dst}")
                else:
                    print(f"[median {model_name}] Could not find model at {best_model_src} — check folder structure.")
        else:
            print("[results] No test_results.csv files found — check your res_dir paths.")