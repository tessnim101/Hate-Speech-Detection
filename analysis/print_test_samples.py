"""
Print all test samples with root and parent text.

# all samples
python analysis/print_test_samples.py \
    --dataset_path "data/spanish_subset_collapsed/"

# stereotype only
python analysis/print_test_samples.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --label 1

# not stereotype only
python analysis/print_test_samples.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --label 0
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text, clean_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--label",        type=int, default=None,
                   help="Filter by label: 0=not stereotype, 1=stereotype")
    return p.parse_args()


def main():
    args = parse_args()

    df_train, df_test = load_data(args.dataset_path)
    df_train = clean_df(df_train)
    df_test  = clean_df(df_test)
    combined = pd.concat([df_train, df_test], ignore_index=True)
    df_train = ids_to_text(df_train, lookup_df=combined)
    df_test  = ids_to_text(df_test,  lookup_df=combined)
    df_test  = filter_contextual_tweets(df_test)
    df_test  = df_test.reset_index(drop=True)

    if args.label is not None:
        df_test = df_test[df_test["stereotype"] == args.label]
        print(f"Filtered to label={args.label} — {len(df_test)} samples\n")
    else:
        print(f"Total test samples with context: {len(df_test)}\n")

    print("=" * 70)
    for idx, row in df_test.iterrows():
        print(f"\n--- idx={idx} | label={'Stereotype' if row['stereotype'] == 1 else 'Not Stereotype'} ---")
        print(f"  Tweet:   {row['text'][:120]}")
        print(f"  Root:    {str(row['root_text'])[:80]}")
        print(f"  Parent:  {str(row['parent_text'])[:80]}")
        print("=" * 70)


if __name__ == "__main__":
    main()