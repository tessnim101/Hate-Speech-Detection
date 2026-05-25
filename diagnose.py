"""
Diagnostic script to check comment_id duplicates in the dataset.

python3 diagnose.py --dataset_path "data/spanish_subset/"
"""

import argparse
import pandas as pd
from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, split_train_validation, clean_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df_train, _ = load_data(args.dataset_path)
    df_train     = clean_df(df_train)
    df_train     = filter_contextual_tweets(df_train)
    df_train, _  = split_train_validation(df_train)

    id_to_text = df_train[["comment_id", "text"]]

    print(f"Total rows:         {len(id_to_text)}")
    print(f"Unique comment_ids: {id_to_text['comment_id'].nunique()}")

    dupes = id_to_text[id_to_text.duplicated(subset="comment_id", keep=False)]
    print(f"\nDuplicated comment_ids: {dupes['comment_id'].nunique()}")
    print(f"Duplicated rows total:  {len(dupes)}")

    if len(dupes) == 0:
        print("\n✓ No duplicates — ids_to_text merge is safe.")
        return

    # Check if duplicated ids have same or different text
    same_text  = 0
    diff_text  = 0
    diff_examples = []

    for cid, group in dupes.groupby("comment_id"):
        if group["text"].nunique() == 1:
            same_text += 1
        else:
            diff_text += 1
            diff_examples.append(group)

    print(f"\nDuplicated ids with same text:      {same_text}  (safe to drop)")
    print(f"Duplicated ids with different text: {diff_text}  (data quality issue)")

    if diff_text > 0:
        print("\n--- Examples of same id / different text ---")
        for group in diff_examples[:3]:
            print(group.to_string())
            print()
    else:
        print("\n✓ All duplicates have identical text — drop_duplicates is safe.")

    print(df_train[["comment_id", "level2", "level3", "text"]].head(20).to_string())


if __name__ == "__main__":
    main()