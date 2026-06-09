"""
Used to pick test samples for baseline vs context wins 

# context wins over baseline — stereotype only
python analysis/pick_index.py \
    --per_sample     "results/inference_per_sample.csv" \
    --dataset_path   "data/spanish_subset_collapsed/" \
    --n              10 \
    --verdict        context_wins \
    --min_confidence 0.6 \
    --label          1

# context wins over baseline — not stereotype only
python analysis/pick_index.py \
    --per_sample     "results/inference_per_sample.csv" \
    --dataset_path   "data/spanish_subset_collapsed/" \
    --n              10 \
    --verdict        context_wins \
    --min_confidence 0.65 \
    --label          0

# baseline wins over context
python analysis/pick_index.py \
    --per_sample     "results/inference_per_sample.csv" \
    --dataset_path   "data/spanish_subset_collapsed/" \
    --n              10 \
    --verdict        baseline_wins \
    --min_confidence 0.65

# all samples
python analysis/pick_index.py \
    --per_sample     "results/inference_per_sample.csv" \
    --dataset_path   "data/spanish_subset_collapsed/" \
    --n              10 \
    --verdict        all
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
    p.add_argument("--per_sample",     required=True)
    p.add_argument("--dataset_path",   required=True)
    p.add_argument("--n",              type=int,   default=10)
    p.add_argument("--min_confidence", type=float, default=0.0,
                   help="Minimum confidence of cross-attention prediction (0.0 = no filter)")
    p.add_argument("--label",          type=int,   default=None,
                   help="Filter by label: 0=not stereotype, 1=stereotype")
    p.add_argument(
        "--verdict",
        choices=["context_wins", "baseline_wins", "all"],
        default="context_wins",
        help="Which verdict to filter: context_wins, baseline_wins, or all",
    )
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

    per_sample = pd.read_csv(args.per_sample)
    per_sample = per_sample[per_sample["text"].isin(df_test["text"])].reset_index(drop=True)

    merged = per_sample.merge(
        df_test[["text", "root_text", "parent_text"]].reset_index().rename(
            columns={"index": "tweet_idx"}
        ),
        on="text", how="inner",
    )

    # Verdict filter
    if args.verdict == "all":
        filtered = merged.copy()
    else:
        filtered = merged[merged["verdict"] == args.verdict].copy()

    # Label filter
    if args.label is not None:
        filtered = filtered[filtered["stereotype"] == args.label]
        label_name = "Stereotype" if args.label == 1 else "Not Stereotype"
        print(f"[filter] Label={args.label} ({label_name}) — {len(filtered)} remaining")

    # Confidence filter — uses predicted class probability regardless of class
    filtered["ca_pred_prob"] = filtered.apply(
        lambda r: r["ca_prob1"] if r["ca_pred"] == 1 else 1 - r["ca_prob1"], axis=1
    )
    if args.min_confidence > 0.0:
        filtered = filtered[filtered["ca_pred_prob"] >= args.min_confidence]
        print(f"[filter] Keeping samples with predicted confidence >= {args.min_confidence} "
              f"— {len(filtered)} remaining")

    filtered = filtered.head(args.n)

    print(f"\nShowing {len(filtered)} samples (verdict='{args.verdict}')\n")
    print("=" * 70)

    for _, row in filtered.iterrows():
        label       = int(row["stereotype"])
        label_name  = "Stereotype" if label == 1 else "Not Stereotype"
        hier_result = "✓ correct" if row["hier_correct"] == 1 else "✗ wrong"
        ca_result   = "✓ correct" if row["ca_correct"]   == 1 else "✗ wrong"

        baseline_conf = row["baseline_prob1"] if row["baseline_pred"] == 1 else 1 - row["baseline_prob1"]
        hier_conf     = row["hier_prob1"]     if row["hier_pred"]     == 1 else 1 - row["hier_prob1"]

        print(f"\n--- tweet_idx={int(row['tweet_idx'])} | verdict={row['verdict']} | "
              f"ca_confidence={row['ca_pred_prob']:.3f} ---")
        print(f"  Tweet:   {row['text'][:120]}")
        print(f"  Root:    {str(row['root_text'])[:80]}")
        print(f"  Parent:  {str(row['parent_text'])[:80]}")
        print(f"  Label:   {label_name}")
        print(f"  Baseline pred:      {'Stereotype' if row['baseline_pred'] == 1 else 'Not Stereotype'} "
              f"(confidence={baseline_conf:.3f}) "
              f"{'✓ correct' if row['baseline_correct'] == 1 else '✗ wrong'}")
        print(f"  Hierarchical pred:  {'Stereotype' if row['hier_pred'] == 1 else 'Not Stereotype'} "
              f"(confidence={hier_conf:.3f}) {hier_result}")
        print(f"  Cross-Attn pred:    {'Stereotype' if row['ca_pred']   == 1 else 'Not Stereotype'} "
              f"(confidence={row['ca_pred_prob']:.3f}) {ca_result}")
        print("=" * 70)


if __name__ == "__main__":
    main()