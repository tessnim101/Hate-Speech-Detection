import sys
sys.path.insert(0, "..").parent

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, split_train_validation

df_train, df_test = load_data("spanish_subset/")

df_train = filter_contextual_tweets(df_train)
df_test  = filter_contextual_tweets(df_test)

df_train, df_val = split_train_validation(df_train)

for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    counts = df["stereotype"].value_counts().sort_index()
    total  = len(df)
    print(f"\n{name} — {total} samples")
    for label, count in counts.items():
        print(f"  class {label}: {count:>4}  ({100*count/total:.1f}%)")

print("\nSample data:")
print(df_train.head())