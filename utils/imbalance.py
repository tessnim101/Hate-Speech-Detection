import torch
import pandas as pd

def compute_class_weights(df, label_col="stereotype"):
    counts  = df[label_col].value_counts().sort_index()
    freqs   = counts.values.astype(float)
    weights = 1.0 / freqs
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float)


def oversample_minority_classes(df, label_col="stereotype", random_state=42):
    max_count = df[label_col].value_counts().max()
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            group = group.sample(max_count, replace=True, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
