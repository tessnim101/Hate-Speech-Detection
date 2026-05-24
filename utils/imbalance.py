"""
Class-imbalance utilities for training data preparation.

Provides two complementary strategies:
- Inverse-frequency class weights (used with weighted loss functions)
- Oversampling (duplicates minority samples to balance the training set)

Only one strategy should be active per training run, controlled by --imbalance_strategy in main.py.
"""

import torch
import pandas as pd


def compute_class_weights(df, label_col="stereotype"):
    """
    Compute inverse-frequency class weights for use in a weighted loss function.

    Args:
        df: Training DataFrame.
        label_col: Column containing the class labels.

    Returns:
        FloatTensor of shape (num_classes,), one weight per class sorted by label.
    """
    counts  = df[label_col].value_counts().sort_index()
    freqs   = counts.values.astype(float)
    weights = 1.0 / freqs                              # inverse frequency
    weights = weights / weights.sum() * len(weights)   # normalize to sum to num_classes
    return torch.tensor(weights, dtype=torch.float)


def oversample_minority_classes(df, label_col="stereotype", random_state=42):
    """
    Oversample minority classes to match the majority class count.

    Args:
        df: Training DataFrame.
        label_col: Column containing the class labels.
        random_state: Seed for reproducibility.

    Returns:
        Balanced and shuffled DataFrame.
    """
    max_count = df[label_col].value_counts().max()
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            # Sample with replacement to reach majority class size
            group = group.sample(max_count, replace=True, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)