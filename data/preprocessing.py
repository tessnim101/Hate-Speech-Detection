"""
Pre-process data
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def filter_contextual_tweets(df):
    """
    Keep only tweets that have both a parent and a root tweet.

    Tweets without context (level2 == 0 or level3 == 0) are dropped so that
    the baseline and hierarchical models are evaluated on the same population.
    """
    before = len(df)
    df = df[(df["level2"] != "0") & (df["level3"] != "0")].reset_index(drop=True)
    after = len(df)
    print(f"[filter] Kept {after:,} / {before:,} tweets with full context "
          f"({before - after:,} dropped).")
    return df


def stratified_sample(df, n):
    """
    Reduce dataset size for CPU-friendly tests
    """
    groups = []
    for _, group in df.groupby('stereotype'):
        groups.append(group.sample(min(len(group), n // 2), random_state=42))
    return pd.concat(groups).reset_index(drop=True)


def split_train_validation(df, val_size=0.2):
    """
    Split training data into training and evaluation
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["stereotype"],
        random_state=42
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def ids_to_text(df):
    """
    Replace parent and root tweet ids by actual textual tweet.
    """
    id_to_text = df[['comment_id', 'text']].rename(
        columns={'comment_id': 'lookup_id', 'text': 'lookup_text'}
    )

    df = df.merge(
        id_to_text,
        left_on='level2',
        right_on='lookup_id',
        how='left'
    ).rename(columns={'lookup_text': 'parent_text'}).drop(columns=['lookup_id'])

    df = df.merge(
        id_to_text,
        left_on='level3',
        right_on='lookup_id',
        how='left'
    ).rename(columns={'lookup_text': 'root_text'}).drop(columns=['lookup_id'])

    df['parent_text'] = df['parent_text'].fillna("")
    df['root_text']   = df['root_text'].fillna("")

    return df


def tokenize_baseline(dataset, tokenizer, max_len):
    """
    Tokenize for the baseline model (tweet only, no context)
    """
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
    return dataset.map(tokenize, batched=True)


def tokenize_hierarchical(dataset, tokenizer, max_len: int):
    """
    Tokenizer for the hierarchical model (context-aware).
    Tokenizes root, parent, and tweet independently.
    """
    def tokenize(example):
        def enc(texts):
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

        root   = enc(example["root_text"])
        parent = enc(example["parent_text"])
        tweet  = enc(example["text"])

        return {
            "root_input_ids":        root["input_ids"],
            "root_attention_mask":   root["attention_mask"],
            "parent_input_ids":      parent["input_ids"],
            "parent_attention_mask": parent["attention_mask"],
            "tweet_input_ids":       tweet["input_ids"],
            "tweet_attention_mask":  tweet["attention_mask"],
        }

    return dataset.map(tokenize, batched=True)

def tokenize_cross_attention(dataset, tokenizer, max_len: int):
    """
    Tokenizer for the cross-attention model.
    Context (hoax + root + parent) and tweet are tokenized separately.
    """
    def tokenize(example):
        context = []
        for h, r, p in zip(example["hoax"], example["root_text"], example["parent_text"]):
            parts = []
            if h and h.strip(): parts.append(f"Hoax: {h}")
            if r and r.strip(): parts.append(f"Thread: {r}")
            if p and p.strip(): parts.append(f"Reply to: {p}")
            context.append(" | ".join(parts))

        ctx   = tokenizer(context,         truncation=True, padding="max_length", max_length=max_len)
        tweet = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_len)
        return {
            "context_input_ids":      ctx["input_ids"],
            "context_attention_mask": ctx["attention_mask"],
            "tweet_input_ids":        tweet["input_ids"],
            "tweet_attention_mask":   tweet["attention_mask"],
        }
    return dataset.map(tokenize, batched=True)

def augment_with_backtranslation(df):
    """
    Expand the training set by appending back-translated tweets as new rows.

    For each row where backtranslated_text is non-empty, a copy of the row is
    created with text replaced by the back-translated version and the same label.
    This follows the data-augmentation approach from Beddiar et al. (2021).
    """
    bt_rows = df[df["backtranslated_text"].ne("")].copy()
    bt_rows["text"] = bt_rows["backtranslated_text"]
    augmented = pd.concat([df, bt_rows], ignore_index=True)
    print(f"[augment] Added {len(bt_rows):,} back-translated rows → {len(augmented):,} total.")
    return augmented